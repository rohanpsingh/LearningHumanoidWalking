"""Persistent Ray Actor for trajectory collection.

This module implements a RolloutWorker that maintains a persistent environment
instance across training iterations, avoiding the expensive environment
recreation that occurs with stateless Ray remote functions.
"""

import traceback
from copy import deepcopy

import ray
import torch

from rl.envs.vectorized_env import VectorizedEnv
from rl.storage.rollout_storage import BatchData, PPOBuffer
from rl.utils.seeding import set_global_seeds


class RolloutWorkerError(Exception):
    """Exception raised when a RolloutWorker encounters an error during sampling."""

    pass


@ray.remote
class RolloutWorker:
    """Persistent worker that holds an environment for trajectory collection.

    Instead of recreating the environment on every sample() call (which involves
    expensive MuJoCo model compilation), this actor creates the environment once
    and reuses it across all training iterations.

    Supports vectorized environments for batched policy inference.
    """

    def __init__(self, env_fn, policy_template, critic_template, num_envs_per_worker=1, seed=None, worker_id=0):
        """Initialize worker with persistent environment(s).

        Args:
            env_fn: Factory function to create the environment (called once per env)
            policy_template: Policy network to clone for local inference
            critic_template: Critic network to clone for local inference
            num_envs_per_worker: Number of environments per worker (default: 1 for backward compatibility)
            seed: Worker-specific seed for reproducibility
            worker_id: Identifier for this worker (for error reporting)
        """
        self.seed = seed
        self.worker_id = worker_id

        # Seed worker-local RNG BEFORE creating environment
        if seed is not None:
            set_global_seeds(seed, cuda_deterministic=False)

        # Create environment(s) ONCE - this is the key optimization
        self.num_envs_per_worker = num_envs_per_worker

        if num_envs_per_worker > 1:
            # Use vectorized environment for batched inference
            self.env = VectorizedEnv(env_fn, num_envs_per_worker)
            self.is_vectorized = True
        else:
            # Single environment (backward compatibility)
            self.env = env_fn()
            self.is_vectorized = False

        # Create local copies of networks for inference
        # These will be updated via set_weights() before each sampling round
        self.policy = deepcopy(policy_template)
        self.critic = deepcopy(critic_template)

        # Cache dimensions
        self.state_dim = self.policy.state_dim
        self.action_dim = self.policy.action_dim

        # Track current episode state for persistence across sample() calls
        # If an episode is ongoing when buffer fills, we continue it next time
        self.current_state = None
        self.current_traj_len = 0
        # Track episode stats incrementally (for logging)
        self.current_ep_reward = 0.0
        self.current_ep_len = 0

    def sync_state(self, policy_state_dict, critic_state_dict, obs_mean, obs_std, iteration_count):
        """Sync all worker state from main process in a single call.

        Combines weight updates, observation normalization, and iteration count
        into one remote call to minimize Ray communication overhead.

        Args:
            policy_state_dict: Policy network state dict
            critic_state_dict: Critic network state dict
            obs_mean: Mean tensor for observation normalization
            obs_std: Std tensor for observation normalization
            iteration_count: Current training iteration (for curriculum learning)
        """
        # Update network weights
        self.policy.load_state_dict(policy_state_dict)
        self.critic.load_state_dict(critic_state_dict)

        # Update observation normalization
        self.policy.obs_mean = obs_mean
        self.policy.obs_std = obs_std
        self.critic.obs_mean = obs_mean
        self.critic.obs_std = obs_std

        # Update iteration count for curriculum learning
        if self.is_vectorized:
            self.env.set_iteration_count(iteration_count)
        else:
            self.env.robot.iteration_count = iteration_count

    @torch.no_grad()
    def sample(self, gamma, max_steps, max_traj_len, deterministic=False):
        """Collect trajectory data using the persistent environment(s).

        Collects exactly max_steps timesteps. Episodes may span multiple sample()
        calls - if the buffer fills mid-episode, the episode continues on the
        next call.

        Args:
            gamma: Discount factor for returns
            max_steps: Maximum number of timesteps to collect
            max_traj_len: Maximum length of a single trajectory
            deterministic: Whether to use deterministic actions

        Returns:
            dict: Collected trajectory data (states, actions, rewards, etc.)

        Raises:
            RolloutWorkerError: If an error occurs during sampling, with context
                about which worker failed and the original exception.
        """
        if self.is_vectorized:
            return self._sample_vectorized(gamma, max_steps, max_traj_len, deterministic)
        else:
            return self._sample_single(gamma, max_steps, max_traj_len, deterministic)

    def _sample_single(self, gamma, max_steps, max_traj_len, deterministic):
        """Original single-environment sampling (backward compatibility)."""
        policy = self.policy
        critic = self.critic
        env = self.env

        memory = PPOBuffer(self.state_dim, self.action_dim, gamma=gamma, size=max_steps)
        completed_ep_lens = []
        completed_ep_rewards = []

        try:
            # Initialize or continue from previous episode state
            if self.current_state is None:
                state = torch.as_tensor(env.reset(), dtype=torch.float)
                self.current_traj_len = 0
                self.current_ep_reward = 0.0
                self.current_ep_len = 0
                if hasattr(policy, "init_hidden_state"):
                    policy.init_hidden_state()
                if hasattr(critic, "init_hidden_state"):
                    critic.init_hidden_state()
            else:
                state = self.current_state

            # Collect exactly max_steps timesteps
            while len(memory) < max_steps:
                action = policy(state, deterministic=deterministic)
                value = critic(state)

                next_state, reward, done, _ = env.step(action.numpy())
                self.current_traj_len += 1
                self.current_ep_len += 1
                self.current_ep_reward += float(reward)

                # Check if trajectory reached max length (truncation)
                truncated = self.current_traj_len >= max_traj_len
                episode_ended = done or truncated

                reward = torch.as_tensor(reward, dtype=torch.float)
                memory.store(state, action, reward, value, episode_ended)

                if episode_ended:
                    # Record completed episode stats
                    completed_ep_lens.append(self.current_ep_len)
                    completed_ep_rewards.append(self.current_ep_reward)

                    # Compute returns for this trajectory
                    # Bootstrap with value estimate if truncated, 0 if truly done
                    next_state_tensor = torch.as_tensor(next_state, dtype=torch.float)
                    bootstrap_value = (not done) * critic(next_state_tensor)
                    memory.finish_path(last_val=bootstrap_value)

                    # Reset for new episode
                    state = torch.as_tensor(env.reset(), dtype=torch.float)
                    self.current_traj_len = 0
                    self.current_ep_reward = 0.0
                    self.current_ep_len = 0
                    if hasattr(policy, "init_hidden_state"):
                        policy.init_hidden_state()
                    if hasattr(critic, "init_hidden_state"):
                        critic.init_hidden_state()
                else:
                    state = torch.as_tensor(next_state, dtype=torch.float)

            # Handle case where buffer filled mid-episode
            # Check if the last transition was not an episode end
            if not memory.dones[memory.ptr - 1]:
                # Episode is ongoing - finish path with bootstrap and save state
                bootstrap_value = critic(state)
                memory.finish_path(last_val=bootstrap_value)
                self.current_state = state
            else:
                # Episode ended cleanly, no state to preserve
                self.current_state = None

            return memory.get_data(ep_lens=completed_ep_lens, ep_rewards=completed_ep_rewards)

        except Exception as e:
            tb = traceback.format_exc()
            raise RolloutWorkerError(
                f"Worker {self.worker_id} failed at step {len(memory)}, "
                f"ep_len={self.current_ep_len}: {type(e).__name__}: {e}\n{tb}"
            ) from e

    def _sample_vectorized(self, gamma, max_steps, max_traj_len, deterministic):
        """Vectorized environment sampling with batched policy inference."""
        policy = self.policy
        critic = self.critic
        vec_env = self.env
        num_envs = self.num_envs_per_worker

        # Create separate buffer for each environment
        buffers = [
            PPOBuffer(self.state_dim, self.action_dim, gamma=gamma, size=max_traj_len * 2) for _ in range(num_envs)
        ]

        # Reset all environments
        states = torch.as_tensor(vec_env.reset_all(), dtype=torch.float)  # [num_envs, obs_dim]
        traj_lens = torch.zeros(num_envs, dtype=torch.int32)
        ep_lens_per_env = [[] for _ in range(num_envs)]
        ep_rewards_per_env = [[] for _ in range(num_envs)]
        ep_reward_accum = [0.0] * num_envs
        ep_len_accum = [0] * num_envs

        # For recurrent policies
        if hasattr(policy, "init_hidden_state"):
            # Note: This might need modification for vectorized recurrent policies
            # Currently resets to same initial state for all envs
            policy.init_hidden_state()

        if hasattr(critic, "init_hidden_state"):
            critic.init_hidden_state()

        total_steps = 0
        while total_steps < max_steps:
            # Batched policy inference - KEY OPTIMIZATION
            actions = policy(states, deterministic=deterministic)  # [num_envs, act_dim]
            values = critic(states)  # [num_envs, 1]

            # Step all environments
            next_states, rewards, dones, infos = vec_env.step(actions)

            # Convert to tensors
            next_states = torch.as_tensor(next_states, dtype=torch.float)
            rewards = torch.as_tensor(rewards, dtype=torch.float)

            # Store in individual buffers
            for i in range(num_envs):
                buffers[i].store(
                    states[i],
                    actions[i],
                    rewards[i],
                    values[i],
                    bool(dones[i]),  # Convert numpy.bool_ to Python bool
                )
                traj_lens[i] += 1
                ep_len_accum[i] += 1
                ep_reward_accum[i] += float(rewards[i])

                # Finish trajectory if done or max length reached
                if dones[i] or traj_lens[i] >= max_traj_len:
                    # Use terminal observation for bootstrapping (not the auto-reset obs)
                    if dones[i] and "terminal_observation" in infos[i]:
                        terminal_obs = torch.as_tensor(infos[i]["terminal_observation"], dtype=torch.float)
                        final_value = critic(terminal_obs.unsqueeze(0)).squeeze(0)
                    else:
                        final_value = critic(next_states[i].unsqueeze(0)).squeeze(0)
                    # Bootstrap with final value if not done (truncated episode)
                    buffers[i].finish_path(last_val=(not dones[i]) * final_value)
                    traj_lens[i] = 0

                    # Record completed episode stats
                    ep_lens_per_env[i].append(ep_len_accum[i])
                    ep_rewards_per_env[i].append(ep_reward_accum[i])
                    ep_len_accum[i] = 0
                    ep_reward_accum[i] = 0.0

                    # Reset hidden states for recurrent policies
                    # Note: This is a simplification - ideally we'd track hidden states per env
                    if hasattr(policy, "init_hidden_state"):
                        policy.init_hidden_state()
                    if hasattr(critic, "init_hidden_state"):
                        critic.init_hidden_state()

            states = next_states
            total_steps += num_envs

        # Finish any incomplete trajectories before aggregating
        # This is critical: without finish_path(), returns are never computed!
        for i in range(num_envs):
            # Check if buffer has unfinished trajectory (data after last finish_path call)
            if len(buffers[i]) > 0 and len(buffers[i].traj_idx) > 0:
                # Check if there's data after the last trajectory marker
                last_traj_end = int(buffers[i].traj_idx[-1])
                if buffers[i].ptr > last_traj_end:
                    # Unfinished trajectory exists - compute final value and finish it
                    final_value = critic(states[i].unsqueeze(0)).squeeze(0)
                    # Bootstrap with final value (trajectory was truncated, not done)
                    buffers[i].finish_path(last_val=final_value)

        # Flatten episode stats across all envs
        all_ep_lens = [ep_len for per_env in ep_lens_per_env for ep_len in per_env]
        all_ep_rewards = [r for per_env in ep_rewards_per_env for r in per_env]

        # Aggregate all buffers
        return self._aggregate_buffers(buffers, all_ep_lens, all_ep_rewards)

    def _aggregate_buffers(self, buffers, ep_lens=None, ep_rewards=None):
        """Aggregate multiple PPOBuffer instances into a single BatchData.

        Args:
            buffers: List of PPOBuffer instances
            ep_lens: Completed episode lengths (from vectorized sampling)
            ep_rewards: Completed episode rewards (from vectorized sampling)

        Returns:
            BatchData: Aggregated trajectory data
        """
        # Get data from all buffers (episode stats handled separately)
        buffer_data = [buf.get_data() for buf in buffers]

        # Fix traj_idx: offset each buffer's indices by cumulative sample count
        traj_idx_list = []
        offset = 0
        for data in buffer_data:
            worker_traj_idx = data.traj_idx
            # Skip the first 0 from subsequent buffers (it's redundant)
            if offset > 0:
                worker_traj_idx = worker_traj_idx[1:]
            traj_idx_list.append(worker_traj_idx + offset)
            offset += len(data.states)

        return BatchData(
            states=torch.cat([d.states for d in buffer_data]),
            actions=torch.cat([d.actions for d in buffer_data]),
            rewards=torch.cat([d.rewards for d in buffer_data]),
            values=torch.cat([d.values for d in buffer_data]),
            returns=torch.cat([d.returns for d in buffer_data]),
            dones=torch.cat([d.dones for d in buffer_data]),
            traj_idx=torch.cat(traj_idx_list),
            ep_lens=torch.tensor(ep_lens) if ep_lens is not None else torch.cat([d.ep_lens for d in buffer_data]),
            ep_rewards=(
                torch.tensor(ep_rewards) if ep_rewards is not None else torch.cat([d.ep_rewards for d in buffer_data])
            ),
        )

    def get_env_info(self):
        """Return environment observation/action space info."""
        return {
            "obs_dim": self.env.observation_space.shape[0],
            "action_dim": self.env.action_space.shape[0],
        }
