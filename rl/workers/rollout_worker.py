"""Persistent Ray Actor for trajectory collection.

This module implements a RolloutWorker that maintains a persistent environment
instance across training iterations, avoiding the expensive environment
recreation that occurs with stateless Ray remote functions.
"""

from copy import deepcopy

import ray
import torch

from rl.envs.vectorized_env import VectorizedEnv
from rl.storage.rollout_storage import PPOBuffer
from rl.utils.seeding import set_global_seeds


@ray.remote
class RolloutWorker:
    """Persistent worker that holds an environment for trajectory collection.

    Instead of recreating the environment on every sample() call (which involves
    expensive MuJoCo model compilation), this actor creates the environment once
    and reuses it across all training iterations.

    Supports vectorized environments for batched policy inference.
    """

    def __init__(self, env_fn, policy_template, critic_template, num_envs_per_worker=1, seed=None):
        """Initialize worker with persistent environment(s).

        Args:
            env_fn: Factory function to create the environment (called once per env)
            policy_template: Policy network to clone for local inference
            critic_template: Critic network to clone for local inference
            num_envs_per_worker: Number of environments per worker (default: 1 for backward compatibility)
            seed: Worker-specific seed for reproducibility
        """
        self.seed = seed

        # Seed worker-local RNG BEFORE creating environment
        if seed is not None:
            set_global_seeds(seed, cuda_deterministic=False)

        # Create environment ONCE - this is the key optimization
        # Global RNG is already seeded above, so env will use seeded np.random
        self.env = env_fn()

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

    def set_weights(self, policy_state_dict, critic_state_dict):
        """Update local network weights from main process.

        This is much cheaper than pickling entire networks because:
        1. state_dict is just tensors (no graph structure, no Python objects)
        2. Ray can use shared memory for tensor transfer

        Args:
            policy_state_dict: Policy network state dict
            critic_state_dict: Critic network state dict
        """
        self.policy.load_state_dict(policy_state_dict)
        self.critic.load_state_dict(critic_state_dict)

    def set_iteration_count(self, iteration_count):
        """Update iteration count for curriculum learning."""
        if self.is_vectorized:
            self.env.set_iteration_count(iteration_count)
        else:
            self.env.robot.iteration_count = iteration_count

    @torch.no_grad()
    def sample(self, gamma, lam, max_steps, max_traj_len, deterministic=False):
        """Collect trajectory data using the persistent environment(s).

        Args:
            gamma: Discount factor for returns
            lam: GAE lambda (unused currently but kept for compatibility)
            max_steps: Maximum number of timesteps to collect
            max_traj_len: Maximum length of a single trajectory
            deterministic: Whether to use deterministic actions

        Returns:
            dict: Collected trajectory data (states, actions, rewards, etc.)
        """
        if self.is_vectorized:
            return self._sample_vectorized(gamma, lam, max_steps, max_traj_len, deterministic)
        else:
            return self._sample_single(gamma, lam, max_steps, max_traj_len, deterministic)

    def _sample_single(self, gamma, lam, max_steps, max_traj_len, deterministic):
        """Original single-environment sampling (backward compatibility)."""
        policy = self.policy
        critic = self.critic
        env = self.env

        memory = PPOBuffer(self.state_dim, self.action_dim, gamma, lam, size=max_traj_len * 2)
        memory_full = False

        while not memory_full:
            state = torch.as_tensor(env.reset(), dtype=torch.float)
            done = False
            traj_len = 0

            if hasattr(policy, "init_hidden_state"):
                policy.init_hidden_state()

            if hasattr(critic, "init_hidden_state"):
                critic.init_hidden_state()

            while not done and traj_len < max_traj_len:
                action = policy(state, deterministic=deterministic)
                value = critic(state)

                next_state, reward, done, _ = env.step(action.numpy())

                reward = torch.as_tensor(reward, dtype=torch.float)
                memory.store(state, action, reward, value, done)
                memory_full = len(memory) >= max_steps

                state = torch.as_tensor(next_state, dtype=torch.float)
                traj_len += 1

            value = critic(state)
            memory.finish_path(last_val=(not done) * value)

        return memory.get_data()

    def _sample_vectorized(self, gamma, lam, max_steps, max_traj_len, deterministic):
        """Vectorized environment sampling with batched policy inference."""
        policy = self.policy
        critic = self.critic
        vec_env = self.env
        num_envs = self.num_envs_per_worker

        # Create separate buffer for each environment
        buffers = [
            PPOBuffer(self.state_dim, self.action_dim, gamma, lam, size=max_traj_len * 2) for _ in range(num_envs)
        ]

        # Reset all environments
        states = torch.as_tensor(vec_env.reset_all(), dtype=torch.float)  # [num_envs, obs_dim]
        traj_lens = torch.zeros(num_envs, dtype=torch.int32)

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
            next_states, rewards, dones, _ = vec_env.step(actions)

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

                # Finish trajectory if done or max length reached
                if dones[i] or traj_lens[i] >= max_traj_len:
                    # Compute final value for bootstrapping
                    final_value = critic(next_states[i].unsqueeze(0)).squeeze(0)
                    # Bootstrap with final value if not done (truncated episode)
                    buffers[i].finish_path(last_val=(not dones[i]) * final_value)
                    traj_lens[i] = 0

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

        # Aggregate all buffers
        return self._aggregate_buffers(buffers)

    def _aggregate_buffers(self, buffers):
        """Aggregate multiple PPOBuffer instances into a single data dictionary.

        Args:
            buffers: List of PPOBuffer instances

        Returns:
            dict: Aggregated trajectory data
        """
        # Get data from all buffers
        buffer_data = [buf.get_data() for buf in buffers]

        # Standard data keys to concatenate
        data_keys = ["states", "actions", "rewards", "values", "returns", "dones"]
        aggregated_data = {k: torch.cat([d[k] for d in buffer_data]) for k in data_keys}

        # Concatenate episode metrics
        aggregated_data["ep_lens"] = torch.cat([d["ep_lens"] for d in buffer_data])
        aggregated_data["ep_rewards"] = torch.cat([d["ep_rewards"] for d in buffer_data])

        # Fix traj_idx: offset each buffer's indices by cumulative sample count
        traj_idx_list = []
        offset = 0
        for data in buffer_data:
            worker_traj_idx = data["traj_idx"]
            # Skip the first 0 from subsequent buffers (it's redundant)
            if offset > 0:
                worker_traj_idx = worker_traj_idx[1:]
            traj_idx_list.append(worker_traj_idx + offset)
            offset += len(data["states"])

        aggregated_data["traj_idx"] = torch.cat(traj_idx_list)

        return aggregated_data

    def get_env_info(self):
        """Return environment observation/action space info."""
        return {
            "obs_dim": self.env.observation_space.shape[0],
            "action_dim": self.env.action_space.shape[0],
        }
