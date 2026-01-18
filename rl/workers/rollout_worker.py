"""Persistent Ray Actor for trajectory collection.

This module implements a RolloutWorker that maintains a persistent environment
instance across training iterations, avoiding the expensive environment
recreation that occurs with stateless Ray remote functions.
"""

import traceback
from copy import deepcopy

import ray
import torch

from rl.storage.rollout_storage import PPOBuffer
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
    """

    def __init__(self, env_fn, policy_template, critic_template, seed=None, worker_id=0):
        """Initialize worker with persistent environment.

        Args:
            env_fn: Factory function to create the environment (called once)
            policy_template: Policy network to clone for local inference
            critic_template: Critic network to clone for local inference
            seed: Worker-specific seed for reproducibility
            worker_id: Identifier for this worker (for error reporting)
        """
        self.seed = seed
        self.worker_id = worker_id

        # Seed worker-local RNG BEFORE creating environment
        if seed is not None:
            set_global_seeds(seed, cuda_deterministic=False)

        # Create environment ONCE - this is the key optimization
        # Global RNG is already seeded above, so env will use seeded np.random
        self.env = env_fn()

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
        self.env.robot.iteration_count = iteration_count

    @torch.no_grad()
    def sample(self, gamma, lam, max_steps, max_traj_len, deterministic=False):
        """Collect trajectory data using the persistent environment.

        Collects exactly max_steps timesteps. Episodes may span multiple sample()
        calls - if the buffer fills mid-episode, the episode continues on the
        next call.

        Args:
            gamma: Discount factor for returns
            lam: GAE lambda (unused currently but kept for compatibility)
            max_steps: Maximum number of timesteps to collect
            max_traj_len: Maximum length of a single trajectory
            deterministic: Whether to use deterministic actions

        Returns:
            dict: Collected trajectory data (states, actions, rewards, etc.)

        Raises:
            RolloutWorkerError: If an error occurs during sampling, with context
                about which worker failed and the original exception.
        """
        policy = self.policy
        critic = self.critic
        env = self.env

        memory = PPOBuffer(self.state_dim, self.action_dim, gamma, lam, size=max_steps)
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
