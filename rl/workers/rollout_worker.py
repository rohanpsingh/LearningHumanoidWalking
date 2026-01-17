"""Persistent Ray Actor for trajectory collection.

This module implements a RolloutWorker that maintains a persistent environment
instance across training iterations, avoiding the expensive environment
recreation that occurs with stateless Ray remote functions.
"""
import torch
import ray
from copy import deepcopy

from rl.storage.rollout_storage import PPOBuffer
from rl.utils.seeding import set_global_seeds


@ray.remote
class RolloutWorker:
    """Persistent worker that holds an environment for trajectory collection.

    Instead of recreating the environment on every sample() call (which involves
    expensive MuJoCo model compilation), this actor creates the environment once
    and reuses it across all training iterations.
    """

    def __init__(self, env_fn, policy_template, critic_template, seed=None):
        """Initialize worker with persistent environment.

        Args:
            env_fn: Factory function to create the environment (called once)
            policy_template: Policy network to clone for local inference
            critic_template: Critic network to clone for local inference
            seed: Worker-specific seed for reproducibility
        """
        self.seed = seed

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
        self.env.robot.iteration_count = iteration_count

    @torch.no_grad()
    def sample(self, gamma, lam, max_steps, max_traj_len, deterministic=False):
        """Collect trajectory data using the persistent environment.

        Args:
            gamma: Discount factor for returns
            lam: GAE lambda (unused currently but kept for compatibility)
            max_steps: Maximum number of timesteps to collect
            max_traj_len: Maximum length of a single trajectory
            deterministic: Whether to use deterministic actions

        Returns:
            dict: Collected trajectory data (states, actions, rewards, etc.)
        """
        policy = self.policy
        critic = self.critic
        env = self.env

        memory = PPOBuffer(self.state_dim, self.action_dim, gamma, lam, size=max_traj_len*2)
        memory_full = False

        while not memory_full:
            state = torch.as_tensor(env.reset(), dtype=torch.float)
            done = False
            traj_len = 0

            if hasattr(policy, 'init_hidden_state'):
                policy.init_hidden_state()

            if hasattr(critic, 'init_hidden_state'):
                critic.init_hidden_state()

            while not done and traj_len < max_traj_len:
                action = policy(state, deterministic=deterministic)
                value = critic(state)

                next_state, reward, done, _ = env.step(action.numpy())

                reward = torch.as_tensor(reward, dtype=torch.float)
                memory.store(state, action, reward, value, done)
                memory_full = (len(memory) >= max_steps)

                state = torch.as_tensor(next_state, dtype=torch.float)
                traj_len += 1

            value = critic(state)
            memory.finish_path(last_val=(not done) * value)

        return memory.get_data()

    def get_env_info(self):
        """Return environment observation/action space info."""
        return {
            'obs_dim': self.env.observation_space.shape[0],
            'action_dim': self.env.action_space.shape[0],
        }
