"""Vectorized environment wrapper for parallel environment execution.

This module implements a vectorized environment that manages multiple environment
instances and enables batched policy inference for improved efficiency.
"""

import numpy as np
import torch


class VectorizedEnv:
    """Manages multiple environments with synchronized stepping.

    This wrapper allows batched policy inference across multiple environments,
    which is significantly faster than sequential inference, especially on GPU.

    Key benefits:
    - Batched policy inference (4x faster for 4 envs)
    - Better GPU utilization
    - Reduced per-step inference overhead

    Args:
        env_fn: Factory function to create a single environment
        num_envs: Number of environments to run in parallel
    """

    def __init__(self, env_fn, num_envs):
        """Initialize vectorized environment with multiple instances.

        Args:
            env_fn: Callable that returns a new environment instance
            num_envs: Number of parallel environments
        """
        self.envs = [env_fn() for _ in range(num_envs)]
        self.num_envs = num_envs

        # Cache environment spaces from first env
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space

        # Track which environments have terminated
        self.dones = np.zeros(num_envs, dtype=bool)

    def reset(self, env_ids=None):
        """Reset specific environments or all environments.

        Args:
            env_ids: List/array of environment indices to reset, or None for all

        Returns:
            np.ndarray: Stacked observations [num_envs, obs_dim]
        """
        if env_ids is None:
            env_ids = range(self.num_envs)

        obs_list = []
        for i in env_ids:
            obs = self.envs[i].reset()
            obs_list.append(obs)
            self.dones[i] = False

        return np.stack(obs_list)

    def reset_all(self):
        """Reset all environments.

        Returns:
            np.ndarray: Stacked observations [num_envs, obs_dim]
        """
        return self.reset(env_ids=None)

    def step(self, actions):
        """Step all environments with batched actions.

        Episodes are automatically reset when they terminate, making this
        wrapper suitable for continuous sampling.

        Args:
            actions: Batched actions [num_envs, action_dim] or list of actions

        Returns:
            tuple: (observations, rewards, dones, infos)
                - observations: np.ndarray [num_envs, obs_dim]
                - rewards: np.ndarray [num_envs]
                - dones: np.ndarray [num_envs] (bool)
                - infos: list of info dicts
        """
        obs_list, rewards, dones, infos = [], [], [], []

        # Convert to numpy if tensor
        if torch.is_tensor(actions):
            actions = actions.cpu().numpy()

        for _i, (env, action) in enumerate(zip(self.envs, actions, strict=False)):
            obs, reward, done, info = env.step(action)

            # Auto-reset on episode termination
            if done:
                obs = env.reset()

            obs_list.append(obs)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)

        return (
            np.stack(obs_list),  # [num_envs, obs_dim]
            np.array(rewards),  # [num_envs]
            np.array(dones),  # [num_envs]
            infos,  # list of dicts
        )

    def set_iteration_count(self, iteration_count):
        """Update iteration count on all environments for curriculum learning.

        Args:
            iteration_count: Current training iteration
        """
        for env in self.envs:
            if hasattr(env, "robot"):
                env.robot.iteration_count = iteration_count

    def close(self):
        """Close all environments."""
        for env in self.envs:
            env.close()

    def __len__(self):
        """Return number of environments."""
        return self.num_envs
