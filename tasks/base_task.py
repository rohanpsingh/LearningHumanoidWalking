"""Base class for all task implementations.

This module defines the TaskBase abstract base class that all task
implementations must inherit from.
"""

from abc import ABC, abstractmethod

import numpy as np


class BaseTask(ABC):
    """Abstract base class defining the task interface.

    All task implementations must inherit from this class and implement
    the required abstract methods. This ensures that tasks can be used
    interchangeably with RobotBase and environment classes.

    The task lifecycle is:
    1. __init__: Create the task with required configuration
    2. reset(): Called at episode start to initialize task state
    3. step() + calc_reward() + done(): Called each control step

    Attributes:
        _client: Reference to the RobotInterface for accessing robot state.
    """

    @abstractmethod
    def reset(self, iter_count: int = 0) -> None:
        """Reset task state for a new episode.

        Called at the beginning of each episode to reinitialize task state.
        May use iter_count for curriculum learning.

        Args:
            iter_count: Current training iteration, can be used for curriculum.
        """
        pass

    @abstractmethod
    def step(self) -> None:
        """Update task state during each control step.

        Called once per control step to update internal task state
        (e.g., phase tracking, goal updates).
        """
        pass

    @abstractmethod
    def calc_reward(
        self,
        prev_torque: np.ndarray,
        prev_action: np.ndarray,
        action: np.ndarray,
    ) -> dict[str, float]:
        """Calculate reward components for the current step.

        Args:
            prev_torque: Joint torques from the previous step.
            prev_action: Action from the previous step.
            action: Current action.

        Returns:
            Dictionary mapping reward component names to their values.
            The total reward is typically computed as sum of all components.
        """
        pass

    @abstractmethod
    def done(self) -> bool:
        """Check if the episode should terminate.

        Returns:
            True if the episode should terminate, False otherwise.
        """
        pass

    def substep(self) -> None:  # noqa: B027
        """Optional hook for sub-control-step updates.

        Called for updates at a finer granularity than the control step.
        Default implementation does nothing.
        """
