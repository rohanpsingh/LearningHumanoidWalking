"""Task definitions for humanoid robot learning.

This package provides task implementations that define objectives,
rewards, and termination conditions for training humanoid robots.

All tasks inherit from TaskBase which defines the required interface
for interaction with RobotBase and environment classes.
"""

from tasks.base_task import BaseTask
from tasks.standing_task import StandingTask
from tasks.stepping_task import SteppingTask
from tasks.walking_task import WalkingTask

__all__ = [
    "BaseTask",
    "WalkingTask",
    "SteppingTask",
    "StandingTask",
]
