"""Observation term functions for humanoid environments.

This module provides functions to compute various observation terms
from the robot state. Each function takes a RobotInterface and returns
the corresponding observation as a numpy array.
"""

import numpy as np
import transforms3d as tf3


def get_root_orientation(interface):
    """Get root roll and pitch angles.

    Args:
        interface: RobotInterface instance.

    Returns:
        Tuple of (roll, pitch) as 1D arrays.
    """
    qpos = np.copy(interface.get_qpos())
    root_r, root_p = tf3.euler.quat2euler(qpos[3:7])[0:2]
    return np.array([root_r]), np.array([root_p])


def get_root_angular_velocity(interface):
    """Get root angular velocity.

    Args:
        interface: RobotInterface instance.

    Returns:
        3D angular velocity vector.
    """
    qvel = np.copy(interface.get_qvel())
    return qvel[3:6]


def get_motor_positions(interface):
    """Get motor/actuator positions at joint level.

    Args:
        interface: RobotInterface instance.

    Returns:
        Array of joint positions.
    """
    return interface.get_act_joint_positions()


def get_motor_velocities(interface):
    """Get motor/actuator velocities at joint level.

    Args:
        interface: RobotInterface instance.

    Returns:
        Array of joint velocities.
    """
    return interface.get_act_joint_velocities()


def get_motor_torques(interface):
    """Get motor/actuator torques at joint level.

    Args:
        interface: RobotInterface instance.

    Returns:
        Array of joint torques.
    """
    return interface.get_act_joint_torques()
