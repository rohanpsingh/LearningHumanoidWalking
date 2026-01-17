"""Reward functions for locomotion tasks.

Functions return scalar reward values, typically in range [0, 1] for exponential rewards.
"""
import numpy as np


def calc_fwd_vel_reward(root_vel: float, goal_speed: float) -> float:
    """Reward for forward velocity tracking.

    Args:
        root_vel: Current forward velocity of the root body.
        goal_speed: Target forward velocity.

    Returns:
        Exponential reward based on velocity error.
    """
    error = np.abs(root_vel - goal_speed)
    return np.exp(-10 * (error ** 2))


def calc_yaw_vel_reward(yaw_vel: float, yaw_vel_ref: float = 0) -> float:
    """Reward for yaw angular velocity tracking.

    Args:
        yaw_vel: Current yaw angular velocity.
        yaw_vel_ref: Target yaw angular velocity.

    Returns:
        Exponential reward based on yaw velocity error.
    """
    error = np.abs(yaw_vel - yaw_vel_ref)
    return np.exp(-10 * (error ** 3))


def calc_action_reward(action: np.ndarray, prev_action: np.ndarray) -> float:
    """Reward for smooth actions (penalizes action changes).

    Args:
        action: Current action.
        prev_action: Previous action.

    Returns:
        Exponential reward penalizing action discontinuities.
    """
    penalty = 5 * np.sum(np.abs(prev_action - action)) / len(action)
    return np.exp(-penalty)


def calc_torque_reward(torque: np.ndarray, prev_torque: np.ndarray) -> float:
    """Reward for smooth torques (penalizes torque changes).

    Args:
        torque: Current joint torques.
        prev_torque: Previous joint torques.

    Returns:
        Exponential reward penalizing torque discontinuities.
    """
    penalty = 0.25 * (np.sum(np.abs(prev_torque - torque)) / len(torque))
    return np.exp(-penalty)


def calc_height_reward(
    current_height: float,
    goal_height: float,
    goal_speed: float,
    contact_point_z: float = 0,
) -> float:
    """Reward for maintaining target height.

    Args:
        current_height: Current height of root body.
        goal_height: Target height.
        goal_speed: Current goal speed (affects deadzone size).
        contact_point_z: Z position of lowest contact point (0 if no contact).

    Returns:
        Exponential reward based on height error.
    """
    relative_height = current_height - contact_point_z
    error = np.abs(relative_height - goal_height)
    deadzone_size = 0.01 + 0.05 * goal_speed
    if error < deadzone_size:
        error = 0
    return np.exp(-40 * np.square(error))


def calc_root_accel_reward(qvel: np.ndarray, qacc: np.ndarray) -> float:
    """Reward for minimizing root body acceleration and angular velocity.

    Args:
        qvel: Generalized velocities (expecting indices 3:6 for angular).
        qacc: Generalized accelerations (expecting indices 0:3 for linear).

    Returns:
        Exponential reward penalizing acceleration.
    """
    error = 0.25 * (np.abs(qvel[3:6]).sum() + np.abs(qacc[0:3]).sum())
    return np.exp(-error)


def calc_foot_frc_clock_reward(
    l_foot_frc: float,
    r_foot_frc: float,
    phase: int,
    left_frc_fn,
    right_frc_fn,
    robot_mass: float,
) -> float:
    """Reward for foot forces matching the gait phase clock.

    Args:
        l_foot_frc: Left foot ground reaction force magnitude.
        r_foot_frc: Right foot ground reaction force magnitude.
        phase: Current gait phase.
        left_frc_fn: Phase function for left foot force.
        right_frc_fn: Phase function for right foot force.
        robot_mass: Total robot mass (for normalization).

    Returns:
        Score based on force-phase alignment.
    """
    desired_max_foot_frc = robot_mass * 9.8 * 0.5
    normed_left_frc = min(l_foot_frc, desired_max_foot_frc) / desired_max_foot_frc
    normed_right_frc = min(r_foot_frc, desired_max_foot_frc) / desired_max_foot_frc
    normed_left_frc = normed_left_frc * 2 - 1
    normed_right_frc = normed_right_frc * 2 - 1

    left_frc_clock = left_frc_fn(phase)
    right_frc_clock = right_frc_fn(phase)

    left_frc_score = np.tan(np.pi / 4 * left_frc_clock * normed_left_frc)
    right_frc_score = np.tan(np.pi / 4 * right_frc_clock * normed_right_frc)

    return (left_frc_score + right_frc_score) / 2


def calc_foot_vel_clock_reward(
    l_foot_vel: np.ndarray,
    r_foot_vel: np.ndarray,
    phase: int,
    left_vel_fn,
    right_vel_fn,
) -> float:
    """Reward for foot velocities matching the gait phase clock.

    Args:
        l_foot_vel: Left foot velocity vector.
        r_foot_vel: Right foot velocity vector.
        phase: Current gait phase.
        left_vel_fn: Phase function for left foot velocity.
        right_vel_fn: Phase function for right foot velocity.

    Returns:
        Score based on velocity-phase alignment.
    """
    desired_max_foot_vel = 0.2
    normed_left_vel = min(np.linalg.norm(l_foot_vel), desired_max_foot_vel) / desired_max_foot_vel
    normed_right_vel = min(np.linalg.norm(r_foot_vel), desired_max_foot_vel) / desired_max_foot_vel
    normed_left_vel = normed_left_vel * 2 - 1
    normed_right_vel = normed_right_vel * 2 - 1

    left_vel_clock = left_vel_fn(phase)
    right_vel_clock = right_vel_fn(phase)

    left_vel_score = np.tan(np.pi / 4 * left_vel_clock * normed_left_vel)
    right_vel_score = np.tan(np.pi / 4 * right_vel_clock * normed_right_vel)

    return (left_vel_score + right_vel_score) / 2


def calc_body_orient_reward(
    body_quat: np.ndarray,
    target_quat: np.ndarray = None,
) -> float:
    """Reward for body orientation matching target.

    Args:
        body_quat: Current body quaternion [w, x, y, z].
        target_quat: Target quaternion [w, x, y, z]. Defaults to identity.

    Returns:
        Exponential reward based on orientation error.
    """
    if target_quat is None:
        target_quat = np.array([1, 0, 0, 0])
    error = 10 * (1 - np.inner(target_quat, body_quat) ** 2)
    return np.exp(-error)


def create_phase_reward(swing_duration, stance_duration, strict_relaxer, stance_mode, FREQ=40):
    """Create phase-based reward functions for gait timing.

    Args:
        swing_duration: Duration of swing phase in seconds.
        stance_duration: Duration of stance phase in seconds.
        strict_relaxer: Fraction to relax phase boundaries.
        stance_mode: One of "grounded", "aerial", or "zero".
        FREQ: Control frequency in Hz.

    Returns:
        Tuple of (right_clock, left_clock) where each is [force_fn, velocity_fn].
    """
    from scipy.interpolate import PchipInterpolator

    right_swing = np.array([0.0, swing_duration]) * FREQ
    first_dblstance = np.array([swing_duration, swing_duration + stance_duration]) * FREQ
    left_swing = np.array([swing_duration + stance_duration, 2 * swing_duration + stance_duration]) * FREQ
    second_dblstance = np.array([2 * swing_duration + stance_duration, 2 * (swing_duration + stance_duration)]) * FREQ

    r_frc_phase_points = np.zeros((2, 8))
    r_vel_phase_points = np.zeros((2, 8))
    l_frc_phase_points = np.zeros((2, 8))
    l_vel_phase_points = np.zeros((2, 8))

    right_swing_relax_offset = (right_swing[1] - right_swing[0]) * strict_relaxer
    l_frc_phase_points[0, 0] = r_frc_phase_points[0, 0] = right_swing[0] + right_swing_relax_offset
    l_frc_phase_points[0, 1] = r_frc_phase_points[0, 1] = right_swing[1] - right_swing_relax_offset
    l_vel_phase_points[0, 0] = r_vel_phase_points[0, 0] = right_swing[0] + right_swing_relax_offset
    l_vel_phase_points[0, 1] = r_vel_phase_points[0, 1] = right_swing[1] - right_swing_relax_offset
    l_vel_phase_points[1, :2] = r_frc_phase_points[1, :2] = np.negative(np.ones(2))
    l_frc_phase_points[1, :2] = r_vel_phase_points[1, :2] = np.ones(2)

    dbl_stance_relax_offset = (first_dblstance[1] - first_dblstance[0]) * strict_relaxer
    l_frc_phase_points[0, 2] = r_frc_phase_points[0, 2] = first_dblstance[0] + dbl_stance_relax_offset
    l_frc_phase_points[0, 3] = r_frc_phase_points[0, 3] = first_dblstance[1] - dbl_stance_relax_offset
    l_vel_phase_points[0, 2] = r_vel_phase_points[0, 2] = first_dblstance[0] + dbl_stance_relax_offset
    l_vel_phase_points[0, 3] = r_vel_phase_points[0, 3] = first_dblstance[1] - dbl_stance_relax_offset
    if stance_mode == "aerial":
        l_frc_phase_points[1, 2:4] = r_frc_phase_points[1, 2:4] = np.negative(np.ones(2))
        l_vel_phase_points[1, 2:4] = r_vel_phase_points[1, 2:4] = np.ones(2)
    elif stance_mode == "zero":
        l_frc_phase_points[1, 2:4] = r_frc_phase_points[1, 2:4] = np.zeros(2)
        l_vel_phase_points[1, 2:4] = r_vel_phase_points[1, 2:4] = np.zeros(2)
    else:
        l_frc_phase_points[1, 2:4] = r_frc_phase_points[1, 2:4] = np.ones(2)
        l_vel_phase_points[1, 2:4] = r_vel_phase_points[1, 2:4] = np.negative(np.ones(2))

    left_swing_relax_offset = (left_swing[1] - left_swing[0]) * strict_relaxer
    l_frc_phase_points[0, 4] = r_frc_phase_points[0, 4] = left_swing[0] + left_swing_relax_offset
    l_frc_phase_points[0, 5] = r_frc_phase_points[0, 5] = left_swing[1] - left_swing_relax_offset
    l_vel_phase_points[0, 4] = r_vel_phase_points[0, 4] = left_swing[0] + left_swing_relax_offset
    l_vel_phase_points[0, 5] = r_vel_phase_points[0, 5] = left_swing[1] - left_swing_relax_offset
    l_vel_phase_points[1, 4:6] = r_frc_phase_points[1, 4:6] = np.ones(2)
    l_frc_phase_points[1, 4:6] = r_vel_phase_points[1, 4:6] = np.negative(np.ones(2))

    dbl_stance_relax_offset = (second_dblstance[1] - second_dblstance[0]) * strict_relaxer
    l_frc_phase_points[0, 6] = r_frc_phase_points[0, 6] = second_dblstance[0] + dbl_stance_relax_offset
    l_frc_phase_points[0, 7] = r_frc_phase_points[0, 7] = second_dblstance[1] - dbl_stance_relax_offset
    l_vel_phase_points[0, 6] = r_vel_phase_points[0, 6] = second_dblstance[0] + dbl_stance_relax_offset
    l_vel_phase_points[0, 7] = r_vel_phase_points[0, 7] = second_dblstance[1] - dbl_stance_relax_offset
    if stance_mode == "aerial":
        l_frc_phase_points[1, 6:] = r_frc_phase_points[1, 6:] = np.negative(np.ones(2))
        l_vel_phase_points[1, 6:] = r_vel_phase_points[1, 6:] = np.ones(2)
    elif stance_mode == "zero":
        l_frc_phase_points[1, 6:] = r_frc_phase_points[1, 6:] = np.zeros(2)
        l_vel_phase_points[1, 6:] = r_vel_phase_points[1, 6:] = np.zeros(2)
    else:
        l_frc_phase_points[1, 6:] = r_frc_phase_points[1, 6:] = np.ones(2)
        l_vel_phase_points[1, 6:] = r_vel_phase_points[1, 6:] = np.negative(np.ones(2))

    # Extend data to three cycles for continuity
    r_frc_prev_cycle = np.copy(r_frc_phase_points)
    r_vel_prev_cycle = np.copy(r_vel_phase_points)
    l_frc_prev_cycle = np.copy(l_frc_phase_points)
    l_vel_prev_cycle = np.copy(l_vel_phase_points)
    l_frc_prev_cycle[0] = r_frc_prev_cycle[0] = r_frc_phase_points[0] - r_frc_phase_points[0, -1] - dbl_stance_relax_offset
    l_vel_prev_cycle[0] = r_vel_prev_cycle[0] = r_vel_phase_points[0] - r_vel_phase_points[0, -1] - dbl_stance_relax_offset

    r_frc_second_cycle = np.copy(r_frc_phase_points)
    r_vel_second_cycle = np.copy(r_vel_phase_points)
    l_frc_second_cycle = np.copy(l_frc_phase_points)
    l_vel_second_cycle = np.copy(l_vel_phase_points)
    l_frc_second_cycle[0] = r_frc_second_cycle[0] = r_frc_phase_points[0] + r_frc_phase_points[0, -1] + dbl_stance_relax_offset
    l_vel_second_cycle[0] = r_vel_second_cycle[0] = r_vel_phase_points[0] + r_vel_phase_points[0, -1] + dbl_stance_relax_offset

    r_frc_phase_points_repeated = np.hstack((r_frc_prev_cycle, r_frc_phase_points, r_frc_second_cycle))
    r_vel_phase_points_repeated = np.hstack((r_vel_prev_cycle, r_vel_phase_points, r_vel_second_cycle))
    l_frc_phase_points_repeated = np.hstack((l_frc_prev_cycle, l_frc_phase_points, l_frc_second_cycle))
    l_vel_phase_points_repeated = np.hstack((l_vel_prev_cycle, l_vel_phase_points, l_vel_second_cycle))

    r_frc_phase_spline = PchipInterpolator(r_frc_phase_points_repeated[0], r_frc_phase_points_repeated[1])
    r_vel_phase_spline = PchipInterpolator(r_vel_phase_points_repeated[0], r_vel_phase_points_repeated[1])
    l_frc_phase_spline = PchipInterpolator(l_frc_phase_points_repeated[0], l_frc_phase_points_repeated[1])
    l_vel_phase_spline = PchipInterpolator(l_vel_phase_points_repeated[0], l_vel_phase_points_repeated[1])

    return [r_frc_phase_spline, r_vel_phase_spline], [l_frc_phase_spline, l_vel_phase_spline]
