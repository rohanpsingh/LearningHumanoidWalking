"""Standing balance task for humanoid robots.

This module defines a simple standing task where the robot must maintain
an upright posture without walking.
"""
from typing import Dict

import numpy as np

from tasks.base_task import BaseTask


class StandingTask(BaseTask):
    """Standing balance task for humanoid robots.

    The robot must maintain an upright standing posture, minimizing
    velocity, keeping the torso aligned, and maintaining a target height.

    Attributes:
        _client: RobotInterface for accessing robot state.
        neutral_pose: Target joint positions for standing.
    """

    def __init__(self, client, neutral_pose):
        """Initialize the standing task.

        Args:
            client: RobotInterface instance for robot state access.
            neutral_pose: Target joint positions for the standing posture.
        """
        self._client = client
        self.neutral_pose = neutral_pose

    def reset(self, iter_count: int = 0) -> None:
        """Reset task state for a new episode.

        Standing task has no state to reset.

        Args:
            iter_count: Current training iteration (unused).
        """
        pass

    def step(self) -> None:
        pass

    def substep(self) -> None:
        pass

    def calc_reward(
        self,
        prev_torque: np.ndarray,
        prev_action: np.ndarray,
        action: np.ndarray,
    ) -> Dict[str, float]:
        """Calculate reward components for standing.

        Rewards the robot for:
        - Maintaining target height
        - Keeping upper body aligned over pelvis
        - Maintaining neutral posture
        - Minimizing joint torques
        - Minimizing forward velocity (staying in place)
        - Minimizing yaw velocity (not spinning)

        Args:
            prev_torque: Joint torques from the previous step (unused).
            prev_action: Action from the previous step (unused).
            action: Current action (unused).

        Returns:
            Dictionary of reward components.
        """
        root_pose = self._client.get_object_affine_by_name("pelvis", 'OBJ_BODY')

        # height reward
        target_root_h = 0.98
        root_h = root_pose[2, 3]
        height_error = np.linalg.norm(root_h - target_root_h)

        # upperbody reward
        head_pose_offset = np.zeros(2)
        head_pose = self._client.get_object_affine_by_name("torso_link", 'OBJ_BODY')
        head_pos_in_robot_base = np.linalg.inv(root_pose).dot(head_pose)[:2, 3] - head_pose_offset
        upperbody_error = np.linalg.norm(head_pos_in_robot_base)

        # posture reward
        current_pose = np.array(self._client.get_act_joint_positions())[:10]
        posture_error = np.linalg.norm(current_pose - self.neutral_pose)

        # torque reward
        tau_error = np.linalg.norm(self._client.get_act_joint_torques())

        # velocity reward
        root_vel = self._client.get_body_vel("pelvis", frame=1)[0][:2]
        fwd_vel_error = np.linalg.norm(root_vel)
        yaw_vel = self._client.get_qvel()[5]
        yaw_vel_error = np.linalg.norm(yaw_vel)

        reward = {
            "com_vel_error": 0.3 * np.exp(-4 * np.square(fwd_vel_error)),
            "yaw_vel_error": 0.3 * np.exp(-4 * np.square(yaw_vel_error)),
            "height": 0.1 * np.exp(-0.5 * np.square(height_error)),
            "upperbody": 0.1 * np.exp(-40 * np.square(upperbody_error)),
            "joint_torque_reward": 0.1 * np.exp(-5e-5 * np.square(tau_error)),
            "posture": 0.1 * np.exp(-1 * np.square(posture_error)),
        }
        return reward

    def done(self) -> bool:
        """Check if the episode should terminate.

        Terminates if:
        - Root height drops below 0.9m (falling)
        - Root height exceeds 1.4m (unrealistic)
        - Self-collision detected

        Returns:
            True if episode should terminate, False otherwise.
        """
        root_jnt_adr = self._client.model.body("pelvis").jntadr[0]
        root_qpos_adr = self._client.model.joint(root_jnt_adr).qposadr[0]
        qpos = self._client.get_qpos()[root_qpos_adr:root_qpos_adr + 7]
        contact_flag = self._client.check_self_collisions()

        terminate_conditions = {
            "qpos[2]_ll": (qpos[2] < 0.9),
            "qpos[2]_ul": (qpos[2] > 1.4),
            "contact_flag": contact_flag,
        }

        return True in terminate_conditions.values()
