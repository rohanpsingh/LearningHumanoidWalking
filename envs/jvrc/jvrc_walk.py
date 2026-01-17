import os
import numpy as np
import transforms3d as tf3
import mujoco

from tasks import walking_task

from .jvrc_base import JvrcBaseEnv
from .gen_xml import builder


class JvrcWalkEnv(JvrcBaseEnv):
    """JVRC humanoid walking environment."""

    def _build_xml(self) -> str:
        export_dir = self._get_xml_export_dir('jvrc_walk')
        path_to_xml = os.path.join(export_dir, 'jvrc.xml')
        if not os.path.exists(path_to_xml):
            builder(export_dir, config={})
        return path_to_xml

    def _setup_task(self, control_dt: float) -> None:
        self.task = walking_task.WalkingTask(
            client=self.interface,
            dt=control_dt,
            neutral_foot_orient=np.array([1, 0, 0, 0]),
            root_body=self.ROOT_BODY,
            lfoot_body=self.LFOOT_BODY,
            rfoot_body=self.RFOOT_BODY,
            head_body=self.HEAD_BODY,
        )
        self.task._neutral_pose = self.nominal_pose

        # Set task parameters from config
        task_cfg = self.cfg.task
        self.task._goal_height_ref = task_cfg.goal_height
        self.task._total_duration = task_cfg.total_duration
        self.task._swing_duration = task_cfg.swing_duration
        self.task._stance_duration = task_cfg.stance_duration

    def _get_num_external_obs(self) -> int:
        return 6  # clock(2) + mode_encode(3) + mode_ref(1)

    def _setup_obs_normalization(self) -> None:
        self.obs_mean = np.concatenate((
            np.zeros(5),
            np.deg2rad(self.half_sitting_pose), np.zeros(12),
            [0.5, 0.5, 0.5, 0, 0, 0]
        ))
        self.obs_std = np.concatenate((
            [0.2, 0.2, 1, 1, 1],
            0.5 * np.ones(12), 4 * np.ones(12),
            [1, 1, 1, 1, 1, 1]
        ))
        self.obs_mean = np.tile(self.obs_mean, self.history_len)
        self.obs_std = np.tile(self.obs_std, self.history_len)

    def _get_external_state(self) -> np.ndarray:
        clock = self._get_clock_signal()
        return np.concatenate((clock, self.task.mode.encode(), [self.task.mode_ref]))

    def draw_markers(self, marker_drawer):
        """Draw an arrow above the robot's head indicating walking mode and reference."""
        if not hasattr(self.task, 'mode'):
            return

        arrow = mujoco.mjtGeom.mjGEOM_ARROW

        # Get head position and add offset above
        head_pos = self.interface.get_object_xpos_by_name(self.task._head_body_name, 'OBJ_BODY')
        arrow_pos = [head_pos[0], head_pos[1], head_pos[2] + 0.3]

        # Get root body orientation for forward direction
        root_quat = self.interface.get_object_xquat_by_name(self.task._root_body_name, 'OBJ_BODY')
        root_yaw = tf3.euler.quat2euler(root_quat)[2]

        # Determine arrow direction and size based on mode
        mode = self.task.mode
        mode_ref = self.task.mode_ref
        rgba_blue = np.array([0, 0, 1, 0.5])
        rgba_green = np.array([0, 1, 0, 0.5])

        if mode == walking_task.WalkModes.FORWARD:
            arrow_length = abs(mode_ref)
            arrow_mat = tf3.euler.euler2mat(0, np.pi/2, root_yaw)
        elif mode == walking_task.WalkModes.INPLACE:
            arrow_length = mode_ref
            arrow_mat = tf3.euler.euler2mat(0, 0, 0)
        else:  # STANDING
            arrow_length = 0.0
            arrow_mat = tf3.euler.euler2mat(0, np.pi, 0)

        arrow_size = [0.05, 0.05, 2*arrow_length]
        marker_drawer.add_marker(pos=arrow_pos, mat=arrow_mat,
                                 size=arrow_size, rgba=rgba_blue, type=arrow)

        # Draw green arrow showing actual velocity
        qvel = self.interface.get_qvel()
        if mode == walking_task.WalkModes.FORWARD:
            vel_x, vel_y = qvel[0], qvel[1]
            actual_length = np.sqrt(vel_x**2 + vel_y**2)
            vel_yaw = np.arctan2(vel_y, vel_x)
            actual_mat = tf3.euler.euler2mat(0, np.pi/2, vel_yaw)
        elif mode == walking_task.WalkModes.INPLACE:
            actual_length = qvel[5]
            actual_mat = tf3.euler.euler2mat(0, 0, 0)
        else:  # STANDING
            actual_length = 0.0
            actual_mat = tf3.euler.euler2mat(0, np.pi, 0)

        actual_arrow_size = [0.05, 0.05, 2*actual_length]
        marker_drawer.add_marker(pos=arrow_pos, mat=actual_mat,
                                 size=actual_arrow_size, rgba=rgba_green, type=arrow)
