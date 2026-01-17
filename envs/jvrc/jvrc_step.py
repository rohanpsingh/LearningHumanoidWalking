import os

import mujoco
import numpy as np
import transforms3d as tf3

from tasks import stepping_task

from .gen_xml import builder
from .jvrc_base import JvrcBaseEnv


class JvrcStepEnv(JvrcBaseEnv):
    """JVRC humanoid stepping environment with footstep targets."""

    def _build_xml(self) -> str:
        export_dir = self._get_xml_export_dir("jvrc_step")
        path_to_xml = os.path.join(export_dir, "jvrc.xml")
        if not os.path.exists(path_to_xml):
            builder(export_dir, config={"boxes": True})
        return path_to_xml

    def _setup_task(self, control_dt: float) -> None:
        self.task = stepping_task.SteppingTask(
            client=self.interface,
            dt=control_dt,
            neutral_foot_orient=np.array([1, 0, 0, 0]),
            root_body=self.ROOT_BODY,
            lfoot_body=self.LFOOT_BODY,
            rfoot_body=self.RFOOT_BODY,
            head_body=self.HEAD_BODY,
        )

        # Set task parameters from config
        task_cfg = self.cfg.task
        self.task._goal_height_ref = task_cfg.goal_height
        self.task._total_duration = task_cfg.total_duration
        self.task._swing_duration = task_cfg.swing_duration
        self.task._stance_duration = task_cfg.stance_duration

    def _get_num_external_obs(self) -> int:
        return 10  # clock(2) + goal_steps_x(2) + y(2) + z(2) + theta(2)

    def _setup_obs_normalization(self) -> None:
        self.obs_mean = np.concatenate(
            (
                np.zeros(5),  # root_r, root_p, root_ang_vel
                np.deg2rad(self.half_sitting_pose),
                np.zeros(12),  # motor_pos, motor_vel
                [0.5, 0.5],  # clock
                np.zeros(8),  # goal step coords (x, y, z, theta for 2 steps)
            )
        )
        self.obs_std = np.concatenate(
            (
                [0.2, 0.2, 1, 1, 1],  # root orient and ang vel
                0.5 * np.ones(12),
                4 * np.ones(12),  # motor pos and vel
                [1, 1],  # clock
                np.ones(8),  # goal step coords
            )
        )
        self.obs_mean = np.tile(self.obs_mean, self.history_len)
        self.obs_std = np.tile(self.obs_std, self.history_len)

    def _get_external_state(self) -> np.ndarray:
        clock = self._get_clock_signal()
        return np.concatenate(
            (
                clock,
                np.asarray(self.task._goal_steps_x).flatten(),
                np.asarray(self.task._goal_steps_y).flatten(),
                np.asarray(self.task._goal_steps_z).flatten(),
                np.asarray(self.task._goal_steps_theta).flatten(),
            )
        )

    def draw_markers(self, marker_drawer):
        """Draw step targets, goal markers, and foot poses for visualization."""
        if not hasattr(self.task, "l_foot_quat"):
            return

        arrow_size = [0.02, 0.02, 0.5]
        sphere = mujoco.mjtGeom.mjGEOM_SPHERE
        arrow = mujoco.mjtGeom.mjGEOM_ARROW

        # Draw step sequence
        if hasattr(self.task, "sequence"):
            for step in self.task.sequence:
                step_pos = [step[0], step[1], step[2]]
                step_theta = step[3]
                if step_pos not in [
                    self.task.sequence[self.task.t1][0:3].tolist(),
                    self.task.sequence[self.task.t2][0:3].tolist(),
                ]:
                    marker_drawer.add_marker(
                        pos=step_pos, size=np.ones(3) * 0.05, rgba=np.array([0, 1, 1, 1]), type=sphere
                    )
                    marker_drawer.add_marker(
                        pos=step_pos,
                        mat=tf3.euler.euler2mat(0, np.pi / 2, step_theta),
                        size=arrow_size,
                        rgba=np.array([0, 1, 1, 1]),
                        type=arrow,
                    )

            target_radius = self.task.target_radius

            # t1 target (red)
            step_pos = self.task.sequence[self.task.t1][0:3].tolist()
            step_theta = self.task.sequence[self.task.t1][3]
            marker_drawer.add_marker(pos=step_pos, size=np.ones(3) * 0.05, rgba=np.array([1, 0, 0, 1]), type=sphere)
            marker_drawer.add_marker(
                pos=step_pos,
                mat=tf3.euler.euler2mat(0, np.pi / 2, step_theta),
                size=arrow_size,
                rgba=np.array([1, 0, 0, 1]),
                type=arrow,
            )
            marker_drawer.add_marker(
                pos=step_pos, size=np.ones(3) * target_radius, rgba=np.array([1, 0, 0, 0.1]), type=sphere
            )

            # t2 target (blue)
            step_pos = self.task.sequence[self.task.t2][0:3].tolist()
            step_theta = self.task.sequence[self.task.t2][3]
            marker_drawer.add_marker(pos=step_pos, size=np.ones(3) * 0.05, rgba=np.array([0, 0, 1, 1]), type=sphere)
            marker_drawer.add_marker(
                pos=step_pos,
                mat=tf3.euler.euler2mat(0, np.pi / 2, step_theta),
                size=arrow_size,
                rgba=np.array([0, 0, 1, 1]),
                type=arrow,
            )
            marker_drawer.add_marker(
                pos=step_pos, size=np.ones(3) * target_radius, rgba=np.array([0, 0, 1, 0.1]), type=sphere
            )

        # Draw observed goal targets (cyan)
        goalx = self.task._goal_steps_x
        goaly = self.task._goal_steps_y
        goaltheta = self.task._goal_steps_theta
        marker_drawer.add_marker(
            pos=[goalx[0], goaly[0], 0], size=np.ones(3) * 0.05, rgba=np.array([0, 1, 1, 1]), type=sphere
        )
        marker_drawer.add_marker(
            pos=[goalx[0], goaly[0], 0],
            mat=tf3.euler.euler2mat(0, np.pi / 2, goaltheta[0]),
            size=arrow_size,
            rgba=np.array([0, 1, 1, 1]),
            type=arrow,
        )
        marker_drawer.add_marker(
            pos=[goalx[1], goaly[1], 0], size=np.ones(3) * 0.05, rgba=np.array([0, 1, 1, 1]), type=sphere
        )
        marker_drawer.add_marker(
            pos=[goalx[1], goaly[1], 0],
            mat=tf3.euler.euler2mat(0, np.pi / 2, goaltheta[1]),
            size=arrow_size,
            rgba=np.array([0, 1, 1, 1]),
            type=arrow,
        )

        # Draw feet poses (gray)
        lfoot_orient = (tf3.quaternions.quat2mat(self.task.l_foot_quat)).dot(tf3.euler.euler2mat(0, np.pi / 2, 0))
        rfoot_orient = (tf3.quaternions.quat2mat(self.task.r_foot_quat)).dot(tf3.euler.euler2mat(0, np.pi / 2, 0))
        marker_drawer.add_marker(pos=self.task.l_foot_pos, size=np.ones(3) * 0.05, rgba=[0.5, 0.5, 0.5, 1], type=sphere)
        marker_drawer.add_marker(
            pos=self.task.l_foot_pos, mat=lfoot_orient, size=arrow_size, rgba=[0.5, 0.5, 0.5, 1], type=arrow
        )
        marker_drawer.add_marker(pos=self.task.r_foot_pos, size=np.ones(3) * 0.05, rgba=[0.5, 0.5, 0.5, 1], type=sphere)
        marker_drawer.add_marker(
            pos=self.task.r_foot_pos, mat=rfoot_orient, size=arrow_size, rgba=[0.5, 0.5, 0.5, 1], type=arrow
        )

        # Draw origin axes
        marker_drawer.add_marker(pos=[0, 0, 0], size=np.ones(3) * 0.05, rgba=np.array([1, 1, 1, 1]), type=sphere)
        marker_drawer.add_marker(
            pos=[0, 0, 0],
            mat=tf3.euler.euler2mat(0, 0, 0),
            size=[0.01, 0.01, 2],
            rgba=np.array([0, 0, 1, 0.2]),
            type=arrow,
        )
        marker_drawer.add_marker(
            pos=[0, 0, 0],
            mat=tf3.euler.euler2mat(0, np.pi / 2, 0),
            size=[0.01, 0.01, 2],
            rgba=np.array([1, 0, 0, 0.2]),
            type=arrow,
        )
        marker_drawer.add_marker(
            pos=[0, 0, 0],
            mat=tf3.euler.euler2mat(-np.pi / 2, np.pi / 2, 0),
            size=[0.01, 0.01, 2],
            rgba=np.array([0, 1, 0, 0.2]),
            type=arrow,
        )
