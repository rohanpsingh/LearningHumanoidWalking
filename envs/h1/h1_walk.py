"""H1 humanoid walking environment.

Bootstraps a 3-mode walking policy (STANDING / INPLACE / FORWARD) on the
H1 lower body. Used by itself for the walking expert and as the source
policy for `--imitate` when training the mimic env.
"""

import os

import mujoco
import numpy as np
import transforms3d as tf3

from tasks import walking_task

from .gen_xml import ARM_JOINTS, WAIST_JOINTS, builder
from .h1_base import H1BaseEnv


class H1WalkEnv(H1BaseEnv):
    """Unitree H1 humanoid walking environment."""

    def _get_default_config_path(self) -> str:
        return os.path.join(os.path.dirname(os.path.realpath(__file__)), "configs/walk.yaml")

    def _build_xml(self) -> str:
        export_dir = self._get_xml_export_dir("h1_walk")
        path_to_xml = os.path.join(export_dir, "h1.xml")
        if not os.path.exists(path_to_xml):
            builder(
                export_dir,
                config={
                    "unused_joints": [WAIST_JOINTS, ARM_JOINTS],
                    "rangefinder": False,
                    "raisedplatform": False,
                    "ctrllimited": self.cfg.ctrllimited,
                    "jointlimited": self.cfg.jointlimited,
                    "minimal": self.cfg.reduced_xml,
                },
            )
        return path_to_xml

    def _setup_task(self, control_dt: float) -> None:
        task_cfg = self.cfg.task
        self.task = walking_task.WalkingTask(
            client=self.interface,
            dt=control_dt,
            neutral_pose=self.half_sitting_pose,
            root_body="pelvis",
            lfoot_body=self.LFOOT_BODY,
            rfoot_body=self.RFOOT_BODY,
            head_body="torso_link",
        )
        self.task._goal_height_ref = task_cfg.goal_height
        self.task._total_duration = task_cfg.total_duration
        self.task._swing_duration = task_cfg.swing_duration
        self.task._stance_duration = task_cfg.stance_duration

    def _setup_robot(self) -> None:
        super()._setup_robot()
        self._setup_mirror_indices()

    def _setup_mirror_indices(self) -> None:
        # Mirror indices over the 35-D robot state + 8-D external state.
        # Order: root_orient(2), root_ang_vel(3), motor_pos(10), motor_vel(10),
        # motor_tau(10), clock(2), mode_one_hot(3), mode_ref(3).
        # Joint order in motor blocks: left(5) then right(5); within a leg:
        # hip_yaw, hip_roll, hip_pitch, knee, ankle.
        base_mir_obs = [
            -0.1,
            1,  # root orient (roll, pitch)
            -2,
            3,
            -4,  # root ang vel
            -10,
            -11,
            12,
            13,
            14,  # motor pos [1] -> right leg
            -5,
            -6,
            7,
            8,
            9,  # motor pos [2] -> left leg
            -20,
            -21,
            22,
            23,
            24,  # motor vel [1]
            -15,
            -16,
            17,
            18,
            19,  # motor vel [2]
            -30,
            -31,
            32,
            33,
            34,  # motor torque [1]
            -25,
            -26,
            27,
            28,
            29,  # motor torque [2]
        ]
        num_ext = self._get_num_external_obs()
        append_obs = [(len(base_mir_obs) + i) for i in range(num_ext)]
        self.robot.clock_inds = append_obs[0:2]
        self.robot.mirrored_obs = np.array(base_mir_obs + append_obs, copy=True).tolist()
        # Action ordering: [left_leg(5), right_leg(5)]; mirror swaps and flips
        # signs of yaw/roll dofs (idx 0,1).
        self.robot.mirrored_acts = [-5, -6, 7, 8, 9, -0.1, -1, 2, 3, 4]

    def _get_num_external_obs(self) -> int:
        # clock(2) + mode_one_hot(3) + mode_ref(3) = 8
        return 8

    def _get_external_state(self) -> np.ndarray:
        clock = [
            np.sin(2 * np.pi * self.task._phase / self.task._period),
            np.cos(2 * np.pi * self.task._phase / self.task._period),
        ]
        return np.concatenate((clock, self.task.mode.encode(), self.task.mode_ref))

    def _setup_obs_normalization(self) -> None:
        self.obs_mean = np.concatenate(
            (
                np.zeros(5),
                self.half_sitting_pose,
                np.zeros(10),
                np.zeros(10),
                [0, 0],
                [0.5, 0.5, 0.5, 0, 0, 0],
            )
        )
        self.obs_std = np.concatenate(
            (
                [0.2, 0.2, 1, 1, 1],
                0.5 * np.ones(10),
                4 * np.ones(10),
                100 * np.ones(10),
                [1, 1],
                [1, 1, 1, 0.5, 0.5, 0.5],
            )
        )
        self.obs_mean = np.tile(self.obs_mean, self.history_len)
        self.obs_std = np.tile(self.obs_std, self.history_len)

    def draw_markers(self, marker_drawer):
        """Draw an arrow over the robot showing walking mode and reference."""
        if not hasattr(self.task, "mode"):
            return

        arrow = mujoco.mjtGeom.mjGEOM_ARROW
        head_pos = self.interface.get_object_xpos_by_name(self.task._head_body_name, "OBJ_BODY")
        arrow_pos = [head_pos[0], head_pos[1], head_pos[2] + 0.5]

        root_quat = self.interface.get_object_xquat_by_name(self.task._root_body_name, "OBJ_BODY")
        root_yaw = tf3.euler.quat2euler(root_quat)[2]

        mode = self.task.mode
        yaw_ref, vx_ref, vy_ref = self.task.mode_ref
        rgba_blue = np.array([0, 0, 1, 0.5])

        if mode == walking_task.WalkModes.FORWARD:
            length = float(np.linalg.norm([vx_ref, vy_ref]))
            mat = tf3.euler.euler2mat(0, np.pi / 2, root_yaw)
        elif mode == walking_task.WalkModes.INPLACE:
            length = float(yaw_ref)
            mat = tf3.euler.euler2mat(0, 0, 0)
        else:
            length = 0.0
            mat = tf3.euler.euler2mat(0, np.pi, 0)
        marker_drawer.add_marker(pos=arrow_pos, mat=mat, size=[0.05, 0.05, 2 * length], rgba=rgba_blue, type=arrow)
