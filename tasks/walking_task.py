"""Generic mode-conditioned walking task for biped humanoids.

3-mode command (STANDING / INPLACE / FORWARD) with a 3-D velocity reference
[yaw_vel, vx, vy]. Mode-dependent zeroing is applied during reward
computation, so unused dimensions of `mode_ref` are ignored.

Designed to be robot-agnostic: robot-specific configuration (root/foot/head
body names, gait timing, neutral pose) is supplied via __init__. For richer
behavior (curriculum, custom mode-switching, custom termination), subclass
WalkingTask.
"""

from enum import Enum, auto

import numpy as np

from tasks import rewards
from tasks.base_task import BaseTask


class WalkModes(Enum):
    STANDING = auto()
    INPLACE = auto()
    FORWARD = auto()

    def encode(self):
        if self.name == "STANDING":
            return np.array([0, 0, 1])
        elif self.name == "INPLACE":
            return np.array([0, 1, 0])
        elif self.name == "FORWARD":
            return np.array([1, 0, 0])

    def sample_ref(self) -> np.ndarray:
        """Sample a 3-D velocity reference [yaw_vel, vx, vy]."""
        if self.name == "STANDING":
            return np.random.uniform(-1.0, 1.0, 3)
        if self.name == "INPLACE":
            return np.array([np.random.uniform(-0.5, 0.5), 0.0, 0.0])
        if self.name == "FORWARD":
            return np.array([0.0, np.random.uniform(0.0, 0.4), 0.0])


class WalkingTask(BaseTask):
    """Mode-conditioned walking task for biped humanoids."""

    def __init__(
        self,
        client=None,
        dt=0.025,
        neutral_foot_orient=None,
        neutral_pose=None,
        root_body="pelvis",
        lfoot_body="lfoot",
        rfoot_body="rfoot",
        head_body="head",
        manip_hfield=False,
    ):
        if neutral_foot_orient is None:
            neutral_foot_orient = []
        if neutral_pose is None:
            neutral_pose = []

        self._client = client
        self._control_dt = dt
        self._neutral_foot_orient = neutral_foot_orient
        self._neutral_pose = np.array(neutral_pose)
        self.manip_hfield = manip_hfield

        self._mass = self._client.get_robot_mass()

        # Set by the env after construction; left empty here so subclasses
        # / envs can populate from config.
        self.mode_ref = []
        self._goal_height_ref = []
        self._swing_duration = []
        self._stance_duration = []
        self._total_duration = []

        self._root_body_name = root_body
        self._lfoot_body_name = lfoot_body
        self._rfoot_body_name = rfoot_body
        self._head_body_name = head_body

    def calc_reward(self, prev_torque, prev_action, action):
        # Gather state from client
        l_foot_vel = self._client.get_lfoot_body_vel(frame=1)[0]
        r_foot_vel = self._client.get_rfoot_body_vel(frame=1)[0]
        l_foot_frc = self._client.get_lfoot_grf()
        r_foot_frc = self._client.get_rfoot_grf()
        head_pos = self._client.get_object_xpos_by_name(self._head_body_name, "OBJ_BODY")[0:2]
        root_pos = self._client.get_object_xpos_by_name(self._root_body_name, "OBJ_BODY")[0:2]
        root_height = self._client.get_object_xpos_by_name(self._root_body_name, "OBJ_BODY")[2]
        root_vel_xy = self._client.get_body_vel(self._root_body_name, frame=1)[0][:2]
        yaw_vel = self._client.get_qvel()[5]
        qvel = self._client.get_qvel()
        qacc = self._client.get_qacc()
        current_torque = np.asarray(self._client.get_act_joint_torques())
        n_joints = len(self._neutral_pose)
        current_pose = np.array(self._client.get_act_joint_positions())[:n_joints]

        # Get contact point for height calculation
        if self._client.check_rfoot_floor_collision() or self._client.check_lfoot_floor_collision():
            contact_point_z = min(
                c.pos[2] for _, c in (self._client.get_rfoot_floor_contacts() + self._client.get_lfoot_floor_contacts())
            )
        else:
            contact_point_z = 0

        # Determine clock functions based on mode
        r_frc_fn = self.right_clock[0]
        l_frc_fn = self.left_clock[0]
        r_vel_fn = self.right_clock[1]
        l_vel_fn = self.left_clock[1]
        if self.mode == WalkModes.STANDING:
            r_frc_fn = lambda _: 1
            l_frc_fn = lambda _: 1
            r_vel_fn = lambda _: -1
            l_vel_fn = lambda _: -1

        # Decompose 3-D mode_ref [yaw_vel, vx, vy] and apply mode-dependent zeroing
        yaw_vel_ref, vx_ref, vy_ref = self.mode_ref
        if self.mode == WalkModes.STANDING:
            yaw_vel_ref, vx_ref, vy_ref = 0.0, 0.0, 0.0
        elif self.mode == WalkModes.INPLACE:
            vx_ref, vy_ref = 0.0, 0.0
        elif self.mode == WalkModes.FORWARD:
            yaw_vel_ref = 0.0
        goal_vel_xy = np.array([vx_ref, vy_ref])
        goal_speed = float(np.linalg.norm(goal_vel_xy))

        reward = dict(
            foot_frc_score=0.225
            * rewards.calc_foot_frc_clock_reward(l_foot_frc, r_foot_frc, self._phase, l_frc_fn, r_frc_fn, self._mass),
            foot_vel_score=0.225
            * rewards.calc_foot_vel_clock_reward(l_foot_vel, r_foot_vel, self._phase, l_vel_fn, r_vel_fn),
            root_accel=0.050 * rewards.calc_root_accel_reward(qvel, qacc),
            height_error=0.050
            * rewards.calc_height_reward(root_height, self._goal_height_ref, goal_speed, contact_point_z),
            com_vel_error=0.150 * rewards.calc_fwd_vel_reward(root_vel_xy, goal_vel_xy),
            yaw_vel_error=0.150 * rewards.calc_yaw_vel_reward(yaw_vel, yaw_vel_ref),
            upper_body_reward=0.050 * np.exp(-10 * np.linalg.norm(head_pos - root_pos)),
            posture_error=0.050 * np.exp(-np.linalg.norm(self._neutral_pose - current_pose)),
            torque_penalty=0.025 * rewards.calc_torque_reward(current_torque, prev_torque),
            action_penalty=0.025 * rewards.calc_action_reward(action, prev_action),
        )
        return reward

    def step(self):
        # increment phase
        self._phase += 1
        if self._phase >= self._period:
            self._phase = 0

        # random switch between INPLACE and STANDING (only in double support)
        in_double_support = self.right_clock[0](self._phase) == 1 and self.left_clock[0](self._phase) == 1
        if np.random.randint(100) == 0 and in_double_support:
            if self.mode == WalkModes.INPLACE:
                self.mode = WalkModes.STANDING
            elif self.mode == WalkModes.STANDING:
                self.mode = WalkModes.INPLACE
            self.mode_ref = self.mode.sample_ref()

        # random switch between INPLACE and FORWARD
        if np.random.randint(200) == 0 and self.mode != WalkModes.STANDING:
            if self.mode == WalkModes.FORWARD:
                self.mode = WalkModes.INPLACE
            elif self.mode == WalkModes.INPLACE:
                self.mode = WalkModes.FORWARD
            self.mode_ref = self.mode.sample_ref()

        # manipulate hfield
        if self.manip_hfield:
            if np.random.randint(200) == 0 and self.mode != WalkModes.STANDING:
                self._client.model.geom("hfield").pos[:] = [
                    np.random.uniform(-0.5, 0.5),
                    np.random.uniform(-0.5, 0.5),
                    np.random.uniform(-0.015, -0.035),
                ]

    def substep(self) -> None:
        pass

    def done(self):
        contact_flag = self._client.check_self_collisions()
        qpos = self._client.get_qpos()
        terminate_conditions = {
            "qpos[2]_ll": (qpos[2] < 0.6),
            "qpos[2]_ul": (qpos[2] > 1.4),
            "contact_flag": contact_flag,
        }
        return True in terminate_conditions.values()

    def reset(self, iter_count=0):
        self.mode = np.random.choice([WalkModes.STANDING, WalkModes.INPLACE, WalkModes.FORWARD], p=[0.6, 0.2, 0.2])
        self.mode_ref = self.mode.sample_ref()

        self.right_clock, self.left_clock = rewards.create_phase_reward(
            self._swing_duration, self._stance_duration, 0.1, "grounded", 1 / self._control_dt
        )

        # number of control steps in one full cycle
        # (one full cycle includes left swing + right swing)
        self._period = np.floor(2 * self._total_duration * (1 / self._control_dt))
        self._phase = np.random.randint(0, self._period)
