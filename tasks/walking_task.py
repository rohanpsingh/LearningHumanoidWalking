import numpy as np
import transforms3d as tf3
from tasks import rewards
from tasks.base_task import BaseTask
from enum import Enum, auto

class WalkModes(Enum):
    STANDING = auto()
    INPLACE  = auto()
    FORWARD  = auto()

    def encode(self):
        if self.name=='STANDING':
            return np.array([0, 0, 1])
        elif self.name=='INPLACE':
            return np.array([0, 1, 0])
        elif self.name=='FORWARD':
            return np.array([1, 0, 0])

    def sample_ref(self):
        if self.name == 'STANDING':
            return np.random.uniform(-1, 1)
        if self.name == 'INPLACE':
            return np.random.uniform(-0.5, 0.5)
        if self.name == 'FORWARD':
            return np.random.uniform(0., 0.4)

class WalkingTask(BaseTask):
    """Dynamically stable walking on biped."""

    def __init__(self,
                 client=None,
                 dt=0.025,
                 neutral_foot_orient=[],
                 neutral_pose=[],
                 root_body='pelvis',
                 lfoot_body='lfoot',
                 rfoot_body='rfoot',
                 head_body='head',
                 waist_r_joint='waist_r',
                 waist_p_joint='waist_p',
                 manip_hfield=False,
    ):

        self._client = client
        self._control_dt = dt
        self._neutral_foot_orient=neutral_foot_orient
        self._neutral_pose = np.array(neutral_pose)
        self.manip_hfield = manip_hfield

        self._mass = self._client.get_robot_mass()

        # These depend on the robot, hardcoded for now
        # Ideally, they should be arguments to __init__
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
        self.l_foot_vel = self._client.get_lfoot_body_vel(frame=1)[0]
        self.r_foot_vel = self._client.get_rfoot_body_vel(frame=1)[0]
        self.l_foot_frc = self._client.get_lfoot_grf()
        self.r_foot_frc = self._client.get_rfoot_grf()
        head_pos = self._client.get_object_xpos_by_name(self._head_body_name, 'OBJ_BODY')[0:2]
        root_pos = self._client.get_object_xpos_by_name(self._root_body_name, 'OBJ_BODY')[0:2]
        current_pose = np.array(self._client.get_act_joint_positions())

        r_frc = self.right_clock[0]
        l_frc = self.left_clock[0]
        r_vel = self.right_clock[1]
        l_vel = self.left_clock[1]
        if self.mode == WalkModes.STANDING:
            r_frc = (lambda _:1)
            l_frc = (lambda _:1)
            r_vel = (lambda _:-1)
            l_vel = (lambda _:-1)

        if self.mode == WalkModes.STANDING:
            self._goal_speed_ref = 0
            yaw_vel_ref = 0
        if self.mode == WalkModes.INPLACE:
            self._goal_speed_ref = 0
            yaw_vel_ref = self.mode_ref
        if self.mode == WalkModes.FORWARD:
            self._goal_speed_ref = self.mode_ref
            yaw_vel_ref = 0

        # WARNING: This assumes leg joints to be at [:12]
        reward = dict(foot_frc_score=0.225 * rewards._calc_foot_frc_clock_reward(self, l_frc, r_frc),
                      foot_vel_score=0.225 * rewards._calc_foot_vel_clock_reward(self, l_vel, r_vel),
                      root_accel=0.050 * rewards._calc_root_accel_reward(self),
                      height_error=0.050 * rewards._calc_height_reward(self),
                      com_vel_error=0.150 * rewards._calc_fwd_vel_reward(self),
                      yaw_vel_error=0.150 * rewards._calc_yaw_vel_reward(self, yaw_vel_ref),
                      upper_body_reward=0.050 * np.exp(-10*np.linalg.norm(head_pos-root_pos)),
                      posture_error=0.050 * np.exp(-np.linalg.norm(self._neutral_pose[:12]-current_pose[:12])),
                      torque_penalty=0.025 * rewards._calc_torque_reward(self, prev_torque),
                      action_penalty=0.025 * rewards._calc_action_reward(self, action, prev_action),
        )
        return reward

    def step(self):
        # increment phase
        self._phase+=1
        if self._phase>=self._period:
            self._phase=0

        # random switch between INPLACE and STANDING( (only in double support)
        in_double_support = self.right_clock[0](self._phase)==1 and self.left_clock[0](self._phase)==1
        if np.random.randint(100)==0 and in_double_support:
            if self.mode == WalkModes.INPLACE:
                self.mode = WalkModes.STANDING
            elif self.mode == WalkModes.STANDING:
                self.mode = WalkModes.INPLACE
            self.mode_ref = self.mode.sample_ref()

        # random switch between INPLACE and FORWARD
        if np.random.randint(200)==0 and self.mode!=WalkModes.STANDING:
            if self.mode == WalkModes.FORWARD:
                self.mode = WalkModes.INPLACE
            elif self.mode == WalkModes.INPLACE:
                self.mode = WalkModes.FORWARD
            self.mode_ref = self.mode.sample_ref()

        # manipulate hfield
        if self.manip_hfield:
            if np.random.randint(200)==0 and self.mode!=WalkModes.STANDING:
                self._client.model.geom("hfield").pos[:] = [np.random.uniform(-0.5, 0.5),
                                                            np.random.uniform(-0.5, 0.5),
                                                            np.random.uniform(-0.015, -0.035)]
        return

    def substep(self) -> None:
        pass

    def done(self):
        contact_flag = self._client.check_self_collisions()
        qpos = self._client.get_qpos()
        terminate_conditions = {"qpos[2]_ll":(qpos[2] < 0.6),
                                "qpos[2]_ul":(qpos[2] > 1.4),
                                "contact_flag":contact_flag,
        }

        done = True in terminate_conditions.values()
        return done

    def reset(self, iter_count=0):
        # select a walking 'mode'
        self.mode = np.random.choice(
            [WalkModes.STANDING, WalkModes.INPLACE, WalkModes.FORWARD],
            p=[0.6, 0.2, 0.2])
        self.mode_ref = self.mode.sample_ref()

        self.right_clock, self.left_clock = rewards.create_phase_reward(self._swing_duration,
                                                                        self._stance_duration,
                                                                        0.1,
                                                                        "grounded",
                                                                        1/self._control_dt)

        # number of control steps in one full cycle
        # (one full cycle includes left swing + right swing)
        self._period = np.floor(2*self._total_duration*(1/self._control_dt))
        # randomize phase during initialization
        self._phase = np.random.randint(0, self._period)
