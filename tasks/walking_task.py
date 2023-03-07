import numpy as np
import transforms3d as tf3
from tasks import rewards

class WalkingTask(object):
    """Dynamically stable walking on biped."""

    def __init__(self,
                 client=None,
                 dt=0.025,
                 neutral_foot_orient=[],
                 root_body='pelvis',
                 lfoot_body='lfoot',
                 rfoot_body='rfoot',
                 waist_r_joint='waist_r',
                 waist_p_joint='waist_p',
    ):

        self._client = client
        self._control_dt = dt
        self._neutral_foot_orient=neutral_foot_orient

        self._mass = self._client.get_robot_mass()

        # These depend on the robot, hardcoded for now
        # Ideally, they should be arguments to __init__
        self._goal_speed_ref = []
        self._goal_height_ref = []
        self._swing_duration = []
        self._stance_duration = []
        self._total_duration = []

        self._root_body_name = root_body
        self._lfoot_body_name = lfoot_body
        self._rfoot_body_name = rfoot_body

    def calc_reward(self, prev_torque, prev_action, action):
        self.l_foot_vel = self._client.get_lfoot_body_vel()[0]
        self.r_foot_vel = self._client.get_rfoot_body_vel()[0]
        self.l_foot_frc = self._client.get_lfoot_grf()
        self.r_foot_frc = self._client.get_rfoot_grf()
        r_frc = self.right_clock[0]
        l_frc = self.left_clock[0]
        r_vel = self.right_clock[1]
        l_vel = self.left_clock[1]
        reward = dict(foot_frc_score=0.150 * rewards._calc_foot_frc_clock_reward(self, l_frc, r_frc),
                      foot_vel_score=0.150 * rewards._calc_foot_vel_clock_reward(self, l_vel, r_vel),
                      orient_cost=0.050 * (rewards._calc_body_orient_reward(self, self._lfoot_body_name) +
                                           rewards._calc_body_orient_reward(self, self._rfoot_body_name) +
                                           rewards._calc_body_orient_reward(self, self._root_body_name))/3,
                      root_accel=0.050 * rewards._calc_root_accel_reward(self),
                      height_error=0.050 * rewards._calc_height_reward(self),
                      com_vel_error=0.200 * rewards._calc_fwd_vel_reward(self),
                      torque_penalty=0.050 * rewards._calc_torque_reward(self, prev_torque),
                      action_penalty=0.050 * rewards._calc_action_reward(self, prev_action),
        )
        return reward

    def step(self):
        if self._phase>self._period:
            self._phase=0
        self._phase+=1
        return

    def done(self):
        contact_flag = self._client.check_self_collisions()
        qpos = self._client.get_qpos()
        terminate_conditions = {"qpos[2]_ll":(qpos[2] < 0.6),
                                "qpos[2]_ul":(qpos[2] > 1.4),
                                "contact_flag":contact_flag,
        }

        done = True in terminate_conditions.values()
        return done

    def reset(self):
        self._goal_speed_ref = np.random.choice([0, np.random.uniform(0.3, 0.4)])
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
