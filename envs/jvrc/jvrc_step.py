import os
import numpy as np
import transforms3d as tf3
import collections

from tasks import stepping_task
from robots.robot_base import RobotBase
from envs.common import mujoco_env
from envs.common import robot_interface
from envs.common import config_builder
from envs.jvrc.jvrc_walk import JvrcWalkEnv

from .gen_xml import *

class JvrcStepEnv(JvrcWalkEnv):
    def __init__(self, path_to_yaml = None):

        ## Load CONFIG from yaml ##
        if path_to_yaml is None:
            path_to_yaml = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'configs/base.yaml')

        self.cfg = config_builder.load_yaml(path_to_yaml)

        sim_dt = self.cfg.sim_dt
        control_dt = self.cfg.control_dt
        frame_skip = (control_dt/sim_dt)

        self.history_len = self.cfg.obs_history_len

        path_to_xml = '/tmp/mjcf-export/jvrc_step/jvrc.xml'
        if not os.path.exists(path_to_xml):
            export_dir = os.path.dirname(path_to_xml)
            builder(export_dir, config={
                "boxes": True,
            })

        mujoco_env.MujocoEnv.__init__(self, path_to_xml, sim_dt, control_dt)

        pdgains = np.zeros((2, 12))
        pdgains[0] = self.cfg.kp
        pdgains[1] = self.cfg.kd

        # list of desired actuators
        # RHIP_P, RHIP_R, RHIP_Y, RKNEE, RANKLE_R, RANKLE_P
        # LHIP_P, LHIP_R, LHIP_Y, LKNEE, LANKLE_R, LANKLE_P
        self.actuators = LEG_JOINTS

        # define nominal pose
        base_position = [0, 0, 0.81]
        base_orientation = [1, 0, 0, 0]
        half_sitting_pose = [-30,  0, 0, 50, 0, -24,
                             -30,  0, 0, 50, 0, -24,
        ] # degrees
        self.nominal_pose = base_position + base_orientation + np.deg2rad(half_sitting_pose).tolist()

        # set up interface
        self.interface = robot_interface.RobotInterface(self.model, self.data, 'R_ANKLE_P_S', 'L_ANKLE_P_S', None)

        # set up task
        self.task = stepping_task.SteppingTask(client=self.interface,
                                               dt=control_dt,
                                               neutral_foot_orient=np.array([1, 0, 0, 0]),
                                               root_body='PELVIS_S',
                                               lfoot_body='L_ANKLE_P_S',
                                               rfoot_body='R_ANKLE_P_S',
                                               head_body='NECK_P_S',
        )
        # set goal height
        self.task._goal_height_ref = 0.80
        self.task._total_duration = 1.1
        self.task._swing_duration = 0.75
        self.task._stance_duration = 0.35

        # set up robot
        self.robot = RobotBase(pdgains, control_dt, self.interface, self.task)

        # define indices for action and obs mirror fns
        base_mir_obs = [-0.1, 1,                   # root orient
                        -2, 3, -4,                 # root ang vel
                        11, -12, -13, 14, -15, 16, # motor pos [1]
                         5,  -6,  -7,  8,  -9, 10, # motor pos [2]
                        23, -24, -25, 26, -27, 28, # motor vel [1]
                        17, -18, -19, 20, -21, 22, # motor vel [2]
        ]
        append_obs = [(len(base_mir_obs)+i) for i in range(10)]
        self.robot.clock_inds = append_obs[0:2]
        self.robot.mirrored_obs = np.array(base_mir_obs + append_obs, copy=True).tolist()
        self.robot.mirrored_acts = [6, -7, -8, 9, -10, 11,
                                    0.1, -1, -2, 3, -4, 5,]

        # set action space
        action_space_size = len(self.actuators)
        action = np.zeros(action_space_size)
        self.action_space = np.zeros(action_space_size)
        self.prev_prediction = np.zeros(action_space_size)

        # set observation space
        self.base_obs_len = 39
        self.observation_history = collections.deque(maxlen=self.history_len)
        self.observation_space = np.zeros(self.base_obs_len*self.history_len)

    def get_obs(self):
        # external state
        clock = [np.sin(2 * np.pi * self.task._phase / self.task._period),
                 np.cos(2 * np.pi * self.task._phase / self.task._period)]
        ext_state = np.concatenate((clock,
                                    np.asarray(self.task._goal_steps_x).flatten(),
                                    np.asarray(self.task._goal_steps_y).flatten(),
                                    np.asarray(self.task._goal_steps_z).flatten(),
                                    np.asarray(self.task._goal_steps_theta).flatten()))

        # internal state
        qpos = np.copy(self.interface.get_qpos())
        qvel = np.copy(self.interface.get_qvel())
        root_r, root_p = tf3.euler.quat2euler(qpos[3:7])[0:2]
        root_r = np.array([root_r])
        root_p = np.array([root_p])
        root_ang_vel = qvel[3:6]
        motor_pos = self.interface.get_act_joint_positions()
        motor_vel = self.interface.get_act_joint_velocities()

        robot_state = np.concatenate([
            root_r, root_p, root_ang_vel, motor_pos, motor_vel,
        ])

        state = np.concatenate([robot_state, ext_state])
        assert state.shape==(self.base_obs_len,), \
            "State vector length expected to be: {} but is {}".format(self.base_obs_len, len(state))

        if len(self.observation_history)==0:
            for _ in range(self.history_len):
                self.observation_history.appendleft(np.zeros_like(state))
            self.observation_history.appendleft(state)
        else:
            self.observation_history.appendleft(state)
        return np.array(self.observation_history).flatten()
