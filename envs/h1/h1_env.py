import os
import copy
import numpy as np
import transforms3d as tf3
import collections

from robots.robot_base import RobotBase
from envs.common import mujoco_env
from envs.common import robot_interface
from envs.common import config_builder
from tasks.standing_task import StandingTask

from .gen_xml import *


class H1Env(mujoco_env.MujocoEnv):
    def __init__(self, path_to_yaml = None):

        ## Load CONFIG from yaml ##
        if path_to_yaml is None:
            path_to_yaml = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'configs/base.yaml')

        self.cfg = config_builder.load_yaml(path_to_yaml)

        sim_dt = self.cfg.sim_dt
        control_dt = self.cfg.control_dt
        frame_skip = (control_dt/sim_dt)

        self.dynrand_interval = self.cfg.dynamics_randomization.interval/control_dt
        self.perturb_interval = self.cfg.perturbation.interval/control_dt
        self.history_len = self.cfg.obs_history_len

        path_to_xml = '/tmp/mjcf-export/h1/h1.xml'
        if not os.path.exists(path_to_xml):
            export_dir = os.path.dirname(path_to_xml)
            builder(export_dir, config={
                'unused_joints': [WAIST_JOINTS, ARM_JOINTS],
                'rangefinder': False,
                'raisedplatform': False,
                'ctrllimited': self.cfg.ctrllimited,
                'jointlimited': self.cfg.jointlimited,
                'minimal': self.cfg.reduced_xml,
            })

        mujoco_env.MujocoEnv.__init__(self, path_to_xml, sim_dt, control_dt)

        # the actual weight of the robot is about 7kg higher
        self.model.body("pelvis").mass = 8.89
        self.model.body("torso_link").mass = 21.289

        # list of desired actuators
        self.leg_names = LEG_JOINTS
        gains_dict = self.cfg.pdgains.to_dict()
        kp, kd = zip(*[gains_dict[jn] for jn in self.leg_names])
        pdgains = np.array([kp, kd])

        # define nominal pose
        base_position = [0, 0, 0.98]
        base_orientation = [1, 0, 0, 0]
        half_sitting_pose = [
            0, 0, -0.2, 0.6, -0.4,
            0, 0, -0.2, 0.6, -0.4,
        ]

        # set up interface
        self.interface = robot_interface.RobotInterface(self.model, self.data, 'right_ankle_link', 'left_ankle_link', None)
        self.nominal_pose = base_position + base_orientation + half_sitting_pose

        # set up task
        self.task = StandingTask(self.interface, half_sitting_pose)

        self.robot = RobotBase(pdgains, control_dt, self.interface, self.task)

        # set action space
        action_space_size = len(self.leg_names)
        action = np.zeros(action_space_size)
        self.action_space = np.zeros(action_space_size)
        self.prev_prediction = np.zeros(action_space_size)

        # set observation space
        self.base_obs_len = 35
        self.observation_history = collections.deque(maxlen=self.history_len)
        self.observation_space = np.zeros(self.base_obs_len*self.history_len)

        # manually define observation mean and std
        self.obs_mean = np.concatenate((
            np.zeros(5),
            half_sitting_pose, np.zeros(10), np.zeros(10),
        ))

        self.obs_std = np.concatenate((
            [0.2, 0.2, 1, 1, 1],
            0.5*np.ones(10), 4*np.ones(10), 100*np.ones(10),
        ))

        self.obs_mean = np.tile(self.obs_mean, self.history_len)
        self.obs_std = np.tile(self.obs_std, self.history_len)

        # copy the original model
        self.default_model = copy.deepcopy(self.model)

    def get_obs(self):
        # internal state
        qpos = np.copy(self.interface.get_qpos())
        qvel = np.copy(self.interface.get_qvel())
        root_r, root_p = tf3.euler.quat2euler(qpos[3:7])[0:2]
        root_r = np.array([root_r])
        root_p = np.array([root_p])
        root_ang_vel = qvel[3:6]
        motor_pos = self.interface.get_act_joint_positions()
        motor_vel = self.interface.get_act_joint_velocities()
        motor_tau = self.interface.get_act_joint_torques()

        # add some Gaussian noise to observations
        if self.cfg.observation_noise.enabled:
            scales = self.cfg.observation_noise.scales
            level = self.cfg.observation_noise.multiplier

        # add some noise to observations
        if self.cfg.observation_noise.enabled:
            noise_type = self.cfg.observation_noise.type
            scales = self.cfg.observation_noise.scales
            level = self.cfg.observation_noise.multiplier
            if noise_type=="uniform":
                noise = lambda x, n : np.random.uniform(-x, x, n)
            elif noise_type=="gaussian":
                noise = lambda x, n : np.random.randn(n) * x
            else:
                raise Exception("Observation noise type can only be \"uniform\" or \"gaussian\"")
            root_r += noise(scales.root_orient * level, 1)
            root_p += noise(scales.root_orient * level, 1)
            root_ang_vel += noise(scales.root_ang_vel * level, len(root_ang_vel))
            motor_pos += noise(scales.motor_pos * level, len(motor_pos))
            motor_vel += noise(scales.motor_vel * level, len(motor_vel))
            motor_tau += noise(scales.motor_tau * level, len(motor_tau))

        robot_state = np.concatenate([
            root_r, root_p, root_ang_vel, motor_pos, motor_vel, motor_tau,
        ])

        state = robot_state.copy()
        assert state.shape==(self.base_obs_len,), \
            "State vector length expected to be: {} but is {}".format(self.base_obs_len, len(state))

        if len(self.observation_history)==0:
            for _ in range(self.history_len):
                self.observation_history.appendleft(np.zeros_like(state))
            self.observation_history.appendleft(state)
        else:
            self.observation_history.appendleft(state)
        return np.array(self.observation_history).flatten()

    def step(self, action):
        # Compute the applied action to actuators
        # (targets <- Policy predictions)
        # (offsets <- Half-sitting pose)

        # action vector assumed to be in the following order:
        # [leg_0, leg_1, ..., leg_n, waist_0, ..., waist_n, arm_0, arm_1, ..., arm_n]
        targets = self.cfg.action_smoothing * action + \
            (1 - self.cfg.action_smoothing) * self.prev_prediction
        offsets = [
            self.nominal_pose[self.interface.get_jnt_qposadr_by_name(jnt)[0]]
            for jnt in self.leg_names
        ]

        rewards, done = self.robot.step(targets, np.asarray(offsets))
        obs = self.get_obs()

        if self.cfg.dynamics_randomization.enable and np.random.randint(self.dynrand_interval)==0:
            self.randomize_dyn()

        if self.cfg.perturbation.enable and np.random.randint(self.perturb_interval)==0:
            self.randomize_perturb()

        self.prev_prediction = action

        return obs, sum(rewards.values()), done, rewards

    def reset_model(self):
        if self.cfg.dynamics_randomization.enable:
            self.randomize_dyn()

        init_qpos, init_qvel = self.nominal_pose.copy(), [0] * self.interface.nv()

        # add some initialization noise to root orientation (roll, pitch)
        c = self.cfg.init_noise * np.pi/180
        root_adr = self.interface.get_jnt_qposadr_by_name('root')[0]
        init_qpos[root_adr+2] = np.random.uniform(1.0, 1.02)
        init_qpos[root_adr+3:root_adr+7] = tf3.euler.euler2quat(np.random.uniform(-c, c), np.random.uniform(-c, c), 0)
        init_qpos[root_adr+7:] += np.random.uniform(-c, c, len(self.leg_names))

        # set up init state
        self.set_state(
            np.asarray(init_qpos),
            np.asarray(init_qvel)
        )
        # do a few simulation steps to avoid big contact forces in the start
        for _ in range(3):
            self.interface.step()

        self.task.reset()

        self.prev_prediction = np.zeros_like(self.prev_prediction)
        self.observation_history = collections.deque(maxlen=self.history_len)
        obs = self.get_obs()
        return obs

    #### randomizations and other utility functions ###########
    def randomize_perturb(self):
        frc_mag = self.cfg.perturbation.force_magnitude
        tau_mag = self.cfg.perturbation.force_magnitude
        for body in self.cfg.perturbation.bodies:
            self.data.body(body).xfrc_applied[:3] = np.random.uniform(-frc_mag, frc_mag, 3)
            self.data.body(body).xfrc_applied[3:] = np.random.uniform(-tau_mag, tau_mag, 3)
            if np.random.randint(2)==0:
                self.data.xfrc_applied = np.zeros_like(self.data.xfrc_applied)

    def randomize_dyn(self):
        # dynamics randomization
        dofadr = [self.interface.get_jnt_qveladr_by_name(jn)
                  for jn in self.leg_names]
        for jnt in dofadr:
            self.model.dof_frictionloss[jnt] = np.random.uniform(0, 2)    # actuated joint frictionloss
            self.model.dof_damping[jnt] = np.random.uniform(0.02, 2)      # actuated joint damping

        # randomize com
        bodies = ["pelvis"]
        for legjoint in self.leg_names:
            bodyid = self.model.joint(legjoint).bodyid
            bodyname = self.model.body(bodyid).name
            bodies.append(bodyname)

        for body in bodies:
            default_mass = self.default_model.body(body).mass[0]
            default_ipos = self.default_model.body(body).ipos
            self.model.body(body).mass[0] = default_mass*np.random.uniform(0.95, 1.05)
            self.model.body(body).ipos = default_ipos + np.random.uniform(-0.01, 0.01, 3)

    def viewer_setup(self):
        super().viewer_setup()
        self.viewer.cam.distance = 5
        self.viewer.cam.lookat[2] = 1.5
        self.viewer.cam.lookat[0] = 1.0
