import os
import numpy as np
import transforms3d as tf3
import collections

from tasks import stepping_task
from envs.common import mujoco_env
from envs.common import robot_interface
from envs.jvrc import robot

from .gen_xml import builder


class JvrcStepEnv(mujoco_env.MujocoEnv):
    def __init__(self):
        sim_dt = 0.0025
        control_dt = 0.025
        frame_skip = (control_dt/sim_dt)

        path_to_xml_out = '/tmp/mjcf-export/jvrc_step/jvrc1.xml'
        if not os.path.exists(path_to_xml_out):
            builder(path_to_xml_out)
        mujoco_env.MujocoEnv.__init__(self, path_to_xml_out, sim_dt, control_dt)

        pdgains = np.zeros((12, 2))
        coeff = 0.5
        pdgains.T[0] = coeff * np.array([200, 200, 200, 250, 80, 80,
                                         200, 200, 200, 250, 80, 80,])
        pdgains.T[1] = coeff * np.array([20, 20, 20, 25, 8, 8,
                                         20, 20, 20, 25, 8, 8,])

        # list of desired actuators
        # RHIP_P, RHIP_R, RHIP_Y, RKNEE, RANKLE_R, RANKLE_P
        # LHIP_P, LHIP_R, LHIP_Y, LKNEE, LANKLE_R, LANKLE_P
        self.actuators = [0, 1, 2, 3, 4, 5,
                          6, 7, 8, 9, 10, 11]

        # set up interface
        self.interface = robot_interface.RobotInterface(self.model, self.data, 'R_ANKLE_P_S', 'L_ANKLE_P_S')
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

        self.robot = robot.JVRC(pdgains.T, control_dt, self.actuators, self.interface)

        # define indices for action and obs mirror fns
        base_mir_obs = [0.1, -1, 2, -3,              # root orient
                        -4, 5, -6,                   # root ang vel
                        13, -14, -15, 16, -17, 18,   # motor pos [1]
                         7,  -8,  -9, 10, -11, 12,   # motor pos [2]
                        25, -26, -27, 28, -29, 30,   # motor vel [1]
                        19, -20, -21, 22, -23, 24,   # motor vel [2]
        ]
        append_obs = [(len(base_mir_obs)+i) for i in range(10)]
        self.robot.clock_inds = append_obs[0:2]
        self.robot.mirrored_obs = np.array(base_mir_obs + append_obs, copy=True).tolist()
        self.robot.mirrored_acts = [6, -7, -8, 9, -10, 11,
                                    0.1, -1, -2, 3, -4, 5,]

        # set action space
        action_space_size = len(self.robot.actuators)
        action = np.zeros(action_space_size)
        self.action_space = np.zeros(action_space_size)

        # set observation space
        self.base_obs_len = 41
        self.observation_space = np.zeros(self.base_obs_len)

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
        root_orient = tf3.euler.euler2quat(root_r, root_p, 0)
        root_ang_vel = qvel[3:6]

        motor_pos = self.interface.get_act_joint_positions()
        motor_vel = self.interface.get_act_joint_velocities()
        motor_pos = [motor_pos[i] for i in self.actuators]
        motor_vel = [motor_vel[i] for i in self.actuators]

        robot_state = np.concatenate([
            root_orient,
            root_ang_vel,
            motor_pos,
            motor_vel,
        ])
        state = np.concatenate([robot_state, ext_state])
        assert state.shape==(self.base_obs_len,)
        return state.flatten()

    def step(self, a):
        # make one control step
        applied_action = self.robot.step(a)

        # compute reward
        self.task.step()
        rewards = self.task.calc_reward(self.robot.prev_torque, self.robot.prev_action, applied_action)
        total_reward = sum([float(i) for i in rewards.values()])

        # check if terminate
        done = self.task.done()

        obs = self.get_obs()
        return obs, total_reward, done, rewards

    def reset_model(self):
        '''
        # dynamics randomization
        dofadr = [self.interface.get_jnt_qveladr_by_name(jn)
                  for jn in self.interface.get_actuated_joint_names()]
        for jnt in dofadr:
            self.model.dof_frictionloss[jnt] = np.random.uniform(0,10)    # actuated joint frictionloss
            self.model.dof_damping[jnt] = np.random.uniform(0.2,5)        # actuated joint damping
            self.model.dof_armature[jnt] *= np.random.uniform(0.90, 1.10) # actuated joint armature
        '''

        c = 0.02
        self.init_qpos = list(self.robot.init_qpos_)
        self.init_qvel = list(self.robot.init_qvel_)
        self.init_qpos = self.init_qpos + np.random.uniform(low=-c, high=c, size=self.model.nq)
        self.init_qvel = self.init_qvel + np.random.uniform(low=-c, high=c, size=self.model.nv)

        # modify init state acc to task
        root_adr = self.interface.get_jnt_qposadr_by_name('root')[0]
        self.init_qpos[root_adr+0] = np.random.uniform(-1, 1)
        self.init_qpos[root_adr+1] = np.random.uniform(-1, 1)
        self.init_qpos[root_adr+2] = 0.81
        self.init_qpos[root_adr+3:root_adr+7] = tf3.euler.euler2quat(0, np.random.uniform(-5, 5)*np.pi/180, np.random.uniform(-np.pi, np.pi))
        self.set_state(
            self.init_qpos,
            self.init_qvel
        )
        self.task.reset(iter_count = self.robot.iteration_count)
        obs = self.get_obs()
        return obs
