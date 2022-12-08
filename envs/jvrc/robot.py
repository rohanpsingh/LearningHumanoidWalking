import numpy as np

class JVRC:
    def __init__(self, pdgains, dt, active, client):

        self.client = client
        self.control_dt = dt

        # list of desired actuators
        self.actuators = active

        # set PD gains
        self.kp = pdgains[0]
        self.kd = pdgains[1]
        assert self.kp.shape==self.kd.shape==(self.client.nu(),)
        self.client.set_pd_gains(self.kp, self.kd)
        
        # define init qpos and qvel
        self.init_qpos_ = [0] * self.client.nq()
        self.init_qvel_ = [0] * self.client.nv()

        self.prev_action = None
        self.prev_torque = None
        self.iteration_count = np.inf

        # frame skip parameter
        if (np.around(self.control_dt%self.client.sim_dt(), 6)):
            raise Exception("Control dt should be an integer multiple of Simulation dt.")
        self.frame_skip = int(self.control_dt/self.client.sim_dt())

        # define nominal pose
        base_position = [0, 0, 0.81]
        base_orientation = [1, 0, 0, 0]
        half_sitting_pose = [-30,  0, 0, 50, 0, -24,
                             -30,  0, 0, 50, 0, -24,
                             -3, -9.74, -30,
                             -3,  9.74, -30,
        ] # degrees

        # number of all joints
        self.num_joints = len(half_sitting_pose)
        
        # define init qpos and qvel
        nominal_pose = [q*np.pi/180.0 for q in half_sitting_pose]
        robot_pose = base_position + base_orientation + nominal_pose
        assert len(robot_pose)==self.client.nq()
        self.init_qpos_[-len(robot_pose):] = base_position + base_orientation + nominal_pose

        # define actuated joint nominal pose
        motor_qposadr = self.client.get_motor_qposadr()
        self.motor_offset = [self.init_qpos_[i] for i in motor_qposadr]

    def step(self, action):
        filtered_action = np.zeros(len(self.motor_offset))
        for idx, act_id in enumerate(self.actuators):
            filtered_action[act_id] = action[idx]

        # add fixed motor offset
        filtered_action += self.motor_offset

        if self.prev_action is None:
            self.prev_action = filtered_action
        if self.prev_torque is None:
            self.prev_torque = np.asarray(self.client.get_act_joint_torques())

        self.client.set_pd_gains(self.kp, self.kd)
        self.do_simulation(filtered_action, self.frame_skip)

        self.prev_action = filtered_action
        self.prev_torque = np.asarray(self.client.get_act_joint_torques())
        return filtered_action

    def do_simulation(self, target, n_frames):
        ratio = self.client.get_gear_ratios()
        for _ in range(n_frames):
            tau = self.client.step_pd(target, np.zeros(self.client.nu()))
            tau = [(i/j) for i,j in zip(tau, ratio)]
            self.client.set_motor_torque(tau)
            self.client.step()
