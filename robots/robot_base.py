import numpy as np

class RobotBase(object):
    def __init__(self, pdgains, dt, client, task, pdrand_k = 0, sim_bemf = False, sim_motor_dyn = False):

        self.client = client
        self.task = task
        self.control_dt = dt
        self.pdrand_k = pdrand_k
        self.sim_bemf = sim_bemf
        self.sim_motor_dyn = sim_motor_dyn

        assert (self.sim_bemf & self.sim_motor_dyn)==False, \
            "You cannot simulate back-EMF and motor dynamics simultaneously!"

        # set PD gains
        self.kp = pdgains[0]
        self.kd = pdgains[1]
        assert self.kp.shape==self.kd.shape==(self.client.nu(),), \
            f"kp shape {self.kp.shape} and kd shape {self.kd.shape} must be {(self.client.nu(),)}"

        # torque damping param
        self.tau_d = np.zeros(self.client.nu())

        self.client.set_pd_gains(self.kp, self.kd)
        tau = self.client.step_pd(np.zeros(self.client.nu()), np.zeros(self.client.nu()))
        w = self.client.get_act_joint_velocities()
        assert len(w)==len(tau)
        
        self.prev_action = None
        self.prev_torque = None
        self.iteration_count = np.inf

        # frame skip parameter
        if (np.around(self.control_dt%self.client.sim_dt(), 6)):
            raise Exception("Control dt should be an integer multiple of Simulation dt.")
        self.frame_skip = int(self.control_dt/self.client.sim_dt())

    def _do_simulation(self, target, n_frames):
        # randomize PD gains
        if self.pdrand_k:
            k = self.pdrand_k
            kp = np.random.uniform((1-k)*self.kp, (1+k)*self.kp)
            kd = np.random.uniform((1-k)*self.kd, (1+k)*self.kd)
            self.client.set_pd_gains(kp, kd)

        assert target.shape == (self.client.nu(),), \
            f"Target shape must be {(self.client.nu(),)}"

        ratio = self.client.get_gear_ratios()

        if self.sim_bemf and np.random.randint(10)==0:
            self.tau_d = np.random.uniform(5, 40, self.client.nu())

        for _ in range(n_frames):
            w = self.client.get_act_joint_velocities()
            tau = self.client.step_pd(target, np.zeros(self.client.nu()))
            tau = tau - self.tau_d*w
            tau /= ratio
            self.client.set_motor_torque(tau, self.sim_motor_dyn)
            self.client.step()

    def step(self, action, offset=None):

        if not isinstance(action, np.ndarray):
            raise TypeError("Expected action to be a numpy array")

        action = np.copy(action)

        assert action.shape == (self.client.nu(),), \
            f"Action vector length expected to be: {self.client.nu()} but is {action.shape}"

        # If offset is provided, add to action vector
        if offset is not None:
            if not isinstance(offset, np.ndarray):
                raise TypeError("Expected offset to be a numpy array")
            assert offset.shape == action.shape, \
                f"Offset shape {offset} must match action shape {action.shape}"
            offset = np.copy(offset)
            action += offset

        if self.prev_action is None:
            self.prev_action = action
        if self.prev_torque is None:
            self.prev_torque = np.asarray(self.client.get_act_joint_torques())

        # Perform the simulation
        self._do_simulation(action, self.frame_skip)

        # Task-related operations
        self.task.step()
        rewards = self.task.calc_reward(self.prev_torque, self.prev_action, action)
        done = self.task.done()

        self.prev_action = action
        self.prev_torque = np.asarray(self.client.get_act_joint_torques())

        return rewards, done
