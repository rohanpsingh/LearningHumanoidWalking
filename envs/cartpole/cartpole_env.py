"""Minimal cartpole environment - single-file implementation.

This serves as a simple example for creating new environments.
Contains everything in one file: env, observation space, reward function.
"""

import os

import numpy as np

from envs.common.mujoco_env import MujocoEnv
from envs.common.robot_interface import RobotInterface

# Path to the cartpole XML model
CARTPOLE_XML = os.path.join(os.path.dirname(__file__), "cartpole.xml")


class CartpoleRobot:
    """Simple robot wrapper with PD control for cartpole."""

    def __init__(self, interface: RobotInterface, control_dt: float, kp: float, kd: float):
        self.interface = interface
        self.control_dt = control_dt

        # Frame skip (control_dt / sim_dt)
        self.frame_skip = int(control_dt / interface.sim_dt())

        # Set PD gains on interface
        kp_arr = np.array([kp])
        kd_arr = np.array([kd])
        self.interface.set_pd_gains(kp_arr, kd_arr)

        # For training infrastructure compatibility
        self.iteration_count = 0

    def step(self, target_position: np.ndarray) -> None:
        """Execute PD control loop for one control timestep.

        Args:
            target_position: Target cart position [-1, 1].
        """
        target = np.atleast_1d(target_position)
        zero_vel = np.zeros_like(target)

        for _ in range(self.frame_skip):
            # Compute PD torque using interface
            torque = self.interface.step_pd(target, zero_vel)

            # Apply torque and step simulation
            self.interface.set_motor_torque(torque)
            self.interface.step()


class CartpoleEnv(MujocoEnv):
    """Simple cartpole swing-up environment with PD control.

    Observation (4-dim):
        - cart position
        - pole angle (0 = upright)
        - cart velocity
        - pole angular velocity

    Action (1-dim):
        - target cart position [-1, 1]

    Reward:
        - upright bonus: cos(pole_angle)
        - torque penalty: -0.01 * action^2
    """

    def __init__(self, path_to_yaml=None):
        # Simulation parameters
        sim_dt = 0.005  # Physics timestep
        control_dt = 0.02  # Control timestep

        super().__init__(CARTPOLE_XML, sim_dt, control_dt)

        # Create robot interface
        self.interface = RobotInterface(self.model, self.data)

        # Create robot with PD control
        self.robot = CartpoleRobot(self.interface, control_dt, kp=100.0, kd=10.0)

        # Cache qpos/qvel indices for observations
        self._slider_qpos_idx = self.model.jnt_qposadr[self.model.joint("slider").id]
        self._hinge_qpos_idx = self.model.jnt_qposadr[self.model.joint("hinge").id]
        self._slider_qvel_idx = self.model.jnt_dofadr[self.model.joint("slider").id]
        self._hinge_qvel_idx = self.model.jnt_dofadr[self.model.joint("hinge").id]

        # Observation and action spaces (as numpy arrays, matching project convention)
        self.observation_space = np.zeros(4)
        self.action_space = np.zeros(1)

    def reset_model(self):
        """Reset to random initial state with pole hanging down."""
        # Start with pole at random position
        pole_init = np.random.uniform(-np.pi, np.pi)
        qpos = np.array([0.0, pole_init])
        qvel = np.array([0.0, 0.0])

        # Add small random perturbation
        qpos += np.random.uniform(-0.1, 0.1, size=2)
        qvel += np.random.uniform(-0.1, 0.1, size=2)

        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        """Get current observation."""
        cart_pos = self.data.qpos[self._slider_qpos_idx]
        pole_angle = self.data.qpos[self._hinge_qpos_idx] % (2 * np.pi)
        cart_vel = self.data.qvel[self._slider_qvel_idx]
        pole_vel = self.data.qvel[self._hinge_qvel_idx]

        return np.array([cart_pos, pole_angle, cart_vel, pole_vel])

    def step(self, action):
        """Take one environment step with PD control."""

        # Execute robot step with PD control
        self.robot.step(action)

        obs = self._get_obs()
        reward = self._compute_reward(obs, action)
        done = self._check_termination(obs)

        return obs, float(reward), float(done), {}

    def _compute_reward(self, obs, action):
        """Compute reward for swing-up task. Total reward in [0, 1]."""
        cart_pos = obs[0]
        pole_angle = obs[1]
        pole_vel = obs[3]

        # upright error
        upright_error = 1 - np.cos(pole_angle)
        upright_reward = np.exp(-2.0 * upright_error**2)

        # Center reward: 1 when at center, decays as cart moves away
        center_reward = np.exp(-1.0 * cart_pos**2)

        # velocity error
        vel_reward = np.exp(-0.01 * pole_vel**2)

        # Action penalty
        action_reward = np.exp(-0.5 * np.sum(action**2))

        return 0.7 * upright_reward + 0.1 * vel_reward + 0.1 * action_reward + 0.1 * center_reward

    def _check_termination(self, obs):
        """Check if episode should terminate."""
        cart_pos = obs[0]
        # Terminate if cart goes out of bounds
        return np.abs(cart_pos) > 0.99
