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
        sim_dt = 0.01  # Physics timestep
        control_dt = 0.04  # Control timestep (4x frame skip)

        super().__init__(CARTPOLE_XML, sim_dt, control_dt)

        # Create robot interface
        self.interface = RobotInterface(self.model, self.data)

        # Create robot with PD control
        self.robot = CartpoleRobot(self.interface, control_dt, kp=10.0, kd=2.0)

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
        # Start with pole hanging down (angle ~pi) with small noise
        qpos = np.array([0.0, np.pi])  # cart at center, pole hanging down
        qvel = np.array([0.0, 0.0])

        # Add small random perturbation
        qpos += np.random.uniform(-0.1, 0.1, size=2)
        qvel += np.random.uniform(-0.1, 0.1, size=2)

        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        """Get current observation."""
        cart_pos = self.data.qpos[self._slider_qpos_idx]
        pole_angle = self.data.qpos[self._hinge_qpos_idx]
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
        """Compute reward: upright bonus - torque penalty."""
        pole_angle = obs[1]

        # Upright reward: +1 when upright, -1 when hanging down
        upright_reward = np.cos(pole_angle)

        # Torque penalty
        torque_penalty = 0.01 * np.sum(action**2)

        return upright_reward - torque_penalty

    def _check_termination(self, obs):
        """Check if episode should terminate."""
        cart_pos = obs[0]
        # Terminate if cart goes out of bounds
        return np.abs(cart_pos) > 0.95

    def viewer_setup(self):
        """Setup viewer camera."""
        with self.viewer.lock():
            self.viewer.cam.trackbodyid = 0
            self.viewer.cam.distance = 4.0
            self.viewer.cam.lookat[2] = 0.5
            self.viewer.cam.elevation = -20
