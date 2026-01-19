"""Minimal cartpole environment - single-file implementation.

This serves as a simple example for creating new environments.
Contains everything in one file: env, observation space, reward function.
"""

import os

import mujoco
import numpy as np

from envs.common.mujoco_env import MujocoEnv

# Path to the cartpole XML model
CARTPOLE_XML = os.path.join(os.path.dirname(__file__), "cartpole.xml")


class _MinimalRobot:
    """Minimal robot interface for compatibility with training infrastructure."""

    def __init__(self):
        self.iteration_count = 0


class CartpoleEnv(MujocoEnv):
    """Simple cartpole swing-up environment.

    Observation (4-dim):
        - cart position
        - pole angle (0 = upright)
        - cart velocity
        - pole angular velocity

    Action (1-dim):
        - force applied to cart [-1, 1]

    Reward:
        - upright bonus: cos(pole_angle)
        - torque penalty: -0.01 * action^2
    """

    def __init__(self, path_to_yaml=None):
        # Simulation parameters
        sim_dt = 0.01  # Physics timestep
        control_dt = 0.04  # Control timestep (4x frame skip)

        super().__init__(CARTPOLE_XML, sim_dt, control_dt)

        # Cache joint/actuator indices
        self._slider_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "slider")
        self._hinge_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "hinge")
        self._actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "slide")

        # Get qpos/qvel indices for joints
        self._slider_qpos_idx = self.model.jnt_qposadr[self._slider_joint_id]
        self._hinge_qpos_idx = self.model.jnt_qposadr[self._hinge_joint_id]
        self._slider_qvel_idx = self.model.jnt_dofadr[self._slider_joint_id]
        self._hinge_qvel_idx = self.model.jnt_dofadr[self._hinge_joint_id]

        # Observation and action spaces (as numpy arrays, matching project convention)
        self.observation_space = np.zeros(4)
        self.action_space = np.zeros(1)

        # Minimal robot interface for training infrastructure compatibility
        self.robot = _MinimalRobot()

        # Observation normalization (mean, std)
        self.obs_mean = np.zeros(4)
        self.obs_std = np.array([1.0, np.pi, 2.0, 5.0])

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
        """Take one environment step."""
        action = np.clip(action, -1.0, 1.0)

        # Apply action
        self.data.ctrl[self._actuator_id] = action[0]

        # Step simulation
        for _ in range(int(self.frame_skip)):
            mujoco.mj_step(self.model, self.data)

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
