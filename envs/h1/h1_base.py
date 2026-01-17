"""Base class for H1 robot environments.

This module provides common functionality shared across H1 environment
implementations (standing, walking, etc.).
"""
import os
from abc import abstractmethod

import numpy as np

from robots.robot_base import RobotBase
from envs.common.base_humanoid_env import BaseHumanoidEnv
from envs.common import robot_interface
from tasks import observations as obs_terms

from .gen_xml import LEG_JOINTS


class H1BaseEnv(BaseHumanoidEnv):
    """Base class for H1 humanoid environments.

    Provides H1-specific common functionality:
    - Nominal pose and interface setup
    - PD gains configuration
    - Robot state with motor torques
    """

    # H1 body names
    RFOOT_BODY = 'right_ankle_link'
    LFOOT_BODY = 'left_ankle_link'

    def _get_default_config_path(self) -> str:
        return os.path.join(os.path.dirname(os.path.realpath(__file__)), 'configs/base.yaml')

    def _setup_robot(self) -> None:
        control_dt = self.cfg.control_dt

        # Adjust body masses (actual robot is ~7kg heavier)
        self.model.body("pelvis").mass = 8.89
        self.model.body("torso_link").mass = 21.289

        # PD gains from config
        self.leg_names = LEG_JOINTS
        gains_dict = self.cfg.pdgains.to_dict()
        kp, kd = zip(*[gains_dict[jn] for jn in self.leg_names])
        pdgains = np.array([kp, kd])

        # Get half-sitting pose from config
        self.half_sitting_pose = self.cfg.half_sitting_pose

        # Define nominal pose
        base_position = [0, 0, 0.98]
        base_orientation = [1, 0, 0, 0]
        self.nominal_pose = base_position + base_orientation + list(self.half_sitting_pose)

        # Setup interface
        self.interface = robot_interface.RobotInterface(
            self.model, self.data, self.RFOOT_BODY, self.LFOOT_BODY, None)

        # Setup task (implemented by subclasses)
        self._setup_task(control_dt)

        # Setup robot
        self.robot = RobotBase(pdgains, control_dt, self.interface, self.task)

    @abstractmethod
    def _setup_task(self, control_dt: float) -> None:
        """Setup the task instance. Must set self.task."""
        pass

    def _setup_spaces(self) -> None:
        action_space_size = len(self.leg_names)
        self.action_space = np.zeros(action_space_size)
        self.prev_prediction = np.zeros(action_space_size)

        self.base_obs_len = self._get_robot_state_len() + self._get_num_external_obs()
        self.observation_space = np.zeros(self.base_obs_len * self.history_len)

        self._setup_obs_normalization()

    def _get_robot_state_len(self) -> int:
        """Return length of robot state vector."""
        # root_r(1) + root_p(1) + root_ang_vel(3) + motor_pos(10) + motor_vel(10) + motor_tau(10)
        return 35

    def _get_num_external_obs(self) -> int:
        """Return number of external observation dimensions. Override if needed."""
        return 0

    @abstractmethod
    def _setup_obs_normalization(self) -> None:
        """Setup obs_mean and obs_std for observation normalization."""
        pass

    def _get_robot_state(self) -> np.ndarray:
        root_r, root_p = obs_terms.get_root_orientation(self.interface)
        root_ang_vel = obs_terms.get_root_angular_velocity(self.interface)
        motor_pos = obs_terms.get_motor_positions(self.interface)
        motor_vel = obs_terms.get_motor_velocities(self.interface)
        motor_tau = obs_terms.get_motor_torques(self.interface)

        # Apply observation noise if enabled
        if hasattr(self.cfg, 'observation_noise') and self.cfg.observation_noise.enabled:
            observations = {
                'root_orient': np.concatenate([root_r, root_p]),
                'root_ang_vel': root_ang_vel,
                'motor_pos': motor_pos,
                'motor_vel': motor_vel,
                'motor_tau': motor_tau,
            }
            observations = self._apply_observation_noise(observations)
            root_r = observations['root_orient'][:1]
            root_p = observations['root_orient'][1:]
            root_ang_vel = observations['root_ang_vel']
            motor_pos = observations['motor_pos']
            motor_vel = observations['motor_vel']
            motor_tau = observations['motor_tau']

        return np.concatenate([root_r, root_p, root_ang_vel, motor_pos, motor_vel, motor_tau])

    def viewer_setup(self):
        super().viewer_setup()
        self.viewer.cam.distance = 5
        self.viewer.cam.lookat[2] = 1.5
        self.viewer.cam.lookat[0] = 1.0
