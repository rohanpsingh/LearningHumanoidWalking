"""Base class for humanoid robot environments.

This module provides common functionality shared across all humanoid
environment implementations, reducing code duplication.
"""

import collections
import copy
import os
from abc import abstractmethod

import numpy as np
import transforms3d as tf3

from envs.common import config_builder, mujoco_env
from envs.common.domain_randomization import apply_perturbation, randomize_dynamics


class BaseHumanoidEnv(mujoco_env.MujocoEnv):
    """Base class for humanoid robot environments.

    Provides common functionality:
    - Configuration loading from YAML
    - Observation history management
    - Action smoothing
    - Domain randomization
    - Observation noise
    - Initialization noise
    - Standard step/reset patterns

    Subclasses must implement:
    - _build_xml(): Generate and return path to MuJoCo XML
    - _setup_robot(): Configure robot interface, task, and RobotBase
    - _get_robot_state(): Return current robot state observation
    - _get_external_state(): Return task-specific external state
    """

    def __init__(self, path_to_yaml: str | None = None):
        """Initialize the humanoid environment.

        Args:
            path_to_yaml: Path to configuration YAML file. If None, uses
                default config from the environment's configs/ directory.
        """
        # Load configuration
        if path_to_yaml is None:
            path_to_yaml = self._get_default_config_path()
        self.cfg = config_builder.load_yaml(path_to_yaml)

        # Extract common config values
        sim_dt = self.cfg.sim_dt
        control_dt = self.cfg.control_dt
        self.history_len = self.cfg.obs_history_len
        self._action_smoothing = self.cfg.action_smoothing

        # Build XML and initialize MuJoCo
        path_to_xml = self._build_xml()
        mujoco_env.MujocoEnv.__init__(self, path_to_xml, sim_dt, control_dt)

        # Setup robot-specific components (interface, task, robot)
        self._setup_robot()

        # Store default model for domain randomization (before any randomization)
        self.default_model = copy.deepcopy(self.model)

        # Setup domain randomization intervals
        self._setup_domain_randomization()

        # Initialize action/observation spaces
        self._setup_spaces()

        # Initialize observation history
        self.observation_history = collections.deque(maxlen=self.history_len)
        self.prev_prediction = np.zeros_like(self.action_space)

    def _setup_domain_randomization(self) -> None:
        """Setup domain randomization intervals from config."""
        control_dt = self.cfg.control_dt

        # Check for dynamics randomization config
        dyn_cfg = getattr(self.cfg, "dynamics_randomization", None)
        if dyn_cfg is not None and getattr(dyn_cfg, "enable", False):
            self.dynrand_interval = int(dyn_cfg.interval / control_dt)
        else:
            self.dynrand_interval = 0

        # Check for perturbation config
        perturb_cfg = getattr(self.cfg, "perturbation", None)
        if perturb_cfg is not None and getattr(perturb_cfg, "enable", False):
            self.perturb_interval = int(perturb_cfg.interval / control_dt)
        else:
            self.perturb_interval = 0

    @abstractmethod
    def _get_default_config_path(self) -> str:
        """Return path to default config YAML for this environment."""
        pass

    @abstractmethod
    def _build_xml(self) -> str:
        """Build MuJoCo XML and return the path to it.

        Should check if XML already exists and build only if necessary.
        Returns absolute path to the generated XML file.
        """
        pass

    @abstractmethod
    def _setup_robot(self) -> None:
        """Setup robot interface, task, and RobotBase.

        Must set:
        - self.interface: RobotInterface instance
        - self.task: Task instance
        - self.robot: RobotBase instance
        - self.nominal_pose: List of nominal joint positions
        - self.actuators or self.leg_names: List of actuator names
        - self.half_sitting_pose: Robot's half-sitting pose
        """
        pass

    @abstractmethod
    def _setup_spaces(self) -> None:
        """Setup action and observation spaces.

        Must set:
        - self.action_space: np.ndarray defining action space
        - self.observation_space: np.ndarray defining observation space
        - self.base_obs_len: int, length of single observation
        - self.obs_mean: np.ndarray for normalization
        - self.obs_std: np.ndarray for normalization
        """
        pass

    @abstractmethod
    def _get_robot_state(self) -> np.ndarray:
        """Return the current robot proprioceptive state.

        Typically includes: root orientation, angular velocity,
        motor positions, motor velocities.
        """
        pass

    @abstractmethod
    def _get_external_state(self) -> np.ndarray:
        """Return task-specific external state.

        Examples: clock signals, goal velocities, step targets.
        """
        pass

    def _get_xml_export_dir(self, env_name: str) -> str:
        """Get the XML export directory for this environment.

        Uses config value if available, otherwise creates a temp directory.

        Args:
            env_name: Name of the environment (e.g., 'jvrc_walk', 'h1')

        Returns:
            Path to export directory.
        """
        if hasattr(self.cfg, "xml_export_path") and self.cfg.xml_export_path:
            base_path = self.cfg.xml_export_path
        else:
            base_path = "/tmp/mjcf-export"
        return os.path.join(base_path, env_name)

    def _get_joint_names(self) -> list[str]:
        """Get list of actuated joint names.

        Returns:
            List of joint names (from actuators or leg_names attribute).
        """
        return getattr(self, "actuators", getattr(self, "leg_names", []))

    def get_obs(self) -> np.ndarray:
        """Get the current observation including history.

        Returns:
            Flattened array of current and historical observations.
        """
        robot_state = self._get_robot_state()
        ext_state = self._get_external_state()
        state = np.concatenate([robot_state, ext_state])

        assert state.shape == (self.base_obs_len,), (
            f"State vector length expected to be: {self.base_obs_len} but is {len(state)}"
        )

        # Manage observation history
        if len(self.observation_history) == 0:
            for _ in range(self.history_len):
                self.observation_history.appendleft(np.zeros_like(state))
        self.observation_history.appendleft(state)

        return np.array(self.observation_history).flatten()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        """Execute one environment step.

        Args:
            action: Action from policy.

        Returns:
            Tuple of (observation, reward, done, info_dict).
        """
        # Apply action smoothing
        targets = self._action_smoothing * action + (1 - self._action_smoothing) * self.prev_prediction

        # Get offsets from nominal pose
        offsets = self._get_action_offsets()

        # Execute robot step
        rewards, done = self.robot.step(targets, np.asarray(offsets))
        obs = self.get_obs()

        self.prev_prediction = action

        # Apply domain randomization if enabled
        if self.dynrand_interval > 0 and np.random.randint(self.dynrand_interval) == 0:
            self._randomize_dynamics()

        if self.perturb_interval > 0 and np.random.randint(self.perturb_interval) == 0:
            self._apply_perturbation()

        return obs, sum(rewards.values()), done, rewards

    def _randomize_dynamics(self) -> None:
        """Apply dynamics randomization."""
        joint_names = self._get_joint_names()
        randomize_dynamics(self.model, self.default_model, self.interface, joint_names, self.cfg.dynamics_randomization)

    def _apply_perturbation(self) -> None:
        """Apply perturbation forces."""
        apply_perturbation(self.data, self.cfg.perturbation)

    def _get_action_offsets(self) -> list[float]:
        """Get action offsets from nominal pose.

        Returns:
            List of offset values for each actuator.
        """
        actuator_names = self._get_joint_names()
        return [self.nominal_pose[self.interface.get_jnt_qposadr_by_name(jnt)[0]] for jnt in actuator_names]

    def reset_model(self) -> np.ndarray:
        """Reset the environment to initial state.

        Returns:
            Initial observation.
        """
        # Apply dynamics randomization on reset if enabled
        if self.dynrand_interval > 0:
            self._randomize_dynamics()

        init_qpos = self.nominal_pose.copy()
        init_qvel = [0] * self.interface.nv()

        # Apply initialization noise if configured
        init_noise = getattr(self.cfg, "init_noise", None)
        if init_noise is not None and init_noise > 0:
            init_qpos = self._apply_init_noise(init_qpos)

        self.set_state(np.asarray(init_qpos), np.asarray(init_qvel))

        # Do a few simulation steps to avoid big contact forces at start
        for _ in range(3):
            self.interface.step()

        self.task.reset(iter_count=self.robot.iteration_count)

        self.prev_prediction = np.zeros_like(self.prev_prediction)
        self.observation_history = collections.deque(maxlen=self.history_len)

        return self.get_obs()

    def _apply_init_noise(self, init_qpos: list) -> list:
        """Apply initialization noise to initial pose.

        Args:
            init_qpos: Initial joint positions.

        Returns:
            Noised initial joint positions.
        """
        c = self.cfg.init_noise * np.pi / 180
        joint_names = self._get_joint_names()

        root_adr = self.interface.get_jnt_qposadr_by_name("root")[0]

        # Add noise to root height
        init_qpos[root_adr + 2] = np.random.uniform(init_qpos[root_adr + 2], init_qpos[root_adr + 2] + 0.02)

        # Add noise to root orientation
        init_qpos[root_adr + 3 : root_adr + 7] = tf3.euler.euler2quat(
            np.random.uniform(-c, c), np.random.uniform(-c, c), 0
        )

        # Add noise to joint positions
        init_qpos[root_adr + 7 : root_adr + 7 + len(joint_names)] = [
            init_qpos[root_adr + 7 + i] + np.random.uniform(-c, c) for i in range(len(joint_names))
        ]

        return init_qpos

    def _apply_observation_noise(self, observations: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Apply noise to observations based on config.

        Args:
            observations: Dictionary of observation arrays.

        Returns:
            Dictionary of noised observation arrays.
        """
        if not hasattr(self.cfg, "observation_noise") or not self.cfg.observation_noise.enabled:
            return observations

        noise_cfg = self.cfg.observation_noise
        noise_type = noise_cfg.type
        scales = noise_cfg.scales
        level = noise_cfg.multiplier

        if noise_type == "uniform":
            noise_fn = lambda x, n: np.random.uniform(-x, x, n)
        elif noise_type == "gaussian":
            noise_fn = lambda x, n: np.random.randn(n) * x
        else:
            raise ValueError("Observation noise type must be 'uniform' or 'gaussian'")

        result = {}
        for key, value in observations.items():
            if hasattr(scales, key):
                scale = getattr(scales, key)
                result[key] = value + noise_fn(scale * level, len(value))
            else:
                result[key] = value
        return result
