"""Unit tests for all environment implementations.

These tests are parametrized to run against ALL environments discovered
under envs/. When new environments are added, they will automatically
be included in the test suite.
"""
import pytest
import numpy as np

from conftest import get_all_env_instances, close_all_envs, DISCOVERED_ENVIRONMENTS


class TestEnvironmentBasics:
    """Basic tests that run for every discovered environment."""

    def test_initialization(self, env_instance, env_name):
        """Test environment initializes correctly."""
        assert env_instance is not None
        assert env_instance.model is not None
        assert env_instance.data is not None

    def test_has_observation_space(self, env_instance, env_name):
        """Test environment has observation_space attribute."""
        assert hasattr(env_instance, 'observation_space')
        assert env_instance.observation_space is not None
        assert hasattr(env_instance.observation_space, 'shape')
        obs_dim = env_instance.observation_space.shape[0]
        assert obs_dim > 0

    def test_has_action_space(self, env_instance, env_name):
        """Test environment has action_space attribute."""
        assert hasattr(env_instance, 'action_space')
        assert env_instance.action_space is not None
        assert hasattr(env_instance.action_space, 'shape')
        action_dim = env_instance.action_space.shape[0]
        assert action_dim > 0

    def test_reset_returns_valid_observation(self, env_instance, env_name):
        """Test reset returns observation with correct shape and valid values."""
        obs = env_instance.reset()

        expected_shape = env_instance.observation_space.shape
        assert obs.shape == expected_shape, \
            f"Expected obs shape {expected_shape}, got {obs.shape}"
        assert not np.any(np.isnan(obs)), "Observation contains NaN"
        assert not np.any(np.isinf(obs)), "Observation contains Inf"

    def test_step_with_zero_action(self, env_instance, env_name):
        """Test stepping with zero action."""
        env_instance.reset()
        action_dim = env_instance.action_space.shape[0]
        action = np.zeros(action_dim)

        obs, reward, done, info = env_instance.step(action)

        assert obs.shape == env_instance.observation_space.shape
        assert isinstance(reward, (int, float))
        assert isinstance(done, bool)
        assert isinstance(info, dict)
        assert not np.any(np.isnan(obs)), "Observation contains NaN after step"

    def test_step_with_random_action(self, env_instance, env_name):
        """Test stepping with random action."""
        env_instance.reset()
        action_dim = env_instance.action_space.shape[0]
        action = np.random.uniform(-1, 1, action_dim)

        obs, reward, done, info = env_instance.step(action)

        assert obs.shape == env_instance.observation_space.shape
        assert isinstance(reward, (int, float))
        assert not np.any(np.isnan(obs))

    def test_multiple_steps(self, env_instance, env_name):
        """Test taking multiple steps without errors."""
        env_instance.reset()
        action_dim = env_instance.action_space.shape[0]

        for _ in range(100):
            action = np.random.uniform(-1, 1, action_dim)
            obs, reward, done, info = env_instance.step(action)
            if done:
                obs = env_instance.reset()
            assert not np.any(np.isnan(obs))

    def test_reward_is_dict(self, env_instance, env_name):
        """Test that step returns reward info as a dict."""
        env_instance.reset()
        action_dim = env_instance.action_space.shape[0]
        action = np.zeros(action_dim)

        _, _, _, info = env_instance.step(action)

        assert isinstance(info, dict)
        # All reward components should be finite
        for key, value in info.items():
            if isinstance(value, (int, float)):
                assert np.isfinite(value), f"Reward component {key} is not finite"

    def test_consecutive_resets(self, env_instance, env_name):
        """Test consecutive resets don't cause issues."""
        for _ in range(5):
            obs = env_instance.reset()
            assert not np.any(np.isnan(obs))

    def test_handles_extreme_actions(self, env_instance, env_name):
        """Test environment handles extreme action values."""
        env_instance.reset()
        action_dim = env_instance.action_space.shape[0]
        extreme_action = np.ones(action_dim) * 10

        obs, reward, done, _ = env_instance.step(extreme_action)

        assert not np.any(np.isnan(obs)), "NaN in observation after extreme action"
        assert np.isfinite(reward), "Reward not finite after extreme action"


class TestEnvironmentAttributes:
    """Tests for required environment attributes."""

    def test_has_robot_attribute(self, env_instance, env_name):
        """Test environment has robot attribute."""
        assert hasattr(env_instance, 'robot')
        assert env_instance.robot is not None

    def test_has_task_attribute(self, env_instance, env_name):
        """Test environment has task attribute."""
        assert hasattr(env_instance, 'task')
        assert env_instance.task is not None

    def test_has_interface_attribute(self, env_instance, env_name):
        """Test environment has interface attribute."""
        assert hasattr(env_instance, 'interface')
        assert env_instance.interface is not None

    def test_has_obs_normalization_stats(self, env_instance, env_name):
        """Test environment has observation normalization stats."""
        assert hasattr(env_instance, 'obs_mean'), \
            f"Environment {env_name} missing obs_mean attribute"
        assert hasattr(env_instance, 'obs_std'), \
            f"Environment {env_name} missing obs_std attribute"

        obs_dim = env_instance.observation_space.shape[0]
        assert env_instance.obs_mean.shape == (obs_dim,), \
            f"obs_mean shape mismatch: expected ({obs_dim},), got {env_instance.obs_mean.shape}"
        assert env_instance.obs_std.shape == (obs_dim,), \
            f"obs_std shape mismatch: expected ({obs_dim},), got {env_instance.obs_std.shape}"
        assert np.all(env_instance.obs_std > 0), "obs_std should be positive"


class TestEnvironmentRewards:
    """Tests for reward computation."""

    def test_rewards_are_bounded(self, env_instance, env_name):
        """Test that rewards are reasonably bounded."""
        env_instance.reset()
        action_dim = env_instance.action_space.shape[0]

        total_rewards = []
        for _ in range(50):
            action = np.random.uniform(-1, 1, action_dim)
            _, reward, done, _ = env_instance.step(action)
            total_rewards.append(reward)
            if done:
                env_instance.reset()

        assert np.all(np.isfinite(total_rewards)), "Some rewards are not finite"
        # Rewards should be reasonably bounded (adjust if needed)
        assert np.max(np.abs(total_rewards)) < 1000, "Rewards seem unbounded"

    def test_reward_components_sum_to_total(self, env_instance, env_name):
        """Test that reward dict values sum to the returned reward."""
        env_instance.reset()
        action_dim = env_instance.action_space.shape[0]
        action = np.zeros(action_dim)

        _, total_reward, _, info = env_instance.step(action)

        # Sum numeric values in info dict (reward components)
        component_sum = sum(v for v in info.values() if isinstance(v, (int, float)))

        # Allow small floating point tolerance
        assert abs(total_reward - component_sum) < 1e-6, \
            f"Total reward {total_reward} != sum of components {component_sum}"


class TestMirrorSymmetry:
    """Tests for environments that support mirror symmetry learning."""

    def test_mirror_indices_valid_if_present(self, env_instance, env_name):
        """Test mirror indices are valid if environment supports symmetry."""
        if not hasattr(env_instance.robot, 'mirrored_obs'):
            pytest.skip(f"{env_name} does not support mirror symmetry")

        obs_dim = env_instance.observation_space.shape[0] // env_instance.history_len
        action_dim = env_instance.action_space.shape[0]

        # Check mirrored_obs indices are valid
        mirrored_obs = env_instance.robot.mirrored_obs
        assert len(mirrored_obs) > 0
        for idx in mirrored_obs:
            abs_idx = abs(idx) if isinstance(idx, (int, float)) else abs(int(idx))
            # Index should be within observation dimension (accounting for sign encoding)
            assert abs_idx < obs_dim or abs(idx) < 1, \
                f"Invalid mirror obs index: {idx}, obs_dim={obs_dim}"

        # Check mirrored_acts indices are valid
        if hasattr(env_instance.robot, 'mirrored_acts'):
            mirrored_acts = env_instance.robot.mirrored_acts
            assert len(mirrored_acts) == action_dim, \
                f"mirrored_acts length {len(mirrored_acts)} != action_dim {action_dim}"

    def test_clock_indices_valid_if_present(self, env_instance, env_name):
        """Test clock indices are valid if environment has clock."""
        if not hasattr(env_instance.robot, 'clock_inds'):
            pytest.skip(f"{env_name} does not have clock indices")

        clock_inds = env_instance.robot.clock_inds
        obs_dim = env_instance.observation_space.shape[0] // env_instance.history_len

        for idx in clock_inds:
            assert 0 <= idx < obs_dim, f"Invalid clock index: {idx}"


class TestEnvironmentConsistency:
    """Cross-environment consistency tests."""

    def test_all_envs_have_required_attributes(self):
        """Test all environments have required attributes."""
        required_attrs = [
            'observation_space',
            'action_space',
            'obs_mean',
            'obs_std',
            'robot',
            'task',
            'interface',
        ]

        envs = get_all_env_instances()
        try:
            for env_name, env in envs:
                for attr in required_attrs:
                    assert hasattr(env, attr), \
                        f"Environment {env_name} missing attribute: {attr}"
        finally:
            close_all_envs(envs)

    def test_all_envs_return_consistent_step_signature(self):
        """Test all environments return (obs, reward, done, info) from step."""
        envs = get_all_env_instances()
        try:
            for env_name, env in envs:
                env.reset()
                action = np.zeros(env.action_space.shape[0])
                result = env.step(action)

                assert len(result) == 4, \
                    f"{env_name} step() should return 4 values, got {len(result)}"
                obs, reward, done, info = result
                assert isinstance(obs, np.ndarray), \
                    f"{env_name}: obs should be ndarray"
                assert isinstance(reward, (int, float)), \
                    f"{env_name}: reward should be numeric"
                assert isinstance(done, bool), \
                    f"{env_name}: done should be bool"
                assert isinstance(info, dict), \
                    f"{env_name}: info should be dict"
        finally:
            close_all_envs(envs)

    def test_all_envs_dimensions_consistent(self):
        """Test observation and action dimensions are consistent across methods."""
        envs = get_all_env_instances()
        try:
            for env_name, env in envs:
                obs_dim = env.observation_space.shape[0]
                action_dim = env.action_space.shape[0]

                # Reset should return correct obs dim
                obs = env.reset()
                assert obs.shape[0] == obs_dim, \
                    f"{env_name}: reset obs dim mismatch"

                # Step should return correct obs dim
                action = np.zeros(action_dim)
                obs, _, _, _ = env.step(action)
                assert obs.shape[0] == obs_dim, \
                    f"{env_name}: step obs dim mismatch"

                # obs_mean/obs_std should match obs dim
                assert env.obs_mean.shape[0] == obs_dim, \
                    f"{env_name}: obs_mean dim mismatch"
                assert env.obs_std.shape[0] == obs_dim, \
                    f"{env_name}: obs_std dim mismatch"
        finally:
            close_all_envs(envs)


class TestDiscoveredEnvironments:
    """Meta-tests to verify environment discovery."""

    def test_environments_discovered(self):
        """Test that at least some environments were discovered."""
        assert len(DISCOVERED_ENVIRONMENTS) > 0, \
            "No environments discovered under envs/"

    def test_discovered_environments_info(self):
        """Print discovered environments for debugging."""
        print(f"\nDiscovered {len(DISCOVERED_ENVIRONMENTS)} environments:")
        for env_name, info in DISCOVERED_ENVIRONMENTS.items():
            print(f"  - {env_name}: {info['class'].__name__} from {info['robot']}")
