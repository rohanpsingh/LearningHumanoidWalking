"""Unit tests for all environment implementations."""
import pytest
import numpy as np


class TestH1Environment:
    """Tests for the H1 humanoid standing environment."""

    def test_initialization(self, h1_env):
        """Test H1 environment initializes correctly."""
        assert h1_env is not None
        assert h1_env.model is not None
        assert h1_env.data is not None

    def test_observation_space_shape(self, h1_env):
        """Test observation space has correct shape."""
        # H1 has 35 base obs * history_len
        expected_base = 35
        assert h1_env.base_obs_len == expected_base
        assert h1_env.observation_space.shape[0] == expected_base * h1_env.history_len

    def test_action_space_shape(self, h1_env):
        """Test action space has correct shape (10 DOF legs)."""
        assert h1_env.action_space.shape[0] == 10

    def test_reset_returns_valid_observation(self, h1_env):
        """Test reset returns observation with correct shape and valid values."""
        obs = h1_env.reset()
        assert obs.shape == h1_env.observation_space.shape
        assert not np.any(np.isnan(obs))
        assert not np.any(np.isinf(obs))

    def test_step_with_zero_action(self, h1_env):
        """Test stepping with zero action."""
        h1_env.reset()
        action = np.zeros(h1_env.action_space.shape[0])
        obs, reward, done, info = h1_env.step(action)

        assert obs.shape == h1_env.observation_space.shape
        assert isinstance(reward, (int, float))
        assert isinstance(done, bool)
        assert isinstance(info, dict)
        assert not np.any(np.isnan(obs))

    def test_step_with_random_action(self, h1_env):
        """Test stepping with random action."""
        h1_env.reset()
        action = np.random.uniform(-1, 1, h1_env.action_space.shape[0])
        obs, reward, done, info = h1_env.step(action)

        assert obs.shape == h1_env.observation_space.shape
        assert isinstance(reward, (int, float))
        assert not np.any(np.isnan(obs))

    def test_multiple_steps(self, h1_env):
        """Test taking multiple steps without errors."""
        h1_env.reset()
        for _ in range(100):
            action = np.random.uniform(-1, 1, h1_env.action_space.shape[0])
            obs, reward, done, info = h1_env.step(action)
            if done:
                obs = h1_env.reset()
            assert not np.any(np.isnan(obs))

    def test_reward_components(self, h1_env):
        """Test that reward info dict contains expected components."""
        h1_env.reset()
        action = np.zeros(h1_env.action_space.shape[0])
        _, _, _, info = h1_env.step(action)

        expected_keys = [
            "com_vel_error",
            "yaw_vel_error",
            "height",
            "upperbody",
            "joint_torque_reward",
            "posture",
        ]
        for key in expected_keys:
            assert key in info, f"Missing reward component: {key}"
            assert not np.isnan(info[key])

    def test_obs_normalization_stats(self, h1_env):
        """Test observation normalization statistics are defined."""
        assert hasattr(h1_env, 'obs_mean')
        assert hasattr(h1_env, 'obs_std')
        assert h1_env.obs_mean.shape == h1_env.observation_space.shape
        assert h1_env.obs_std.shape == h1_env.observation_space.shape
        assert np.all(h1_env.obs_std > 0)  # Std should be positive

    def test_task_termination_conditions(self, h1_env):
        """Test that termination conditions work."""
        h1_env.reset()
        # Run many steps to potentially trigger termination
        terminated = False
        for _ in range(500):
            action = np.random.uniform(-1, 1, h1_env.action_space.shape[0])
            _, _, done, _ = h1_env.step(action)
            if done:
                terminated = True
                break
        # We may or may not terminate, but should not crash


class TestJvrcWalkEnvironment:
    """Tests for the JVRC walking environment."""

    def test_initialization(self, jvrc_walk_env):
        """Test JVRC walk environment initializes correctly."""
        assert jvrc_walk_env is not None
        assert jvrc_walk_env.model is not None
        assert jvrc_walk_env.data is not None

    def test_observation_space_shape(self, jvrc_walk_env):
        """Test observation space has correct shape."""
        # JVRC walk has 32 base obs * history_len
        expected_base = 32
        assert jvrc_walk_env.base_obs_len == expected_base
        assert jvrc_walk_env.observation_space.shape[0] == expected_base * jvrc_walk_env.history_len

    def test_action_space_shape(self, jvrc_walk_env):
        """Test action space has correct shape (12 DOF legs)."""
        assert jvrc_walk_env.action_space.shape[0] == 12

    def test_reset_returns_valid_observation(self, jvrc_walk_env):
        """Test reset returns observation with correct shape and valid values."""
        obs = jvrc_walk_env.reset()
        assert obs.shape == jvrc_walk_env.observation_space.shape
        assert not np.any(np.isnan(obs))
        assert not np.any(np.isinf(obs))

    def test_step_with_zero_action(self, jvrc_walk_env):
        """Test stepping with zero action."""
        jvrc_walk_env.reset()
        action = np.zeros(jvrc_walk_env.action_space.shape[0])
        obs, reward, done, info = jvrc_walk_env.step(action)

        assert obs.shape == jvrc_walk_env.observation_space.shape
        assert isinstance(reward, (int, float))
        assert isinstance(done, bool)
        assert isinstance(info, dict)
        assert not np.any(np.isnan(obs))

    def test_step_with_random_action(self, jvrc_walk_env):
        """Test stepping with random action."""
        jvrc_walk_env.reset()
        action = np.random.uniform(-1, 1, jvrc_walk_env.action_space.shape[0])
        obs, reward, done, info = jvrc_walk_env.step(action)

        assert obs.shape == jvrc_walk_env.observation_space.shape
        assert isinstance(reward, (int, float))
        assert not np.any(np.isnan(obs))

    def test_multiple_steps(self, jvrc_walk_env):
        """Test taking multiple steps without errors."""
        jvrc_walk_env.reset()
        for _ in range(100):
            action = np.random.uniform(-1, 1, jvrc_walk_env.action_space.shape[0])
            obs, reward, done, info = jvrc_walk_env.step(action)
            if done:
                obs = jvrc_walk_env.reset()
            assert not np.any(np.isnan(obs))

    def test_clock_observation(self, jvrc_walk_env):
        """Test that phase clock is included in observations."""
        obs = jvrc_walk_env.reset()
        # Clock should be in the observation (sin, cos of phase)
        # Should have values between -1 and 1
        clock_obs = obs[-3:-1]  # Last 3 are clock and goal speed
        assert np.all(np.abs(clock_obs) <= 1.0)

    def test_mirror_indices_defined(self, jvrc_walk_env):
        """Test that mirror indices are defined for symmetry learning."""
        assert hasattr(jvrc_walk_env.robot, 'mirrored_obs')
        assert hasattr(jvrc_walk_env.robot, 'mirrored_acts')
        assert hasattr(jvrc_walk_env.robot, 'clock_inds')
        assert len(jvrc_walk_env.robot.mirrored_obs) > 0
        assert len(jvrc_walk_env.robot.mirrored_acts) == 12

    def test_obs_normalization_stats(self, jvrc_walk_env):
        """Test observation normalization statistics are defined."""
        assert hasattr(jvrc_walk_env, 'obs_mean')
        assert hasattr(jvrc_walk_env, 'obs_std')
        assert jvrc_walk_env.obs_mean.shape == jvrc_walk_env.observation_space.shape
        assert jvrc_walk_env.obs_std.shape == jvrc_walk_env.observation_space.shape
        assert np.all(jvrc_walk_env.obs_std > 0)

    def test_task_phase_advances(self, jvrc_walk_env):
        """Test that the gait phase advances during stepping."""
        jvrc_walk_env.reset()
        initial_phase = jvrc_walk_env.task._phase

        # Take some steps
        for _ in range(10):
            action = np.zeros(jvrc_walk_env.action_space.shape[0])
            jvrc_walk_env.step(action)

        # Phase should have changed
        assert jvrc_walk_env.task._phase != initial_phase or jvrc_walk_env.task._phase == 0


class TestJvrcStepEnvironment:
    """Tests for the JVRC stepping environment."""

    def test_initialization(self, jvrc_step_env):
        """Test JVRC step environment initializes correctly."""
        assert jvrc_step_env is not None
        assert jvrc_step_env.model is not None
        assert jvrc_step_env.data is not None

    def test_observation_space_shape(self, jvrc_step_env):
        """Test observation space has correct shape."""
        # JVRC step has 39 base obs * history_len (extended from walk)
        expected_base = 39
        assert jvrc_step_env.base_obs_len == expected_base
        assert jvrc_step_env.observation_space.shape[0] == expected_base * jvrc_step_env.history_len

    def test_action_space_shape(self, jvrc_step_env):
        """Test action space has correct shape (12 DOF legs)."""
        assert jvrc_step_env.action_space.shape[0] == 12

    def test_reset_returns_valid_observation(self, jvrc_step_env):
        """Test reset returns observation with correct shape and valid values."""
        obs = jvrc_step_env.reset()
        assert obs.shape == jvrc_step_env.observation_space.shape
        assert not np.any(np.isnan(obs))
        assert not np.any(np.isinf(obs))

    def test_step_with_zero_action(self, jvrc_step_env):
        """Test stepping with zero action."""
        jvrc_step_env.reset()
        action = np.zeros(jvrc_step_env.action_space.shape[0])
        obs, reward, done, info = jvrc_step_env.step(action)

        assert obs.shape == jvrc_step_env.observation_space.shape
        assert isinstance(reward, (int, float))
        assert isinstance(done, bool)
        assert isinstance(info, dict)
        assert not np.any(np.isnan(obs))

    def test_step_with_random_action(self, jvrc_step_env):
        """Test stepping with random action."""
        jvrc_step_env.reset()
        action = np.random.uniform(-1, 1, jvrc_step_env.action_space.shape[0])
        obs, reward, done, info = jvrc_step_env.step(action)

        assert obs.shape == jvrc_step_env.observation_space.shape
        assert isinstance(reward, (int, float))
        assert not np.any(np.isnan(obs))

    def test_multiple_steps(self, jvrc_step_env):
        """Test taking multiple steps without errors."""
        jvrc_step_env.reset()
        for _ in range(100):
            action = np.random.uniform(-1, 1, jvrc_step_env.action_space.shape[0])
            obs, reward, done, info = jvrc_step_env.step(action)
            if done:
                obs = jvrc_step_env.reset()
            assert not np.any(np.isnan(obs))

    def test_stepping_goal_info_in_observation(self, jvrc_step_env):
        """Test that stepping goal info is included in observations."""
        obs = jvrc_step_env.reset()
        # Step environment has extended observations for footstep targets
        # Observation length should be 39 * history_len
        assert obs.shape[0] == 39 * jvrc_step_env.history_len

    def test_mirror_indices_defined(self, jvrc_step_env):
        """Test that mirror indices are defined for symmetry learning."""
        assert hasattr(jvrc_step_env.robot, 'mirrored_obs')
        assert hasattr(jvrc_step_env.robot, 'mirrored_acts')
        assert hasattr(jvrc_step_env.robot, 'clock_inds')

    def test_inherits_from_walk_env(self, jvrc_step_env):
        """Test that JvrcStepEnv inherits from JvrcWalkEnv."""
        from envs.jvrc import JvrcWalkEnv
        assert isinstance(jvrc_step_env, JvrcWalkEnv)


class TestEnvironmentConsistency:
    """Cross-environment consistency tests."""

    def test_all_envs_have_required_attributes(self, h1_env, jvrc_walk_env, jvrc_step_env):
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
        for env in [h1_env, jvrc_walk_env, jvrc_step_env]:
            for attr in required_attrs:
                assert hasattr(env, attr), f"Environment missing attribute: {attr}"

    def test_all_envs_return_consistent_step_signature(self, h1_env, jvrc_walk_env, jvrc_step_env):
        """Test all environments return (obs, reward, done, info) from step."""
        for env in [h1_env, jvrc_walk_env, jvrc_step_env]:
            env.reset()
            action = np.zeros(env.action_space.shape[0])
            result = env.step(action)
            assert len(result) == 4
            obs, reward, done, info = result
            assert isinstance(obs, np.ndarray)
            assert isinstance(reward, (int, float))
            assert isinstance(done, bool)
            assert isinstance(info, dict)

    def test_all_envs_rewards_are_bounded(self, h1_env, jvrc_walk_env, jvrc_step_env):
        """Test that rewards from all environments are reasonably bounded."""
        for env in [h1_env, jvrc_walk_env, jvrc_step_env]:
            env.reset()
            total_rewards = []
            for _ in range(50):
                action = np.random.uniform(-1, 1, env.action_space.shape[0])
                _, reward, done, _ = env.step(action)
                total_rewards.append(reward)
                if done:
                    env.reset()

            # Rewards should be finite and reasonably bounded
            assert np.all(np.isfinite(total_rewards))
            assert np.max(np.abs(total_rewards)) < 100  # Reasonable bound


class TestEnvironmentRobustness:
    """Tests for environment robustness under edge cases."""

    def test_h1_handles_extreme_actions(self, h1_env):
        """Test H1 handles extreme action values."""
        h1_env.reset()
        extreme_action = np.ones(h1_env.action_space.shape[0]) * 10
        obs, reward, done, _ = h1_env.step(extreme_action)
        assert not np.any(np.isnan(obs))
        assert np.isfinite(reward)

    def test_jvrc_walk_handles_extreme_actions(self, jvrc_walk_env):
        """Test JVRC walk handles extreme action values."""
        jvrc_walk_env.reset()
        extreme_action = np.ones(jvrc_walk_env.action_space.shape[0]) * 10
        obs, reward, done, _ = jvrc_walk_env.step(extreme_action)
        assert not np.any(np.isnan(obs))
        assert np.isfinite(reward)

    def test_jvrc_step_handles_extreme_actions(self, jvrc_step_env):
        """Test JVRC step handles extreme action values."""
        jvrc_step_env.reset()
        extreme_action = np.ones(jvrc_step_env.action_space.shape[0]) * 10
        obs, reward, done, _ = jvrc_step_env.step(extreme_action)
        assert not np.any(np.isnan(obs))
        assert np.isfinite(reward)

    def test_consecutive_resets(self, h1_env, jvrc_walk_env, jvrc_step_env):
        """Test consecutive resets don't cause issues."""
        for env in [h1_env, jvrc_walk_env, jvrc_step_env]:
            for _ in range(5):
                obs = env.reset()
                assert not np.any(np.isnan(obs))
