"""Integration tests for training functionality.

These tests are parametrized to run against ALL environments discovered
under envs/. No hardcoded dimensions - everything is derived from the
environment instances.
"""
import pytest
import numpy as np
import torch
from pathlib import Path
from functools import partial

import ray

from conftest import DISCOVERED_ENVIRONMENTS, get_env_info


class TestPPOInitialization:
    """Tests for PPO algorithm initialization with any environment."""

    def test_ppo_initializes(self, env_factory, train_args, env_name):
        """Test PPO initializes correctly with any environment."""
        from rl.algos.ppo import PPO

        ppo = PPO(env_factory, train_args)

        assert ppo.policy is not None
        assert ppo.critic is not None
        assert ppo.old_policy is not None

    def test_policy_network_dimensions(self, env_factory, train_args, env_name):
        """Test policy network has correct input/output dimensions."""
        from rl.algos.ppo import PPO

        env = env_factory()
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        env.close()

        ppo = PPO(env_factory, train_args)

        assert ppo.policy.state_dim == obs_dim, \
            f"Policy state_dim {ppo.policy.state_dim} != env obs_dim {obs_dim}"
        assert ppo.policy.action_dim == action_dim, \
            f"Policy action_dim {ppo.policy.action_dim} != env action_dim {action_dim}"

    def test_critic_produces_correct_output(self, env_factory, train_args, env_name):
        """Test critic network produces scalar value."""
        from rl.algos.ppo import PPO

        env = env_factory()
        obs_dim = env.observation_space.shape[0]
        env.close()

        ppo = PPO(env_factory, train_args)

        # Test critic produces scalar value for single observation
        obs = torch.randn(obs_dim)
        value = ppo.critic(obs)
        assert value.shape == (1,), f"Critic output shape {value.shape} != (1,)"

    def test_observation_normalization_loaded(self, env_factory, train_args, env_name):
        """Test observation normalization stats are loaded into policy."""
        from rl.algos.ppo import PPO

        ppo = PPO(env_factory, train_args)

        assert hasattr(ppo.policy, 'obs_mean')
        assert hasattr(ppo.policy, 'obs_std')
        assert ppo.policy.obs_mean is not None
        assert ppo.policy.obs_std is not None

        # Dimensions should match
        env = env_factory()
        obs_dim = env.observation_space.shape[0]
        env.close()

        assert ppo.policy.obs_mean.shape[0] == obs_dim
        assert ppo.policy.obs_std.shape[0] == obs_dim


class TestPPOSampling:
    """Tests for PPO trajectory sampling."""

    def test_sample_parallel_returns_valid_data(self, env_factory, train_args, env_name):
        """Test parallel sampling returns valid batch data."""
        from rl.algos.ppo import PPO

        ppo = PPO(env_factory, train_args)
        policy_ref = ray.put(ppo.policy)
        critic_ref = ray.put(ppo.critic)

        batch = ppo.sample_parallel(env_factory, policy_ref, critic_ref)

        required_attrs = ['states', 'actions', 'rewards', 'returns', 'values', 'ep_rewards', 'ep_lens']
        for attr in required_attrs:
            assert hasattr(batch, attr), f"Batch missing attribute: {attr}"

    def test_sampled_states_match_obs_dimension(self, env_factory, train_args, env_name):
        """Test sampled states have correct observation dimension."""
        from rl.algos.ppo import PPO

        env = env_factory()
        obs_dim = env.observation_space.shape[0]
        env.close()

        ppo = PPO(env_factory, train_args)
        policy_ref = ray.put(ppo.policy)
        critic_ref = ray.put(ppo.critic)

        batch = ppo.sample_parallel(env_factory, policy_ref, critic_ref)

        assert batch.states.shape[1] == obs_dim, \
            f"Sampled states dim {batch.states.shape[1]} != obs_dim {obs_dim}"

    def test_sampled_actions_match_action_dimension(self, env_factory, train_args, env_name):
        """Test sampled actions have correct action dimension."""
        from rl.algos.ppo import PPO

        env = env_factory()
        action_dim = env.action_space.shape[0]
        env.close()

        ppo = PPO(env_factory, train_args)
        policy_ref = ray.put(ppo.policy)
        critic_ref = ray.put(ppo.critic)

        batch = ppo.sample_parallel(env_factory, policy_ref, critic_ref)

        assert batch.actions.shape[1] == action_dim, \
            f"Sampled actions dim {batch.actions.shape[1]} != action_dim {action_dim}"

    def test_sampled_values_are_scalars(self, env_factory, train_args, env_name):
        """Test sampled values are scalar per timestep."""
        from rl.algos.ppo import PPO

        ppo = PPO(env_factory, train_args)
        policy_ref = ray.put(ppo.policy)
        critic_ref = ray.put(ppo.critic)

        batch = ppo.sample_parallel(env_factory, policy_ref, critic_ref)

        assert len(batch.values.shape) == 2
        assert batch.values.shape[1] == 1


class TestPPOUpdate:
    """Tests for PPO policy update."""

    def test_update_actor_critic_runs_without_error(self, env_factory, train_args, env_name):
        """Test actor-critic update completes without errors."""
        from rl.algos.ppo import PPO

        ppo = PPO(env_factory, train_args)
        ppo.actor_optimizer = torch.optim.Adam(ppo.policy.parameters(), lr=train_args.lr)
        ppo.critic_optimizer = torch.optim.Adam(ppo.critic.parameters(), lr=train_args.lr)

        policy_ref = ray.put(ppo.policy)
        critic_ref = ray.put(ppo.critic)
        batch = ppo.sample_parallel(env_factory, policy_ref, critic_ref)

        observations = batch.states.float()
        actions = batch.actions.float()
        returns = batch.returns.float()
        values = batch.values.float()
        advantages = returns - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

        ppo.old_policy.load_state_dict(ppo.policy.state_dict())

        # Take a minibatch
        indices = list(range(min(32, len(observations))))
        obs_batch = observations[indices]
        action_batch = actions[indices]
        return_batch = returns[indices]
        advantage_batch = advantages[indices]

        result = ppo.update_actor_critic(
            obs_batch, action_batch, return_batch, advantage_batch, mask=1
        )

        assert len(result) == 7  # Returns 7 scalars
        actor_loss, _, critic_loss, _, _, _, _ = result
        assert torch.isfinite(actor_loss)
        assert torch.isfinite(critic_loss)

    def test_policy_weights_change_after_update(self, env_factory, train_args, env_name):
        """Test policy weights change after update."""
        from rl.algos.ppo import PPO

        ppo = PPO(env_factory, train_args)
        ppo.actor_optimizer = torch.optim.Adam(ppo.policy.parameters(), lr=1e-3)
        ppo.critic_optimizer = torch.optim.Adam(ppo.critic.parameters(), lr=1e-3)

        # Store original weights
        original_weights = {name: param.clone() for name, param in ppo.policy.named_parameters()}

        policy_ref = ray.put(ppo.policy)
        critic_ref = ray.put(ppo.critic)
        batch = ppo.sample_parallel(env_factory, policy_ref, critic_ref)

        observations = batch.states.float()
        actions = batch.actions.float()
        returns = batch.returns.float()
        values = batch.values.float()
        advantages = returns - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

        ppo.old_policy.load_state_dict(ppo.policy.state_dict())

        indices = list(range(min(32, len(observations))))
        ppo.update_actor_critic(
            observations[indices],
            actions[indices],
            returns[indices],
            advantages[indices],
            mask=1
        )

        # Check weights changed
        weights_changed = False
        for name, param in ppo.policy.named_parameters():
            if not torch.allclose(original_weights[name], param):
                weights_changed = True
                break
        assert weights_changed, "Policy weights should change after update"


@pytest.mark.slow
class TestTrainingLoop:
    """Tests for full training loop (marked slow)."""

    @pytest.mark.timeout(120)
    def test_training_one_iteration(self, env_factory, train_args, env_name):
        """Test training runs for one iteration without errors."""
        from rl.algos.ppo import PPO

        train_args.n_itr = 1
        train_args.eval_freq = 1

        ppo = PPO(env_factory, train_args)
        ppo.train(env_factory, n_itr=1)

        # Check that weights were saved
        assert (train_args.logdir / "actor_0.pt").exists(), \
            f"actor_0.pt not saved for {env_name}"
        assert (train_args.logdir / "critic_0.pt").exists(), \
            f"critic_0.pt not saved for {env_name}"


class TestSymmetricEnvWrapper:
    """Tests for the SymmetricEnv wrapper used in training."""

    def test_symmetric_env_wraps_if_supported(self, env_factory, env_name):
        """Test SymmetricEnv wrapper works with environments that support it."""
        from rl.envs.wrappers import SymmetricEnv

        base_env = env_factory()

        if not hasattr(base_env.robot, 'mirrored_obs') or not hasattr(base_env.robot, 'mirrored_acts'):
            base_env.close()
            pytest.skip(f"{env_name} does not support mirror symmetry")

        wrapped_env_fn = partial(
            SymmetricEnv,
            env_factory,
            mirrored_obs=base_env.robot.mirrored_obs,
            mirrored_act=base_env.robot.mirrored_acts,
            clock_inds=getattr(base_env.robot, 'clock_inds', [])
        )

        wrapped_env = wrapped_env_fn()
        obs = wrapped_env.reset()

        assert obs.shape == base_env.observation_space.shape
        assert hasattr(wrapped_env, 'mirror_observation')
        assert hasattr(wrapped_env, 'mirror_action')

        base_env.close()
        wrapped_env.close()

    def test_symmetric_env_step_works(self, env_factory, env_name):
        """Test stepping through SymmetricEnv wrapper works."""
        from rl.envs.wrappers import SymmetricEnv

        base_env = env_factory()

        if not hasattr(base_env.robot, 'mirrored_obs') or not hasattr(base_env.robot, 'mirrored_acts'):
            base_env.close()
            pytest.skip(f"{env_name} does not support mirror symmetry")

        action_dim = base_env.action_space.shape[0]

        wrapped_env_fn = partial(
            SymmetricEnv,
            env_factory,
            mirrored_obs=base_env.robot.mirrored_obs,
            mirrored_act=base_env.robot.mirrored_acts,
            clock_inds=getattr(base_env.robot, 'clock_inds', [])
        )

        wrapped_env = wrapped_env_fn()
        wrapped_env.reset()

        for _ in range(10):
            action = np.random.uniform(-1, 1, action_dim)
            obs, reward, done, info = wrapped_env.step(action)
            assert not np.any(np.isnan(obs))
            if done:
                wrapped_env.reset()

        base_env.close()
        wrapped_env.close()


class TestPolicyNetworks:
    """Tests for actor/critic network architectures."""

    def test_ff_actor_forward_pass(self, env_factory, env_name):
        """Test feed-forward actor forward pass with environment dimensions."""
        from rl.policies.actor import Gaussian_FF_Actor

        env = env_factory()
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        env.close()

        actor = Gaussian_FF_Actor(obs_dim, action_dim, init_std=0.2, learn_std=False)

        obs = torch.randn(32, obs_dim)
        action = actor(obs)

        assert action.shape == (32, action_dim)
        assert torch.all(torch.isfinite(action))

    def test_ff_actor_distribution(self, env_factory, env_name):
        """Test feed-forward actor distribution."""
        from rl.policies.actor import Gaussian_FF_Actor

        env = env_factory()
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        env.close()

        actor = Gaussian_FF_Actor(obs_dim, action_dim, init_std=0.2, learn_std=False)

        obs = torch.randn(32, obs_dim)
        dist = actor.distribution(obs)

        assert dist is not None
        sample = dist.sample()
        assert sample.shape == (32, action_dim)

    def test_ff_critic_forward_pass(self, env_factory, env_name):
        """Test feed-forward critic forward pass with environment dimensions."""
        from rl.policies.critic import FF_V

        env = env_factory()
        obs_dim = env.observation_space.shape[0]
        obs_mean = torch.zeros(obs_dim)
        obs_std = torch.ones(obs_dim)
        env.close()

        critic = FF_V(obs_dim, obs_mean=obs_mean, obs_std=obs_std)

        obs = torch.randn(32, obs_dim)
        value = critic(obs)

        assert value.shape == (32, 1)
        assert torch.all(torch.isfinite(value))

    def test_lstm_actor_forward_pass(self, env_factory, env_name):
        """Test LSTM actor forward pass with environment dimensions."""
        from rl.policies.actor import Gaussian_LSTM_Actor

        env = env_factory()
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        env.close()

        actor = Gaussian_LSTM_Actor(obs_dim, action_dim, init_std=0.2, learn_std=False)
        actor.init_hidden_state()

        obs = torch.randn(obs_dim)  # Single observation
        action = actor(obs)

        assert action.shape == (action_dim,)
        assert torch.all(torch.isfinite(action))

    def test_lstm_critic_forward_pass(self, env_factory, env_name):
        """Test LSTM critic forward pass with environment dimensions."""
        from rl.policies.critic import LSTM_V

        env = env_factory()
        obs_dim = env.observation_space.shape[0]
        env.close()

        critic = LSTM_V(obs_dim)
        critic.init_hidden_state()
        # LSTM_V requires obs_mean/obs_std to be set
        critic.obs_mean = torch.zeros(obs_dim)
        critic.obs_std = torch.ones(obs_dim)

        obs = torch.randn(obs_dim)
        value = critic(obs)

        assert value.shape == (1,)
        assert torch.all(torch.isfinite(value))


class TestRecurrentTraining:
    """Tests for recurrent (LSTM) policy training."""

    def test_ppo_initializes_with_recurrent_flag(self, env_factory, train_args, env_name):
        """Test PPO initializes with LSTM networks when recurrent flag set."""
        from rl.algos.ppo import PPO
        from rl.policies.actor import Gaussian_LSTM_Actor
        from rl.policies.critic import LSTM_V

        train_args.recurrent = True
        ppo = PPO(env_factory, train_args)

        assert isinstance(ppo.policy, Gaussian_LSTM_Actor)
        assert isinstance(ppo.critic, LSTM_V)

    @pytest.mark.timeout(120)
    @pytest.mark.slow
    def test_recurrent_training_one_iteration(self, env_factory, train_args, env_name):
        """Test recurrent training runs for one iteration."""
        from rl.algos.ppo import PPO

        train_args.recurrent = True
        train_args.n_itr = 1
        train_args.eval_freq = 1

        ppo = PPO(env_factory, train_args)
        ppo.train(env_factory, n_itr=1)

        assert (train_args.logdir / "actor_0.pt").exists()
