"""Tests for evaluation functionality.

These tests are parametrized to run against ALL environments discovered
under envs/. No hardcoded dimensions.
"""
import pytest
import numpy as np
import torch
import pickle
from pathlib import Path

from conftest import DISCOVERED_ENVIRONMENTS


class TestModelSaveLoad:
    """Tests for saving and loading trained models."""

    def test_save_and_load_actor(self, env_factory, train_args, env_name):
        """Test actor network can be saved and loaded."""
        from rl.algos.ppo import PPO

        ppo = PPO(env_factory, train_args)

        # Save
        actor_path = train_args.logdir / "actor_test.pt"
        torch.save(ppo.policy, actor_path)

        # Load
        loaded_actor = torch.load(actor_path, weights_only=False)

        # Test loaded model works
        env = env_factory()
        obs_dim = env.observation_space.shape[0]
        env.close()

        obs = torch.randn(obs_dim)
        original_action = ppo.policy(obs, deterministic=True)
        loaded_action = loaded_actor(obs, deterministic=True)

        assert torch.allclose(original_action, loaded_action)

    def test_save_and_load_critic(self, env_factory, train_args, env_name):
        """Test critic network can be saved and loaded."""
        from rl.algos.ppo import PPO

        ppo = PPO(env_factory, train_args)

        # Save
        critic_path = train_args.logdir / "critic_test.pt"
        torch.save(ppo.critic, critic_path)

        # Load
        loaded_critic = torch.load(critic_path, weights_only=False)

        # Test loaded model works
        env = env_factory()
        obs_dim = env.observation_space.shape[0]
        env.close()

        obs = torch.randn(obs_dim)
        original_value = ppo.critic(obs)
        loaded_value = loaded_critic(obs)

        assert torch.allclose(original_value, loaded_value)

    def test_save_and_load_experiment_pickle(self, train_args, env_name):
        """Test experiment args can be pickled and loaded."""
        pkl_path = train_args.logdir / "experiment.pkl"

        # Save
        with open(pkl_path, 'wb') as f:
            pickle.dump(train_args, f)

        # Load
        with open(pkl_path, 'rb') as f:
            loaded_args = pickle.load(f)

        assert loaded_args.env == train_args.env
        assert loaded_args.lr == train_args.lr
        assert loaded_args.gamma == train_args.gamma


class TestDeterministicEvaluation:
    """Tests for deterministic policy evaluation."""

    def test_deterministic_action_is_consistent(self, env_factory, train_args, env_name):
        """Test deterministic action is consistent for same input."""
        from rl.algos.ppo import PPO

        ppo = PPO(env_factory, train_args)
        ppo.policy.eval()

        env = env_factory()
        obs_dim = env.observation_space.shape[0]
        env.close()

        obs = torch.randn(obs_dim)
        action1 = ppo.policy(obs, deterministic=True)
        action2 = ppo.policy(obs, deterministic=True)

        assert torch.allclose(action1, action2)

    def test_stochastic_action_varies(self, env_factory, train_args, env_name):
        """Test stochastic actions vary (with high probability)."""
        from rl.algos.ppo import PPO

        ppo = PPO(env_factory, train_args)

        env = env_factory()
        obs_dim = env.observation_space.shape[0]
        env.close()

        obs = torch.randn(obs_dim)
        actions = [ppo.policy(obs, deterministic=False) for _ in range(10)]

        # At least some actions should differ
        all_same = all(torch.allclose(actions[0], a) for a in actions[1:])
        assert not all_same, "Stochastic actions should vary"


class TestEvaluationRollout:
    """Tests for evaluation rollouts."""

    def test_evaluation_rollout(self, env_factory, train_args, env_name):
        """Test evaluation rollout works for any environment."""
        from rl.algos.ppo import PPO

        ppo = PPO(env_factory, train_args)
        ppo.policy.eval()

        env = env_factory()
        obs = env.reset()

        total_reward = 0
        steps = 0
        max_steps = 100

        while steps < max_steps:
            obs_tensor = torch.tensor(obs, dtype=torch.float32)
            with torch.no_grad():
                action = ppo.policy(obs_tensor, deterministic=True).numpy()

            obs, reward, done, _ = env.step(action)
            total_reward += reward
            steps += 1

            if done:
                break

        env.close()

        assert steps > 0
        assert np.isfinite(total_reward)


@pytest.mark.slow
class TestFullEvaluationPipeline:
    """Tests for the full evaluation pipeline including model loading."""

    @pytest.mark.timeout(180)
    def test_train_save_load_evaluate(self, env_factory, train_args, env_name):
        """Test full pipeline: train, save, load, evaluate for any environment."""
        from rl.algos.ppo import PPO

        # Train for 1 iteration
        train_args.n_itr = 1
        train_args.eval_freq = 1
        ppo = PPO(env_factory, train_args)
        ppo.train(env_factory, n_itr=1)

        # Save experiment args
        pkl_path = train_args.logdir / "experiment.pkl"
        with open(pkl_path, 'wb') as f:
            pickle.dump(train_args, f)

        # Load model and evaluate
        actor_path = train_args.logdir / "actor.pt"
        if not actor_path.exists():
            actor_path = train_args.logdir / "actor_0.pt"

        loaded_policy = torch.load(actor_path, weights_only=False)
        loaded_policy.eval()

        # Run evaluation rollout
        env = env_factory()
        obs = env.reset()

        steps = 0
        for _ in range(50):
            obs_tensor = torch.tensor(obs, dtype=torch.float32)
            with torch.no_grad():
                action = loaded_policy(obs_tensor, deterministic=True).numpy()
            obs, reward, done, _ = env.step(action)
            steps += 1
            if done:
                break

        env.close()
        assert steps > 0


class TestObservationNormalization:
    """Tests for observation normalization in evaluation."""

    def test_policy_normalizes_observations(self, env_factory, train_args, env_name):
        """Test policy applies observation normalization."""
        from rl.algos.ppo import PPO

        ppo = PPO(env_factory, train_args)

        # Policy should have normalization stats
        assert hasattr(ppo.policy, 'obs_mean')
        assert hasattr(ppo.policy, 'obs_std')

        # Get obs dim from env
        env = env_factory()
        obs_dim = env.observation_space.shape[0]
        env.close()

        # Normalized input should give different output than raw input
        obs = torch.randn(obs_dim) * 10  # Large values

        # The policy internally normalizes
        with torch.no_grad():
            action = ppo.policy(obs, deterministic=True)

        assert torch.all(torch.isfinite(action))

    def test_critic_normalizes_observations(self, env_factory, train_args, env_name):
        """Test critic applies observation normalization."""
        from rl.algos.ppo import PPO

        ppo = PPO(env_factory, train_args)

        # Critic should have normalization stats
        assert hasattr(ppo.critic, 'obs_mean')
        assert hasattr(ppo.critic, 'obs_std')

        # Get obs dim from env
        env = env_factory()
        obs_dim = env.observation_space.shape[0]
        env.close()

        obs = torch.randn(obs_dim) * 10

        with torch.no_grad():
            value = ppo.critic(obs)

        assert torch.all(torch.isfinite(value))


@pytest.mark.slow
class TestContinuedTraining:
    """Tests for continuing training from saved weights."""

    @pytest.mark.timeout(240)
    def test_continued_training(self, env_factory, train_args, env_name, temp_logdir):
        """Test continuing training from saved weights for any environment."""
        from rl.algos.ppo import PPO
        from argparse import Namespace

        # First training run
        train_args.n_itr = 1
        train_args.eval_freq = 1
        ppo1 = PPO(env_factory, train_args)
        ppo1.train(env_factory, n_itr=1)

        # Get path to saved actor
        actor_path = train_args.logdir / "actor_0.pt"
        assert actor_path.exists()

        # Create new args for continued training
        continued_args = Namespace(**vars(train_args))
        continued_args.logdir = temp_logdir / "continued"
        Path.mkdir(continued_args.logdir, parents=True, exist_ok=True)
        continued_args.continued = actor_path
        continued_args.n_itr = 1
        continued_args.eval_freq = 1

        # Continue training
        ppo2 = PPO(env_factory, continued_args)
        ppo2.train(env_factory, n_itr=1)

        # Verify new weights were saved
        assert (continued_args.logdir / "actor_0.pt").exists()
