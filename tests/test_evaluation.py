"""Tests for evaluation functionality."""
import pytest
import numpy as np
import torch
import pickle
import tempfile
import shutil
from pathlib import Path
from functools import partial
from argparse import Namespace


class TestModelSaveLoad:
    """Tests for saving and loading trained models."""

    def test_save_and_load_actor(self, h1_env_fn, h1_train_args):
        """Test actor network can be saved and loaded."""
        from rl.algos.ppo import PPO

        ppo = PPO(h1_env_fn, h1_train_args)

        # Save
        actor_path = h1_train_args.logdir / "actor_test.pt"
        torch.save(ppo.policy, actor_path)

        # Load
        loaded_actor = torch.load(actor_path, weights_only=False)

        # Test loaded model works
        obs = torch.randn(35)
        original_action = ppo.policy(obs, deterministic=True)
        loaded_action = loaded_actor(obs, deterministic=True)

        assert torch.allclose(original_action, loaded_action)

    def test_save_and_load_critic(self, h1_env_fn, h1_train_args):
        """Test critic network can be saved and loaded."""
        from rl.algos.ppo import PPO

        ppo = PPO(h1_env_fn, h1_train_args)

        # Save
        critic_path = h1_train_args.logdir / "critic_test.pt"
        torch.save(ppo.critic, critic_path)

        # Load
        loaded_critic = torch.load(critic_path, weights_only=False)

        # Test loaded model works
        obs = torch.randn(35)
        original_value = ppo.critic(obs)
        loaded_value = loaded_critic(obs)

        assert torch.allclose(original_value, loaded_value)

    def test_save_and_load_experiment_pickle(self, h1_train_args):
        """Test experiment args can be pickled and loaded."""
        pkl_path = h1_train_args.logdir / "experiment.pkl"

        # Save
        with open(pkl_path, 'wb') as f:
            pickle.dump(h1_train_args, f)

        # Load
        with open(pkl_path, 'rb') as f:
            loaded_args = pickle.load(f)

        assert loaded_args.env == h1_train_args.env
        assert loaded_args.lr == h1_train_args.lr
        assert loaded_args.gamma == h1_train_args.gamma


class TestDeterministicEvaluation:
    """Tests for deterministic policy evaluation."""

    def test_deterministic_action_is_consistent(self, h1_env_fn, h1_train_args):
        """Test deterministic action is consistent for same input."""
        from rl.algos.ppo import PPO

        ppo = PPO(h1_env_fn, h1_train_args)
        ppo.policy.eval()

        obs = torch.randn(35)
        action1 = ppo.policy(obs, deterministic=True)
        action2 = ppo.policy(obs, deterministic=True)

        assert torch.allclose(action1, action2)

    def test_stochastic_action_varies(self, h1_env_fn, h1_train_args):
        """Test stochastic actions vary (with high probability)."""
        from rl.algos.ppo import PPO

        ppo = PPO(h1_env_fn, h1_train_args)

        obs = torch.randn(35)
        actions = [ppo.policy(obs, deterministic=False) for _ in range(10)]

        # At least some actions should differ
        all_same = all(torch.allclose(actions[0], a) for a in actions[1:])
        assert not all_same, "Stochastic actions should vary"


class TestEvaluationRollout:
    """Tests for evaluation rollouts."""

    def test_evaluation_rollout_h1(self, h1_env_fn, h1_train_args):
        """Test evaluation rollout works for H1 environment."""
        from rl.algos.ppo import PPO

        ppo = PPO(h1_env_fn, h1_train_args)
        ppo.policy.eval()

        env = h1_env_fn()
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

    def test_evaluation_rollout_jvrc_walk(self, jvrc_walk_env_fn, jvrc_walk_train_args):
        """Test evaluation rollout works for JVRC walk environment."""
        from rl.algos.ppo import PPO

        ppo = PPO(jvrc_walk_env_fn, jvrc_walk_train_args)
        ppo.policy.eval()

        env = jvrc_walk_env_fn()
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

    def test_evaluation_rollout_jvrc_step(self, jvrc_step_env_fn, jvrc_step_train_args):
        """Test evaluation rollout works for JVRC step environment."""
        from rl.algos.ppo import PPO

        ppo = PPO(jvrc_step_env_fn, jvrc_step_train_args)
        ppo.policy.eval()

        env = jvrc_step_env_fn()
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
    def test_train_save_load_evaluate_h1(self, h1_env_fn, h1_train_args):
        """Test full pipeline: train, save, load, evaluate for H1."""
        from rl.algos.ppo import PPO

        # Train for 1 iteration
        h1_train_args.n_itr = 1
        h1_train_args.eval_freq = 1
        ppo = PPO(h1_env_fn, h1_train_args)
        ppo.train(h1_env_fn, n_itr=1)

        # Save experiment args
        pkl_path = h1_train_args.logdir / "experiment.pkl"
        with open(pkl_path, 'wb') as f:
            pickle.dump(h1_train_args, f)

        # Load model and evaluate
        actor_path = h1_train_args.logdir / "actor.pt"
        if not actor_path.exists():
            actor_path = h1_train_args.logdir / "actor_0.pt"

        loaded_policy = torch.load(actor_path, weights_only=False)
        loaded_policy.eval()

        # Run evaluation rollout
        env = h1_env_fn()
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

    @pytest.mark.timeout(180)
    def test_train_save_load_evaluate_jvrc_walk(self, jvrc_walk_env_fn, jvrc_walk_train_args):
        """Test full pipeline: train, save, load, evaluate for JVRC walk."""
        from rl.algos.ppo import PPO

        # Train for 1 iteration
        jvrc_walk_train_args.n_itr = 1
        jvrc_walk_train_args.eval_freq = 1
        ppo = PPO(jvrc_walk_env_fn, jvrc_walk_train_args)
        ppo.train(jvrc_walk_env_fn, n_itr=1)

        # Save experiment args
        pkl_path = jvrc_walk_train_args.logdir / "experiment.pkl"
        with open(pkl_path, 'wb') as f:
            pickle.dump(jvrc_walk_train_args, f)

        # Load model and evaluate
        actor_path = jvrc_walk_train_args.logdir / "actor.pt"
        if not actor_path.exists():
            actor_path = jvrc_walk_train_args.logdir / "actor_0.pt"

        loaded_policy = torch.load(actor_path, weights_only=False)
        loaded_policy.eval()

        # Run evaluation rollout
        env = jvrc_walk_env_fn()
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

    @pytest.mark.timeout(180)
    def test_train_save_load_evaluate_jvrc_step(self, jvrc_step_env_fn, jvrc_step_train_args):
        """Test full pipeline: train, save, load, evaluate for JVRC step."""
        from rl.algos.ppo import PPO

        # Train for 1 iteration
        jvrc_step_train_args.n_itr = 1
        jvrc_step_train_args.eval_freq = 1
        ppo = PPO(jvrc_step_env_fn, jvrc_step_train_args)
        ppo.train(jvrc_step_env_fn, n_itr=1)

        # Save experiment args
        pkl_path = jvrc_step_train_args.logdir / "experiment.pkl"
        with open(pkl_path, 'wb') as f:
            pickle.dump(jvrc_step_train_args, f)

        # Load model and evaluate
        actor_path = jvrc_step_train_args.logdir / "actor.pt"
        if not actor_path.exists():
            actor_path = jvrc_step_train_args.logdir / "actor_0.pt"

        loaded_policy = torch.load(actor_path, weights_only=False)
        loaded_policy.eval()

        # Run evaluation rollout
        env = jvrc_step_env_fn()
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

    def test_policy_normalizes_observations(self, h1_env_fn, h1_train_args):
        """Test policy applies observation normalization."""
        from rl.algos.ppo import PPO

        ppo = PPO(h1_env_fn, h1_train_args)

        # Policy should have normalization stats
        assert hasattr(ppo.policy, 'obs_mean')
        assert hasattr(ppo.policy, 'obs_std')

        # Normalized input should give different output than raw input
        obs = torch.randn(35) * 10  # Large values

        # The policy internally normalizes
        with torch.no_grad():
            action = ppo.policy(obs, deterministic=True)

        assert torch.all(torch.isfinite(action))

    def test_critic_normalizes_observations(self, h1_env_fn, h1_train_args):
        """Test critic applies observation normalization."""
        from rl.algos.ppo import PPO

        ppo = PPO(h1_env_fn, h1_train_args)

        # Critic should have normalization stats
        assert hasattr(ppo.critic, 'obs_mean')
        assert hasattr(ppo.critic, 'obs_std')

        obs = torch.randn(35) * 10

        with torch.no_grad():
            value = ppo.critic(obs)

        assert torch.all(torch.isfinite(value))


class TestContinuedTraining:
    """Tests for continuing training from saved weights."""

    @pytest.mark.slow
    @pytest.mark.timeout(240)
    def test_continued_training_h1(self, h1_env_fn, h1_train_args, temp_logdir):
        """Test continuing training from saved weights for H1."""
        from rl.algos.ppo import PPO

        # First training run
        h1_train_args.n_itr = 1
        h1_train_args.eval_freq = 1
        ppo1 = PPO(h1_env_fn, h1_train_args)
        ppo1.train(h1_env_fn, n_itr=1)

        # Get path to saved actor
        actor_path = h1_train_args.logdir / "actor_0.pt"
        assert actor_path.exists()

        # Create new args for continued training
        from argparse import Namespace
        continued_args = Namespace(**vars(h1_train_args))
        continued_args.logdir = temp_logdir / "continued"
        Path.mkdir(continued_args.logdir, parents=True, exist_ok=True)
        continued_args.continued = actor_path
        continued_args.n_itr = 1
        continued_args.eval_freq = 1

        # Continue training
        ppo2 = PPO(h1_env_fn, continued_args)
        ppo2.train(h1_env_fn, n_itr=1)

        # Verify new weights were saved
        assert (continued_args.logdir / "actor_0.pt").exists()
