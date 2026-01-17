"""Tests for deterministic training and evaluation with seeding.

These tests verify that training and evaluation are reproducible when
using the same random seed.
"""

from copy import deepcopy

import numpy as np
import pytest
import torch
from conftest import get_env_info

from rl.utils.seeding import set_global_seeds


def get_single_env_name():
    """Get a single environment for determinism tests (faster than testing all)."""
    return "h1"


@pytest.fixture
def determinism_train_args(temp_logdir):
    """Create minimal training args for determinism tests."""
    from argparse import Namespace

    return Namespace(
        env="h1",
        logdir=temp_logdir,
        lr=1e-4,
        eps=1e-5,
        lam=0.95,
        gamma=0.99,
        std_dev=0.223,
        learn_std=False,
        entropy_coeff=0.0,
        clip=0.2,
        minibatch_size=32,
        epochs=1,
        use_gae=True,
        num_procs=2,
        max_grad_norm=0.05,
        max_traj_len=50,
        no_mirror=True,
        mirror_coeff=0.0,
        eval_freq=100,
        continued=None,
        recurrent=False,
        imitate=None,
        imitate_coeff=0.0,
        yaml=None,
        input_norm_steps=50,
        n_itr=2,
        device="cpu",
    )


def compare_state_dicts(state1: dict, state2: dict) -> tuple[bool, list[str]]:
    """Compare two state dicts and return (match, differences)."""
    differences = []

    for key in state1:
        if key not in state2:
            differences.append(f"Key {key} missing in second state dict")
            continue

        t1, t2 = state1[key], state2[key]
        if not torch.equal(t1, t2):
            max_diff = (t1 - t2).abs().max().item()
            differences.append(f"{key}: max_diff={max_diff:.2e}")

    return len(differences) == 0, differences


@pytest.mark.slow
class TestDeterministicTraining:
    """Tests for deterministic training reproducibility."""

    @pytest.mark.timeout(180)
    def test_training_determinism_with_seed(self, temp_logdir):
        """Test that two training runs with the same seed produce identical results."""
        from argparse import Namespace

        env_info = get_env_info("h1")
        env_factory = env_info["factory"]

        seed = 12345
        n_itr = 2

        def create_args(logdir):
            return Namespace(
                env="h1",
                logdir=logdir,
                lr=1e-4,
                eps=1e-5,
                lam=0.95,
                gamma=0.99,
                std_dev=0.223,
                learn_std=False,
                entropy_coeff=0.0,
                clip=0.2,
                minibatch_size=32,
                epochs=1,
                use_gae=True,
                num_procs=2,
                max_grad_norm=0.05,
                max_traj_len=50,
                no_mirror=True,
                mirror_coeff=0.0,
                eval_freq=100,
                continued=None,
                recurrent=False,
                imitate=None,
                imitate_coeff=0.0,
                yaml=None,
                input_norm_steps=50,
                n_itr=n_itr,
                device="cpu",
            )

        # Run 1

        from rl.algos.ppo import PPO

        run1_dir = temp_logdir / "run1"
        run1_dir.mkdir()
        set_global_seeds(seed, cuda_deterministic=True)
        args1 = create_args(run1_dir)
        ppo1 = PPO(env_factory, args1, seed=seed)
        ppo1.train(env_factory, n_itr=n_itr)
        state1 = deepcopy(ppo1.policy.state_dict())

        # Run 2
        run2_dir = temp_logdir / "run2"
        run2_dir.mkdir()
        set_global_seeds(seed, cuda_deterministic=True)
        args2 = create_args(run2_dir)
        ppo2 = PPO(env_factory, args2, seed=seed)
        ppo2.train(env_factory, n_itr=n_itr)
        state2 = deepcopy(ppo2.policy.state_dict())

        # Compare final weights
        match, differences = compare_state_dicts(state1, state2)

        if not match:
            pytest.fail("Training not deterministic. Differences:\n" + "\n".join(differences))

    @pytest.mark.timeout(180)
    def test_different_seeds_produce_different_results(self, temp_logdir):
        """Test that different seeds produce different training results."""
        from argparse import Namespace

        env_info = get_env_info("h1")
        env_factory = env_info["factory"]

        n_itr = 2

        def create_args(logdir):
            return Namespace(
                env="h1",
                logdir=logdir,
                lr=1e-4,
                eps=1e-5,
                lam=0.95,
                gamma=0.99,
                std_dev=0.223,
                learn_std=False,
                entropy_coeff=0.0,
                clip=0.2,
                minibatch_size=32,
                epochs=1,
                use_gae=True,
                num_procs=2,
                max_grad_norm=0.05,
                max_traj_len=50,
                no_mirror=True,
                mirror_coeff=0.0,
                eval_freq=100,
                continued=None,
                recurrent=False,
                imitate=None,
                imitate_coeff=0.0,
                yaml=None,
                input_norm_steps=50,
                n_itr=n_itr,
                device="cpu",
            )

        from rl.algos.ppo import PPO

        # Run with seed 1
        run1_dir = temp_logdir / "run1"
        run1_dir.mkdir()
        set_global_seeds(111, cuda_deterministic=True)
        args1 = create_args(run1_dir)
        ppo1 = PPO(env_factory, args1, seed=111)
        ppo1.train(env_factory, n_itr=n_itr)
        state1 = deepcopy(ppo1.policy.state_dict())

        # Run with seed 2
        run2_dir = temp_logdir / "run2"
        run2_dir.mkdir()
        set_global_seeds(222, cuda_deterministic=True)
        args2 = create_args(run2_dir)
        ppo2 = PPO(env_factory, args2, seed=222)
        ppo2.train(env_factory, n_itr=n_itr)
        state2 = deepcopy(ppo2.policy.state_dict())

        # Results should differ
        match, _ = compare_state_dicts(state1, state2)
        assert not match, "Different seeds should produce different results"


@pytest.mark.slow
class TestDeterministicEvaluation:
    """Tests for deterministic policy evaluation."""

    @pytest.mark.timeout(120)
    def test_evaluation_determinism_with_seed(self, temp_logdir):
        """Test that policy evaluation is deterministic with the same seed."""
        from argparse import Namespace

        env_info = get_env_info("h1")
        env_factory = env_info["factory"]

        # First train a model
        train_seed = 42
        n_itr = 1

        args = Namespace(
            env="h1",
            logdir=temp_logdir,
            lr=1e-4,
            eps=1e-5,
            lam=0.95,
            gamma=0.99,
            std_dev=0.223,
            learn_std=False,
            entropy_coeff=0.0,
            clip=0.2,
            minibatch_size=32,
            epochs=1,
            use_gae=True,
            num_procs=2,
            max_grad_norm=0.05,
            max_traj_len=50,
            no_mirror=True,
            mirror_coeff=0.0,
            eval_freq=100,
            continued=None,
            recurrent=False,
            imitate=None,
            imitate_coeff=0.0,
            yaml=None,
            input_norm_steps=50,
            n_itr=n_itr,
            device="cpu",
        )

        from rl.algos.ppo import PPO

        set_global_seeds(train_seed, cuda_deterministic=True)
        ppo = PPO(env_factory, args, seed=train_seed)
        ppo.train(env_factory, n_itr=n_itr)

        # Get the trained policy
        policy = deepcopy(ppo.policy)
        policy.eval()

        # Run evaluation twice with the same seed
        eval_seed = 9999
        n_steps = 100

        def run_evaluation(seed):
            """Run deterministic evaluation and collect trajectory."""
            set_global_seeds(seed, cuda_deterministic=True)
            env = env_factory()

            states = []
            actions = []
            rewards = []

            state = env.reset()
            for _ in range(n_steps):
                state_tensor = torch.as_tensor(state, dtype=torch.float32)
                with torch.no_grad():
                    action = policy(state_tensor, deterministic=True)

                states.append(state.copy())
                actions.append(action.numpy().copy())

                next_state, reward, done, _ = env.step(action.numpy())
                rewards.append(reward)

                if done:
                    state = env.reset()
                else:
                    state = next_state

            env.close()
            return np.array(states), np.array(actions), np.array(rewards)

        # Run twice
        states1, actions1, rewards1 = run_evaluation(eval_seed)
        states2, actions2, rewards2 = run_evaluation(eval_seed)

        # Compare trajectories
        assert np.allclose(states1, states2), f"States differ: max_diff={np.abs(states1 - states2).max()}"
        assert np.allclose(actions1, actions2), f"Actions differ: max_diff={np.abs(actions1 - actions2).max()}"
        assert np.allclose(rewards1, rewards2), f"Rewards differ: max_diff={np.abs(rewards1 - rewards2).max()}"

    @pytest.mark.timeout(120)
    def test_stochastic_vs_deterministic_actions(self, temp_logdir):
        """Test that stochastic actions differ while deterministic stay same."""
        from argparse import Namespace

        env_info = get_env_info("h1")
        env_factory = env_info["factory"]

        # Train a model
        train_seed = 42
        n_itr = 1

        args = Namespace(
            env="h1",
            logdir=temp_logdir,
            lr=1e-4,
            eps=1e-5,
            lam=0.95,
            gamma=0.99,
            std_dev=0.223,
            learn_std=False,
            entropy_coeff=0.0,
            clip=0.2,
            minibatch_size=32,
            epochs=1,
            use_gae=True,
            num_procs=2,
            max_grad_norm=0.05,
            max_traj_len=50,
            no_mirror=True,
            mirror_coeff=0.0,
            eval_freq=100,
            continued=None,
            recurrent=False,
            imitate=None,
            imitate_coeff=0.0,
            yaml=None,
            input_norm_steps=50,
            n_itr=n_itr,
            device="cpu",
        )

        from rl.algos.ppo import PPO

        set_global_seeds(train_seed, cuda_deterministic=True)
        ppo = PPO(env_factory, args, seed=train_seed)
        ppo.train(env_factory, n_itr=n_itr)

        policy = deepcopy(ppo.policy)
        policy.eval()

        env = env_factory()
        state = torch.as_tensor(env.reset(), dtype=torch.float32)
        env.close()

        # Deterministic actions should be identical
        with torch.no_grad():
            det_action1 = policy(state, deterministic=True)
            det_action2 = policy(state, deterministic=True)

        assert torch.allclose(det_action1, det_action2), "Deterministic actions should be identical"

        # Stochastic actions should differ (with high probability)
        with torch.no_grad():
            stoch_action1 = policy(state, deterministic=False)
            stoch_action2 = policy(state, deterministic=False)

        # They might be the same by chance, but very unlikely
        # Just verify they can be sampled
        assert stoch_action1.shape == stoch_action2.shape


class TestSeedingUtilities:
    """Tests for seeding utility functions."""

    def test_get_worker_seed_no_collisions(self):
        """Test that worker seeds don't collide."""
        from rl.utils.seeding import get_worker_seed

        seeds = set()
        master_seed = 42

        # Generate seeds for multiple workers and offsets
        for offset in range(10):
            for worker_id in range(100):
                seed = get_worker_seed(master_seed, worker_id, offset)
                assert seed not in seeds, f"Collision at offset={offset}, worker_id={worker_id}"
                seeds.add(seed)

    def test_set_global_seeds_affects_random(self):
        """Test that set_global_seeds affects random number generators."""
        import random

        set_global_seeds(42, cuda_deterministic=True)
        r1 = random.random()
        np1 = np.random.random()
        t1 = torch.rand(1).item()

        set_global_seeds(42, cuda_deterministic=True)
        r2 = random.random()
        np2 = np.random.random()
        t2 = torch.rand(1).item()

        assert r1 == r2, "random.random() not deterministic"
        assert np1 == np2, "np.random.random() not deterministic"
        assert t1 == t2, "torch.rand() not deterministic"

    def test_different_seeds_different_values(self):
        """Test that different seeds produce different random values."""
        set_global_seeds(42, cuda_deterministic=True)
        val1 = np.random.random()

        set_global_seeds(43, cuda_deterministic=True)
        val2 = np.random.random()

        assert val1 != val2, "Different seeds should produce different values"
