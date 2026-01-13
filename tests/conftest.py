"""Shared fixtures for learninghumanoidwalking tests."""
import pytest
import sys
import os
import shutil
import tempfile
from pathlib import Path
from argparse import Namespace
from functools import partial

import numpy as np
import torch
import ray

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(scope="session", autouse=True)
def setup_ray():
    """Initialize Ray once for the entire test session."""
    # Unset RAY_ADDRESS to ensure local mode
    os.environ.pop("RAY_ADDRESS", None)
    if not ray.is_initialized():
        ray.init(num_cpus=2, ignore_reinit_error=True, local_mode=False)
    yield
    # Don't shutdown Ray as other tests may need it


@pytest.fixture
def h1_env():
    """Create H1 environment instance."""
    from envs.h1 import H1Env
    env = H1Env()
    yield env
    env.close()


@pytest.fixture
def jvrc_walk_env():
    """Create JVRC walking environment instance."""
    from envs.jvrc import JvrcWalkEnv
    env = JvrcWalkEnv()
    yield env
    env.close()


@pytest.fixture
def jvrc_step_env():
    """Create JVRC stepping environment instance."""
    from envs.jvrc import JvrcStepEnv
    env = JvrcStepEnv()
    yield env
    env.close()


@pytest.fixture
def temp_logdir():
    """Create a temporary directory for logs/weights."""
    tmpdir = tempfile.mkdtemp(prefix="test_logs_")
    yield Path(tmpdir)
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def base_train_args(temp_logdir):
    """Create base training arguments with minimal settings for fast tests."""
    return Namespace(
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
        max_traj_len=50,  # Short trajectory for fast tests
        no_mirror=True,   # Disable mirror for faster tests
        mirror_coeff=0.0,
        eval_freq=100,
        continued=None,
        recurrent=False,
        imitate=None,
        imitate_coeff=0.0,
        yaml=None,
        input_norm_steps=100,  # Minimal normalization steps
    )


@pytest.fixture
def h1_train_args(base_train_args):
    """Training arguments for H1 environment."""
    base_train_args.env = "h1"
    base_train_args.no_mirror = True  # H1 doesn't have mirror indices
    return base_train_args


@pytest.fixture
def jvrc_walk_train_args(base_train_args):
    """Training arguments for JVRC walking environment."""
    base_train_args.env = "jvrc_walk"
    return base_train_args


@pytest.fixture
def jvrc_step_train_args(base_train_args):
    """Training arguments for JVRC stepping environment."""
    base_train_args.env = "jvrc_step"
    return base_train_args


@pytest.fixture
def h1_env_fn():
    """Factory function for H1 environment."""
    from envs.h1 import H1Env
    return partial(H1Env, path_to_yaml=None)


@pytest.fixture
def jvrc_walk_env_fn():
    """Factory function for JVRC walking environment."""
    from envs.jvrc import JvrcWalkEnv
    return partial(JvrcWalkEnv, path_to_yaml=None)


@pytest.fixture
def jvrc_step_env_fn():
    """Factory function for JVRC stepping environment."""
    from envs.jvrc import JvrcStepEnv
    return partial(JvrcStepEnv, path_to_yaml=None)
