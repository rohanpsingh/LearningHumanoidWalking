"""Shared fixtures for learninghumanoidwalking tests.

This module provides dynamic environment discovery and parametrized fixtures
that automatically test all environments under envs/.
"""

import os
import shutil
import sys
import tempfile
from argparse import Namespace
from functools import partial
from pathlib import Path

import pytest
import ray

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def discover_environments():
    """
    Discover all environment classes from envs.ENVIRONMENTS registry.

    Returns a dict mapping env_name to environment info dict.
    New environments should be registered in envs/__init__.py.
    """
    from envs import ENVIRONMENTS

    environments = {}
    for env_name, (env_class, robot_name) in ENVIRONMENTS.items():
        env_fn = partial(env_class, path_to_yaml=None)
        environments[env_name] = {
            "class": env_class,
            "factory": env_fn,
            "name": env_class.__name__,
            "robot": robot_name,
        }

    return environments


# Discover all environments at module load time
DISCOVERED_ENVIRONMENTS = discover_environments()

# Create list of env names for parametrization
ENV_NAMES = list(DISCOVERED_ENVIRONMENTS.keys())


def get_env_info(env_name):
    """Get environment info by name."""
    return DISCOVERED_ENVIRONMENTS[env_name]


@pytest.fixture(scope="session", autouse=True)
def setup_ray():
    """Initialize Ray once for the entire test session."""
    # Unset RAY_ADDRESS to ensure local mode
    os.environ.pop("RAY_ADDRESS", None)
    if not ray.is_initialized():
        ray.init(num_cpus=2, ignore_reinit_error=True, local_mode=False)
    yield


@pytest.fixture
def temp_logdir():
    """Create a temporary directory for logs/weights."""
    tmpdir = tempfile.mkdtemp(prefix="test_logs_")
    yield Path(tmpdir)
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture(params=ENV_NAMES)
def env_name(request):
    """Parametrized fixture that yields each environment name."""
    return request.param


@pytest.fixture
def env_info(env_name):
    """Get environment info dict for the current env_name."""
    return get_env_info(env_name)


@pytest.fixture
def env_class(env_info):
    """Get the environment class."""
    return env_info["class"]


@pytest.fixture
def env_factory(env_info):
    """Get the environment factory function."""
    return env_info["factory"]


@pytest.fixture
def env_instance(env_factory):
    """Create an environment instance."""
    env = env_factory()
    yield env
    env.close()


@pytest.fixture
def train_args(temp_logdir, env_info):
    """Create training arguments for the current environment."""
    # Determine env string for run_experiment.py compatibility
    robot = env_info["robot"]
    class_name = env_info["name"]

    # Map to the env names used in run_experiment.py
    if robot == "h1":
        env_str = "h1"
    elif robot == "jvrc":
        if "Step" in class_name:
            env_str = "jvrc_step"
        else:
            env_str = "jvrc_walk"
    else:
        # For new environments, use robot name as fallback
        env_str = robot

    # Check if environment has mirror indices
    test_env = env_info["factory"]()
    has_mirror = hasattr(test_env.robot, "mirrored_obs") and hasattr(test_env.robot, "mirrored_acts")
    test_env.close()

    return Namespace(
        env=env_str,
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
        no_mirror=not has_mirror,
        mirror_coeff=0.4 if has_mirror else 0.0,
        eval_freq=100,
        continued=None,
        recurrent=False,
        imitate=None,
        imitate_coeff=0.0,
        yaml=None,
        input_norm_steps=100,
        device="cpu",  # Force CPU for tests to avoid Ray serialization issues with CUDA tensors
    )


# Helper functions for tests that need multiple environments at once
def get_all_env_instances():
    """Create instances of all discovered environments."""
    envs = []
    for env_name, info in DISCOVERED_ENVIRONMENTS.items():
        env = info["factory"]()
        envs.append((env_name, env))
    return envs


def close_all_envs(envs):
    """Close all environment instances."""
    for _, env in envs:
        env.close()
