"""Tests for the env-owned imitation-projector contract.

Covers:
- PPO loads ``env.imitation_projector()`` when ``--imitate`` is set.
- ``update_actor_critic`` produces a finite imitation loss when the projector's
  mask is non-empty, and zero when the mask is empty.
- PPO raises a clear error when ``--imitate`` is set against an env that does
  not implement ``imitation_projector()``.
"""

from argparse import Namespace
from functools import partial

import pytest
import torch
import torch.optim as optim

from envs import ENVIRONMENTS
from rl.algos.imitation import ImitationQuery
from rl.algos.ppo import PPO
from rl.policies.actor import Gaussian_FF_Actor

CARTPOLE_OBS_DIM = 5
CARTPOLE_ACT_DIM = 1


class _StubProjector:
    """Minimal projector for tests: slice obs[:, :expert_obs_dim], mask sin(obs[:,0])>0."""

    def __init__(self, expert_obs_dim: int, action_indices: list[int], mask_thresh: float = 0.0):
        self._expert_obs_dim = expert_obs_dim
        self._action_indices = torch.tensor(action_indices, dtype=torch.long)
        self._mask_thresh = mask_thresh

    def __call__(self, obs_batch: torch.Tensor) -> ImitationQuery:
        mask = obs_batch[:, 0] > self._mask_thresh
        sliced = obs_batch[:, : self._expert_obs_dim]
        return ImitationQuery(
            expert_obs=sliced[mask],
            sample_mask=mask,
            action_indices=self._action_indices.to(obs_batch.device),
        )


class _AlwaysEmptyProjector:
    """Projector whose mask is always all-False; exercises the fast-path branch."""

    def __init__(self, expert_obs_dim: int, action_indices: list[int]):
        self._expert_obs_dim = expert_obs_dim
        self._action_indices = torch.tensor(action_indices, dtype=torch.long)

    def __call__(self, obs_batch: torch.Tensor) -> ImitationQuery:
        mask = torch.zeros(obs_batch.shape[0], dtype=torch.bool, device=obs_batch.device)
        return ImitationQuery(
            expert_obs=obs_batch[:0, : self._expert_obs_dim],
            sample_mask=mask,
            action_indices=self._action_indices.to(obs_batch.device),
        )


def _save_dummy_expert(path, expert_obs_dim: int, expert_action_dim: int) -> None:
    expert = Gaussian_FF_Actor(
        state_dim=expert_obs_dim,
        action_dim=expert_action_dim,
        layers=(16,),
        init_std=0.2,
        learn_std=False,
    )
    torch.save(expert, path)


def _train_args(logdir, imitate_path=None) -> Namespace:
    return Namespace(
        env="cartpole",
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
        imitate=imitate_path,
        imitate_coeff=0.3,
        yaml=None,
        input_norm_steps=100,
        device="cpu",
    )


def _cartpole_factory():
    env_cls, _ = ENVIRONMENTS["cartpole"]
    return partial(env_cls, path_to_yaml=None)


def _attach_projector(env_cls, projector):
    """Patch a class-level ``imitation_projector()`` returning ``projector``."""
    env_cls.imitation_projector = lambda self, _p=projector: _p


def _detach_projector(env_cls):
    if "imitation_projector" in env_cls.__dict__:
        delattr(env_cls, "imitation_projector")


def _synthetic_batch(ppo, batch_size=8):
    obs_dim = ppo.policy.state_dim
    action_dim = ppo.policy.action_dim
    obs = torch.randn(batch_size, obs_dim)
    actions = torch.randn(batch_size, action_dim)
    returns = torch.randn(batch_size, 1)
    advantages = torch.randn(batch_size, 1)
    return obs, actions, returns, advantages


def _set_optimizers(ppo):
    ppo.actor_optimizer = optim.Adam(ppo.policy.parameters(), lr=ppo.lr, eps=ppo.eps)
    ppo.critic_optimizer = optim.Adam(ppo.critic.parameters(), lr=ppo.lr, eps=ppo.eps)


def test_projector_drives_imitation_loss(tmp_path):
    """Projector with a non-empty mask produces a finite, non-zero imitation loss."""
    expert_path = tmp_path / "expert.pt"
    _save_dummy_expert(expert_path, expert_obs_dim=3, expert_action_dim=CARTPOLE_ACT_DIM)

    env_cls, _ = ENVIRONMENTS["cartpole"]
    projector = _StubProjector(expert_obs_dim=3, action_indices=[0])
    _attach_projector(env_cls, projector)
    try:
        args = _train_args(tmp_path, imitate_path=str(expert_path))
        ppo = PPO(_cartpole_factory(), args)

        assert ppo.imitation_projector is projector
        assert ppo.base_policy is not None

        _set_optimizers(ppo)

        # Force a deterministic mask hit by writing a positive value into obs[:, 0]
        obs, actions, returns, advantages = _synthetic_batch(ppo)
        obs[:, 0] = 1.0  # all samples pass the stub's mask

        scalars = ppo.update_actor_critic(obs, actions, returns, advantages, mask=1)
        imitation_loss = scalars[5]

        assert torch.isfinite(imitation_loss)
        assert imitation_loss.item() > 0.0
    finally:
        _detach_projector(env_cls)


def test_empty_mask_yields_zero_loss(tmp_path):
    """Projector that masks out every sample short-circuits to zero loss."""
    expert_path = tmp_path / "expert.pt"
    _save_dummy_expert(expert_path, expert_obs_dim=3, expert_action_dim=CARTPOLE_ACT_DIM)

    env_cls, _ = ENVIRONMENTS["cartpole"]
    _attach_projector(env_cls, _AlwaysEmptyProjector(expert_obs_dim=3, action_indices=[0]))
    try:
        args = _train_args(tmp_path, imitate_path=str(expert_path))
        ppo = PPO(_cartpole_factory(), args)
        _set_optimizers(ppo)

        obs, actions, returns, advantages = _synthetic_batch(ppo)
        scalars = ppo.update_actor_critic(obs, actions, returns, advantages, mask=1)
        imitation_loss = scalars[5]

        assert torch.isfinite(imitation_loss)
        assert imitation_loss.item() == 0.0
    finally:
        _detach_projector(env_cls)


def test_missing_projector_raises(tmp_path):
    """--imitate against an env without imitation_projector() raises a clear error."""
    expert_path = tmp_path / "expert.pt"
    _save_dummy_expert(expert_path, expert_obs_dim=3, expert_action_dim=CARTPOLE_ACT_DIM)

    env_cls, _ = ENVIRONMENTS["cartpole"]
    _detach_projector(env_cls)  # ensure no leftover patch from another test

    args = _train_args(tmp_path, imitate_path=str(expert_path))
    with pytest.raises(ValueError, match="imitation_projector"):
        PPO(_cartpole_factory(), args)
