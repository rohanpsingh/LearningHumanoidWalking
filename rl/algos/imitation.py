"""Contract for env-owned imitation-loss adapters.

PPO needs to ask "for this batch of policy obs, what should I feed an expert and
what should I compare its output against?" without knowing anything about the
env's obs/action layout. The env answers via an :class:`ImitationAdapter` it
returns from ``env.imitation_adapter()``.
"""

from dataclasses import dataclass
from typing import Protocol

import torch


@dataclass
class ImitationQuery:
    """One batched question to the expert.

    Attributes:
        expert_obs: ``(N_active, expert_obs_dim)`` tensor — already filtered to the
            samples that should contribute to the loss. Callers pass this straight
            into ``base_policy(...)``.
        sample_mask: ``(B,)`` bool tensor — which entries of the original policy
            obs batch fed the expert. Used to select the matching student actions.
        action_indices: ``(k,)`` long tensor — which student-action dims to
            compare against the expert's output.
    """

    expert_obs: torch.Tensor
    sample_mask: torch.Tensor
    action_indices: torch.Tensor


class ImitationAdapter(Protocol):
    """Translator from policy obs to an expert query.

    Implementations are stateless w.r.t. env time: they capture obs-layout
    constants at construction and operate purely on the obs batch tensor.
    """

    def __call__(self, obs_batch: torch.Tensor) -> ImitationQuery: ...
