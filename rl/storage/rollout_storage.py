from dataclasses import dataclass

import torch


@dataclass
class BatchData:
    """Typed container for batch data from trajectory collection.

    Provides a typed interface instead of anonymous dicts, enabling IDE support,
    type checking, and clearer code documentation.
    """

    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    values: torch.Tensor
    returns: torch.Tensor
    dones: torch.Tensor
    traj_idx: torch.Tensor
    ep_lens: torch.Tensor
    ep_rewards: torch.Tensor


class PPOBuffer:
    def __init__(self, obs_len=1, act_len=1, gamma=0.99, lam=0.95, size=1):
        self.states = torch.zeros(size, obs_len, dtype=float)
        self.actions = torch.zeros(size, act_len, dtype=float)
        self.rewards = torch.zeros(size, 1, dtype=float)
        self.values = torch.zeros(size, 1, dtype=float)
        self.returns = torch.zeros(size, 1, dtype=float)
        self.dones = torch.zeros(size, 1, dtype=float)

        self.gamma = gamma
        self.lam = lam
        self.ptr = 0
        self.traj_idx = [0]

    def __len__(self):
        return self.ptr

    def store(self, state, action, reward, value, done):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value
        self.dones[self.ptr] = done
        self.ptr += 1

    def finish_path(self, last_val=None):
        """Finish a trajectory and compute GAE λ-returns.

        Uses Generalized Advantage Estimation:
            δ_t = r_t + γ*V(s_{t+1}) - V(s_t)
            A_t = δ_t + (γλ)*δ_{t+1} + (γλ)²*δ_{t+2} + ...
            returns_t = A_t + V(s_t)

        Args:
            last_val: Bootstrap value for return computation (0 if truly done)
        """
        self.traj_idx += [self.ptr]
        start = self.traj_idx[-2]
        end = self.traj_idx[-1]
        rewards = self.rewards[start:end, 0]
        values = self.values[start:end, 0]
        T = len(rewards)

        if T == 0:
            return

        last_val_scalar = last_val.squeeze(0) if last_val.dim() > 0 else last_val

        # Compute TD residuals: δ_t = r_t + γ*V(s_{t+1}) - V(s_t)
        # Append bootstrap value for V(s_T)
        next_values = torch.cat([values[1:], last_val_scalar.unsqueeze(0)])
        deltas = rewards + self.gamma * next_values - values

        # Compute GAE advantages via reverse accumulation:
        # A_t = δ_t + (γλ)*A_{t+1}
        gae = torch.zeros_like(rewards)
        running_gae = 0.0
        for t in reversed(range(T)):
            running_gae = deltas[t] + self.gamma * self.lam * running_gae
            gae[t] = running_gae

        # Returns = advantages + values
        self.returns[start:end, 0] = gae + values

    def get_data(self, ep_lens=None, ep_rewards=None) -> BatchData:
        """Return collected data as BatchData.

        Args:
            ep_lens: List of completed episode lengths (from worker)
            ep_rewards: List of completed episode rewards (from worker)

        Returns:
            BatchData with collected trajectory data
        """
        return BatchData(
            states=self.states[: self.ptr],
            actions=self.actions[: self.ptr],
            rewards=self.rewards[: self.ptr],
            values=self.values[: self.ptr],
            returns=self.returns[: self.ptr],
            dones=self.dones[: self.ptr],
            traj_idx=torch.tensor(self.traj_idx),
            ep_lens=torch.tensor(ep_lens if ep_lens else []),
            ep_rewards=torch.tensor(ep_rewards if ep_rewards else []),
        )
