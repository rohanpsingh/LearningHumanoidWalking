import torch

class PPOBuffer:
    def __init__(self, obs_len=1, act_len=1, gamma=0.99, lam=0.95, use_gae=False, size=1):
        self.states  = torch.zeros(size, obs_len, dtype=float)
        self.actions = torch.zeros(size, act_len, dtype=float)
        self.rewards = torch.zeros(size, 1, dtype=float)
        self.values  = torch.zeros(size, 1, dtype=float)
        self.returns = torch.zeros(size, 1, dtype=float)
        self.dones   = torch.zeros(size, 1, dtype=float)

        self.gamma, self.lam = gamma, lam
        self.ptr = 0
        self.traj_idx = [0]

    def __len__(self):
        return self.ptr

    def store(self, state, action, reward, value, done):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        self.states[self.ptr]= state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value
        self.dones[self.ptr] = done
        self.ptr += 1

    def finish_path(self, last_val=None):
        self.traj_idx += [self.ptr]
        rewards = self.rewards[self.traj_idx[-2]:self.traj_idx[-1], 0]
        R = last_val.squeeze(0)
        returns = torch.zeros_like(rewards)
        for i in range(len(rewards) - 1, -1, -1):
            R = self.gamma * R + rewards[i]
            returns[i] = R
        self.returns[self.traj_idx[-2]:self.traj_idx[-1], 0] = returns
        self.dones[-1] = True

    def get_data(self):
        """
        Return collected data and reset buffer.

        Returns:
            dict: Collected trajectory data
        """
        ep_lens = [j - i for i, j in zip(self.traj_idx, self.traj_idx[1:])]
        ep_rewards = [
            float(sum(self.rewards[int(i):int(j)])) for i, j in zip(self.traj_idx, self.traj_idx[1:])
        ]
        data = {
            'states': self.states[:self.ptr],
            'actions': self.actions[:self.ptr],
            'rewards': self.rewards[:self.ptr],
            'values': self.values[:self.ptr],
            'returns': self.returns[:self.ptr],
            'dones': self.dones[:self.ptr],
            'traj_idx': torch.Tensor(self.traj_idx),
            'ep_lens': torch.Tensor(ep_lens),
            'ep_rewards': torch.Tensor(ep_rewards),
        }
        return data
