import torch
import torch.nn as nn

from rl.policies.base import Net, normc_fn


# The base class for a critic. Includes functions for normalizing reward and state (optional)
class Critic(Net):
    def __init__(self):
        super().__init__()

        self.welford_reward_mean = 0.0
        self.welford_reward_mean_diff = 1.0
        self.welford_reward_n = 1

    def forward(self):
        raise NotImplementedError

    def normalize_reward(self, r, update=True):
        if update:
            if len(r.size()) == 1:
                r_old = self.welford_reward_mean
                self.welford_reward_mean += (r - r_old) / self.welford_reward_n
                self.welford_reward_mean_diff += (r - r_old) * (r - r_old)
                self.welford_reward_n += 1
            elif len(r.size()) == 2:
                for r_n in r:
                    r_old = self.welford_reward_mean
                    self.welford_reward_mean += (r_n - r_old) / self.welford_reward_n
                    self.welford_reward_mean_diff += (r_n - r_old) * (r_n - r_old)
                    self.welford_reward_n += 1
            else:
                raise NotImplementedError

        return (r - self.welford_reward_mean) / torch.sqrt(self.welford_reward_mean_diff / self.welford_reward_n)


class FF_V(Critic):
    def __init__(
        self,
        state_dim,
        layers=(256, 256),
        nonlinearity=torch.nn.functional.relu,
        normc_init=True,
        obs_std=None,
        obs_mean=None,
    ):
        super().__init__()

        self.critic_layers = nn.ModuleList()
        self.critic_layers += [nn.Linear(state_dim, layers[0])]
        for i in range(len(layers) - 1):
            self.critic_layers += [nn.Linear(layers[i], layers[i + 1])]
        self.network_out = nn.Linear(layers[-1], 1)

        self.nonlinearity = nonlinearity

        self.obs_std = obs_std
        self.obs_mean = obs_mean

        # weight initialization scheme used in PPO paper experiments
        self.normc_init = normc_init

        self.init_parameters()

    def init_parameters(self):
        if self.normc_init:
            print("Doing norm column initialization.")
            self.apply(normc_fn)

    def forward(self, inputs):
        inputs = (inputs - self.obs_mean) / self.obs_std

        x = inputs
        for layer in self.critic_layers:
            x = self.nonlinearity(layer(x))
        value = self.network_out(x)

        return value


class LSTM_V(Critic):
    def __init__(self, input_dim, layers=(128, 128), normc_init=True):
        super().__init__()

        self.critic_layers = nn.ModuleList()
        self.critic_layers += [nn.LSTMCell(input_dim, layers[0])]
        for i in range(len(layers) - 1):
            self.critic_layers += [nn.LSTMCell(layers[i], layers[i + 1])]
        self.network_out = nn.Linear(layers[-1], 1)

        self.init_hidden_state()

        if normc_init:
            self.initialize_parameters()

    def get_hidden_state(self):
        return self.hidden, self.cells

    def _get_device(self):
        """Get device from network parameters."""
        return next(self.parameters()).device

    def init_hidden_state(self, batch_size=1, device=None):
        if device is None:
            device = self._get_device()
        self.hidden = [torch.zeros(batch_size, layer.hidden_size, device=device) for layer in self.critic_layers]
        self.cells = [torch.zeros(batch_size, layer.hidden_size, device=device) for layer in self.critic_layers]

    def forward(self, state):
        state = (state - self.obs_mean) / self.obs_std
        dims = len(state.size())

        if dims == 3:  # if we get a batch of trajectories
            self.init_hidden_state(batch_size=state.size(1), device=state.device)
            value = []
            for _t, state_batch_t in enumerate(state):
                x_t = state_batch_t
                for idx, layer in enumerate(self.critic_layers):
                    c, h = self.cells[idx], self.hidden[idx]
                    self.hidden[idx], self.cells[idx] = layer(x_t, (h, c))
                    x_t = self.hidden[idx]
                x_t = self.network_out(x_t)
                value.append(x_t)

            x = torch.stack([a.float() for a in value])

        else:
            x = state
            if dims == 1:
                x = x.view(1, -1)

            for idx, layer in enumerate(self.critic_layers):
                c, h = self.cells[idx], self.hidden[idx]
                self.hidden[idx], self.cells[idx] = layer(x, (h, c))
                x = self.hidden[idx]
            x = self.network_out(x)

            if dims == 1:
                x = x.view(-1)

        return x


GaussianMLP_Critic = FF_V
