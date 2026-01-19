import torch
import torch.nn as nn

from rl.policies.base import Net


class Critic(Net):
    def __init__(self):
        super().__init__()

    def forward(self, state):
        raise NotImplementedError


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
        self.normc_init = normc_init

        self.init_parameters()

    def forward(self, state):
        state = (state - self.obs_mean) / self.obs_std

        x = state
        for layer in self.critic_layers:
            x = self.nonlinearity(layer(x))
        value = self.network_out(x)

        return value


class LSTM_V(Critic):
    def __init__(self, input_dim, layers=(256, 256), normc_init=True):
        super().__init__()

        self.critic_layers = nn.ModuleList()
        self.critic_layers += [nn.LSTMCell(input_dim, layers[0])]
        for i in range(len(layers) - 1):
            self.critic_layers += [nn.LSTMCell(layers[i], layers[i + 1])]
        self.network_out = nn.Linear(layers[-1], 1)

        self.init_hidden_state()

        self.normc_init = normc_init
        self.init_parameters()

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
