import torch
import torch.nn as nn
import torch.nn.functional as F

from rl.policies.base import Net


class Actor(Net):
    def __init__(self):
        super().__init__()

    def forward(self, state, deterministic=True):
        raise NotImplementedError


class Linear_Actor(Actor):
    def __init__(self, state_dim, action_dim, hidden_size=32):
        super().__init__()

        self.l1 = nn.Linear(state_dim, hidden_size)
        self.l2 = nn.Linear(hidden_size, action_dim)

        self.action_dim = action_dim

        for p in self.parameters():
            p.data = torch.zeros(p.shape)

    def forward(self, state):
        a = self.l1(state)
        a = self.l2(a)
        return a


class FF_Actor(Actor):
    def __init__(self, state_dim, action_dim, layers=(256, 256), nonlinearity=F.relu):
        super().__init__()

        self.actor_layers = nn.ModuleList()
        self.actor_layers += [nn.Linear(state_dim, layers[0])]
        for i in range(len(layers) - 1):
            self.actor_layers += [nn.Linear(layers[i], layers[i + 1])]
        self.network_out = nn.Linear(layers[-1], action_dim)

        self.action_dim = action_dim
        self.nonlinearity = nonlinearity

        self.init_parameters()

    def forward(self, state, deterministic=True):
        x = state
        for _idx, layer in enumerate(self.actor_layers):
            x = self.nonlinearity(layer(x))

        action = torch.tanh(self.network_out(x))
        return action


class LSTM_Actor(Actor):
    def __init__(self, state_dim, action_dim, layers=(128, 128), nonlinearity=torch.tanh):
        super().__init__()

        self.actor_layers = nn.ModuleList()
        self.actor_layers += [nn.LSTMCell(state_dim, layers[0])]
        for i in range(len(layers) - 1):
            self.actor_layers += [nn.LSTMCell(layers[i], layers[i + 1])]
        self.network_out = nn.Linear(layers[-1], action_dim)

        self.action_dim = action_dim
        self.init_hidden_state()
        self.nonlinearity = nonlinearity

    def get_hidden_state(self):
        return self.hidden, self.cells

    def set_hidden_state(self, data):
        if len(data) != 2:
            print("Got invalid hidden state data.")
            exit(1)

        self.hidden, self.cells = data

    def _get_device(self):
        """Get device from network parameters."""
        return next(self.parameters()).device

    def init_hidden_state(self, batch_size=1, device=None):
        if device is None:
            device = self._get_device()
        self.hidden = [torch.zeros(batch_size, layer.hidden_size, device=device) for layer in self.actor_layers]
        self.cells = [torch.zeros(batch_size, layer.hidden_size, device=device) for layer in self.actor_layers]

    def forward(self, x, deterministic=True):
        dims = len(x.size())

        if dims == 3:  # if we get a batch of trajectories
            self.init_hidden_state(batch_size=x.size(1), device=x.device)
            y = []
            for _t, x_t in enumerate(x):
                for idx, layer in enumerate(self.actor_layers):
                    c, h = self.cells[idx], self.hidden[idx]
                    self.hidden[idx], self.cells[idx] = layer(x_t, (h, c))
                    x_t = self.hidden[idx]
                y.append(x_t)
            x = torch.stack([x_t for x_t in y])

        else:
            if dims == 1:  # if we get a single timestep (if not, assume we got a batch of single timesteps)
                x = x.view(1, -1)

            for idx, layer in enumerate(self.actor_layers):
                h, c = self.hidden[idx], self.cells[idx]
                self.hidden[idx], self.cells[idx] = layer(x, (h, c))
                x = self.hidden[idx]

            if dims == 1:
                x = x.view(-1)

        action = self.nonlinearity(self.network_out(x))
        return action


class Gaussian_FF_Actor(Actor):
    def __init__(
        self,
        state_dim,
        action_dim,
        layers=(256, 256),
        nonlinearity=torch.nn.functional.relu,
        init_std=0.2,
        learn_std=False,
        bounded=False,
        normc_init=True,
    ):
        super().__init__()

        self.actor_layers = nn.ModuleList()
        self.actor_layers += [nn.Linear(state_dim, layers[0])]
        for i in range(len(layers) - 1):
            self.actor_layers += [nn.Linear(layers[i], layers[i + 1])]
        self.means = nn.Linear(layers[-1], action_dim)

        self.learn_std = learn_std
        if self.learn_std:
            self.stds = nn.Parameter(init_std * torch.ones(action_dim))
        else:
            self.stds = init_std * torch.ones(action_dim)

        self.action_dim = action_dim
        self.state_dim = state_dim
        self.nonlinearity = nonlinearity

        self.obs_std = 1.0
        self.obs_mean = 0.0

        self.bounded = bounded

        self.normc_init = normc_init
        self.init_parameters(self.means)

    def _get_dist_params(self, state):
        state = (state - self.obs_mean) / self.obs_std

        x = state
        for layer in self.actor_layers:
            x = self.nonlinearity(layer(x))
        mean = self.means(x)

        if self.bounded:
            mean = torch.tanh(mean)

        sd = torch.zeros_like(mean)
        if hasattr(self, "stds"):
            sd = self.stds
        return mean, sd

    def forward(self, state, deterministic=True):
        mu, sd = self._get_dist_params(state)

        if not deterministic:
            action = torch.distributions.Normal(mu, sd).sample()
        else:
            action = mu

        return action

    def distribution(self, inputs):
        mu, sd = self._get_dist_params(inputs)
        return torch.distributions.Normal(mu, sd)


class Gaussian_LSTM_Actor(Actor):
    def __init__(
        self,
        state_dim,
        action_dim,
        layers=(256, 256),
        nonlinearity=F.tanh,
        normc_init=True,
        init_std=0.2,
        learn_std=False,
        bounded=False,
    ):
        super().__init__()

        self.actor_layers = nn.ModuleList()
        self.actor_layers += [nn.LSTMCell(state_dim, layers[0])]
        for i in range(len(layers) - 1):
            self.actor_layers += [nn.LSTMCell(layers[i], layers[i + 1])]
        self.network_out = nn.Linear(layers[-1], action_dim)

        self.action_dim = action_dim
        self.state_dim = state_dim
        self.init_hidden_state()
        self.nonlinearity = nonlinearity

        self.obs_std = 1.0
        self.obs_mean = 0.0
        self.learn_std = learn_std
        if self.learn_std:
            self.stds = nn.Parameter(init_std * torch.ones(action_dim))
        else:
            self.stds = init_std * torch.ones(action_dim)

        self.bounded = bounded

        self.normc_init = normc_init
        self.init_parameters(self.network_out)

    def _get_device(self):
        """Get device from network parameters."""
        return next(self.parameters()).device

    def _get_dist_params(self, state):
        state = (state - self.obs_mean) / self.obs_std

        dims = len(state.size())

        x = state
        if dims == 3:  # if we get a batch of trajectories
            self.init_hidden_state(batch_size=x.size(1), device=x.device)
            y = []
            for _t, x_t in enumerate(x):
                for idx, layer in enumerate(self.actor_layers):
                    c, h = self.cells[idx], self.hidden[idx]
                    self.hidden[idx], self.cells[idx] = layer(x_t, (h, c))
                    x_t = self.hidden[idx]
                y.append(x_t)
            x = torch.stack([x_t for x_t in y])

        else:
            if dims == 1:  # if we get a single timestep (if not, assume we got a batch of single timesteps)
                x = x.view(1, -1)

            for idx, layer in enumerate(self.actor_layers):
                h, c = self.hidden[idx], self.cells[idx]
                self.hidden[idx], self.cells[idx] = layer(x, (h, c))
                x = self.hidden[idx]

            if dims == 1:
                x = x.view(-1)

        mu = self.network_out(x)
        if self.bounded:
            mu = torch.tanh(mu)
        sd = self.stds
        return mu, sd

    def init_hidden_state(self, batch_size=1, device=None):
        if device is None:
            device = self._get_device()
        self.hidden = [torch.zeros(batch_size, layer.hidden_size, device=device) for layer in self.actor_layers]
        self.cells = [torch.zeros(batch_size, layer.hidden_size, device=device) for layer in self.actor_layers]

    def forward(self, state, deterministic=True):
        mu, sd = self._get_dist_params(state)

        if not deterministic:
            action = torch.distributions.Normal(mu, sd).sample()
        else:
            action = mu

        return action

    def distribution(self, inputs):
        mu, sd = self._get_dist_params(inputs)
        return torch.distributions.Normal(mu, sd)
