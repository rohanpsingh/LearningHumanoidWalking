import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

# TODO: extend these for arbitrary bounds

"""A beta distribution, but where the pdf is scaled to (-1, 1)"""
class BoundedBeta(torch.distributions.Beta):
    def log_prob(self, x):
        return super().log_prob((x + 1) / 2)

class Beta(nn.Module):
    def __init__(self, action_dim):
        super(Beta, self).__init__()

        self.action_dim = action_dim

    def forward(self, alpha_beta):
        alpha = 1 + F.softplus(alpha_beta[:, :self.action_dim])
        beta  = 1 + F.softplus(alpha_beta[:, self.action_dim:])
        return alpha, beta

    def sample(self, x, deterministic):
        if deterministic is False:
            action = self.evaluate(x).sample()
        else:
            # E = alpha / (alpha + beta)
            return self.evaluate(x).mean

        return 2 * action - 1

    def evaluate(self, x):
        alpha, beta = self(x)
        return BoundedBeta(alpha, beta)


# TODO: think of a better name for this
"""Beta distribution parameterized by mean and variance."""
class Beta2(nn.Module):
    def __init__(self, action_dim, init_std=0.25, learn_std=False):
        super(Beta2, self).__init__()

        assert init_std < 0.5, "Beta distribution has a max std dev of 0.5"

        self.action_dim = action_dim

        self.logstd = nn.Parameter(
            torch.ones(1, action_dim) * np.log(init_std),
            requires_grad=learn_std
        )

        self.learn_std = learn_std


    def forward(self, x):
        mean = torch.sigmoid(x) 

        var = self.logstd.exp().pow(2)

        """
        alpha = ((1 - mu) / sigma^2 - 1 / mu) * mu^2
        beta  = alpha * (1 / mu - 1)

        Implemented slightly differently for numerical stability.
        """
        alpha = ((1 - mean) / var) * mean.pow(2) - mean
        beta  = ((1 - mean) / var) * mean - 1 - alpha

        # PROBLEM: if alpha or beta < 1 thats not good

        #assert np.allclose(alpha, ((1 - mean) / var - 1 / mean) * mean.pow(2))
        #assert np.allclose(beta, ((1 - mean) / var - 1 / mean) * mean.pow(2) * (1 / mean - 1))

        #alpha = 1 + F.softplus(alpha)
        #beta  = 1 + F.softplus(beta)

        # print("alpha",alpha)
        # print("beta",beta)

        # #print(alpha / (alpha + beta))
        # print("mu",mean)

        # #print(torch.sqrt(alpha * beta / ((alpha+beta)**2 * (alpha + beta + 1))))
        # print("var", var)

        # import pdb
        # pdb.set_trace()

        return alpha, beta

    def sample(self, x, deterministic):
        if deterministic is False:
            action = self.evaluate(x).sample()
        else:
            # E = alpha / (alpha + beta)
            return self.evaluate(x).mean

        # 2 * a - 1 puts a in (-1, 1)
        return 2 * action - 1

    def evaluate(self, x):
        alpha, beta = self(x)
        return BoundedBeta(alpha, beta)