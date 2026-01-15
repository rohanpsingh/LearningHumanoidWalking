import torch
import torch.nn as nn


def normc_fn(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)


class Net(nn.Module):
    """Base class for actor and critic networks."""

    def __init__(self):
        super(Net, self).__init__()

    def forward(self):
        raise NotImplementedError

    def initialize_parameters(self):
        self.apply(normc_fn)
