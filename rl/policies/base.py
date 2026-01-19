import torch
import torch.nn as nn


def normc_fn(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)


class Net(nn.Module):
    def __init__(self):
        super().__init__()

    def init_parameters(self, output_layer=None):
        if getattr(self, "normc_init", True):
            self.apply(normc_fn)
            if output_layer is not None:
                output_layer.weight.data.mul_(0.01)
