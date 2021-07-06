"""
Some common layers for deep networks
"""

import torch
from torch import nn
from torch.nn import init


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class Dense2Conv(nn.Module):
    def forward(self, input):
        out = torch.repeat_interleave(input, 64 * 64)
        out = out.view(-1, input.shape[1], 64, 64)
        return out


def init_weights(model):
    if type(model) in [nn.Linear, nn.Conv2d]:
        init.xavier_uniform_(model.weight)
        init.constant_(model.bias, 0)
    elif type(model) in [nn.LSTMCell]:
        init.constant_(model.bias_ih, 0)
        init.constant_(model.bias_hh, 0)
