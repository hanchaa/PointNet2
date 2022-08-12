import torch
from torch import nn


class PointNet(nn.Module):
    def __init__(self, in_channel, hidden_dims):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.relu = nn.ReLU()

        previous_channel = in_channel
        for dim in hidden_dims:
            self.convs.append(nn.Conv2d(previous_channel, dim, 1))
            self.bns.append(nn.BatchNorm2d(dim))
            previous_channel = dim

    def forward(self, x):
        """
        :param x: input point features [B, C, K, N]
        :return: sampled point features [B, C', K]
        """
        for conv, bn in zip(self.convs, self.bns):
            x = self.relu(bn(conv(x)))

        x = torch.max(x, dim=-1)[0]

        return x
