from torch import nn
from torch.nn import functional as F


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
        :param x: input point features [B, K, N, C]
        :return: sampled point features [B, K, C']
        """
        x = x.permute(0, 3, 1, 2)

        for conv, bn in zip(self.convs, self.bns):
            x = self.relu(bn(conv(x)))

        x = F.max_pool2d(x, kernel_size=(1, x.shape[-1])).squeeze(-1)
        x = x.permute(0, 2, 1)

        return x
