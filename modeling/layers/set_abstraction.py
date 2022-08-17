import torch
from torch import nn

from .pointnet import PointNet
from ..utils.pointnet2_utils import sampling_and_grouping


class SetAbstraction(nn.Module):
    def __init__(self, in_channel, hidden_dims, radius, num_query, num_sample_per_ball, group_all=False):
        super().__init__()
        self.radius = radius
        self.num_query = num_query
        self.num_sample_per_ball = num_sample_per_ball
        self.group_all = group_all

        self.pointnet = PointNet(in_channel, hidden_dims)

    def forward(self, xyz, features):
        """
        :param xyz: xyz coordinates of points [B, N, 3]
        :param features: point features [B, N, D]
        :return: sampled points xyz coordinates and features
        """
        if self.group_all:
            assert features is not None, "features should not be None!"
            features = torch.cat([xyz, features], dim=-1).unsqueeze(1)

        else:
            xyz, features = sampling_and_grouping(xyz, features, self.radius, self.num_query, self.num_sample_per_ball)

        features = self.pointnet(features)

        if self.group_all:
            features = features.squeeze(1)

        return xyz, features