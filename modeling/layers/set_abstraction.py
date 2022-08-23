import torch
from torch import nn

from .pointnet import PointNet
from ..utils import farthest_point_sampling, sampling_and_grouping


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


class SetAbstractionMSG(nn.Module):
    def __init__(self, in_channel, hidden_dims_list, radius_list, num_query, num_sample_per_ball_list):
        super().__init__()
        self.radius_list = radius_list
        self.num_query = num_query
        self.num_sample_per_ball_list = num_sample_per_ball_list

        self.pointnets = nn.ModuleList()

        for dims in hidden_dims_list:
            self.pointnets.append(PointNet(in_channel, dims))

    def forward(self, xyz, features):
        """
        :param xyz: xyz coordinates of points [B, N, 3]
        :param features: point features [B, N, D]
        :return: sampled points xyz coordinates and features
        """
        fps_idx = farthest_point_sampling(xyz, self.num_query)
        msg_features = []

        for radius, num_sample_per_ball, pointnet in zip(self.radius_list, self.num_sample_per_ball_list, self.pointnets):
            xyz, sampled_features = sampling_and_grouping(xyz, features, radius, self.num_query, num_sample_per_ball, fps_idx)
            msg_features.append(pointnet(sampled_features))

        features = torch.cat(msg_features, dim=-1)

        return xyz, features
