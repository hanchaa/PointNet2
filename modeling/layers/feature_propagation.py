import torch
from torch import nn


class FeaturePropagation(nn.Module):
    def __init__(self, in_channel, hidden_dims, num_k):
        super().__init__()

        self.num_k = num_k

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.relu = nn.ReLU()

        previous_channel = in_channel
        for dim in hidden_dims:
            self.convs.append(nn.Conv1d(previous_channel, dim, 1))
            self.bns.append(nn.BatchNorm1d(dim))
            previous_channel = dim

    def forward(self, xyz1, xyz2, features1, features2):
        """
        :param xyz1: coordinates of abstracted points [B, N, 3]
        :param xyz2: coordinates of points to be propagated [B, N', 3]
        :param features1: features of abstracted points [B, N, D]
        :param features2: features of points to be propagated [B, N', D]
        :return: new features of points [B, N', D]
        """
        B, N, N_prime = xyz1.shape[0], xyz1.shape[1], xyz2.shape[1]

        xyz1 = xyz1.unsqueeze(1).repeat(1, N_prime, 1, 1)
        xyz2 = xyz2.unsqueeze(2).repeat(1, 1, N, 1)

        dist = ((xyz2 - xyz1) ** 2).sum(dim=-1)
        dist = 1 / (dist + torch.finfo(torch.float32).eps)
        neighbor = dist.topk(self.num_k)
        weights = neighbor.values / neighbor.values.sum(dim=-1).unsqueeze(-1)

        batch_idx = xyz1.new_tensor(range(B), dtype=torch.long).view(B, 1, 1).repeat(1, N_prime, self.num_k)
        query_idx = xyz1.new_tensor(range(N_prime), dtype=torch.long).view(1, N_prime, 1).repeat(B, 1, self.num_k)

        interpolated_features = features1.unsqueeze(1).repeat(1, N_prime, 1, 1)[batch_idx, query_idx, neighbor.indices]
        interpolated_features = (interpolated_features * weights.unsqueeze(-1)).sum(dim=-2)

        features = torch.cat([interpolated_features, features2], dim=-1)
        features = features.permute(0, 2, 1)

        for conv, bn in zip(self.convs, self.bns):
            features = self.relu(bn(conv(features)))

        features = features.permute(0, 2, 1)

        return features
