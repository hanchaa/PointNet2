import torch


def farthest_point_sampling(xyz, num_query):
    """
    :param xyz: xyz coordinates of points [B, N, 3]
    :param num_query: the number of points to sample
    :return: sampled points [B, num_points]
    """
    B, N, _ = xyz.shape
    device = xyz.device

    centroids = xyz.new_zeros((B, num_query), dtype=torch.long)
    centroids[:, 0] = torch.randint(0, N, (B, )).to(device)
    distances = xyz.new_ones((B, N)) * 1e10

    for i in range(num_query - 1):
        center = xyz[range(B), centroids[:, i]].unsqueeze(1)
        dist = torch.sum((center - xyz) ** 2, dim=-1)
        distances = torch.where(dist < distances, dist, distances)
        centroids[:, i + 1] = torch.argmax(distances, dim=-1)

    return centroids


def _ball_query(xyz, query_xyz, radius, num_sample_per_ball):
    """
    :param xyz: xyz coordinates of original points [B, N, 3]
    :param query_xyz: xyz coordinates of query points [B, K, 3]
    :param radius: radius of ball
    :param num_sample_per_ball: the number of points to sample inside the ball
    :return: indices of points inside each ball [B, N, K]
    """
    B, N, _ = xyz.shape
    K = query_xyz.shape[1]

    xyz = xyz.unsqueeze(1).repeat((1, K, 1, 1))
    query_xyz = query_xyz.unsqueeze(2).repeat((1, 1, N, 1))

    dist = torch.sqrt(torch.sum((query_xyz - xyz) ** 2, dim=-1))

    group_idx = xyz.new_tensor(range(N), dtype=torch.long).repeat((B, K, 1))
    group_idx[dist > radius] = N
    group_idx = torch.sort(group_idx, dim=-1)[0][:, :, :num_sample_per_ball]

    over_sample = group_idx[:, :, 0].unsqueeze(-1).repeat((1, 1, num_sample_per_ball))
    mask = group_idx == N
    group_idx[mask] = over_sample[mask]

    return group_idx


def sampling_and_grouping(xyz, features, radius, num_query, num_sample_per_ball, fps_idx=None):
    """
    :param xyz: xyz coordinates of points [B, N, 3]
    :param features: point features [B, N, D]
    :param radius: radius of ball query
    :param num_query: the number of points to sample using FPS
    :param num_sample_per_ball: the number of  points to sample inside ball
    :param fps_idx: indices of points of sampled by farthest point sampling
    :return: sampled points xyz coordinates and features
    """
    B = xyz.shape[0]

    batch_idx = xyz.new_tensor(range(B), dtype=torch.long).unsqueeze(-1).repeat((1, num_query))
    if fps_idx is None:
        fps_idx = farthest_point_sampling(xyz, num_query)
    query_xyz = xyz[batch_idx, fps_idx]

    batch_idx = batch_idx.unsqueeze(-1).repeat((1, 1, num_sample_per_ball))
    query_idx = xyz.new_tensor(range(num_query), dtype=torch.long).view(1, num_query, 1).repeat((B, 1, num_sample_per_ball))
    group_idx = _ball_query(xyz, query_xyz, radius, num_sample_per_ball)
    grouped_xyz = xyz.unsqueeze(1).repeat((1, num_query, 1, 1))[batch_idx, query_idx, group_idx]

    grouped_xyz -= query_xyz.unsqueeze(2)

    if features is not None:
        grouped_features = features.unsqueeze(1).repeat((1, num_query, 1, 1,))[batch_idx, query_idx, group_idx]
        grouped_features = torch.cat([grouped_xyz, grouped_features], dim=-1)

    else:
        grouped_features = grouped_xyz

    return query_xyz, grouped_features
