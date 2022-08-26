import torch


def normalize_points(points):
    centroid = torch.mean(points, dim=0)
    points = points - centroid
    max_dist = torch.max(torch.sqrt(torch.sum(points ** 2, dim=1)))
    points /= max_dist

    return points
