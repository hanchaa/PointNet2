import torch

from .modelnet_dataset import ModelNetDataset
from .shapenet_dataset import ShapeNetDataset

def normalize_points(points):
    centroid = torch.mean(points, dim=0)
    points = points - centroid
    max_dist = torch.max(torch.sqrt(torch.sum(points ** 2, dim=1)))
    points /= max_dist

    return points


def build_dataset(args, split):
    dataset = args.dataset

    if dataset == "ModelNet10" or dataset == "ModelNet40":
        num_category = int(dataset[-2:])
        dataset = ModelNetDataset("./datasets/modelnet", args, num_category, split)

    elif "ShapeNet" in dataset:
        dataset = ShapeNetDataset("./datasets/shapenet", args, split)

    else:
        raise Exception("Unsupported dataset! Supporting datasets: ModelNet10, ModelNet40, ShapeNet")

    return dataset
