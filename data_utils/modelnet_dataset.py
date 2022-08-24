import os

import numpy as np
import torch
from torch.utils.data import Dataset

from .utils import normalize_points


class ModelNetDataset(Dataset):
    def __init__(self, path, args, split="train"):
        self.path = path
        self.num_points = args.num_point
        self.num_category = args.num_category
        self.use_normal = args.use_normal

        assert self.num_category == 10 or self.num_category == 40, "The number of categories should be 10 or 40!"
        assert split == "train" or split == "test", "Split should be train or test!"

        category_file = os.path.join(self.path, f"modelnet{self.num_category}_shape_names.txt")
        shape_id_file = os.path.join(self.path, f"modelnet{self.num_category}_{split}.txt")

        with open(category_file) as f:
            self.categories = [line.strip() for line in f]
            self.categories = dict(zip(self.categories, range(len(self.categories))))

        with open(shape_id_file) as f:
            shape_id = [line.strip() for line in f]

        self.data_path = [(shape.rsplit("_", 1)[0], os.path.join(self.path, shape.rsplit("_", 1)[0], f"{shape}.txt")) for shape in shape_id]

    def __len__(self):
        return len(self.data_path)

    def __getitem__(self, index):
        data = self.data_path[index]

        label = self.categories[data[0]]
        label = torch.tensor([label], dtype=torch.int64)

        points = torch.from_numpy(np.loadtxt(data[1], delimiter=",").astype(np.float32))
        random_idx = torch.multinomial(torch.ones(points.shape[0]), num_samples=self.num_points, replacement=False)
        points = points[random_idx]

        points[:, 0:3] = normalize_points(points[:, 0:3])
        if not self.use_normal:
            points = points[:, 0:3]

        return points, label[0]
