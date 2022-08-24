import os
import json

import torch
from torch.utils.data import Dataset
import numpy as np

from .utils import normalize_points


class ShapeNetDataset(Dataset):
    def __init__(self, path, args, split="train"):
        self.num_points = args.num_point
        self.use_normal = args.use_normal
        cat_file = os.path.join(path, "synsetoffset2category.txt")

        category = {}

        with open(cat_file, "r") as f:
            for line in f:
                category, directory = line.strip().split()
                category[directory] = category

        self.category_id = dict(zip(category, range(len(category))))

        assert split == "test" or split == "train" or split == "val", "Split should be train, val or test!"

        data_list_file = os.path.join(path, "train_test_split", f"shuffled_{split}_file_list.json")
        with open(data_list_file) as f:
            data_path = json.load(f)

        self.data_path = [os.path.join(path, file.split("/", 1)[1]) + ".txt" for file in data_path]

    def __len__(self):
        return len(self.data_path)

    def __getitem__(self, index):
        data = torch.from_numpy(np.loadtxt(self.data_path[index]).astype(np.float32))

        cls_label = self.category_id[self.data_path[index].split("/")[-2]]

        random_idx = torch.multinomial(torch.ones(data.shape[0]), num_samples=self.num_points, replacement=False)
        points = data[random_idx, 0:6] if self.use_normal else data[random_idx, 0:3]
        points[:, 0:3] = normalize_points(points[:, 0:3])

        seg_label = points[random_idx, 6]

        return points, cls_label, seg_label
