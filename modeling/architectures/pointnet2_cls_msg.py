from torch import nn
from torch.nn import functional as F

from ..layers import SetAbstraction, SetAbstractionMSG, ClassificationHead


class _Model(nn.Module):
    def __init__(self, num_class, use_normal=False):
        super().__init__()
        self.use_normal = use_normal

        in_channel = 6 if use_normal else 3
        self.sa1 = SetAbstractionMSG(
            in_channel=in_channel,
            hidden_dims_list=[[32, 32, 64], [64, 64, 128], [64, 96, 128]],
            radius_list=[0.1, 0.2, 0.4],
            num_query=512,
            num_sample_per_ball_list=[16, 32, 128]
        )
        self.sa2 = SetAbstractionMSG(
            in_channel=3 + 64 + 128 + 128,
            hidden_dims_list=[[64, 64, 128], [128, 128, 256], [128, 128, 256]],
            radius_list=[0.2, 0.4, 0.8],
            num_query=128,
            num_sample_per_ball_list=[32, 64, 128]
        )
        self.sa3 = SetAbstraction(
            in_channel=3 + 128 + 256 + 256,
            hidden_dims=[256, 512, 1024],
            radius=None,
            num_query=None,
            num_sample_per_ball=None,
            group_all=True
        )

        self.classification_head = ClassificationHead(
            in_channel=1024,
            hidden_dims=[512, 256],
            dropout_rates=[0.5, 0.5],
            num_class=num_class
        )

    def forward(self, x):
        if self.use_normal:
            xyz, features = x[:, :, :3], x[:, :, 3:]
        else:
            xyz, features = x, None

        xyz, features = self.sa1(xyz, features)
        xyz, features = self.sa2(xyz, features)
        xyz, features = self.sa3(xyz, features)
        logits = self.classification_head(features)

        if self.training:
            return logits

        prob = F.softmax(logits, dim=-1)
        return prob


def get_model(args):
    return _Model(args.num_class, args.use_normal)


def get_loss_fn():
    return F.cross_entropy
