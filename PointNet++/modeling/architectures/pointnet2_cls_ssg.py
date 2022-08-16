from torch import nn
from torch.nn import functional as F

from ..layers import SetAbstraction, ClassificationHead


class Model(nn.Module):
    def __int__(self, num_class, use_normal=False):
        super().__init__()
        self.use_normal = use_normal

        in_channel = 6 if use_normal else 3
        self.sa = nn.Sequential(
            SetAbstraction(
                in_channel=in_channel,
                hidden_dims=[64, 64, 128],
                radius=0.2,
                num_query=512,
                num_sample_per_ball=32
            ),
            SetAbstraction(
                in_channel=3 + 128,
                hidden_dims=[128, 128, 256],
                radius=0.4,
                num_query=128,
                num_sample_per_ball=64
            ),
            SetAbstraction(
                in_channel=3 + 256,
                hidden_dims=[256, 512, 1024],
                radius=None,
                num_query=None,
                num_sample_per_ball=None
            )
        )

        self.classification_head = ClassificationHead(
            in_channel=1024,
            hidden_dims=[512, 256],
            dropout_rates=[0.5, 0.5],
            num_class=num_class
        )

    def forward(self, x):
        if self.use_normal:
            xyz, features = x[:, :, 3], x[:, :, 3:]
        else:
            xyz, features = x, None

        xyz, features = self.sa(xyz, features)
        logits = self.classification_head(features)

        if self.training:
            return logits

        prob = F.softmax(logits)
        return prob

    def loss(self, predicted, gt):
        loss = F.cross_entropy(predicted, gt)

        return loss
