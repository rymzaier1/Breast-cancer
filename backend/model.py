import torch.nn as nn
from torchvision import models


class BreastCancerModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.model = models.efficientnet_b0(
            weights=None
        )

        self.model.classifier[1] = nn.Linear(
            self.model.classifier[1].in_features,
            2
        )

    def forward(self, x):

        return self.model(x)