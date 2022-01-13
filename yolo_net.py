import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np


class YoloBody(nn.Module):
    def __init__(self, num_classes=80):
        super(YoloBody, self).__init__()

        self.backbone = models.resnet50(pretrained=True)
        # resnet50 fc output channels is 1000
        self.fc = nn.Linear(1000, 7 * 7 * (10 + num_classes))

    def forward(self, x):
        x = self.backbone(x)

        return self.fc(x)

