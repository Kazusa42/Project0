import math

import torch.nn as nn
import torchvision.models as models


def resnet50(pretrain=True):
    model = models.resnet50(pretrained=pretrain)

    features = list([model.conv1, model.bn1, model.relu, model.maxpool, model.layer1, model.layer2, model.layer3])
    classifier = list([model.layer4, model.avgpool])
    
    features = nn.Sequential(*features)
    classifier = nn.Sequential(*classifier)
    return features, classifier

