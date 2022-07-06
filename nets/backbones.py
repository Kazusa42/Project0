import torch.nn as nn
import torch
from torch.hub import load_state_dict_from_url

from .resnet import Bottleneck, ResNet
from .convnext import ConvNeXt


def resnet50(pretrained=False):
    model = ResNet(Bottleneck, [3, 4, 6, 3])
    if pretrained:
        state_dict = load_state_dict_from_url("https://download.pytorch.org/models/resnet50-19c8e357.pth",
                                              model_dir="./model_data")
        model.load_state_dict(state_dict)
    features = list([model.conv1, model.bn1, model.relu, model.maxpool, model.layer1, model.layer2, model.layer3])
    classifier = list([model.layer4, model.avgpool])

    features = nn.Sequential(*features)
    classifier = nn.Sequential(*classifier)
    return features, classifier


model_urls = {
    "convnext_tiny_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth",
}


def convnext_tiny(trans_neck=True, pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], transneck=trans_neck, **kwargs)
    if pretrained:
        print('Load pre-trained backbone.')
        url = model_urls['convnext_tiny_22k'] if in_22k else model_urls['convnext_tiny_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint["model"], strict=False)

    features = list([model.downsample_layers[0], model.stages[0], model.downsample_layers[1], model.stages[1],
                     model.downsample_layers[2], model.stages[2], model.downsample_layers[3], model.stages[3]])
    classifier = list([nn.AvgPool2d(7)])

    features = nn.Sequential(*features)
    classifier = nn.Sequential(*classifier)
    return features, classifier

