import torch
import torch.nn as nn
import torchvision.models as models

net = models.resnet34(pretrained=True)
my_net = nn.Sequential(*list(net.children())[:-1])
print(my_net)

