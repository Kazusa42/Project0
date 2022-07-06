import torch
from torchvision.models import resnet50
from nets.backbones import convnext_tiny, resnet50
from nets.resnet import ResNet, Bottleneck
from nets.convnext import ConvNeXt
from nets.frcnn import FasterRCNN

from thop import profile

model1, model2 = convnext_tiny(trans_neck=True, pretrained=False)
tmp1, tmp2 = resnet50(pretrained=False)
model = ResNet(Bottleneck, [3, 4, 6, 3])
test = ConvNeXt(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], transneck=False)
"""
print("tmp1")
print(tmp1)
print()
print("tmp2")
print(tmp2)
"""
print(type(model.layer1))
print('----------------------------')
print(type(test.downsample_layers[0]))

model = FasterRCNN(15, anchor_scales=[4, 16, 32], backbone='convnext_tiny',
                   pretrained=False, attentionneck=True)

img = torch.rand([1, 3, 640, 640])

"""x1 = model1.forward(img)
print(x1.shape)

x2 = tmp1.forward(img)
print(x2.shape)"""
model.forward(img)
