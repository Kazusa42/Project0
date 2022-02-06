import torch
import torchvision.models as models
import torchvision.transforms as transforms

from PIL import Image

from yolo_net import YoloBody

IMAGE = r'./test.png'

norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]

img = Image.open(IMAGE).convert('RGB')
# print(img)
inf_trans = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(416),
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std),
])


def img_trans(img_rgb, transform=None):
    if transform is None:
        raise ValueError('Can not find transform')
    img_t = transform(img_rgb)

    return img_t


x = img_trans(img, inf_trans)
# insert a dimention at position i: torch.unsqueeze(x, i)
x = torch.unsqueeze(x, 0)
net = YoloBody()

out = net(x)
print(out.shape)
# print(net)
