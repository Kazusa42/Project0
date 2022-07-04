import torch
from torch import nn
import torch.nn.functional as F
from timm.models.layers import DropPath

from configure import RESOLUTION


def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        n = 1
        for s in list(p.size()):
            n = n * s
        pp += n
    return pp


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class MHSA(nn.Module):
    def __init__(self, n_dims, width, height, heads=4):
        super(MHSA, self).__init__()
        self.heads = heads

        self.query = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        self.key = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        self.value = nn.Conv2d(n_dims, n_dims, kernel_size=1)

        self.rel_h = nn.Parameter(torch.randn([1, heads, n_dims // heads, 1, height]), requires_grad=True)
        self.rel_w = nn.Parameter(torch.randn([1, heads, n_dims // heads, width, 1]), requires_grad=True)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        n_batch, channel, width, height = x.size()
        q = self.query(x).view(n_batch, self.heads, channel // self.heads, -1)
        k = self.key(x).view(n_batch, self.heads, channel // self.heads, -1)
        v = self.value(x).view(n_batch, self.heads, channel // self.heads, -1)

        content_content = torch.matmul(q.permute(0, 1, 3, 2), k)

        content_position = (self.rel_h + self.rel_w).view(1, self.heads, channel // self.heads, -1).permute(0, 1, 3, 2)
        content_position = torch.matmul(content_position, q)

        energy = content_content + content_position
        attention = self.softmax(energy)

        out = torch.matmul(v, attention.permute(0, 1, 3, 2))
        out = out.view(n_batch, channel, width, height)

        return out


class AttentionNeck1(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, heads=4, drop_path=0., resolution=RESOLUTION):
        super(AttentionNeck1, self).__init__()

        self.dsconv = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)

        self.attention = nn.ModuleList()
        self.attention.append(MHSA(planes, width=int(resolution[0]), height=int(resolution[1]), heads=heads))
        self.attention = nn.Sequential(*self.attention)

        self.layernorm = LayerNorm(planes)

        self.conv1 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        out = self.dsconv(x)
        out = self.attention(out)
        out = out.permute(0, 2, 3, 1)
        out = self.layernorm(out)
        out = out.permute(0, 3, 1, 2)
        out = F.gelu(self.conv1(out))
        # print(out.shape)
        return out + self.drop_path(x)


class AttentionNeck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, heads=4, resolution=RESOLUTION):
        super(AttentionNeck, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.ModuleList()
        self.conv2.append(MHSA(planes, width=int(resolution[0]), height=int(resolution[1]), heads=heads))
        if stride == 2:
            self.conv2.append(nn.AvgPool2d(2, 2))
        self.conv2 = nn.Sequential(*self.conv2)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.gelu(self.bn1(self.conv1(x)))
        out = F.gelu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.gelu(out)
        return out
