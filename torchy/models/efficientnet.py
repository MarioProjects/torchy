"""EfficientNet in PyTorch.
Paper: "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks".
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def swish_activation(x):
    return x * torch.sigmoid(x)


# activation_function = F.relu
activation_function = swish_activation


class Block(nn.Module):
    """expand + depthwise + pointwise + squeeze-excitation"""

    def __init__(self, in_planes, out_planes, expansion, stride, kernel_size):
        super(Block, self).__init__()
        self.stride = stride

        planes = expansion * in_planes
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        if kernel_size == 3:
            padding = 1
        elif kernel_size == 5:
            padding = 2
        else:
            assert False, "Kernel size {} not contempled".format(kernel_size)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=kernel_size,
                               stride=stride, padding=padding, groups=planes, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1,
                          stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_planes),
            )

        # SE layers
        self.fc1 = nn.Conv2d(out_planes, out_planes // 16, kernel_size=1)
        self.fc2 = nn.Conv2d(out_planes // 16, out_planes, kernel_size=1)

    def forward(self, x):
        out = activation_function(self.bn1(self.conv1(x)))
        out = activation_function(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        shortcut = self.shortcut(x) if self.stride == 1 else out
        # Squeeze-Excitation
        w = F.avg_pool2d(out, out.size(2))
        w = activation_function(self.fc1(w))
        w = self.fc2(w).sigmoid()
        out = out * w + shortcut
        return out


class EfficientNet(nn.Module):
    def __init__(self, cfg, num_classes=10):
        super(EfficientNet, self).__init__()
        self.cfg = cfg
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)

        self.layers = self._make_layers(in_planes=32)

        self.conv1x1 = nn.Conv2d(cfg[-2][2], cfg[-1][2], kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1x1 = nn.BatchNorm2d(cfg[-1][2])

        self.linear = nn.Linear(cfg[-1][2], num_classes)

    def _make_layers(self, in_planes):
        layers = []
        for kernel_size, expansion, out_planes, num_blocks, stride in self.cfg[:-1]:
            strides = [stride] + [1] * (num_blocks - 1)
            for stride in strides:
                layers.append(Block(in_planes, out_planes, expansion, stride, kernel_size))
                in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = activation_function(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = activation_function(self.bn1x1(self.conv1x1(out)))
        out = nn.AdaptiveAvgPool2d((1))(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def EfficientNetB0():
    # (kernel_size, expansion, out_planes, num_blocks, stride)
    cfg = [[3, 1, 16, 1, 2],
           [3, 6, 24, 2, 1],
           [5, 6, 40, 2, 2],
           [3, 6, 80, 3, 2],
           [5, 6, 112, 3, 1],
           [5, 6, 192, 4, 2],
           [3, 6, 320, 1, 2],
           [1, 1, 1280, 1, 1]]  # Last -> Conv1x1 & Pooling & F
    return EfficientNet(cfg)


def EfficientNet_Constants(d, w):
    # (kernel_size, expansion, out_planes, num_blocks, stride)
    cfg = [[3, 1, 16, 1, 2],
           [3, 6, 24, 2, 1],
           [5, 6, 40, 2, 2],
           [3, 6, 80, 3, 2],
           [5, 6, 112, 3, 1],
           [5, 6, 192, 4, 2],
           [3, 6, 320, 1, 2],
           [1, 1, 1280, 1, 1]]  # Last -> Conv1x1 & Pooling & F

    for index, cfg_row in enumerate(cfg):
        if index != len(cfg) - 1:  # No quiero añadir num_blocks en el last conv1x1
            cfg[index][-2] = math.ceil(cfg[index][-2] * d)
        cfg[index][2] = math.ceil(cfg[index][2] * w)

    return EfficientNet(cfg)


def test():
    net = EfficientNetB0()
    x = torch.randn(2, 3, 224, 224)
    y = net(x)
    print(y.shape)


def test_constants():
    net = EfficientNet_Constants(1.2, 1.1)
    x = torch.randn(2, 3, 224, 224)
    y = net(x)
    print(y.shape)


#test()
#test_constants()
