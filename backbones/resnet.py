"""

ResNet using the resnet from torchvision.

"""

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from backbones import tv_resnet


class ResNet(nn.Module):
    def __init__(self, block, layers, final_dim: int, pretrain_v: str, progress: bool = True, **kwargs: Any):
        super(ResNet, self).__init__()
        self.in_planes = 64

        if pretrain_v == 'V1':
            self.tv_layers = tv_resnet.resnet50(weights=tv_resnet.ResNet50_Weights.IMAGENET1K_V1)
        elif pretrain_v == 'V2':
            self.tv_layers = tv_resnet.resnet50(weights=tv_resnet.ResNet50_Weights.IMAGENET1K_V2)
        else:
            self.tv_layers = tv_resnet._resnet(block, layers = layers, weights = None, progress=progress, **kwargs)

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        
        self.linear = nn.Linear(512*block.expansion, out_features = final_dim)

        self.repr_dim = final_dim

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.tv_layers.conv1(x)
        x = self.tv_layers.bn1(x)
        x = self.tv_layers.relu(x)
        x = self.tv_layers.maxpool(x)

        x = self.tv_layers.layer1(x)
        x = self.tv_layers.layer2(x)
        x = self.tv_layers.layer3(x)
        x = self.tv_layers.layer4(x)

        x = self.tv_layers.avgpool(x)
        x = torch.flatten(x, 1)
        out = self.linear(x)
        return out


def ResNet18(final_dim: int):
    return ResNet(tv_resnet.BasicBlock, [2, 2, 2, 2], final_dim, pretrain_v='')

def ResNet34(final_dim: int):
    return ResNet(tv_resnet.BasicBlock, [3, 4, 6, 3], final_dim, pretrain_v='')

def ResNet50(final_dim: int):
    return ResNet(tv_resnet.Bottleneck, [3, 4, 6, 3], final_dim, pretrain_v='')

def pretrain_ResNet50_V1(final_dim: int):
    return ResNet(tv_resnet.Bottleneck, [3, 4, 6, 3], final_dim, pretrain_v='V1')

def pretrain_ResNet50_V2(final_dim: int):
    return ResNet(tv_resnet.Bottleneck, [3, 4, 6, 3], final_dim, pretrain_v='V2')


def test():
    final_dim = 3
    net = ResNet18(final_dim)
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())

# test()
