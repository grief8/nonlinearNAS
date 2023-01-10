from collections import OrderedDict

import torch
import torch.nn as tnn
import torch.nn.functional as F
from torch import Tensor
import nni.retiarii.nn.pytorch as nn
from typing import Type, Any, Callable, Union, List, Optional, Tuple

from models.sim_ops import *


class DenseNet(nn.Module):
    def __init__(
        self,
        growth_rate: int = 32,
        num_init_features: int = 64,
        block_config: Tuple[int, int, int, int] = (6, 12, 24, 16),
        num_classes: int = 1000,
        pretrained: bool = False
    ) -> None:

        super(DenseNet, self).__init__()

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2,
                                padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        self.samples = nn.ModuleList()
        self.aggeregate = nn.ModuleList()
        channels = [num_init_features]
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = SampleBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                growth_rate=growth_rate
            )
            self.samples.append(block)
            # self.features.add_module('sampleblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                channels.append(num_features)
                # print('channels: ', channels)
                trans = AggregateBlock(inplanes=channels,
                                    outplanes=num_features // 2)
                # self.features.add_module('aggregate%d' % (i + 1), trans)
                self.aggeregate.append(trans)
                num_features = num_features // 2
                channels[-1] = num_features
        
        # Linear layer
        self.classifier = nn.Linear(num_init_features * 16, num_classes)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                tnn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                tnn.init.constant_(m.weight, 1)
                tnn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                tnn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        features = [self.features(x)]
        for idx in range(len(self.aggeregate)):
            # print(idx)
            features.append(self.samples[idx](features[-1]))
            # print(len(features))
            # print(features[-1].shape)
            features[-1] = self.aggeregate[idx](features)
        out = self.samples[-1](features[-1])

        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out


class CifarDenseNet(nn.Module):
    def __init__(
        self,
        growth_rate: int = 32,
        num_init_features: int = 64,
        block_config: Tuple[int, int, int, int] = (6, 12, 24),
        num_classes: int = 1000,
        pretrained: bool = False
    ) -> None:

        super(CifarDenseNet, self).__init__()

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2,
                                padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        self.samples = nn.ModuleList()
        self.aggeregate = nn.ModuleList()
        channels = [num_init_features]
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = SampleBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                growth_rate=growth_rate
            )
            self.samples.append(block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                channels.append(num_features)
                trans = AggregateBlock(inplanes=channels,
                                    outplanes=num_features // 2)
                self.aggeregate.append(trans)
                num_features = num_features // 2
                channels[-1] = num_features
        
        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                tnn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                tnn.init.constant_(m.weight, 1)
                tnn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                tnn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        features = [self.features(x)]
        for idx in range(len(self.aggeregate)):
            features.append(self.samples[idx](features[-1]))
            features[-1] = self.aggeregate[idx](features)
        out = self.samples[-1](features[-1])

        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out

