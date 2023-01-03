from collections import OrderedDict

import torch
import torch.nn as tnn
import torch.nn.functional as F
from torch import Tensor
import nni.retiarii.nn.pytorch as nn
from typing import Type, Any, Callable, Union, List, Optional, Tuple

from models.sim_ops import AggregateBlock


class DenseNet(nn.Module):
    def __init__(
        self,
        num_init_features: int = 64,
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

        self.block1 = AggregateBlock([num_init_features], num_init_features*2, 1)
        self.block2 = AggregateBlock([num_init_features*2, num_init_features], num_init_features*4, 1)
        self.block3 = AggregateBlock([num_init_features*4, num_init_features*2, num_init_features],
                                     num_init_features*8, 1)
        self.block4 = AggregateBlock([num_init_features*8, num_init_features*4, num_init_features*2, num_init_features],
                                     num_init_features*16, 1)

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
        features.insert(0, self.block1(features))
        features.insert(0, self.block2(features))
        features.insert(0, self.block3(features))
        out = self.block4(features)

        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out


class CifarDenseNet(nn.Module):
    def __init__(
        self,
        num_init_features: int = 64,
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
        ]))

        self.block1 = AggregateBlock([num_init_features], num_init_features*2, 1)
        self.block2 = AggregateBlock([num_init_features*2, num_init_features], num_init_features*4, 1)
        self.block3 = AggregateBlock([num_init_features*4, num_init_features*2, num_init_features],
                                     num_init_features*8, 1)
        # self.block4 = AggregateBlock([num_init_features*8, num_init_features*4, num_init_features*2, num_init_features],
        #                              num_init_features*16, 1)

        # Linear layer
        self.classifier = nn.Linear(2048, num_classes)

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
        features.insert(0, self.block1(features))
        features.insert(0, self.block2(features))
        out = self.block3(features)

        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out
