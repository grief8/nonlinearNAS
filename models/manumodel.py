from collections import OrderedDict
import torch
from torch import Tensor
import torch.nn.functional as F
import nni.nas.nn.pytorch as nn
from typing import Tuple, Type, Any, Callable, Union, List, Optional
from nni.nas.hub.pytorch.nasnet import OPS
from models.sim_ops import *


class Supermodel(nn.Module):
    def __init__(
            self,
            dataset: str = 'imagenet',
            num_init_features: int = 32,
            block_config: Tuple[int, int, int, int] = (2, 2, 2, 2),
            num_classes: int = 1000
    ) -> None:

        super(Supermodel, self).__init__()

        # First convolution
        if dataset == 'imagenet':
            init_stride = 4
            self.features = nn.Sequential(OrderedDict([
                ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=init_stride,
                                    padding=3, bias=False)),
                ('norm0', nn.BatchNorm2d(num_init_features)),
            ]))
        else:
            self.features = nn.Sequential(OrderedDict([
                ('conv0', nn.Conv2d(3, num_init_features, kernel_size=3, stride=1,
                                    padding=1, bias=False)),
                ('norm0', nn.BatchNorm2d(num_init_features)),
            ]))

        self.samples = nn.ModuleList()
        self.aggeregate = nn.ModuleList()
        channels = [num_init_features]
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            # FIXME: Maybe unreasonable without channel adding
            block = nn.Sequential(SampleBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                growth_rate=32
                )
            )
            self.samples.append(block)
            num_features = num_features + num_layers * 32
            if i != len(block_config) - 1:
                channels.append(num_features)
                trans = AggregateBlock(channels, num_features)
                self.aggeregate.append(trans)

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

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



def supermodel16(num_classes: int = 1000, pretrained: bool = False):
    return Supermodel(block_config=(2, 4), num_classes=num_classes)


def supermodel8(num_classes: int = 1000, pretrained: bool = False):
    return Supermodel(block_config=(2, 2, 2), num_classes=num_classes)


def supermodel50(num_classes: int = 1000, pretrained: bool = False):
    return Supermodel(block_config=(3, 4, 6, 3), num_classes=num_classes)

def cifarsupermodel16(num_classes: int = 100, pretrained: bool = False):
    return Supermodel(dataset='cifar', block_config=(2, 2, 2, 2), num_classes=num_classes)


def cifarsupermodel22(num_classes: int = 100, pretrained: bool = False):
    return Supermodel(dataset='cifar', block_config=(2, 2, 2), num_classes=num_classes)


def cifarsupermodel26(num_classes: int = 100, pretrained: bool = False):
    return Supermodel(dataset='cifar', block_config=(2, 2), num_classes=num_classes)


def cifarsupermodel50(num_classes: int = 100, pretrained: bool = False):
    return Supermodel(dataset='cifar', block_config=(3, 4, 6, 3), num_classes=num_classes)


def cifarsupermodel101(num_classes: int = 100, pretrained: bool = False):
    return Supermodel(dataset='cifar', block_config=(3, 4, 23, 3), num_classes=num_classes)