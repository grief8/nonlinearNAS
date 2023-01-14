from collections import OrderedDict
import torch
from torch import Tensor
import torch.nn.functional as F
import nni.nas.nn.pytorch as nn
from typing import Tuple, Type, Any, Callable, Union, List, Optional
from nni.nas.hub.pytorch.nasnet import OPS
from nni.nas import model_wrapper


class _SampleLayer(nn.Module):
    SAMPLE_OPS = [
        'skip_connect',
        'conv_3x3',
        'conv_1x1',
        'sep_conv_3x3',
        'sep_conv_5x5',
        'sep_conv_7x7',
        'avg_pool_3x3',
        'max_pool_3x3',
        'dil_sep_conv_3x3',
        # 'conv_3x1_1x3',
        # 'conv_7x1_1x7',
    ]

    def __init__(
            self,
            inplanes: int,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(_SampleLayer, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        # extract feature from low dimension
        self.paths = nn.ModuleList()
        for _ in range(3):
            self.paths.append(nn.Sequential(
                nn.Conv2d(inplanes, inplanes//4, kernel_size=1, stride=1),
                nn.LayerChoice([OPS[op](inplanes//4, 1, True) for op in self.SAMPLE_OPS]),
                norm_layer(inplanes//4),
                nn.ReLU()
            ))
        # feature aggregation
        self.agg_node = nn.Sequential(
            norm_layer(inplanes//4),
            nn.Conv2d(inplanes//4, inplanes, kernel_size=1, stride=1),
            norm_layer(inplanes),
            nn.ReLU()
        )

    def forward(self, x: Tensor) -> Tensor:
        out = None
        for idx, _ in enumerate(self.paths):
            if out is None:
                out = self.paths[idx](x)
            else:
                out = out + self.paths[idx](x)
        # maybe division is needed
        out = self.agg_node(out/3)
        return out


class SampleBlock(nn.ModuleDict):
    def __init__(
        self,
        num_layers: int,
        inplanes: int,
    ) -> None:
        super(SampleBlock, self).__init__()
        for i in range(num_layers):
            # FIXME: nn.InputChoice maybe needed
            layer = _SampleLayer(inplanes)
            self.add_module('samplelayer%d' % (i + 1), layer)

    def forward(self, init_features: Tensor) -> Tensor:
        features = [init_features] 
        for _, layer in self.items():
            new_features = layer(sum(features)/len(features))
            features.append(new_features)
        return features[-1]


class TransitionBlock(nn.Module):
    def __init__(
            self,
            inplanes: list,
            outplanes: int,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(TransitionBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.layers = nn.ModuleList()
        compensation = 1
        config = inplanes[:]
        config.reverse()
        for idx, inp in enumerate(config):
            self.layers.insert(0, nn.Sequential(
                nn.Conv2d(inp, outplanes, kernel_size=1, stride=compensation, bias=False),
                norm_layer(outplanes)
            ))
            compensation = 2 ** (idx)
        self.transition = nn.Sequential(
            nn.Conv2d(outplanes, outplanes, kernel_size=3, stride=2,
                      padding=1, bias=False),
            norm_layer(outplanes),
            nn.ReLU(),
        )

    def forward(self, x: List) -> Tensor:
        out = None
        for idx, _ in enumerate(self.layers):
            if out is None:
                out = self.layers[idx](x[idx])
            else:
                out = out + self.layers[idx](x[idx])

        out = self.transition(out)
        return out


# @model_wrapper
class Supermodel(nn.Module):
    def __init__(
        self,
        dataset: str = 'imagenet',
        num_init_features: int = 64,
        block_config: Tuple[int, int, int, int] = (6, 12, 24, 16),
        num_classes: int = 1000
    ) -> None:

        super(Supermodel, self).__init__()

        # First convolution
        if dataset == 'imagenet':
            self.features = nn.Sequential(OrderedDict([
                ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2,
                                    padding=3, bias=False)),
                ('norm0', nn.BatchNorm2d(num_init_features)),
                ('relu0', nn.ReLU()),
                ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
            ]))
        else:
            self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=3, stride=1,
                                padding=1, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU()),
        ]))

        self.samples = nn.ModuleList()
        self.aggeregate = nn.ModuleList()
        channels = [num_init_features]
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            # FIXME: Maybe unreasonable without channel adding
            block = SampleBlock(
                num_layers=num_layers,
                inplanes=num_features
            )
            self.samples.append(block)
            if i != len(block_config) - 1:
                channels.append(num_features)
                trans = TransitionBlock(inplanes=channels,
                                    outplanes=num_features * 2)
                self.aggeregate.append(trans)
                num_features = num_features * 2
                channels[-1] = num_features
        
        # Linear layer
        self.classifier = nn.Linear(num_init_features * 8, num_classes)

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
            features.append(self.samples[idx](features[-1].clone()))
            features[-1] = self.aggeregate[idx](features)
        out = self.samples[-1](features[-1].clone())
        
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out


def supermodel121(num_classes: int = 1000, pretrained: bool = False):
    return Supermodel(block_config=[6,12,24,16], num_classes=num_classes)

def supermodel169(num_classes: int = 1000, pretrained: bool = False):
    return Supermodel(block_config=[6,12,32,32], num_classes=num_classes)

def supermodel201(num_classes: int = 1000, pretrained: bool = False):
    return Supermodel(block_config=[6,12,48,32], num_classes=num_classes)

def supermodel161(num_classes: int = 1000, pretrained: bool = False):
    return Supermodel(block_config=[6,12,36,24], num_classes=num_classes)

def cifarsupermodel121(num_classes: int = 100, pretrained: bool = False):
    return Supermodel(dataset='cifar', block_config=[6,12,24,16], num_classes=num_classes)

def cifarsupermodel169(num_classes: int = 100, pretrained: bool = False):
    return Supermodel(dataset='cifar', block_config=[6,12,32,32], num_classes=num_classes)

def cifarsupermodel201(num_classes: int = 100, pretrained: bool = False):
    return Supermodel(dataset='cifar', block_config=[6,12,48,32], num_classes=num_classes)

def cifarsupermodel161(num_classes: int = 100, pretrained: bool = False):
    return Supermodel(dataset='cifar', block_config=[6,12,36,24], num_classes=num_classes)
