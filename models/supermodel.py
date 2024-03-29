from collections import OrderedDict
import torch
from torch import Tensor
import torch.nn.functional as F
import nni.nas.nn.pytorch as nn
from typing import Tuple, Type, Any, Callable, Union, List, Optional
from models.ops import OPS, DropPath_, ZeroLayer
from nni.nas import model_wrapper
import copy


class Swish(nn.Module):
    def __init__(self, inplace=True):
        super(Swish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        if self.inplace:
            x.mul_(torch.sigmoid(x))
            return x
        else:
            return x * torch.sigmoid(x)


class _SampleLayer(nn.Module):
    SAMPLE_OPS = [
        'skip_connect',
        'conv_3x3',
        'conv_1x1',
        'sep_conv_3x3',
        'sep_conv_5x5',
        # 'sep_conv_7x7',
        'dil_conv_3x3',
        'avg_pool_3x3',
        # 'max_pool_3x3',
        'dil_sep_conv_3x3',
        'conv_3x1_1x3',
        'conv_7x1_1x7',
        'van_conv_3x3'
    ]
    # all ops
    # SAMPLE_OPS = ['none', 
    #               'avg_pool_3x3', 'avg_pool_5x5', 'avg_pool_7x7', 
    #               'skip_connect', 
    #               'conv_1x1', 'conv_3x3', 'conv_5x5', 'conv_7x7', 
    #               'sep_conv_3x3', 'sep_conv_5x5', 'sep_conv_7x7', 
    #               'dil_conv_3x3', 'dil_conv_5x5', 'dil_conv_7x7', 
    #               'dil_sep_conv_3x3', 'dil_sep_conv_5x5', 'dil_sep_conv_7x7', 
    #               'group_8_conv_3x3', 'group_8_conv_5x5', 'group_8_conv_7x7', 
    #               'conv_3x1_1x3', 'conv_5x1_1x5', 'conv_7x1_1x7', 
    #               'van_conv_3x3', 'van_conv_5x5', 'van_conv_7x7']
    # # remove 7x7
    # SAMPLE_OPS = ['none',  
    #                 'avg_pool_3x3', 'avg_pool_5x5', 
    #                 'skip_connect',
    #                 'conv_1x1', 'conv_3x3', 'conv_5x5',
    #                 'sep_conv_3x3', 'sep_conv_5x5',
    #                 'dil_conv_3x3', 'dil_conv_5x5',
    #                 'dil_sep_conv_3x3', 'dil_sep_conv_5x5',
    #                 'group_8_conv_3x3', 'group_8_conv_5x5',
    #                 'conv_3x1_1x3', 'conv_5x1_1x5',
    #                 'van_conv_3x3', 'van_conv_5x5']
    # remove avg_pool
    # SAMPLE_OPS = ['none', 
    #               'skip_connect', 
    #               'conv_1x1', 'conv_3x3', 'conv_5x5', 'conv_7x7', 
    #               'sep_conv_3x3', 'sep_conv_5x5', 'sep_conv_7x7', 
    #               'dil_conv_3x3', 'dil_conv_5x5', 'dil_conv_7x7', 
    #               'dil_sep_conv_3x3', 'dil_sep_conv_5x5', 'dil_sep_conv_7x7', 
    #               'group_8_conv_3x3', 'group_8_conv_5x5', 'group_8_conv_7x7', 
    #               'conv_3x1_1x3', 'conv_5x1_1x5', 'conv_7x1_1x7', 
    #               'van_conv_3x3', 'van_conv_5x5', 'van_conv_7x7']
    # remove 3x3, 5x5
    # SAMPLE_OPS = ['none',
    #               'avg_pool_7x7',
    #               'skip_connect',
    #               'conv_1x1', 'conv_7x7',
    #               'sep_conv_7x7',
    #               'dil_conv_7x7',
    #               'dil_sep_conv_7x7',
    #               'group_8_conv_7x7',
    #               'conv_7x1_1x7',
    #               'van_conv_7x7']
    # remove 3x3, 7x7
    # SAMPLE_OPS = ['none',
    #               'avg_pool_5x5',
    #               'skip_connect',
    #               'conv_1x1', 'conv_5x5',
    #               'sep_conv_5x5',
    #               'dil_conv_5x5',
    #               'dil_sep_conv_5x5',
    #               'group_8_conv_5x5',
    #               'conv_5x1_1x5',
    #               'van_conv_5x5']

    # remove 5x5, 7x7
    # SAMPLE_OPS = ['none',
    #               'avg_pool_3x3',
    #               'skip_connect',
    #               'conv_3x3',
    #               'sep_conv_3x3',
    #               'dil_conv_3x3',
    #               'conv_3x1_1x3',
    #               'group_8_conv_3x3',
    #               'van_conv_3x3']

    def __init__(
            self,
            inplanes: int,
            label: str,
            clamp=False
    ) -> None:
        super(_SampleLayer, self).__init__()
        self.clamp = clamp
        # extract feature from low dimension
        self.paths = nn.ModuleList([nn.Sequential(OPS[op](inplanes, 1, True), DropPath_()) for op in self.SAMPLE_OPS])
        # self.paths = nn.LayerChoice([OPS[op](inplanes, 1, True) for op in self.SAMPLE_OPS])
        self.input_switch = nn.InputChoice(n_candidates=len(self.SAMPLE_OPS), n_chosen=4, reduction='sum')
        # self.alpha = nn.Parameter(torch.rand(len(self.SAMPLE_OPS)) * 1E-3)
        self.nonlinear = nn.LayerChoice([nn.Identity(), nn.Hardswish()])
        # self.nonlinear = nn.ModuleList([nn.Identity(), nn.Hardswish()])
        # self.beta = nn.Parameter(torch.rand(2))
        self.softmax = nn.Softmax(dim=-1)

    def freeze(self, topk=1):
        _, indices = torch.topk(self.alpha, topk)
        y = torch.zeros_like(self.alpha)
        y[indices] = 1
        print(y)
        self.alpha = nn.Parameter(y)
        self.alpha.requires_grad_(False)

    def reset_clamp(self, clamp):
        self.clamp = clamp

    def replace_zero_layers(self, threshold=1e-2):
        print(self.alpha)
        print(self.softmax(self.alpha))
        # Iterate through paths and alpha
        for i, (path, alpha) in enumerate(zip(self.paths, self.softmax(self.alpha))):
            # Replace the path with a ZeroLayer if alpha is below the threshold
            if alpha.item() < threshold:
                self.paths[i] = ZeroLayer()
        # self.softmax = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
<<<<<<< HEAD
        weights = self.softmax(self.alpha)
        # if self.clamp:
        #     weights = self.alpha.clone().clamp(0, 1)
        # else:
        #     weights = self.alpha.clone()
        out = None
=======
        out = []
>>>>>>> pruning
        for idx, _ in enumerate(self.paths):
            out.append(self.paths[idx](x))
        out = self.input_switch(out) 
        out = self.nonlinear(out)    
        # out = self.nonlinear[0](out) * self.beta[0] + self.nonlinear[1](out) * self.beta[1] 
        # out = []
        # for idx, _ in enumerate(self.paths):
        #     out.append(self.paths[idx](x))
        # out = self.input_switch(out)
        return out


class SampleBlock(nn.ModuleDict):
    def __init__(
            self,
            num_layers: int,
            inplanes: int,
    ) -> None:
        super(SampleBlock, self).__init__()
        layer = nn.Sequential(
            nn.Conv2d(inplanes, inplanes * 2, kernel_size=1, stride=1),
            nn.BatchNorm2d(inplanes * 2),
        )
        self.add_module('upsamplelayer', layer)
        for i in range(num_layers):
            # FIXME: nn.InputChoice maybe needed
            layer = _SampleLayer(inplanes * 2, 'sampleunit')
            self.add_module('samplelayer%d' % (i + 1), layer)
        # self.add_module('samplelayer', PathSamplingRepeat(_SampleLayer(inplanes*2, 'sampleunit'), nn.ValueChoice([i+1 for i in range(num_layers)], label='samplelayer')))
        # self.add_module('relu', nn.Hardswish(inplace=True))

    def forward(self, init_features: Tensor) -> Tensor:
        features = init_features
        for _, layer in self.items():
            features = layer(features)
        return features


class AggregateBlock(nn.Module):
    def __init__(
            self,
            inplanes: list,
            outplanes: int,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(AggregateBlock, self).__init__()
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
        self.nonlinear = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Hardswish(inplace=True)
        )
        self.input_switch = nn.InputChoice(n_candidates=len(self.layers), n_chosen=2, reduction='sum')

    def freeze(self, topk=1):
        if topk == -1:
            topk = int(len(self.layers)/2)
        elif topk > len(self.layers):
            topk = len(self.layers) 
        _, indices = torch.topk(self.alpha, topk)
        y = torch.zeros_like(self.alpha)
        y[indices] = 1
        print(y)
        self.alpha = nn.Parameter(y)
        self.alpha.requires_grad_(False)

    def forward(self, x: List) -> Tensor:
        # weights = F.softmax(self.alpha, dim=-1)
        out = []
        for idx, _ in enumerate(self.layers):
            out.append(self.layers[idx](x[idx]))
        out = self.input_switch(out) 
        out = self.nonlinear(out)
        return out


# @model_wrapper
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
        elif dataset == 'tiny':
            init_stride = 2
            self.features = nn.Sequential(OrderedDict([
                ('conv0', nn.Conv2d(3, num_init_features, kernel_size=3, stride=init_stride,
                                    padding=1, bias=False)),
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
            block = SampleBlock(
                num_layers=num_layers,
                inplanes=num_features
            )
            self.samples.append(block)
            num_features = num_features * 2
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


def supermodel101(num_classes: int = 1000, pretrained: bool = False):
    return Supermodel(block_config=(3, 4, 23, 3), num_classes=num_classes)


def cifarsupermodel22(num_classes: int = 100, pretrained: bool = False):
    return Supermodel(dataset='cifar', block_config=(2, 2, 2), num_classes=num_classes)


def cifarsupermodel26(num_classes: int = 100, pretrained: bool = False):
    return Supermodel(dataset='cifar', block_config=(2, 2), num_classes=num_classes)


def cifarsupermodel50(num_classes: int = 100, pretrained: bool = False):
    return Supermodel(dataset='cifar', block_config=(3, 4, 6, 3), num_classes=num_classes)


def cifarsupermodel101(num_classes: int = 100, pretrained: bool = False):
    return Supermodel(dataset='cifar', block_config=(3, 4, 23, 3), num_classes=num_classes)


def tinysupermodel50(num_classes: int = 200, pretrained: bool = False):
    return Supermodel(dataset='tiny', block_config=(3, 4, 6, 3), num_classes=num_classes)