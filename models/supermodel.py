from collections import OrderedDict
import torch
from torch import Tensor
import torch.nn.functional as F
import nni.nas.nn.pytorch as nn
from typing import Tuple, Type, Any, Callable, Union, List, Optional
from models.ops import OPS, DropPath_
from nni.nas import model_wrapper


class Swish(nn.Module):
	def __init__(self,inplace=True):
		super(Swish,self).__init__()
		self.inplace = inplace

	def forward(self,x):
		if self.inplace:
			x.mul_(torch.sigmoid(x))
			return x
		else:
			return x*torch.sigmoid(x)	


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

    def __init__(
            self,
            inplanes: int,
            label: str
    ) -> None:
        super(_SampleLayer, self).__init__()
        # extract feature from low dimension
        # self.paths = nn.ModuleList([nn.Sequential(OPS[op](inplanes, 1, True), DropPath_() )  for op in self.SAMPLE_OPS])
        self.paths = nn.LayerChoice([OPS[op](inplanes, 1, True) for op in self.SAMPLE_OPS])
        # self.input_switch = nn.InputChoice(n_candidates=len(self.SAMPLE_OPS), n_chosen=4, reduction='sum')

    def forward(self, x: Tensor) -> Tensor:
        # out = None
        # for idx, _ in enumerate(self.paths):
        #     if out is None:
        #         out = self.paths[idx](x)
        #     else:
        #         out = out + self.paths[idx](x)
        # out = []
        # for idx, _ in enumerate(self.paths):
        #     out.append(self.paths[idx](x))
        # out = self.input_switch(out)
        out = self.paths(x)
        return out


class SampleBlock(nn.ModuleDict):
    def __init__(
        self,
        num_layers: int,
        inplanes: int,
    ) -> None:
        super(SampleBlock, self).__init__()
        layer = nn.Sequential(
                nn.Conv2d(inplanes, inplanes*2, kernel_size=1, stride=1),
                nn.BatchNorm2d(inplanes*2),
            )
        self.add_module('upsamplelayer', layer)
        for i in range(num_layers):
            # FIXME: nn.InputChoice maybe needed
            layer = _SampleLayer(inplanes*2, 'sampleunit')
            self.add_module('samplelayer%d' % (i + 1), layer)
        # self.add_module('samplelayer', PathSamplingRepeat(_SampleLayer(inplanes*2, 'sampleunit'), nn.ValueChoice([i+1 for i in range(num_layers)], label='samplelayer')))
        self.add_module('relu', nn.Hardswish(inplace=True))

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
        
    def forward(self, x: List) -> Tensor:
        out = None
        for idx, _ in enumerate(self.layers):
            if out is None:
                out = self.layers[idx](x[idx])
            else:
                out = out + self.layers[idx](x[idx])

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
            init_stride = 2* 2** (4 - len(block_config))
            self.features = nn.Sequential(OrderedDict([
                ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=init_stride,
                                    padding=3, bias=False)),
                ('norm0', nn.BatchNorm2d(num_init_features)),
                ('relu0', nn.Hardswish()),
                ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
            ]))
        else:
            init_stride = 2** (4 - len(block_config))
            self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=3, stride=init_stride,
                                padding=1, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.Hardswish()),
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

def cifarsupermodel16(num_classes: int = 100, pretrained: bool = False):
    return Supermodel(dataset='cifar', block_config=(2, 2, 2, 2), num_classes=num_classes)

def cifarsupermodel22(num_classes: int = 100, pretrained: bool = False):
    return Supermodel(dataset='cifar', block_config=(4, 6, 8, 4), num_classes=num_classes)

def cifarsupermodel26(num_classes: int = 100, pretrained: bool = False):
    return Supermodel(dataset='cifar', block_config=(4, 6, 8, 8), num_classes=num_classes)