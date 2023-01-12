import torch
from torch import Tensor
import nni.nas.nn.pytorch as nn
from typing import Type, Any, Callable, Union, List, Optional
from nni.nas.hub.pytorch.nasnet import OPS

from sim_ops import AggregateBlock


class _SampleLayer(nn.nn.Module):
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
        'conv_3x1_1x3',
        'conv_7x1_1x7',
    ]

    def __init__(
            self,
            inplanes: int,
            norm_layer: Optional[Callable[..., nn.nn.Module]] = None
    ) -> None:
        super(_SampleLayer, self).__init__()
        if norm_layer is None:
            norm_layer = nn.nn.BatchNorm2d

        # extract feature from low dimension
        self.paths = nn.nn.ModuleList()
        for _ in range(3):
            self.paths.append(nn.nn.Sequential(
                nn.nn.Conv2d(inplanes, inplanes//4, kernel_size=1, stride=1),
                nn.LayerChoice([OPS[op](inplanes//4, 1, True) for op in self.SAMPLE_OPS]),
                norm_layer(inplanes//4),
                nn.nn.Relu()
            ))
        # feature aggregation
        self.agg_node = nn.nn.Sequential(
            norm_layer(inplanes//4),
            nn.nn.Conv2d(inplanes//4, inplanes, kernel_size=1, stride=1),
            norm_layer(inplanes),
            nn.nn.ReLU()
        )

    def forward(self, x: Tensor) -> Tensor:
        out = None
        for idx, _ in enumerate(self.paths):
            if out is None:
                out = self.paths[idx](x[idx])
            else:
                out += self.paths[idx](x[idx])
        # maybe division is needed
        out = self.agg_node(out/3)
        return out


class SampleBlock(torch.nn.ModuleDict):
    def __init__(
        self,
        num_layers: int,
        inplanes: int,
        outplanes: int,
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