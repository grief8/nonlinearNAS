import torch
from torch import Tensor
import nni.retiarii.nn.pytorch as nn
from typing import Type, Any, Callable, Union, List, Optional


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class ShortcutBlock(nn.Module):
    """
    This layer is about reusing nonlinear output
    """
    expansion: int = 1

    def __init__(
            self,
            inplanes: int,
            planes: int,
            kernel_size: int,
            stride: int = 1,
            downsample: bool = False,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(ShortcutBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        compensation = int((kernel_size - 3) / 2)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=kernel_size, stride=stride,
                               padding=dilation + compensation, groups=groups, bias=False, dilation=dilation)
        self.bn1 = norm_layer(planes)
        self.sim_relu = conv1x1(planes, planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=kernel_size, stride=1,
                               padding=1 + compensation, groups=1, bias=False, dilation=1)
        self.bn2 = norm_layer(planes)
        if downsample:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * self.expansion, kernel_size=kernel_size - 2, stride=stride, bias=False,
                          padding=compensation),
                norm_layer(planes * self.expansion),
            )
        else:
            self.downsample = None
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.sim_relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out = out + identity

        return out


class SampleBlock(torch.nn.ModuleDict):
    def __init__(
        self,
        num_layers: int,
        num_input_features: int,
        growth_rate: int,
    ) -> None:
        super(SampleBlock, self).__init__()
        for i in range(num_layers):
            layer = _SampleLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate
            )
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features: Tensor) -> Tensor:
        features = [init_features]
        for name, layer in self.items():
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)


class AggregateBlock(nn.Module):
    def __init__(
            self,
            inplanes: list,
            outplanes: int,
            kernel_size: int = 1,
            stride: int = 1,
            groups: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(AggregateBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.layers = nn.ModuleList()
        compensation = stride
        config = inplanes[:]
        config.reverse()
        for idx, inp in enumerate(config):
            self.layers.insert(0, nn.Sequential(
                nn.Conv2d(inp, 4 * outplanes, kernel_size=kernel_size, stride=compensation,
                          groups=groups, bias=False),
                norm_layer(4 * outplanes)
            ))
            # self.layers.append(nn.Sequential(
            #     nn.Conv2d(inp, 4 * outplanes, kernel_size=kernel_size, stride=compensation,
            #               groups=groups, bias=False),
            #     norm_layer(4 * outplanes)
            # ))
            compensation = 2 ** (idx)
        self.transition = nn.Sequential(
            nn.Conv2d(4 * outplanes, outplanes, kernel_size=3, stride=2,
                      padding=1, groups=groups, bias=False),
            norm_layer(outplanes),
            nn.ReLU(),
        )

    def forward(self, x: List) -> Tensor:
        out = None
        for idx, _ in enumerate(self.layers):
            # print(x[idx].shape)
            if out is None:
                out = self.layers[idx](x[idx])
            else:
                out += self.layers[idx](x[idx])

        out = self.transition(out)
        return out


class _SampleLayer(nn.Module):
    def __init__(
            self,
            inplanes: list,
            growth_rate: int,
            groups: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(_SampleLayer, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.layer = nn.Sequential(
            nn.Conv2d(inplanes, growth_rate, kernel_size=3, stride=1,
                      padding=1, groups=groups, bias=False),
            norm_layer(growth_rate),
            nn.ReLU(),
        )

    def forward(self, x: List[Tensor]) -> Tensor:
        out = torch.cat(x, 1)
        out = self.layer(out)
        return out
