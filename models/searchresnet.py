from collections import OrderedDict

import torch
import torch.nn as tnn
import torch.nn.functional as F
from torch import Tensor
import nni.retiarii.nn.pytorch as nn
from typing import Type, Any, Callable, Union, List, Optional, Tuple

from models.sim_ops import ShortcutBlock


class ResNet(nn.Module):

    def __init__(
            self,
            block: Type[Union[ShortcutBlock]],
            layers,
            num_classes: int = 1000,
            zero_init_residual: bool = False,
            groups: int = 1,
            width_per_group: int = 64,
            replace_stride_with_dilation: Optional[List[bool]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        # self.relu1 = nn.LayerChoice([nn.ReLU(), nn.Identity()])
        self.relu1 = nn.ReLU()
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.relu2 = nn.ReLU()
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.relu3 = nn.ReLU()
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.relu4 = nn.ReLU()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block: Type[Union[ShortcutBlock]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = False
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = True

        layers = [nn.LayerChoice([block(self.inplanes, planes, 3, stride, downsample, self.groups,
                                        self.base_width, previous_dilation, norm_layer),
                                  block(self.inplanes, planes, 5, stride, downsample, self.groups,
                                        self.base_width, previous_dilation, norm_layer),
                                  block(self.inplanes, planes, 7, stride, downsample, self.groups,
                                        self.base_width, previous_dilation, norm_layer),
                                  ])]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(nn.LayerChoice([block(self.inplanes, planes, 3, groups=self.groups,
                                                base_width=self.base_width, dilation=self.dilation,
                                                norm_layer=norm_layer),
                                          block(self.inplanes, planes, 5, groups=self.groups,
                                                base_width=self.base_width, dilation=self.dilation,
                                                norm_layer=norm_layer),
                                          block(self.inplanes, planes, 7, groups=self.groups,
                                                base_width=self.base_width, dilation=self.dilation,
                                                norm_layer=norm_layer),
                                          ]))
            layers.append(nn.ReLU())
        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.relu1(x)
        x = self.layer2(x)
        x = self.relu2(x)
        x = self.layer3(x)
        x = self.relu3(x)
        x = self.layer4(x)
        x = self.relu4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def resnet18(pretrained=True, **kwargs: Any) -> ResNet:
    model = ResNet(ShortcutBlock, [2, 2, 2, 2], **kwargs)
    return model


def resnet34(pretrained=True, **kwargs: Any) -> ResNet:
    model = ResNet(ShortcutBlock, [3, 4, 6, 3], **kwargs)
    return model


def resnet101(pretrained=True, **kwargs: Any) -> ResNet:
    model = ResNet(ShortcutBlock, [3, 4, 23, 3], **kwargs)
    return model


def resnet152(pretrained=True, **kwargs: Any) -> ResNet:
    model = ResNet(ShortcutBlock, [3, 8, 36, 3], **kwargs)
    return model