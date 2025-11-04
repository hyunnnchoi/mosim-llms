"""ResNet implementations for CIFAR-style datasets."""

from __future__ import annotations

import math
from typing import Callable, Dict, Type

import torch
import torch.nn as nn


def _conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample: nn.Module | None = None) -> None:
        super().__init__()
        self.conv1 = _conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = _conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class ResNetCifar(nn.Module):
    """ResNet for CIFAR (depth = 6n + 2)."""

    def __init__(self, depth: int, num_classes: int = 10, base_channels: int = 16) -> None:
        super().__init__()
        if (depth - 2) % 6 != 0:
            raise ValueError(f"ResNet depth should be 6n+2, but got depth={depth}.")

        n = (depth - 2) // 6
        block: Type[BasicBlock] = BasicBlock

        self.inplanes = base_channels
        self.conv1 = _conv3x3(3, base_channels)
        self.bn1 = nn.BatchNorm2d(base_channels)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, base_channels, n)
        self.layer2 = self._make_layer(block, base_channels * 2, n, stride=2)
        self.layer3 = self._make_layer(block, base_channels * 4, n, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(base_channels * 4 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, math.sqrt(2.0 / m.out_features))
                nn.init.zeros_(m.bias)

    def _make_layer(
        self,
        block: Type[BasicBlock],
        planes: int,
        blocks: int,
        stride: int = 1,
    ) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def resnet44(num_classes: int = 10) -> ResNetCifar:
    return ResNetCifar(depth=44, num_classes=num_classes)


def resnet110(num_classes: int = 10) -> ResNetCifar:
    return ResNetCifar(depth=110, num_classes=num_classes)


RESNET_CIFAR_FACTORIES: Dict[str, Callable[[int], nn.Module]] = {
    "resnet44": resnet44,
    "resnet110": resnet110,
}


__all__ = ["ResNetCifar", "resnet44", "resnet110", "RESNET_CIFAR_FACTORIES", "BasicBlock"]

