"""DenseNet variants for CIFAR-style inputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


def _num_layers_per_block(depth: int) -> int:
    if (depth - 4) % 3 != 0:
        raise ValueError(
            f"DenseNet depth must satisfy (depth - 4) % 3 == 0, but got depth={depth}."
        )
    return (depth - 4) // 3


class _DenseLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        growth_rate: int,
        bn_size: int = 4,
        drop_rate: float = 0.0,
    ) -> None:
        super().__init__()
        inter_channels = bn_size * growth_rate

        self.norm1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, inter_channels, kernel_size=1, stride=1, bias=False)

        self.norm2 = nn.BatchNorm2d(inter_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(inter_channels, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)

        self.drop_rate = drop_rate

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        out = self.conv1(self.relu1(self.norm1(x)))
        out = self.conv2(self.relu2(self.norm2(out)))
        if self.drop_rate > 0.0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        return torch.cat([x, out], 1)


class _DenseBlock(nn.Module):
    def __init__(
        self,
        num_layers: int,
        in_channels: int,
        growth_rate: int,
        bn_size: int = 4,
        drop_rate: float = 0.0,
    ) -> None:
        super().__init__()
        layers = []
        channels = in_channels
        for _ in range(num_layers):
            layer = _DenseLayer(channels, growth_rate, bn_size=bn_size, drop_rate=drop_rate)
            layers.append(layer)
            channels += growth_rate
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        for layer in self.layers:
            x = layer(x)
        return x


class _Transition(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.norm = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.conv(self.relu(self.norm(x)))
        x = self.pool(x)
        return x


@dataclass(frozen=True)
class DenseNetConfig:
    depth: int
    growth_rate: int
    bn_size: int = 4
    theta: float = 0.5
    drop_rate: float = 0.0


class DenseNetCifar(nn.Module):
    """DenseNet architecture tailored for CIFAR-style inputs (32x32)."""

    def __init__(self, config: DenseNetConfig, num_classes: int = 10) -> None:
        super().__init__()
        num_layers_per_block = _num_layers_per_block(config.depth)

        init_channels = 2 * config.growth_rate
        current_channels = init_channels

        self.features = nn.Sequential()
        self.features.add_module(
            "conv0",
            nn.Conv2d(3, init_channels, kernel_size=3, stride=1, padding=1, bias=False),
        )

        block_configs = []
        for block_idx in range(3):
            block = _DenseBlock(
                num_layers=num_layers_per_block,
                in_channels=current_channels,
                growth_rate=config.growth_rate,
                bn_size=config.bn_size,
                drop_rate=config.drop_rate,
            )
            self.features.add_module(f"denseblock{block_idx + 1}", block)
            current_channels = current_channels + num_layers_per_block * config.growth_rate
            block_configs.append(current_channels)

            if block_idx != 2:
                out_channels = int(current_channels * config.theta)
                transition = _Transition(current_channels, out_channels)
                self.features.add_module(f"transition{block_idx + 1}", transition)
                current_channels = out_channels

        self.features.add_module("norm_final", nn.BatchNorm2d(current_channels))
        self.features.add_module("relu_final", nn.ReLU(inplace=True))

        self.classifier = nn.Linear(current_channels, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        features = self.features(x)
        out = F.adaptive_avg_pool2d(features, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out


def densenet40_k12(num_classes: int = 10, drop_rate: float = 0.0) -> DenseNetCifar:
    return DenseNetCifar(DenseNetConfig(depth=40, growth_rate=12, drop_rate=drop_rate), num_classes)


def densenet100_k12(num_classes: int = 10, drop_rate: float = 0.0) -> DenseNetCifar:
    return DenseNetCifar(DenseNetConfig(depth=100, growth_rate=12, drop_rate=drop_rate), num_classes)


DENSENET_FACTORIES: Dict[str, nn.Module] = {
    "densenet40_k12": densenet40_k12,
    "densenet100_k12": densenet100_k12,
}


__all__ = [
    "DenseNetCifar",
    "DenseNetConfig",
    "densenet40_k12",
    "densenet100_k12",
    "DENSENET_FACTORIES",
]

