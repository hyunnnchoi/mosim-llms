"""Dataset utilities for PyTorch CNN benchmarks."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler, RandomSampler
from torchvision import datasets as tv_datasets
from torchvision import transforms


_CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
_CIFAR10_STD = (0.2023, 0.1994, 0.2010)
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class SyntheticDataset(Dataset[Tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        num_samples: int,
        image_size: int,
        num_classes: int,
        channels: int = 3,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.num_samples = num_samples
        self.image_size = image_size
        self.num_classes = num_classes
        self.channels = channels
        self.dtype = dtype

    def __len__(self) -> int:  # type: ignore[override]
        return self.num_samples

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:  # type: ignore[override]
        data = torch.randn(
            self.channels,
            self.image_size,
            self.image_size,
            dtype=self.dtype,
        )
        target = torch.randint(0, self.num_classes, (1,), dtype=torch.int64)
        return data, target.squeeze(0)


@dataclass
class DatasetInfo:
    train_size: int
    val_size: int
    num_classes: int
    image_size: int


def _resolve_mean_std(dataset: str) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    if dataset in {"cifar10", "cifar100"}:
        return _CIFAR10_MEAN, _CIFAR10_STD
    if dataset == "imagenet":
        return _IMAGENET_MEAN, _IMAGENET_STD
    return (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)


def create_dataloaders(
    dataset: str,
    data_dir: Optional[str],
    batch_size: int,
    val_batch_size: Optional[int],
    num_workers: int,
    image_size: int,
    distributed: bool,
    synthetic: bool,
    num_classes: Optional[int],
    drop_last: bool = True,
) -> Tuple[DataLoader, Optional[DataLoader], DatasetInfo, Optional[DistributedSampler]]:
    if val_batch_size is None:
        val_batch_size = batch_size

    mean, std = _resolve_mean_std(dataset)

    if synthetic:
        if num_classes is None:
            raise ValueError("num_classes must be provided when using synthetic data.")
        train_dataset = SyntheticDataset(num_samples=10_000, image_size=image_size, num_classes=num_classes)
        val_dataset: Optional[Dataset[Tuple[torch.Tensor, torch.Tensor]]] = None
        train_sampler = None
        if distributed:
            train_sampler = DistributedSampler(train_dataset)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            shuffle=train_sampler is None,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=drop_last,
        )
        return train_loader, None, DatasetInfo(len(train_dataset), 0, num_classes, image_size), train_sampler

    if dataset == "cifar10":
        if data_dir is None:
            raise ValueError("CIFAR-10 requires --data-dir when not using synthetic data.")
        train_transforms = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        eval_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        train_dataset = tv_datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_transforms)
        val_dataset = tv_datasets.CIFAR10(root=data_dir, train=False, download=True, transform=eval_transforms)
        inferred_classes = 10
    elif dataset == "cifar100":
        if data_dir is None:
            raise ValueError("CIFAR-100 requires --data-dir when not using synthetic data.")
        train_transforms = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        eval_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        train_dataset = tv_datasets.CIFAR100(root=data_dir, train=True, download=True, transform=train_transforms)
        val_dataset = tv_datasets.CIFAR100(root=data_dir, train=False, download=True, transform=eval_transforms)
        inferred_classes = 100
    elif dataset == "imagenet":
        if data_dir is None:
            raise ValueError("ImageNet requires --data-dir pointing to the dataset root.")
        train_dir = os.path.join(data_dir, "train")
        val_dir = os.path.join(data_dir, "val")
        train_transforms = transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        eval_transforms = transforms.Compose(
            [
                transforms.Resize(int(image_size * 256 / 224)),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        train_dataset = tv_datasets.ImageFolder(train_dir, transform=train_transforms)
        val_dataset = tv_datasets.ImageFolder(val_dir, transform=eval_transforms)
        inferred_classes = len(train_dataset.classes)
    else:
        raise ValueError(f"Unsupported dataset '{dataset}'.")

    if num_classes is None:
        num_classes = inferred_classes

    if distributed:
        train_sampler = DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=train_sampler is None,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return (
        train_loader,
        val_loader,
        DatasetInfo(len(train_dataset), len(val_dataset) if val_dataset is not None else 0, num_classes, image_size),
        train_sampler,
    )


__all__ = ["create_dataloaders", "SyntheticDataset", "DatasetInfo"]

