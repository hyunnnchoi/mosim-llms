"""CNN DDP training for A100 interference experiments.

Supports all CNN models: resnet44, resnet110, resnet50, vgg16,
googlenet, inception3, densenet40_k12, densenet100_k12.

Usage:
    torchrun --nproc_per_node=4 --master_port=29500 \
        A100/train_cnn.py --model vgg16 --mode solo --total-steps 100
"""

import os
import sys
import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Dataset
import torchvision.transforms as transforms
import torchvision.datasets as tv_datasets

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from torch_cnn_benchmarks.models import create_model, get_model_defaults
from A100.config import MODEL_CONFIGS, DEFAULT_TOTAL_STEPS, DEFAULT_WARMUP_STEPS, CNN_MODELS
from A100.metrics import ExperimentMetrics


# ── Synthetic dataset for ImageNet-scale models ──

class SyntheticDataset(Dataset):
    """Generates random images + labels. No disk I/O overhead."""

    def __init__(self, image_size: int, num_classes: int, length: int = 50000):
        self.image_size = image_size
        self.num_classes = num_classes
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        img = torch.randn(3, self.image_size, self.image_size)
        label = torch.randint(0, self.num_classes, (1,)).item()
        return img, label


# ── Data loaders ──

def get_cifar10_loader(batch_size: int, image_size: int, num_workers: int,
                       distributed: bool):
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    dataset = tv_datasets.CIFAR10(root="./data", train=True, download=True,
                                  transform=transform)
    sampler = DistributedSampler(dataset, shuffle=True) if distributed else None
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                        shuffle=(sampler is None), num_workers=num_workers,
                        pin_memory=True, drop_last=True)
    return loader


def get_synthetic_loader(batch_size: int, image_size: int, num_classes: int,
                         num_workers: int, distributed: bool):
    dataset = SyntheticDataset(image_size=image_size, num_classes=num_classes)
    sampler = DistributedSampler(dataset, shuffle=True) if distributed else None
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                        shuffle=(sampler is None), num_workers=num_workers,
                        pin_memory=True, drop_last=True)
    return loader


def get_dataloader(model_name: str, batch_size: int, num_workers: int,
                   distributed: bool):
    cfg = MODEL_CONFIGS[model_name]
    dataset_name = cfg["dataset"]
    image_size = cfg["image_size"]
    num_classes = cfg["num_classes"]

    if dataset_name == "cifar10":
        return get_cifar10_loader(batch_size, image_size, num_workers, distributed)
    else:  # synthetic
        return get_synthetic_loader(batch_size, image_size, num_classes,
                                    num_workers, distributed)


# ── CLI ──

def parse_args():
    p = argparse.ArgumentParser(description="CNN training — A100 interference experiment")
    p.add_argument("--model", type=str, required=True, choices=CNN_MODELS)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--total-steps", type=int, default=DEFAULT_TOTAL_STEPS)
    p.add_argument("--warmup-steps", type=int, default=DEFAULT_WARMUP_STEPS)
    p.add_argument("--learning-rate", type=float, default=None)
    p.add_argument("--mode", type=str, choices=["solo", "pair"], required=True)
    p.add_argument("--partner", type=str, default=None)
    p.add_argument("--output-dir", type=str, default="./A100/results")
    p.add_argument("--num-workers", type=int, default=4)
    return p.parse_args()


# ── Main ──

def main():
    args = parse_args()
    model_name = args.model
    cfg = MODEL_CONFIGS[model_name]

    batch_size = args.batch_size or cfg["batch_size"]
    lr = args.learning_rate or cfg["learning_rate"]
    num_classes = cfg["num_classes"]

    # ── DDP init ──
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    visible_gpus = [int(g) for g in os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3").split(",")]

    tag = model_name.upper()
    if rank == 0:
        print(f"[{tag}] mode={args.mode}, partner={args.partner}, "
              f"world_size={world_size}, batch_size={batch_size}, "
              f"dataset={cfg['dataset']}, gpus={visible_gpus}")

    # ── Model ──
    model = create_model(model_name, num_classes=num_classes)
    model = model.to(device)
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])

    # ── Data ──
    train_loader = get_dataloader(model_name, batch_size, args.num_workers,
                                  distributed=(world_size > 1))

    # ── Optimizer ──
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                                weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    # ── Metrics ──
    metrics = ExperimentMetrics(
        model=model_name, mode=args.mode, partner=args.partner, gpus=visible_gpus,
        batch_size=batch_size, total_steps=args.total_steps, warmup_steps=args.warmup_steps,
    )

    # ── Training loop ──
    model.train()
    data_iter = iter(train_loader)
    e2e_start = time.perf_counter()

    for step in range(args.total_steps):
        try:
            inputs, targets = next(data_iter)
        except StopIteration:
            if world_size > 1:
                train_loader.sampler.set_epoch(step)
            data_iter = iter(train_loader)
            inputs, targets = next(data_iter)

        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        torch.cuda.synchronize()
        t0 = time.perf_counter()

        outputs = model(inputs)
        # Handle inception-style auxiliary outputs
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        loss = criterion(outputs, targets)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        torch.cuda.synchronize()
        iter_time = time.perf_counter() - t0

        loss_val = loss.detach().item()

        if step >= args.warmup_steps:
            metrics.iter_times.append(iter_time)
            metrics.losses.append(loss_val)

        if rank == 0 and (step % 10 == 0 or step == args.total_steps - 1):
            marker = " (warmup)" if step < args.warmup_steps else ""
            print(f"  [{tag}] step {step:>4d}/{args.total_steps} | "
                  f"loss={loss_val:.4f} | iter={iter_time:.4f}s{marker}")

    torch.cuda.synchronize()
    metrics.end_to_end_sec = time.perf_counter() - e2e_start
    metrics.measured_steps = len(metrics.iter_times)
    metrics.gpu_memory_allocated_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    metrics.gpu_memory_reserved_mb = torch.cuda.max_memory_reserved(device) / (1024 ** 2)

    if rank == 0:
        metrics.save(args.output_dir)
        print(f"[{tag}] Done. E2E={metrics.end_to_end_sec:.2f}s | "
              f"iter_mean={metrics.iter_mean:.4f}s | iter_std={metrics.iter_std:.4f}s | "
              f"throughput={metrics.throughput:.1f} samples/s")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
