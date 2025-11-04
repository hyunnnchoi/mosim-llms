"""Training utilities for PyTorch CNN benchmarks."""

from __future__ import annotations

import datetime as dt
import os
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


@dataclass
class DistributedState:
    is_initialized: bool
    world_size: int
    global_rank: int
    local_rank: int


def setup_distributed(backend: str = "nccl", timeout_seconds: int = 1800) -> DistributedState:
    if dist.is_available() and dist.is_initialized():
        return DistributedState(True, dist.get_world_size(), dist.get_rank(), int(os.environ.get("LOCAL_RANK", 0)))

    required_env = {"RANK", "WORLD_SIZE"}
    if not required_env.issubset(os.environ.keys()):
        return DistributedState(False, 1, 0, 0)

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    timeout = dt.timedelta(seconds=timeout_seconds)
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size, timeout=timeout)
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return DistributedState(True, world_size, rank, local_rank)


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(state: DistributedState) -> bool:
    return state.global_rank == 0


def barrier(state: DistributedState) -> None:
    if state.is_initialized:
        dist.barrier()


def seed_everything(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
        torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]


def accuracy(output: torch.Tensor, target: torch.Tensor, topk: Sequence[int] = (1,)) -> Sequence[torch.Tensor]:
    maxk = min(max(topk), output.size(1))
    batch_size = target.size(0)

    if output.dim() > 2:
        raise ValueError("Output tensor must be 2D (batch_size x num_classes).")

    _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    results = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        results.append(correct_k.mul_(100.0 / batch_size))
    return results


def reduce_tensor(tensor: torch.Tensor, state: DistributedState, average: bool = True) -> torch.Tensor:
    if not state.is_initialized:
        return tensor

    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    if average:
        rt /= state.world_size
    return rt


class MetricTracker:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.sum += value * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / max(self.count, 1)


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        device: torch.device,
        state: DistributedState,
        scaler: Optional[torch.cuda.amp.GradScaler] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        log_interval: int = 50,
        grad_accum_steps: int = 1,
        autocast_enabled: bool = False,
        autocast_dtype: Optional[torch.dtype] = None,
        max_norm: Optional[float] = None,
        tracer: Optional[Any] = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.state = state
        self.scaler = scaler
        self.scheduler = scheduler
        self.log_interval = log_interval
        self.grad_accum_steps = grad_accum_steps
        self.autocast_enabled = autocast_enabled
        self.autocast_dtype = autocast_dtype
        self.max_norm = max_norm
        self.tracer = tracer

        self._global_step = 0

    def _forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.model(inputs)
        if isinstance(outputs, tuple):
            # Handle inception-style auxiliary outputs (main output is first element)
            outputs = outputs[0]
        return outputs

    def train_one_epoch(self, dataloader: Iterable[Tuple[torch.Tensor, torch.Tensor]], epoch: int) -> Dict[str, float]:
        self.model.train()
        loss_meter = MetricTracker()
        top1_meter = MetricTracker()
        top5_meter = MetricTracker()

        start_time = time.time()

        self.optimizer.zero_grad(set_to_none=True)

        for batch_idx, (inputs, targets) in enumerate(dataloader, start=1):
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=self.autocast_enabled, dtype=self.autocast_dtype):
                outputs = self._forward(inputs)
                loss = self.criterion(outputs, targets)

            loss_value = loss.detach()

            if self.grad_accum_steps > 1:
                loss = loss / self.grad_accum_steps

            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            if (batch_idx % self.grad_accum_steps) == 0:
                if self.max_norm is not None:
                    if self.scaler is not None:
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_norm)
                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

            with torch.no_grad():
                top1, top5 = accuracy(outputs, targets, topk=(1, 5))

                loss_value = reduce_tensor(loss_value, self.state, average=True)
                top1 = reduce_tensor(top1, self.state, average=True)
                top5 = reduce_tensor(top5, self.state, average=True)

            loss_meter.update(loss_value.item(), inputs.size(0))
            top1_meter.update(top1.item(), inputs.size(0))
            top5_meter.update(top5.item(), inputs.size(0))

            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)

            if is_main_process(self.state) and batch_idx % self.log_interval == 0:
                elapsed = time.time() - start_time
                samples_per_sec = (self.log_interval * inputs.size(0) * self.state.world_size) / elapsed
                lr = self.optimizer.param_groups[0]["lr"]
                print(
                    f"Epoch {epoch} [{batch_idx}/{len(dataloader)}] "
                    f"Loss: {loss_meter.avg:.4f} | Top1: {top1_meter.avg:.2f}% | "
                    f"Top5: {top5_meter.avg:.2f}% | LR: {lr:.5f} | {samples_per_sec:.1f} img/s"
                )
                start_time = time.time()

            # Step the tracer (advances profiler schedule)
            if self.tracer is not None:
                self.tracer.step()

            self._global_step += 1

        metrics = {"train_loss": loss_meter.avg, "train_top1": top1_meter.avg, "train_top5": top5_meter.avg}
        return metrics

    @torch.no_grad()
    def evaluate(self, dataloader: Optional[Iterable[Tuple[torch.Tensor, torch.Tensor]]]) -> Dict[str, float]:
        if dataloader is None:
            return {}

        self.model.eval()
        loss_meter = MetricTracker()
        top1_meter = MetricTracker()
        top5_meter = MetricTracker()

        for inputs, targets in dataloader:
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            outputs = self._forward(inputs)
            loss = self.criterion(outputs, targets)
            top1, top5 = accuracy(outputs, targets, topk=(1, 5))

            loss = reduce_tensor(loss, self.state, average=True)
            top1 = reduce_tensor(top1, self.state, average=True)
            top5 = reduce_tensor(top5, self.state, average=True)

            loss_meter.update(loss.item(), inputs.size(0))
            top1_meter.update(top1.item(), inputs.size(0))
            top5_meter.update(top5.item(), inputs.size(0))

        return {"val_loss": loss_meter.avg, "val_top1": top1_meter.avg, "val_top5": top5_meter.avg}


def wrap_ddp(model: torch.nn.Module, state: DistributedState, use_sync_bn: bool, device: torch.device) -> torch.nn.Module:
    if not state.is_initialized:
        return model

    if use_sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = DDP(model.to(device), device_ids=[state.local_rank] if device.type == "cuda" else None)
    return model


__all__ = [
    "Trainer",
    "DistributedState",
    "setup_distributed",
    "cleanup_distributed",
    "is_main_process",
    "barrier",
    "seed_everything",
    "wrap_ddp",
]

