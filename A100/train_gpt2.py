"""GPT-2 DDP training for A100 interference experiments.

Usage (launched by run_solo.sh / run_pair.sh via torchrun):
    torchrun --nproc_per_node=4 --master_port=29500 \
        A100/train_gpt2.py --mode solo --total-steps 100
"""

import os
import sys
import argparse
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gpt2.model import GPT2Model
from gpt2.config import GPT2Config
from utils.data_utils import get_dataloaders
from A100.config import MODEL_CONFIGS, DEFAULT_TOTAL_STEPS, DEFAULT_WARMUP_STEPS
from A100.metrics import ExperimentMetrics


def parse_args():
    p = argparse.ArgumentParser(description="GPT-2 training — A100 interference experiment")
    p.add_argument("--batch-size", type=int, default=None, help="Override batch size per GPU")
    p.add_argument("--total-steps", type=int, default=DEFAULT_TOTAL_STEPS)
    p.add_argument("--warmup-steps", type=int, default=DEFAULT_WARMUP_STEPS)
    p.add_argument("--max-seq-length", type=int, default=512)
    p.add_argument("--learning-rate", type=float, default=None)
    p.add_argument("--mode", type=str, choices=["solo", "pair"], required=True)
    p.add_argument("--partner", type=str, default=None)
    p.add_argument("--output-dir", type=str, default="./A100/results")
    p.add_argument("--num-workers", type=int, default=4)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = MODEL_CONFIGS["gpt2"]

    batch_size = args.batch_size or cfg["batch_size"]
    lr = args.learning_rate or cfg["learning_rate"]

    # ── DDP init ──
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    visible_gpus = [int(g) for g in os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3").split(",")]

    if rank == 0:
        print(f"[GPT2] mode={args.mode}, partner={args.partner}, "
              f"world_size={world_size}, batch_size={batch_size}, gpus={visible_gpus}")

    # ── Model ──
    gpt2_cfg = GPT2Config(model_name="gpt2", batch_size=batch_size,
                          max_seq_length=args.max_seq_length, learning_rate=lr)
    model = GPT2Model(gpt2_cfg).to(device)
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])

    # ── Data ──
    train_loader, _, _ = get_dataloaders(
        model_type="gpt2", tokenizer_name="gpt2",
        batch_size=batch_size, max_length=args.max_seq_length,
        num_workers=args.num_workers, use_distributed=(world_size > 1),
    )

    # ── Optimizer / Scheduler ──
    no_decay = ["bias", "LayerNorm.weight"]
    param_groups = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": 0.01},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    optimizer = AdamW(param_groups, lr=lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=50,
                                                num_training_steps=args.total_steps)

    # ── Metrics collector ──
    metrics = ExperimentMetrics(
        model="gpt2", mode=args.mode, partner=args.partner, gpus=visible_gpus,
        batch_size=batch_size, total_steps=args.total_steps, warmup_steps=args.warmup_steps,
    )

    # ── Training loop ──
    model.train()
    data_iter = iter(train_loader)
    e2e_start = time.perf_counter()

    for step in range(args.total_steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            if world_size > 1:
                train_loader.sampler.set_epoch(step)
            data_iter = iter(train_loader)
            batch = next(data_iter)

        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        torch.cuda.synchronize()
        t0 = time.perf_counter()

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs["loss"]

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        torch.cuda.synchronize()
        iter_time = time.perf_counter() - t0

        loss_val = loss.detach().item()

        if step >= args.warmup_steps:
            metrics.iter_times.append(iter_time)
            metrics.losses.append(loss_val)

        if rank == 0 and (step % 10 == 0 or step == args.total_steps - 1):
            tag = " (warmup)" if step < args.warmup_steps else ""
            print(f"  [GPT2] step {step:>4d}/{args.total_steps} | "
                  f"loss={loss_val:.4f} | iter={iter_time:.4f}s{tag}")

    torch.cuda.synchronize()
    metrics.end_to_end_sec = time.perf_counter() - e2e_start
    metrics.measured_steps = len(metrics.iter_times)
    metrics.gpu_memory_allocated_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    metrics.gpu_memory_reserved_mb = torch.cuda.max_memory_reserved(device) / (1024 ** 2)

    if rank == 0:
        metrics.save(args.output_dir)
        print(f"[GPT2] Done. E2E={metrics.end_to_end_sec:.2f}s | "
              f"iter_mean={metrics.iter_mean:.4f}s | iter_std={metrics.iter_std:.4f}s | "
              f"throughput={metrics.throughput:.1f} samples/s")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
