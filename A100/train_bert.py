"""BERT DDP training for A100 interference experiments.

Usage (launched by run_solo.sh / run_pair.sh via torchrun):
    torchrun --nproc_per_node=4 --master_port=29500 \
        A100/train_bert.py --mode solo --total-steps 100
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

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bert.model import BERTModel
from bert.config import BERTConfig
from utils.data_utils import get_dataloaders
from A100.config import MODEL_CONFIGS, DEFAULT_TOTAL_STEPS, DEFAULT_WARMUP_STEPS
from A100.metrics import ExperimentMetrics
from A100.barrier import wait_for_partner, signal_done, should_stop


def parse_args():
    p = argparse.ArgumentParser(description="BERT training — A100 interference experiment")
    p.add_argument("--batch-size", type=int, default=None, help="Override batch size per GPU")
    p.add_argument("--total-steps", type=int, default=DEFAULT_TOTAL_STEPS)
    p.add_argument("--warmup-steps", type=int, default=DEFAULT_WARMUP_STEPS, help="Timing warmup (excluded from measurement)")
    p.add_argument("--max-seq-length", type=int, default=512)
    p.add_argument("--learning-rate", type=float, default=None)
    p.add_argument("--mode", type=str, choices=["solo", "pair"], required=True)
    p.add_argument("--partner", type=str, default=None, help="Name of co-located model (pair mode)")
    p.add_argument("--output-dir", type=str, default="./A100/results")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--barrier-dir", type=str, default=None)
    p.add_argument("--role", type=str, choices=["primary", "interferer"], default="primary",
                   help="primary: fixed steps + save results; interferer: run until primary is done")
    p.add_argument("--job-id", type=str, default=None,
                   help="Unique job ID for barrier signaling (defaults to model name)")
    p.add_argument("--partner-id", type=str, default=None,
                   help="Partner job ID for barrier signaling (defaults to partner name)")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = MODEL_CONFIGS["bert"]

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
        print(f"[BERT] mode={args.mode}, partner={args.partner}, "
              f"world_size={world_size}, batch_size={batch_size}, gpus={visible_gpus}")

    # ── Model ──
    bert_cfg = BERTConfig(model_name="bert-base-uncased", batch_size=batch_size,
                          max_seq_length=args.max_seq_length, learning_rate=lr)
    model = BERTModel(bert_cfg).to(device)
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])

    # ── Data ──
    train_loader, _, _ = get_dataloaders(
        model_type="bert", tokenizer_name="bert-base-uncased",
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
        model="bert", mode=args.mode, partner=args.partner, gpus=visible_gpus,
        batch_size=batch_size, total_steps=args.total_steps, warmup_steps=args.warmup_steps,
    )

    # ── Synchronize with partner job before starting ──
    job_id = args.job_id or "bert"
    partner_id = args.partner_id or args.partner
    wait_for_partner(args.mode, job_id, partner_id, args.barrier_dir, rank)

    # ── Training loop ──
    model.train()
    data_iter = iter(train_loader)
    e2e_start = time.perf_counter()
    is_primary = args.role == "primary"
    max_steps = args.total_steps if is_primary else 999999999

    step = 0
    while step < max_steps:
        # Interferer: check if primary is done
        if not is_primary and step > 0 and should_stop(args.barrier_dir, partner_id, rank):
            if rank == 0:
                print(f"  [BERT] interferer stopping at step {step} (primary is done)")
            break

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

        if is_primary and step >= args.warmup_steps:
            metrics.iter_times.append(iter_time)
            metrics.losses.append(loss_val)

        if rank == 0 and (step % 10 == 0 or (is_primary and step == args.total_steps - 1)):
            role_tag = "P" if is_primary else "I"
            marker = " (warmup)" if (is_primary and step < args.warmup_steps) else ""
            print(f"  [BERT/{role_tag}] step {step:>4d}/{args.total_steps if is_primary else '?'} | "
                  f"loss={loss_val:.4f} | iter={iter_time:.4f}s{marker}")

        step += 1

    torch.cuda.synchronize()

    # Primary: signal done and save results
    if is_primary:
        signal_done(args.barrier_dir, job_id, rank)
        metrics.end_to_end_sec = time.perf_counter() - e2e_start
        metrics.measured_steps = len(metrics.iter_times)
        metrics.gpu_memory_allocated_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        metrics.gpu_memory_reserved_mb = torch.cuda.max_memory_reserved(device) / (1024 ** 2)

        if rank == 0:
            metrics.save(args.output_dir)
            print(f"[BERT] Done. E2E={metrics.end_to_end_sec:.2f}s | "
                  f"iter_mean={metrics.iter_mean:.4f}s | iter_std={metrics.iter_std:.4f}s | "
                  f"throughput={metrics.throughput:.1f} samples/s")
    else:
        if rank == 0:
            print(f"[BERT] Interferer finished after {step} steps.")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
