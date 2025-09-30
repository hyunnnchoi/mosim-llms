"""GPT-2 training script with DDP support and Chakra tracing."""

import os
import sys
import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import get_linear_schedule_with_warmup
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from gpt2.config import GPT2Config
from gpt2.model import GPT2Model
from utils.distributed import (
    setup_distributed,
    cleanup_distributed,
    is_main_process,
    reduce_value
)
from utils.data_utils import get_dataloaders
from utils.chakra_tracer import ChakraTracer


def train_one_epoch(
    model: nn.Module,
    train_loader,
    optimizer,
    scheduler,
    epoch: int,
    config: GPT2Config,
    tracer: ChakraTracer = None
):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    step = 0
    
    # Progress bar (only on main process)
    if is_main_process():
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    else:
        pbar = train_loader
    
    for batch_idx, batch in enumerate(pbar):
        # Move to device
        input_ids = batch["input_ids"].cuda()
        attention_mask = batch["attention_mask"].cuda()
        labels = batch["labels"].cuda()
        
        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        loss = outputs["loss"]
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        
        # Optimizer step
        optimizer.step()
        scheduler.step()
        
        # Chakra profiling step
        if tracer is not None:
            tracer.step()
        
        # Reduce loss across all processes
        loss_val = reduce_value(loss, average=True)
        total_loss += loss_val
        step += 1
        
        # Logging
        if is_main_process() and batch_idx % config.logging_steps == 0:
            pbar.set_postfix({
                "loss": f"{loss_val:.4f}",
                "lr": f"{scheduler.get_last_lr()[0]:.2e}"
            })
    
    avg_loss = total_loss / step
    return avg_loss


def evaluate(model: nn.Module, val_loader, config: GPT2Config):
    """Evaluate the model."""
    model.eval()
    total_loss = 0
    step = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating", disable=not is_main_process()):
            input_ids = batch["input_ids"].cuda()
            attention_mask = batch["attention_mask"].cuda()
            labels = batch["labels"].cuda()
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs["loss"]
            
            # Reduce loss
            loss_val = reduce_value(loss, average=True)
            total_loss += loss_val
            step += 1
    
    avg_loss = total_loss / step
    return avg_loss


def train(config: GPT2Config):
    """Main training function."""
    # Setup distributed training
    rank, world_size = setup_distributed(backend=config.backend)
    
    if is_main_process():
        print(f"Training GPT-2 on SQuAD dataset")
        print(f"Number of GPUs: {world_size}")
        print(f"Batch size per GPU: {config.batch_size}")
        print(f"Effective batch size: {config.batch_size * world_size}")
    
    # Create dataloaders
    train_loader, val_loader, tokenizer = get_dataloaders(
        model_type="gpt2",
        tokenizer_name=config.model_name,
        batch_size=config.batch_size,
        max_length=config.max_seq_length,
        num_workers=config.num_workers,
        use_distributed=(world_size > 1),
        cache_dir=config.cache_dir
    )
    
    # Create model
    model = GPT2Model(config)
    model = model.cuda()
    
    # Wrap with DDP if using multiple GPUs
    if world_size > 1:
        model = DDP(model, device_ids=[rank], output_device=rank)
    
    # Optimizer and scheduler
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": config.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    
    optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate)
    
    total_steps = len(train_loader) * config.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=total_steps
    )
    
    # Chakra tracer (only on main process)
    tracer = None
    if config.enable_tracing and is_main_process():
        tracer = ChakraTracer(
            output_dir=config.trace_output_dir,
            trace_name=config.trace_name,
            enabled=True,
            wait_steps=config.trace_wait_steps,
            warmup_steps=config.trace_warmup_steps,
            active_steps=config.trace_active_steps
        )
    
    # Training loop
    if is_main_process():
        print("\n" + "="*50)
        print("Starting training...")
        print("="*50 + "\n")
    
    best_val_loss = float("inf")
    
    for epoch in range(config.num_epochs):
        # Set epoch for distributed sampler
        if world_size > 1:
            train_loader.sampler.set_epoch(epoch)
        
        # Train
        if tracer is not None:
            with tracer:
                train_loss = train_one_epoch(
                    model, train_loader, optimizer, scheduler, epoch, config, tracer
                )
        else:
            train_loss = train_one_epoch(
                model, train_loader, optimizer, scheduler, epoch, config, tracer
            )
        
        # Evaluate
        val_loss = evaluate(model, val_loader, config)
        
        if is_main_process():
            print(f"\nEpoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_path = Path(config.save_dir) / "best_model"
                save_path.mkdir(parents=True, exist_ok=True)
                
                if isinstance(model, DDP):
                    model.module.save_pretrained(str(save_path))
                else:
                    model.save_pretrained(str(save_path))
                
                print(f"Saved best model to {save_path}")
    
    # Cleanup
    cleanup_distributed()
    
    if is_main_process():
        print("\n" + "="*50)
        print("Training completed!")
        print("="*50 + "\n")


def main():
    """Entry point."""
    parser = argparse.ArgumentParser(description="GPT-2 Pretraining on SQuAD")
    
    # Model
    parser.add_argument("--model-name", type=str, default="gpt2", help="Model name")
    parser.add_argument("--max-seq-length", type=int, default=512, help="Max sequence length")
    
    # Training
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size per GPU")
    parser.add_argument("--learning-rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--num-epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--num-gpus", type=int, default=1, choices=[1, 2, 8], help="Number of GPUs")
    
    # Chakra tracing
    parser.add_argument("--enable-tracing", action="store_true", help="Enable Chakra tracing")
    parser.add_argument("--trace-output-dir", type=str, default="./outputs", help="Trace output directory")
    parser.add_argument("--trace-name", type=str, default="gpt2_trace", help="Trace name")
    
    # Paths
    parser.add_argument("--save-dir", type=str, default="./checkpoints/gpt2", help="Save directory")
    parser.add_argument("--cache-dir", type=str, default=None, help="Cache directory")
    
    args = parser.parse_args()
    
    # Create config
    config = GPT2Config(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        num_gpus=args.num_gpus,
        enable_tracing=args.enable_tracing,
        trace_output_dir=args.trace_output_dir,
        trace_name=args.trace_name,
        save_dir=args.save_dir,
        cache_dir=args.cache_dir,
    )
    
    # Train
    train(config)


if __name__ == "__main__":
    main()
