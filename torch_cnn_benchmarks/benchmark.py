"""Command-line entry point for PyTorch CNN benchmarks."""

from __future__ import annotations

import argparse
import json
import os
from contextlib import contextmanager
from typing import Any, Dict, Optional

try:
    import torch
except ModuleNotFoundError as err:  # pragma: no cover - dependency guard
    raise ModuleNotFoundError(
        "PyTorch가 설치되어 있지 않습니다. 'pip install torch torchvision'으로 설치해 주세요."
    ) from err

# Setup Python path for imports
import sys
from pathlib import Path
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

# import submodules
import torch_datasets
import models, trainer

# Import ChakraTracer from parent utils directory
from utils.chakra_tracer import ChakraTracer


@contextmanager
def _null_context():
    """Null context manager for when tracing is disabled"""
    yield


def _parse_args(argv: Optional[Any] = None) -> argparse.Namespace:
    available_models = list(models.list_models())
    parser = argparse.ArgumentParser(description="PyTorch CNN Benchmark")
    parser.add_argument("--model", type=str, choices=available_models, required=True, help="사용할 모델 이름")
    parser.add_argument("--dataset", type=str, choices=["cifar10", "cifar100", "imagenet", "synthetic"], help="데이터셋 선택")
    parser.add_argument("--data-dir", type=str, default=None, help="데이터셋 경로")
    parser.add_argument("--batch-size", type=int, default=128, help="GPU/프로세스당 배치 크기")
    parser.add_argument("--val-batch-size", type=int, default=None, help="검증 배치 크기")
    parser.add_argument("--epochs", type=int, default=90, help="학습 epoch 수")
    parser.add_argument("--learning-rate", "--lr", dest="lr", type=float, default=0.1, help="초기 학습률")
    parser.add_argument("--momentum", type=float, default=0.9, help="SGD 모멘텀")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="가중치 감쇠")
    parser.add_argument("--lr-scheduler", choices=["none", "step", "cosine"], default="cosine", help="학습률 스케줄러")
    parser.add_argument("--lr-step-size", type=int, default=30, help="StepLR step size")
    parser.add_argument("--lr-gamma", type=float, default=0.1, help="StepLR 감쇠 비율")
    parser.add_argument("--grad-accum-steps", type=int, default=1, help="gradient accumulation step 수")
    parser.add_argument("--clip-grad", type=float, default=None, help="gradient clipping 최대 norm")
    parser.add_argument("--log-interval", type=int, default=50, help="로그 출력 주기 (step)" )
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader worker 수")
    parser.add_argument("--seed", type=int, default=42, help="난수 시드")
    parser.add_argument("--deterministic", action="store_true", help="Deterministic CUDNN 모드 사용")
    parser.add_argument("--mixed-precision", action="store_true", help="torch.cuda.amp 사용 여부")
    parser.add_argument("--synthetic", action="store_true", help="합성 데이터로 학습")
    parser.add_argument("--num-classes", type=int, default=None, help="클래스 수 지정 (자동 추론 가능)")
    parser.add_argument("--image-size", type=int, default=None, help="입력 이미지 크기")
    parser.add_argument("--backend", choices=["nccl", "gloo"], default=None, help="분산 학습 backend")
    parser.add_argument("--sync-bn", action="store_true", help="SyncBatchNorm 사용")
    parser.add_argument("--aux-logits", action="store_true", help="보조 로짓 사용 (GoogLeNet/Inception)" )
    parser.add_argument("--output", type=str, default=None, help="결과를 JSON으로 저장할 경로")
    parser.add_argument("--no-eval", action="store_true", help="검증 단계 생략")
    parser.add_argument("--resume", type=str, default=None, help="체크포인트 경로")
    parser.add_argument("--resume-strict", action="store_true", help="체크포인트 로드시 strict 모드")

    # Chakra Tracing arguments
    parser.add_argument("--enable-tracing", action="store_true", help="Chakra 트레이스 캡처 활성화")
    parser.add_argument("--trace-name", type=str, default=None, help="트레이스 파일 이름 (기본값: {model}_{num_gpus}gpu)")
    parser.add_argument("--trace-output-dir", type=str, default="./outputs", help="트레이스 출력 디렉토리")
    parser.add_argument("--trace-wait-steps", type=int, default=0, help="프로파일링 대기 step (권장: 0)")
    parser.add_argument("--trace-warmup-steps", type=int, default=0, help="프로파일링 워밍업 step (권장: 0)")
    parser.add_argument("--trace-active-steps", type=int, default=1, help="프로파일링 활성 step (권장: 1)")

    return parser.parse_args(argv)


def _build_scheduler(name: str, optimizer: torch.optim.Optimizer, epochs: int, args: argparse.Namespace) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    if name == "none":
        return None
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    raise ValueError(f"Unknown scheduler '{name}'")


def _load_checkpoint(path: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer, scheduler: Optional[torch.optim.lr_scheduler._LRScheduler], strict: bool) -> Dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=strict)
    if "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])
    return checkpoint


def main(argv: Optional[Any] = None) -> None:
    args = _parse_args(argv)

    model_defaults = models.get_model_defaults(args.model)
    if args.dataset is None:
        args.dataset = model_defaults["dataset"]
    if args.num_classes is None:
        args.num_classes = model_defaults["num_classes"]
    if args.image_size is None:
        args.image_size = model_defaults["image_size"]

    use_cuda = torch.cuda.is_available()
    backend = args.backend or ("nccl" if use_cuda else "gloo")

    dist_state = trainer.setup_distributed(backend=backend)
    trainer.seed_everything(args.seed, deterministic=args.deterministic)

    if use_cuda:
        device = torch.device("cuda", dist_state.local_rank if dist_state.is_initialized else 0)
    else:
        device = torch.device("cpu")

    train_loader, val_loader, dataset_info, train_sampler = torch_datasets.create_dataloaders(
        dataset=args.dataset,
        data_dir=None if args.dataset == "synthetic" else args.data_dir,
        batch_size=args.batch_size,
        val_batch_size=args.val_batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        distributed=dist_state.is_initialized,
        synthetic=args.synthetic or args.dataset == "synthetic",
        num_classes=args.num_classes,
    )

    if args.num_classes is None:
        args.num_classes = dataset_info.num_classes

    model = models.create_model(
        name=args.model,
        num_classes=args.num_classes,
        aux_logits=args.aux_logits,
    )
    model.to(device)

    if dist_state.is_initialized:
        model = trainer.wrap_ddp(model, dist_state, use_sync_bn=args.sync_bn, device=device)

    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    scheduler = _build_scheduler(args.lr_scheduler, optimizer, args.epochs, args)

    scaler = torch.cuda.amp.GradScaler() if args.mixed_precision and use_cuda else None

    # Initialize ChakraTracer if tracing is enabled (all ranks)
    tracer = None
    if args.enable_tracing:
        # Default trace name: {model}_{num_gpus}gpu
        if args.trace_name is None:
            args.trace_name = f"{args.model}_{dist_state.world_size}gpu"

        tracer = ChakraTracer(
            output_dir=args.trace_output_dir,
            trace_name=args.trace_name,
            enabled=True,
            wait_steps=args.trace_wait_steps,
            warmup_steps=args.trace_warmup_steps,
            active_steps=args.trace_active_steps,
            rank=dist_state.global_rank,
            world_size=dist_state.world_size,
        )

        if trainer.is_main_process(dist_state):
            print(f"\nChakra tracing enabled:")
            print(f"  Trace name: {args.trace_name}")
            print(f"  Output dir: {args.trace_output_dir}")
            print(f"  Schedule: wait={args.trace_wait_steps}, warmup={args.trace_warmup_steps}, active={args.trace_active_steps}")

    benchmark_trainer = trainer.Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        state=dist_state,
        scaler=scaler,
        scheduler=scheduler,
        log_interval=args.log_interval,
        grad_accum_steps=args.grad_accum_steps,
        autocast_enabled=args.mixed_precision and use_cuda,
        autocast_dtype=torch.float16 if args.mixed_precision else None,
        max_norm=args.clip_grad,
        tracer=tracer,
    )

    if args.resume is not None and os.path.isfile(args.resume):
        checkpoint = _load_checkpoint(args.resume, model, optimizer, scheduler, args.resume_strict)
        if trainer.is_main_process(dist_state):
            print(f"Resumed from checkpoint '{args.resume}' (epoch={checkpoint.get('epoch')})")

    history: Dict[str, Any] = {
        "train": [],
        "val": [],
        "config": vars(args),
        "world_size": dist_state.world_size,
        "global_batch_size": args.batch_size * dist_state.world_size,
    }

    # Wrap training loop with tracer context if enabled
    training_context = tracer if tracer is not None else _null_context()

    try:
        with training_context:
            for epoch in range(1, args.epochs + 1):
                if dist_state.is_initialized and train_sampler is not None:
                    train_sampler.set_epoch(epoch)

                train_metrics = benchmark_trainer.train_one_epoch(train_loader, epoch)
                history["train"].append({"epoch": epoch, **train_metrics})

                if not args.no_eval:
                    val_metrics = benchmark_trainer.evaluate(val_loader)
                    if val_metrics:
                        history["val"].append({"epoch": epoch, **val_metrics})

                if scheduler is not None:
                    scheduler.step()

                if trainer.is_main_process(dist_state):
                    summary = {"epoch": epoch, **train_metrics}
                    if not args.no_eval and history["val"]:
                        summary.update(history["val"][-1])
                    print(json.dumps(summary))

    finally:
        if trainer.is_main_process(dist_state) and args.output:
            output_dir = os.path.dirname(args.output)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(history, f, indent=2)

        trainer.cleanup_distributed()


if __name__ == "__main__":
    main()

