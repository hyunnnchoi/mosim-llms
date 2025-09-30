"""Distributed training utilities for Data Parallelism (similar to TensorFlow's MultiWorkerMirroredStrategy)."""

import os
import torch
import torch.distributed as dist


def setup_distributed(backend="nccl"):
    """
    Initialize distributed training environment.
    
    PyTorch DDP를 사용하여 TensorFlow의 MultiWorkerMirroredStrategy와 유사한
    Data Parallelism을 구현합니다.
    
    Args:
        backend: 'nccl' for GPU, 'gloo' for CPU
    
    Returns:
        rank: 현재 프로세스의 rank
        world_size: 전체 프로세스 수 (GPU 개수)
    """
    # 환경 변수에서 설정 가져오기
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
    else:
        # 단일 GPU 또는 비분산 모드
        rank = 0
        world_size = 1
        local_rank = 0
    
    if world_size > 1:
        # 분산 환경 초기화
        dist.init_process_group(backend=backend)
        
        # 각 프로세스에 GPU 할당
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        
        print(f"[Rank {rank}/{world_size}] Distributed training initialized")
    else:
        print("Single GPU/CPU mode")
    
    return rank, world_size


def cleanup_distributed():
    """분산 환경 정리"""
    if dist.is_initialized():
        dist.destroy_process_group()


def get_rank():
    """현재 프로세스의 rank 반환"""
    if dist.is_initialized():
        return dist.get_rank()
    return 0


def get_world_size():
    """전체 프로세스 수 반환"""
    if dist.is_initialized():
        return dist.get_world_size()
    return 1


def is_main_process():
    """메인 프로세스인지 확인 (rank 0)"""
    return get_rank() == 0


def reduce_value(value, average=True):
    """
    모든 프로세스의 값을 reduce (평균 또는 합)
    
    Args:
        value: reduce할 값 (tensor 또는 scalar)
        average: True면 평균, False면 합
    """
    world_size = get_world_size()
    if world_size < 2:
        return value
    
    with torch.no_grad():
        if not isinstance(value, torch.Tensor):
            value = torch.tensor(value).cuda()
        
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        
        if average:
            value = value / world_size
    
    return value.item() if value.numel() == 1 else value
