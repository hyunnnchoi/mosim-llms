"""Chakra execution trace capture utilities."""

import os
import torch
import torch.profiler as profiler
from typing import Optional, Callable
from pathlib import Path


class ChakraTracer:
    """
    Chakra Execution Trace 캡처를 위한 래퍼 클래스.
    
    PyTorch Profiler를 사용하여 compute, memory, communication 작업을 추적하고
    Chakra ET 파일 형식으로 내보냅니다.
    """
    
    def __init__(
        self,
        output_dir: str = "./outputs",
        trace_name: str = "trace",
        enabled: bool = True,
        wait_steps: int = 2,
        warmup_steps: int = 2,
        active_steps: int = 6,
        record_shapes: bool = True,
        profile_memory: bool = True,
        with_stack: bool = True,
        with_flops: bool = True
    ):
        """
        Args:
            output_dir: 출력 디렉토리
            trace_name: Trace 파일 이름
            enabled: Profiling 활성화 여부
            wait_steps: Profiling 시작 전 대기 스텝
            warmup_steps: Warmup 스텝
            active_steps: 실제 profiling 스텝
            record_shapes: Tensor shape 기록
            profile_memory: Memory profiling 활성화
            with_stack: Python stack trace 포함
            with_flops: FLOPs 계산 포함
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.trace_name = trace_name
        self.enabled = enabled
        
        if not self.enabled:
            self.profiler = None
            return
        
        # PyTorch Profiler 설정
        activities = [
            profiler.ProfilerActivity.CPU,
        ]
        
        if torch.cuda.is_available():
            activities.append(profiler.ProfilerActivity.CUDA)
        
        self.profiler = profiler.profile(
            activities=activities,
            schedule=profiler.schedule(
                wait=wait_steps,
                warmup=warmup_steps,
                active=active_steps,
                repeat=1
            ),
            on_trace_ready=self._trace_handler,
            record_shapes=record_shapes,
            profile_memory=profile_memory,
            with_stack=with_stack,
            with_flops=with_flops,
        )
    
    def _trace_handler(self, prof):
        """Trace 준비 완료 시 호출되는 핸들러"""
        # Chrome trace 형식으로 저장
        trace_path = self.output_dir / f"{self.trace_name}_chrome.json"
        prof.export_chrome_trace(str(trace_path))
        print(f"[ChakraTracer] Chrome trace saved to {trace_path}")
        
        # Stacks 형식으로 저장
        stacks_path = self.output_dir / f"{self.trace_name}_stacks.txt"
        with open(stacks_path, "w") as f:
            f.write(prof.key_averages(group_by_stack_n=5).table(
                sort_by="self_cuda_time_total", row_limit=50
            ))
        print(f"[ChakraTracer] Stack trace saved to {stacks_path}")
        
        # TODO: Chakra ET 형식으로 변환
        # 현재는 Chrome trace 형식으로 저장하고,
        # 추후 Chakra 도구를 사용하여 ET 형식으로 변환 가능
        print(f"[ChakraTracer] To convert to Chakra ET format, use:")
        print(f"  chakra_converter --input {trace_path} --output {self.output_dir}/{self.trace_name}.et")
    
    def __enter__(self):
        """Context manager 진입"""
        if self.profiler is not None:
            self.profiler.__enter__()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager 종료"""
        if self.profiler is not None:
            self.profiler.__exit__(exc_type, exc_val, exc_tb)
    
    def step(self):
        """Profiling step (각 iteration마다 호출)"""
        if self.profiler is not None:
            self.profiler.step()
    
    def start(self):
        """Profiling 시작"""
        if self.profiler is not None:
            self.profiler.start()
    
    def stop(self):
        """Profiling 종료"""
        if self.profiler is not None:
            self.profiler.stop()


def profile_training(
    train_fn: Callable,
    output_dir: str = "./outputs",
    trace_name: str = "trace",
    enabled: bool = True,
    **profiler_kwargs
):
    """
    Training 함수를 profiling하는 데코레이터.
    
    Args:
        train_fn: Training 함수
        output_dir: 출력 디렉토리
        trace_name: Trace 이름
        enabled: Profiling 활성화 여부
        **profiler_kwargs: ChakraTracer에 전달할 추가 인자
    
    Example:
        @profile_training(output_dir="./outputs", trace_name="gpt2_train")
        def train_one_epoch(model, dataloader, optimizer):
            ...
    """
    def wrapper(*args, **kwargs):
        tracer = ChakraTracer(
            output_dir=output_dir,
            trace_name=trace_name,
            enabled=enabled,
            **profiler_kwargs
        )
        
        with tracer:
            result = train_fn(*args, **kwargs, tracer=tracer)
        
        return result
    
    return wrapper
