"""Chakra execution trace capture utilities.

Chakra Workflow:
1. PyTorch Profiler -> kineto trace (Chrome JSON)
2. chakra_converter -> convert to Chakra ET (protobuf)

Note: PyTorch Profiler's Kineto trace already includes both host and device events,
so we skip chakra_trace_link and directly use chakra_converter.
"""

import os
import torch
import torch.profiler as profiler
from typing import Optional, Callable
from pathlib import Path


class ChakraTracer:
    """
    Chakra Execution Trace 캡처를 위한 래퍼 클래스.
    
    PyTorch Profiler를 사용하여 Kineto trace를 생성하고,
    chakra_converter를 통해 ET 파일로 변환합니다.
    
    Workflow:
        PyTorch Profiler -> Kineto trace -> chakra_converter -> .et file
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
        with_flops: bool = True,
        convert_to_et: bool = True,
        rank: int = 0
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
            convert_to_et: Chakra ET 형식으로 자동 변환 여부
            rank: Distributed training rank (for chakra_trace_link)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.trace_name = trace_name
        self.enabled = enabled
        self.convert_to_et = convert_to_et
        self.rank = rank
        
        if not self.enabled:
            self.profiler = None
            return
        
        # PyTorch Profiler 설정 (Kineto trace 생성)
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
    
    def _convert_to_chakra_et(self, kineto_trace_path: Path):
        """
        Kineto trace를 Chakra ET 파일로 변환
        
        Chakra 워크플로우:
        PyTorch Kineto trace -> PyTorchConverter -> .et (protobuf)
        
        Note: chakra.et_converter.pytorch.PyTorchConverter API를 사용하여
        Kineto JSON을 직접 Chakra ET 형식으로 변환합니다.
        """
        # 파일 경로 설정
        base_name = kineto_trace_path.stem  # _kineto.json 제외
        if base_name.endswith("_kineto"):
            base_name = base_name[:-7]  # "_kineto" 제거
        
        # ET 파일 경로
        et_path = self.output_dir / f"{base_name}.et"
        
        try:
            # Chakra PyTorchConverter를 사용하여 변환
            print(f"[ChakraTracer] Converting Kineto trace to Chakra ET format...")
            
            from chakra.et_converter.pytorch import PyTorchConverter
            
            converter = PyTorchConverter()
            converter.convert(
                input_filename=str(kineto_trace_path),
                output_filename=str(et_path)
            )
            
            print(f"[ChakraTracer] ✓ Chakra ET file saved to {et_path}")
            return True
                    
        except ImportError as e:
            print(f"[ChakraTracer] Error: Chakra converter not found: {e}")
            print(f"  Make sure Chakra is properly installed.")
            print(f"  pip install git+https://github.com/mlcommons/chakra.git")
            print(f"\n[ChakraTracer] Alternative: use convert_to_et.py script")
            print(f"  python convert_to_et.py")
            return False
        except Exception as e:
            print(f"[ChakraTracer] Warning: Failed to convert to Chakra ET: {e}")
            print(f"\n[ChakraTracer] Alternative: use convert_to_et.py script")
            print(f"  python convert_to_et.py")
            return False
    
    def _trace_handler(self, prof):
        """Trace 준비 완료 시 호출되는 핸들러"""
        print(f"\n{'='*60}")
        print(f"[ChakraTracer] Processing profiler trace...")
        print(f"{'='*60}")
        
        # 1. Kineto trace (JSON) 저장
        kineto_path = self.output_dir / f"{self.trace_name}_kineto.json"
        prof.export_chrome_trace(str(kineto_path))
        print(f"[ChakraTracer] ✓ Kineto trace saved: {kineto_path}")
        
        # 2. Stacks 분석 저장
        stacks_path = self.output_dir / f"{self.trace_name}_stacks.txt"
        with open(stacks_path, "w") as f:
            f.write(prof.key_averages(group_by_stack_n=5).table(
                sort_by="self_cuda_time_total", row_limit=50
            ))
        print(f"[ChakraTracer] ✓ Stack trace saved: {stacks_path}")
        
        # 3. Chakra ET 형식으로 변환
        if self.convert_to_et:
            print(f"\n[ChakraTracer] Converting to Chakra ET format...")
            success = self._convert_to_chakra_et(kineto_path)
            if success:
                print(f"\n{'='*60}")
                print(f"[ChakraTracer] ✓ Trace capture complete!")
                print(f"{'='*60}\n")
            else:
                print(f"\n{'='*60}")
                print(f"[ChakraTracer] ⚠ Kineto trace saved, manual conversion needed")
                print(f"{'='*60}\n")
        else:
            print(f"\n[ChakraTracer] To convert manually, use:")
            print(f"  python convert_to_et.py\n")
    
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
