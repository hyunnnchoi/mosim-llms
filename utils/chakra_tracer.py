"""Chakra execution trace capture utilities.

Chakra Workflow:
1. Collect PyTorch host trace (ExecutionTraceObserver) and device trace (Kineto)
2. chakra_trace_link -> merge host + device traces
3. chakra_converter -> convert to Chakra ET (protobuf)
"""

import os
import torch
import torch.profiler as profiler
from torch.profiler import ExecutionTraceObserver
from typing import Optional, Callable
from pathlib import Path


class ChakraTracer:
    """
    Chakra Execution Trace 캡처를 위한 래퍼 클래스.

    PyTorch ExecutionTraceObserver와 Profiler를 사용하여
    host trace와 device trace를 수집합니다.

    Workflow:
        1. ExecutionTraceObserver -> PyTorch host trace
        2. PyTorch Profiler -> Kineto device trace
        3. chakra_trace_link -> merge traces
        4. chakra_converter -> .et file
    """

    def __init__(
        self,
        output_dir: str = "./outputs",
        trace_name: str = "trace",
        enabled: bool = True,
        wait_steps: int = 2,
        warmup_steps: int = 2,
        active_steps: int = 1,  # MUST be 1 for ExecutionTraceObserver
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
            self.et_observer = None
            return

        # 1. ExecutionTraceObserver for host trace
        self.host_trace_path = self.output_dir / f"{self.trace_name}_host.json"
        self.et_observer = ExecutionTraceObserver()
        self.et_observer.register_callback(str(self.host_trace_path))
        self.et_started = False  # Track if ET observer has been started
        self.device_trace_path = None  # Will be set in _trace_handler

        # 2. PyTorch Profiler 설정 (Kineto device trace 생성)
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
            execution_trace_observer=self.et_observer,  # Link ET observer to profiler
        )
    
    def _link_and_convert_traces(self, device_trace_path: Path):
        """
        Host trace와 device trace를 병합하고 Chakra ET로 변환

        Chakra 워크플로우:
        1. chakra_trace_link: merge host + device traces -> merged JSON
        2. chakra_converter: merged JSON -> .et (protobuf)
        """
        import subprocess

        # 파일 경로 설정
        base_name = self.trace_name
        merged_trace_path = self.output_dir / f"{base_name}_merged.json"
        et_path_base = self.output_dir / base_name
        et_path = self.output_dir / f"{base_name}.et"

        try:
            # Step 1: Link host and device traces
            print(f"[ChakraTracer] Linking host and device traces...")
            print(f"  Host trace: {self.host_trace_path}")
            print(f"  Device trace: {device_trace_path}")

            result = subprocess.run(
                [
                    "chakra_trace_link",
                    "--rank", str(self.rank),
                    "--chakra-host-trace", str(self.host_trace_path),
                    "--chakra-device-trace", str(device_trace_path),
                    "--output-file", str(merged_trace_path)
                ],
                capture_output=True,
                text=True,
                timeout=300  # 5분 타임아웃
            )

            if result.returncode != 0:
                print(f"[ChakraTracer] Warning: chakra_trace_link failed:")
                print(f"  {result.stderr}")
                return False

            print(f"[ChakraTracer] ✓ Traces linked: {merged_trace_path}")

            # Step 2: Convert merged trace to ET format
            print(f"[ChakraTracer] Converting to Chakra ET format...")

            result = subprocess.run(
                [
                    "chakra_converter", "PyTorch",
                    "--input", str(merged_trace_path),
                    "--output", str(et_path_base)  # .et는 자동 추가됨
                ],
                capture_output=True,
                text=True,
                timeout=300  # 5분 타임아웃
            )

            if result.returncode != 0:
                print(f"[ChakraTracer] Warning: chakra_converter failed:")
                print(f"  {result.stderr}")
                return False

            print(f"[ChakraTracer] ✓ Chakra ET file saved to {et_path}")
            return True

        except FileNotFoundError as e:
            print(f"[ChakraTracer] Error: Command not found: {e}")
            print(f"  Make sure Chakra is properly installed.")
            return False
        except subprocess.TimeoutExpired:
            print(f"[ChakraTracer] Warning: Conversion timed out (>5 minutes)")
            return False
        except Exception as e:
            print(f"[ChakraTracer] Warning: Failed to process traces: {e}")
            return False
    
    def _trace_handler(self, prof):
        """Trace 준비 완료 시 호출되는 핸들러"""
        print(f"\n{'='*60}")
        print(f"[ChakraTracer] Processing profiler trace...")
        print(f"{'='*60}")

        # Note: DO NOT stop et_observer here - let __exit__ handle it
        # Stopping here causes duplicate JSON objects in the file

        # 1. Kineto device trace (JSON) 저장
        device_trace_path = self.output_dir / f"{self.trace_name}_device.json"
        prof.export_chrome_trace(str(device_trace_path))
        print(f"[ChakraTracer] ✓ Device trace saved: {device_trace_path}")

        # 2. Stacks 분석 저장
        stacks_path = self.output_dir / f"{self.trace_name}_stacks.txt"
        with open(stacks_path, "w") as f:
            f.write(prof.key_averages(group_by_stack_n=5).table(
                sort_by="self_cuda_time_total", row_limit=50
            ))
        print(f"[ChakraTracer] ✓ Stack trace saved: {stacks_path}")

        # Note: Trace linking will be done in __exit__ after et_observer is properly stopped
        self.device_trace_path = device_trace_path  # Save for later use
    
    def __enter__(self):
        """Context manager 진입"""
        # Start ExecutionTraceObserver before profiler
        if self.et_observer is not None and not self.et_started:
            self.et_observer.start()
            self.et_started = True
        if self.profiler is not None:
            self.profiler.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager 종료"""
        if self.profiler is not None:
            self.profiler.__exit__(exc_type, exc_val, exc_tb)

        # Stop and unregister ET observer
        if self.et_observer is not None:
            if self.et_started:
                self.et_observer.stop()
                self.et_started = False
            self.et_observer.unregister_callback()
            print(f"[ChakraTracer] ✓ Host trace saved: {self.host_trace_path}")

        # Now that both traces are complete, link and convert them
        if self.convert_to_et and self.device_trace_path is not None:
            print(f"\n[ChakraTracer] Linking and converting to Chakra ET format...")
            success = self._link_and_convert_traces(self.device_trace_path)
            if success:
                print(f"\n{'='*60}")
                print(f"[ChakraTracer] ✓ Trace capture complete!")
                print(f"{'='*60}\n")
            else:
                print(f"\n{'='*60}")
                print(f"[ChakraTracer] ⚠ Traces saved, manual conversion needed")
                print(f"  Manual conversion:")
                print(f"  1. chakra_trace_link --rank {self.rank} --chakra-host-trace {self.host_trace_path} --chakra-device-trace {self.device_trace_path} --output-file merged.json")
                print(f"  2. chakra_converter PyTorch --input merged.json --output output_trace")
                print(f"{'='*60}\n")

    def step(self):
        """Profiling step (각 iteration마다 호출)"""
        if self.profiler is not None:
            self.profiler.step()

    def start(self):
        """Profiling 시작"""
        if self.et_observer is not None and not self.et_started:
            self.et_observer.start()
            self.et_started = True
        if self.profiler is not None:
            self.profiler.start()

    def stop(self):
        """Profiling 종료"""
        if self.profiler is not None:
            self.profiler.stop()
        if self.et_observer is not None and self.et_started:
            self.et_observer.stop()
            self.et_started = False


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
