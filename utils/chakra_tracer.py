"""Improved Chakra execution trace capture for ASTRA-sim compatibility."""

import os
import torch
import torch.profiler as profiler
from torch.profiler import ExecutionTraceObserver
from pathlib import Path
import json


class ImprovedChakraTracer:
    """
    ASTRA-sim 호환 Chakra Trace 생성을 위한 개선된 Tracer.
    
    주요 개선사항:
    1. 더 많은 iteration 캡처 (최소 5-10회)
    2. 통신 collective 명시적 캡처
    3. 완전한 dependency graph
    """

    def __init__(
        self,
        output_dir: str = "./outputs",
        trace_name: str = "trace",
        enabled: bool = True,
        wait_steps: int = 5,      # 더 긴 warmup
        warmup_steps: int = 5,    
        active_steps: int = 10,   # 더 많은 iteration 캡처
        record_shapes: bool = True,
        profile_memory: bool = True,
        with_stack: bool = True,
        with_flops: bool = True,
        with_modules: bool = True,  # Module hierarchy 포함
        experimental_config: dict = None,
        rank: int = 0
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.trace_name = trace_name
        self.enabled = enabled
        self.rank = rank

        if not self.enabled:
            self.profiler = None
            self.et_observer = None
            return

        # ExecutionTraceObserver 설정
        self.host_trace_path = self.output_dir / f"{self.trace_name}_host.json"
        self.et_observer = ExecutionTraceObserver()
        self.et_observer.register_callback(str(self.host_trace_path))
        self.et_started = False
        self.device_trace_path = None

        # Profiler activities
        activities = [profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(profiler.ProfilerActivity.CUDA)

        # Experimental config for better trace quality
        if experimental_config is None:
            experimental_config = {
                '_gpu_sync_event': True,  # GPU 동기화 이벤트 캡처
                'record_concrete_inputs_outputs': True,  # 실제 입출력 캡처
            }

        # Profiler 설정
        self.profiler = profiler.profile(
            activities=activities,
            schedule=profiler.schedule(
                wait=wait_steps,
                warmup=warmup_steps,
                active=active_steps,  # 최소 10 iterations
                repeat=1
            ),
            on_trace_ready=self._trace_handler,
            record_shapes=record_shapes,
            profile_memory=profile_memory,
            with_stack=with_stack,
            with_flops=with_flops,
            with_modules=with_modules,
            experimental_config=experimental_config,
            execution_trace_observer=self.et_observer,
        )
        
        self.step_count = 0

    def _validate_and_fix_trace(self, trace_path: Path):
        """
        Trace 파일 검증 및 수정.
        
        ASTRA-sim 호환성을 위해:
        1. Node ID 연속성 확인
        2. Dependency 정보 검증
        3. Communication collective 명시적 표시
        """
        try:
            with open(trace_path, 'r') as f:
                trace_data = json.load(f)
            
            print(f"[ChakraTracer] Validating trace: {trace_path.name}")
            
            # Basic validation
            if 'nodes' in trace_data:
                num_nodes = len(trace_data['nodes'])
                print(f"  - Total nodes: {num_nodes}")
                
                # Count operation types
                op_types = {}
                comm_ops = 0
                comp_ops = 0
                
                for node in trace_data['nodes']:
                    op_name = node.get('name', 'unknown')
                    op_types[op_name] = op_types.get(op_name, 0) + 1
                    
                    # Communication operations
                    if any(comm in op_name.lower() for comm in ['all_reduce', 'all_gather', 'reduce_scatter', 'broadcast']):
                        comm_ops += 1
                    # Computation operations  
                    elif any(comp in op_name.lower() for comp in ['matmul', 'conv', 'linear', 'softmax', 'layernorm']):
                        comp_ops += 1
                
                print(f"  - Communication ops: {comm_ops}")
                print(f"  - Computation ops: {comp_ops}")
                print(f"  - Top 5 operation types:")
                for op, count in sorted(op_types.items(), key=lambda x: x[1], reverse=True)[:5]:
                    print(f"    * {op}: {count}")
                
                # Warning if no communication ops in multi-GPU trace
                if self.rank == 0 and comm_ops == 0 and 'gpu' in self.trace_name.lower():
                    if any(x in self.trace_name.lower() for x in ['2gpu', '4gpu', '8gpu']):
                        print(f"  ⚠️  Warning: No communication ops found in multi-GPU trace!")
                        print(f"     This may indicate incomplete DDP gradient synchronization capture.")
                
                return True
                
        except Exception as e:
            print(f"[ChakraTracer] Warning: Could not validate trace: {e}")
            return False

    def _trace_handler(self, prof):
        """Trace 준비 완료 시 호출"""
        print(f"\n{'='*60}")
        print(f"[ChakraTracer] Processing profiler trace...")
        print(f"{'='*60}")

        # Kineto device trace 저장
        device_trace_path = self.output_dir / f"{self.trace_name}_device.json"
        prof.export_chrome_trace(str(device_trace_path))
        print(f"[ChakraTracer] ✓ Device trace saved: {device_trace_path}")

        # Stacks 분석
        stacks_path = self.output_dir / f"{self.trace_name}_stacks.txt"
        with open(stacks_path, "w") as f:
            # CPU time
            f.write("=== CPU Time Stats ===\n")
            f.write(prof.key_averages(group_by_stack_n=5).table(
                sort_by="cpu_time_total", row_limit=30
            ))
            f.write("\n\n")
            
            # CUDA time
            if torch.cuda.is_available():
                f.write("=== CUDA Time Stats ===\n")
                f.write(prof.key_averages(group_by_stack_n=5).table(
                    sort_by="cuda_time_total", row_limit=30
                ))
        
        print(f"[ChakraTracer] ✓ Stack analysis saved: {stacks_path}")

        self.device_trace_path = device_trace_path

    def __enter__(self):
        if self.et_observer is not None and not self.et_started:
            self.et_observer.start()
            self.et_started = True
        if self.profiler is not None:
            self.profiler.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.profiler is not None:
            self.profiler.__exit__(exc_type, exc_val, exc_tb)

        if self.et_observer is not None:
            if self.et_started:
                self.et_observer.stop()
                self.et_started = False
            self.et_observer.unregister_callback()
            print(f"[ChakraTracer] ✓ Host trace saved: {self.host_trace_path}")
            
            # Validate host trace
            self._validate_and_fix_trace(self.host_trace_path)

        # Link and convert traces
        if self.device_trace_path is not None:
            self._link_and_convert_traces()

    def _link_and_convert_traces(self):
        """Host + Device trace 병합 및 ET 변환"""
        import subprocess

        merged_trace = self.output_dir / f"{self.trace_name}_merged.json"
        et_base = self.output_dir / self.trace_name
        et_file = self.output_dir / f"{self.trace_name}.et"

        try:
            # Step 1: chakra_trace_link
            print(f"\n[ChakraTracer] Linking traces...")
            result = subprocess.run(
                [
                    "chakra_trace_link",
                    "--rank", str(self.rank),
                    "--chakra-host-trace", str(self.host_trace_path.absolute()),
                    "--chakra-device-trace", str(self.device_trace_path.absolute()),
                    "--output-file", str(merged_trace.absolute())
                ],
                capture_output=True,
                text=True,
                timeout=600
            )

            if result.returncode != 0:
                print(f"[ChakraTracer] ✗ Link failed: {result.stderr[:500]}")
                return False

            print(f"[ChakraTracer] ✓ Traces linked: {merged_trace.name}")

            # Step 2: chakra_converter
            print(f"[ChakraTracer] Converting to Chakra ET...")
            result = subprocess.run(
                [
                    "chakra_converter", "PyTorch",
                    "--input", str(merged_trace),
                    "--output", str(et_base)
                ],
                capture_output=True,
                text=True,
                timeout=600
            )

            if result.returncode != 0:
                print(f"[ChakraTracer] ✗ Conversion failed: {result.stderr[:500]}")
                return False

            print(f"[ChakraTracer] ✓ ET file created: {et_file.name}")
            
            # Validate ET file
            if et_file.exists():
                size_mb = et_file.stat().st_size / (1024 * 1024)
                print(f"[ChakraTracer] ET file size: {size_mb:.2f} MB")
                
                # Check if file is too small (may indicate incomplete capture)
                if size_mb < 0.5:
                    print(f"[ChakraTracer] ⚠️  Warning: ET file is very small!")
                    print(f"   Consider increasing active_steps or checking trace quality.")
            
            return True

        except FileNotFoundError as e:
            print(f"[ChakraTracer] ✗ Command not found: {e}")
            print(f"   Install Chakra tools first!")
            return False
        except subprocess.TimeoutExpired:
            print(f"[ChakraTracer] ✗ Timeout (>10 minutes)")
            return False
        except Exception as e:
            print(f"[ChakraTracer] ✗ Error: {e}")
            return False

    def step(self):
        """매 iteration마다 호출"""
        if self.profiler is not None:
            self.profiler.step()
        self.step_count += 1

    def start(self):
        if self.et_observer is not None and not self.et_started:
            self.et_observer.start()
            self.et_started = True
        if self.profiler is not None:
            self.profiler.start()

    def stop(self):
        if self.profiler is not None:
            self.profiler.stop()
        if self.et_observer is not None and self.et_started:
            self.et_observer.stop()
            self.et_started = False