"""
Multi-GPU Chakra tracer with ASTRA-sim compatible file naming.

Key fixes:
1. Each GPU generates separate ET file: trace_name.{rank}.et
2. All ranks save traces, not just rank 0
3. Proper file naming for ASTRA-sim
"""

import os
import torch
import torch.profiler as profiler
from torch.profiler import ExecutionTraceObserver
from pathlib import Path
import subprocess


class MultiGPUChakraTracer:
    """
    ASTRA-sim 호환 Multi-GPU Chakra Tracer.
    
    파일 네이밍: {trace_name}.{rank}.et
    """

    def __init__(
        self,
        output_dir: str = "./outputs",
        trace_name: str = "trace",
        enabled: bool = True,
        wait_steps: int = 5,
        warmup_steps: int = 5,
        active_steps: int = 10,  # 최소 10 iterations
        record_shapes: bool = True,
        profile_memory: bool = True,
        with_stack: bool = True,
        with_flops: bool = True,
        with_modules: bool = True,
        rank: int = 0,
        world_size: int = 1,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.trace_name = trace_name
        self.enabled = enabled
        self.rank = rank
        self.world_size = world_size

        if not self.enabled:
            self.profiler = None
            self.et_observer = None
            return

        # ASTRA-sim 파일 네이밍: trace_name.rank.et
        # 중간 파일들도 rank 포함
        self.host_trace_path = self.output_dir / f"{self.trace_name}_rank{self.rank}_host.json"
        self.device_trace_path = None
        
        print(f"[Rank {rank}] ChakraTracer initialized")
        print(f"  Output: {self.host_trace_path}")

        # ExecutionTraceObserver
        self.et_observer = ExecutionTraceObserver()
        self.et_observer.register_callback(str(self.host_trace_path))
        self.et_started = False

        # Profiler activities
        activities = [profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(profiler.ProfilerActivity.CUDA)

        # Profiler 설정
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
            with_modules=with_modules,
            execution_trace_observer=self.et_observer,
        )

    def _trace_handler(self, prof):
        """Trace 저장"""
        print(f"\n[Rank {self.rank}] Saving traces...")

        # Device trace
        self.device_trace_path = self.output_dir / f"{self.trace_name}_rank{self.rank}_device.json"
        prof.export_chrome_trace(str(self.device_trace_path))
        print(f"[Rank {self.rank}] ✓ Device trace: {self.device_trace_path.name}")

        # Stack analysis
        if self.rank == 0:  # 대표로 rank 0만 저장
            stacks_path = self.output_dir / f"{self.trace_name}_stacks.txt"
            with open(stacks_path, "w") as f:
                f.write("=== CPU Time ===\n")
                f.write(prof.key_averages().table(sort_by="cpu_time_total", row_limit=30))
                if torch.cuda.is_available():
                    f.write("\n\n=== CUDA Time ===\n")
                    f.write(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))
            print(f"[Rank {self.rank}] ✓ Stack analysis: {stacks_path.name}")

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
            print(f"[Rank {self.rank}] ✓ Host trace saved: {self.host_trace_path.name}")

        # Convert to Chakra ET with ASTRA-sim naming
        if self.device_trace_path is not None:
            success = self._convert_to_chakra_et()
            
            if success and self.rank == 0:
                self._print_summary()

    def _convert_to_chakra_et(self):
        """
        Host + Device trace를 병합하고 Chakra ET로 변환.
        
        최종 파일명: {trace_name}.{rank}.et (ASTRA-sim 규칙)
        """
        merged_trace = self.output_dir / f"{self.trace_name}_rank{self.rank}_merged.json"
        
        # ASTRA-sim 네이밍: trace_name.rank.et
        et_file = self.output_dir / f"{self.trace_name}.{self.rank}.et"

        try:
            # Step 1: Link traces
            print(f"[Rank {self.rank}] Linking traces...")
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
                print(f"[Rank {self.rank}] ✗ Link failed: {result.stderr[:300]}")
                return False

            print(f"[Rank {self.rank}] ✓ Traces linked")

            # Step 2: Convert to ET
            print(f"[Rank {self.rank}] Converting to Chakra ET...")
            
            # chakra_converter는 .et 확장자를 자동으로 추가하므로
            # 확장자 없이 전달
            et_base = self.output_dir / f"{self.trace_name}.{self.rank}"
            
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
                print(f"[Rank {self.rank}] ✗ Conversion failed: {result.stderr[:300]}")
                return False

            # 파일 확인
            if et_file.exists():
                size_mb = et_file.stat().st_size / (1024 * 1024)
                print(f"[Rank {self.rank}] ✓ Chakra ET created: {et_file.name} ({size_mb:.2f} MB)")
                return True
            else:
                print(f"[Rank {self.rank}] ✗ ET file not found: {et_file.name}")
                return False

        except FileNotFoundError as e:
            print(f"[Rank {self.rank}] ✗ Command not found: {e}")
            print(f"  Install Chakra tools first!")
            return False
        except subprocess.TimeoutExpired:
            print(f"[Rank {self.rank}] ✗ Timeout")
            return False
        except Exception as e:
            print(f"[Rank {self.rank}] ✗ Error: {e}")
            return False

    def _print_summary(self):
        """Rank 0가 전체 결과 요약 출력"""
        print(f"\n{'='*60}")
        print(f"Chakra ET Generation Summary")
        print(f"{'='*60}")
        print(f"Trace name: {self.trace_name}")
        print(f"World size: {self.world_size}")
        print(f"\nGenerated files (ASTRA-sim format):")
        
        for rank in range(self.world_size):
            et_file = self.output_dir / f"{self.trace_name}.{rank}.et"
            if et_file.exists():
                size_mb = et_file.stat().st_size / (1024 * 1024)
                print(f"  ✓ {et_file.name} ({size_mb:.2f} MB)")
            else:
                print(f"  ✗ {et_file.name} (missing)")
        
        print(f"\nASTRA-sim usage:")
        print(f"  ./AstraSim_Analytical_Congestion_Unaware \\")
        print(f"    --workload-configuration={self.output_dir / self.trace_name} \\")
        print(f"    --system-configuration=system.json \\")
        print(f"    --network-configuration=network.yml \\")
        print(f"    --remote-memory-configuration=memory.json")
        print(f"{'='*60}\n")

    def step(self):
        """매 iteration마다 호출"""
        if self.profiler is not None:
            self.profiler.step()

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


# Backward compatibility: alias
ChakraTracer = MultiGPUChakraTracer