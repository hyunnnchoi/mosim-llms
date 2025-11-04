"""
Multi-GPU Chakra tracer following the official wiki guidelines.

This implementation follows the Chakra Wiki ResNet-50 example (Section 4):
https://github.com/mlcommons/chakra/wiki/Chakra-Execution-Trace-Collection

Key features:
1. ExecutionTraceObserver is passed to profiler as execution_trace_observer parameter
2. PyTorch automatically synchronizes ET observer with profiler schedule
3. Simplified file naming: host_{rank}.json, device_{rank}.json
4. Automatic conversion to Chakra ET format on exit
"""

import os
import sys
import torch
import torch.profiler as profiler
from torch.profiler import ExecutionTraceObserver
from torch._C._profiler import _ExperimentalConfig
from pathlib import Path
import subprocess


class MultiGPUChakraTracer:
    """
    ASTRA-sim compatible Multi-GPU Chakra Tracer following official wiki.
    
    References:
    - https://github.com/mlcommons/chakra/wiki/Chakra-Execution-Trace-Collection
    """

    def __init__(
        self,
        output_dir: str = "./outputs",
        trace_name: str = "trace",
        enabled: bool = True,
        wait_steps: int = 0,      # Wiki recommends 0
        warmup_steps: int = 0,    # Wiki recommends 0
        active_steps: int = 1,    # Wiki recommends 1
        record_shapes: bool = True,
        profile_memory: bool = True,
        with_stack: bool = True,
        with_flops: bool = True,
        with_modules: bool = True,
        rank: int = 0,
        world_size: int = 1,
    ):
        # Create trace-specific subdirectory
        base_output_dir = Path(output_dir)
        self.output_dir = base_output_dir / trace_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create analysis subdirectory for optional outputs
        self.analysis_dir = self.output_dir / "analysis"
        self.analysis_dir.mkdir(exist_ok=True)

        self.trace_name = trace_name
        self.enabled = enabled
        self.rank = rank
        self.world_size = world_size
        
        # Schedule settings
        self.wait_steps = wait_steps
        self.warmup_steps = warmup_steps
        self.active_steps = active_steps

        if not self.enabled:
            self.profiler = None
            self.et_observer = None
            return

        # Simplified file naming (following wiki convention)
        self.host_trace_path = self.output_dir / f"host_{self.rank}.json"
        self.device_trace_path = self.output_dir / f"device_{self.rank}.json"
        
        print(f"[Rank {rank}] ChakraTracer initialized")
        print(f"  Output directory: {self.output_dir}")
        print(f"  Host trace: {self.host_trace_path.name}")
        print(f"  Device trace: {self.device_trace_path.name}")

        # ========================================
        # ExecutionTraceObserver setup
        # Following Chakra Wiki Section 4 (ResNet-50 example)
        # ========================================
        self.et_observer = ExecutionTraceObserver()
        self.et_observer.register_callback(str(self.host_trace_path))

        # Profiler activities
        activities = [profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(profiler.ProfilerActivity.CUDA)

        # ========================================
        # Profiler setup with ExecutionTraceObserver
        # Wiki example (line 241-252): Pass et_observer to profiler
        # PyTorch automatically synchronizes ET observer with schedule
        # ========================================
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
            # ✅ Pass ET observer to profiler (Wiki standard practice)
            execution_trace_observer=self.et_observer,
            # Enable CUDA sync events for accurate GPU timestamps
            experimental_config=_ExperimentalConfig(enable_cuda_sync_events=True),
        )

    def _trace_handler(self, prof):
        """Trace handler called when profiler completes active phase"""
        print(f"\n[Rank {self.rank}] Saving traces...")

        # Export Kineto device trace
        prof.export_chrome_trace(str(self.device_trace_path))
        print(f"[Rank {self.rank}] ✓ Device trace: {self.device_trace_path.name}")

        # Optional: Stack analysis (only rank 0)
        if self.rank == 0:
            stacks_path = self.analysis_dir / "stacks.txt"
            with open(stacks_path, "w") as f:
                f.write("=== CPU Time ===\n")
                f.write(prof.key_averages().table(sort_by="cpu_time_total", row_limit=30))
                if torch.cuda.is_available():
                    f.write("\n\n=== CUDA Time ===\n")
                    f.write(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))
            print(f"[Rank {self.rank}] ✓ Stack analysis: {stacks_path.relative_to(self.output_dir)}")

    def __enter__(self):
        """
        Context manager entry.

        PyTorch profiler automatically manages ET observer lifecycle.
        """
        if self.profiler is not None:
            self.profiler.__enter__()
            print(f"[Rank {self.rank}] ✓ Profiler started (ET observer auto-managed)")

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Context manager exit.

        Profiler automatically handles ET observer cleanup.
        We just need to unregister callback and convert traces.
        """
        # Stop profiler (auto-stops ET observer)
        if self.profiler is not None:
            self.profiler.__exit__(exc_type, exc_val, exc_tb)
            print(f"[Rank {self.rank}] ✓ Profiler stopped")

        # Unregister ET observer callback
        if self.et_observer is not None:
            self.et_observer.unregister_callback()
            print(f"[Rank {self.rank}] ✓ Host trace saved: {self.host_trace_path.name}")

        # Convert to Chakra ET
        if self.device_trace_path.exists() and self.host_trace_path.exists():
            success = self._convert_to_chakra_et()

            if success and self.rank == 0:
                self._print_summary()
        else:
            print(f"[Rank {self.rank}] ✗ Missing trace files, skipping conversion")

    def _fix_malformed_json(self, json_path):
        """
        Fix malformed JSON files with multiple concatenated objects.
        
        ExecutionTraceObserver sometimes outputs multiple JSON objects:
        {...}{...}{...} which is not valid JSON.
        
        This extracts the first valid JSON object and overwrites the file.
        Uses brace matching to find the first complete JSON object.
        """
        import json
        import shutil
        from pathlib import Path

        # Ensure json_path is a Path object
        json_path = Path(json_path)

        try:
            # [NOTE, hyunnnchoi, 2025.11.04] UTF-8 디코딩 에러 처리 - device 파일에 잘못된 바이트가 있을 수 있음
            content = None
            utf8_error_occurred = False
            
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            except UnicodeDecodeError:
                # Try with error handling for corrupted UTF-8
                utf8_error_occurred = True
                with open(json_path, 'rb') as f:
                    raw_content = f.read()
                # Decode with error handling
                content = raw_content.decode('utf-8', errors='replace')
                # Remove control characters immediately
                import re
                content = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F]', '', content)
                print(f"[Rank {self.rank}] ⚠ Warning: File {json_path.name} contains invalid UTF-8 bytes, cleaning...")

            # Try to parse normally first
            try:
                json.loads(content)
                # Already valid JSON, no fix needed
                # But if UTF-8 error occurred, we should save the cleaned version
                if utf8_error_occurred:
                    # Save cleaned version to ensure chakra_trace_link can read it
                    backup_path = json_path.with_suffix(json_path.suffix + '.backup')
                    if not backup_path.exists():
                        json_path.rename(backup_path)
                    else:
                        shutil.copy2(json_path, backup_path)
                    with open(json_path, 'w', encoding='utf-8') as f:
                        json.dump(json.loads(content), f, indent=2)
                    print(f"[Rank {self.rank}] ✓ Saved cleaned UTF-8 version: {json_path.name}")
                return True
            except json.JSONDecodeError as e:
                # Check if it's a Chrome trace format (array format)
                # Chrome trace files are usually arrays, not objects
                if content.strip().startswith('['):
                    # It's already a valid Chrome trace format, no fix needed
                    return True
                # Check if error is due to control characters
                if 'control character' in str(e).lower():
                    # Remove control characters except newlines and tabs
                    import re
                    cleaned_content = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F]', '', content)
                    try:
                        obj = json.loads(cleaned_content)
                        # Write cleaned version
                        backup_path = json_path.with_suffix(json_path.suffix + '.backup')
                        if not backup_path.exists():
                            json_path.rename(backup_path)
                        else:
                            shutil.copy2(json_path, backup_path)
                        with open(json_path, 'w', encoding='utf-8') as f:
                            json.dump(obj, f, indent=2)
                        print(f"[Rank {self.rank}] ✓ Fixed control characters in JSON: {json_path.name}")
                        return True
                    except json.JSONDecodeError:
                        pass
                # If JSON decode fails but file has UTF-8 issues, try cleaning UTF-8 first
                # Read file again with error handling and save cleaned version
                try:
                    with open(json_path, 'rb') as f:
                        raw_content = f.read()
                    # Decode with error handling
                    cleaned_utf8 = raw_content.decode('utf-8', errors='replace')
                    # Remove control characters
                    import re
                    cleaned_content = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F]', '', cleaned_utf8)
                    # Try parsing
                    obj = json.loads(cleaned_content)
                    # Save cleaned version
                    backup_path = json_path.with_suffix(json_path.suffix + '.backup')
                    if not backup_path.exists():
                        json_path.rename(backup_path)
                    else:
                        shutil.copy2(json_path, backup_path)
                    with open(json_path, 'w', encoding='utf-8') as f:
                        json.dump(obj, f, indent=2)
                    print(f"[Rank {self.rank}] ✓ Fixed UTF-8 and control characters in JSON: {json_path.name}")
                    return True
                except Exception:
                    pass
            except Exception as e:
                # Unexpected error, but continue to try fixing
                if 'control character' in str(e).lower():
                    import re
                    cleaned_content = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F]', '', content)
                    try:
                        json.loads(cleaned_content)
                        backup_path = json_path.with_suffix(json_path.suffix + '.backup')
                        if not backup_path.exists():
                            json_path.rename(backup_path)
                        else:
                            shutil.copy2(json_path, backup_path)
                        with open(json_path, 'w', encoding='utf-8') as f:
                            f.write(cleaned_content)
                        print(f"[Rank {self.rank}] ✓ Fixed control characters in JSON: {json_path.name}")
                        return True
                    except json.JSONDecodeError:
                        pass
                # Continue to try other methods
                pass

            # [NOTE, hyunnnchoi, 2025.11.04] 여러 JSON 객체가 {...}{...}{...} 형식으로 있을 때 처리 개선
            # Method 1: Find }{ pattern and extract first object
            # This is more reliable than brace matching for very large files
            split_positions = []
            
            # Find all }{ patterns (not inside strings)
            in_string = False
            escape_next = False
            
            for i in range(len(content) - 1):
                if escape_next:
                    escape_next = False
                    continue
                
                if content[i] == '\\':
                    escape_next = True
                    continue
                
                if content[i] == '"' and not escape_next:
                    in_string = not in_string
                    continue
                
                if not in_string and content[i] == '}' and content[i+1] == '{':
                    split_positions.append(i + 1)
            
            if split_positions:
                # [NOTE, hyunnnchoi, 2025.11.04] 여러 객체가 있을 때 마지막 완전한 객체 사용
                # Try each object from last to first, use the first complete one
                usable_obj_content = None
                usable_obj_idx = None
                
                # Try last object first (most likely to be complete)
                if len(split_positions) > 0:
                    last_obj_content = content[split_positions[-1]:]
                    try:
                        json.loads(last_obj_content)
                        usable_obj_content = last_obj_content
                        usable_obj_idx = len(split_positions)
                    except json.JSONDecodeError:
                        pass
                
                # If last object not valid, try first object
                if usable_obj_content is None:
                    first_obj_content = content[:split_positions[0]]
                    try:
                        json.loads(first_obj_content)
                        usable_obj_content = first_obj_content
                        usable_obj_idx = 0
                    except json.JSONDecodeError:
                        pass
                
                # If neither works, try middle objects
                if usable_obj_content is None:
                    for idx in range(len(split_positions) - 1, 0, -1):
                        obj_content = content[split_positions[idx-1]:split_positions[idx]]
                        try:
                            json.loads(obj_content)
                            usable_obj_content = obj_content
                            usable_obj_idx = idx
                            break
                        except json.JSONDecodeError:
                            continue
                
                if usable_obj_content:
                    try:
                        obj = json.loads(usable_obj_content)
                        
                        # Backup original file
                        backup_path = json_path.with_suffix(json_path.suffix + '.backup')
                        if not backup_path.exists():
                            json_path.rename(backup_path)
                        else:
                            shutil.copy2(json_path, backup_path)

                        # Write fixed JSON
                        with open(json_path, 'w') as f:
                            json.dump(obj, f, indent=2)
                        
                        print(f"[Rank {self.rank}] ✓ Fixed malformed JSON (extracted object {usable_obj_idx + 1}): {json_path.name}")
                        print(f"[Rank {self.rank}]   Found {len(split_positions) + 1} concatenated objects, using object {usable_obj_idx + 1}")
                        print(f"[Rank {self.rank}]   Backup saved: {backup_path.name}")
                        return True
                    except json.JSONDecodeError as e:
                        print(f"[Rank {self.rank}] ✗ Object extraction failed: {e}")
                else:
                    print(f"[Rank {self.rank}] ✗ No complete JSON object found in file")
            
            # Method 2: Use brace matching as fallback
            first_obj_end = None
            brace_count = 0
            in_string = False
            escape_next = False
            
            for i, char in enumerate(content):
                if escape_next:
                    escape_next = False
                    continue
                
                if char == '\\':
                    escape_next = True
                    continue
                
                if char == '"' and not escape_next:
                    in_string = not in_string
                    continue
                
                if not in_string:
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            # Found end of first JSON object
                            first_obj_end = i + 1
                            break
            
            if first_obj_end is not None:
                # Extract first JSON object
                first_obj_content = content[:first_obj_end]
                
                try:
                    # Validate it's valid JSON
                    obj = json.loads(first_obj_content)
                    
                    # Backup original file
                    backup_path = json_path.with_suffix(json_path.suffix + '.backup')
                    if not backup_path.exists():
                        json_path.rename(backup_path)
                    else:
                        shutil.copy2(json_path, backup_path)

                    # Write fixed JSON
                    with open(json_path, 'w') as f:
                        json.dump(obj, f, indent=2)
                    
                    print(f"[Rank {self.rank}] ✓ Fixed malformed JSON (brace matching): {json_path.name}")
                    print(f"[Rank {self.rank}]   Backup saved: {backup_path.name}")
                    return True
                except json.JSONDecodeError as e:
                    print(f"[Rank {self.rank}] ✗ Brace matching extraction failed: {e}")
            
            # Method 3: Try raw_decode as last resort
            try:
                decoder = json.JSONDecoder()
                obj, idx = decoder.raw_decode(content)
                
                # Backup original file
                backup_path = json_path.with_suffix(json_path.suffix + '.backup')
                if not backup_path.exists():
                    json_path.rename(backup_path)
                else:
                    shutil.copy2(json_path, backup_path)

                # Write fixed JSON
                with open(json_path, 'w') as f:
                    json.dump(obj, f, indent=2)
                
                print(f"[Rank {self.rank}] ✓ Fixed malformed JSON (raw_decode method): {json_path.name}")
                print(f"[Rank {self.rank}]   Backup saved: {backup_path.name}")
                return True
            except (json.JSONDecodeError, ValueError) as e:
                print(f"[Rank {self.rank}] ✗ Failed to fix JSON: {e}")
                print(f"[Rank {self.rank}]   File might be corrupted. Check backup if exists.")
                return False

        except Exception as e:
            print(f"[Rank {self.rank}] ✗ Failed to fix JSON: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _convert_to_chakra_et(self):
        """
        Merge host + device traces and convert to Chakra ET.

        Final file naming: {trace_name}.{rank}.et (ASTRA-sim convention)

        Steps:
        0. Fix malformed JSON if needed
        1. chakra_trace_link: merge host + device → merged.json
        2. chakra_converter: merged.json → trace_name.rank.et
        """
        merged_trace = self.output_dir / f"merged_{self.rank}.json"

        # ASTRA-sim naming: trace_name.rank.et
        et_file = self.output_dir / f"{self.trace_name}.{self.rank}.et"

        # Increase recursion limit for large traces
        import sys
        old_limit = sys.getrecursionlimit()
        sys.setrecursionlimit(50000)

        try:
            # Step 0: Fix malformed JSON
            print(f"[Rank {self.rank}] Validating JSON files...")
            if not self._fix_malformed_json(self.host_trace_path):
                print(f"[Rank {self.rank}] ✗ Host JSON validation failed")
                return False
            
            # [NOTE, hyunnnchoi, 2025.11.04] device 파일 수정 실패해도 계속 진행 (경고만 출력)
            device_fixed = self._fix_malformed_json(self.device_trace_path)
            if not device_fixed:
                print(f"[Rank {self.rank}] ⚠ Device JSON validation failed, but continuing...")
            
            # Step 1: Link traces
            print(f"[Rank {self.rank}] Linking host and device traces...")
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
                print(f"[Rank {self.rank}] ✗ Link failed:")
                print(f"  {result.stderr[:300]}")
                return False

            print(f"[Rank {self.rank}] ✓ Traces linked → {merged_trace.name}")

            # Step 2: Convert to Chakra ET
            print(f"[Rank {self.rank}] Converting to Chakra ET...")

            # Import converter within same process to inherit recursion limit
            try:
                from chakra.src.converter.converter import main as converter_main

                # Save original argv and replace it
                original_argv = sys.argv.copy()
                sys.argv = ['chakra_converter', 'PyTorch',
                           '--input', str(merged_trace.absolute()),
                           '--output', str(et_file.absolute())]

                # Call converter directly in same process
                converter_main()

                # Restore original argv
                sys.argv = original_argv

            except Exception as e:
                print(f"[Rank {self.rank}] ✗ Conversion failed: {e}")
                import traceback
                traceback.print_exc()
                return False

            # Verify output file
            if et_file.exists():
                size_mb = et_file.stat().st_size / (1024 * 1024)
                print(f"[Rank {self.rank}] ✓ Chakra ET created: {et_file.name} ({size_mb:.2f} MB)")
                return True
            else:
                print(f"[Rank {self.rank}] ✗ ET file not found: {et_file.name}")
                return False

        except FileNotFoundError as e:
            print(f"[Rank {self.rank}] ✗ Command not found: {e}")
            print(f"  Please install Chakra tools:")
            print(f"    git clone --recurse-submodules https://github.com/mlcommons/chakra.git")
            print(f"    cd chakra && pip install .")
            return False
        except subprocess.TimeoutExpired:
            print(f"[Rank {self.rank}] ✗ Conversion timeout (>10 minutes)")
            return False
        except Exception as e:
            print(f"[Rank {self.rank}] ✗ Unexpected error: {e}")
            return False
        finally:
            # Restore original recursion limit
            sys.setrecursionlimit(old_limit)

    def _print_summary(self):
        """Print summary (only rank 0)"""
        print(f"\n{'='*60}")
        print(f"Chakra ET Generation Summary")
        print(f"{'='*60}")
        print(f"Trace name: {self.trace_name}")
        print(f"Output directory: {self.output_dir}")
        print(f"World size: {self.world_size}")
        print(f"\nGenerated files (ASTRA-sim format):")
        
        total_size = 0
        for rank in range(self.world_size):
            et_file = self.output_dir / f"{self.trace_name}.{rank}.et"
            if et_file.exists():
                size_mb = et_file.stat().st_size / (1024 * 1024)
                total_size += size_mb
                print(f"  ✓ {et_file.name} ({size_mb:.2f} MB)")
            else:
                print(f"  ✗ {et_file.name} (missing)")
        
        print(f"\nTotal size: {total_size:.2f} MB")
        
        print(f"\nDirectory structure:")
        print(f"  {self.output_dir}/")
        print(f"    ├── host_*.json (raw PyTorch ET)")
        print(f"    ├── device_*.json (raw Kineto trace)")
        print(f"    ├── merged_*.json (linked traces)")
        print(f"    ├── {self.trace_name}.*.et (final Chakra ET)")
        print(f"    └── analysis/")
        print(f"        └── stacks.txt")
        
        print(f"\nASTRA-sim usage:")
        print(f"  ./AstraSim_Analytical_Congestion_Unaware \\")
        print(f"    --workload-configuration={self.output_dir / self.trace_name} \\")
        print(f"    --system-configuration=system.json \\")
        print(f"    --network-configuration=network.yml \\")
        print(f"    --remote-memory-configuration=memory.json")
        print(f"{'='*60}\n")

    def step(self):
        """
        Call this at the end of each training iteration.

        Advances profiler schedule. PyTorch automatically manages
        ET observer start/stop based on the schedule.
        """
        if self.profiler is not None:
            self.profiler.step()

    def start(self):
        """Explicit start (alternative to context manager)"""
        if self.profiler is not None:
            self.profiler.start()

    def stop(self):
        """Explicit stop (alternative to context manager)"""
        if self.profiler is not None:
            self.profiler.stop()


# Backward compatibility
ChakraTracer = MultiGPUChakraTracer