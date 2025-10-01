#!/usr/bin/env python3
"""Convert PyTorch host + device traces to Chakra ET format."""

import os
import sys
from pathlib import Path
import subprocess

def find_trace_pairs(outputs_dir):
    """Find pairs of host and device trace files."""
    host_files = list(outputs_dir.glob("*_host.json"))
    pairs = []
    
    for host_file in host_files:
        # Generate corresponding device file name
        base_name = host_file.name.replace("_host.json", "")
        device_file = outputs_dir / f"{base_name}_device.json"
        
        if device_file.exists():
            pairs.append((base_name, host_file, device_file))
    
    return pairs

def link_traces(base_name, host_file, device_file, merged_file, rank=0):
    """Link host and device traces using chakra_trace_link."""
    try:
        print(f"  [1/2] Linking traces...")
        print(f"    Host:   {host_file.name}")
        print(f"    Device: {device_file.name}")
        
        # Use absolute paths to avoid chakra_trace_link path resolution issues
        result = subprocess.run(
            [
                "chakra_trace_link",
                "--rank", str(rank),
                "--chakra-host-trace", str(host_file.absolute()),
                "--chakra-device-trace", str(device_file.absolute()),
                "--output-file", str(merged_file.absolute())
            ],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        if result.returncode != 0:
            print(f"✗ chakra_trace_link failed:")
            print(f"  {result.stderr}")
            return False
        
        print(f"  ✓ Merged: {merged_file.name}")
        return True
        
    except FileNotFoundError:
        print(f"✗ Error: chakra_trace_link command not found")
        print("\nTo install required tools:")
        print("  # Install PARAM")
        print("  git clone https://github.com/facebookresearch/param.git")
        print("  cd param/et_replay && git checkout 7b19f586dd8b267333114992833a0d7e0d601630")
        print("  pip install .")
        print("")
        print("  # Install HolisticTraceAnalysis")
        print("  git clone https://github.com/facebookresearch/HolisticTraceAnalysis.git")
        print("  cd HolisticTraceAnalysis && git checkout d731cc2e2249976c97129d409a83bd53d93051f6")
        print("  git submodule update --init")
        print("  pip install -r requirements.txt && pip install -e .")
        print("")
        print("  # Install Chakra")
        print("  pip install git+https://github.com/mlcommons/chakra.git")
        return False
    except subprocess.TimeoutExpired:
        print(f"✗ Linking timed out (>5 minutes)")
        return False
    except Exception as e:
        print(f"✗ Error linking traces: {e}")
        return False

def convert_to_et(merged_file, et_file):
    """Convert merged trace to Chakra ET format using chakra_converter."""
    try:
        print(f"  [2/2] Converting to ET format...")
        
        result = subprocess.run(
            [
                "chakra_converter", "PyTorch",
                "--input", str(merged_file),
                "--output", str(et_file.with_suffix(""))  # .et is added automatically
            ],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        if result.returncode != 0:
            print(f"✗ chakra_converter failed:")
            print(f"  {result.stderr}")
            return False
        
        print(f"  ✓ Converted: {et_file.name}")
        return True
        
    except FileNotFoundError:
        print(f"✗ Error: chakra_converter command not found")
        print("\nTo install Chakra:")
        print("  pip install git+https://github.com/mlcommons/chakra.git")
        return False
    except subprocess.TimeoutExpired:
        print(f"✗ Conversion timed out (>5 minutes)")
        return False
    except Exception as e:
        print(f"✗ Error converting: {e}")
        return False

def convert_trace_pair(base_name, host_file, device_file, outputs_dir):
    """Convert a pair of host and device traces to Chakra ET format."""
    merged_file = outputs_dir / f"{base_name}_merged.json"
    et_file = outputs_dir / f"{base_name}.et"
    
    print(f"\nProcessing: {base_name}")
    print("="*60)
    
    # Step 1: Link traces
    if not link_traces(base_name, host_file, device_file, merged_file):
        return False
    
    # Step 2: Convert to ET
    if not convert_to_et(merged_file, et_file):
        return False
    
    # Cleanup merged file (optional)
    # merged_file.unlink()
    
    return True

def main():
    """Convert all host+device trace pairs in outputs directory."""
    outputs_dir = Path("outputs")
    
    if not outputs_dir.exists():
        print(f"✗ Error: {outputs_dir} directory not found")
        sys.exit(1)
    
    # Find all trace pairs
    trace_pairs = find_trace_pairs(outputs_dir)
    
    if not trace_pairs:
        print(f"✗ No trace pairs found in {outputs_dir}")
        print("\nExpected files:")
        print("  *_host.json (ExecutionTraceObserver output)")
        print("  *_device.json (Kineto output)")
        sys.exit(1)
    
    print(f"Found {len(trace_pairs)} trace pairs")
    print("="*60)
    
    success_count = 0
    fail_count = 0
    
    for base_name, host_file, device_file in sorted(trace_pairs):
        if convert_trace_pair(base_name, host_file, device_file, outputs_dir):
            success_count += 1
        else:
            fail_count += 1
    
    print("\n" + "="*60)
    print(f"Conversion complete!")
    print(f"  Success: {success_count}")
    print(f"  Failed:  {fail_count}")
    print("="*60)
    
    if success_count > 0:
        print(f"\nET files saved to: {outputs_dir.absolute()}")
        # List generated ET files
        et_files = list(outputs_dir.glob("*.et"))
        for et_file in sorted(et_files):
            size_mb = et_file.stat().st_size / (1024 * 1024)
            print(f"  {et_file.name} ({size_mb:.2f} MB)")

if __name__ == "__main__":
    main()
