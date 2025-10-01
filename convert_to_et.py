#!/usr/bin/env python3
"""Convert Kineto JSON traces to Chakra ET format."""

import os
import sys
from pathlib import Path

def convert_trace(kineto_file, et_file):
    """Convert a single Kineto trace to ET format using chakra_converter CLI."""
    import subprocess
    
    try:
        # chakra_converter CLI를 사용 (Python API와 달리 Kineto JSON을 직접 처리 가능)
        result = subprocess.run(
            [
                "chakra_converter", "PyTorch",
                "--input", str(kineto_file),
                "--output", str(et_file.with_suffix(""))  # .et는 자동 추가됨
            ],
            capture_output=True,
            text=True,
            timeout=300  # 5분 타임아웃
        )
        
        if result.returncode != 0:
            print(f"✗ Error converting {kineto_file.name}: {result.stderr}")
            return False
        
        print(f"✓ Converted: {kineto_file.name} → {et_file.name}")
        return True
        
    except FileNotFoundError:
        print(f"✗ Error: chakra_converter command not found")
        print("\nTo install Chakra:")
        print("  pip install https://github.com/mlcommons/chakra/archive/refs/heads/main.zip")
        return False
    except subprocess.TimeoutExpired:
        print(f"✗ Error: Conversion timed out (>5 minutes)")
        return False
    except Exception as e:
        print(f"✗ Error converting {kineto_file.name}: {e}")
        return False

def main():
    """Convert all Kineto traces in outputs directory."""
    outputs_dir = Path("outputs")
    
    if not outputs_dir.exists():
        print(f"✗ Error: {outputs_dir} directory not found")
        sys.exit(1)
    
    # Find all Kineto JSON files
    kineto_files = list(outputs_dir.glob("*_kineto.json"))
    
    if not kineto_files:
        print(f"✗ No Kineto trace files found in {outputs_dir}")
        sys.exit(1)
    
    print(f"Found {len(kineto_files)} Kineto trace files")
    print("="*50)
    
    success_count = 0
    fail_count = 0
    
    for kineto_file in sorted(kineto_files):
        # Generate ET filename
        et_file = kineto_file.with_name(
            kineto_file.name.replace("_kineto.json", ".et")
        )
        
        if convert_trace(kineto_file, et_file):
            success_count += 1
        else:
            fail_count += 1
            # Stop on first error
            break
    
    print("="*50)
    print(f"\nConversion complete!")
    print(f"  Success: {success_count}")
    print(f"  Failed:  {fail_count}")
    
    if success_count > 0:
        print(f"\nET files saved to: {outputs_dir.absolute()}")
        # List generated ET files
        et_files = list(outputs_dir.glob("*.et"))
        for et_file in sorted(et_files):
            size_mb = et_file.stat().st_size / (1024 * 1024)
            print(f"  {et_file.name} ({size_mb:.2f} MB)")

if __name__ == "__main__":
    main()
