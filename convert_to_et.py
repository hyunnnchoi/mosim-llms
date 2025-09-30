#!/usr/bin/env python3
"""Convert Kineto JSON traces to Chakra ET format."""

import os
import sys
from pathlib import Path

def convert_trace(kineto_file, et_file):
    """Convert a single Kineto trace to ET format."""
    try:
        from chakra.et_converter.pytorch import PyTorchConverter
        
        converter = PyTorchConverter()
        converter.convert(
            input_filename=str(kineto_file),
            output_filename=str(et_file)
        )
        print(f"✓ Converted: {kineto_file.name} → {et_file.name}")
        return True
    except ImportError as e:
        print(f"✗ Error: Chakra converter not found: {e}")
        print("\nTo install Chakra:")
        print("  pip install git+https://github.com/mlcommons/chakra.git")
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
