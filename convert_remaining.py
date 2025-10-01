#!/usr/bin/env python3
"""Convert remaining traces with increased recursion limit."""

import sys
import subprocess
from pathlib import Path

# Increase recursion limit for complex multi-GPU traces
sys.setrecursionlimit(10000)

def convert_with_increased_limit(merged_file, output_base):
    """Convert with increased recursion limit."""
    print(f"\n변환 중: {merged_file.name}")
    print("  재귀 한도: 10000")
    
    try:
        result = subprocess.run(
            [
                "chakra_converter", "PyTorch",
                "--input", str(merged_file),
                "--output", str(output_base)
            ],
            capture_output=True,
            text=True,
            timeout=600,  # 10분 타임아웃
            env={**subprocess.os.environ, 'PYTHONRECURSIONLIMIT': '10000'}
        )
        
        if result.returncode != 0:
            print(f"  ✗ 실패:")
            print(f"    {result.stderr[:500]}")
            return False
        
        # Rename to add .et extension
        output_file = Path(str(output_base))
        if output_file.exists():
            et_file = output_file.with_suffix('.et')
            output_file.rename(et_file)
            print(f"  ✓ 성공: {et_file.name}")
            return True
        
        return False
        
    except subprocess.TimeoutExpired:
        print(f"  ✗ 타임아웃 (>10분)")
        return False
    except Exception as e:
        print(f"  ✗ 에러: {e}")
        return False

def main():
    outputs_dir = Path("outputs")
    
    # Remaining files
    remaining = [
        "gpt2_2gpu_quick_trace",
        "gpt2_4gpu_quick_trace"
    ]
    
    print("="*60)
    print("남은 파일 변환 (재귀 한도 증가)")
    print("="*60)
    
    success_count = 0
    fail_count = 0
    
    for base_name in remaining:
        merged_file = outputs_dir / f"{base_name}_merged.json"
        
        if not merged_file.exists():
            print(f"\n✗ 파일 없음: {merged_file.name}")
            fail_count += 1
            continue
        
        output_base = outputs_dir / base_name
        
        if convert_with_increased_limit(merged_file, output_base):
            success_count += 1
        else:
            fail_count += 1
    
    print("\n" + "="*60)
    print(f"변환 완료!")
    print(f"  성공: {success_count}")
    print(f"  실패: {fail_count}")
    print("="*60)
    
    if success_count > 0:
        print("\n생성된 파일:")
        for et_file in sorted(outputs_dir.glob("*.et")):
            size_mb = et_file.stat().st_size / (1024 * 1024)
            print(f"  {et_file.name} ({size_mb:.1f} MB)")

if __name__ == "__main__":
    main()

