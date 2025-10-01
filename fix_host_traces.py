#!/usr/bin/env python3
"""
Fix corrupted host trace files.

Host trace files may contain multiple JSON objects concatenated together.
This script extracts the last valid JSON object from each file.
"""

import json
from pathlib import Path
import re

def fix_host_trace(file_path):
    """
    Fix a host trace file by extracting the last complete JSON object.
    
    The issue: ExecutionTraceObserver may write multiple JSON objects to the same file,
    causing "}{" patterns where one JSON ends and another begins.
    
    Solution: Find all complete JSON objects and keep only the last one.
    """
    print(f"\n처리 중: {file_path.name}")
    
    # Read the file
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Try to find all JSON objects (separated by "}{" pattern)
    # Split by "}{" and add back the braces
    parts = content.split('}{')
    
    if len(parts) == 1:
        # No duplicate objects, file is OK
        print(f"  ✓ 이미 정상 파일입니다")
        return True
    
    print(f"  발견: {len(parts)}개의 JSON 객체")
    
    # Take the last part and add back the opening brace
    last_json = '{' + parts[-1]
    
    # Verify it's valid JSON
    try:
        data = json.loads(last_json)
        print(f"  ✓ 유효한 JSON 객체 추출 성공")
    except json.JSONDecodeError as e:
        print(f"  ✗ JSON 파싱 실패: {e}")
        return False
    
    # Create backup
    backup_path = file_path.with_suffix('.json.backup')
    file_path.rename(backup_path)
    print(f"  ✓ 백업 생성: {backup_path.name}")
    
    # Write the fixed JSON
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"  ✓ 수정 완료: {file_path.name}")
    return True

def main():
    """Fix all host trace files in outputs directory."""
    outputs_dir = Path("outputs")
    
    if not outputs_dir.exists():
        print(f"✗ Error: {outputs_dir} directory not found")
        return
    
    # Find all host trace files
    host_files = list(outputs_dir.glob("*_host.json"))
    
    if not host_files:
        print(f"✗ No host trace files found in {outputs_dir}")
        return
    
    print("="*60)
    print(f"Host Trace 파일 수정")
    print("="*60)
    print(f"\n찾은 파일: {len(host_files)}개")
    
    success_count = 0
    fail_count = 0
    
    for host_file in sorted(host_files):
        if fix_host_trace(host_file):
            success_count += 1
        else:
            fail_count += 1
    
    print("\n" + "="*60)
    print(f"수정 완료!")
    print(f"  성공: {success_count}개")
    print(f"  실패: {fail_count}개")
    print("="*60)
    
    if fail_count > 0:
        print("\n백업 파일을 확인하세요: *_host.json.backup")
    
    print("\n다음 단계:")
    print("  python convert_to_et.py")
    print("  # 또는")
    print("  ./setup_and_convert.sh")

if __name__ == "__main__":
    main()

