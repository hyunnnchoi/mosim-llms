#!/usr/bin/env python3
"""
Validate Chakra ET files for ASTRA-sim compatibility.

ASTRA-sim requires:
1. Valid protobuf format
2. Complete computation graph
3. Communication collectives for multi-GPU
4. Proper node dependencies
"""

import sys
from pathlib import Path
import struct


def validate_et_file(et_path: Path):
    """
    Validate a Chakra ET file for ASTRA-sim compatibility.
    
    Returns:
        (is_valid, issues)
    """
    issues = []
    
    print(f"\n{'='*60}")
    print(f"Validating: {et_path.name}")
    print(f"{'='*60}")
    
    # 1. Check file exists
    if not et_path.exists():
        return False, [f"File does not exist: {et_path}"]
    
    # 2. Check file size
    size_bytes = et_path.stat().st_size
    size_mb = size_bytes / (1024 * 1024)
    print(f"✓ File size: {size_mb:.2f} MB")
    
    if size_mb < 0.1:
        issues.append(f"File is very small ({size_mb:.2f} MB) - may be incomplete")
    
    # 3. Check if it's a valid protobuf file
    try:
        with open(et_path, 'rb') as f:
            header = f.read(4)
            
            # Protobuf files typically don't have a magic header,
            # but we can check if it's binary data
            if len(header) < 4:
                issues.append("File is too small to be valid protobuf")
            
            # Try to read some protobuf structure
            f.seek(0)
            data = f.read(1024)
            
            # Look for common protobuf patterns
            # Protobuf uses varint encoding, field tags, etc.
            has_binary = any(b > 127 for b in data[:100])
            
            if not has_binary:
                issues.append("File doesn't appear to be binary protobuf format")
            else:
                print(f"✓ File appears to be binary format (protobuf)")
                
    except Exception as e:
        issues.append(f"Cannot read file: {e}")
    
    # 4. Try to parse with protobuf (if chakra is installed)
    try:
        from chakra.et_def.et_def_pb2 import GlobalMetadata, Node
        from google.protobuf import text_format
        
        print(f"✓ Chakra protobuf definitions available")
        
        # Try to read the ET file
        with open(et_path, 'rb') as f:
            # ET files contain multiple protobuf messages
            # We'll try to read the first few
            
            message_count = 0
            node_count = 0
            
            while True:
                # Read message length (varint)
                try:
                    length_bytes = []
                    while True:
                        byte = f.read(1)
                        if not byte:
                            break
                        length_bytes.append(ord(byte))
                        if ord(byte) < 128:
                            break
                    
                    if not length_bytes:
                        break
                    
                    # Decode varint
                    length = 0
                    for i, b in enumerate(length_bytes):
                        length |= (b & 0x7F) << (7 * i)
                    
                    # Read message
                    msg_data = f.read(length)
                    if len(msg_data) < length:
                        break
                    
                    message_count += 1
                    
                    # Try to parse as Node
                    try:
                        node = Node()
                        node.ParseFromString(msg_data)
                        node_count += 1
                    except:
                        pass
                    
                    # Stop after reading a few messages
                    if message_count > 100:
                        break
                        
                except:
                    break
            
            print(f"✓ Read {message_count} protobuf messages")
            print(f"✓ Parsed {node_count} Chakra nodes")
            
            if node_count == 0:
                issues.append("Could not parse any Chakra nodes - format may be incorrect")
            elif node_count < 10:
                issues.append(f"Only {node_count} nodes found - trace may be incomplete")
                
    except ImportError:
        print(f"⚠️  Cannot validate protobuf structure (chakra not installed)")
        issues.append("Install chakra to validate protobuf structure: pip install git+https://github.com/mlcommons/chakra.git")
    except Exception as e:
        issues.append(f"Error parsing protobuf: {e}")
    
    # 5. Check filename convention
    if not et_path.name.endswith('.et'):
        issues.append("File should have .et extension")
    
    # Summary
    print(f"\n{'='*60}")
    if issues:
        print(f"⚠️  Found {len(issues)} potential issue(s):")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
        is_valid = False
    else:
        print(f"✓ File appears valid for ASTRA-sim")
        is_valid = True
    
    print(f"{'='*60}\n")
    
    return is_valid, issues


def main():
    """Validate all .et files in outputs directory."""
    
    if len(sys.argv) > 1:
        # Validate specific file
        et_path = Path(sys.argv[1])
        if not et_path.exists():
            print(f"✗ File not found: {et_path}")
            sys.exit(1)
        
        is_valid, issues = validate_et_file(et_path)
        sys.exit(0 if is_valid else 1)
    
    # Validate all .et files in outputs
    outputs_dir = Path("outputs")
    if not outputs_dir.exists():
        print(f"✗ outputs directory not found")
        sys.exit(1)
    
    et_files = list(outputs_dir.glob("*.et"))
    
    if not et_files:
        print(f"✗ No .et files found in {outputs_dir}")
        sys.exit(1)
    
    print(f"Found {len(et_files)} .et files")
    print(f"{'='*60}\n")
    
    results = {}
    for et_file in sorted(et_files):
        is_valid, issues = validate_et_file(et_file)
        results[et_file.name] = (is_valid, issues)
    
    # Summary
    print(f"\n{'='*60}")
    print(f"VALIDATION SUMMARY")
    print(f"{'='*60}")
    
    valid_count = sum(1 for v, _ in results.values() if v)
    invalid_count = len(results) - valid_count
    
    print(f"Valid files:   {valid_count}/{len(results)}")
    print(f"Invalid files: {invalid_count}/{len(results)}")
    
    if invalid_count > 0:
        print(f"\nFiles with issues:")
        for name, (is_valid, issues) in results.items():
            if not is_valid:
                print(f"  ✗ {name}")
                for issue in issues[:3]:  # Show first 3 issues
                    print(f"    - {issue}")
    
    print(f"\nRecommendations:")
    print(f"1. Use improved tracer with more iterations (active_steps >= 10)")
    print(f"2. Ensure DDP communication is captured")
    print(f"3. Verify multi-GPU traces have collective operations")
    print(f"4. Consider using Chakra synthetic workload generator")
    
    sys.exit(0 if invalid_count == 0 else 1)


if __name__ == "__main__":
    main()