#!/bin/bash

# ASTRA-sim으로 Chakra ET 파일 테스트

set -e

echo "=========================================="
echo "Testing Chakra ET with ASTRA-sim"
echo "=========================================="
echo ""

# 1. ASTRA-sim 경로 확인
ASTRASIM_DIR=${ASTRASIM_DIR:-"../astra-sim"}

if [ ! -d "$ASTRASIM_DIR" ]; then
    echo "✗ ASTRA-sim not found at: $ASTRASIM_DIR"
    echo ""
    echo "Please set ASTRASIM_DIR environment variable:"
    echo "  export ASTRASIM_DIR=/path/to/astra-sim"
    echo ""
    echo "Or clone ASTRA-sim:"
    echo "  git clone --recurse-submodules https://github.com/astra-sim/astra-sim.git"
    exit 1
fi

echo "✓ ASTRA-sim found: $ASTRASIM_DIR"
echo ""

# 2. ET 파일 선택
ET_FILE=${1:-"outputs/gpt2_1gpu_quick_trace.et"}

if [ ! -f "$ET_FILE" ]; then
    echo "✗ ET file not found: $ET_FILE"
    echo ""
    echo "Usage: $0 <et_file>"
    echo ""
    echo "Available ET files:"
    ls -1 outputs/*.et 2>/dev/null || echo "  (none found)"
    exit 1
fi

echo "Testing with: $ET_FILE"
echo ""

# 3. ET 파일 검증
echo "Validating ET file..."
python3 validate_et_for_astrasim.py "$ET_FILE"

if [ $? -ne 0 ]; then
    echo ""
    echo "⚠️  ET file validation failed!"
    echo "The file may not work with ASTRA-sim."
    echo ""
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo ""

# 4. ASTRA-sim 설정 파일 생성
CONFIG_DIR="$(pwd)/astrasim_configs"
mkdir -p "$CONFIG_DIR"

# System configuration
cat > "$CONFIG_DIR/system.txt" << 'EOF'
# Simple system configuration for testing
topology-name: Ring
dimensions-count: 1
dimension[0]: 4
local-dimension: 0
EOF

# Network configuration
cat > "$CONFIG_DIR/network.txt" << 'EOF'
# Simple network configuration
topology: Ring
npus-count: 4
link-latency: 100
link-bandwidth: 25
EOF

# Workload configuration
WORKLOAD_CONFIG="$CONFIG_DIR/workload.txt"
cat > "$WORKLOAD_CONFIG" << EOF
# Workload configuration
workload-type: Chakra
workload-file: $(realpath $ET_FILE)
EOF

echo "✓ Configuration files created in $CONFIG_DIR/"
echo ""

# 5. ASTRA-sim 실행
cd "$ASTRASIM_DIR"

if [ ! -f "./build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Unaware" ]; then
    echo "Building ASTRA-sim..."
    ./build/astra_analytical/build.sh
fi

echo "Running ASTRA-sim..."
echo "=========================================="

RESULT_DIR="$(pwd)/results/test_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULT_DIR"

./build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Unaware \
    --workload-configuration="$(realpath $ET_FILE)" \
    --system-configuration="$CONFIG_DIR/system.txt" \
    --network-configuration="$CONFIG_DIR/network.txt" \
    --num-passes=1 \
    --num-queues-per-dim=1 \
    2>&1 | tee "$RESULT_DIR/output.log"

EXIT_CODE=$?

echo "=========================================="
echo ""

if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ ASTRA-sim completed successfully!"
    echo ""
    echo "Results saved to: $RESULT_DIR"
    echo ""
    echo "Log file: $RESULT_DIR/output.log"
else
    echo "✗ ASTRA-sim failed with exit code: $EXIT_CODE"
    echo ""
    echo "Common issues:"
    echo "1. ET file format incompatible"
    echo "2. Missing communication collectives"
    echo "3. Incomplete dependency graph"
    echo ""
    echo "Check the log: $RESULT_DIR/output.log"
    echo ""
    echo "Recommendations:"
    echo "- Use the improved tracer (chakra_tracer_improved.py)"
    echo "- Increase active_steps to 10-20"
    echo "- Ensure DDP communication is captured"
    echo "- Try synthetic workload generator"
fi

exit $EXIT_CODE