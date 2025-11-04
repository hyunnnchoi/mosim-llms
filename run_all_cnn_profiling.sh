#!/bin/bash

# 모든 CNN 모델 프로파일링 실험 자동 실행
# 8 Models × {1, 2, 4, 8} GPU × 1 epoch

EPOCHS=1

# 8개 CNN 모델
CNN_MODELS=(
    "densenet40_k12"
    "densenet100_k12"
    "googlenet"
    "inception3"
    "resnet44"
    "resnet110"
    "resnet50"
    "vgg16"
)

# 사용 가능한 GPU 수 확인
if command -v nvidia-smi &> /dev/null; then
    AVAILABLE_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
else
    AVAILABLE_GPUS=0
fi

echo "=========================================="
echo "Running All CNN Profiling Experiments"
echo "Models: ${CNN_MODELS[@]}"
echo "Available GPUs: $AVAILABLE_GPUS"
echo "Epochs: $EPOCHS"
echo "=========================================="
echo ""

# GPU 설정 결정
if [ $AVAILABLE_GPUS -ge 8 ]; then
    GPU_CONFIGS=(1 2 4 8)
elif [ $AVAILABLE_GPUS -ge 4 ]; then
    GPU_CONFIGS=(1 2 4)
    echo "⚠️  Only 4 GPUs available, skipping 8 GPU experiments"
elif [ $AVAILABLE_GPUS -ge 2 ]; then
    GPU_CONFIGS=(1 2)
    echo "⚠️  Only 2 GPUs available, skipping 4 and 8 GPU experiments"
else
    GPU_CONFIGS=(1)
    echo "⚠️  Only 1 GPU available, skipping multi-GPU experiments"
fi

TOTAL_EXP=$((${#CNN_MODELS[@]} * ${#GPU_CONFIGS[@]}))

echo "GPU configs to run: ${GPU_CONFIGS[@]}"
echo "Total experiments: $TOTAL_EXP (${#CNN_MODELS[@]} models × ${#GPU_CONFIGS[@]} GPU configs)"
echo ""

# 출력 디렉토리 생성
mkdir -p outputs

# 실험 카운터
counter=1
failed_experiments=()

# 각 모델별 프로파일링
for model in "${CNN_MODELS[@]}"; do
    echo ""
    echo "=== $model Experiments ==="

    for ngpu in "${GPU_CONFIGS[@]}"; do
        echo ""
        echo "[$counter/$TOTAL_EXP] $model with $ngpu GPU(s)"

        # 스크립트 실행
        if ./run_cnn_quick.sh $model $ngpu $EPOCHS; then
            echo "✓ $model with $ngpu GPU(s) completed"
        else
            echo "✗ $model with $ngpu GPU(s) failed"
            failed_experiments+=("$model with $ngpu GPU(s)")
        fi

        echo ""
        ((counter++))

        # GPU 메모리 정리를 위한 짧은 대기
        sleep 2
    done
done

# 결과 요약
echo ""
echo "=========================================="
echo "All CNN profiling experiments completed!"
echo "=========================================="
echo ""

if [ ${#failed_experiments[@]} -gt 0 ]; then
    echo "⚠️  Failed experiments:"
    for exp in "${failed_experiments[@]}"; do
        echo "   - $exp"
    done
    echo ""
fi

echo "Generated directories:"
echo ""

# 각 trace 디렉토리 확인
for dir in outputs/*/; do
    if [ -d "$dir" ]; then
        trace_name=$(basename "$dir")

        # CNN 모델 trace만 표시
        if [[ $trace_name == *"densenet"* ]] || \
           [[ $trace_name == *"googlenet"* ]] || \
           [[ $trace_name == *"inception"* ]] || \
           [[ $trace_name == *"resnet"* ]] || \
           [[ $trace_name == *"vgg"* ]]; then

            echo "📁 $trace_name"

            # ET 파일 크기 확인
            et_files=$(find "$dir" -name "*.et" 2>/dev/null)
            if [ -n "$et_files" ]; then
                total_size=0
                count=0
                for et in $et_files; do
                    if [ -f "$et" ]; then
                        size=$(stat -f%z "$et" 2>/dev/null || stat -c%s "$et" 2>/dev/null)
                        total_mb=$(echo "scale=2; $size / 1048576" | bc 2>/dev/null || echo "0")
                        total_size=$(echo "$total_size + $total_mb" | bc 2>/dev/null || echo "0")
                        ((count++))
                        echo "   ✓ $(basename $et) (${total_mb} MB)"
                    fi
                done
                echo "   Total: ${count} files, ${total_size} MB"
            else
                echo "   ✗ No ET files found"
            fi

            # 분석 파일 확인
            if [ -f "$dir/analysis/stacks.txt" ]; then
                echo "   ✓ analysis/stacks.txt"
            fi

            echo ""
        fi
    fi
done

# 전체 통계
echo "Summary:"
echo "  Models: ${#CNN_MODELS[@]}"
echo "  GPU configs: ${#GPU_CONFIGS[@]}"
echo "  Total experiments: $TOTAL_EXP"
echo "  Successful: $((TOTAL_EXP - ${#failed_experiments[@]}))"
echo "  Failed: ${#failed_experiments[@]}"
echo ""

echo "Directory structure example:"
echo "  ./outputs/"
echo "  ├── resnet50_1gpu_trace/"
echo "  │   ├── host_0.json"
echo "  │   ├── device_0.json"
echo "  │   ├── merged_0.json"
echo "  │   ├── resnet50_1gpu_trace.0.et"
echo "  │   └── analysis/stacks.txt"
echo "  ├── resnet50_2gpu_trace/"
echo "  │   ├── host_{0,1}.json"
echo "  │   ├── device_{0,1}.json"
echo "  │   ├── merged_{0,1}.json"
echo "  │   ├── resnet50_2gpu_trace.{0,1}.et"
echo "  │   └── analysis/stacks.txt"
echo "  └── ..."
echo ""
echo "To view traces in Chrome:"
echo "  1. Open Chrome browser"
echo "  2. Navigate to: chrome://tracing"
echo "  3. Load: ./outputs/<trace_name>/device_0.json"
echo ""
echo "To use with ASTRA-sim:"
echo "  ./AstraSim_Analytical_Congestion_Unaware \\"
echo "    --workload-configuration=./outputs/<trace_name>/<trace_name> \\"
echo "    --system-configuration=system.json"
