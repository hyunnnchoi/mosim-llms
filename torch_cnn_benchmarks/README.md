# TensorFlow Benchmarks

## Compatibility Status of CNN Models in TensorFlow 2

| Model             | Compatibility | Last Verified |
|-------------------|---------------|---------------|
| AlexNet           | ✅            | 2024-10-28    |
| DenseNet100_k12   | ✅            | 2024-11-02    |
| DenseNet40_k12    | ✅            | 2024-10-28    |
| GoogLeNet         | ✅            | 2024-11-02    |
| Inception3        | ✅            | 2024-11-02    |
| ResNet110         | ✅            | 2024-11-02    |
| ResNet44          | ✅            | 2024-11-02    |
| ResNet50          | ✅            | 2024-10-28    |
| ResNet56          | ✅            | 2024-11-02    |
| VGG16             | ✅            | 2024-10-28    |

*All models listed have been confirmed to run in TensorFlow 2 in a distributed training environment.*

This repository contains various TensorFlow benchmarks. Currently, it consists of two projects:

1. [PerfZero](https://github.com/tensorflow/benchmarks/tree/master/perfzero): A benchmark framework for TensorFlow.

2. [scripts/tf_cnn_benchmarks](https://github.com/tensorflow/benchmarks/tree/master/scripts/tf_cnn_benchmarks) (no longer maintained): The TensorFlow CNN benchmarks contain TensorFlow 1 benchmarks for several convolutional neural networks.

If you want to run TensorFlow models and measure their performance, also consider the [TensorFlow Official Models](https://github.com/tensorflow/models/tree/master/official)

---

## PyTorch CNN Benchmarks

The repository now includes a PyTorch port located in `scripts/torch_cnn_benchmarks`. The port supports the following models:

- `densenet40_k12`
- `densenet100_k12`
- `googlenet`
- `inception3`
- `resnet44`
- `resnet110`
- `resnet50`
- `vgg16`

### Usage

Install dependencies (PyTorch and TorchVision):

```bash
pip install torch torchvision
```

Run single-process training:

```bash
python torch_cnn_benchmarks/benchmark.py --model resnet44 --dataset cifar10 --data-dir /path/to/cifar --epochs 100
```

Benchmark with synthetic data:

```bash
python torch_cnn_benchmarks/benchmark.py --model resnet50 --dataset synthetic --synthetic --num-classes 1000 --batch-size 256
```

Launch distributed data parallel (DDP) training (FSDP is intentionally not used):

```bash
torchrun --nproc_per_node=4 torch_cnn_benchmarks/benchmark.py \
  --model resnet50 --dataset imagenet --data-dir /path/to/imagenet --batch-size 128 --epochs 90
```

Quick training with Chakra tracing:

```bash
# Single GPU
./run_cnn_quick.sh resnet50 1 1

# Multi-GPU (2 GPUs)
./run_cnn_quick.sh resnet50 2 1

# Multi-GPU (4 GPUs)
./run_cnn_quick.sh resnet50 4 1
```

Additional options are available via `--help`, including mixed-precision training, SyncBatchNorm, gradient accumulation, and JSON logging.

