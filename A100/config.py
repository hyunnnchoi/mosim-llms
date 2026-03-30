"""A100 interference experiment configurations.

A100 80GB optimized batch sizes - designed to stress GPU memory/compute
enough to reveal interference when two models share the same node.
"""

# Per-model configuration for A100 8-GPU experiments
MODEL_CONFIGS = {
    # ── LLM models (SQuAD dataset, seq_len=512) ──
    "gpt2": {
        "type": "llm",
        "script": "train_gpt2.py",
        "batch_size": 48,
        "max_seq_length": 512,
        "learning_rate": 5e-5,
        "model_name": "gpt2",
    },
    "bert": {
        "type": "llm",
        "script": "train_bert.py",
        "batch_size": 48,
        "max_seq_length": 512,
        "learning_rate": 5e-5,
        "model_name": "bert-base-uncased",
    },
    # ── CNN models - CIFAR-10 (32x32) ──
    "resnet44": {
        "type": "cnn",
        "script": "train_cnn.py",
        "batch_size": 512,
        "dataset": "cifar10",
        "num_classes": 10,
        "image_size": 32,
        "learning_rate": 0.1,
    },
    "resnet110": {
        "type": "cnn",
        "script": "train_cnn.py",
        "batch_size": 512,
        "dataset": "cifar10",
        "num_classes": 10,
        "image_size": 32,
        "learning_rate": 0.1,
    },
    "densenet40_k12": {
        "type": "cnn",
        "script": "train_cnn.py",
        "batch_size": 512,
        "dataset": "cifar10",
        "num_classes": 10,
        "image_size": 32,
        "learning_rate": 0.1,
    },
    "densenet100_k12": {
        "type": "cnn",
        "script": "train_cnn.py",
        "batch_size": 256,
        "dataset": "cifar10",
        "num_classes": 10,
        "image_size": 32,
        "learning_rate": 0.1,
    },
    # ── CNN models - ImageNet scale (synthetic data) ──
    "resnet50": {
        "type": "cnn",
        "script": "train_cnn.py",
        "batch_size": 256,
        "dataset": "synthetic",
        "num_classes": 1000,
        "image_size": 224,
        "learning_rate": 0.1,
    },
    "vgg16": {
        "type": "cnn",
        "script": "train_cnn.py",
        "batch_size": 128,
        "dataset": "synthetic",
        "num_classes": 1000,
        "image_size": 224,
        "learning_rate": 0.01,
    },
    "googlenet": {
        "type": "cnn",
        "script": "train_cnn.py",
        "batch_size": 256,
        "dataset": "synthetic",
        "num_classes": 1000,
        "image_size": 224,
        "learning_rate": 0.1,
    },
    "inception3": {
        "type": "cnn",
        "script": "train_cnn.py",
        "batch_size": 128,
        "dataset": "synthetic",
        "num_classes": 1000,
        "image_size": 299,
        "learning_rate": 0.1,
    },
}

ALL_MODELS = list(MODEL_CONFIGS.keys())
LLM_MODELS = [k for k, v in MODEL_CONFIGS.items() if v["type"] == "llm"]
CNN_MODELS = [k for k, v in MODEL_CONFIGS.items() if v["type"] == "cnn"]

# Experiment defaults
DEFAULT_TOTAL_STEPS = 100
DEFAULT_WARMUP_STEPS = 10
