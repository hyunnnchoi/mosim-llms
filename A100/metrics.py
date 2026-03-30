"""Metrics collection for A100 interference experiments."""

import json
import os
import time
from dataclasses import dataclass, field
from typing import List, Optional

import torch


@dataclass
class ExperimentMetrics:
    """Collects and exports timing/performance metrics for one training run."""

    model: str
    mode: str  # "solo" or "pair"
    partner: Optional[str]
    gpus: List[int]
    batch_size: int
    total_steps: int
    warmup_steps: int

    # Filled during / after training
    end_to_end_sec: float = 0.0
    measured_steps: int = 0
    iter_times: List[float] = field(default_factory=list)
    losses: List[float] = field(default_factory=list)
    gpu_memory_allocated_mb: float = 0.0
    gpu_memory_reserved_mb: float = 0.0

    # ── derived properties ──

    @property
    def iter_mean(self) -> float:
        return sum(self.iter_times) / len(self.iter_times) if self.iter_times else 0.0

    @property
    def iter_std(self) -> float:
        if len(self.iter_times) < 2:
            return 0.0
        mean = self.iter_mean
        variance = sum((t - mean) ** 2 for t in self.iter_times) / (len(self.iter_times) - 1)
        return variance ** 0.5

    @property
    def iter_min(self) -> float:
        return min(self.iter_times) if self.iter_times else 0.0

    @property
    def iter_max(self) -> float:
        return max(self.iter_times) if self.iter_times else 0.0

    @property
    def throughput(self) -> float:
        """Effective samples/sec across all GPUs."""
        if self.iter_mean == 0:
            return 0.0
        return self.batch_size * len(self.gpus) / self.iter_mean

    # ── serialization ──

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "mode": self.mode,
            "partner": self.partner,
            "gpus": self.gpus,
            "batch_size_per_gpu": self.batch_size,
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
            "end_to_end_sec": round(self.end_to_end_sec, 3),
            "measured_steps": self.measured_steps,
            "iter_times_sec": {
                "mean": round(self.iter_mean, 6),
                "std": round(self.iter_std, 6),
                "min": round(self.iter_min, 6),
                "max": round(self.iter_max, 6),
                "per_step": [round(t, 6) for t in self.iter_times],
            },
            "throughput_samples_per_sec": round(self.throughput, 2),
            "avg_loss": round(sum(self.losses) / len(self.losses), 6) if self.losses else None,
            "gpu_memory_allocated_mb": round(self.gpu_memory_allocated_mb, 1),
            "gpu_memory_reserved_mb": round(self.gpu_memory_reserved_mb, 1),
        }

    def save(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        if self.mode == "solo":
            filename = f"{self.model}_solo.json"
        else:
            filename = f"{self.model}_with_{self.partner}.json"
        filepath = os.path.join(output_dir, filename)
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        print(f"[Metrics] Saved to {filepath}")
