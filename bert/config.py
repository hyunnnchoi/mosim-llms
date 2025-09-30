"""BERT configuration."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class BERTConfig:
    """BERT training configuration."""
    
    # Model architecture
    model_name: str = "bert-base-uncased"  # or "bert-large-uncased"
    vocab_size: int = 30522
    max_position_embeddings: int = 512
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    
    # Training
    batch_size: int = 4
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    num_epochs: int = 3
    warmup_steps: int = 500
    max_grad_norm: float = 1.0
    
    # Data
    max_seq_length: int = 512
    num_workers: int = 4
    cache_dir: Optional[str] = None
    
    # Distributed
    num_gpus: int = 1  # 1, 2, or 8
    backend: str = "nccl"
    
    # Checkpointing
    save_dir: str = "./checkpoints/bert"
    save_steps: int = 1000
    logging_steps: int = 100
    
    # Chakra tracing
    enable_tracing: bool = True
    trace_output_dir: str = "./outputs"
    trace_name: str = "bert_trace"
    trace_wait_steps: int = 2
    trace_warmup_steps: int = 2
    trace_active_steps: int = 6
    
    # Evaluation
    eval_steps: int = 500
    
    def __post_init__(self):
        """Validate configuration."""
        assert self.num_gpus in [1, 2, 8], "num_gpus must be 1, 2, or 8"
        assert self.batch_size > 0, "batch_size must be positive"
