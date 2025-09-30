"""GPT-2 model implementation using HuggingFace."""

import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Config as HFGPTConfig
from typing import Optional, Dict


class GPT2Model(nn.Module):
    """GPT-2 wrapper for causal language modeling."""
    
    def __init__(self, config):
        """
        Args:
            config: GPT2Config instance
        """
        super().__init__()
        
        # HuggingFace GPT-2 config
        hf_config = HFGPTConfig(
            vocab_size=config.vocab_size,
            n_positions=config.max_position_embeddings,
            n_embd=config.hidden_size,
            n_layer=config.num_hidden_layers,
            n_head=config.num_attention_heads,
            n_inner=config.intermediate_size,
        )
        
        # Load pretrained or initialize from scratch
        try:
            self.model = GPT2LMHeadModel.from_pretrained(
                config.model_name,
                config=hf_config
            )
            print(f"Loaded pretrained GPT-2 model: {config.model_name}")
        except:
            self.model = GPT2LMHeadModel(hf_config)
            print("Initialized GPT-2 model from scratch")
        
        self.config = config
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            labels: [batch_size, seq_len]
        
        Returns:
            dict with 'loss' and 'logits'
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        return {
            "loss": outputs.loss,
            "logits": outputs.logits
        }
    
    def save_pretrained(self, save_path: str):
        """Save model checkpoint."""
        self.model.save_pretrained(save_path)
    
    def load_pretrained(self, load_path: str):
        """Load model checkpoint."""
        self.model = GPT2LMHeadModel.from_pretrained(load_path)
