"""BERT model implementation using HuggingFace."""

import torch
import torch.nn as nn
from transformers import BertForMaskedLM, BertConfig as HFBertConfig
from typing import Optional, Dict


class BERTModel(nn.Module):
    """BERT wrapper for masked language modeling."""
    
    def __init__(self, config):
        """
        Args:
            config: BERTConfig instance
        """
        super().__init__()
        
        # HuggingFace BERT config
        hf_config = HFBertConfig(
            vocab_size=config.vocab_size,
            max_position_embeddings=config.max_position_embeddings,
            hidden_size=config.hidden_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            intermediate_size=config.intermediate_size,
        )
        
        # Load pretrained or initialize from scratch
        try:
            self.model = BertForMaskedLM.from_pretrained(
                config.model_name,
                config=hf_config
            )
            print(f"Loaded pretrained BERT model: {config.model_name}")
        except:
            self.model = BertForMaskedLM(hf_config)
            print("Initialized BERT model from scratch")
        
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
        self.model = BertForMaskedLM.from_pretrained(load_path)
