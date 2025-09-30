"""Data loading utilities for SQuAD dataset."""

import os
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from datasets import load_dataset
from transformers import AutoTokenizer
from typing import Optional, Dict, List


class SQuADDataset(Dataset):
    """SQuAD dataset wrapper for both GPT-2 and BERT."""
    
    def __init__(
        self,
        dataset,
        tokenizer,
        max_length: int = 512,
        model_type: str = "bert"  # "bert" or "gpt2"
    ):
        """
        Args:
            dataset: HuggingFace dataset
            tokenizer: Tokenizer instance
            max_length: Maximum sequence length
            model_type: "bert" for MLM, "gpt2" for CLM
        """
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.model_type = model_type.lower()
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # SQuAD의 context와 question을 결합
        context = item["context"]
        question = item["question"]
        
        if self.model_type == "bert":
            # BERT: [CLS] question [SEP] context [SEP]
            text = f"{question} {self.tokenizer.sep_token} {context}"
        else:
            # GPT-2: question + context (causal)
            text = f"{question} {context}"
        
        # Tokenization
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Squeeze to remove batch dimension
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        
        if self.model_type == "bert":
            # BERT: Masked Language Modeling을 위한 labels
            labels = input_ids.clone()
            
            # 15% of tokens를 masking
            probability_matrix = torch.full(labels.shape, 0.15)
            special_tokens_mask = self.tokenizer.get_special_tokens_mask(
                labels.tolist(), already_has_special_tokens=True
            )
            probability_matrix.masked_fill_(
                torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0
            )
            
            masked_indices = torch.bernoulli(probability_matrix).bool()
            labels[~masked_indices] = -100  # Only compute loss on masked tokens
            
            # 80% of the time, replace with [MASK]
            indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
            input_ids[indices_replaced] = self.tokenizer.mask_token_id
            
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels
            }
        else:
            # GPT-2: Causal Language Modeling
            # Labels are the same as input_ids, shifted inside the model
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": input_ids.clone()
            }


def load_squad_dataset(split: str = "train", cache_dir: Optional[str] = None):
    """
    Load SQuAD dataset from HuggingFace.
    
    Args:
        split: "train" or "validation"
        cache_dir: Cache directory for datasets
    
    Returns:
        Dataset object
    """
    dataset = load_dataset("squad", split=split, cache_dir=cache_dir)
    return dataset


def get_dataloaders(
    model_type: str,
    tokenizer_name: str,
    batch_size: int = 4,
    max_length: int = 512,
    num_workers: int = 4,
    use_distributed: bool = False,
    cache_dir: Optional[str] = None
):
    """
    Create DataLoaders for training and validation.
    
    Args:
        model_type: "bert" or "gpt2"
        tokenizer_name: Tokenizer name (e.g., "bert-base-uncased", "gpt2")
        batch_size: Batch size per GPU
        max_length: Maximum sequence length
        num_workers: Number of workers for data loading
        use_distributed: Use DistributedSampler for DDP
        cache_dir: Cache directory
    
    Returns:
        train_loader, val_loader, tokenizer
    """
    # Docker 이미지 내 사전 다운로드된 토크나이저 경로
    pretrained_path = f"/workspace/pretrained_models/{tokenizer_name}"
    
    # 사전 다운로드된 토크나이저가 있으면 사용, 없으면 다운로드
    if os.path.exists(pretrained_path):
        print(f"Loading tokenizer from Docker image: {pretrained_path}")
        tokenizer = AutoTokenizer.from_pretrained(pretrained_path, local_files_only=True)
    else:
        print(f"Downloading tokenizer: {tokenizer_name}")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, cache_dir=cache_dir)
    
    # GPT-2는 pad_token이 없으므로 추가
    if model_type == "gpt2" and tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load datasets
    train_dataset = load_squad_dataset(split="train", cache_dir=cache_dir)
    val_dataset = load_squad_dataset(split="validation", cache_dir=cache_dir)
    
    # Wrap datasets
    train_dataset = SQuADDataset(
        train_dataset, tokenizer, max_length=max_length, model_type=model_type
    )
    val_dataset = SQuADDataset(
        val_dataset, tokenizer, max_length=max_length, model_type=model_type
    )
    
    # Create samplers
    if use_distributed:
        train_sampler = DistributedSampler(
            train_dataset, shuffle=True, drop_last=True
        )
        val_sampler = DistributedSampler(
            val_dataset, shuffle=False, drop_last=False
        )
    else:
        train_sampler = None
        val_sampler = None
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    return train_loader, val_loader, tokenizer
