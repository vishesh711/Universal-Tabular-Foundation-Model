"""Tensor manipulation utilities."""

import torch
import numpy as np
from typing import List, Optional, Union


def create_attention_mask(
    sequence_lengths: List[int],
    max_length: Optional[int] = None
) -> torch.Tensor:
    """
    Create attention mask from sequence lengths.
    
    Args:
        sequence_lengths: List of sequence lengths
        max_length: Maximum sequence length (auto-computed if None)
        
    Returns:
        Attention mask tensor [batch_size, max_length]
    """
    if max_length is None:
        max_length = max(sequence_lengths)
    
    batch_size = len(sequence_lengths)
    mask = torch.zeros(batch_size, max_length, dtype=torch.bool)
    
    for i, length in enumerate(sequence_lengths):
        mask[i, :length] = True
    
    return mask


def pad_sequences(
    sequences: List[torch.Tensor],
    padding_value: float = 0.0,
    max_length: Optional[int] = None
) -> torch.Tensor:
    """
    Pad sequences to same length.
    
    Args:
        sequences: List of tensors to pad
        padding_value: Value to use for padding
        max_length: Maximum length (auto-computed if None)
        
    Returns:
        Padded tensor [batch_size, max_length, ...]
    """
    if not sequences:
        raise ValueError("Empty sequence list")
    
    if max_length is None:
        max_length = max(seq.size(0) for seq in sequences)
    
    # Get dimensions
    batch_size = len(sequences)
    feature_dims = sequences[0].shape[1:]
    
    # Create padded tensor
    padded = torch.full(
        (batch_size, max_length, *feature_dims),
        padding_value,
        dtype=sequences[0].dtype
    )
    
    # Fill with actual sequences
    for i, seq in enumerate(sequences):
        length = min(seq.size(0), max_length)
        padded[i, :length] = seq[:length]
    
    return padded


def apply_mask(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    fill_value: float = 0.0
) -> torch.Tensor:
    """
    Apply mask to tensor.
    
    Args:
        tensor: Input tensor
        mask: Boolean mask (True = keep, False = mask)
        fill_value: Value to fill masked positions
        
    Returns:
        Masked tensor
    """
    return tensor.masked_fill(~mask.unsqueeze(-1), fill_value)


def compute_sequence_lengths(attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Compute sequence lengths from attention mask.
    
    Args:
        attention_mask: Boolean attention mask [batch_size, seq_length]
        
    Returns:
        Sequence lengths [batch_size]
    """
    return attention_mask.sum(dim=1)