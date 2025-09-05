"""Masking utilities for pre-training objectives."""

import torch
import numpy as np
from typing import Tuple, List, Optional
import random


class MaskingStrategy:
    """Base class for masking strategies."""
    
    def __init__(self, mask_probability: float = 0.15):
        self.mask_probability = mask_probability
    
    def apply_mask(
        self, 
        tokens: torch.Tensor, 
        attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply masking to tokens.
        
        Args:
            tokens: Input tokens [batch_size, seq_len, embedding_dim]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Tuple of (masked_tokens, mask_positions, original_tokens)
        """
        raise NotImplementedError


class RandomCellMasking(MaskingStrategy):
    """Random cell masking for Masked Cell Modeling objective."""
    
    def __init__(
        self, 
        mask_probability: float = 0.15,
        replace_probability: float = 0.8,
        random_probability: float = 0.1
    ):
        super().__init__(mask_probability)
        self.replace_probability = replace_probability
        self.random_probability = random_probability
    
    def apply_mask(
        self, 
        tokens: torch.Tensor, 
        attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply random cell masking."""
        batch_size, seq_len, embedding_dim = tokens.shape
        
        # Create mask for positions to mask
        mask_positions = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        
        for i in range(batch_size):
            # Get valid positions (where attention_mask is True)
            valid_positions = attention_mask[i].nonzero(as_tuple=True)[0]
            
            if len(valid_positions) > 0:
                # Sample positions to mask
                n_mask = max(1, int(len(valid_positions) * self.mask_probability))
                mask_indices = torch.randperm(len(valid_positions))[:n_mask]
                positions_to_mask = valid_positions[mask_indices]
                mask_positions[i, positions_to_mask] = True
        
        # Create masked tokens
        masked_tokens = tokens.clone()
        original_tokens = tokens.clone()
        
        # Apply masking strategy
        for i in range(batch_size):
            for j in range(seq_len):
                if mask_positions[i, j]:
                    rand_val = random.random()
                    
                    if rand_val < self.replace_probability:
                        # Replace with mask token (zeros)
                        masked_tokens[i, j] = torch.zeros(embedding_dim)
                    elif rand_val < self.replace_probability + self.random_probability:
                        # Replace with random token
                        masked_tokens[i, j] = torch.randn(embedding_dim)
                    # Else keep original (remaining 10%)
        
        return masked_tokens, mask_positions, original_tokens


class ColumnMasking(MaskingStrategy):
    """Column-level masking for Masked Column Modeling objective."""
    
    def __init__(self, mask_probability: float = 0.2):
        super().__init__(mask_probability)
    
    def apply_mask(
        self, 
        tokens: torch.Tensor, 
        attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply column-level masking."""
        batch_size, seq_len, embedding_dim = tokens.shape
        
        # Create mask for columns to mask
        mask_positions = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        
        # Sample columns to mask (same across batch)
        n_mask = max(1, int(seq_len * self.mask_probability))
        columns_to_mask = torch.randperm(seq_len)[:n_mask]
        
        # Apply mask to all samples in batch
        mask_positions[:, columns_to_mask] = True
        
        # Only mask where attention is valid
        mask_positions = mask_positions & attention_mask
        
        # Create masked tokens
        masked_tokens = tokens.clone()
        masked_tokens[mask_positions] = torch.zeros(embedding_dim)
        
        return masked_tokens, mask_positions, tokens.clone()


class ContrastiveAugmentation:
    """Data augmentation for contrastive learning."""
    
    def __init__(
        self,
        noise_std: float = 0.1,
        dropout_prob: float = 0.1,
        perturbation_std: float = 0.05
    ):
        self.noise_std = noise_std
        self.dropout_prob = dropout_prob
        self.perturbation_std = perturbation_std
    
    def augment(
        self, 
        tokens: torch.Tensor, 
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply augmentation to create positive pairs.
        
        Args:
            tokens: Input tokens [batch_size, seq_len, embedding_dim]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Augmented tokens
        """
        augmented = tokens.clone()
        
        # Gaussian noise injection
        noise = torch.randn_like(augmented) * self.noise_std
        augmented = augmented + noise
        
        # Feature dropout
        dropout_mask = torch.rand(augmented.shape[:2]) > self.dropout_prob
        dropout_mask = dropout_mask.unsqueeze(-1).expand_as(augmented)
        dropout_mask = dropout_mask & attention_mask.unsqueeze(-1)
        augmented = augmented * dropout_mask.float()
        
        # Value perturbation
        perturbation = torch.randn_like(augmented) * self.perturbation_std
        augmented = augmented + perturbation
        
        # Zero out invalid positions according to attention mask
        attention_mask_expanded = attention_mask.unsqueeze(-1).expand_as(augmented)
        augmented = augmented * attention_mask_expanded.float()
        
        return augmented
    
    def create_positive_pairs(
        self, 
        tokens: torch.Tensor, 
        attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create positive pairs for contrastive learning.
        
        Returns:
            Tuple of (anchor_tokens, positive_tokens)
        """
        anchor = self.augment(tokens, attention_mask)
        positive = self.augment(tokens, attention_mask)
        
        return anchor, positive


def create_attention_mask_with_masking(
    original_mask: torch.Tensor,
    mask_positions: torch.Tensor
) -> torch.Tensor:
    """
    Create attention mask that accounts for masking.
    
    Args:
        original_mask: Original attention mask
        mask_positions: Positions that are masked
        
    Returns:
        Updated attention mask
    """
    # Keep original attention pattern
    # Masked positions still participate in attention
    return original_mask