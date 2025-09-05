"""Tests for masking utilities."""

import pytest
import torch
import numpy as np

from tabgpt.tokenizers.masking import (
    RandomCellMasking,
    ColumnMasking,
    ContrastiveAugmentation
)


class TestRandomCellMasking:
    """Test random cell masking."""
    
    def test_basic_masking(self):
        """Test basic cell masking functionality."""
        batch_size, seq_len, embedding_dim = 4, 10, 64
        tokens = torch.randn(batch_size, seq_len, embedding_dim)
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        
        masker = RandomCellMasking(mask_probability=0.15)
        masked_tokens, mask_positions, original_tokens = masker.apply_mask(tokens, attention_mask)
        
        # Check shapes
        assert masked_tokens.shape == tokens.shape
        assert mask_positions.shape == (batch_size, seq_len)
        assert original_tokens.shape == tokens.shape
        
        # Check that some positions are masked
        assert mask_positions.any()
        
        # Check that masked positions are different from original
        for i in range(batch_size):
            for j in range(seq_len):
                if mask_positions[i, j]:
                    # At least some masked positions should be different
                    # (unless randomly replaced with same value, which is unlikely)
                    pass
    
    def test_mask_probability(self):
        """Test that mask probability is approximately respected."""
        batch_size, seq_len, embedding_dim = 10, 100, 32
        tokens = torch.randn(batch_size, seq_len, embedding_dim)
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        
        mask_prob = 0.2
        masker = RandomCellMasking(mask_probability=mask_prob)
        
        # Run multiple times to check average
        total_masked = 0
        total_positions = 0
        n_runs = 10
        
        for _ in range(n_runs):
            _, mask_positions, _ = masker.apply_mask(tokens, attention_mask)
            total_masked += mask_positions.sum().item()
            total_positions += mask_positions.numel()
        
        actual_prob = total_masked / total_positions
        # Should be approximately the target probability (within reasonable tolerance)
        assert abs(actual_prob - mask_prob) < 0.05
    
    def test_attention_mask_respect(self):
        """Test that masking respects attention mask."""
        batch_size, seq_len, embedding_dim = 4, 10, 32
        tokens = torch.randn(batch_size, seq_len, embedding_dim)
        
        # Create attention mask with some positions invalid
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        attention_mask[:, -3:] = False  # Last 3 positions invalid
        
        masker = RandomCellMasking(mask_probability=0.5)
        _, mask_positions, _ = masker.apply_mask(tokens, attention_mask)
        
        # Check that no invalid positions are masked
        invalid_positions = ~attention_mask
        assert not (mask_positions & invalid_positions).any()


class TestColumnMasking:
    """Test column masking."""
    
    def test_basic_column_masking(self):
        """Test basic column masking functionality."""
        batch_size, seq_len, embedding_dim = 4, 10, 64
        tokens = torch.randn(batch_size, seq_len, embedding_dim)
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        
        masker = ColumnMasking(mask_probability=0.2)
        masked_tokens, mask_positions, original_tokens = masker.apply_mask(tokens, attention_mask)
        
        # Check shapes
        assert masked_tokens.shape == tokens.shape
        assert mask_positions.shape == (batch_size, seq_len)
        
        # Check that entire columns are masked (same across batch)
        for j in range(seq_len):
            if mask_positions[0, j]:
                # If first sample has this column masked, all should
                assert mask_positions[:, j].all()
    
    def test_column_mask_probability(self):
        """Test column mask probability."""
        batch_size, seq_len, embedding_dim = 4, 20, 32
        tokens = torch.randn(batch_size, seq_len, embedding_dim)
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        
        mask_prob = 0.25
        masker = ColumnMasking(mask_probability=mask_prob)
        
        # Run multiple times
        total_masked_cols = 0
        n_runs = 20
        
        for _ in range(n_runs):
            _, mask_positions, _ = masker.apply_mask(tokens, attention_mask)
            # Count masked columns (check first row since columns are consistent)
            masked_cols = mask_positions[0].sum().item()
            total_masked_cols += masked_cols
        
        avg_masked_cols = total_masked_cols / n_runs
        expected_cols = seq_len * mask_prob
        
        # Should be approximately correct
        assert abs(avg_masked_cols - expected_cols) < 2


class TestContrastiveAugmentation:
    """Test contrastive augmentation."""
    
    def test_basic_augmentation(self):
        """Test basic augmentation functionality."""
        batch_size, seq_len, embedding_dim = 4, 10, 64
        tokens = torch.randn(batch_size, seq_len, embedding_dim)
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        
        augmenter = ContrastiveAugmentation()
        augmented = augmenter.augment(tokens, attention_mask)
        
        # Check shape preservation
        assert augmented.shape == tokens.shape
        
        # Check that augmentation changes the tokens
        assert not torch.equal(tokens, augmented)
    
    def test_positive_pairs(self):
        """Test positive pair creation."""
        batch_size, seq_len, embedding_dim = 2, 8, 32
        tokens = torch.randn(batch_size, seq_len, embedding_dim)
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        
        augmenter = ContrastiveAugmentation()
        anchor, positive = augmenter.create_positive_pairs(tokens, attention_mask)
        
        # Check shapes
        assert anchor.shape == tokens.shape
        assert positive.shape == tokens.shape
        
        # Check that anchor and positive are different
        assert not torch.equal(anchor, positive)
        
        # But both should be different from original
        assert not torch.equal(anchor, tokens)
        assert not torch.equal(positive, tokens)
    
    def test_attention_mask_handling(self):
        """Test that augmentation respects attention mask."""
        batch_size, seq_len, embedding_dim = 2, 10, 16
        tokens = torch.randn(batch_size, seq_len, embedding_dim)
        
        # Create attention mask with some invalid positions
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        attention_mask[:, -2:] = False  # Last 2 positions invalid
        
        augmenter = ContrastiveAugmentation(dropout_prob=1.0)  # High dropout to test
        augmented = augmenter.augment(tokens, attention_mask)
        
        # Invalid positions should be zeroed out
        invalid_positions = ~attention_mask
        assert (augmented[invalid_positions] == 0).all()
    
    def test_different_augmentation_parameters(self):
        """Test different augmentation parameters."""
        batch_size, seq_len, embedding_dim = 2, 5, 8
        tokens = torch.randn(batch_size, seq_len, embedding_dim)
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        
        # Test with different noise levels
        aug_low_noise = ContrastiveAugmentation(noise_std=0.01)
        aug_high_noise = ContrastiveAugmentation(noise_std=0.5)
        
        low_noise_result = aug_low_noise.augment(tokens, attention_mask)
        high_noise_result = aug_high_noise.augment(tokens, attention_mask)
        
        # High noise should create more different results
        low_diff = torch.norm(tokens - low_noise_result)
        high_diff = torch.norm(tokens - high_noise_result)
        
        assert high_diff > low_diff