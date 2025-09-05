"""Tests for FT-Transformer-based row encoder."""

import pytest
import torch
import numpy as np

from tabgpt.models.row_encoder import (
    RowEncoder, 
    MultiHeadAttention, 
    TransformerBlock,
    PositionalEncoding,
    FeedForward
)
from tabgpt.config import TabGPTConfig


class TestMultiHeadAttention:
    """Test multi-head attention mechanism."""
    
    def test_attention_creation(self):
        """Test attention module creation."""
        attention = MultiHeadAttention(d_model=256, n_heads=8)
        assert attention.d_model == 256
        assert attention.n_heads == 8
        assert attention.d_k == 32
    
    def test_attention_forward(self):
        """Test attention forward pass."""
        batch_size, seq_len, d_model = 4, 10, 256
        n_heads = 8
        
        attention = MultiHeadAttention(d_model, n_heads)
        
        # Create input tensors
        x = torch.randn(batch_size, seq_len, d_model)
        
        # Forward pass
        output, attn_weights = attention(x, x, x)
        
        # Check output shapes
        assert output.shape == (batch_size, seq_len, d_model)
        assert attn_weights.shape == (batch_size, seq_len, seq_len)
        
        # Check attention weights sum to 1
        assert torch.allclose(attn_weights.sum(dim=-1), torch.ones(batch_size, seq_len), atol=1e-6)
    
    def test_attention_with_mask(self):
        """Test attention with masking."""
        batch_size, seq_len, d_model = 2, 8, 128
        n_heads = 4
        
        attention = MultiHeadAttention(d_model, n_heads)
        
        # Create input and mask
        x = torch.randn(batch_size, seq_len, d_model)
        mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        mask[0, -2:] = False  # Mask last 2 positions for first sample
        
        output, attn_weights = attention(x, x, x, mask)
        
        # Check that masked positions have zero attention
        assert torch.allclose(attn_weights[0, :, -2:], torch.zeros(seq_len, 2), atol=1e-6)


class TestPositionalEncoding:
    """Test positional encoding."""
    
    def test_positional_encoding_creation(self):
        """Test positional encoding creation."""
        pos_enc = PositionalEncoding(d_model=256, max_len=512)
        assert pos_enc.pe.shape == (1, 512, 256)
    
    def test_positional_encoding_forward(self):
        """Test positional encoding forward pass."""
        batch_size, seq_len, d_model = 4, 20, 256
        
        pos_enc = PositionalEncoding(d_model)
        x = torch.randn(batch_size, seq_len, d_model)
        
        output = pos_enc(x)
        assert output.shape == x.shape
        
        # Output should be different from input (due to added positional encoding)
        assert not torch.equal(output, x)


class TestFeedForward:
    """Test feed-forward network."""
    
    def test_feedforward_creation(self):
        """Test feed-forward network creation."""
        ff = FeedForward(d_model=256, d_ff=1024)
        assert ff.linear1.in_features == 256
        assert ff.linear1.out_features == 1024
        assert ff.linear2.in_features == 1024
        assert ff.linear2.out_features == 256
    
    def test_feedforward_forward(self):
        """Test feed-forward forward pass."""
        batch_size, seq_len, d_model = 4, 10, 256
        d_ff = 1024
        
        ff = FeedForward(d_model, d_ff)
        x = torch.randn(batch_size, seq_len, d_model)
        
        output = ff(x)
        assert output.shape == x.shape


class TestTransformerBlock:
    """Test transformer block."""
    
    def test_transformer_block_creation(self):
        """Test transformer block creation."""
        block = TransformerBlock(d_model=256, n_heads=8, d_ff=1024)
        assert isinstance(block.attention, MultiHeadAttention)
        assert isinstance(block.feed_forward, FeedForward)
    
    def test_transformer_block_forward(self):
        """Test transformer block forward pass."""
        batch_size, seq_len, d_model = 4, 10, 256
        
        block = TransformerBlock(d_model=d_model, n_heads=8, d_ff=1024)
        x = torch.randn(batch_size, seq_len, d_model)
        
        output, attn_weights = block(x)
        
        assert output.shape == x.shape
        assert attn_weights.shape == (batch_size, seq_len, seq_len)
    
    def test_transformer_block_with_mask(self):
        """Test transformer block with attention mask."""
        batch_size, seq_len, d_model = 2, 8, 256
        
        block = TransformerBlock(d_model=d_model, n_heads=8, d_ff=1024)
        x = torch.randn(batch_size, seq_len, d_model)
        mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        mask[0, -2:] = False
        
        output, attn_weights = block(x, mask)
        
        assert output.shape == x.shape
        assert attn_weights.shape == (batch_size, seq_len, seq_len)


class TestRowEncoder:
    """Test complete row encoder."""
    
    def test_row_encoder_creation(self):
        """Test row encoder creation."""
        config = TabGPTConfig(
            d_model=256,
            n_heads=8,
            n_layers=6,
            embedding_dim=128
        )
        
        encoder = RowEncoder(config)
        assert len(encoder.transformer_blocks) == 6
        assert encoder.feature_projection.in_features == 128
        assert encoder.feature_projection.out_features == 256
    
    def test_row_encoder_forward_cls_pooling(self):
        """Test row encoder forward pass with CLS pooling."""
        config = TabGPTConfig(
            d_model=256,
            n_heads=8,
            n_layers=3,
            embedding_dim=128,
            pooling_strategy='cls'
        )
        
        encoder = RowEncoder(config)
        
        batch_size, n_features, embedding_dim = 4, 10, 128
        features = torch.randn(batch_size, n_features, embedding_dim)
        mask = torch.ones(batch_size, n_features, dtype=torch.bool)
        
        outputs = encoder(features, mask)
        
        assert 'last_hidden_state' in outputs
        assert 'pooler_output' in outputs
        assert 'attention_weights' in outputs
        
        assert outputs['last_hidden_state'].shape == (batch_size, n_features, 256)
        assert outputs['pooler_output'].shape == (batch_size, 256)
        assert len(outputs['attention_weights']) == 3  # n_layers
    
    def test_row_encoder_forward_mean_pooling(self):
        """Test row encoder forward pass with mean pooling."""
        config = TabGPTConfig(
            d_model=128,
            n_heads=4,
            n_layers=2,
            embedding_dim=64,
            pooling_strategy='mean'
        )
        
        encoder = RowEncoder(config)
        
        batch_size, n_features, embedding_dim = 2, 8, 64
        features = torch.randn(batch_size, n_features, embedding_dim)
        mask = torch.ones(batch_size, n_features, dtype=torch.bool)
        mask[0, -2:] = False  # Mask last 2 features for first sample
        
        outputs = encoder(features, mask)
        
        assert outputs['last_hidden_state'].shape == (batch_size, n_features, 128)
        assert outputs['pooler_output'].shape == (batch_size, 128)
    
    def test_row_encoder_forward_max_pooling(self):
        """Test row encoder forward pass with max pooling."""
        config = TabGPTConfig(
            d_model=128,
            n_heads=4,
            n_layers=2,
            embedding_dim=64,
            pooling_strategy='max'
        )
        
        encoder = RowEncoder(config)
        
        batch_size, n_features, embedding_dim = 2, 6, 64
        features = torch.randn(batch_size, n_features, embedding_dim)
        
        outputs = encoder(features)
        
        assert outputs['last_hidden_state'].shape == (batch_size, n_features, 128)
        assert outputs['pooler_output'].shape == (batch_size, 128)
    
    def test_row_encoder_with_positional_encoding(self):
        """Test row encoder with positional encoding."""
        config = TabGPTConfig(
            d_model=256,
            n_heads=8,
            n_layers=2,
            embedding_dim=128,
            use_positional_encoding=True,
            max_features=20
        )
        
        encoder = RowEncoder(config)
        
        batch_size, n_features, embedding_dim = 3, 15, 128
        features = torch.randn(batch_size, n_features, embedding_dim)
        
        outputs = encoder(features)
        
        assert outputs['last_hidden_state'].shape == (batch_size, n_features, 256)
        assert outputs['pooler_output'].shape == (batch_size, 256)
    
    def test_attention_patterns_extraction(self):
        """Test attention pattern extraction."""
        config = TabGPTConfig(
            d_model=128,
            n_heads=4,
            n_layers=2,
            embedding_dim=64
        )
        
        encoder = RowEncoder(config)
        
        batch_size, n_features, embedding_dim = 2, 8, 64
        features = torch.randn(batch_size, n_features, embedding_dim)
        
        # Extract attention patterns from last layer
        attention_patterns = encoder.get_attention_patterns(features, layer_idx=-1)
        
        # With CLS pooling, sequence length is n_features + 1
        expected_seq_len = n_features + 1 if config.pooling_strategy == 'cls' else n_features
        assert attention_patterns.shape == (batch_size, expected_seq_len, expected_seq_len)
    
    def test_feature_importance_computation(self):
        """Test feature importance computation."""
        config = TabGPTConfig(
            d_model=128,
            n_heads=4,
            n_layers=2,
            embedding_dim=64,
            pooling_strategy='cls'
        )
        
        encoder = RowEncoder(config)
        
        batch_size, n_features, embedding_dim = 2, 6, 64
        features = torch.randn(batch_size, n_features, embedding_dim)
        
        importance_scores = encoder.compute_feature_importance(features)
        
        assert importance_scores.shape == (batch_size, n_features)
        
        # Importance scores should be non-negative (attention weights are non-negative)
        assert (importance_scores >= 0).all()
    
    def test_gradient_flow(self):
        """Test that gradients flow properly through the model."""
        config = TabGPTConfig(
            d_model=128,
            n_heads=4,
            n_layers=2,
            embedding_dim=64
        )
        
        encoder = RowEncoder(config)
        
        batch_size, n_features, embedding_dim = 2, 8, 64
        features = torch.randn(batch_size, n_features, embedding_dim, requires_grad=True)
        
        outputs = encoder(features)
        loss = outputs['pooler_output'].sum()
        loss.backward()
        
        # Check that gradients exist
        assert features.grad is not None
        assert not torch.isnan(features.grad).any()
        
        # Check that model parameters have gradients
        for param in encoder.parameters():
            if param.requires_grad:
                assert param.grad is not None
                assert not torch.isnan(param.grad).any()


class TestRowEncoderIntegration:
    """Test row encoder integration with other components."""
    
    def test_different_input_sizes(self):
        """Test row encoder with different input sizes."""
        config = TabGPTConfig(
            d_model=128,
            n_heads=4,
            n_layers=2,
            embedding_dim=64
        )
        
        encoder = RowEncoder(config)
        
        # Test different batch sizes and feature counts
        test_cases = [
            (1, 5, 64),
            (4, 10, 64),
            (8, 20, 64),
            (2, 50, 64)
        ]
        
        for batch_size, n_features, embedding_dim in test_cases:
            features = torch.randn(batch_size, n_features, embedding_dim)
            outputs = encoder(features)
            
            expected_seq_len = n_features + 1 if config.pooling_strategy == 'cls' else n_features
            if config.pooling_strategy == 'cls':
                assert outputs['last_hidden_state'].shape == (batch_size, n_features, 128)
            else:
                assert outputs['last_hidden_state'].shape == (batch_size, n_features, 128)
            assert outputs['pooler_output'].shape == (batch_size, 128)
    
    def test_memory_efficiency(self):
        """Test memory usage with larger inputs."""
        config = TabGPTConfig(
            d_model=256,
            n_heads=8,
            n_layers=3,
            embedding_dim=128
        )
        
        encoder = RowEncoder(config)
        
        # Test with larger input
        batch_size, n_features, embedding_dim = 16, 100, 128
        features = torch.randn(batch_size, n_features, embedding_dim)
        
        # Should not raise memory errors
        outputs = encoder(features)
        
        assert outputs['last_hidden_state'].shape[0] == batch_size
        assert outputs['pooler_output'].shape[0] == batch_size