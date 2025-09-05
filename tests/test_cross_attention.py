"""Tests for cross-attention fusion mechanism."""

import pytest
import torch
import numpy as np

from tabgpt.models.cross_attention import (
    CrossAttentionLayer,
    CrossAttentionFusion
)


class TestCrossAttentionLayer:
    """Test cross-attention layer."""
    
    def test_cross_attention_creation(self):
        """Test cross-attention layer creation."""
        layer = CrossAttentionLayer(d_model=256, n_heads=8)
        assert layer.d_model == 256
        assert layer.n_heads == 8
        assert layer.d_k == 32
    
    def test_cross_attention_forward(self):
        """Test cross-attention forward pass."""
        batch_size, n_features, d_model = 4, 10, 64
        layer = CrossAttentionLayer(d_model=d_model, n_heads=8)
        
        row_embeddings = torch.randn(batch_size, n_features, d_model)
        column_embeddings = torch.randn(n_features, d_model)
        
        outputs = layer(row_embeddings, column_embeddings)
        
        assert 'fused_row_embeddings' in outputs
        assert 'fused_column_embeddings' in outputs
        assert 'row_to_column_attention' in outputs
        assert 'column_to_row_attention' in outputs
        
        assert outputs['fused_row_embeddings'].shape == (batch_size, n_features, d_model)
        assert outputs['fused_column_embeddings'].shape == (batch_size, n_features, d_model)
        assert outputs['row_to_column_attention'].shape == (batch_size, n_features, n_features)
        assert outputs['column_to_row_attention'].shape == (batch_size, n_features, n_features)
    
    def test_cross_attention_with_masks(self):
        """Test cross-attention with attention masks."""
        batch_size, n_features, d_model = 3, 8, 32
        layer = CrossAttentionLayer(d_model=d_model, n_heads=4)
        
        row_embeddings = torch.randn(batch_size, n_features, d_model)
        column_embeddings = torch.randn(n_features, d_model)
        
        # Create masks
        row_mask = torch.ones(batch_size, n_features, dtype=torch.bool)
        row_mask[:, -2:] = False  # Mask last 2 features
        
        col_mask = torch.ones(n_features, dtype=torch.bool)
        col_mask[-3:] = False  # Mask last 3 features
        
        outputs = layer(row_embeddings, column_embeddings, row_mask, col_mask)
        
        assert outputs['fused_row_embeddings'].shape == (batch_size, n_features, d_model)
        assert outputs['fused_column_embeddings'].shape == (batch_size, n_features, d_model)
        
        # Check that attention to masked positions is zero
        row_to_col_attn = outputs['row_to_column_attention']
        col_to_row_attn = outputs['column_to_row_attention']
        
        # Attention from masked rows should be zero
        assert row_to_col_attn[:, -2:, :].sum() == 0
        # Attention to masked columns should be zero
        assert row_to_col_attn[:, :, -3:].sum() == 0
    
    def test_attention_weights_properties(self):
        """Test attention weights have correct properties."""
        batch_size, n_features, d_model = 2, 6, 64
        layer = CrossAttentionLayer(d_model=d_model, n_heads=8)
        
        row_embeddings = torch.randn(batch_size, n_features, d_model)
        column_embeddings = torch.randn(n_features, d_model)
        
        outputs = layer(row_embeddings, column_embeddings)
        
        row_to_col_attn = outputs['row_to_column_attention']
        col_to_row_attn = outputs['column_to_row_attention']
        
        # Attention weights should sum to 1 along the last dimension
        assert torch.allclose(row_to_col_attn.sum(dim=-1), torch.ones(batch_size, n_features), atol=1e-6)
        assert torch.allclose(col_to_row_attn.sum(dim=-1), torch.ones(batch_size, n_features), atol=1e-6)
        
        # Attention weights should be non-negative
        assert (row_to_col_attn >= 0).all()
        assert (col_to_row_attn >= 0).all()


class TestCrossAttentionFusion:
    """Test complete cross-attention fusion mechanism."""
    
    def test_fusion_creation(self):
        """Test fusion mechanism creation."""
        fusion = CrossAttentionFusion(d_model=128, n_heads=8, n_layers=2)
        assert fusion.d_model == 128
        assert fusion.n_layers == 2
        assert len(fusion.cross_attention_layers) == 2
        assert len(fusion.row_layer_norms) == 2
        assert len(fusion.col_layer_norms) == 2
    
    def test_fusion_forward(self):
        """Test fusion forward pass."""
        batch_size, n_features, d_model = 3, 12, 64
        fusion = CrossAttentionFusion(d_model=d_model, n_heads=8, n_layers=2)
        
        row_embeddings = torch.randn(batch_size, n_features, d_model)
        column_embeddings = torch.randn(n_features, d_model)
        
        outputs = fusion(row_embeddings, column_embeddings)
        
        assert 'fused_representations' in outputs
        assert 'enhanced_row_embeddings' in outputs
        assert 'enhanced_column_embeddings' in outputs
        
        assert outputs['fused_representations'].shape == (batch_size, n_features, d_model)
        assert outputs['enhanced_row_embeddings'].shape == (batch_size, n_features, d_model)
        assert outputs['enhanced_column_embeddings'].shape == (n_features, d_model)
    
    def test_fusion_with_attention_weights(self):
        """Test fusion with attention weights return."""
        batch_size, n_features, d_model = 2, 8, 32
        fusion = CrossAttentionFusion(d_model=d_model, n_heads=4, n_layers=2)
        
        row_embeddings = torch.randn(batch_size, n_features, d_model)
        column_embeddings = torch.randn(n_features, d_model)
        
        outputs = fusion(row_embeddings, column_embeddings, return_attention_weights=True)
        
        assert 'attention_weights' in outputs
        attention_weights = outputs['attention_weights']
        assert len(attention_weights) == 2  # n_layers
        
        for layer_attn in attention_weights:
            assert 'row_to_column' in layer_attn
            assert 'column_to_row' in layer_attn
            assert layer_attn['row_to_column'].shape == (batch_size, n_features, n_features)
            assert layer_attn['column_to_row'].shape == (batch_size, n_features, n_features)
    
    def test_different_fusion_strategies(self):
        """Test different fusion strategies."""
        batch_size, n_features, d_model = 2, 6, 64
        
        strategies = ['add', 'concat', 'gate']
        
        for strategy in strategies:
            fusion = CrossAttentionFusion(
                d_model=d_model, 
                n_heads=8, 
                n_layers=1,
                fusion_strategy=strategy
            )
            
            row_embeddings = torch.randn(batch_size, n_features, d_model)
            column_embeddings = torch.randn(n_features, d_model)
            
            outputs = fusion(row_embeddings, column_embeddings)
            
            assert outputs['fused_representations'].shape == (batch_size, n_features, d_model)
            print(f"Strategy '{strategy}' works correctly")
    
    def test_fusion_with_masks(self):
        """Test fusion with attention masks."""
        batch_size, n_features, d_model = 3, 10, 32
        fusion = CrossAttentionFusion(d_model=d_model, n_heads=4, n_layers=2)
        
        row_embeddings = torch.randn(batch_size, n_features, d_model)
        column_embeddings = torch.randn(n_features, d_model)
        
        # Create masks
        row_mask = torch.ones(batch_size, n_features, dtype=torch.bool)
        row_mask[:, -3:] = False
        
        col_mask = torch.ones(n_features, dtype=torch.bool)
        col_mask[-2:] = False
        
        outputs = fusion(row_embeddings, column_embeddings, row_mask, col_mask)
        
        assert outputs['fused_representations'].shape == (batch_size, n_features, d_model)
        
        # Outputs should not contain NaN values
        assert not torch.isnan(outputs['fused_representations']).any()
        assert not torch.isnan(outputs['enhanced_row_embeddings']).any()
        assert not torch.isnan(outputs['enhanced_column_embeddings']).any()
    
    def test_residual_connections(self):
        """Test residual connections."""
        batch_size, n_features, d_model = 2, 5, 64
        
        # Test with residual connections
        fusion_with_residual = CrossAttentionFusion(
            d_model=d_model, n_heads=8, n_layers=1, use_residual=True
        )
        
        # Test without residual connections
        fusion_without_residual = CrossAttentionFusion(
            d_model=d_model, n_heads=8, n_layers=1, use_residual=False
        )
        
        row_embeddings = torch.randn(batch_size, n_features, d_model)
        column_embeddings = torch.randn(n_features, d_model)
        
        outputs_with = fusion_with_residual(row_embeddings, column_embeddings)
        outputs_without = fusion_without_residual(row_embeddings, column_embeddings)
        
        # Outputs should be different
        assert not torch.equal(
            outputs_with['enhanced_row_embeddings'],
            outputs_without['enhanced_row_embeddings']
        )
    
    def test_attention_patterns(self):
        """Test attention pattern extraction."""
        batch_size, n_features, d_model = 2, 6, 32
        fusion = CrossAttentionFusion(d_model=d_model, n_heads=4, n_layers=2)
        
        row_embeddings = torch.randn(batch_size, n_features, d_model)
        column_embeddings = torch.randn(n_features, d_model)
        
        patterns = fusion.get_attention_patterns(row_embeddings, column_embeddings)
        
        assert 'row_to_column_attention' in patterns
        assert 'column_to_row_attention' in patterns
        assert 'layer_attention_weights' in patterns
        
        assert patterns['row_to_column_attention'].shape == (batch_size, n_features, n_features)
        assert patterns['column_to_row_attention'].shape == (batch_size, n_features, n_features)
    
    def test_interaction_scores(self):
        """Test feature interaction score computation."""
        batch_size, n_features, d_model = 3, 8, 64
        fusion = CrossAttentionFusion(d_model=d_model, n_heads=8, n_layers=2)
        
        row_embeddings = torch.randn(batch_size, n_features, d_model)
        column_embeddings = torch.randn(n_features, d_model)
        
        interaction_scores = fusion.compute_interaction_scores(row_embeddings, column_embeddings)
        
        assert 'interaction_matrix' in interaction_scores
        assert 'feature_importance' in interaction_scores
        assert 'row_to_column_strength' in interaction_scores
        assert 'column_to_row_strength' in interaction_scores
        
        interaction_matrix = interaction_scores['interaction_matrix']
        feature_importance = interaction_scores['feature_importance']
        
        assert interaction_matrix.shape == (batch_size, n_features, n_features)
        assert feature_importance.shape == (batch_size, n_features)
        
        # Feature importance should sum to 1
        assert torch.allclose(feature_importance.sum(dim=1), torch.ones(batch_size), atol=1e-6)
        
        # Feature importance should be non-negative
        assert (feature_importance >= 0).all()
        
        # Interaction matrix values should be reasonable (between 0 and 1)
        assert (interaction_matrix >= 0).all()
        assert (interaction_matrix <= 1).all()
    
    def test_interaction_scores_with_mask(self):
        """Test interaction scores with attention mask."""
        batch_size, n_features, d_model = 2, 10, 32
        fusion = CrossAttentionFusion(d_model=d_model, n_heads=4, n_layers=1)
        
        row_embeddings = torch.randn(batch_size, n_features, d_model)
        column_embeddings = torch.randn(n_features, d_model)
        
        # Create mask
        row_mask = torch.ones(batch_size, n_features, dtype=torch.bool)
        row_mask[:, -3:] = False  # Mask last 3 features
        
        interaction_scores = fusion.compute_interaction_scores(
            row_embeddings, column_embeddings, row_mask
        )
        
        feature_importance = interaction_scores['feature_importance']
        
        # Masked features should have zero importance
        assert (feature_importance[:, -3:] == 0).all()
        
        # Valid features should sum to 1
        valid_importance = feature_importance[:, :-3]
        assert torch.allclose(valid_importance.sum(dim=1), torch.ones(batch_size), atol=1e-6)
    
    def test_gradient_flow(self):
        """Test gradient flow through fusion mechanism."""
        batch_size, n_features, d_model = 2, 5, 32
        fusion = CrossAttentionFusion(d_model=d_model, n_heads=4, n_layers=2)
        
        row_embeddings = torch.randn(batch_size, n_features, d_model, requires_grad=True)
        column_embeddings = torch.randn(n_features, d_model, requires_grad=True)
        
        outputs = fusion(row_embeddings, column_embeddings)
        
        # Compute loss
        loss = outputs['fused_representations'].sum()
        loss.backward()
        
        # Check gradients
        assert row_embeddings.grad is not None
        assert column_embeddings.grad is not None
        assert not torch.isnan(row_embeddings.grad).any()
        assert not torch.isnan(column_embeddings.grad).any()
        
        # Check model parameter gradients
        for param in fusion.parameters():
            if param.requires_grad:
                assert param.grad is not None
                assert not torch.isnan(param.grad).any()
    
    def test_temperature_scaling(self):
        """Test temperature scaling in attention."""
        batch_size, n_features, d_model = 2, 6, 32
        
        # Test different temperatures
        fusion_low_temp = CrossAttentionFusion(
            d_model=d_model, n_heads=4, n_layers=1, temperature=0.5
        )
        fusion_high_temp = CrossAttentionFusion(
            d_model=d_model, n_heads=4, n_layers=1, temperature=2.0
        )
        
        row_embeddings = torch.randn(batch_size, n_features, d_model)
        column_embeddings = torch.randn(n_features, d_model)
        
        patterns_low = fusion_low_temp.get_attention_patterns(row_embeddings, column_embeddings)
        patterns_high = fusion_high_temp.get_attention_patterns(row_embeddings, column_embeddings)
        
        # Low temperature should produce sharper attention (higher max values)
        low_temp_max = patterns_low['row_to_column_attention'].max(dim=-1)[0].mean()
        high_temp_max = patterns_high['row_to_column_attention'].max(dim=-1)[0].mean()
        
        assert low_temp_max > high_temp_max
    
    def test_multiple_layers_effect(self):
        """Test effect of multiple cross-attention layers."""
        batch_size, n_features, d_model = 2, 8, 64
        
        fusion_1_layer = CrossAttentionFusion(d_model=d_model, n_heads=8, n_layers=1)
        fusion_3_layers = CrossAttentionFusion(d_model=d_model, n_heads=8, n_layers=3)
        
        row_embeddings = torch.randn(batch_size, n_features, d_model)
        column_embeddings = torch.randn(n_features, d_model)
        
        outputs_1 = fusion_1_layer(row_embeddings, column_embeddings)
        outputs_3 = fusion_3_layers(row_embeddings, column_embeddings)
        
        # Multiple layers should produce different outputs
        assert not torch.equal(
            outputs_1['fused_representations'],
            outputs_3['fused_representations']
        )
        
        # Both should have valid shapes
        assert outputs_1['fused_representations'].shape == (batch_size, n_features, d_model)
        assert outputs_3['fused_representations'].shape == (batch_size, n_features, d_model)