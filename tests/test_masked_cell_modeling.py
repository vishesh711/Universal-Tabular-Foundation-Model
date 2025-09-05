"""Tests for Masked Cell Modeling pre-training objective."""

import pytest
import torch
import numpy as np
import pandas as pd

from tabgpt.training.masked_cell_modeling import (
    MaskedCellModelingObjective,
    MaskedCellModelingHead,
    CategoricalPredictionHead,
    NumericalPredictionHead,
    MaskedCellOutput
)
from tabgpt.tokenizers import TabularTokenizer
from tabgpt.models import TabGPTModel
from tabgpt.config import TabGPTConfig


class TestCategoricalPredictionHead:
    """Test categorical prediction head."""
    
    def test_categorical_head_creation(self):
        """Test categorical prediction head creation."""
        head = CategoricalPredictionHead(d_model=128, vocab_size=1000)
        assert head.d_model == 128
        assert head.vocab_size == 1000
    
    def test_categorical_head_forward(self):
        """Test categorical prediction head forward pass."""
        batch_size, seq_len, d_model = 4, 10, 64
        vocab_size = 500
        
        head = CategoricalPredictionHead(d_model=d_model, vocab_size=vocab_size)
        hidden_states = torch.randn(batch_size, seq_len, d_model)
        
        logits = head(hidden_states)
        
        assert logits.shape == (batch_size, seq_len, vocab_size)
        assert not torch.isnan(logits).any()


class TestNumericalPredictionHead:
    """Test numerical prediction head."""
    
    def test_numerical_head_creation(self):
        """Test numerical prediction head creation."""
        head = NumericalPredictionHead(d_model=128)
        assert head.d_model == 128
    
    def test_numerical_head_forward(self):
        """Test numerical prediction head forward pass."""
        batch_size, seq_len, d_model = 3, 8, 32
        
        head = NumericalPredictionHead(d_model=d_model)
        hidden_states = torch.randn(batch_size, seq_len, d_model)
        
        predictions = head(hidden_states)
        
        assert predictions.shape == (batch_size, seq_len, 1)
        assert not torch.isnan(predictions).any()


class TestMaskedCellModelingHead:
    """Test complete masked cell modeling head."""
    
    def test_mcm_head_creation(self):
        """Test MCM head creation."""
        head = MaskedCellModelingHead(
            d_model=128,
            categorical_vocab_size=1000,
            dropout=0.1
        )
        assert head.d_model == 128
        assert head.categorical_vocab_size == 1000
    
    def test_mcm_head_forward_categorical_only(self):
        """Test MCM head with only categorical features."""
        batch_size, seq_len, d_model = 2, 5, 64
        
        head = MaskedCellModelingHead(d_model=d_model, categorical_vocab_size=100)
        
        hidden_states = torch.randn(batch_size, seq_len, d_model)
        masked_positions = torch.tensor([
            [True, False, True, False, False],
            [False, True, False, True, False]
        ])
        original_values = torch.randint(0, 100, (batch_size, seq_len)).float()
        feature_types = ['categorical'] * seq_len
        
        output = head(
            hidden_states=hidden_states,
            masked_positions=masked_positions,
            original_values=original_values,
            feature_types=feature_types
        )
        
        assert isinstance(output, MaskedCellOutput)
        assert output.loss is not None
        assert output.categorical_loss is not None
        assert output.numerical_loss is None
        assert 'categorical' in output.predictions
        assert 'numerical' in output.predictions
        assert output.masked_positions.shape == masked_positions.shape
    
    def test_mcm_head_forward_numerical_only(self):
        """Test MCM head with only numerical features."""
        batch_size, seq_len, d_model = 2, 4, 32
        
        head = MaskedCellModelingHead(d_model=d_model)
        
        hidden_states = torch.randn(batch_size, seq_len, d_model)
        masked_positions = torch.tensor([
            [True, False, True, False],
            [False, True, False, True]
        ])
        original_values = torch.randn(batch_size, seq_len)
        feature_types = ['numerical'] * seq_len
        
        output = head(
            hidden_states=hidden_states,
            masked_positions=masked_positions,
            original_values=original_values,
            feature_types=feature_types
        )
        
        assert isinstance(output, MaskedCellOutput)
        assert output.loss is not None
        assert output.categorical_loss is None
        assert output.numerical_loss is not None
    
    def test_mcm_head_forward_mixed_types(self):
        """Test MCM head with mixed feature types."""
        batch_size, seq_len, d_model = 3, 6, 64
        
        head = MaskedCellModelingHead(d_model=d_model, categorical_vocab_size=50)
        
        hidden_states = torch.randn(batch_size, seq_len, d_model)
        masked_positions = torch.ones(batch_size, seq_len, dtype=torch.bool)
        
        # Mixed original values
        original_values = torch.randn(batch_size, seq_len)
        original_values[:, :3] = torch.randint(0, 50, (batch_size, 3)).float()  # Categorical
        
        feature_types = ['categorical', 'categorical', 'categorical', 'numerical', 'numerical', 'numerical']
        
        output = head(
            hidden_states=hidden_states,
            masked_positions=masked_positions,
            original_values=original_values,
            feature_types=feature_types
        )
        
        assert isinstance(output, MaskedCellOutput)
        assert output.loss is not None
        assert output.categorical_loss is not None
        assert output.numerical_loss is not None
        assert output.accuracy['categorical'] >= 0.0
        assert output.accuracy['numerical_mae'] >= 0.0
    
    def test_mcm_head_with_attention_mask(self):
        """Test MCM head with attention mask."""
        batch_size, seq_len, d_model = 2, 8, 32
        
        head = MaskedCellModelingHead(d_model=d_model)
        
        hidden_states = torch.randn(batch_size, seq_len, d_model)
        masked_positions = torch.ones(batch_size, seq_len, dtype=torch.bool)
        original_values = torch.randn(batch_size, seq_len)
        feature_types = ['numerical'] * seq_len
        
        # Create attention mask (mask out last 2 positions)
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        attention_mask[:, -2:] = False
        
        output = head(
            hidden_states=hidden_states,
            masked_positions=masked_positions,
            original_values=original_values,
            feature_types=feature_types,
            attention_mask=attention_mask
        )
        
        assert isinstance(output, MaskedCellOutput)
        assert output.loss is not None
        # Should only compute loss for non-padded positions
        assert output.accuracy['total_masked'] <= batch_size * (seq_len - 2)


class TestMaskedCellModelingObjective:
    """Test complete masked cell modeling objective."""
    
    def test_mcm_objective_creation(self):
        """Test MCM objective creation."""
        objective = MaskedCellModelingObjective(
            d_model=128,
            mask_probability=0.15,
            categorical_vocab_size=1000
        )
        assert objective.mask_probability == 0.15
        assert objective.categorical_vocab_size == 1000
    
    def test_create_masked_inputs(self):
        """Test masked input creation."""
        batch_size, seq_len, d_model = 3, 10, 64
        
        objective = MaskedCellModelingObjective(d_model=d_model, mask_probability=0.3)
        
        input_embeddings = torch.randn(batch_size, seq_len, d_model)
        input_values = torch.randn(batch_size, seq_len)
        feature_types = ['numerical'] * seq_len
        
        masked_embeddings, masked_positions, original_values = objective.create_masked_inputs(
            input_embeddings, input_values, feature_types
        )
        
        assert masked_embeddings.shape == input_embeddings.shape
        assert masked_positions.shape == (batch_size, seq_len)
        assert original_values.shape == input_values.shape
        assert masked_positions.dtype == torch.bool
        
        # Should mask approximately 30% of positions
        mask_ratio = masked_positions.float().mean().item()
        assert 0.1 < mask_ratio < 0.5  # Allow some variance due to randomness
    
    def test_create_masked_inputs_with_attention_mask(self):
        """Test masked input creation with attention mask."""
        batch_size, seq_len, d_model = 2, 8, 32
        
        objective = MaskedCellModelingObjective(d_model=d_model, mask_probability=0.5)
        
        input_embeddings = torch.randn(batch_size, seq_len, d_model)
        input_values = torch.randn(batch_size, seq_len)
        feature_types = ['numerical'] * seq_len
        
        # Create attention mask (mask out last 3 positions)
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        attention_mask[:, -3:] = False
        
        masked_embeddings, masked_positions, original_values = objective.create_masked_inputs(
            input_embeddings, input_values, feature_types, attention_mask
        )
        
        # Should not mask padded positions
        assert not masked_positions[:, -3:].any()
    
    def test_mcm_objective_forward(self):
        """Test MCM objective forward pass."""
        batch_size, seq_len, d_model = 2, 6, 64
        
        # Create mock model forward function
        def mock_model_forward(input_features, attention_mask=None, **kwargs):
            return {
                'last_hidden_state': torch.randn(batch_size, seq_len, d_model)
            }
        
        objective = MaskedCellModelingObjective(
            d_model=d_model,
            mask_probability=0.2,
            categorical_vocab_size=100
        )
        
        input_embeddings = torch.randn(batch_size, seq_len, d_model)
        input_values = torch.randint(0, 100, (batch_size, seq_len)).float()
        feature_types = ['categorical'] * seq_len
        
        output = objective(
            input_embeddings=input_embeddings,
            input_values=input_values,
            feature_types=feature_types,
            model_forward_fn=mock_model_forward
        )
        
        assert isinstance(output, MaskedCellOutput)
        assert output.loss is not None
        assert not torch.isnan(output.loss)
        assert output.masked_positions.shape == (batch_size, seq_len)
    
    def test_compute_metrics(self):
        """Test metrics computation."""
        objective = MaskedCellModelingObjective(d_model=64)
        
        # Create mock output
        output = MaskedCellOutput(
            loss=torch.tensor(1.5),
            categorical_loss=torch.tensor(0.8),
            numerical_loss=torch.tensor(0.7),
            predictions={'categorical': torch.randn(2, 5, 100), 'numerical': torch.randn(2, 5)},
            masked_positions=torch.ones(2, 5, dtype=torch.bool),
            accuracy={'categorical': 0.6, 'numerical_mae': 0.3, 'total_masked': 10}
        )
        
        metrics = objective.compute_metrics(output)
        
        assert 'mcm_loss' in metrics
        assert 'categorical_accuracy' in metrics
        assert 'numerical_mae' in metrics
        assert 'total_masked_cells' in metrics
        assert 'mask_ratio' in metrics
        assert 'categorical_loss' in metrics
        assert 'numerical_loss' in metrics
        
        assert metrics['mcm_loss'] == 1.5
        assert metrics['categorical_accuracy'] == 0.6
        assert metrics['numerical_mae'] == 0.3


class TestMCMIntegration:
    """Test MCM integration with TabGPT model."""
    
    def test_mcm_with_tabgpt_model(self):
        """Test MCM objective with actual TabGPT model."""
        # Create test data
        df = pd.DataFrame({
            'cat_feature': ['A', 'B', 'C', 'A', 'B'],
            'num_feature': [1.0, 2.5, 3.2, 1.8, 2.1],
            'bool_feature': [True, False, True, False, True]
        })
        
        # Tokenize data
        tokenizer = TabularTokenizer(embedding_dim=64)
        tokenized = tokenizer.fit_transform(df)
        
        # Create model
        config = TabGPTConfig(
            d_model=64,
            n_heads=4,
            n_layers=2,
            embedding_dim=64,
            max_features=10
        )
        model = TabGPTModel(config)
        
        # Create MCM objective
        mcm_objective = MaskedCellModelingObjective(
            d_model=64,
            mask_probability=0.3,
            categorical_vocab_size=tokenizer.vocab_size
        )
        
        # Extract feature types from tokenizer
        feature_types = []
        for metadata in tokenizer.column_metadata:
            feature_types.append(metadata.dtype)
        
        # Forward pass
        output = mcm_objective(
            input_embeddings=tokenized.tokens,
            input_values=tokenized.tokens.sum(dim=-1),  # Simplified values
            feature_types=feature_types,
            model_forward_fn=model.forward,
            attention_mask=tokenized.attention_mask
        )
        
        assert isinstance(output, MaskedCellOutput)
        assert output.loss is not None
        assert not torch.isnan(output.loss)
        
        # Test metrics computation
        metrics = mcm_objective.compute_metrics(output)
        assert isinstance(metrics, dict)
        assert 'mcm_loss' in metrics
    
    def test_mcm_gradient_flow(self):
        """Test gradient flow through MCM objective."""
        batch_size, seq_len, d_model = 2, 4, 32
        
        # Create simple model
        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(d_model, d_model)
            
            def forward(self, input_features, **kwargs):
                return {'last_hidden_state': self.linear(input_features)}
        
        model = SimpleModel()
        
        # Create MCM objective
        mcm_objective = MaskedCellModelingObjective(
            d_model=d_model,
            mask_probability=0.5
        )
        
        input_embeddings = torch.randn(batch_size, seq_len, d_model, requires_grad=True)
        input_values = torch.randn(batch_size, seq_len)
        feature_types = ['numerical'] * seq_len
        
        # Forward pass
        output = mcm_objective(
            input_embeddings=input_embeddings,
            input_values=input_values,
            feature_types=feature_types,
            model_forward_fn=model.forward
        )
        
        # Backward pass
        output.loss.backward()
        
        # Check gradients
        assert input_embeddings.grad is not None
        assert not torch.isnan(input_embeddings.grad).any()
        
        for param in model.parameters():
            assert param.grad is not None
            assert not torch.isnan(param.grad).any()
        
        # Check that at least some MCM parameters have gradients
        # (Not all parameters may have gradients if certain feature types aren't masked)
        mcm_params_with_grad = [p for p in mcm_objective.parameters() if p.grad is not None]
        assert len(mcm_params_with_grad) > 0
        
        for param in mcm_params_with_grad:
            assert not torch.isnan(param.grad).any()
    
    def test_mcm_different_mask_probabilities(self):
        """Test MCM with different mask probabilities."""
        batch_size, seq_len, d_model = 2, 10, 32
        
        def mock_model_forward(input_features, **kwargs):
            return {'last_hidden_state': torch.randn_like(input_features)}
        
        input_embeddings = torch.randn(batch_size, seq_len, d_model)
        input_values = torch.randn(batch_size, seq_len)
        feature_types = ['numerical'] * seq_len
        
        mask_probs = [0.1, 0.15, 0.3, 0.5]
        
        for mask_prob in mask_probs:
            objective = MaskedCellModelingObjective(
                d_model=d_model,
                mask_probability=mask_prob
            )
            
            output = objective(
                input_embeddings=input_embeddings,
                input_values=input_values,
                feature_types=feature_types,
                model_forward_fn=mock_model_forward
            )
            
            # Check that mask ratio is approximately correct
            actual_mask_ratio = output.masked_positions.float().mean().item()
            assert abs(actual_mask_ratio - mask_prob) < 0.1  # Allow some variance
    
    def test_mcm_no_masked_positions(self):
        """Test MCM when no positions are masked."""
        batch_size, seq_len, d_model = 2, 5, 32
        
        objective = MaskedCellModelingObjective(
            d_model=d_model,
            mask_probability=0.0  # No masking
        )
        
        def mock_model_forward(input_features, **kwargs):
            return {'last_hidden_state': torch.randn_like(input_features)}
        
        input_embeddings = torch.randn(batch_size, seq_len, d_model)
        input_values = torch.randn(batch_size, seq_len)
        feature_types = ['numerical'] * seq_len
        
        output = objective(
            input_embeddings=input_embeddings,
            input_values=input_values,
            feature_types=feature_types,
            model_forward_fn=mock_model_forward
        )
        
        # Should have zero loss when nothing is masked
        assert output.loss == 0.0
        assert not output.masked_positions.any()
        assert output.accuracy['total_masked'] == 0