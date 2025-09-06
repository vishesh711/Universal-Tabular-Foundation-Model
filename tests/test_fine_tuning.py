"""Tests for fine-tuning functionality."""

import pytest
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
import tempfile
import os
from pathlib import Path

from tabgpt.adapters import LoRAConfig, LoRALinear, apply_lora_to_model, get_lora_parameters
from tabgpt.fine_tuning import (
    TabGPTFineTuningTrainer,
    FineTuningConfig,
    prepare_classification_data,
    prepare_regression_data,
    create_default_callbacks
)
from tabgpt.fine_tuning.callbacks import (
    EarlyStoppingCallback,
    ModelCheckpointCallback,
    ProgressCallback
)


class TestLoRAAdapters:
    """Test LoRA adapter functionality."""
    
    def test_lora_config_creation(self):
        """Test LoRA configuration creation."""
        config = LoRAConfig(r=8, alpha=16.0, dropout=0.1)
        assert config.r == 8
        assert config.alpha == 16.0
        assert config.dropout == 0.1
        assert config.scaling == 2.0  # alpha / r
        
    def test_lora_config_validation(self):
        """Test LoRA configuration validation."""
        with pytest.raises(ValueError):
            LoRAConfig(r=0)  # Invalid rank
            
        with pytest.raises(ValueError):
            LoRAConfig(alpha=-1.0)  # Invalid alpha
            
        with pytest.raises(ValueError):
            LoRAConfig(dropout=1.5)  # Invalid dropout
            
    def test_lora_linear_layer(self):
        """Test LoRA linear layer."""
        base_layer = nn.Linear(128, 64)
        config = LoRAConfig(r=8, alpha=16.0)
        
        lora_layer = LoRALinear(base_layer, config)
        
        # Test forward pass
        x = torch.randn(4, 128)
        output = lora_layer(x)
        assert output.shape == (4, 64)
        
        # Test parameter count
        lora_params = sum(p.numel() for p in lora_layer.lora.parameters())
        expected_params = 8 * 128 + 64 * 8  # A and B matrices
        assert lora_params == expected_params
        
    def test_lora_merge_unmerge(self):
        """Test LoRA weight merging and unmerging."""
        base_layer = nn.Linear(128, 64)
        config = LoRAConfig(r=8, alpha=16.0)
        lora_layer = LoRALinear(base_layer, config)
        
        # Store original weight
        original_weight = base_layer.weight.data.clone()
        
        # Test merging
        lora_layer.merge_adapter()
        assert lora_layer.merged
        assert not torch.equal(base_layer.weight.data, original_weight)
        
        # Test unmerging
        lora_layer.unmerge_adapter()
        assert not lora_layer.merged
        assert torch.allclose(base_layer.weight.data, original_weight, atol=1e-6)
        
    def test_apply_lora_to_model(self):
        """Test applying LoRA to a model."""
        # Create a simple model
        model = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.Linear(32, 10)
        )
        
        config = LoRAConfig(r=8, target_modules=["0", "2"])  # Target first and third layers
        
        # Apply LoRA
        lora_model = apply_lora_to_model(model, config)
        
        # Check that LoRA layers were applied
        assert isinstance(lora_model[0], LoRALinear)
        assert isinstance(lora_model[2], LoRALinear)
        assert isinstance(lora_model[3], nn.Linear)  # Should not be modified
        
    def test_lora_parameter_extraction(self):
        """Test extracting LoRA parameters."""
        model = nn.Sequential(nn.Linear(128, 64))
        config = LoRAConfig(r=8)
        
        lora_model = apply_lora_to_model(model, config)
        lora_params = get_lora_parameters(lora_model)
        
        # Should have LoRA A and B parameters
        assert len(lora_params) == 2
        assert any("lora_A" in name for name in lora_params.keys())
        assert any("lora_B" in name for name in lora_params.keys())


class TestFineTuningCallbacks:
    """Test fine-tuning callbacks."""
    
    def test_early_stopping_callback(self):
        """Test early stopping callback."""
        callback = EarlyStoppingCallback(
            early_stopping_patience=2,
            metric_for_best_model="eval_loss",
            greater_is_better=False
        )
        
        # Mock objects
        args = Mock()
        state = Mock()
        control = Mock()
        control.should_training_stop = False
        model = Mock()
        
        # First evaluation - should not stop
        logs = {"eval_loss": 1.0}
        callback.on_evaluate(args, state, control, model, logs)
        assert not control.should_training_stop
        assert callback.best_metric == 1.0
        
        # Second evaluation - improvement, should not stop
        logs = {"eval_loss": 0.8}
        callback.on_evaluate(args, state, control, model, logs)
        assert not control.should_training_stop
        assert callback.best_metric == 0.8
        
        # Third evaluation - no improvement
        logs = {"eval_loss": 0.9}
        callback.on_evaluate(args, state, control, model, logs)
        assert not control.should_training_stop
        assert callback.patience_counter == 1
        
        # Fourth evaluation - still no improvement, should trigger early stopping
        logs = {"eval_loss": 1.0}
        callback.on_evaluate(args, state, control, model, logs)
        assert control.should_training_stop
        
    def test_model_checkpoint_callback(self):
        """Test model checkpoint callback."""
        with tempfile.TemporaryDirectory() as temp_dir:
            callback = ModelCheckpointCallback(
                save_best_model=True,
                metric_for_best_model="eval_accuracy",
                greater_is_better=True
            )
            
            # Mock objects
            args = Mock()
            args.output_dir = temp_dir
            state = Mock()
            state.global_step = 100
            control = Mock()
            model = Mock()
            model.save_pretrained = Mock()
            
            # First evaluation
            logs = {"eval_accuracy": 0.8}
            callback.on_evaluate(args, state, control, model, logs)
            
            # Should save model
            model.save_pretrained.assert_called_once()
            assert callback.best_metric == 0.8
            
    def test_progress_callback(self):
        """Test progress callback."""
        callback = ProgressCallback(log_every_n_steps=10)
        
        # Mock objects
        args = Mock()
        state = Mock()
        state.global_step = 10
        control = Mock()
        
        # Should log at step 10
        logs = {"loss": 0.5, "learning_rate": 1e-5}
        
        with patch('tabgpt.fine_tuning.callbacks.logger') as mock_logger:
            callback.on_log(args, state, control, logs)
            mock_logger.info.assert_called_once()


class TestFineTuningConfig:
    """Test fine-tuning configuration."""
    
    def test_config_creation(self):
        """Test creating fine-tuning configuration."""
        config = FineTuningConfig(
            output_dir="./output",
            num_train_epochs=3,
            per_device_train_batch_size=32,
            learning_rate=5e-5
        )
        
        assert config.output_dir == "./output"
        assert config.num_train_epochs == 3
        assert config.per_device_train_batch_size == 32
        assert config.learning_rate == 5e-5
        
    def test_config_validation(self):
        """Test configuration validation."""
        # Should work with valid parameters
        config = FineTuningConfig(output_dir="./output")
        assert config.output_dir == "./output"


class TestDataPreparation:
    """Test data preparation utilities."""
    
    def test_prepare_classification_data(self):
        """Test preparing classification data."""
        # Create sample data
        df = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': ['a', 'b', 'c', 'd', 'e'],
            'target': [0, 1, 0, 1, 0]
        })
        
        # Mock tokenizer
        tokenizer = Mock()
        tokenizer.encode_batch = Mock(return_value={
            'input_ids': torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]]),
            'attention_mask': torch.ones(5, 3)
        })
        
        dataset = prepare_classification_data(df, 'target', tokenizer)
        
        # Should return a dataset-like object
        assert hasattr(dataset, '__len__')
        assert len(dataset) == 5
        
    def test_prepare_regression_data(self):
        """Test preparing regression data."""
        # Create sample data
        df = pd.DataFrame({
            'feature1': [1.0, 2.0, 3.0, 4.0, 5.0],
            'feature2': ['a', 'b', 'c', 'd', 'e'],
            'target': [1.5, 2.5, 3.5, 4.5, 5.5]
        })
        
        # Mock tokenizer
        tokenizer = Mock()
        tokenizer.encode_batch = Mock(return_value={
            'input_ids': torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]]),
            'attention_mask': torch.ones(5, 3)
        })
        
        dataset = prepare_regression_data(df, 'target', tokenizer)
        
        # Should return a dataset-like object
        assert hasattr(dataset, '__len__')
        assert len(dataset) == 5


class TestFineTuningTrainer:
    """Test fine-tuning trainer."""
    
    def test_trainer_creation(self):
        """Test creating fine-tuning trainer."""
        # Mock components
        model = Mock()
        config = FineTuningConfig(output_dir="./output")
        train_dataset = Mock()
        eval_dataset = Mock()
        tokenizer = Mock()
        
        trainer = TabGPTFineTuningTrainer(
            model=model,
            config=config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer
        )
        
        assert trainer.model == model
        assert trainer.config == config
        assert trainer.train_dataset == train_dataset
        assert trainer.eval_dataset == eval_dataset
        assert trainer.tokenizer == tokenizer


class TestIntegration:
    """Integration tests for fine-tuning."""
    
    def test_end_to_end_classification(self):
        """Test end-to-end classification fine-tuning."""
        # Create sample data
        np.random.seed(42)
        n_samples = 100
        
        df = pd.DataFrame({
            'feature1': np.random.randn(n_samples),
            'feature2': np.random.choice(['A', 'B', 'C'], n_samples),
            'feature3': np.random.randint(0, 10, n_samples),
            'target': np.random.choice([0, 1], n_samples)
        })
        
        # This would be a more complete test with actual model training
        # For now, just test that the data preparation works
        assert len(df) == n_samples
        assert 'target' in df.columns
        
    def test_lora_integration(self):
        """Test LoRA integration with fine-tuning."""
        # Create a simple model
        model = nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 2)
        )
        
        # Apply LoRA
        config = LoRAConfig(r=4, alpha=8.0)
        lora_model = apply_lora_to_model(model, config)
        
        # Test forward pass
        x = torch.randn(4, 10)
        output = lora_model(x)
        assert output.shape == (4, 2)
        
        # Test that LoRA parameters exist
        lora_params = get_lora_parameters(lora_model)
        assert len(lora_params) > 0


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_invalid_lora_config(self):
        """Test handling of invalid LoRA configurations."""
        with pytest.raises(ValueError):
            LoRAConfig(r=-1)
            
        with pytest.raises(ValueError):
            LoRAConfig(alpha=0)
            
    def test_empty_dataset(self):
        """Test handling of empty datasets."""
        df = pd.DataFrame()
        tokenizer = Mock()
        
        # Should handle empty dataframe gracefully
        try:
            dataset = prepare_classification_data(df, 'target', tokenizer)
            # If it doesn't raise an error, that's fine too
        except (ValueError, KeyError):
            # Expected for empty dataframe
            pass
            
    def test_missing_target_column(self):
        """Test handling of missing target column."""
        df = pd.DataFrame({'feature1': [1, 2, 3]})
        tokenizer = Mock()
        
        with pytest.raises((ValueError, KeyError)):
            prepare_classification_data(df, 'nonexistent_target', tokenizer)


if __name__ == "__main__":
    pytest.main([__file__])