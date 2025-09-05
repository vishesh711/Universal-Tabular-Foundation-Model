"""Tests for HuggingFace integration."""
import pytest
import torch
import pandas as pd
import numpy as np
import tempfile
import os
from unittest.mock import Mock, patch

from tabgpt.models import (
    TabGPTConfig,
    TabGPTModel,
    TabGPTForSequenceClassification,
    TabGPTForRegression,
    TabGPTPreTrainedModel
)
from tabgpt.tokenizers import TabGPTTokenizer


class TestTabGPTConfig:
    """Test TabGPT configuration."""
    
    def test_config_creation(self):
        """Test creating TabGPT configuration."""
        config = TabGPTConfig()
        
        # Check default values
        assert config.hidden_size == 768
        assert config.num_hidden_layers == 12
        assert config.num_attention_heads == 12
        assert config.max_columns == 100
        assert config.max_rows == 512
        assert config.use_masked_cell_modeling is True
        assert config.use_contrastive_row_learning is True
    
    def test_config_custom_values(self):
        """Test creating config with custom values."""
        config = TabGPTConfig(
            hidden_size=512,
            num_hidden_layers=6,
            num_attention_heads=8,
            max_columns=50,
            use_masked_cell_modeling=False
        )
        
        assert config.hidden_size == 512
        assert config.num_hidden_layers == 6
        assert config.num_attention_heads == 8
        assert config.max_columns == 50
        assert config.use_masked_cell_modeling is False
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Invalid hidden_size / num_attention_heads
        with pytest.raises(ValueError):
            TabGPTConfig(hidden_size=100, num_attention_heads=12)
        
        # Invalid max_columns
        with pytest.raises(ValueError):
            TabGPTConfig(max_columns=0)
        
        # Invalid binning strategy
        with pytest.raises(ValueError):
            TabGPTConfig(numerical_binning_strategy="invalid")
    
    def test_config_serialization(self):
        """Test config serialization and deserialization."""
        config = TabGPTConfig(
            hidden_size=480,  # 480 is divisible by 12
            max_columns=50,
            use_masked_cell_modeling=False
        )
        
        # Test to_dict
        config_dict = config.to_dict()
        assert isinstance(config_dict, dict)
        assert config_dict["hidden_size"] == 480
        assert config_dict["max_columns"] == 50
        
        # Test from_dict
        new_config = TabGPTConfig.from_dict(config_dict)
        assert new_config.hidden_size == 480
        assert new_config.max_columns == 50
        assert new_config.use_masked_cell_modeling is False
    
    def test_config_utility_methods(self):
        """Test config utility methods."""
        config = TabGPTConfig(hidden_size=768, num_attention_heads=12)
        
        # Test get_head_dim
        assert config.get_head_dim() == 64  # 768 / 12
        
        # Test get_loss_weights
        loss_weights = config.get_loss_weights()
        assert isinstance(loss_weights, dict)
        assert "mcm" in loss_weights
        assert "crl" in loss_weights
        
        # Test enable/disable objectives
        config.disable_objective("mcm")
        assert config.use_masked_cell_modeling is False
        
        config.enable_objective("mcm")
        assert config.use_masked_cell_modeling is True
        
        # Test set_loss_weight
        config.set_loss_weight("mcm", 2.0)
        assert config.mcm_loss_weight == 2.0


class TestTabGPTTokenizer:
    """Test TabGPT tokenizer."""
    
    def test_tokenizer_creation(self):
        """Test creating TabGPT tokenizer."""
        tokenizer = TabGPTTokenizer()
        
        assert tokenizer.pad_token == "[PAD]"
        assert tokenizer.unk_token == "[UNK]"
        assert tokenizer.mask_token == "[MASK]"
        assert tokenizer.missing_token == "[MISSING]"
        assert tokenizer.vocab_size > 0
    
    def test_tokenizer_special_tokens(self):
        """Test special tokens handling."""
        tokenizer = TabGPTTokenizer()
        
        # Test token to ID conversion
        pad_id = tokenizer._convert_token_to_id(tokenizer.pad_token)
        assert isinstance(pad_id, int)
        
        # Test ID to token conversion
        pad_token = tokenizer._convert_id_to_token(pad_id)
        assert pad_token == tokenizer.pad_token
    
    def test_tokenizer_dataframe_fitting(self):
        """Test fitting tokenizer on DataFrame."""
        # Create sample DataFrame
        df = pd.DataFrame({
            'num_col': [1.0, 2.0, 3.0, 4.0],
            'cat_col': ['A', 'B', 'A', 'C'],
            'target': [0, 1, 0, 1]
        })
        
        tokenizer = TabGPTTokenizer()
        tokenizer.fit_on_dataframe(df, target_column='target')
        
        # Check that tokenizer is fitted
        assert tokenizer.preprocessor is not None
        assert tokenizer.base_tokenizer.is_fitted
        assert len(tokenizer.column_metadata) > 0
    
    def test_tokenizer_dataframe_tokenization(self):
        """Test tokenizing DataFrame."""
        # Create and fit tokenizer
        df = pd.DataFrame({
            'num_col': [1.0, 2.0, 3.0],
            'cat_col': ['A', 'B', 'A']
        })
        
        tokenizer = TabGPTTokenizer()
        tokenizer.fit_on_dataframe(df)
        
        # Tokenize DataFrame
        encoding = tokenizer.tokenize_dataframe(
            df,
            max_length=10,
            padding=True,
            return_tensors="pt"
        )
        
        assert "input_ids" in encoding
        assert "attention_mask" in encoding
        assert isinstance(encoding["input_ids"], torch.Tensor)
        assert encoding["input_ids"].shape[0] == len(df)
    
    def test_tokenizer_save_load(self):
        """Test saving and loading tokenizer."""
        df = pd.DataFrame({
            'num_col': [1.0, 2.0, 3.0],
            'cat_col': ['A', 'B', 'A']
        })
        
        tokenizer = TabGPTTokenizer()
        tokenizer.fit_on_dataframe(df)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save tokenizer
            vocab_file, metadata_file = tokenizer.save_vocabulary(temp_dir)
            
            assert os.path.exists(vocab_file)
            assert os.path.exists(metadata_file)
            
            # Load tokenizer
            new_tokenizer = TabGPTTokenizer(
                vocab_file=vocab_file,
                metadata_file=metadata_file
            )
            
            assert new_tokenizer.vocab_size == tokenizer.vocab_size


class TestTabGPTModel:
    """Test TabGPT model."""
    
    def test_model_creation(self):
        """Test creating TabGPT model."""
        config = TabGPTConfig(
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            max_columns=10
        )
        
        model = TabGPTModel(config)
        
        assert isinstance(model, TabGPTPreTrainedModel)
        assert model.config.hidden_size == 128
        assert model.config.num_hidden_layers == 2
    
    def test_model_forward_basic(self):
        """Test basic model forward pass."""
        config = TabGPTConfig(
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            max_position_embeddings=50
        )
        
        model = TabGPTModel(config)
        
        # Create dummy inputs
        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        
        # Forward pass
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
        
        assert hasattr(outputs, 'last_hidden_state')
        assert outputs.last_hidden_state.shape == (batch_size, seq_len, config.hidden_size)
    
    def test_model_gradient_flow(self):
        """Test gradient flow through model."""
        config = TabGPTConfig(
            hidden_size=64,
            num_hidden_layers=1,
            num_attention_heads=2
        )
        
        model = TabGPTModel(config)
        
        # Create dummy inputs
        batch_size, seq_len = 2, 5
        input_ids = torch.randint(0, 100, (batch_size, seq_len))
        
        # Forward pass with gradients
        outputs = model(input_ids=input_ids, return_dict=True)
        loss = outputs.last_hidden_state.sum()
        
        loss.backward()
        
        # Check that gradients exist
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None


class TestTabGPTForSequenceClassification:
    """Test TabGPT for sequence classification."""
    
    def test_classification_model_creation(self):
        """Test creating classification model."""
        config = TabGPTConfig(
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=4
        )
        config.num_labels = 3  # Multi-class classification
        
        model = TabGPTForSequenceClassification(config)
        
        assert model.num_labels == 3
        assert hasattr(model, 'classifier')
    
    def test_binary_classification_forward(self):
        """Test binary classification forward pass."""
        config = TabGPTConfig(
            hidden_size=64,
            num_hidden_layers=1,
            num_attention_heads=2
        )
        config.num_labels = 2
        
        model = TabGPTForSequenceClassification(config)
        
        # Create dummy inputs
        batch_size, seq_len = 4, 8
        input_ids = torch.randint(0, 100, (batch_size, seq_len))
        labels = torch.randint(0, 2, (batch_size,))
        
        # Forward pass
        outputs = model(
            input_ids=input_ids,
            labels=labels,
            return_dict=True
        )
        
        assert outputs.loss is not None
        assert outputs.logits.shape == (batch_size, 1)
        assert isinstance(outputs.loss.item(), float)
    
    def test_multiclass_classification_forward(self):
        """Test multi-class classification forward pass."""
        config = TabGPTConfig(
            hidden_size=64,
            num_hidden_layers=1,
            num_attention_heads=2
        )
        config.num_labels = 5
        
        model = TabGPTForSequenceClassification(config)
        
        # Create dummy inputs
        batch_size, seq_len = 4, 8
        input_ids = torch.randint(0, 100, (batch_size, seq_len))
        labels = torch.randint(0, 5, (batch_size,))
        
        # Forward pass
        outputs = model(
            input_ids=input_ids,
            labels=labels,
            return_dict=True
        )
        
        assert outputs.loss is not None
        assert outputs.logits.shape == (batch_size, 5)


class TestTabGPTForRegression:
    """Test TabGPT for regression."""
    
    def test_regression_model_creation(self):
        """Test creating regression model."""
        config = TabGPTConfig(
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=4
        )
        config.output_dim = 2  # Multi-target regression
        
        model = TabGPTForRegression(config)
        
        assert model.output_dim == 2
        assert hasattr(model, 'regressor')
    
    def test_regression_forward(self):
        """Test regression forward pass."""
        config = TabGPTConfig(
            hidden_size=64,
            num_hidden_layers=1,
            num_attention_heads=2
        )
        config.output_dim = 1
        
        model = TabGPTForRegression(config)
        
        # Create dummy inputs
        batch_size, seq_len = 4, 8
        input_ids = torch.randint(0, 100, (batch_size, seq_len))
        labels = torch.randn(batch_size, 1)
        
        # Forward pass
        outputs = model(
            input_ids=input_ids,
            labels=labels,
            return_dict=True
        )
        
        assert outputs.loss is not None
        assert outputs.logits.shape == (batch_size, 1)
        assert isinstance(outputs.loss.item(), float)


class TestHuggingFaceCompatibility:
    """Test HuggingFace compatibility features."""
    
    def test_model_save_load(self):
        """Test saving and loading model."""
        config = TabGPTConfig(
            hidden_size=64,
            num_hidden_layers=1,
            num_attention_heads=2
        )
        
        model = TabGPTModel(config)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save model
            model.save_pretrained(temp_dir)
            
            # Check that files exist
            assert os.path.exists(os.path.join(temp_dir, "config.json"))
            assert os.path.exists(os.path.join(temp_dir, "pytorch_model.bin"))
            
            # Load model
            loaded_model = TabGPTModel.from_pretrained(temp_dir)
            
            assert loaded_model.config.hidden_size == config.hidden_size
            assert loaded_model.config.num_hidden_layers == config.num_hidden_layers
    
    def test_config_save_load(self):
        """Test saving and loading configuration."""
        config = TabGPTConfig(
            hidden_size=256,
            max_columns=50,
            use_masked_cell_modeling=False
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = os.path.join(temp_dir, "config.json")
            
            # Save config
            config.save_pretrained(temp_dir)
            assert os.path.exists(config_path)
            
            # Load config
            loaded_config = TabGPTConfig.from_pretrained(temp_dir)
            
            assert loaded_config.hidden_size == 256
            assert loaded_config.max_columns == 50
            assert loaded_config.use_masked_cell_modeling is False
    
    def test_model_device_compatibility(self):
        """Test model device compatibility."""
        config = TabGPTConfig(
            hidden_size=64,
            num_hidden_layers=1,
            num_attention_heads=2
        )
        
        model = TabGPTModel(config)
        
        # Test CPU
        model = model.to("cpu")
        input_ids = torch.randint(0, 100, (2, 5))
        
        with torch.no_grad():
            outputs = model(input_ids=input_ids)
        
        assert outputs[0].device.type == "cpu"
        
        # Test CUDA if available
        if torch.cuda.is_available():
            model = model.to("cuda")
            input_ids = input_ids.to("cuda")
            
            with torch.no_grad():
                outputs = model(input_ids=input_ids)
            
            assert outputs[0].device.type == "cuda"


class TestIntegrationEdgeCases:
    """Test edge cases for HuggingFace integration."""
    
    def test_empty_inputs(self):
        """Test handling empty inputs."""
        config = TabGPTConfig(
            hidden_size=64,
            num_hidden_layers=1,
            num_attention_heads=2
        )
        
        model = TabGPTModel(config)
        
        # Empty input
        input_ids = torch.empty(0, 5, dtype=torch.long)
        
        with torch.no_grad():
            outputs = model(input_ids=input_ids)
        
        assert outputs[0].shape[0] == 0
    
    def test_single_sample(self):
        """Test handling single sample."""
        config = TabGPTConfig(
            hidden_size=64,
            num_hidden_layers=1,
            num_attention_heads=2
        )
        
        model = TabGPTForSequenceClassification(config)
        
        # Single sample
        input_ids = torch.randint(0, 100, (1, 5))
        
        with torch.no_grad():
            outputs = model(input_ids=input_ids, return_dict=True)
        
        assert outputs.logits.shape[0] == 1
    
    def test_large_batch(self):
        """Test handling large batch."""
        config = TabGPTConfig(
            hidden_size=32,  # Small model for memory efficiency
            num_hidden_layers=1,
            num_attention_heads=2
        )
        
        model = TabGPTModel(config)
        
        # Large batch
        batch_size = 100
        input_ids = torch.randint(0, 100, (batch_size, 10))
        
        with torch.no_grad():
            outputs = model(input_ids=input_ids)
        
        assert outputs[0].shape[0] == batch_size


if __name__ == "__main__":
    pytest.main([__file__])