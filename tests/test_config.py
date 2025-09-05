"""Tests for TabGPT configuration."""

import pytest
import tempfile
import os
from tabgpt.config import TabGPTConfig


def test_config_creation():
    """Test basic config creation."""
    config = TabGPTConfig()
    assert config.d_model == 256
    assert config.n_heads == 8
    assert config.n_layers == 6


def test_config_to_dict():
    """Test config serialization to dict."""
    config = TabGPTConfig(d_model=512, n_heads=16)
    config_dict = config.to_dict()
    
    assert config_dict['d_model'] == 512
    assert config_dict['n_heads'] == 16
    assert isinstance(config_dict, dict)


def test_config_from_dict():
    """Test config creation from dict."""
    config_dict = {
        'd_model': 512,
        'n_heads': 16,
        'n_layers': 8
    }
    config = TabGPTConfig.from_dict(config_dict)
    
    assert config.d_model == 512
    assert config.n_heads == 16
    assert config.n_layers == 8


def test_config_save_load():
    """Test config save and load functionality."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create and save config
        config = TabGPTConfig(d_model=512, n_heads=16)
        config.save_pretrained(temp_dir)
        
        # Check file exists
        config_path = os.path.join(temp_dir, "config.json")
        assert os.path.exists(config_path)
        
        # Load config
        loaded_config = TabGPTConfig.from_pretrained(temp_dir)
        
        # Verify values
        assert loaded_config.d_model == 512
        assert loaded_config.n_heads == 16