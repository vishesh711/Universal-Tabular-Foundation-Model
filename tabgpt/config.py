"""Configuration classes for TabGPT models."""

from dataclasses import dataclass
from typing import Dict, Any, Optional
import json


@dataclass
class TabGPTConfig:
    """Configuration class for TabGPT models."""
    
    # Model architecture
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 6
    dropout: float = 0.1
    
    # Tokenizer settings
    vocab_size: int = 10000
    max_features: int = 512
    embedding_dim: int = 256
    
    # Column encoder settings
    column_embedding_dim: int = 128
    statistical_features: int = 8
    
    # Pre-training objectives
    mask_probability: float = 0.15
    column_mask_probability: float = 0.2
    contrastive_temperature: float = 0.1
    
    # Row encoder settings
    use_positional_encoding: bool = False
    pooling_strategy: str = 'cls'  # 'cls', 'mean', 'max'
    
    # Cross-attention fusion settings
    cross_attention_layers: int = 2
    fusion_strategy: str = 'gate'  # 'add', 'concat', 'gate'
    cross_attention_temperature: float = 1.0
    
    # Training settings
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {k: v for k, v in self.__dict__.items()}
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TabGPTConfig':
        """Create config from dictionary."""
        return cls(**config_dict)
    
    def save_pretrained(self, save_directory: str) -> None:
        """Save config to directory."""
        import os
        os.makedirs(save_directory, exist_ok=True)
        config_path = os.path.join(save_directory, "config.json")
        with open(config_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def from_pretrained(cls, model_name_or_path: str) -> 'TabGPTConfig':
        """Load config from pretrained model."""
        import os
        if os.path.isdir(model_name_or_path):
            config_path = os.path.join(model_name_or_path, "config.json")
        else:
            # Handle HuggingFace Hub loading later
            raise NotImplementedError("HuggingFace Hub loading not implemented yet")
        
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)