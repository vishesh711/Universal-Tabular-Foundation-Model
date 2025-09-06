"""LoRA (Low-Rank Adaptation) implementation for efficient fine-tuning."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass
import math
import json
import os
from pathlib import Path


@dataclass
class LoRAConfig:
    """Configuration for LoRA adaptation."""
    
    # LoRA hyperparameters
    r: int = 8  # Rank of adaptation
    alpha: float = 16.0  # LoRA scaling parameter
    dropout: float = 0.1  # Dropout probability
    
    # Target modules to apply LoRA
    target_modules: List[str] = None  # e.g., ["query", "value", "key", "dense"]
    
    # Advanced options
    use_rslora: bool = False  # Use rank-stabilized LoRA
    use_dora: bool = False  # Use DoRA (Weight-Decomposed Low-Rank Adaptation)
    
    # Initialization
    init_lora_weights: str = "gaussian"  # gaussian, kaiming, xavier
    
    def __post_init__(self):
        if self.target_modules is None:
            # Default target modules for transformer layers
            self.target_modules = ["query", "key", "value", "dense"]
        
        # Validate parameters
        if self.r <= 0:
            raise ValueError(f"LoRA rank must be positive, got {self.r}")
        if self.alpha <= 0:
            raise ValueError(f"LoRA alpha must be positive, got {self.alpha}")
        if not 0 <= self.dropout <= 1:
            raise ValueError(f"Dropout must be between 0 and 1, got {self.dropout}")
    
    @property
    def scaling(self) -> float:
        """Get LoRA scaling factor."""
        if self.use_rslora:
            # Rank-stabilized LoRA scaling
            return self.alpha / math.sqrt(self.r)
        else:
            # Standard LoRA scaling
            return self.alpha / self.r
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "r": self.r,
            "alpha": self.alpha,
            "dropout": self.dropout,
            "target_modules": self.target_modules,
            "use_rslora": self.use_rslora,
            "use_dora": self.use_dora,
            "init_lora_weights": self.init_lora_weights
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "LoRAConfig":
        """Create config from dictionary."""
        return cls(**config_dict)
    
    def save(self, path: str):
        """Save config to file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> "LoRAConfig":
        """Load config from file."""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


class LoRALayer(nn.Module):
    """Base LoRA layer."""
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        config: LoRAConfig
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.config = config
        
        # LoRA parameters
        if config.r > 0:
            self.lora_A = nn.Parameter(torch.zeros(config.r, in_features))
            self.lora_B = nn.Parameter(torch.zeros(out_features, config.r))
            self.scaling = config.scaling
            
            # Dropout
            self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()
            
            # DoRA parameters (if enabled)
            if config.use_dora:
                self.magnitude = nn.Parameter(torch.ones(out_features))
            
            # Initialize weights
            self._init_weights()
        
        self.merged = False
    
    def _init_weights(self):
        """Initialize LoRA weights."""
        if self.config.init_lora_weights == "gaussian":
            # Standard Gaussian initialization (like original LoRA)
            nn.init.normal_(self.lora_A, std=1/self.config.r)
            nn.init.zeros_(self.lora_B)
        elif self.config.init_lora_weights == "kaiming":
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
        elif self.config.init_lora_weights == "xavier":
            nn.init.xavier_uniform_(self.lora_A)
            nn.init.zeros_(self.lora_B)
        else:
            raise ValueError(f"Unknown initialization: {self.config.init_lora_weights}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through LoRA layer."""
        if self.config.r == 0:
            return torch.zeros_like(x)
        
        # LoRA computation: x @ A^T @ B^T
        result = self.dropout(x) @ self.lora_A.T @ self.lora_B.T
        result = result * self.scaling
        
        return result
    
    def merge_weights(self, base_weight: torch.Tensor) -> torch.Tensor:
        """Merge LoRA weights with base weights."""
        if self.config.r == 0:
            return base_weight
        
        # Compute LoRA weight delta
        lora_weight = self.lora_B @ self.lora_A * self.scaling
        
        if self.config.use_dora:
            # DoRA: decompose weight into magnitude and direction
            base_norm = torch.norm(base_weight, dim=1, keepdim=True)
            direction = base_weight / (base_norm + 1e-8)
            
            # Apply LoRA to direction
            new_direction = direction + lora_weight
            new_direction_norm = torch.norm(new_direction, dim=1, keepdim=True)
            new_direction = new_direction / (new_direction_norm + 1e-8)
            
            # Combine with learned magnitude
            merged_weight = self.magnitude.unsqueeze(1) * new_direction
        else:
            # Standard LoRA
            merged_weight = base_weight + lora_weight
        
        return merged_weight


class LoRALinear(nn.Module):
    """Linear layer with LoRA adaptation."""
    
    def __init__(
        self,
        base_layer: nn.Linear,
        config: LoRAConfig,
        adapter_name: str = "default"
    ):
        super().__init__()
        self.base_layer = base_layer
        self.config = config
        self.adapter_name = adapter_name
        
        # Create LoRA layer
        self.lora = LoRALayer(
            base_layer.in_features,
            base_layer.out_features,
            config
        )
        
        # Track if weights are merged
        self.merged = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through LoRA linear layer."""
        # Base layer computation
        result = self.base_layer(x)
        
        # Add LoRA adaptation if not merged
        if not self.merged and self.config.r > 0:
            result = result + self.lora(x)
        
        return result
    
    def merge_adapter(self):
        """Merge LoRA weights into base layer."""
        if self.merged or self.config.r == 0:
            return
        
        # Merge weights
        merged_weight = self.lora.merge_weights(self.base_layer.weight.data)
        self.base_layer.weight.data = merged_weight
        self.merged = True
    
    def unmerge_adapter(self):
        """Unmerge LoRA weights from base layer."""
        if not self.merged or self.config.r == 0:
            return
        
        # Subtract LoRA weights
        lora_weight = self.lora.lora_B @ self.lora.lora_A * self.lora.scaling
        self.base_layer.weight.data = self.base_layer.weight.data - lora_weight
        self.merged = False


class LoRAEmbedding(nn.Module):
    """Embedding layer with LoRA adaptation."""
    
    def __init__(
        self,
        base_layer: nn.Embedding,
        config: LoRAConfig,
        adapter_name: str = "default"
    ):
        super().__init__()
        self.base_layer = base_layer
        self.config = config
        self.adapter_name = adapter_name
        
        # Create LoRA parameters for embedding
        if config.r > 0:
            self.lora_A = nn.Parameter(torch.zeros(config.r, base_layer.num_embeddings))
            self.lora_B = nn.Parameter(torch.zeros(base_layer.embedding_dim, config.r))
            self.scaling = config.scaling
            
            # Initialize weights
            nn.init.normal_(self.lora_A, std=1/config.r)
            nn.init.zeros_(self.lora_B)
        
        self.merged = False
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass through LoRA embedding layer."""
        # Base embedding
        result = self.base_layer(input_ids)
        
        # Add LoRA adaptation if not merged
        if not self.merged and self.config.r > 0:
            # LoRA embedding computation
            lora_result = F.embedding(
                input_ids,
                (self.lora_B @ self.lora_A).T * self.scaling,
                self.base_layer.padding_idx,
                self.base_layer.max_norm,
                self.base_layer.norm_type,
                self.base_layer.scale_grad_by_freq,
                self.base_layer.sparse
            )
            result = result + lora_result
        
        return result
    
    def merge_adapter(self):
        """Merge LoRA weights into base embedding."""
        if self.merged or self.config.r == 0:
            return
        
        # Merge weights
        lora_weight = (self.lora_B @ self.lora_A).T * self.scaling
        self.base_layer.weight.data = self.base_layer.weight.data + lora_weight
        self.merged = True
    
    def unmerge_adapter(self):
        """Unmerge LoRA weights from base embedding."""
        if not self.merged or self.config.r == 0:
            return
        
        # Subtract LoRA weights
        lora_weight = (self.lora_B @ self.lora_A).T * self.scaling
        self.base_layer.weight.data = self.base_layer.weight.data - lora_weight
        self.merged = False


def apply_lora_to_model(
    model: nn.Module,
    config: LoRAConfig,
    target_modules: Optional[List[str]] = None
) -> nn.Module:
    """
    Apply LoRA adaptation to specified modules in a model.
    
    Args:
        model: Base model to adapt
        config: LoRA configuration
        target_modules: List of module names to target (overrides config)
        
    Returns:
        Model with LoRA adapters applied
    """
    if target_modules is None:
        target_modules = config.target_modules
    
    # Find and replace target modules
    for name, module in model.named_modules():
        # Check if this module should be adapted
        should_adapt = any(target in name for target in target_modules)
        
        if should_adapt:
            if isinstance(module, nn.Linear):
                # Replace with LoRA linear layer
                parent_name = ".".join(name.split(".")[:-1])
                child_name = name.split(".")[-1]
                
                if parent_name:
                    parent_module = model.get_submodule(parent_name)
                else:
                    parent_module = model
                
                lora_layer = LoRALinear(module, config)
                setattr(parent_module, child_name, lora_layer)
                
            elif isinstance(module, nn.Embedding):
                # Replace with LoRA embedding layer
                parent_name = ".".join(name.split(".")[:-1])
                child_name = name.split(".")[-1]
                
                if parent_name:
                    parent_module = model.get_submodule(parent_name)
                else:
                    parent_module = model
                
                lora_layer = LoRAEmbedding(module, config)
                setattr(parent_module, child_name, lora_layer)
    
    return model


def get_lora_parameters(model: nn.Module) -> Dict[str, torch.Tensor]:
    """
    Get all LoRA parameters from a model.
    
    Args:
        model: Model with LoRA adapters
        
    Returns:
        Dictionary of LoRA parameter names and tensors
    """
    lora_params = {}
    
    for name, param in model.named_parameters():
        if "lora_" in name or "magnitude" in name:
            lora_params[name] = param
    
    return lora_params


def merge_lora_weights(model: nn.Module):
    """
    Merge all LoRA weights into base model weights.
    
    Args:
        model: Model with LoRA adapters
    """
    for module in model.modules():
        if isinstance(module, (LoRALinear, LoRAEmbedding)):
            module.merge_adapter()


def save_lora_weights(
    model: nn.Module,
    save_directory: str,
    config: LoRAConfig
):
    """
    Save LoRA weights and configuration.
    
    Args:
        model: Model with LoRA adapters
        save_directory: Directory to save weights
        config: LoRA configuration
    """
    save_path = Path(save_directory)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Save LoRA parameters
    lora_params = get_lora_parameters(model)
    torch.save(lora_params, save_path / "lora_weights.pt")
    
    # Save configuration
    config.save(save_path / "lora_config.json")
    
    print(f"LoRA weights saved to {save_directory}")


def load_lora_weights(
    model: nn.Module,
    load_directory: str,
    strict: bool = True
) -> LoRAConfig:
    """
    Load LoRA weights and configuration.
    
    Args:
        model: Model to load LoRA weights into
        load_directory: Directory containing LoRA weights
        strict: Whether to strictly match parameter names
        
    Returns:
        LoRA configuration
    """
    load_path = Path(load_directory)
    
    # Load configuration
    config = LoRAConfig.load(load_path / "lora_config.json")
    
    # Load LoRA parameters
    lora_params = torch.load(load_path / "lora_weights.pt", map_location="cpu")
    
    # Load parameters into model
    missing_keys, unexpected_keys = model.load_state_dict(lora_params, strict=False)
    
    if strict and (missing_keys or unexpected_keys):
        raise RuntimeError(
            f"Error loading LoRA weights. "
            f"Missing keys: {missing_keys}, "
            f"Unexpected keys: {unexpected_keys}"
        )
    
    print(f"LoRA weights loaded from {load_directory}")
    return config


def get_lora_model_info(model: nn.Module) -> Dict[str, Any]:
    """
    Get information about LoRA adapters in a model.
    
    Args:
        model: Model with LoRA adapters
        
    Returns:
        Dictionary with LoRA information
    """
    info = {
        "total_parameters": 0,
        "lora_parameters": 0,
        "lora_modules": [],
        "parameter_efficiency": 0.0
    }
    
    # Count parameters
    for name, param in model.named_parameters():
        info["total_parameters"] += param.numel()
        
        if "lora_" in name or "magnitude" in name:
            info["lora_parameters"] += param.numel()
    
    # Find LoRA modules
    for name, module in model.named_modules():
        if isinstance(module, (LoRALinear, LoRAEmbedding)):
            info["lora_modules"].append(name)
    
    # Calculate efficiency
    if info["total_parameters"] > 0:
        info["parameter_efficiency"] = info["lora_parameters"] / info["total_parameters"]
    
    return info