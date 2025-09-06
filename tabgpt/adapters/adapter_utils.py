"""Utility functions for adapter-based fine-tuning."""
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Union, Any, Tuple, Iterator
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class AdapterConfig:
    """General configuration for adapters."""
    
    # Adapter type
    adapter_type: str = "lora"  # lora, prefix, prompt, etc.
    
    # Training parameters
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    
    # Freezing strategy
    freeze_base_model: bool = True
    trainable_modules: Optional[List[str]] = None
    
    # Regularization
    adapter_dropout: float = 0.1
    
    def __post_init__(self):
        """Validate configuration."""
        if self.learning_rate <= 0:
            raise ValueError(f"Learning rate must be positive, got {self.learning_rate}")
        
        if not 0 <= self.adapter_dropout <= 1:
            raise ValueError(f"Dropout must be between 0 and 1, got {self.adapter_dropout}")


def freeze_base_model(
    model: nn.Module,
    exclude_patterns: Optional[List[str]] = None
) -> int:
    """
    Freeze all parameters in the base model except those matching exclude patterns.
    
    Args:
        model: Model to freeze
        exclude_patterns: List of parameter name patterns to exclude from freezing
        
    Returns:
        Number of parameters frozen
    """
    if exclude_patterns is None:
        exclude_patterns = ["lora_", "adapter_", "classifier", "head"]
    
    frozen_count = 0
    
    for name, param in model.named_parameters():
        # Check if parameter should be excluded from freezing
        should_exclude = any(pattern in name for pattern in exclude_patterns)
        
        if not should_exclude:
            param.requires_grad = False
            frozen_count += param.numel()
        else:
            param.requires_grad = True
    
    logger.info(f"Frozen {frozen_count:,} parameters")
    return frozen_count


def unfreeze_base_model(model: nn.Module) -> int:
    """
    Unfreeze all parameters in the model.
    
    Args:
        model: Model to unfreeze
        
    Returns:
        Number of parameters unfrozen
    """
    unfrozen_count = 0
    
    for param in model.parameters():
        if not param.requires_grad:
            param.requires_grad = True
            unfrozen_count += param.numel()
    
    logger.info(f"Unfrozen {unfrozen_count:,} parameters")
    return unfrozen_count


def get_trainable_parameters(model: nn.Module) -> Tuple[int, int, float]:
    """
    Get information about trainable parameters in the model.
    
    Args:
        model: Model to analyze
        
    Returns:
        Tuple of (trainable_params, total_params, trainable_ratio)
    """
    trainable_params = 0
    total_params = 0
    
    for param in model.parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    
    trainable_ratio = trainable_params / total_params if total_params > 0 else 0.0
    
    return trainable_params, total_params, trainable_ratio


def compute_parameter_efficiency(
    model: nn.Module,
    adapter_patterns: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Compute parameter efficiency metrics for adapter-based fine-tuning.
    
    Args:
        model: Model with adapters
        adapter_patterns: List of patterns to identify adapter parameters
        
    Returns:
        Dictionary with efficiency metrics
    """
    if adapter_patterns is None:
        adapter_patterns = ["lora_", "adapter_", "prefix_", "prompt_"]
    
    total_params = 0
    adapter_params = 0
    trainable_params = 0
    
    adapter_modules = {}
    
    for name, param in model.named_parameters():
        total_params += param.numel()
        
        if param.requires_grad:
            trainable_params += param.numel()
        
        # Check if this is an adapter parameter
        is_adapter = any(pattern in name for pattern in adapter_patterns)
        if is_adapter:
            adapter_params += param.numel()
            
            # Group by module type
            module_type = None
            for pattern in adapter_patterns:
                if pattern in name:
                    module_type = pattern.rstrip("_")
                    break
            
            if module_type not in adapter_modules:
                adapter_modules[module_type] = 0
            adapter_modules[module_type] += param.numel()
    
    # Calculate efficiency metrics
    adapter_efficiency = adapter_params / total_params if total_params > 0 else 0.0
    trainable_efficiency = trainable_params / total_params if total_params > 0 else 0.0
    
    return {
        "total_parameters": total_params,
        "adapter_parameters": adapter_params,
        "trainable_parameters": trainable_params,
        "adapter_efficiency": adapter_efficiency,
        "trainable_efficiency": trainable_efficiency,
        "adapter_modules": adapter_modules,
        "total_parameters_millions": total_params / 1e6,
        "adapter_parameters_millions": adapter_params / 1e6,
        "trainable_parameters_millions": trainable_params / 1e6,
    }


def get_adapter_parameters(
    model: nn.Module,
    adapter_patterns: Optional[List[str]] = None
) -> Iterator[nn.Parameter]:
    """
    Get iterator over adapter parameters.
    
    Args:
        model: Model with adapters
        adapter_patterns: List of patterns to identify adapter parameters
        
    Yields:
        Adapter parameters
    """
    if adapter_patterns is None:
        adapter_patterns = ["lora_", "adapter_", "prefix_", "prompt_"]
    
    for name, param in model.named_parameters():
        is_adapter = any(pattern in name for pattern in adapter_patterns)
        if is_adapter:
            yield param


def print_trainable_parameters(model: nn.Module):
    """
    Print information about trainable parameters in the model.
    
    Args:
        model: Model to analyze
    """
    trainable_params, total_params, trainable_ratio = get_trainable_parameters(model)
    
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable ratio: {trainable_ratio:.4f} ({trainable_ratio*100:.2f}%)")
    
    # Print efficiency metrics if adapters are detected
    efficiency_info = compute_parameter_efficiency(model)
    if efficiency_info["adapter_parameters"] > 0:
        print(f"Adapter parameters: {efficiency_info['adapter_parameters']:,}")
        print(f"Adapter efficiency: {efficiency_info['adapter_efficiency']:.4f} ({efficiency_info['adapter_efficiency']*100:.2f}%)")
        
        if efficiency_info["adapter_modules"]:
            print("Adapter modules:")
            for module_type, param_count in efficiency_info["adapter_modules"].items():
                print(f"  {module_type}: {param_count:,} parameters")


def create_optimizer_for_adapters(
    model: nn.Module,
    learning_rate: float = 1e-4,
    weight_decay: float = 0.01,
    adapter_patterns: Optional[List[str]] = None
) -> torch.optim.Optimizer:
    """
    Create optimizer that only updates adapter parameters.
    
    Args:
        model: Model with adapters
        learning_rate: Learning rate for adapter parameters
        weight_decay: Weight decay for adapter parameters
        adapter_patterns: List of patterns to identify adapter parameters
        
    Returns:
        Optimizer for adapter parameters
    """
    adapter_params = list(get_adapter_parameters(model, adapter_patterns))
    
    if not adapter_params:
        raise ValueError("No adapter parameters found in model")
    
    optimizer = torch.optim.AdamW(
        adapter_params,
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    logger.info(f"Created optimizer for {len(adapter_params)} adapter parameter groups")
    return optimizer


def setup_adapter_training(
    model: nn.Module,
    adapter_config: AdapterConfig
) -> Tuple[nn.Module, torch.optim.Optimizer]:
    """
    Setup model and optimizer for adapter-based training.
    
    Args:
        model: Base model
        adapter_config: Adapter configuration
        
    Returns:
        Tuple of (prepared_model, optimizer)
    """
    # Freeze base model if specified
    if adapter_config.freeze_base_model:
        freeze_base_model(model, adapter_config.trainable_modules)
    
    # Create optimizer for trainable parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    
    if not trainable_params:
        raise ValueError("No trainable parameters found in model")
    
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=adapter_config.learning_rate,
        weight_decay=adapter_config.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Print parameter information
    print_trainable_parameters(model)
    
    return model, optimizer


def validate_adapter_setup(model: nn.Module) -> Dict[str, Any]:
    """
    Validate that adapter setup is correct.
    
    Args:
        model: Model with adapters
        
    Returns:
        Dictionary with validation results
    """
    results = {
        "valid": True,
        "warnings": [],
        "errors": [],
        "info": {}
    }
    
    # Get parameter information
    trainable_params, total_params, trainable_ratio = get_trainable_parameters(model)
    efficiency_info = compute_parameter_efficiency(model)
    
    results["info"] = {
        "trainable_parameters": trainable_params,
        "total_parameters": total_params,
        "trainable_ratio": trainable_ratio,
        **efficiency_info
    }
    
    # Check if any parameters are trainable
    if trainable_params == 0:
        results["errors"].append("No trainable parameters found")
        results["valid"] = False
    
    # Check if adapter efficiency is reasonable
    if efficiency_info["adapter_parameters"] > 0:
        if efficiency_info["adapter_efficiency"] > 0.5:
            results["warnings"].append(
                f"High adapter efficiency ({efficiency_info['adapter_efficiency']:.2%}). "
                "Consider reducing adapter size for better efficiency."
            )
        elif efficiency_info["adapter_efficiency"] < 0.001:
            results["warnings"].append(
                f"Very low adapter efficiency ({efficiency_info['adapter_efficiency']:.4%}). "
                "Consider increasing adapter size if performance is poor."
            )
    
    # Check for gradient flow
    has_gradients = False
    for param in model.parameters():
        if param.requires_grad:
            has_gradients = True
            break
    
    if not has_gradients:
        results["errors"].append("No parameters require gradients")
        results["valid"] = False
    
    return results


def compare_models(
    base_model: nn.Module,
    adapted_model: nn.Module
) -> Dict[str, Any]:
    """
    Compare base model with adapted model.
    
    Args:
        base_model: Original base model
        adapted_model: Model with adapters
        
    Returns:
        Dictionary with comparison results
    """
    base_params = sum(p.numel() for p in base_model.parameters())
    adapted_params = sum(p.numel() for p in adapted_model.parameters())
    
    base_trainable = sum(p.numel() for p in base_model.parameters() if p.requires_grad)
    adapted_trainable = sum(p.numel() for p in adapted_model.parameters() if p.requires_grad)
    
    adapter_efficiency = compute_parameter_efficiency(adapted_model)
    
    return {
        "base_model": {
            "total_parameters": base_params,
            "trainable_parameters": base_trainable,
            "trainable_ratio": base_trainable / base_params if base_params > 0 else 0
        },
        "adapted_model": {
            "total_parameters": adapted_params,
            "trainable_parameters": adapted_trainable,
            "trainable_ratio": adapted_trainable / adapted_params if adapted_params > 0 else 0
        },
        "comparison": {
            "parameter_increase": adapted_params - base_params,
            "parameter_increase_ratio": (adapted_params - base_params) / base_params if base_params > 0 else 0,
            "trainable_reduction": base_trainable - adapted_trainable,
            "trainable_reduction_ratio": (base_trainable - adapted_trainable) / base_trainable if base_trainable > 0 else 0,
            "adapter_efficiency": adapter_efficiency["adapter_efficiency"]
        }
    }