"""Adapter modules for efficient fine-tuning."""

from .lora import (
    LoRAConfig,
    LoRALayer,
    LoRALinear,
    LoRAEmbedding,
    apply_lora_to_model,
    get_lora_parameters,
    merge_lora_weights,
    save_lora_weights,
    load_lora_weights,
    get_lora_model_info
)
from .adapter_utils import (
    AdapterConfig,
    freeze_base_model,
    unfreeze_base_model,
    get_trainable_parameters,
    compute_parameter_efficiency
)

__all__ = [
    "LoRAConfig",
    "LoRALayer", 
    "LoRALinear",
    "LoRAEmbedding",
    "apply_lora_to_model",
    "get_lora_parameters",
    "merge_lora_weights",
    "save_lora_weights",
    "load_lora_weights",
    "get_lora_model_info",
    "AdapterConfig",
    "freeze_base_model",
    "unfreeze_base_model", 
    "get_trainable_parameters",
    "compute_parameter_efficiency"
]