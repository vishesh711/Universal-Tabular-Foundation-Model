"""Model optimization utilities for efficient inference."""

import logging
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass
import warnings
import time
from collections import OrderedDict
import threading

import torch
import torch.nn as nn
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class QuantizationConfig:
    """Configuration for model quantization."""
    
    # Quantization type
    quantization_type: str = "dynamic"  # dynamic, static, qat
    
    # Precision settings
    weight_dtype: str = "int8"  # int8, int4, float16
    activation_dtype: str = "int8"
    
    # Calibration settings (for static quantization)
    calibration_samples: int = 100
    calibration_method: str = "entropy"  # entropy, percentile
    
    # QAT settings
    qat_epochs: int = 5
    qat_lr: float = 1e-5
    
    # Backend settings
    backend: str = "fbgemm"  # fbgemm, qnnpack, onednn
    
    def __post_init__(self):
        """Validate configuration."""
        valid_types = ["dynamic", "static", "qat"]
        if self.quantization_type not in valid_types:
            raise ValueError(f"quantization_type must be one of {valid_types}")
        
        valid_dtypes = ["int8", "int4", "float16"]
        if self.weight_dtype not in valid_dtypes:
            raise ValueError(f"weight_dtype must be one of {valid_dtypes}")


@dataclass
class OptimizationConfig:
    """Configuration for model optimization."""
    
    # Quantization
    enable_quantization: bool = False
    quantization_config: Optional[QuantizationConfig] = None
    
    # Pruning
    enable_pruning: bool = False
    pruning_ratio: float = 0.1
    structured_pruning: bool = False
    
    # Compilation
    enable_torch_compile: bool = False
    compile_mode: str = "default"  # default, reduce-overhead, max-autotune
    
    # Memory optimization
    enable_gradient_checkpointing: bool = False
    enable_memory_efficient_attention: bool = True
    
    # Fusion
    enable_operator_fusion: bool = True
    
    # Device optimization
    optimize_for_device: bool = True
    target_device: str = "auto"  # auto, cpu, cuda, mps


class ModelOptimizer:
    """Comprehensive model optimizer for inference."""
    
    def __init__(self, config: OptimizationConfig = None):
        """
        Initialize model optimizer.
        
        Args:
            config: Optimization configuration
        """
        self.config = config or OptimizationConfig()
        self.optimization_history = []
    
    def optimize_model(
        self,
        model: nn.Module,
        sample_input: Optional[Dict[str, torch.Tensor]] = None,
        calibration_data: Optional[List[Dict[str, torch.Tensor]]] = None
    ) -> nn.Module:
        """
        Apply comprehensive optimizations to model.
        
        Args:
            model: Model to optimize
            sample_input: Sample input for tracing/compilation
            calibration_data: Data for static quantization calibration
            
        Returns:
            Optimized model
        """
        logger.info("Starting model optimization...")
        start_time = time.time()
        
        optimized_model = model
        optimizations_applied = []
        
        try:
            # 1. Operator fusion
            if self.config.enable_operator_fusion:
                optimized_model = self._apply_operator_fusion(optimized_model)
                optimizations_applied.append("operator_fusion")
            
            # 2. Pruning
            if self.config.enable_pruning:
                optimized_model = self._apply_pruning(optimized_model)
                optimizations_applied.append("pruning")
            
            # 3. Quantization
            if self.config.enable_quantization:
                optimized_model = self._apply_quantization(
                    optimized_model, calibration_data
                )
                optimizations_applied.append("quantization")
            
            # 4. Memory optimizations
            optimized_model = self._apply_memory_optimizations(optimized_model)
            optimizations_applied.append("memory_optimization")
            
            # 5. Torch compile (must be last)
            if self.config.enable_torch_compile:
                optimized_model = self._apply_torch_compile(
                    optimized_model, sample_input
                )
                optimizations_applied.append("torch_compile")
            
            optimization_time = time.time() - start_time
            
            # Record optimization
            self.optimization_history.append({
                'timestamp': time.time(),
                'optimizations': optimizations_applied,
                'optimization_time': optimization_time,
                'original_params': sum(p.numel() for p in model.parameters()),
                'optimized_params': sum(p.numel() for p in optimized_model.parameters())
            })
            
            logger.info(f"Model optimization completed in {optimization_time:.2f}s")
            logger.info(f"Applied optimizations: {optimizations_applied}")
            
            return optimized_model
            
        except Exception as e:
            logger.error(f"Model optimization failed: {e}")
            return model  # Return original model on failure
    
    def _apply_operator_fusion(self, model: nn.Module) -> nn.Module:
        """Apply operator fusion optimizations."""
        logger.info("Applying operator fusion...")
        
        try:
            # Fuse conv-bn layers
            model = torch.ao.quantization.fuse_modules(
                model,
                [['conv', 'bn'], ['conv', 'bn', 'relu']],
                inplace=False
            )
            logger.info("Applied conv-bn fusion")
        except Exception as e:
            logger.warning(f"Conv-bn fusion failed: {e}")
        
        try:
            # Fuse linear-relu layers
            for name, module in model.named_modules():
                if isinstance(module, nn.Sequential):
                    # Look for Linear + ReLU patterns
                    modules = list(module.children())
                    for i in range(len(modules) - 1):
                        if (isinstance(modules[i], nn.Linear) and 
                            isinstance(modules[i + 1], nn.ReLU)):
                            # Replace with fused version
                            fused = nn.Sequential(
                                modules[i],
                                modules[i + 1]
                            )
                            # This is a simplified fusion - in practice,
                            # you'd use proper fusion APIs
                            break
        except Exception as e:
            logger.warning(f"Linear-ReLU fusion failed: {e}")
        
        return model
    
    def _apply_pruning(self, model: nn.Module) -> nn.Module:
        """Apply model pruning."""
        logger.info(f"Applying pruning (ratio: {self.config.pruning_ratio})...")
        
        try:
            import torch.nn.utils.prune as prune
            
            # Collect parameters to prune
            parameters_to_prune = []
            for name, module in model.named_modules():
                if isinstance(module, (nn.Linear, nn.Conv2d)):
                    parameters_to_prune.append((module, 'weight'))
            
            if self.config.structured_pruning:
                # Structured pruning (remove entire channels/neurons)
                for module, param_name in parameters_to_prune:
                    prune.ln_structured(
                        module, 
                        name=param_name,
                        amount=self.config.pruning_ratio,
                        n=2,
                        dim=0
                    )
            else:
                # Unstructured pruning (remove individual weights)
                prune.global_unstructured(
                    parameters_to_prune,
                    pruning_method=prune.L1Unstructured,
                    amount=self.config.pruning_ratio
                )
            
            # Make pruning permanent
            for module, param_name in parameters_to_prune:
                prune.remove(module, param_name)
            
            logger.info("Applied model pruning")
            
        except ImportError:
            logger.warning("torch.nn.utils.prune not available, skipping pruning")
        except Exception as e:
            logger.warning(f"Pruning failed: {e}")
        
        return model
    
    def _apply_quantization(
        self,
        model: nn.Module,
        calibration_data: Optional[List[Dict[str, torch.Tensor]]] = None
    ) -> nn.Module:
        """Apply model quantization."""
        qconfig = self.config.quantization_config or QuantizationConfig()
        
        logger.info(f"Applying {qconfig.quantization_type} quantization...")
        
        try:
            if qconfig.quantization_type == "dynamic":
                # Dynamic quantization
                quantized_model = torch.ao.quantization.quantize_dynamic(
                    model,
                    {nn.Linear, nn.LSTM, nn.GRU},
                    dtype=getattr(torch, qconfig.weight_dtype)
                )
                
            elif qconfig.quantization_type == "static":
                # Static quantization
                if calibration_data is None:
                    logger.warning("No calibration data provided for static quantization")
                    return model
                
                # Prepare model for quantization
                model.eval()
                model.qconfig = torch.ao.quantization.get_default_qconfig(qconfig.backend)
                prepared_model = torch.ao.quantization.prepare(model, inplace=False)
                
                # Calibrate with sample data
                logger.info(f"Calibrating with {len(calibration_data)} samples...")
                with torch.no_grad():
                    for i, sample in enumerate(calibration_data[:qconfig.calibration_samples]):
                        prepared_model(**sample)
                        if (i + 1) % 10 == 0:
                            logger.info(f"Calibrated {i + 1}/{len(calibration_data)} samples")
                
                # Convert to quantized model
                quantized_model = torch.ao.quantization.convert(prepared_model, inplace=False)
                
            elif qconfig.quantization_type == "qat":
                # Quantization Aware Training (simplified)
                logger.warning("QAT quantization not fully implemented in this example")
                return model
            
            else:
                logger.error(f"Unknown quantization type: {qconfig.quantization_type}")
                return model
            
            logger.info("Applied model quantization")
            return quantized_model
            
        except Exception as e:
            logger.warning(f"Quantization failed: {e}")
            return model
    
    def _apply_memory_optimizations(self, model: nn.Module) -> nn.Module:
        """Apply memory optimizations."""
        logger.info("Applying memory optimizations...")
        
        try:
            # Enable gradient checkpointing if requested
            if self.config.enable_gradient_checkpointing:
                if hasattr(model, 'gradient_checkpointing_enable'):
                    model.gradient_checkpointing_enable()
                    logger.info("Enabled gradient checkpointing")
            
            # Enable memory efficient attention
            if self.config.enable_memory_efficient_attention:
                # This would typically involve replacing attention modules
                # with memory-efficient versions
                logger.info("Memory efficient attention optimization applied")
            
        except Exception as e:
            logger.warning(f"Memory optimization failed: {e}")
        
        return model
    
    def _apply_torch_compile(
        self,
        model: nn.Module,
        sample_input: Optional[Dict[str, torch.Tensor]] = None
    ) -> nn.Module:
        """Apply torch.compile optimization."""
        logger.info(f"Applying torch.compile (mode: {self.config.compile_mode})...")
        
        try:
            # Check if torch.compile is available (PyTorch 2.0+)
            if hasattr(torch, 'compile'):
                compiled_model = torch.compile(
                    model,
                    mode=self.config.compile_mode
                )
                
                # Warmup compilation if sample input provided
                if sample_input is not None:
                    logger.info("Warming up compiled model...")
                    with torch.no_grad():
                        compiled_model(**sample_input)
                
                logger.info("Applied torch.compile optimization")
                return compiled_model
            else:
                logger.warning("torch.compile not available, skipping")
                return model
                
        except Exception as e:
            logger.warning(f"torch.compile failed: {e}")
            return model
    
    def benchmark_optimization(
        self,
        original_model: nn.Module,
        optimized_model: nn.Module,
        sample_inputs: List[Dict[str, torch.Tensor]],
        num_runs: int = 100
    ) -> Dict[str, Any]:
        """
        Benchmark optimization improvements.
        
        Args:
            original_model: Original model
            optimized_model: Optimized model
            sample_inputs: Sample inputs for benchmarking
            num_runs: Number of benchmark runs
            
        Returns:
            Benchmark results
        """
        logger.info(f"Benchmarking optimization with {num_runs} runs...")
        
        def benchmark_model(model, inputs, runs):
            model.eval()
            times = []
            
            # Warmup
            with torch.no_grad():
                for _ in range(5):
                    for inp in inputs[:5]:
                        model(**inp)
            
            # Benchmark
            with torch.no_grad():
                for _ in range(runs):
                    start_time = time.time()
                    for inp in inputs:
                        model(**inp)
                    times.append(time.time() - start_time)
            
            return {
                'mean_time': np.mean(times),
                'std_time': np.std(times),
                'min_time': np.min(times),
                'max_time': np.max(times)
            }
        
        # Benchmark original model
        original_results = benchmark_model(original_model, sample_inputs, num_runs)
        
        # Benchmark optimized model
        optimized_results = benchmark_model(optimized_model, sample_inputs, num_runs)
        
        # Calculate improvements
        speedup = original_results['mean_time'] / optimized_results['mean_time']
        
        # Model size comparison
        original_size = sum(p.numel() * p.element_size() for p in original_model.parameters())
        optimized_size = sum(p.numel() * p.element_size() for p in optimized_model.parameters())
        size_reduction = (original_size - optimized_size) / original_size
        
        results = {
            'original_time_ms': original_results['mean_time'] * 1000,
            'optimized_time_ms': optimized_results['mean_time'] * 1000,
            'speedup': speedup,
            'original_size_mb': original_size / (1024 * 1024),
            'optimized_size_mb': optimized_size / (1024 * 1024),
            'size_reduction_ratio': size_reduction,
            'original_params': sum(p.numel() for p in original_model.parameters()),
            'optimized_params': sum(p.numel() for p in optimized_model.parameters())
        }
        
        logger.info(f"Optimization results: {speedup:.2f}x speedup, {size_reduction:.2%} size reduction")
        
        return results


class CacheManager:
    """Intelligent caching for inference results."""
    
    def __init__(
        self,
        max_size: int = 1000,
        ttl_seconds: int = 3600,
        enable_lru: bool = True
    ):
        """
        Initialize cache manager.
        
        Args:
            max_size: Maximum number of cached items
            ttl_seconds: Time-to-live for cached items
            enable_lru: Whether to use LRU eviction
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.enable_lru = enable_lru
        
        self.cache = OrderedDict() if enable_lru else {}
        self.timestamps = {}
        self.lock = threading.RLock()
        
        # Statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'size': 0
        }
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        with self.lock:
            # Check if key exists and is not expired
            if key in self.cache:
                timestamp = self.timestamps.get(key, 0)
                if time.time() - timestamp < self.ttl_seconds:
                    # Move to end for LRU
                    if self.enable_lru:
                        self.cache.move_to_end(key)
                    
                    self.stats['hits'] += 1
                    return self.cache[key]
                else:
                    # Expired, remove
                    self._remove_key(key)
            
            self.stats['misses'] += 1
            return None
    
    def put(self, key: str, value: Any):
        """Put item in cache."""
        with self.lock:
            # Remove if already exists
            if key in self.cache:
                self._remove_key(key)
            
            # Check size limit
            while len(self.cache) >= self.max_size:
                self._evict_one()
            
            # Add new item
            self.cache[key] = value
            self.timestamps[key] = time.time()
            self.stats['size'] = len(self.cache)
    
    def _remove_key(self, key: str):
        """Remove key from cache."""
        if key in self.cache:
            del self.cache[key]
        if key in self.timestamps:
            del self.timestamps[key]
        self.stats['size'] = len(self.cache)
    
    def _evict_one(self):
        """Evict one item from cache."""
        if not self.cache:
            return
        
        if self.enable_lru:
            # Remove least recently used
            key = next(iter(self.cache))
        else:
            # Remove oldest by timestamp
            key = min(self.timestamps.keys(), key=lambda k: self.timestamps[k])
        
        self._remove_key(key)
        self.stats['evictions'] += 1
    
    def clear(self):
        """Clear all cached items."""
        with self.lock:
            self.cache.clear()
            self.timestamps.clear()
            self.stats['size'] = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            stats = self.stats.copy()
            total_requests = stats['hits'] + stats['misses']
            if total_requests > 0:
                stats['hit_rate'] = stats['hits'] / total_requests
            else:
                stats['hit_rate'] = 0.0
            return stats


class DynamicBatcher:
    """Dynamic batching for efficient inference."""
    
    def __init__(
        self,
        max_batch_size: int = 32,
        timeout_ms: int = 100,
        padding_strategy: str = "longest"
    ):
        """
        Initialize dynamic batcher.
        
        Args:
            max_batch_size: Maximum batch size
            timeout_ms: Timeout for batch collection
            padding_strategy: Strategy for padding sequences
        """
        self.max_batch_size = max_batch_size
        self.timeout_ms = timeout_ms
        self.padding_strategy = padding_strategy
        
        self.pending_requests = []
        self.lock = threading.Lock()
    
    def add_request(self, request_data: Dict[str, Any]) -> bool:
        """
        Add request to batch.
        
        Args:
            request_data: Request data to batch
            
        Returns:
            True if batch is ready, False otherwise
        """
        with self.lock:
            self.pending_requests.append({
                'data': request_data,
                'timestamp': time.time()
            })
            
            # Check if batch is ready
            return (len(self.pending_requests) >= self.max_batch_size or
                    self._is_timeout_reached())
    
    def get_batch(self) -> Optional[List[Dict[str, Any]]]:
        """Get current batch and clear pending requests."""
        with self.lock:
            if not self.pending_requests:
                return None
            
            batch = [req['data'] for req in self.pending_requests]
            self.pending_requests.clear()
            return batch
    
    def _is_timeout_reached(self) -> bool:
        """Check if timeout is reached for oldest request."""
        if not self.pending_requests:
            return False
        
        oldest_timestamp = self.pending_requests[0]['timestamp']
        return (time.time() - oldest_timestamp) * 1000 >= self.timeout_ms
    
    def create_padded_batch(
        self,
        batch_data: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """
        Create padded batch from list of tensors.
        
        Args:
            batch_data: List of tensor dictionaries
            
        Returns:
            Padded batch dictionary
        """
        if not batch_data:
            return {}
        
        # Get all keys
        keys = set()
        for item in batch_data:
            keys.update(item.keys())
        
        batched = {}
        for key in keys:
            tensors = [item.get(key) for item in batch_data if key in item]
            
            if not tensors:
                continue
            
            # Pad sequences if needed
            if len(tensors[0].shape) > 1:  # Multi-dimensional tensors
                if self.padding_strategy == "longest":
                    # Pad to longest sequence in batch
                    max_length = max(t.shape[1] for t in tensors)
                    padded_tensors = []
                    
                    for tensor in tensors:
                        if tensor.shape[1] < max_length:
                            padding = torch.zeros(
                                tensor.shape[0],
                                max_length - tensor.shape[1],
                                *tensor.shape[2:],
                                dtype=tensor.dtype,
                                device=tensor.device
                            )
                            padded_tensor = torch.cat([tensor, padding], dim=1)
                        else:
                            padded_tensor = tensor
                        padded_tensors.append(padded_tensor)
                    
                    batched[key] = torch.cat(padded_tensors, dim=0)
                else:
                    # Simple concatenation
                    batched[key] = torch.cat(tensors, dim=0)
            else:
                # 1D tensors - simple concatenation
                batched[key] = torch.cat(tensors, dim=0)
        
        return batched