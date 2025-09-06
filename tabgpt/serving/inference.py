"""Efficient inference engines for TabGPT models."""

import time
import logging
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass, field
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from queue import Queue, Empty
import gc

import torch
import torch.nn as nn
import pandas as pd
import numpy as np

from ..utils import DataValidator, RobustNormalizer, ValidationError
from ..utils.exceptions import InferenceError, ModelNotTrainedError

logger = logging.getLogger(__name__)


@dataclass
class InferenceConfig:
    """Configuration for inference engines."""
    
    # Batch processing
    batch_size: int = 32
    max_batch_size: int = 128
    batch_timeout_ms: int = 100
    
    # Performance optimization
    use_mixed_precision: bool = False
    use_torch_compile: bool = False
    use_cache: bool = True
    cache_size: int = 1000
    
    # Memory management
    max_memory_mb: int = 4096
    enable_memory_optimization: bool = True
    
    # Threading
    num_workers: int = 4
    max_concurrent_requests: int = 100
    
    # Validation
    validate_inputs: bool = True
    strict_validation: bool = False
    
    # Device settings
    device: str = "auto"  # auto, cpu, cuda, mps
    
    def __post_init__(self):
        """Validate configuration."""
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.max_batch_size < self.batch_size:
            raise ValueError("max_batch_size must be >= batch_size")
        if self.num_workers <= 0:
            raise ValueError("num_workers must be positive")


@dataclass
class InferenceResult:
    """Result of model inference."""
    
    predictions: np.ndarray
    probabilities: Optional[np.ndarray] = None
    features: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Performance metrics
    inference_time_ms: float = 0.0
    preprocessing_time_ms: float = 0.0
    model_time_ms: float = 0.0
    postprocessing_time_ms: float = 0.0
    
    # Quality metrics
    batch_size: int = 0
    memory_usage_mb: float = 0.0
    cache_hit: bool = False


class InferenceEngine:
    """Base inference engine for TabGPT models."""
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        config: InferenceConfig = None
    ):
        """
        Initialize inference engine.
        
        Args:
            model: TabGPT model for inference
            tokenizer: Tokenizer for preprocessing
            config: Inference configuration
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or InferenceConfig()
        
        # Setup device
        self.device = self._setup_device()
        self.model = self.model.to(self.device)
        
        # Setup optimization
        self._setup_optimization()
        
        # Initialize components
        self.validator = None
        self.normalizer = None
        self.cache = {}
        
        # Performance tracking
        self.stats = {
            'total_requests': 0,
            'total_samples': 0,
            'total_time_ms': 0.0,
            'cache_hits': 0,
            'errors': 0
        }
        
        logger.info(f"Initialized inference engine on device: {self.device}")
    
    def _setup_device(self) -> torch.device:
        """Setup computation device."""
        if self.config.device == "auto":
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
        else:
            device = torch.device(self.config.device)
        
        return device
    
    def _setup_optimization(self):
        """Setup model optimizations."""
        # Set model to evaluation mode
        self.model.eval()
        
        # Enable mixed precision if requested
        if self.config.use_mixed_precision and self.device.type == "cuda":
            self.model = self.model.half()
            logger.info("Enabled mixed precision inference")
        
        # Compile model if requested (PyTorch 2.0+)
        if self.config.use_torch_compile:
            try:
                self.model = torch.compile(self.model)
                logger.info("Compiled model with torch.compile")
            except Exception as e:
                logger.warning(f"Failed to compile model: {e}")
    
    def setup_preprocessing(
        self,
        validator: Optional[DataValidator] = None,
        normalizer: Optional[RobustNormalizer] = None
    ):
        """Setup preprocessing components."""
        self.validator = validator
        self.normalizer = normalizer
        
        if validator:
            logger.info("Configured input validation")
        if normalizer:
            logger.info("Configured data normalization")
    
    def predict(
        self,
        data: Union[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]]],
        return_probabilities: bool = False,
        return_features: bool = False
    ) -> InferenceResult:
        """
        Make predictions on input data.
        
        Args:
            data: Input data (DataFrame, dict, or list of dicts)
            return_probabilities: Whether to return prediction probabilities
            return_features: Whether to return extracted features
            
        Returns:
            InferenceResult with predictions and metadata
        """
        start_time = time.time()
        
        try:
            # Convert input to DataFrame
            if isinstance(data, dict):
                df = pd.DataFrame([data])
            elif isinstance(data, list):
                df = pd.DataFrame(data)
            elif isinstance(data, pd.DataFrame):
                df = data.copy()
            else:
                raise InferenceError(f"Unsupported input type: {type(data)}")
            
            # Preprocessing
            preprocess_start = time.time()
            processed_df = self._preprocess_data(df)
            preprocessing_time = (time.time() - preprocess_start) * 1000
            
            # Model inference
            model_start = time.time()
            predictions, probabilities, features = self._run_inference(
                processed_df, return_probabilities, return_features
            )
            model_time = (time.time() - model_start) * 1000
            
            # Postprocessing
            postprocess_start = time.time()
            predictions, probabilities = self._postprocess_predictions(
                predictions, probabilities
            )
            postprocessing_time = (time.time() - postprocess_start) * 1000
            
            # Create result
            total_time = (time.time() - start_time) * 1000
            
            result = InferenceResult(
                predictions=predictions,
                probabilities=probabilities,
                features=features,
                inference_time_ms=total_time,
                preprocessing_time_ms=preprocessing_time,
                model_time_ms=model_time,
                postprocessing_time_ms=postprocessing_time,
                batch_size=len(df),
                memory_usage_mb=self._get_memory_usage()
            )
            
            # Update statistics
            self._update_stats(result)
            
            return result
            
        except Exception as e:
            self.stats['errors'] += 1
            logger.error(f"Inference failed: {e}")
            raise InferenceError(f"Inference failed: {str(e)}") from e
    
    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess input data."""
        # Validate input
        if self.config.validate_inputs and self.validator:
            try:
                validation_result = self.validator.validate_dataframe(
                    df, dataset_name="inference_input"
                )
                if not validation_result['valid'] and self.config.strict_validation:
                    raise ValidationError(f"Input validation failed: {validation_result['errors']}")
            except Exception as e:
                if self.config.strict_validation:
                    raise
                logger.warning(f"Input validation warning: {e}")
        
        # Normalize data
        if self.normalizer:
            try:
                processed_df, _ = self.normalizer.transform(df, handle_errors=True)
                return processed_df
            except Exception as e:
                logger.warning(f"Normalization failed, using original data: {e}")
                return df
        
        return df
    
    def _run_inference(
        self,
        df: pd.DataFrame,
        return_probabilities: bool,
        return_features: bool
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """Run model inference."""
        # Check cache
        cache_key = None
        if self.config.use_cache:
            cache_key = self._compute_cache_key(df)
            if cache_key in self.cache:
                self.stats['cache_hits'] += 1
                cached_result = self.cache[cache_key]
                return cached_result['predictions'], cached_result.get('probabilities'), cached_result.get('features')
        
        # Tokenize input
        try:
            inputs = self.tokenizer.encode_batch(df)
        except Exception as e:
            raise InferenceError(f"Tokenization failed: {str(e)}") from e
        
        # Move to device
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                inputs[key] = value.to(self.device)
        
        # Run model
        with torch.no_grad():
            if self.config.use_mixed_precision and self.device.type == "cuda":
                with torch.cuda.amp.autocast():
                    outputs = self.model(**inputs)
            else:
                outputs = self.model(**inputs)
        
        # Extract predictions
        if hasattr(outputs, 'predictions'):
            predictions = outputs.predictions.cpu().numpy()
        elif hasattr(outputs, 'logits'):
            predictions = outputs.logits.cpu().numpy()
        else:
            raise InferenceError("Model output does not contain predictions or logits")
        
        # Extract probabilities
        probabilities = None
        if return_probabilities:
            if hasattr(outputs, 'probabilities'):
                probabilities = outputs.probabilities.cpu().numpy()
            elif hasattr(outputs, 'logits'):
                # Compute probabilities from logits
                logits = outputs.logits.cpu()
                if logits.shape[-1] == 1:
                    # Binary classification
                    probabilities = torch.sigmoid(logits).numpy()
                else:
                    # Multi-class classification
                    probabilities = torch.softmax(logits, dim=-1).numpy()
        
        # Extract features
        features = None
        if return_features and hasattr(outputs, 'features'):
            features = outputs.features.cpu().numpy()
        
        # Cache result
        if self.config.use_cache and cache_key:
            self.cache[cache_key] = {
                'predictions': predictions,
                'probabilities': probabilities,
                'features': features
            }
            
            # Limit cache size
            if len(self.cache) > self.config.cache_size:
                # Remove oldest entry (simple FIFO)
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
        
        return predictions, probabilities, features
    
    def _postprocess_predictions(
        self,
        predictions: np.ndarray,
        probabilities: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Postprocess model predictions."""
        # Convert logits to class predictions for classification
        if predictions.ndim > 1 and predictions.shape[-1] > 1:
            # Multi-class: take argmax
            predictions = np.argmax(predictions, axis=-1)
        elif predictions.ndim > 1 and predictions.shape[-1] == 1:
            # Binary classification: threshold at 0.5
            if probabilities is not None:
                predictions = (probabilities > 0.5).astype(int).flatten()
            else:
                predictions = (predictions > 0.0).astype(int).flatten()
        
        return predictions, probabilities
    
    def _compute_cache_key(self, df: pd.DataFrame) -> str:
        """Compute cache key for input data."""
        # Simple hash of DataFrame content
        try:
            return str(hash(df.to_string()))
        except Exception:
            # Fallback to shape-based key
            return f"shape_{df.shape[0]}x{df.shape[1]}"
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            if self.device.type == "cuda":
                return torch.cuda.memory_allocated() / (1024 * 1024)
            else:
                import psutil
                process = psutil.Process()
                return process.memory_info().rss / (1024 * 1024)
        except Exception:
            return 0.0
    
    def _update_stats(self, result: InferenceResult):
        """Update performance statistics."""
        self.stats['total_requests'] += 1
        self.stats['total_samples'] += result.batch_size
        self.stats['total_time_ms'] += result.inference_time_ms
        
        if result.cache_hit:
            self.stats['cache_hits'] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        stats = self.stats.copy()
        
        if stats['total_requests'] > 0:
            stats['avg_time_per_request_ms'] = stats['total_time_ms'] / stats['total_requests']
            stats['avg_samples_per_request'] = stats['total_samples'] / stats['total_requests']
            stats['cache_hit_rate'] = stats['cache_hits'] / stats['total_requests']
        
        if stats['total_samples'] > 0:
            stats['avg_time_per_sample_ms'] = stats['total_time_ms'] / stats['total_samples']
        
        return stats
    
    def clear_cache(self):
        """Clear inference cache."""
        self.cache.clear()
        logger.info("Cleared inference cache")
    
    def warmup(self, sample_data: pd.DataFrame, num_warmup: int = 5):
        """Warmup the model with sample data."""
        logger.info(f"Warming up model with {num_warmup} iterations")
        
        for i in range(num_warmup):
            try:
                self.predict(sample_data.iloc[:1])
            except Exception as e:
                logger.warning(f"Warmup iteration {i+1} failed: {e}")
        
        logger.info("Model warmup completed")


class BatchInferenceEngine(InferenceEngine):
    """Batch inference engine with dynamic batching."""
    
    def __init__(self, model: nn.Module, tokenizer: Any, config: InferenceConfig = None):
        super().__init__(model, tokenizer, config)
        
        # Batch processing
        self.batch_queue = Queue()
        self.result_futures = {}
        self.batch_processor_thread = None
        self.shutdown_event = threading.Event()
        
        # Start batch processor
        self._start_batch_processor()
    
    def _start_batch_processor(self):
        """Start background batch processor."""
        self.batch_processor_thread = threading.Thread(
            target=self._batch_processor_loop,
            daemon=True
        )
        self.batch_processor_thread.start()
        logger.info("Started batch processor thread")
    
    def _batch_processor_loop(self):
        """Main loop for batch processing."""
        while not self.shutdown_event.is_set():
            try:
                # Collect batch
                batch_items = []
                batch_data = []
                
                # Wait for first item
                try:
                    item = self.batch_queue.get(timeout=self.config.batch_timeout_ms / 1000)
                    batch_items.append(item)
                    batch_data.append(item['data'])
                except Empty:
                    continue
                
                # Collect additional items up to batch size
                while (len(batch_items) < self.config.max_batch_size and 
                       not self.batch_queue.empty()):
                    try:
                        item = self.batch_queue.get_nowait()
                        batch_items.append(item)
                        batch_data.append(item['data'])
                    except Empty:
                        break
                
                # Process batch
                if batch_data:
                    self._process_batch(batch_items, batch_data)
                    
            except Exception as e:
                logger.error(f"Batch processor error: {e}")
    
    def _process_batch(self, batch_items: List[Dict], batch_data: List[pd.DataFrame]):
        """Process a batch of requests."""
        try:
            # Combine data
            combined_df = pd.concat(batch_data, ignore_index=True)
            
            # Run inference
            batch_result = super().predict(
                combined_df,
                return_probabilities=batch_items[0].get('return_probabilities', False),
                return_features=batch_items[0].get('return_features', False)
            )
            
            # Split results
            start_idx = 0
            for item in batch_items:
                end_idx = start_idx + len(item['data'])
                
                # Extract individual result
                individual_result = InferenceResult(
                    predictions=batch_result.predictions[start_idx:end_idx],
                    probabilities=batch_result.probabilities[start_idx:end_idx] if batch_result.probabilities is not None else None,
                    features=batch_result.features[start_idx:end_idx] if batch_result.features is not None else None,
                    metadata=batch_result.metadata,
                    inference_time_ms=batch_result.inference_time_ms,
                    preprocessing_time_ms=batch_result.preprocessing_time_ms,
                    model_time_ms=batch_result.model_time_ms,
                    postprocessing_time_ms=batch_result.postprocessing_time_ms,
                    batch_size=len(item['data']),
                    memory_usage_mb=batch_result.memory_usage_mb,
                    cache_hit=batch_result.cache_hit
                )
                
                # Set result
                item['future'].set_result(individual_result)
                start_idx = end_idx
                
        except Exception as e:
            # Set exception for all items in batch
            for item in batch_items:
                item['future'].set_exception(e)
    
    def predict_async(
        self,
        data: Union[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]]],
        return_probabilities: bool = False,
        return_features: bool = False
    ) -> 'Future[InferenceResult]':
        """
        Asynchronous prediction with batching.
        
        Args:
            data: Input data
            return_probabilities: Whether to return probabilities
            return_features: Whether to return features
            
        Returns:
            Future that will contain the InferenceResult
        """
        from concurrent.futures import Future
        
        # Convert to DataFrame
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        elif isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, pd.DataFrame):
            df = data.copy()
        else:
            raise InferenceError(f"Unsupported input type: {type(data)}")
        
        # Create future
        future = Future()
        
        # Add to batch queue
        batch_item = {
            'data': df,
            'return_probabilities': return_probabilities,
            'return_features': return_features,
            'future': future
        }
        
        self.batch_queue.put(batch_item)
        
        return future
    
    def shutdown(self):
        """Shutdown batch processor."""
        self.shutdown_event.set()
        if self.batch_processor_thread:
            self.batch_processor_thread.join(timeout=5.0)
        logger.info("Batch inference engine shutdown")


class OptimizedInferenceEngine(BatchInferenceEngine):
    """Highly optimized inference engine with all performance features."""
    
    def __init__(self, model: nn.Module, tokenizer: Any, config: InferenceConfig = None):
        super().__init__(model, tokenizer, config)
        
        # Additional optimizations
        self._setup_advanced_optimizations()
        
        # Memory management
        self.memory_monitor = threading.Thread(
            target=self._memory_monitor_loop,
            daemon=True
        )
        self.memory_monitor.start()
    
    def _setup_advanced_optimizations(self):
        """Setup advanced model optimizations."""
        # Enable optimized attention if available
        try:
            if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
                # PyTorch 2.0+ optimized attention
                logger.info("Using optimized scaled_dot_product_attention")
        except Exception:
            pass
        
        # Enable memory efficient attention patterns
        if self.device.type == "cuda":
            try:
                torch.backends.cuda.enable_flash_sdp(True)
                logger.info("Enabled Flash Attention")
            except Exception:
                pass
    
    def _memory_monitor_loop(self):
        """Monitor memory usage and trigger cleanup if needed."""
        while not self.shutdown_event.is_set():
            try:
                memory_mb = self._get_memory_usage()
                
                if memory_mb > self.config.max_memory_mb * 0.9:  # 90% threshold
                    logger.warning(f"High memory usage: {memory_mb:.1f}MB")
                    self._cleanup_memory()
                
                time.sleep(5.0)  # Check every 5 seconds
                
            except Exception as e:
                logger.error(f"Memory monitor error: {e}")
    
    def _cleanup_memory(self):
        """Cleanup memory when usage is high."""
        # Clear cache
        self.clear_cache()
        
        # Force garbage collection
        gc.collect()
        
        # Clear CUDA cache if available
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        
        logger.info("Performed memory cleanup")
    
    def predict_batch(
        self,
        data_list: List[Union[pd.DataFrame, Dict[str, Any]]],
        return_probabilities: bool = False,
        return_features: bool = False,
        max_workers: Optional[int] = None
    ) -> List[InferenceResult]:
        """
        Predict on multiple datasets in parallel.
        
        Args:
            data_list: List of input data
            return_probabilities: Whether to return probabilities
            return_features: Whether to return features
            max_workers: Maximum number of worker threads
            
        Returns:
            List of InferenceResults
        """
        max_workers = max_workers or self.config.num_workers
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            futures = []
            for data in data_list:
                future = executor.submit(
                    self.predict,
                    data,
                    return_probabilities,
                    return_features
                )
                futures.append(future)
            
            # Collect results
            results = []
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Batch prediction failed: {e}")
                    # Create error result
                    error_result = InferenceResult(
                        predictions=np.array([]),
                        metadata={'error': str(e)}
                    )
                    results.append(error_result)
        
        return results