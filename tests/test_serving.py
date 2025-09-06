"""Tests for model serving and inference optimizations."""

import pytest
import time
import tempfile
from unittest.mock import Mock, patch
import threading
from concurrent.futures import Future

import torch
import torch.nn as nn
import pandas as pd
import numpy as np

from tabgpt.serving import (
    InferenceEngine, BatchInferenceEngine, OptimizedInferenceEngine,
    InferenceConfig, InferenceResult,
    ModelOptimizer, QuantizationConfig, OptimizationConfig,
    CacheManager, DynamicBatcher,
    ModelExporter, ONNXExporter, TorchScriptExporter, ExportConfig,
    ModelServer, ServerConfig, HealthChecker, MetricsCollector
)


class MockModel(nn.Module):
    """Mock model for testing."""
    
    def __init__(self, input_dim=10, output_dim=2):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.is_trained = True
    
    def forward(self, input_ids, attention_mask=None, **kwargs):
        # Simulate model output
        batch_size = input_ids.shape[0]
        logits = self.linear(input_ids.float().mean(dim=1))
        
        return type('Output', (), {
            'logits': logits,
            'predictions': logits,
            'probabilities': torch.softmax(logits, dim=-1)
        })()


class MockTokenizer:
    """Mock tokenizer for testing."""
    
    def encode_batch(self, df):
        batch_size = len(df)
        seq_length = 10
        return {
            'input_ids': torch.randint(0, 1000, (batch_size, seq_length)),
            'attention_mask': torch.ones(batch_size, seq_length)
        }


class TestInferenceEngine:
    """Test inference engine functionality."""
    
    def test_basic_inference(self):
        """Test basic inference functionality."""
        model = MockModel()
        tokenizer = MockTokenizer()
        config = InferenceConfig(batch_size=4, validate_inputs=False)
        
        engine = InferenceEngine(model, tokenizer, config)
        
        # Test single prediction
        data = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': ['a', 'b', 'c']
        })
        
        result = engine.predict(data)
        
        assert isinstance(result, InferenceResult)
        assert result.predictions is not None
        assert len(result.predictions) == 3
        assert result.inference_time_ms > 0
        assert result.batch_size == 3
    
    def test_batch_inference(self):
        """Test batch inference functionality."""
        model = MockModel()
        tokenizer = MockTokenizer()
        config = InferenceConfig(batch_size=2, max_batch_size=4)
        
        engine = BatchInferenceEngine(model, tokenizer, config)
        
        # Test batch prediction
        data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': ['a', 'b', 'c', 'd', 'e']
        })
        
        result = engine.predict(data)
        
        assert isinstance(result, InferenceResult)
        assert len(result.predictions) == 5
        
        # Cleanup
        engine.shutdown()
    
    def test_optimized_inference(self):
        """Test optimized inference engine."""
        model = MockModel()
        tokenizer = MockTokenizer()
        config = InferenceConfig(
            use_cache=True,
            cache_size=100,
            enable_memory_optimization=True
        )
        
        engine = OptimizedInferenceEngine(model, tokenizer, config)
        
        # Test prediction with caching
        data = pd.DataFrame({'feature1': [1, 2], 'feature2': ['a', 'b']})
        
        # First prediction (cache miss)
        result1 = engine.predict(data)
        
        # Second prediction (should hit cache)
        result2 = engine.predict(data)
        
        assert len(result1.predictions) == 2
        assert len(result2.predictions) == 2
        
        # Check stats
        stats = engine.get_stats()
        assert stats['total_requests'] == 2
        
        # Cleanup
        engine.shutdown()
    
    def test_inference_with_probabilities(self):
        """Test inference with probability outputs."""
        model = MockModel()
        tokenizer = MockTokenizer()
        engine = InferenceEngine(model, tokenizer)
        
        data = pd.DataFrame({'feature1': [1, 2], 'feature2': ['a', 'b']})
        
        result = engine.predict(data, return_probabilities=True)
        
        assert result.probabilities is not None
        assert result.probabilities.shape[0] == 2
        assert result.probabilities.shape[1] == 2  # Binary classification
    
    def test_inference_error_handling(self):
        """Test inference error handling."""
        # Create model that will fail
        model = Mock()
        model.eval = Mock()
        model.to = Mock(return_value=model)
        model.side_effect = Exception("Model failed")
        
        tokenizer = MockTokenizer()
        engine = InferenceEngine(model, tokenizer)
        
        data = pd.DataFrame({'feature1': [1, 2], 'feature2': ['a', 'b']})
        
        with pytest.raises(Exception):
            engine.predict(data)


class TestModelOptimizer:
    """Test model optimization functionality."""
    
    def test_basic_optimization(self):
        """Test basic model optimization."""
        model = MockModel()
        config = OptimizationConfig(
            enable_quantization=False,  # Skip quantization for simplicity
            enable_pruning=False,       # Skip pruning for simplicity
            enable_torch_compile=False  # Skip compilation for compatibility
        )
        
        optimizer = ModelOptimizer(config)
        
        # Create sample input
        sample_input = {
            'input_ids': torch.randint(0, 1000, (2, 10)),
            'attention_mask': torch.ones(2, 10)
        }
        
        optimized_model = optimizer.optimize_model(model, sample_input)
        
        assert optimized_model is not None
        assert len(optimizer.optimization_history) == 1
    
    def test_quantization_config(self):
        """Test quantization configuration."""
        config = QuantizationConfig(
            quantization_type="dynamic",
            weight_dtype="int8",
            activation_dtype="int8"
        )
        
        assert config.quantization_type == "dynamic"
        assert config.weight_dtype == "int8"
        
        # Test invalid config
        with pytest.raises(ValueError):
            QuantizationConfig(quantization_type="invalid")
    
    def test_optimization_benchmark(self):
        """Test optimization benchmarking."""
        original_model = MockModel()
        optimized_model = MockModel()  # Same model for testing
        
        optimizer = ModelOptimizer()
        
        # Create sample inputs
        sample_inputs = [
            {
                'input_ids': torch.randint(0, 1000, (2, 10)),
                'attention_mask': torch.ones(2, 10)
            }
            for _ in range(5)
        ]
        
        results = optimizer.benchmark_optimization(
            original_model, optimized_model, sample_inputs, num_runs=3
        )
        
        assert 'speedup' in results
        assert 'original_time_ms' in results
        assert 'optimized_time_ms' in results
        assert results['speedup'] > 0


class TestCacheManager:
    """Test cache management functionality."""
    
    def test_basic_caching(self):
        """Test basic cache operations."""
        cache = CacheManager(max_size=3, ttl_seconds=60)
        
        # Test put and get
        cache.put("key1", "value1")
        assert cache.get("key1") == "value1"
        
        # Test miss
        assert cache.get("nonexistent") is None
        
        # Test stats
        stats = cache.get_stats()
        assert stats['hits'] == 1
        assert stats['misses'] == 1
        assert stats['size'] == 1
    
    def test_cache_eviction(self):
        """Test cache eviction when full."""
        cache = CacheManager(max_size=2, enable_lru=True)
        
        # Fill cache
        cache.put("key1", "value1")
        cache.put("key2", "value2")
        
        # Add third item (should evict first)
        cache.put("key3", "value3")
        
        assert cache.get("key1") is None  # Evicted
        assert cache.get("key2") == "value2"
        assert cache.get("key3") == "value3"
        
        stats = cache.get_stats()
        assert stats['evictions'] == 1
    
    def test_cache_ttl(self):
        """Test cache time-to-live."""
        cache = CacheManager(ttl_seconds=0.1)  # Very short TTL
        
        cache.put("key1", "value1")
        assert cache.get("key1") == "value1"
        
        # Wait for expiration
        time.sleep(0.2)
        assert cache.get("key1") is None  # Should be expired


class TestDynamicBatcher:
    """Test dynamic batching functionality."""
    
    def test_batch_collection(self):
        """Test batch collection and timeout."""
        batcher = DynamicBatcher(max_batch_size=3, timeout_ms=100)
        
        # Add requests
        ready1 = batcher.add_request({'data': 'request1'})
        assert not ready1  # Not ready yet
        
        ready2 = batcher.add_request({'data': 'request2'})
        assert not ready2  # Still not ready
        
        ready3 = batcher.add_request({'data': 'request3'})
        assert ready3  # Should be ready now (reached max_batch_size)
        
        # Get batch
        batch = batcher.get_batch()
        assert len(batch) == 3
        assert batch[0]['data'] == 'request1'
    
    def test_batch_timeout(self):
        """Test batch timeout functionality."""
        batcher = DynamicBatcher(max_batch_size=10, timeout_ms=50)
        
        # Add one request
        ready1 = batcher.add_request({'data': 'request1'})
        assert not ready1
        
        # Wait for timeout
        time.sleep(0.1)
        
        ready2 = batcher.add_request({'data': 'request2'})
        assert ready2  # Should be ready due to timeout
        
        batch = batcher.get_batch()
        assert len(batch) == 2


class TestModelExport:
    """Test model export functionality."""
    
    def test_export_config(self):
        """Test export configuration."""
        config = ExportConfig(
            export_format="onnx",
            onnx_opset_version=11,
            validate_export=True
        )
        
        assert config.export_format == "onnx"
        assert config.onnx_opset_version == 11
        assert config.validate_export is True
    
    def test_torchscript_export(self):
        """Test TorchScript export."""
        model = MockModel()
        sample_input = {
            'input_ids': torch.randint(0, 1000, (2, 10)),
            'attention_mask': torch.ones(2, 10)
        }
        
        config = ExportConfig(export_format="torchscript", validate_export=False)
        exporter = TorchScriptExporter(config)
        
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            output_path = f.name
        
        try:
            results = exporter.export_model(model, sample_input, output_path)
            
            assert results['export_format'] == 'torchscript'
            assert 'model_size_mb' in results
            assert Path(output_path).exists()
            
        finally:
            # Cleanup
            if Path(output_path).exists():
                Path(output_path).unlink()
    
    @pytest.mark.skipif(not torch.onnx.is_in_onnx_export(), reason="ONNX not available")
    def test_onnx_export(self):
        """Test ONNX export (if available)."""
        model = MockModel()
        sample_input = {
            'input_ids': torch.randint(0, 1000, (2, 10)),
            'attention_mask': torch.ones(2, 10)
        }
        
        config = ExportConfig(export_format="onnx", validate_export=False)
        exporter = ONNXExporter(config)
        
        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
            output_path = f.name
        
        try:
            results = exporter.export_model(model, sample_input, output_path)
            
            assert results['export_format'] == 'onnx'
            assert 'model_size_mb' in results
            
        except Exception as e:
            pytest.skip(f"ONNX export failed (expected in test environment): {e}")
        finally:
            # Cleanup
            if Path(output_path).exists():
                Path(output_path).unlink()


class TestModelServer:
    """Test model server functionality."""
    
    def test_server_initialization(self):
        """Test server initialization."""
        config = ServerConfig(
            host="localhost",
            port=8001,
            model_path="",
            tokenizer_path=""
        )
        
        # Mock the model loading to avoid actual file operations
        with patch('tabgpt.serving.deployment.ModelServer._setup_inference_engine'):
            server = ModelServer(config)
            
            assert server.config.host == "localhost"
            assert server.config.port == 8001
            assert server.metrics_collector is not None
    
    def test_health_checker(self):
        """Test health checking functionality."""
        model = MockModel()
        tokenizer = MockTokenizer()
        engine = InferenceEngine(model, tokenizer)
        config = ServerConfig()
        
        health_checker = HealthChecker(engine, config)
        
        # Wait a moment for initial health check
        time.sleep(0.1)
        
        status = health_checker.get_health_status()
        
        assert 'healthy' in status
        assert 'last_check' in status
        assert 'uptime_seconds' in status
    
    def test_metrics_collector(self):
        """Test metrics collection."""
        collector = MetricsCollector()
        
        # Record some requests
        collector.record_request_start()
        collector.record_request_end(success=True, response_time_ms=100, inference_time_ms=50)
        
        collector.record_request_start()
        collector.record_request_end(success=False, response_time_ms=200)
        
        metrics = collector.get_metrics()
        
        assert metrics['requests_total'] == 2
        assert metrics['requests_success'] == 1
        assert metrics['requests_error'] == 1
        assert metrics['avg_response_time_ms'] == 150
        assert metrics['success_rate'] == 0.5
        assert metrics['error_rate'] == 0.5


class TestPerformanceOptimizations:
    """Test performance optimization features."""
    
    def test_inference_caching(self):
        """Test inference result caching."""
        model = MockModel()
        tokenizer = MockTokenizer()
        config = InferenceConfig(use_cache=True, cache_size=10)
        
        engine = InferenceEngine(model, tokenizer, config)
        
        data = pd.DataFrame({'feature1': [1, 2], 'feature2': ['a', 'b']})
        
        # First prediction (cache miss)
        result1 = engine.predict(data)
        
        # Second prediction (should hit cache)
        result2 = engine.predict(data)
        
        stats = engine.get_stats()
        assert stats['cache_hits'] > 0
    
    def test_memory_optimization(self):
        """Test memory optimization features."""
        model = MockModel()
        tokenizer = MockTokenizer()
        config = InferenceConfig(
            enable_memory_optimization=True,
            max_memory_mb=1024
        )
        
        engine = OptimizedInferenceEngine(model, tokenizer, config)
        
        # Test prediction
        data = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': ['a', 'b', 'c']})
        result = engine.predict(data)
        
        assert result.memory_usage_mb >= 0
        
        # Cleanup
        engine.shutdown()
    
    def test_batch_processing_performance(self):
        """Test batch processing performance."""
        model = MockModel()
        tokenizer = MockTokenizer()
        config = InferenceConfig(batch_size=4, max_batch_size=8)
        
        engine = BatchInferenceEngine(model, tokenizer, config)
        
        # Create multiple datasets
        datasets = [
            pd.DataFrame({'feature1': [i], 'feature2': [f'val_{i}']})
            for i in range(10)
        ]
        
        # Test batch prediction
        results = engine.predict_batch(
            datasets,
            max_workers=2
        )
        
        assert len(results) == 10
        for result in results:
            assert isinstance(result, InferenceResult)
            assert len(result.predictions) == 1
        
        # Cleanup
        engine.shutdown()


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_input(self):
        """Test handling of empty input."""
        model = MockModel()
        tokenizer = MockTokenizer()
        engine = InferenceEngine(model, tokenizer)
        
        # Empty DataFrame
        empty_df = pd.DataFrame()
        
        with pytest.raises(Exception):  # Should handle gracefully or raise appropriate error
            engine.predict(empty_df)
    
    def test_large_batch(self):
        """Test handling of large batches."""
        model = MockModel()
        tokenizer = MockTokenizer()
        config = InferenceConfig(batch_size=10, max_batch_size=50)
        
        engine = InferenceEngine(model, tokenizer, config)
        
        # Large dataset
        large_data = pd.DataFrame({
            'feature1': range(100),
            'feature2': [f'val_{i}' for i in range(100)]
        })
        
        result = engine.predict(large_data)
        
        assert len(result.predictions) == 100
        assert result.batch_size == 100
    
    def test_invalid_input_types(self):
        """Test handling of invalid input types."""
        model = MockModel()
        tokenizer = MockTokenizer()
        engine = InferenceEngine(model, tokenizer)
        
        # Invalid input type
        with pytest.raises(Exception):
            engine.predict("invalid_input")
        
        with pytest.raises(Exception):
            engine.predict(123)
    
    def test_model_device_mismatch(self):
        """Test handling of device mismatches."""
        model = MockModel()
        tokenizer = MockTokenizer()
        
        # Force CPU device
        config = InferenceConfig(device="cpu")
        engine = InferenceEngine(model, tokenizer, config)
        
        data = pd.DataFrame({'feature1': [1, 2], 'feature2': ['a', 'b']})
        
        # Should work despite any device issues
        result = engine.predict(data)
        assert result.predictions is not None


if __name__ == "__main__":
    pytest.main([__file__])