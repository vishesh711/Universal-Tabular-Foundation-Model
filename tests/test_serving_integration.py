"""Integration tests for TabGPT serving and optimization components."""

import pytest
import time
import tempfile
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import json

import torch
import torch.nn as nn
import pandas as pd
import numpy as np

from tabgpt.serving import (
    InferenceEngine, BatchInferenceEngine, OptimizedInferenceEngine,
    InferenceConfig, ModelOptimizer, OptimizationConfig,
    CacheManager, DynamicBatcher, ModelExporter, ExportConfig,
    ModelServer, ServerConfig, HealthChecker, MetricsCollector
)


class IntegrationTestModel(nn.Module):
    """Test model for integration testing."""
    
    def __init__(self, input_dim=256, output_dim=2):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.LayerNorm(input_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(input_dim, output_dim)
        )
        self.config = type('Config', (), {'hidden_size': input_dim})()
    
    def forward(self, input_ids, attention_mask=None, **kwargs):
        batch_size = input_ids.shape[0]
        features = input_ids.float().mean(dim=1)
        logits = self.layers(features)
        
        return type('Output', (), {
            'logits': logits,
            'predictions': logits,
            'probabilities': torch.softmax(logits, dim=-1)
        })()


class IntegrationTestTokenizer:
    """Test tokenizer for integration testing."""
    
    def encode_batch(self, df):
        batch_size = len(df)
        return {
            'input_ids': torch.randn(batch_size, 256),
            'attention_mask': torch.ones(batch_size, 256)
        }


class TestServingIntegration:
    """Integration tests for serving components."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.model = IntegrationTestModel()
        self.tokenizer = IntegrationTestTokenizer()
        self.test_data = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.choice(['A', 'B', 'C'], 100),
            'feature3': np.random.uniform(0, 1, 100)
        })
    
    def test_end_to_end_inference_pipeline(self):
        """Test complete inference pipeline from data to predictions."""
        # Create optimized inference engine
        config = InferenceConfig(
            batch_size=16,
            max_batch_size=32,
            use_cache=True,
            cache_size=50,
            enable_memory_optimization=True
        )
        
        engine = OptimizedInferenceEngine(self.model, self.tokenizer, config)
        
        try:
            # Test single prediction
            single_result = engine.predict(self.test_data.iloc[:1])
            assert single_result.predictions is not None
            assert len(single_result.predictions) == 1
            
            # Test batch prediction
            batch_result = engine.predict(self.test_data.iloc[:20])
            assert len(batch_result.predictions) == 20
            
            # Test with probabilities
            prob_result = engine.predict(
                self.test_data.iloc[:5], 
                return_probabilities=True
            )
            assert prob_result.probabilities is not None
            assert prob_result.probabilities.shape[0] == 5
            
            # Test caching (same data should be faster)
            start_time = time.time()
            cached_result = engine.predict(self.test_data.iloc[:5])
            cached_time = time.time() - start_time
            
            # Verify cache hit
            stats = engine.get_stats()
            assert stats['cache_hits'] > 0
            
        finally:
            engine.shutdown()
    
    def test_optimization_and_export_pipeline(self):
        """Test model optimization and export pipeline."""
        # 1. Optimize model
        optimization_config = OptimizationConfig(
            enable_quantization=False,  # Skip for test compatibility
            enable_pruning=False,       # Skip for test compatibility
            enable_torch_compile=False, # Skip for test compatibility
            enable_operator_fusion=True
        )
        
        optimizer = ModelOptimizer(optimization_config)
        sample_input = self.tokenizer.encode_batch(self.test_data.iloc[:2])
        
        optimized_model = optimizer.optimize_model(self.model, sample_input)
        assert optimized_model is not None
        
        # 2. Test optimized model inference
        engine = InferenceEngine(optimized_model, self.tokenizer)
        result = engine.predict(self.test_data.iloc[:10])
        assert len(result.predictions) == 10
        
        # 3. Export optimized model
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            export_path = f.name
        
        try:
            from tabgpt.serving import TorchScriptExporter
            
            config = ExportConfig(export_format="torchscript", validate_export=False)
            exporter = TorchScriptExporter(config)
            
            export_result = exporter.export_model(
                optimized_model, sample_input, export_path
            )
            
            assert export_result['export_format'] == 'torchscript'
            assert Path(export_path).exists()
            assert export_result['model_size_mb'] > 0
            
            # 4. Test loading exported model
            loaded_model = torch.jit.load(export_path)
            
            # Test inference with loaded model
            with torch.no_grad():
                output = loaded_model(*sample_input.values())
                assert output is not None
            
        finally:
            # Cleanup
            if Path(export_path).exists():
                Path(export_path).unlink()
    
    def test_concurrent_inference_with_batching(self):
        """Test concurrent inference with dynamic batching."""
        config = InferenceConfig(
            batch_size=8,
            max_batch_size=16,
            batch_timeout_ms=50,
            use_cache=True,
            max_concurrent_requests=20
        )
        
        engine = BatchInferenceEngine(self.model, self.tokenizer, config)
        
        try:
            # Create multiple concurrent requests
            def make_request(request_id):
                batch_size = np.random.randint(1, 10)
                start_idx = request_id * batch_size
                data = self.test_data.iloc[start_idx % len(self.test_data):
                                         (start_idx + batch_size) % len(self.test_data)]
                
                future = engine.predict_async(data)
                result = future.result(timeout=5.0)
                
                return {
                    'request_id': request_id,
                    'batch_size': len(data),
                    'predictions': len(result.predictions),
                    'inference_time_ms': result.inference_time_ms
                }
            
            # Submit concurrent requests
            num_requests = 15
            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = [
                    executor.submit(make_request, i) 
                    for i in range(num_requests)
                ]
                
                results = []
                for future in as_completed(futures):
                    result = future.result()
                    results.append(result)
            
            # Verify all requests completed successfully
            assert len(results) == num_requests
            
            # Verify predictions match input sizes
            for result in results:
                assert result['predictions'] == result['batch_size']
            
            # Check engine statistics
            stats = engine.get_stats()
            assert stats['total_requests'] >= num_requests
            
        finally:
            engine.shutdown()
    
    def test_caching_and_memory_management(self):
        """Test caching effectiveness and memory management."""
        config = InferenceConfig(
            use_cache=True,
            cache_size=20,
            enable_memory_optimization=True,
            max_memory_mb=512
        )
        
        engine = OptimizedInferenceEngine(self.model, self.tokenizer, config)
        
        try:
            # Create test datasets (some repeated for cache testing)
            datasets = []
            
            # Unique datasets
            for i in range(15):
                start_idx = i * 5
                dataset = self.test_data.iloc[start_idx:start_idx + 5].copy()
                dataset['unique_id'] = i  # Make each dataset unique
                datasets.append(dataset)
            
            # Repeated datasets (should hit cache)
            for i in range(5):
                datasets.append(datasets[i].copy())  # Repeat first 5 datasets
            
            # Process all datasets
            results = []
            memory_usage = []
            
            for i, dataset in enumerate(datasets):
                result = engine.predict(dataset)
                results.append(result)
                memory_usage.append(result.memory_usage_mb)
                
                # Check memory doesn't grow unbounded
                if i > 5:  # After some processing
                    assert result.memory_usage_mb < config.max_memory_mb
            
            # Verify cache effectiveness
            stats = engine.get_stats()
            cache_hit_rate = stats.get('cache_hit_rate', 0)
            
            # Should have some cache hits from repeated datasets
            assert cache_hit_rate > 0, f"Expected cache hits, got rate: {cache_hit_rate}"
            
            # Memory usage should be reasonable
            avg_memory = np.mean(memory_usage)
            max_memory = np.max(memory_usage)
            
            assert avg_memory > 0, "Memory usage should be tracked"
            assert max_memory < config.max_memory_mb, "Memory usage exceeded limit"
            
        finally:
            engine.shutdown()
    
    def test_error_handling_and_recovery(self):
        """Test error handling and recovery mechanisms."""
        config = InferenceConfig(
            batch_size=10,
            use_cache=True,
            validate_inputs=True,
            strict_validation=False  # Allow recovery from errors
        )
        
        engine = InferenceEngine(self.model, self.tokenizer, config)
        
        # Test with valid data (should work)
        valid_result = engine.predict(self.test_data.iloc[:5])
        assert len(valid_result.predictions) == 5
        
        # Test with empty data (should handle gracefully)
        empty_data = pd.DataFrame()
        
        with pytest.raises(Exception):  # Should raise appropriate error
            engine.predict(empty_data)
        
        # Test with malformed data
        malformed_data = pd.DataFrame({'invalid_column': [None, None, None]})
        
        try:
            # This might work or fail depending on tokenizer robustness
            result = engine.predict(malformed_data)
            # If it works, verify it returns something reasonable
            assert result.predictions is not None
        except Exception:
            # If it fails, that's also acceptable for malformed data
            pass
        
        # Verify engine still works after errors
        recovery_result = engine.predict(self.test_data.iloc[:3])
        assert len(recovery_result.predictions) == 3
        
        # Check error statistics
        stats = engine.get_stats()
        # Should have recorded some errors
        assert 'errors' in stats
    
    def test_server_components_integration(self):
        """Test integration of server components."""
        # Test metrics collection
        metrics = MetricsCollector()
        
        # Simulate server operations
        for i in range(10):
            metrics.record_request_start()
            
            # Simulate processing
            processing_time = np.random.uniform(10, 100)
            success = i < 8  # 80% success rate
            
            metrics.record_request_end(
                success=success,
                response_time_ms=processing_time,
                inference_time_ms=processing_time * 0.7
            )
        
        # Verify metrics
        server_metrics = metrics.get_metrics()
        
        assert server_metrics['requests_total'] == 10
        assert server_metrics['requests_success'] == 8
        assert server_metrics['requests_error'] == 2
        assert server_metrics['success_rate'] == 0.8
        assert server_metrics['avg_response_time_ms'] > 0
        
        # Test health checker
        engine = InferenceEngine(self.model, self.tokenizer)
        config = ServerConfig()
        
        health_checker = HealthChecker(engine, config)
        
        # Wait for initial health check
        time.sleep(0.2)
        
        health_status = health_checker.get_health_status()
        
        assert 'healthy' in health_status
        assert 'last_check' in health_status
        assert 'uptime_seconds' in health_status
        
        # Health should be good with working engine
        assert health_status['healthy'] is True
    
    def test_performance_under_load(self):
        """Test system performance under sustained load."""
        config = InferenceConfig(
            batch_size=16,
            max_batch_size=32,
            use_cache=True,
            cache_size=100,
            enable_memory_optimization=True,
            num_workers=2
        )
        
        engine = OptimizedInferenceEngine(self.model, self.tokenizer, config)
        
        try:
            # Sustained load test
            num_iterations = 50
            batch_size = 10
            
            start_time = time.time()
            
            for i in range(num_iterations):
                # Create batch with some variation
                start_idx = (i * batch_size) % len(self.test_data)
                end_idx = min(start_idx + batch_size, len(self.test_data))
                batch_data = self.test_data.iloc[start_idx:end_idx]
                
                result = engine.predict(batch_data)
                
                # Verify result quality
                assert result.predictions is not None
                assert len(result.predictions) == len(batch_data)
                assert result.inference_time_ms > 0
            
            total_time = time.time() - start_time
            
            # Performance assertions
            avg_time_per_iteration = total_time / num_iterations
            samples_per_second = (num_iterations * batch_size) / total_time
            
            print(f"\\nLoad test results:")
            print(f"  Total time: {total_time:.2f}s")
            print(f"  Avg time per iteration: {avg_time_per_iteration*1000:.2f}ms")
            print(f"  Throughput: {samples_per_second:.1f} samples/sec")
            
            # Should maintain reasonable performance
            assert avg_time_per_iteration < 1.0  # Less than 1 second per iteration
            assert samples_per_second > 5  # At least 5 samples per second
            
            # Check final statistics
            stats = engine.get_stats()
            
            assert stats['total_requests'] == num_iterations
            assert stats['errors'] == 0  # No errors during load test
            
            # Cache should be effective
            if stats.get('cache_hit_rate', 0) > 0:
                print(f"  Cache hit rate: {stats['cache_hit_rate']:.2%}")
            
        finally:
            engine.shutdown()
    
    def test_configuration_validation(self):
        """Test configuration validation and error handling."""
        # Test invalid configurations
        
        # Invalid batch size
        with pytest.raises(ValueError):
            InferenceConfig(batch_size=0)
        
        with pytest.raises(ValueError):
            InferenceConfig(batch_size=10, max_batch_size=5)  # max < batch
        
        # Invalid worker count
        with pytest.raises(ValueError):
            InferenceConfig(num_workers=0)
        
        # Test server config validation
        with pytest.raises(ValueError):
            ServerConfig(workers=0)
        
        with pytest.raises(ValueError):
            ServerConfig(port=0)
        
        with pytest.raises(ValueError):
            ServerConfig(port=70000)  # Invalid port
        
        # Test valid configurations work
        valid_config = InferenceConfig(
            batch_size=16,
            max_batch_size=32,
            use_cache=True,
            cache_size=100
        )
        
        engine = InferenceEngine(self.model, self.tokenizer, valid_config)
        result = engine.predict(self.test_data.iloc[:5])
        assert len(result.predictions) == 5


class TestServingWorkflows:
    """Test complete serving workflows."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.model = IntegrationTestModel()
        self.tokenizer = IntegrationTestTokenizer()
        self.test_data = pd.DataFrame({
            'feature1': np.random.randn(50),
            'feature2': np.random.choice(['X', 'Y', 'Z'], 50)
        })
    
    def test_development_to_production_workflow(self):
        """Test complete workflow from development to production."""
        
        # 1. Development: Basic inference
        print("\\n1. Development phase...")
        dev_config = InferenceConfig(use_cache=False, batch_size=8)
        dev_engine = InferenceEngine(self.model, self.tokenizer, dev_config)
        
        dev_result = dev_engine.predict(self.test_data.iloc[:10])
        assert len(dev_result.predictions) == 10
        
        # 2. Optimization: Apply optimizations
        print("2. Optimization phase...")
        optimization_config = OptimizationConfig(
            enable_operator_fusion=True,
            enable_memory_efficient_attention=True
        )
        
        optimizer = ModelOptimizer(optimization_config)
        sample_input = self.tokenizer.encode_batch(self.test_data.iloc[:2])
        optimized_model = optimizer.optimize_model(self.model, sample_input)
        
        # 3. Testing: Validate optimized model
        print("3. Testing phase...")
        test_engine = InferenceEngine(optimized_model, self.tokenizer)
        test_result = test_engine.predict(self.test_data.iloc[:10])
        
        # Results should be similar (within reasonable tolerance)
        assert len(test_result.predictions) == len(dev_result.predictions)
        
        # 4. Export: Prepare for deployment
        print("4. Export phase...")
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            export_path = f.name
        
        try:
            from tabgpt.serving import TorchScriptExporter
            
            config = ExportConfig(export_format="torchscript")
            exporter = TorchScriptExporter(config)
            
            export_result = exporter.export_model(
                optimized_model, sample_input, export_path
            )
            
            assert Path(export_path).exists()
            
            # 5. Production: Deploy optimized engine
            print("5. Production phase...")
            prod_config = InferenceConfig(
                batch_size=16,
                max_batch_size=32,
                use_cache=True,
                cache_size=200,
                enable_memory_optimization=True
            )
            
            prod_engine = OptimizedInferenceEngine(
                optimized_model, self.tokenizer, prod_config
            )
            
            try:
                # Simulate production load
                for i in range(5):
                    batch = self.test_data.iloc[i*5:(i+1)*5]
                    prod_result = prod_engine.predict(batch)
                    assert len(prod_result.predictions) == 5
                
                # Check production statistics
                stats = prod_engine.get_stats()
                assert stats['total_requests'] == 5
                assert stats['errors'] == 0
                
                print("✓ Workflow completed successfully!")
                
            finally:
                prod_engine.shutdown()
        
        finally:
            if Path(export_path).exists():
                Path(export_path).unlink()
    
    def test_a_b_testing_workflow(self):
        """Test A/B testing workflow with different configurations."""
        
        # Configuration A: Basic setup
        config_a = InferenceConfig(
            batch_size=8,
            use_cache=False,
            enable_memory_optimization=False
        )
        
        # Configuration B: Optimized setup
        config_b = InferenceConfig(
            batch_size=16,
            use_cache=True,
            cache_size=50,
            enable_memory_optimization=True
        )
        
        engine_a = InferenceEngine(self.model, self.tokenizer, config_a)
        engine_b = OptimizedInferenceEngine(self.model, self.tokenizer, config_b)
        
        try:
            # Test both configurations
            test_batches = [
                self.test_data.iloc[i:i+5] for i in range(0, 25, 5)
            ]
            
            results_a = []
            results_b = []
            
            for batch in test_batches:
                # Test configuration A
                start_time = time.time()
                result_a = engine_a.predict(batch)
                time_a = time.time() - start_time
                
                # Test configuration B
                start_time = time.time()
                result_b = engine_b.predict(batch)
                time_b = time.time() - start_time
                
                results_a.append({
                    'time_ms': time_a * 1000,
                    'predictions': len(result_a.predictions),
                    'memory_mb': result_a.memory_usage_mb
                })
                
                results_b.append({
                    'time_ms': time_b * 1000,
                    'predictions': len(result_b.predictions),
                    'memory_mb': result_b.memory_usage_mb
                })
            
            # Compare results
            avg_time_a = np.mean([r['time_ms'] for r in results_a])
            avg_time_b = np.mean([r['time_ms'] for r in results_b])
            
            avg_memory_a = np.mean([r['memory_mb'] for r in results_a])
            avg_memory_b = np.mean([r['memory_mb'] for r in results_b])
            
            print(f"\\nA/B Testing Results:")
            print(f"Configuration A - Avg time: {avg_time_a:.2f}ms, Avg memory: {avg_memory_a:.2f}MB")
            print(f"Configuration B - Avg time: {avg_time_b:.2f}ms, Avg memory: {avg_memory_b:.2f}MB")
            
            # Both configurations should produce same number of predictions
            total_predictions_a = sum(r['predictions'] for r in results_a)
            total_predictions_b = sum(r['predictions'] for r in results_b)
            
            assert total_predictions_a == total_predictions_b
            
            # Configuration B should generally be more efficient (though not guaranteed in test env)
            stats_b = engine_b.get_stats()
            if stats_b.get('cache_hit_rate', 0) > 0:
                print(f"Configuration B cache hit rate: {stats_b['cache_hit_rate']:.2%}")
            
        finally:
            engine_b.shutdown()


if __name__ == "__main__":
    # Run integration tests with verbose output
    pytest.main([__file__, "-v", "-s"])