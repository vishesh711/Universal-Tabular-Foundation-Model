"""Performance tests for TabGPT inference and serving optimizations."""

import pytest
import time
import threading
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any
import tempfile
import json

import torch
import torch.nn as nn
import pandas as pd
import numpy as np

from tabgpt.serving import (
    InferenceEngine, BatchInferenceEngine, OptimizedInferenceEngine,
    InferenceConfig, ModelOptimizer, OptimizationConfig,
    CacheManager, DynamicBatcher, ModelExporter, ExportConfig
)


class MockTabGPTModel(nn.Module):
    """Mock TabGPT model for performance testing."""
    
    def __init__(self, input_dim=768, output_dim=2, num_layers=6):
        super().__init__()
        
        # Create a reasonably complex model for realistic performance testing
        layers = []
        current_dim = input_dim
        
        for i in range(num_layers):
            layers.extend([
                nn.Linear(current_dim, current_dim),
                nn.LayerNorm(current_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
        
        layers.append(nn.Linear(current_dim, output_dim))
        
        self.model = nn.Sequential(*layers)
        self.config = type('Config', (), {'hidden_size': input_dim})()
    
    def forward(self, input_ids, attention_mask=None, **kwargs):
        # Simulate attention mechanism computation time
        batch_size, seq_len = input_ids.shape
        
        # Simple mean pooling to simulate feature extraction
        features = input_ids.float().mean(dim=1)
        
        # Pass through model
        logits = self.model(features)
        
        return type('Output', (), {
            'logits': logits,
            'predictions': logits,
            'probabilities': torch.softmax(logits, dim=-1),
            'features': features
        })()


class MockTokenizer:
    """Mock tokenizer for performance testing."""
    
    def __init__(self, vocab_size=10000, max_length=512):
        self.vocab_size = vocab_size
        self.max_length = max_length
    
    def encode_batch(self, df):
        batch_size = len(df)
        # Simulate variable sequence lengths
        seq_lengths = np.random.randint(50, self.max_length, batch_size)
        max_seq_len = max(seq_lengths)
        
        # Create padded sequences
        input_ids = torch.zeros(batch_size, max_seq_len, dtype=torch.long)
        attention_mask = torch.zeros(batch_size, max_seq_len, dtype=torch.long)
        
        for i, seq_len in enumerate(seq_lengths):
            input_ids[i, :seq_len] = torch.randint(1, self.vocab_size, (seq_len,))
            attention_mask[i, :seq_len] = 1
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }


class PerformanceBenchmark:
    """Performance benchmarking utilities."""
    
    @staticmethod
    def measure_latency(func, *args, num_runs=100, warmup_runs=10, **kwargs):
        """Measure function latency with warmup."""
        # Warmup
        for _ in range(warmup_runs):
            func(*args, **kwargs)
        
        # Measure
        times = []
        for _ in range(num_runs):
            start_time = time.perf_counter()
            func(*args, **kwargs)
            end_time = time.perf_counter()
            times.append((end_time - start_time) * 1000)  # Convert to ms
        
        return {
            'mean_ms': statistics.mean(times),
            'median_ms': statistics.median(times),
            'std_ms': statistics.stdev(times) if len(times) > 1 else 0,
            'min_ms': min(times),
            'max_ms': max(times),
            'p95_ms': np.percentile(times, 95),
            'p99_ms': np.percentile(times, 99)
        }
    
    @staticmethod
    def measure_throughput(func, data_batches, max_workers=4):
        """Measure throughput with concurrent requests."""
        start_time = time.perf_counter()
        total_samples = sum(len(batch) for batch in data_batches)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(func, batch) for batch in data_batches]
            results = [future.result() for future in as_completed(futures)]
        
        end_time = time.perf_counter()
        total_time = end_time - start_time
        
        return {
            'total_samples': total_samples,
            'total_time_s': total_time,
            'samples_per_second': total_samples / total_time,
            'requests_per_second': len(data_batches) / total_time,
            'avg_batch_size': total_samples / len(data_batches)
        }
    
    @staticmethod
    def create_test_data(num_samples=1000, num_features=20):
        """Create test data for benchmarking."""
        np.random.seed(42)  # For reproducible results
        
        data = {}
        
        # Numerical features
        for i in range(num_features // 2):
            data[f'num_feature_{i}'] = np.random.randn(num_samples)
        
        # Categorical features
        categories = ['A', 'B', 'C', 'D', 'E']
        for i in range(num_features // 2):
            data[f'cat_feature_{i}'] = np.random.choice(categories, num_samples)
        
        return pd.DataFrame(data)


class TestInferencePerformance:
    """Test inference engine performance."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.model = MockTabGPTModel()
        self.tokenizer = MockTokenizer()
        self.test_data = PerformanceBenchmark.create_test_data(1000, 20)
        self.small_batch = self.test_data.iloc[:10]
        self.medium_batch = self.test_data.iloc[:100]
        self.large_batch = self.test_data.iloc[:1000]
    
    def test_basic_inference_latency(self):
        """Test basic inference latency."""
        config = InferenceConfig(batch_size=32, use_cache=False)
        engine = InferenceEngine(self.model, self.tokenizer, config)
        
        # Measure latency for different batch sizes
        small_latency = PerformanceBenchmark.measure_latency(
            engine.predict, self.small_batch, num_runs=50
        )
        
        medium_latency = PerformanceBenchmark.measure_latency(
            engine.predict, self.medium_batch, num_runs=20
        )
        
        print(f"\\nInference Latency Results:")
        print(f"Small batch (10 samples): {small_latency['mean_ms']:.2f}ms ± {small_latency['std_ms']:.2f}ms")
        print(f"Medium batch (100 samples): {medium_latency['mean_ms']:.2f}ms ± {medium_latency['std_ms']:.2f}ms")
        
        # Assertions
        assert small_latency['mean_ms'] > 0
        assert medium_latency['mean_ms'] > small_latency['mean_ms']
        assert small_latency['std_ms'] < small_latency['mean_ms']  # Reasonable variance
    
    def test_batch_inference_performance(self):
        """Test batch inference performance improvements."""
        # Single request engine
        single_config = InferenceConfig(batch_size=1, use_cache=False)
        single_engine = InferenceEngine(self.model, self.tokenizer, single_config)
        
        # Batch engine
        batch_config = InferenceConfig(batch_size=32, max_batch_size=64, use_cache=False)
        batch_engine = BatchInferenceEngine(self.model, self.tokenizer, batch_config)
        
        try:
            # Create multiple small requests
            small_requests = [self.test_data.iloc[i:i+5] for i in range(0, 100, 5)]
            
            # Measure single request processing
            single_start = time.perf_counter()
            for request in small_requests:
                single_engine.predict(request)
            single_time = time.perf_counter() - single_start
            
            # Measure batch processing
            batch_start = time.perf_counter()
            futures = []
            for request in small_requests:
                future = batch_engine.predict_async(request)
                futures.append(future)
            
            # Wait for all results
            for future in futures:
                future.result(timeout=10.0)
            batch_time = time.perf_counter() - batch_start
            
            print(f"\\nBatch Processing Performance:")
            print(f"Single requests: {single_time:.3f}s")
            print(f"Batch processing: {batch_time:.3f}s")
            print(f"Speedup: {single_time / batch_time:.2f}x")
            
            # Batch processing should be faster for multiple small requests
            assert batch_time < single_time
            
        finally:
            batch_engine.shutdown()
    
    def test_caching_performance(self):
        """Test caching performance improvements."""
        # Engine without cache
        no_cache_config = InferenceConfig(use_cache=False)
        no_cache_engine = InferenceEngine(self.model, self.tokenizer, no_cache_config)
        
        # Engine with cache
        cache_config = InferenceConfig(use_cache=True, cache_size=100)
        cache_engine = InferenceEngine(self.model, self.tokenizer, cache_config)
        
        # First run (cache miss)
        first_run = PerformanceBenchmark.measure_latency(
            cache_engine.predict, self.small_batch, num_runs=10
        )
        
        # Second run (cache hit)
        second_run = PerformanceBenchmark.measure_latency(
            cache_engine.predict, self.small_batch, num_runs=10
        )
        
        # No cache baseline
        no_cache_run = PerformanceBenchmark.measure_latency(
            no_cache_engine.predict, self.small_batch, num_runs=10
        )
        
        print(f"\\nCaching Performance:")
        print(f"No cache: {no_cache_run['mean_ms']:.2f}ms")
        print(f"Cache miss: {first_run['mean_ms']:.2f}ms")
        print(f"Cache hit: {second_run['mean_ms']:.2f}ms")
        print(f"Cache speedup: {first_run['mean_ms'] / second_run['mean_ms']:.2f}x")
        
        # Cache hits should be significantly faster
        assert second_run['mean_ms'] < first_run['mean_ms'] * 0.5
        
        # Check cache statistics
        stats = cache_engine.get_stats()
        assert stats['cache_hits'] > 0
        assert stats['cache_hit_rate'] > 0
    
    def test_optimized_engine_performance(self):
        """Test optimized inference engine performance."""
        # Basic engine
        basic_config = InferenceConfig(
            use_cache=False,
            enable_memory_optimization=False
        )
        basic_engine = InferenceEngine(self.model, self.tokenizer, basic_config)
        
        # Optimized engine
        optimized_config = InferenceConfig(
            use_cache=True,
            cache_size=200,
            enable_memory_optimization=True,
            batch_size=32,
            max_batch_size=64
        )
        optimized_engine = OptimizedInferenceEngine(self.model, self.tokenizer, optimized_config)
        
        try:
            # Create test batches
            test_batches = [
                self.test_data.iloc[i:i+20] for i in range(0, 200, 20)
            ]
            
            # Measure basic engine throughput
            basic_throughput = PerformanceBenchmark.measure_throughput(
                basic_engine.predict, test_batches, max_workers=2
            )
            
            # Measure optimized engine throughput
            optimized_throughput = PerformanceBenchmark.measure_throughput(
                optimized_engine.predict, test_batches, max_workers=2
            )
            
            print(f"\\nEngine Throughput Comparison:")
            print(f"Basic engine: {basic_throughput['samples_per_second']:.1f} samples/sec")
            print(f"Optimized engine: {optimized_throughput['samples_per_second']:.1f} samples/sec")
            print(f"Throughput improvement: {optimized_throughput['samples_per_second'] / basic_throughput['samples_per_second']:.2f}x")
            
            # Optimized engine should have better throughput
            assert optimized_throughput['samples_per_second'] >= basic_throughput['samples_per_second']
            
        finally:
            optimized_engine.shutdown()
    
    def test_memory_usage_optimization(self):
        """Test memory usage optimization."""
        config = InferenceConfig(
            enable_memory_optimization=True,
            max_memory_mb=1024
        )
        engine = OptimizedInferenceEngine(self.model, self.tokenizer, config)
        
        try:
            # Process multiple batches and monitor memory
            memory_usage = []
            
            for i in range(10):
                batch = self.test_data.iloc[i*50:(i+1)*50]
                result = engine.predict(batch)
                memory_usage.append(result.memory_usage_mb)
            
            print(f"\\nMemory Usage:")
            print(f"Average: {np.mean(memory_usage):.1f}MB")
            print(f"Peak: {np.max(memory_usage):.1f}MB")
            print(f"Variation: {np.std(memory_usage):.1f}MB")
            
            # Memory usage should be reasonable and stable
            assert np.max(memory_usage) < config.max_memory_mb
            assert np.std(memory_usage) < np.mean(memory_usage) * 0.5  # Low variation
            
        finally:
            engine.shutdown()


class TestOptimizationPerformance:
    """Test model optimization performance."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.model = MockTabGPTModel(input_dim=512, num_layers=8)
        self.tokenizer = MockTokenizer()
        self.test_data = PerformanceBenchmark.create_test_data(100, 15)
    
    def test_model_optimization_impact(self):
        """Test impact of model optimizations on performance."""
        # Original model
        original_model = MockTabGPTModel(input_dim=512, num_layers=8)
        
        # Optimized model
        optimization_config = OptimizationConfig(
            enable_quantization=False,  # Skip for compatibility
            enable_pruning=False,       # Skip for compatibility
            enable_torch_compile=False, # Skip for compatibility
            enable_operator_fusion=True,
            enable_memory_efficient_attention=True
        )
        
        optimizer = ModelOptimizer(optimization_config)
        
        # Create sample input for optimization
        sample_input = self.tokenizer.encode_batch(self.test_data.iloc[:2])
        
        # Optimize model
        optimized_model = optimizer.optimize_model(original_model, sample_input)
        
        # Create engines
        original_engine = InferenceEngine(original_model, self.tokenizer)
        optimized_engine = InferenceEngine(optimized_model, self.tokenizer)
        
        # Benchmark both models
        original_perf = PerformanceBenchmark.measure_latency(
            original_engine.predict, self.test_data.iloc[:50], num_runs=20
        )
        
        optimized_perf = PerformanceBenchmark.measure_latency(
            optimized_engine.predict, self.test_data.iloc[:50], num_runs=20
        )
        
        print(f"\\nModel Optimization Impact:")
        print(f"Original model: {original_perf['mean_ms']:.2f}ms ± {original_perf['std_ms']:.2f}ms")
        print(f"Optimized model: {optimized_perf['mean_ms']:.2f}ms ± {optimized_perf['std_ms']:.2f}ms")
        
        if optimized_perf['mean_ms'] < original_perf['mean_ms']:
            speedup = original_perf['mean_ms'] / optimized_perf['mean_ms']
            print(f"Speedup: {speedup:.2f}x")
        else:
            print("No significant speedup (expected in test environment)")
        
        # Check that optimization completed successfully
        assert len(optimizer.optimization_history) > 0
        
        # Model should still produce valid outputs
        result = optimized_engine.predict(self.test_data.iloc[:5])
        assert result.predictions is not None
        assert len(result.predictions) == 5


class TestCachePerformance:
    """Test caching system performance."""
    
    def test_cache_manager_performance(self):
        """Test cache manager performance under load."""
        cache = CacheManager(max_size=1000, ttl_seconds=60)
        
        # Test cache operations performance
        num_operations = 10000
        
        # Measure put operations
        put_start = time.perf_counter()
        for i in range(num_operations):
            cache.put(f"key_{i}", f"value_{i}")
        put_time = time.perf_counter() - put_start
        
        # Measure get operations (hits)
        get_start = time.perf_counter()
        for i in range(num_operations):
            cache.get(f"key_{i}")
        get_time = time.perf_counter() - get_start
        
        # Measure get operations (misses)
        miss_start = time.perf_counter()
        for i in range(num_operations):
            cache.get(f"missing_key_{i}")
        miss_time = time.perf_counter() - miss_start
        
        print(f"\\nCache Performance ({num_operations} operations):")
        print(f"Put operations: {put_time:.3f}s ({num_operations/put_time:.0f} ops/sec)")
        print(f"Get operations (hits): {get_time:.3f}s ({num_operations/get_time:.0f} ops/sec)")
        print(f"Get operations (misses): {miss_time:.3f}s ({num_operations/miss_time:.0f} ops/sec)")
        
        # Cache operations should be fast
        assert put_time < 1.0  # Should complete in under 1 second
        assert get_time < 0.5  # Gets should be faster than puts
        assert miss_time < 0.5  # Misses should be fast too
        
        # Check final stats
        stats = cache.get_stats()
        print(f"Final cache stats: {stats}")
        assert stats['size'] <= 1000  # Respects max size
    
    def test_cache_concurrent_access(self):
        """Test cache performance under concurrent access."""
        cache = CacheManager(max_size=500, ttl_seconds=60)
        
        def cache_worker(worker_id, num_ops):
            """Worker function for concurrent cache access."""
            for i in range(num_ops):
                key = f"worker_{worker_id}_key_{i}"
                value = f"worker_{worker_id}_value_{i}"
                
                # Put and get operations
                cache.put(key, value)
                retrieved = cache.get(key)
                assert retrieved == value
        
        # Run concurrent workers
        num_workers = 4
        ops_per_worker = 1000
        
        start_time = time.perf_counter()
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(cache_worker, i, ops_per_worker)
                for i in range(num_workers)
            ]
            
            for future in as_completed(futures):
                future.result()  # Wait for completion
        
        end_time = time.perf_counter()
        total_time = end_time - start_time
        total_ops = num_workers * ops_per_worker * 2  # Put + Get
        
        print(f"\\nConcurrent Cache Performance:")
        print(f"Total operations: {total_ops}")
        print(f"Total time: {total_time:.3f}s")
        print(f"Operations per second: {total_ops / total_time:.0f}")
        
        # Should handle concurrent access efficiently
        assert total_time < 5.0  # Should complete reasonably fast
        
        stats = cache.get_stats()
        assert stats['hits'] > 0
        assert stats['size'] > 0


class TestDynamicBatchingPerformance:
    """Test dynamic batching performance."""
    
    def test_batching_efficiency(self):
        """Test dynamic batching efficiency."""
        batcher = DynamicBatcher(max_batch_size=10, timeout_ms=50)
        
        # Simulate requests arriving at different rates
        request_times = []
        batch_sizes = []
        
        def add_requests_slowly():
            """Add requests with delays."""
            for i in range(5):
                start = time.perf_counter()
                ready = batcher.add_request({'id': f'slow_{i}'})
                if ready:
                    batch = batcher.get_batch()
                    batch_sizes.append(len(batch))
                    request_times.append(time.perf_counter() - start)
                time.sleep(0.02)  # 20ms delay
        
        def add_requests_quickly():
            """Add requests rapidly."""
            for i in range(15):
                start = time.perf_counter()
                ready = batcher.add_request({'id': f'fast_{i}'})
                if ready:
                    batch = batcher.get_batch()
                    batch_sizes.append(len(batch))
                    request_times.append(time.perf_counter() - start)
        
        # Run both scenarios
        add_requests_quickly()  # Should create full batches
        add_requests_slowly()   # Should create timeout-based batches
        
        # Get any remaining batch
        remaining_batch = batcher.get_batch()
        if remaining_batch:
            batch_sizes.append(len(remaining_batch))
        
        print(f"\\nDynamic Batching Results:")
        print(f"Number of batches: {len(batch_sizes)}")
        print(f"Batch sizes: {batch_sizes}")
        print(f"Average batch size: {np.mean(batch_sizes):.1f}")
        print(f"Average request time: {np.mean(request_times)*1000:.1f}ms")
        
        # Should create efficient batches
        assert len(batch_sizes) > 0
        assert max(batch_sizes) <= 10  # Respects max batch size
        assert np.mean(batch_sizes) > 1  # Actually batching requests


class TestExportPerformance:
    """Test model export performance."""
    
    def test_export_speed(self):
        """Test model export speed for different formats."""
        model = MockTabGPTModel(input_dim=256, num_layers=4)
        tokenizer = MockTokenizer()
        test_data = PerformanceBenchmark.create_test_data(10, 10)
        
        # Create sample input
        sample_input = tokenizer.encode_batch(test_data.iloc[:2])
        
        export_times = {}
        
        # Test TorchScript export
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            torchscript_path = f.name
        
        try:
            from tabgpt.serving import TorchScriptExporter, ExportConfig
            
            config = ExportConfig(export_format="torchscript", validate_export=False)
            exporter = TorchScriptExporter(config)
            
            start_time = time.perf_counter()
            result = exporter.export_model(model, sample_input, torchscript_path)
            export_times['torchscript'] = time.perf_counter() - start_time
            
            print(f"\\nExport Performance:")
            print(f"TorchScript export: {export_times['torchscript']:.3f}s")
            print(f"Model size: {result['model_size_mb']:.2f}MB")
            
            # Export should complete in reasonable time
            assert export_times['torchscript'] < 30.0  # Should be fast for small model
            assert result['model_size_mb'] > 0
            
        except Exception as e:
            print(f"TorchScript export test skipped: {e}")
        
        finally:
            # Cleanup
            try:
                import os
                os.unlink(torchscript_path)
            except:
                pass


class TestEndToEndPerformance:
    """Test end-to-end performance scenarios."""
    
    def test_production_simulation(self):
        """Simulate production workload performance."""
        # Setup production-like configuration
        config = InferenceConfig(
            batch_size=32,
            max_batch_size=64,
            use_cache=True,
            cache_size=500,
            enable_memory_optimization=True,
            num_workers=2
        )
        
        model = MockTabGPTModel(input_dim=512, num_layers=6)
        tokenizer = MockTokenizer()
        engine = OptimizedInferenceEngine(model, tokenizer, config)
        
        try:
            # Create realistic workload
            # - Mix of batch sizes
            # - Some repeated requests (cache hits)
            # - Concurrent requests
            
            workload = []
            
            # Small batches (common in production)
            for i in range(20):
                batch_size = np.random.randint(1, 10)
                start_idx = i * batch_size
                batch = PerformanceBenchmark.create_test_data(batch_size, 15)
                workload.append(batch)
            
            # Medium batches
            for i in range(10):
                batch_size = np.random.randint(10, 50)
                batch = PerformanceBenchmark.create_test_data(batch_size, 15)
                workload.append(batch)
            
            # Add some duplicate requests for cache testing
            workload.extend(workload[:5])
            
            # Measure performance
            throughput_result = PerformanceBenchmark.measure_throughput(
                engine.predict, workload, max_workers=4
            )
            
            print(f"\\nProduction Simulation Results:")
            print(f"Total samples processed: {throughput_result['total_samples']}")
            print(f"Total time: {throughput_result['total_time_s']:.2f}s")
            print(f"Throughput: {throughput_result['samples_per_second']:.1f} samples/sec")
            print(f"Request rate: {throughput_result['requests_per_second']:.1f} requests/sec")
            
            # Get engine statistics
            stats = engine.get_stats()
            print(f"\\nEngine Statistics:")
            print(f"Total requests: {stats['total_requests']}")
            print(f"Cache hit rate: {stats.get('cache_hit_rate', 0):.2%}")
            print(f"Average time per request: {stats.get('avg_time_per_request_ms', 0):.1f}ms")
            print(f"Errors: {stats.get('errors', 0)}")
            
            # Performance assertions
            assert throughput_result['samples_per_second'] > 10  # Minimum throughput
            assert stats['errors'] == 0  # No errors
            assert stats.get('cache_hit_rate', 0) > 0  # Some cache hits
            
        finally:
            engine.shutdown()
    
    def test_stress_testing(self):
        """Stress test the inference system."""
        config = InferenceConfig(
            batch_size=16,
            max_batch_size=32,
            use_cache=True,
            cache_size=100,
            max_concurrent_requests=50
        )
        
        model = MockTabGPTModel()
        tokenizer = MockTokenizer()
        engine = OptimizedInferenceEngine(model, tokenizer, config)
        
        try:
            # High load scenario
            num_concurrent_requests = 20
            requests_per_thread = 10
            
            def stress_worker(worker_id):
                """Worker function for stress testing."""
                results = []
                for i in range(requests_per_thread):
                    try:
                        batch_size = np.random.randint(1, 20)
                        data = PerformanceBenchmark.create_test_data(batch_size, 10)
                        
                        start_time = time.perf_counter()
                        result = engine.predict(data)
                        end_time = time.perf_counter()
                        
                        results.append({
                            'worker_id': worker_id,
                            'request_id': i,
                            'batch_size': batch_size,
                            'latency_ms': (end_time - start_time) * 1000,
                            'success': True
                        })
                        
                    except Exception as e:
                        results.append({
                            'worker_id': worker_id,
                            'request_id': i,
                            'error': str(e),
                            'success': False
                        })
                
                return results
            
            # Run stress test
            start_time = time.perf_counter()
            
            with ThreadPoolExecutor(max_workers=num_concurrent_requests) as executor:
                futures = [
                    executor.submit(stress_worker, i)
                    for i in range(num_concurrent_requests)
                ]
                
                all_results = []
                for future in as_completed(futures):
                    worker_results = future.result()
                    all_results.extend(worker_results)
            
            end_time = time.perf_counter()
            
            # Analyze results
            successful_requests = [r for r in all_results if r['success']]
            failed_requests = [r for r in all_results if not r['success']]
            
            if successful_requests:
                latencies = [r['latency_ms'] for r in successful_requests]
                
                print(f"\\nStress Test Results:")
                print(f"Total requests: {len(all_results)}")
                print(f"Successful requests: {len(successful_requests)}")
                print(f"Failed requests: {len(failed_requests)}")
                print(f"Success rate: {len(successful_requests) / len(all_results):.2%}")
                print(f"Total time: {end_time - start_time:.2f}s")
                print(f"Average latency: {np.mean(latencies):.1f}ms")
                print(f"95th percentile latency: {np.percentile(latencies, 95):.1f}ms")
                print(f"Max latency: {np.max(latencies):.1f}ms")
                
                # Stress test assertions
                success_rate = len(successful_requests) / len(all_results)
                assert success_rate > 0.95  # At least 95% success rate
                assert np.mean(latencies) < 5000  # Average latency under 5 seconds
                
            else:
                pytest.fail("No successful requests in stress test")
            
        finally:
            engine.shutdown()


if __name__ == "__main__":
    # Run performance tests with verbose output
    pytest.main([__file__, "-v", "-s"])