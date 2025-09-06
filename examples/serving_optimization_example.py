#!/usr/bin/env python3
"""
Comprehensive example demonstrating TabGPT serving and inference optimizations.

This example showcases:
1. Model optimization techniques (quantization, pruning, compilation)
2. Efficient inference engines with caching and batching
3. Model export for production deployment
4. Performance benchmarking and analysis
5. Production-ready serving setup
"""

import time
import tempfile
from pathlib import Path
import json

import torch
import torch.nn as nn
import pandas as pd
import numpy as np

# TabGPT imports
from tabgpt import TabGPTForSequenceClassification, TabGPTTokenizer
from tabgpt.serving import (
    InferenceEngine, BatchInferenceEngine, OptimizedInferenceEngine,
    InferenceConfig, ModelOptimizer, OptimizationConfig, QuantizationConfig,
    CacheManager, DynamicBatcher, ModelExporter, ExportConfig,
    ModelServer, ServerConfig
)
from tabgpt.utils import RobustNormalizer


class ServingOptimizationDemo:
    """Comprehensive demonstration of serving and optimization features."""
    
    def __init__(self):
        self.results = {}
        
    def create_sample_model_and_data(self):
        """Create sample model and data for demonstration."""
        print("Creating sample model and data...")
        
        # Create a sample TabGPT model (mock for demonstration)
        class MockTabGPTModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.transformer = nn.Sequential(
                    nn.Linear(768, 768),
                    nn.LayerNorm(768),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(768, 768),
                    nn.LayerNorm(768),
                    nn.ReLU(),
                    nn.Linear(768, 2)  # Binary classification
                )
                self.config = type('Config', (), {'hidden_size': 768})()
            
            def forward(self, input_ids, attention_mask=None, **kwargs):
                # Simulate feature extraction
                batch_size = input_ids.shape[0]
                features = input_ids.float().mean(dim=1)  # Simple pooling
                logits = self.transformer(features)
                
                return type('Output', (), {
                    'logits': logits,
                    'predictions': logits,
                    'probabilities': torch.softmax(logits, dim=-1)
                })()
        
        # Create mock tokenizer
        class MockTokenizer:
            def encode_batch(self, df):
                batch_size = len(df)
                return {
                    'input_ids': torch.randn(batch_size, 768),  # Pre-computed features
                    'attention_mask': torch.ones(batch_size, 768)
                }
        
        self.model = MockTabGPTModel()
        self.tokenizer = MockTokenizer()
        
        # Create sample data
        np.random.seed(42)
        self.sample_data = pd.DataFrame({
            'feature_1': np.random.randn(1000),
            'feature_2': np.random.choice(['A', 'B', 'C'], 1000),
            'feature_3': np.random.uniform(0, 100, 1000),
            'feature_4': np.random.choice(['X', 'Y'], 1000),
            'target': np.random.choice([0, 1], 1000)
        })
        
        print(f"Created model with {sum(p.numel() for p in self.model.parameters()):,} parameters")
        print(f"Created dataset with {len(self.sample_data)} samples")
        
    def demonstrate_model_optimization(self):
        """Demonstrate model optimization techniques."""
        print("\\n" + "="*60)
        print("MODEL OPTIMIZATION DEMONSTRATION")
        print("="*60)
        
        # Original model performance
        print("\\n1. Measuring original model performance...")
        original_engine = InferenceEngine(self.model, self.tokenizer)
        
        test_batch = self.sample_data.iloc[:50]
        
        # Measure original performance
        times = []
        for _ in range(10):
            start = time.time()
            original_engine.predict(test_batch)
            times.append(time.time() - start)
        
        original_time = np.mean(times) * 1000
        print(f"Original model average time: {original_time:.2f}ms")
        
        # Model optimization
        print("\\n2. Applying model optimizations...")
        
        optimization_config = OptimizationConfig(
            enable_quantization=False,  # Skip for compatibility in demo
            enable_pruning=False,       # Skip for compatibility in demo
            enable_torch_compile=False, # Skip for compatibility in demo
            enable_operator_fusion=True,
            enable_memory_efficient_attention=True
        )
        
        optimizer = ModelOptimizer(optimization_config)
        
        # Create sample input for optimization
        sample_input = self.tokenizer.encode_batch(test_batch)
        
        # Optimize model
        optimized_model = optimizer.optimize_model(self.model, sample_input)
        
        # Measure optimized performance
        optimized_engine = InferenceEngine(optimized_model, self.tokenizer)
        
        times = []
        for _ in range(10):
            start = time.time()
            optimized_engine.predict(test_batch)
            times.append(time.time() - start)
        
        optimized_time = np.mean(times) * 1000
        print(f"Optimized model average time: {optimized_time:.2f}ms")
        
        if optimized_time < original_time:
            speedup = original_time / optimized_time
            print(f"Optimization speedup: {speedup:.2f}x")
        else:
            print("No significant speedup (expected in demo environment)")
        
        self.results['optimization'] = {
            'original_time_ms': original_time,
            'optimized_time_ms': optimized_time,
            'speedup': original_time / optimized_time if optimized_time > 0 else 1.0
        }
        
        return optimized_model
    
    def demonstrate_inference_engines(self):
        """Demonstrate different inference engine capabilities."""
        print("\\n" + "="*60)
        print("INFERENCE ENGINES DEMONSTRATION")
        print("="*60)
        
        test_data = self.sample_data.iloc[:100]
        
        # 1. Basic Inference Engine
        print("\\n1. Basic Inference Engine")
        basic_config = InferenceConfig(use_cache=False, batch_size=32)
        basic_engine = InferenceEngine(self.model, self.tokenizer, basic_config)
        
        start_time = time.time()
        basic_result = basic_engine.predict(test_data)
        basic_time = time.time() - start_time
        
        print(f"   Time: {basic_time*1000:.2f}ms")
        print(f"   Predictions shape: {basic_result.predictions.shape}")
        
        # 2. Batch Inference Engine
        print("\\n2. Batch Inference Engine")
        batch_config = InferenceConfig(
            batch_size=32, 
            max_batch_size=64, 
            batch_timeout_ms=100,
            use_cache=True
        )
        batch_engine = BatchInferenceEngine(self.model, self.tokenizer, batch_config)
        
        try:
            # Test async prediction
            futures = []
            start_time = time.time()
            
            # Submit multiple small batches
            for i in range(0, len(test_data), 10):
                batch = test_data.iloc[i:i+10]
                future = batch_engine.predict_async(batch)
                futures.append(future)
            
            # Collect results
            batch_results = []
            for future in futures:
                result = future.result(timeout=5.0)
                batch_results.append(result)
            
            batch_time = time.time() - start_time
            
            print(f"   Time: {batch_time*1000:.2f}ms")
            print(f"   Number of batches: {len(batch_results)}")
            print(f"   Total predictions: {sum(len(r.predictions) for r in batch_results)}")
            
        finally:
            batch_engine.shutdown()
        
        # 3. Optimized Inference Engine
        print("\\n3. Optimized Inference Engine")
        optimized_config = InferenceConfig(
            batch_size=32,
            max_batch_size=64,
            use_cache=True,
            cache_size=200,
            enable_memory_optimization=True,
            num_workers=2
        )
        optimized_engine = OptimizedInferenceEngine(self.model, self.tokenizer, optimized_config)
        
        try:
            start_time = time.time()
            optimized_result = optimized_engine.predict(test_data)
            optimized_time = time.time() - start_time
            
            print(f"   Time: {optimized_time*1000:.2f}ms")
            print(f"   Memory usage: {optimized_result.memory_usage_mb:.2f}MB")
            
            # Test caching
            start_time = time.time()
            cached_result = optimized_engine.predict(test_data)  # Same data, should hit cache
            cached_time = time.time() - start_time
            
            print(f"   Cached prediction time: {cached_time*1000:.2f}ms")
            
            # Get engine statistics
            stats = optimized_engine.get_stats()
            print(f"   Cache hit rate: {stats.get('cache_hit_rate', 0):.2%}")
            print(f"   Total requests: {stats.get('total_requests', 0)}")
            
        finally:
            optimized_engine.shutdown()
        
        self.results['inference_engines'] = {
            'basic_time_ms': basic_time * 1000,
            'batch_time_ms': batch_time * 1000,
            'optimized_time_ms': optimized_time * 1000,
            'cached_time_ms': cached_time * 1000,
            'cache_speedup': optimized_time / cached_time if cached_time > 0 else 1.0
        }
    
    def demonstrate_caching_and_batching(self):
        """Demonstrate caching and dynamic batching features."""
        print("\\n" + "="*60)
        print("CACHING AND BATCHING DEMONSTRATION")
        print("="*60)
        
        # 1. Cache Manager
        print("\\n1. Cache Manager Performance")
        cache = CacheManager(max_size=100, ttl_seconds=60)
        
        # Test cache operations
        num_operations = 1000
        
        # Put operations
        start_time = time.time()
        for i in range(num_operations):
            cache.put(f"key_{i}", f"value_{i}")
        put_time = time.time() - start_time
        
        # Get operations (hits)
        start_time = time.time()
        for i in range(num_operations):
            cache.get(f"key_{i}")
        get_time = time.time() - start_time
        
        print(f"   Put operations: {num_operations/put_time:.0f} ops/sec")
        print(f"   Get operations: {num_operations/get_time:.0f} ops/sec")
        
        stats = cache.get_stats()
        print(f"   Cache stats: {stats}")
        
        # 2. Dynamic Batcher
        print("\\n2. Dynamic Batching")
        batcher = DynamicBatcher(max_batch_size=5, timeout_ms=100)
        
        # Simulate requests
        batch_sizes = []
        
        # Add requests quickly (should create full batches)
        for i in range(12):
            ready = batcher.add_request({'id': i, 'data': f'request_{i}'})
            if ready:
                batch = batcher.get_batch()
                batch_sizes.append(len(batch))
                print(f"   Created batch of size: {len(batch)}")
        
        # Get any remaining batch
        remaining = batcher.get_batch()
        if remaining:
            batch_sizes.append(len(remaining))
            print(f"   Final batch of size: {len(remaining)}")
        
        print(f"   Average batch size: {np.mean(batch_sizes):.1f}")
        
        self.results['caching_batching'] = {
            'cache_put_ops_per_sec': num_operations / put_time,
            'cache_get_ops_per_sec': num_operations / get_time,
            'avg_batch_size': np.mean(batch_sizes),
            'num_batches': len(batch_sizes)
        }
    
    def demonstrate_model_export(self):
        """Demonstrate model export for production deployment."""
        print("\\n" + "="*60)
        print("MODEL EXPORT DEMONSTRATION")
        print("="*60)
        
        # Create sample input for export
        sample_input = self.tokenizer.encode_batch(self.sample_data.iloc[:2])
        
        export_results = {}
        
        # 1. TorchScript Export
        print("\\n1. TorchScript Export")
        try:
            with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
                torchscript_path = f.name
            
            from tabgpt.serving import TorchScriptExporter, ExportConfig
            
            config = ExportConfig(export_format="torchscript", validate_export=True)
            exporter = TorchScriptExporter(config)
            
            start_time = time.time()
            result = exporter.export_model(self.model, sample_input, torchscript_path)
            export_time = time.time() - start_time
            
            print(f"   Export time: {export_time:.2f}s")
            print(f"   Model size: {result['model_size_mb']:.2f}MB")
            print(f"   Validation passed: {result.get('validation_results', {}).get('validation_passed', 'N/A')}")
            
            export_results['torchscript'] = result
            
            # Test loading exported model
            try:
                loaded_model = torch.jit.load(torchscript_path)
                print("   ✓ Model loaded successfully")
                
                # Test inference with loaded model
                with torch.no_grad():
                    output = loaded_model(*sample_input.values())
                print("   ✓ Inference test passed")
                
            except Exception as e:
                print(f"   ✗ Loading/inference test failed: {e}")
            
            # Cleanup
            Path(torchscript_path).unlink()
            
        except Exception as e:
            print(f"   TorchScript export failed: {e}")
        
        # 2. ONNX Export (if available)
        print("\\n2. ONNX Export")
        try:
            with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
                onnx_path = f.name
            
            from tabgpt.serving import ONNXExporter
            
            config = ExportConfig(export_format="onnx", validate_export=False)  # Skip validation for demo
            exporter = ONNXExporter(config)
            
            start_time = time.time()
            result = exporter.export_model(self.model, sample_input, onnx_path)
            export_time = time.time() - start_time
            
            print(f"   Export time: {export_time:.2f}s")
            print(f"   Model size: {result['model_size_mb']:.2f}MB")
            
            export_results['onnx'] = result
            
            # Cleanup
            Path(onnx_path).unlink()
            
        except Exception as e:
            print(f"   ONNX export failed (expected in demo): {e}")
        
        self.results['model_export'] = export_results
    
    def demonstrate_production_serving(self):
        """Demonstrate production-ready serving setup."""
        print("\\n" + "="*60)
        print("PRODUCTION SERVING DEMONSTRATION")
        print("="*60)
        
        # Create server configuration
        server_config = ServerConfig(
            host="localhost",
            port=8000,
            workers=2,
            max_concurrent_requests=50,
            request_timeout_seconds=30,
            enable_metrics=True,
            log_level="INFO"
        )
        
        print("\\n1. Server Configuration")
        print(f"   Host: {server_config.host}")
        print(f"   Port: {server_config.port}")
        print(f"   Workers: {server_config.workers}")
        print(f"   Max concurrent requests: {server_config.max_concurrent_requests}")
        print(f"   Request timeout: {server_config.request_timeout_seconds}s")
        
        # Note: In a real scenario, you would start the server with:
        # server = ModelServer(server_config)
        # server.start()
        
        print("\\n2. Simulated Server Operations")
        
        # Simulate server request handling
        from tabgpt.serving.deployment import MetricsCollector, HealthChecker
        
        # Metrics collection
        metrics = MetricsCollector()
        
        # Simulate requests
        for i in range(10):
            metrics.record_request_start()
            
            # Simulate processing time
            processing_time = np.random.uniform(50, 200)  # 50-200ms
            success = np.random.random() > 0.1  # 90% success rate
            
            metrics.record_request_end(
                success=success,
                response_time_ms=processing_time,
                inference_time_ms=processing_time * 0.8
            )
        
        server_metrics = metrics.get_metrics()
        
        print(f"   Total requests: {server_metrics['requests_total']}")
        print(f"   Success rate: {server_metrics['success_rate']:.2%}")
        print(f"   Average response time: {server_metrics['avg_response_time_ms']:.1f}ms")
        print(f"   Average inference time: {server_metrics['avg_inference_time_ms']:.1f}ms")
        
        # Health checking
        print("\\n3. Health Monitoring")
        
        # Create a simple inference engine for health checking
        engine = InferenceEngine(self.model, self.tokenizer)
        health_checker = HealthChecker(engine, server_config)
        
        # Wait a moment for health check
        time.sleep(0.5)
        
        health_status = health_checker.get_health_status()
        print(f"   Health status: {'✓ Healthy' if health_status['healthy'] else '✗ Unhealthy'}")
        print(f"   Uptime: {health_status['uptime_seconds']:.1f}s")
        print(f"   Total requests: {health_status['total_requests']}")
        
        self.results['production_serving'] = {
            'server_config': server_config.__dict__,
            'metrics': server_metrics,
            'health_status': health_status
        }
    
    def demonstrate_performance_analysis(self):
        """Demonstrate performance analysis and benchmarking."""
        print("\\n" + "="*60)
        print("PERFORMANCE ANALYSIS DEMONSTRATION")
        print("="*60)
        
        # Create different engine configurations for comparison
        configs = {
            'Basic': InferenceConfig(use_cache=False, batch_size=16),
            'Cached': InferenceConfig(use_cache=True, cache_size=100, batch_size=16),
            'Optimized': InferenceConfig(
                use_cache=True, 
                cache_size=200, 
                batch_size=32,
                enable_memory_optimization=True
            )
        }
        
        test_data = self.sample_data.iloc[:50]
        results = {}
        
        print("\\n1. Engine Performance Comparison")
        
        for name, config in configs.items():
            print(f"\\n   Testing {name} Engine:")
            
            if name == 'Optimized':
                engine = OptimizedInferenceEngine(self.model, self.tokenizer, config)
            else:
                engine = InferenceEngine(self.model, self.tokenizer, config)
            
            try:
                # Warmup
                for _ in range(3):
                    engine.predict(test_data.iloc[:5])
                
                # Measure performance
                times = []
                memory_usage = []
                
                for _ in range(10):
                    start_time = time.time()
                    result = engine.predict(test_data)
                    end_time = time.time()
                    
                    times.append((end_time - start_time) * 1000)
                    memory_usage.append(getattr(result, 'memory_usage_mb', 0))
                
                # Get statistics
                stats = engine.get_stats() if hasattr(engine, 'get_stats') else {}
                
                results[name] = {
                    'avg_time_ms': np.mean(times),
                    'std_time_ms': np.std(times),
                    'min_time_ms': np.min(times),
                    'max_time_ms': np.max(times),
                    'avg_memory_mb': np.mean(memory_usage),
                    'cache_hit_rate': stats.get('cache_hit_rate', 0),
                    'total_requests': stats.get('total_requests', 0)
                }
                
                print(f"     Average time: {results[name]['avg_time_ms']:.2f}ms ± {results[name]['std_time_ms']:.2f}ms")
                print(f"     Memory usage: {results[name]['avg_memory_mb']:.2f}MB")
                print(f"     Cache hit rate: {results[name]['cache_hit_rate']:.2%}")
                
            finally:
                if hasattr(engine, 'shutdown'):
                    engine.shutdown()
        
        # Performance comparison
        print("\\n2. Performance Summary")
        baseline_time = results['Basic']['avg_time_ms']
        
        for name, metrics in results.items():
            if name != 'Basic':
                speedup = baseline_time / metrics['avg_time_ms']
                print(f"   {name} vs Basic: {speedup:.2f}x speedup")
        
        self.results['performance_analysis'] = results
    
    def print_comprehensive_summary(self):
        """Print comprehensive summary of all demonstrations."""
        print("\\n" + "="*80)
        print("COMPREHENSIVE SUMMARY")
        print("="*80)
        
        print("\\n🚀 OPTIMIZATION RESULTS:")
        if 'optimization' in self.results:
            opt_results = self.results['optimization']
            print(f"   Model optimization speedup: {opt_results['speedup']:.2f}x")
            print(f"   Original time: {opt_results['original_time_ms']:.2f}ms")
            print(f"   Optimized time: {opt_results['optimized_time_ms']:.2f}ms")
        
        print("\\n⚡ INFERENCE ENGINE PERFORMANCE:")
        if 'inference_engines' in self.results:
            ie_results = self.results['inference_engines']
            print(f"   Basic engine: {ie_results['basic_time_ms']:.2f}ms")
            print(f"   Batch engine: {ie_results['batch_time_ms']:.2f}ms")
            print(f"   Optimized engine: {ie_results['optimized_time_ms']:.2f}ms")
            print(f"   Cache speedup: {ie_results['cache_speedup']:.2f}x")
        
        print("\\n💾 CACHING AND BATCHING:")
        if 'caching_batching' in self.results:
            cb_results = self.results['caching_batching']
            print(f"   Cache put operations: {cb_results['cache_put_ops_per_sec']:.0f} ops/sec")
            print(f"   Cache get operations: {cb_results['cache_get_ops_per_sec']:.0f} ops/sec")
            print(f"   Average batch size: {cb_results['avg_batch_size']:.1f}")
        
        print("\\n📦 MODEL EXPORT:")
        if 'model_export' in self.results:
            export_results = self.results['model_export']
            for format_name, result in export_results.items():
                if 'model_size_mb' in result:
                    print(f"   {format_name.upper()}: {result['model_size_mb']:.2f}MB")
        
        print("\\n🌐 PRODUCTION SERVING:")
        if 'production_serving' in self.results:
            ps_results = self.results['production_serving']
            metrics = ps_results['metrics']
            print(f"   Success rate: {metrics['success_rate']:.2%}")
            print(f"   Average response time: {metrics['avg_response_time_ms']:.1f}ms")
            print(f"   Health status: {'✓ Healthy' if ps_results['health_status']['healthy'] else '✗ Unhealthy'}")
        
        print("\\n📊 PERFORMANCE COMPARISON:")
        if 'performance_analysis' in self.results:
            pa_results = self.results['performance_analysis']
            baseline_time = pa_results['Basic']['avg_time_ms']
            
            for name, metrics in pa_results.items():
                speedup = baseline_time / metrics['avg_time_ms']
                cache_rate = metrics['cache_hit_rate']
                print(f"   {name}: {metrics['avg_time_ms']:.2f}ms ({speedup:.2f}x, {cache_rate:.1%} cache)")
        
        print("\\n✅ KEY ACHIEVEMENTS:")
        print("   • Demonstrated comprehensive model optimization")
        print("   • Showcased efficient inference engines with caching")
        print("   • Implemented dynamic batching for throughput")
        print("   • Exported models for production deployment")
        print("   • Set up production-ready serving infrastructure")
        print("   • Provided detailed performance analysis")
        
        print("\\n🎯 PRODUCTION READINESS:")
        print("   • Scalable inference with batching and caching")
        print("   • Memory-optimized processing")
        print("   • Health monitoring and metrics collection")
        print("   • Multiple export formats for deployment")
        print("   • Comprehensive error handling")
        print("   • Performance benchmarking tools")
        
        print("\\n" + "="*80)
        print("Serving and optimization demonstration completed successfully! 🎉")
        print("="*80)


def main():
    """Main demonstration function."""
    print("TabGPT Serving and Optimization Comprehensive Demo")
    print("This demo showcases all serving and optimization features")
    print("=" * 80)
    
    # Initialize demo
    demo = ServingOptimizationDemo()
    
    try:
        # Run all demonstrations
        demo.create_sample_model_and_data()
        optimized_model = demo.demonstrate_model_optimization()
        demo.demonstrate_inference_engines()
        demo.demonstrate_caching_and_batching()
        demo.demonstrate_model_export()
        demo.demonstrate_production_serving()
        demo.demonstrate_performance_analysis()
        
        # Print comprehensive summary
        demo.print_comprehensive_summary()
        
        # Save results
        results_file = "serving_optimization_results.json"
        with open(results_file, 'w') as f:
            json.dump(demo.results, f, indent=2, default=str)
        print(f"\\nDetailed results saved to: {results_file}")
        
    except KeyboardInterrupt:
        print("\\nDemo interrupted by user")
    except Exception as e:
        print(f"\\nDemo failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\\nThank you for exploring TabGPT serving and optimization features! 🚀")


if __name__ == "__main__":
    main()