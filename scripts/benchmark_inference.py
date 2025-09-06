#!/usr/bin/env python3
"""
Comprehensive benchmarking script for TabGPT inference performance.

This script provides detailed performance analysis including:
- Latency measurements across different batch sizes
- Throughput analysis under various loads
- Memory usage profiling
- Optimization impact assessment
- Comparison between different engine configurations
"""

import argparse
import json
import time
import statistics
from pathlib import Path
from typing import Dict, List, Any, Optional
import sys
import os

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ThreadPoolExecutor, as_completed

from tabgpt.serving import (
    InferenceEngine, BatchInferenceEngine, OptimizedInferenceEngine,
    InferenceConfig, ModelOptimizer, OptimizationConfig
)


class BenchmarkModel(nn.Module):
    """Configurable model for benchmarking."""
    
    def __init__(self, input_dim=768, output_dim=2, num_layers=12, hidden_dim=None):
        super().__init__()
        
        if hidden_dim is None:
            hidden_dim = input_dim
        
        # Create transformer-like architecture
        layers = []
        current_dim = input_dim
        
        # Input projection
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.ReLU())
        
        # Hidden layers
        for i in range(num_layers):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
        
        # Output projection
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.model = nn.Sequential(*layers)
        self.config = type('Config', (), {'hidden_size': hidden_dim})()
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize model weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def forward(self, input_ids, attention_mask=None, **kwargs):
        batch_size, seq_len = input_ids.shape
        
        # Simulate feature extraction (mean pooling)
        features = input_ids.float().mean(dim=1)
        
        # Pass through model
        logits = self.model(features)
        
        return type('Output', (), {
            'logits': logits,
            'predictions': logits,
            'probabilities': torch.softmax(logits, dim=-1),
            'features': features
        })()


class BenchmarkTokenizer:
    """Configurable tokenizer for benchmarking."""
    
    def __init__(self, vocab_size=30000, max_length=512, complexity="medium"):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.complexity = complexity
    
    def encode_batch(self, df):
        batch_size = len(df)
        
        if self.complexity == "simple":
            # Fixed length sequences
            seq_len = min(self.max_length, 128)
            input_ids = torch.randint(1, self.vocab_size, (batch_size, seq_len))
            attention_mask = torch.ones(batch_size, seq_len)
        
        elif self.complexity == "medium":
            # Variable length sequences
            seq_lengths = np.random.randint(64, min(self.max_length, 256), batch_size)
            max_seq_len = max(seq_lengths)
            
            input_ids = torch.zeros(batch_size, max_seq_len, dtype=torch.long)
            attention_mask = torch.zeros(batch_size, max_seq_len, dtype=torch.long)
            
            for i, seq_len in enumerate(seq_lengths):
                input_ids[i, :seq_len] = torch.randint(1, self.vocab_size, (seq_len,))
                attention_mask[i, :seq_len] = 1
        
        else:  # complex
            # Highly variable sequences with padding
            seq_lengths = np.random.randint(32, self.max_length, batch_size)
            max_seq_len = max(seq_lengths)
            
            input_ids = torch.zeros(batch_size, max_seq_len, dtype=torch.long)
            attention_mask = torch.zeros(batch_size, max_seq_len, dtype=torch.long)
            
            for i, seq_len in enumerate(seq_lengths):
                # Add some structure to the sequences
                input_ids[i, :seq_len] = torch.randint(1, self.vocab_size, (seq_len,))
                attention_mask[i, :seq_len] = 1
                
                # Add special tokens
                input_ids[i, 0] = 1  # CLS token
                if seq_len > 1:
                    input_ids[i, seq_len-1] = 2  # SEP token
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }


class PerformanceBenchmarker:
    """Comprehensive performance benchmarking suite."""
    
    def __init__(self, output_dir: str = "benchmark_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.results = {}
        self.plots_dir = self.output_dir / "plots"
        self.plots_dir.mkdir(exist_ok=True)
    
    def create_test_data(self, num_samples: int, num_features: int = 20, complexity: str = "medium") -> pd.DataFrame:
        """Create test data with specified complexity."""
        np.random.seed(42)  # For reproducible results
        
        data = {}
        
        if complexity == "simple":
            # Simple numerical features only
            for i in range(num_features):
                data[f'feature_{i}'] = np.random.randn(num_samples)
        
        elif complexity == "medium":
            # Mix of numerical and categorical
            for i in range(num_features // 2):
                data[f'num_feature_{i}'] = np.random.randn(num_samples)
            
            categories = ['A', 'B', 'C', 'D', 'E']
            for i in range(num_features // 2):
                data[f'cat_feature_{i}'] = np.random.choice(categories, num_samples)
        
        else:  # complex
            # Complex mix with missing values and outliers
            for i in range(num_features // 3):
                values = np.random.randn(num_samples)
                # Add outliers
                outlier_mask = np.random.random(num_samples) < 0.05
                values[outlier_mask] *= 10
                # Add missing values
                missing_mask = np.random.random(num_samples) < 0.1
                values[missing_mask] = np.nan
                data[f'num_feature_{i}'] = values
            
            # High cardinality categorical
            high_card_categories = [f'cat_{i}' for i in range(100)]
            for i in range(num_features // 3):
                data[f'high_card_feature_{i}'] = np.random.choice(high_card_categories, num_samples)
            
            # Regular categorical
            categories = ['A', 'B', 'C', 'D', 'E']
            for i in range(num_features // 3):
                data[f'cat_feature_{i}'] = np.random.choice(categories, num_samples)
        
        return pd.DataFrame(data)
    
    def benchmark_latency(
        self,
        engine,
        test_data: pd.DataFrame,
        batch_sizes: List[int],
        num_runs: int = 50,
        warmup_runs: int = 10
    ) -> Dict[str, Any]:
        """Benchmark inference latency across different batch sizes."""
        print(f"Benchmarking latency with batch sizes: {batch_sizes}")
        
        results = {}
        
        for batch_size in batch_sizes:
            print(f"  Testing batch size: {batch_size}")
            
            if batch_size > len(test_data):
                batch_data = test_data.sample(n=batch_size, replace=True)
            else:
                batch_data = test_data.iloc[:batch_size]
            
            # Warmup
            for _ in range(warmup_runs):
                try:
                    engine.predict(batch_data)
                except Exception as e:
                    print(f"    Warmup failed: {e}")
                    break
            
            # Measure
            times = []
            successful_runs = 0
            
            for run in range(num_runs):
                try:
                    start_time = time.perf_counter()
                    result = engine.predict(batch_data)
                    end_time = time.perf_counter()
                    
                    latency_ms = (end_time - start_time) * 1000
                    times.append(latency_ms)
                    successful_runs += 1
                    
                except Exception as e:
                    print(f"    Run {run} failed: {e}")
            
            if times:
                results[batch_size] = {
                    'mean_ms': statistics.mean(times),
                    'median_ms': statistics.median(times),
                    'std_ms': statistics.stdev(times) if len(times) > 1 else 0,
                    'min_ms': min(times),
                    'max_ms': max(times),
                    'p95_ms': np.percentile(times, 95),
                    'p99_ms': np.percentile(times, 99),
                    'successful_runs': successful_runs,
                    'total_runs': num_runs,
                    'samples_per_second': batch_size / (statistics.mean(times) / 1000)
                }
            else:
                results[batch_size] = {'error': 'All runs failed'}
        
        return results
    
    def benchmark_throughput(
        self,
        engine,
        test_batches: List[pd.DataFrame],
        max_workers: int = 4,
        concurrent_requests: List[int] = None
    ) -> Dict[str, Any]:
        """Benchmark throughput under different concurrency levels."""
        if concurrent_requests is None:
            concurrent_requests = [1, 2, 4, 8]
        
        print(f"Benchmarking throughput with concurrency levels: {concurrent_requests}")
        
        results = {}
        
        for num_workers in concurrent_requests:
            print(f"  Testing {num_workers} concurrent requests")
            
            # Limit batches to avoid overwhelming the system
            limited_batches = test_batches[:min(len(test_batches), num_workers * 5)]
            
            start_time = time.perf_counter()
            total_samples = sum(len(batch) for batch in limited_batches)
            successful_requests = 0
            
            try:
                with ThreadPoolExecutor(max_workers=num_workers) as executor:
                    futures = [executor.submit(engine.predict, batch) for batch in limited_batches]
                    
                    for future in as_completed(futures):
                        try:
                            future.result(timeout=30.0)  # 30 second timeout
                            successful_requests += 1
                        except Exception as e:
                            print(f"    Request failed: {e}")
                
                end_time = time.perf_counter()
                total_time = end_time - start_time
                
                results[num_workers] = {
                    'total_samples': total_samples,
                    'total_requests': len(limited_batches),
                    'successful_requests': successful_requests,
                    'total_time_s': total_time,
                    'samples_per_second': total_samples / total_time if total_time > 0 else 0,
                    'requests_per_second': successful_requests / total_time if total_time > 0 else 0,
                    'success_rate': successful_requests / len(limited_batches) if limited_batches else 0
                }
                
            except Exception as e:
                results[num_workers] = {'error': str(e)}
        
        return results
    
    def benchmark_memory_usage(
        self,
        engine,
        test_data: pd.DataFrame,
        batch_sizes: List[int]
    ) -> Dict[str, Any]:
        """Benchmark memory usage across different batch sizes."""
        print(f"Benchmarking memory usage with batch sizes: {batch_sizes}")
        
        results = {}
        
        for batch_size in batch_sizes:
            print(f"  Testing batch size: {batch_size}")
            
            if batch_size > len(test_data):
                batch_data = test_data.sample(n=batch_size, replace=True)
            else:
                batch_data = test_data.iloc[:batch_size]
            
            try:
                # Clear cache before measurement
                if hasattr(engine, 'clear_cache'):
                    engine.clear_cache()
                
                # Force garbage collection
                import gc
                gc.collect()
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                
                # Measure memory before
                memory_before = self._get_memory_usage()
                
                # Run inference
                result = engine.predict(batch_data)
                
                # Measure memory after
                memory_after = self._get_memory_usage()
                
                results[batch_size] = {
                    'memory_before_mb': memory_before,
                    'memory_after_mb': memory_after,
                    'memory_increase_mb': memory_after - memory_before,
                    'result_memory_mb': getattr(result, 'memory_usage_mb', 0),
                    'memory_per_sample_mb': (memory_after - memory_before) / batch_size if batch_size > 0 else 0
                }
                
            except Exception as e:
                results[batch_size] = {'error': str(e)}
        
        return results
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            if torch.cuda.is_available():
                return torch.cuda.memory_allocated() / (1024 * 1024)
            else:
                import psutil
                process = psutil.Process()
                return process.memory_info().rss / (1024 * 1024)
        except Exception:
            return 0.0
    
    def compare_engines(
        self,
        engine_configs: Dict[str, Dict],
        test_data: pd.DataFrame,
        batch_sizes: List[int] = None
    ) -> Dict[str, Any]:
        """Compare performance across different engine configurations."""
        if batch_sizes is None:
            batch_sizes = [1, 10, 50, 100]
        
        print(f"Comparing {len(engine_configs)} engine configurations")
        
        comparison_results = {}
        
        for engine_name, config in engine_configs.items():
            print(f"\\nTesting {engine_name}...")
            
            try:
                # Create engine
                engine_class = config['class']
                engine_config = config['config']
                model = config['model']
                tokenizer = config['tokenizer']
                
                engine = engine_class(model, tokenizer, engine_config)
                
                # Benchmark latency
                latency_results = self.benchmark_latency(
                    engine, test_data, batch_sizes, num_runs=20
                )
                
                # Get engine stats
                stats = engine.get_stats() if hasattr(engine, 'get_stats') else {}
                
                comparison_results[engine_name] = {
                    'latency': latency_results,
                    'stats': stats,
                    'config': {
                        'class': engine_class.__name__,
                        'config_dict': engine_config.__dict__ if hasattr(engine_config, '__dict__') else str(engine_config)
                    }
                }
                
                # Cleanup
                if hasattr(engine, 'shutdown'):
                    engine.shutdown()
                
            except Exception as e:
                print(f"  Failed to test {engine_name}: {e}")
                comparison_results[engine_name] = {'error': str(e)}
        
        return comparison_results
    
    def create_visualizations(self, results: Dict[str, Any]):
        """Create performance visualization plots."""
        print("Creating performance visualizations...")
        
        try:
            # Set style
            plt.style.use('default')
            sns.set_palette("husl")
            
            # 1. Latency vs Batch Size
            if 'latency_comparison' in results:
                self._plot_latency_comparison(results['latency_comparison'])
            
            # 2. Throughput Comparison
            if 'throughput_comparison' in results:
                self._plot_throughput_comparison(results['throughput_comparison'])
            
            # 3. Memory Usage
            if 'memory_usage' in results:
                self._plot_memory_usage(results['memory_usage'])
            
            # 4. Engine Comparison
            if 'engine_comparison' in results:
                self._plot_engine_comparison(results['engine_comparison'])
            
            print(f"Visualizations saved to {self.plots_dir}")
            
        except Exception as e:
            print(f"Failed to create visualizations: {e}")
    
    def _plot_latency_comparison(self, latency_data: Dict):
        """Plot latency comparison across batch sizes."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Extract data
        batch_sizes = []
        mean_latencies = []
        p95_latencies = []
        
        for batch_size, metrics in latency_data.items():
            if isinstance(metrics, dict) and 'mean_ms' in metrics:
                batch_sizes.append(batch_size)
                mean_latencies.append(metrics['mean_ms'])
                p95_latencies.append(metrics['p95_ms'])
        
        if batch_sizes:
            # Latency plot
            ax1.plot(batch_sizes, mean_latencies, 'o-', label='Mean Latency', linewidth=2)
            ax1.plot(batch_sizes, p95_latencies, 's--', label='95th Percentile', linewidth=2)
            ax1.set_xlabel('Batch Size')
            ax1.set_ylabel('Latency (ms)')
            ax1.set_title('Inference Latency vs Batch Size')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Throughput plot
            throughput = [batch_size / (latency / 1000) for batch_size, latency in zip(batch_sizes, mean_latencies)]
            ax2.plot(batch_sizes, throughput, 'o-', color='green', linewidth=2)
            ax2.set_xlabel('Batch Size')
            ax2.set_ylabel('Throughput (samples/sec)')
            ax2.set_title('Throughput vs Batch Size')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'latency_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_throughput_comparison(self, throughput_data: Dict):
        """Plot throughput comparison across concurrency levels."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        concurrency_levels = []
        samples_per_sec = []
        requests_per_sec = []
        
        for level, metrics in throughput_data.items():
            if isinstance(metrics, dict) and 'samples_per_second' in metrics:
                concurrency_levels.append(level)
                samples_per_sec.append(metrics['samples_per_second'])
                requests_per_sec.append(metrics['requests_per_second'])
        
        if concurrency_levels:
            ax.plot(concurrency_levels, samples_per_sec, 'o-', label='Samples/sec', linewidth=2)
            ax.plot(concurrency_levels, requests_per_sec, 's--', label='Requests/sec', linewidth=2)
            ax.set_xlabel('Concurrent Requests')
            ax.set_ylabel('Throughput')
            ax.set_title('Throughput vs Concurrency Level')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'throughput_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_memory_usage(self, memory_data: Dict):
        """Plot memory usage across batch sizes."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        batch_sizes = []
        memory_per_sample = []
        
        for batch_size, metrics in memory_data.items():
            if isinstance(metrics, dict) and 'memory_per_sample_mb' in metrics:
                batch_sizes.append(batch_size)
                memory_per_sample.append(metrics['memory_per_sample_mb'])
        
        if batch_sizes:
            ax.plot(batch_sizes, memory_per_sample, 'o-', linewidth=2, color='red')
            ax.set_xlabel('Batch Size')
            ax.set_ylabel('Memory per Sample (MB)')
            ax.set_title('Memory Usage Efficiency')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'memory_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_engine_comparison(self, engine_data: Dict):
        """Plot comparison between different engines."""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Extract data for comparison
        engines = []
        batch_sizes = set()
        
        for engine_name, data in engine_data.items():
            if 'latency' in data:
                engines.append(engine_name)
                batch_sizes.update(data['latency'].keys())
        
        batch_sizes = sorted([bs for bs in batch_sizes if isinstance(bs, int)])
        
        if engines and batch_sizes:
            # Create comparison plot
            x = np.arange(len(batch_sizes))
            width = 0.8 / len(engines)
            
            for i, engine_name in enumerate(engines):
                latencies = []
                for bs in batch_sizes:
                    if (bs in engine_data[engine_name].get('latency', {}) and
                        'mean_ms' in engine_data[engine_name]['latency'][bs]):
                        latencies.append(engine_data[engine_name]['latency'][bs]['mean_ms'])
                    else:
                        latencies.append(0)
                
                ax.bar(x + i * width, latencies, width, label=engine_name, alpha=0.8)
            
            ax.set_xlabel('Batch Size')
            ax.set_ylabel('Mean Latency (ms)')
            ax.set_title('Engine Performance Comparison')
            ax.set_xticks(x + width * (len(engines) - 1) / 2)
            ax.set_xticklabels(batch_sizes)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'engine_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_results(self, results: Dict[str, Any], filename: str = "benchmark_results.json"):
        """Save benchmark results to file."""
        output_file = self.output_dir / filename
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        converted_results = convert_numpy(results)
        
        with open(output_file, 'w') as f:
            json.dump(converted_results, f, indent=2, default=str)
        
        print(f"Results saved to {output_file}")


def main():
    """Main benchmarking function."""
    parser = argparse.ArgumentParser(description="TabGPT Inference Performance Benchmark")
    parser.add_argument("--model-size", choices=["small", "medium", "large"], default="medium",
                       help="Model size for benchmarking")
    parser.add_argument("--data-complexity", choices=["simple", "medium", "complex"], default="medium",
                       help="Test data complexity")
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 5, 10, 25, 50, 100],
                       help="Batch sizes to test")
    parser.add_argument("--num-samples", type=int, default=1000,
                       help="Number of samples in test dataset")
    parser.add_argument("--num-runs", type=int, default=50,
                       help="Number of runs per benchmark")
    parser.add_argument("--output-dir", type=str, default="benchmark_results",
                       help="Output directory for results")
    parser.add_argument("--skip-plots", action="store_true",
                       help="Skip generating plots")
    parser.add_argument("--engines", nargs="+", 
                       choices=["basic", "batch", "optimized", "all"], 
                       default=["all"],
                       help="Engines to benchmark")
    
    args = parser.parse_args()
    
    print("TabGPT Inference Performance Benchmark")
    print("=" * 50)
    print(f"Model size: {args.model_size}")
    print(f"Data complexity: {args.data_complexity}")
    print(f"Batch sizes: {args.batch_sizes}")
    print(f"Number of samples: {args.num_samples}")
    print(f"Number of runs: {args.num_runs}")
    print()
    
    # Initialize benchmarker
    benchmarker = PerformanceBenchmarker(args.output_dir)
    
    # Create test data
    print("Creating test data...")
    test_data = benchmarker.create_test_data(
        args.num_samples, 
        num_features=20, 
        complexity=args.data_complexity
    )
    
    # Create model and tokenizer
    print("Creating model and tokenizer...")
    
    model_configs = {
        "small": {"input_dim": 256, "num_layers": 4, "hidden_dim": 256},
        "medium": {"input_dim": 512, "num_layers": 8, "hidden_dim": 512},
        "large": {"input_dim": 768, "num_layers": 12, "hidden_dim": 768}
    }
    
    model_config = model_configs[args.model_size]
    model = BenchmarkModel(**model_config)
    tokenizer = BenchmarkTokenizer(complexity=args.data_complexity)
    
    # Determine which engines to test
    engines_to_test = args.engines
    if "all" in engines_to_test:
        engines_to_test = ["basic", "batch", "optimized"]
    
    # Create engine configurations
    engine_configs = {}
    
    if "basic" in engines_to_test:
        engine_configs["Basic"] = {
            "class": InferenceEngine,
            "config": InferenceConfig(use_cache=False, batch_size=32),
            "model": model,
            "tokenizer": tokenizer
        }
    
    if "batch" in engines_to_test:
        engine_configs["Batch"] = {
            "class": BatchInferenceEngine,
            "config": InferenceConfig(
                batch_size=32, 
                max_batch_size=64, 
                use_cache=True,
                batch_timeout_ms=100
            ),
            "model": model,
            "tokenizer": tokenizer
        }
    
    if "optimized" in engines_to_test:
        engine_configs["Optimized"] = {
            "class": OptimizedInferenceEngine,
            "config": InferenceConfig(
                batch_size=32,
                max_batch_size=64,
                use_cache=True,
                cache_size=500,
                enable_memory_optimization=True,
                num_workers=2
            ),
            "model": model,
            "tokenizer": tokenizer
        }
    
    # Run benchmarks
    all_results = {}
    
    # 1. Engine comparison
    print("\\nRunning engine comparison...")
    engine_comparison = benchmarker.compare_engines(
        engine_configs, test_data, args.batch_sizes
    )
    all_results["engine_comparison"] = engine_comparison
    
    # 2. Detailed analysis with optimized engine
    if "optimized" in engines_to_test:
        print("\\nRunning detailed analysis with optimized engine...")
        
        optimized_engine = OptimizedInferenceEngine(
            model, tokenizer,
            InferenceConfig(
                batch_size=32,
                max_batch_size=64,
                use_cache=True,
                cache_size=500,
                enable_memory_optimization=True
            )
        )
        
        try:
            # Latency analysis
            latency_results = benchmarker.benchmark_latency(
                optimized_engine, test_data, args.batch_sizes, args.num_runs
            )
            all_results["latency_analysis"] = latency_results
            
            # Throughput analysis
            test_batches = [
                test_data.iloc[i:i+20] for i in range(0, min(200, len(test_data)), 20)
            ]
            throughput_results = benchmarker.benchmark_throughput(
                optimized_engine, test_batches, max_workers=4
            )
            all_results["throughput_analysis"] = throughput_results
            
            # Memory analysis
            memory_results = benchmarker.benchmark_memory_usage(
                optimized_engine, test_data, args.batch_sizes[:4]  # Limit for memory test
            )
            all_results["memory_analysis"] = memory_results
            
        finally:
            optimized_engine.shutdown()
    
    # Save results
    benchmarker.save_results(all_results)
    
    # Create visualizations
    if not args.skip_plots:
        try:
            benchmarker.create_visualizations(all_results)
        except ImportError:
            print("Matplotlib not available, skipping plots")
        except Exception as e:
            print(f"Failed to create plots: {e}")
    
    # Print summary
    print("\\n" + "=" * 50)
    print("BENCHMARK SUMMARY")
    print("=" * 50)
    
    if "engine_comparison" in all_results:
        print("\\nEngine Performance (mean latency for batch size 50):")
        for engine_name, data in all_results["engine_comparison"].items():
            if "latency" in data and 50 in data["latency"]:
                latency = data["latency"][50].get("mean_ms", "N/A")
                throughput = data["latency"][50].get("samples_per_second", "N/A")
                print(f"  {engine_name}: {latency:.2f}ms ({throughput:.1f} samples/sec)")
    
    if "memory_analysis" in all_results:
        print("\\nMemory Usage (per sample):")
        for batch_size, metrics in all_results["memory_analysis"].items():
            if isinstance(metrics, dict) and "memory_per_sample_mb" in metrics:
                memory_per_sample = metrics["memory_per_sample_mb"]
                print(f"  Batch size {batch_size}: {memory_per_sample:.3f} MB/sample")
    
    print(f"\\nDetailed results saved to: {benchmarker.output_dir}")
    print("Benchmark completed successfully!")


if __name__ == "__main__":
    main()