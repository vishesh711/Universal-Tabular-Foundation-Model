"""Benchmarking framework for TabGPT models."""

import time
import json
import logging
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
from dataclasses import dataclass, field
from pathlib import Path
from abc import ABC, abstractmethod
import warnings

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, mean_squared_error, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression

from .metrics import EvaluationMetrics, compute_all_metrics

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Result of a benchmark evaluation."""
    
    model_name: str
    dataset_name: str
    task_type: str
    metrics: Dict[str, float]
    training_time: float
    inference_time: float
    model_size: int
    memory_usage: float
    additional_info: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'model_name': self.model_name,
            'dataset_name': self.dataset_name,
            'task_type': self.task_type,
            'metrics': self.metrics,
            'training_time': self.training_time,
            'inference_time': self.inference_time,
            'model_size': self.model_size,
            'memory_usage': self.memory_usage,
            'additional_info': self.additional_info
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BenchmarkResult':
        """Create from dictionary."""
        return cls(**data)


class TabularBenchmark(ABC):
    """Abstract base class for tabular benchmarks."""
    
    def __init__(
        self,
        name: str,
        description: str = "",
        task_type: str = "classification"
    ):
        self.name = name
        self.description = description
        self.task_type = task_type
        self.results: List[BenchmarkResult] = []
    
    @abstractmethod
    def load_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Load benchmark dataset."""
        pass
    
    @abstractmethod
    def evaluate_model(
        self,
        model: Any,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        **kwargs
    ) -> BenchmarkResult:
        """Evaluate a model on the benchmark."""
        pass
    
    def run_benchmark(
        self,
        models: Dict[str, Any],
        test_size: float = 0.2,
        random_state: int = 42,
        **kwargs
    ) -> List[BenchmarkResult]:
        """Run benchmark on multiple models."""
        logger.info(f"Running benchmark: {self.name}")
        
        # Load data
        X, y = self.load_data()
        
        # Split data
        stratify = y if self.task_type == 'classification' else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=stratify
        )
        
        results = []
        for model_name, model in models.items():
            logger.info(f"Evaluating model: {model_name}")
            try:
                result = self.evaluate_model(
                    model, X_train, y_train, X_test, y_test, **kwargs
                )
                results.append(result)
                self.results.append(result)
            except Exception as e:
                logger.error(f"Error evaluating {model_name}: {e}")
                continue
        
        return results
    
    def get_leaderboard(self, metric: str = "accuracy") -> pd.DataFrame:
        """Get leaderboard sorted by metric."""
        if not self.results:
            return pd.DataFrame()
        
        data = []
        for result in self.results:
            row = {
                'Model': result.model_name,
                'Dataset': result.dataset_name,
                'Task': result.task_type,
                'Training Time (s)': result.training_time,
                'Inference Time (s)': result.inference_time,
                'Model Size (MB)': result.model_size / (1024 * 1024),
                'Memory Usage (MB)': result.memory_usage / (1024 * 1024)
            }
            row.update(result.metrics)
            data.append(row)
        
        df = pd.DataFrame(data)
        if metric in df.columns:
            ascending = metric.lower() in ['loss', 'error', 'mse', 'mae']
            df = df.sort_values(metric, ascending=ascending)
        
        return df


class ClassificationBenchmark(TabularBenchmark):
    """Benchmark for classification tasks."""
    
    def __init__(
        self,
        name: str,
        data_loader: Callable[[], Tuple[pd.DataFrame, pd.Series]],
        description: str = "",
        num_classes: Optional[int] = None
    ):
        super().__init__(name, description, "classification")
        self.data_loader = data_loader
        self.num_classes = num_classes
    
    def load_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Load classification dataset."""
        return self.data_loader()
    
    def evaluate_model(
        self,
        model: Any,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        **kwargs
    ) -> BenchmarkResult:
        """Evaluate classification model."""
        model_name = model.__class__.__name__
        
        # Training
        start_time = time.time()
        if hasattr(model, 'fit'):
            model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        # Inference
        start_time = time.time()
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)
            y_pred = np.argmax(y_pred_proba, axis=1)
        else:
            y_pred = model.predict(X_test)
            y_pred_proba = None
        inference_time = time.time() - start_time
        
        # Compute metrics
        metrics = compute_all_metrics(
            y_true=y_test,
            y_pred=y_pred,
            y_pred_proba=y_pred_proba,
            task_type="classification"
        )
        
        # Model size and memory
        model_size = self._get_model_size(model)
        memory_usage = self._get_memory_usage()
        
        return BenchmarkResult(
            model_name=model_name,
            dataset_name=self.name,
            task_type=self.task_type,
            metrics=metrics,
            training_time=training_time,
            inference_time=inference_time,
            model_size=model_size,
            memory_usage=memory_usage
        )
    
    def _get_model_size(self, model: Any) -> int:
        """Estimate model size in bytes."""
        if hasattr(model, 'get_params'):
            # Sklearn-like model
            return len(str(model.get_params())) * 8  # Rough estimate
        elif hasattr(model, 'state_dict'):
            # PyTorch model
            param_size = 0
            for param in model.parameters():
                param_size += param.nelement() * param.element_size()
            return param_size
        else:
            return 0
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in bytes."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss
        except ImportError:
            return 0.0


class RegressionBenchmark(TabularBenchmark):
    """Benchmark for regression tasks."""
    
    def __init__(
        self,
        name: str,
        data_loader: Callable[[], Tuple[pd.DataFrame, pd.Series]],
        description: str = ""
    ):
        super().__init__(name, description, "regression")
        self.data_loader = data_loader
    
    def load_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Load regression dataset."""
        return self.data_loader()
    
    def evaluate_model(
        self,
        model: Any,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        **kwargs
    ) -> BenchmarkResult:
        """Evaluate regression model."""
        model_name = model.__class__.__name__
        
        # Training
        start_time = time.time()
        if hasattr(model, 'fit'):
            model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        # Inference
        start_time = time.time()
        y_pred = model.predict(X_test)
        inference_time = time.time() - start_time
        
        # Compute metrics
        metrics = compute_all_metrics(
            y_true=y_test,
            y_pred=y_pred,
            task_type="regression"
        )
        
        # Model size and memory
        model_size = self._get_model_size(model)
        memory_usage = self._get_memory_usage()
        
        return BenchmarkResult(
            model_name=model_name,
            dataset_name=self.name,
            task_type=self.task_type,
            metrics=metrics,
            training_time=training_time,
            inference_time=inference_time,
            model_size=model_size,
            memory_usage=memory_usage
        )
    
    def _get_model_size(self, model: Any) -> int:
        """Estimate model size in bytes."""
        if hasattr(model, 'get_params'):
            return len(str(model.get_params())) * 8
        elif hasattr(model, 'state_dict'):
            param_size = 0
            for param in model.parameters():
                param_size += param.nelement() * param.element_size()
            return param_size
        else:
            return 0
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in bytes."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss
        except ImportError:
            return 0.0


class AnomalyDetectionBenchmark(TabularBenchmark):
    """Benchmark for anomaly detection tasks."""
    
    def __init__(
        self,
        name: str,
        data_loader: Callable[[], Tuple[pd.DataFrame, pd.Series]],
        description: str = ""
    ):
        super().__init__(name, description, "anomaly_detection")
        self.data_loader = data_loader
    
    def load_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Load anomaly detection dataset."""
        return self.data_loader()
    
    def evaluate_model(
        self,
        model: Any,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        **kwargs
    ) -> BenchmarkResult:
        """Evaluate anomaly detection model."""
        model_name = model.__class__.__name__
        
        # Training (usually unsupervised)
        start_time = time.time()
        if hasattr(model, 'fit'):
            # For unsupervised methods, only use normal data
            normal_data = X_train[y_train == 0] if len(np.unique(y_train)) > 1 else X_train
            model.fit(normal_data)
        training_time = time.time() - start_time
        
        # Inference
        start_time = time.time()
        if hasattr(model, 'decision_function'):
            anomaly_scores = model.decision_function(X_test)
        elif hasattr(model, 'score_samples'):
            anomaly_scores = -model.score_samples(X_test)  # Negative log-likelihood
        else:
            anomaly_scores = model.predict(X_test)
        inference_time = time.time() - start_time
        
        # Compute metrics
        metrics = {}
        if len(np.unique(y_test)) > 1:
            # Supervised evaluation
            try:
                from sklearn.metrics import roc_auc_score, average_precision_score
                metrics['auc'] = roc_auc_score(y_test, anomaly_scores)
                metrics['average_precision'] = average_precision_score(y_test, anomaly_scores)
            except Exception as e:
                logger.warning(f"Could not compute AUC metrics: {e}")
        
        metrics['mean_anomaly_score'] = np.mean(anomaly_scores)
        metrics['std_anomaly_score'] = np.std(anomaly_scores)
        
        # Model size and memory
        model_size = self._get_model_size(model)
        memory_usage = self._get_memory_usage()
        
        return BenchmarkResult(
            model_name=model_name,
            dataset_name=self.name,
            task_type=self.task_type,
            metrics=metrics,
            training_time=training_time,
            inference_time=inference_time,
            model_size=model_size,
            memory_usage=memory_usage
        )
    
    def _get_model_size(self, model: Any) -> int:
        """Estimate model size in bytes."""
        if hasattr(model, 'get_params'):
            return len(str(model.get_params())) * 8
        elif hasattr(model, 'state_dict'):
            param_size = 0
            for param in model.parameters():
                param_size += param.nelement() * param.element_size()
            return param_size
        else:
            return 0
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in bytes."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss
        except ImportError:
            return 0.0


class BenchmarkSuite:
    """Suite of benchmarks for comprehensive evaluation."""
    
    def __init__(self, name: str = "TabGPT Benchmark Suite"):
        self.name = name
        self.benchmarks: List[TabularBenchmark] = []
        self.results: List[BenchmarkResult] = []
    
    def add_benchmark(self, benchmark: TabularBenchmark):
        """Add a benchmark to the suite."""
        self.benchmarks.append(benchmark)
    
    def run_all(
        self,
        models: Dict[str, Any],
        **kwargs
    ) -> List[BenchmarkResult]:
        """Run all benchmarks in the suite."""
        logger.info(f"Running benchmark suite: {self.name}")
        
        all_results = []
        for benchmark in self.benchmarks:
            logger.info(f"Running benchmark: {benchmark.name}")
            results = benchmark.run_benchmark(models, **kwargs)
            all_results.extend(results)
            self.results.extend(results)
        
        return all_results
    
    def get_summary(self) -> pd.DataFrame:
        """Get summary of all benchmark results."""
        if not self.results:
            return pd.DataFrame()
        
        # Group by model and compute average metrics
        model_results = {}
        for result in self.results:
            model_name = result.model_name
            if model_name not in model_results:
                model_results[model_name] = {
                    'datasets': [],
                    'metrics': {},
                    'training_times': [],
                    'inference_times': [],
                    'model_sizes': [],
                    'memory_usages': []
                }
            
            model_results[model_name]['datasets'].append(result.dataset_name)
            model_results[model_name]['training_times'].append(result.training_time)
            model_results[model_name]['inference_times'].append(result.inference_time)
            model_results[model_name]['model_sizes'].append(result.model_size)
            model_results[model_name]['memory_usages'].append(result.memory_usage)
            
            # Aggregate metrics
            for metric, value in result.metrics.items():
                if metric not in model_results[model_name]['metrics']:
                    model_results[model_name]['metrics'][metric] = []
                model_results[model_name]['metrics'][metric].append(value)
        
        # Create summary DataFrame
        summary_data = []
        for model_name, data in model_results.items():
            row = {
                'Model': model_name,
                'Datasets': len(data['datasets']),
                'Avg Training Time (s)': np.mean(data['training_times']),
                'Avg Inference Time (s)': np.mean(data['inference_times']),
                'Avg Model Size (MB)': np.mean(data['model_sizes']) / (1024 * 1024),
                'Avg Memory Usage (MB)': np.mean(data['memory_usages']) / (1024 * 1024)
            }
            
            # Add average metrics
            for metric, values in data['metrics'].items():
                row[f'Avg {metric}'] = np.mean(values)
                row[f'Std {metric}'] = np.std(values)
            
            summary_data.append(row)
        
        return pd.DataFrame(summary_data)
    
    def save_results(self, filepath: str):
        """Save benchmark results to file."""
        results_data = [result.to_dict() for result in self.results]
        
        if filepath.endswith('.json'):
            with open(filepath, 'w') as f:
                json.dump(results_data, f, indent=2)
        elif filepath.endswith('.csv'):
            df = pd.DataFrame(results_data)
            df.to_csv(filepath, index=False)
        else:
            raise ValueError("Filepath must end with .json or .csv")
        
        logger.info(f"Benchmark results saved to {filepath}")
    
    def load_results(self, filepath: str):
        """Load benchmark results from file."""
        if filepath.endswith('.json'):
            with open(filepath, 'r') as f:
                results_data = json.load(f)
        elif filepath.endswith('.csv'):
            df = pd.read_csv(filepath)
            results_data = df.to_dict('records')
        else:
            raise ValueError("Filepath must end with .json or .csv")
        
        self.results = [BenchmarkResult.from_dict(data) for data in results_data]
        logger.info(f"Benchmark results loaded from {filepath}")


def run_benchmark_suite(
    benchmarks: List[TabularBenchmark],
    models: Dict[str, Any],
    suite_name: str = "Custom Benchmark Suite",
    **kwargs
) -> BenchmarkSuite:
    """Run a custom benchmark suite."""
    suite = BenchmarkSuite(suite_name)
    
    for benchmark in benchmarks:
        suite.add_benchmark(benchmark)
    
    suite.run_all(models, **kwargs)
    
    return suite