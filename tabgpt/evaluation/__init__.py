"""Evaluation and benchmarking framework for TabGPT."""

from .benchmarks import (
    TabularBenchmark,
    ClassificationBenchmark,
    RegressionBenchmark,
    AnomalyDetectionBenchmark,
    BenchmarkSuite,
    BenchmarkResult,
    run_benchmark_suite
)
from .metrics import (
    EvaluationMetrics,
    ClassificationMetrics,
    RegressionMetrics,
    PretrainingMetrics,
    compute_all_metrics,
    compare_models
)
from .evaluator import (
    ModelEvaluator,
    CrossValidationEvaluator,
    HoldoutEvaluator,
    TimeSeriesEvaluator,
    EvaluationConfig,
    EvaluationResult,
    BatchEvaluator
)
from .datasets import (
    BenchmarkDataset,
    TabularDatasetLoader,
    SyntheticDataGenerator,
    load_benchmark_datasets,
    create_synthetic_dataset
)
from .baselines import (
    BaselineModel,
    RandomForestBaseline,
    XGBoostBaseline,
    LogisticRegressionBaseline,
    LinearRegressionBaseline,
    create_baseline_models
)
from .transfer_learning import (
    TransferLearningEvaluator,
    PretrainingBenefitAnalyzer,
    DomainAdaptationEvaluator,
    evaluate_transfer_learning
)

__all__ = [
    # Benchmarks
    "TabularBenchmark",
    "ClassificationBenchmark", 
    "RegressionBenchmark",
    "AnomalyDetectionBenchmark",
    "BenchmarkSuite",
    "BenchmarkResult",
    "run_benchmark_suite",
    
    # Metrics
    "EvaluationMetrics",
    "ClassificationMetrics",
    "RegressionMetrics", 
    "PretrainingMetrics",
    "compute_all_metrics",
    "compare_models",
    
    # Evaluators
    "ModelEvaluator",
    "CrossValidationEvaluator",
    "HoldoutEvaluator",
    "TimeSeriesEvaluator",
    "EvaluationConfig",
    "EvaluationResult",
    "BatchEvaluator",
    
    # Datasets
    "BenchmarkDataset",
    "TabularDatasetLoader",
    "SyntheticDataGenerator",
    "load_benchmark_datasets",
    "create_synthetic_dataset",
    
    # Baselines
    "BaselineModel",
    "RandomForestBaseline",
    "XGBoostBaseline",
    "LogisticRegressionBaseline",
    "LinearRegressionBaseline",
    "create_baseline_models",
    
    # Transfer Learning
    "TransferLearningEvaluator",
    "PretrainingBenefitAnalyzer",
    "DomainAdaptationEvaluator",
    "evaluate_transfer_learning"
]