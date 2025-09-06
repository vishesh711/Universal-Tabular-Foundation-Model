"""
Example: Comprehensive TabGPT Evaluation

This example demonstrates how to use the TabGPT evaluation framework
to benchmark models, compare with baselines, and evaluate transfer learning.
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tabgpt.evaluation import (
    BenchmarkSuite, ClassificationBenchmark, RegressionBenchmark,
    SyntheticDataGenerator, create_baseline_models,
    EvaluationConfig, BatchEvaluator,
    TransferLearningEvaluator, PretrainingBenefitAnalyzer,
    compute_all_metrics
)


def create_mock_tabgpt_model(task_type="classification"):
    """Create a mock TabGPT model for demonstration."""
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    
    class MockTabGPTModel:
        def __init__(self, task_type):
            self.task_type = task_type
            if task_type == "classification":
                self.model = RandomForestClassifier(n_estimators=50, random_state=42)
            else:
                self.model = RandomForestRegressor(n_estimators=50, random_state=42)
        
        def fit(self, X, y):
            return self.model.fit(X, y)
        
        def predict(self, X):
            return self.model.predict(X)
        
        def predict_proba(self, X):
            if hasattr(self.model, 'predict_proba'):
                return self.model.predict_proba(X)
            else:
                raise AttributeError("Model does not support probability predictions")
        
        @property
        def __class__(self):
            class MockClass:
                __name__ = "TabGPT"
            return MockClass()
    
    return MockTabGPTModel(task_type)


def demonstrate_metrics_computation():
    """Demonstrate metrics computation."""
    print("="*60)
    print("METRICS COMPUTATION DEMONSTRATION")
    print("="*60)
    
    # Generate sample predictions
    np.random.seed(42)
    y_true = np.random.choice([0, 1], 100)
    y_pred = np.random.choice([0, 1], 100)
    y_pred_proba = np.random.rand(100, 2)
    y_pred_proba = y_pred_proba / y_pred_proba.sum(axis=1, keepdims=True)  # Normalize
    
    # Compute classification metrics
    print("\nClassification Metrics:")
    metrics = compute_all_metrics(y_true, y_pred, y_pred_proba, "classification")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Generate regression data
    y_true_reg = np.random.randn(100)
    y_pred_reg = y_true_reg + np.random.randn(100) * 0.1  # Add some noise
    
    print("\nRegression Metrics:")
    metrics_reg = compute_all_metrics(y_true_reg, y_pred_reg, task_type="regression")
    for metric, value in metrics_reg.items():
        print(f"  {metric}: {value:.4f}")


def demonstrate_synthetic_data_generation():
    """Demonstrate synthetic data generation."""
    print("\n" + "="*60)
    print("SYNTHETIC DATA GENERATION DEMONSTRATION")
    print("="*60)
    
    generator = SyntheticDataGenerator()
    
    # Classification dataset
    print("\nGenerating Classification Dataset:")
    X_cls, y_cls = generator.generate_classification_dataset(
        n_samples=1000, n_features=10, n_classes=3, random_state=42
    )
    print(f"  Shape: X={X_cls.shape}, y={y_cls.shape}")
    print(f"  Classes: {np.unique(y_cls)}")
    print(f"  Class distribution: {pd.Series(y_cls).value_counts().to_dict()}")
    
    # Regression dataset
    print("\nGenerating Regression Dataset:")
    X_reg, y_reg = generator.generate_regression_dataset(
        n_samples=1000, n_features=10, noise=0.1, random_state=42
    )
    print(f"  Shape: X={X_reg.shape}, y={y_reg.shape}")
    print(f"  Target range: [{y_reg.min():.2f}, {y_reg.max():.2f}]")
    print(f"  Target mean: {y_reg.mean():.2f}, std: {y_reg.std():.2f}")
    
    # Mixed types dataset
    print("\nGenerating Mixed Types Dataset:")
    X_mixed, y_mixed = generator.generate_mixed_types_dataset(
        n_samples=1000, n_numerical=5, n_categorical=3, random_state=42
    )
    print(f"  Shape: X={X_mixed.shape}, y={y_mixed.shape}")
    print(f"  Data types: {X_mixed.dtypes.to_dict()}")
    
    return {
        'classification': (X_cls, y_cls),
        'regression': (X_reg, y_reg),
        'mixed_types': (X_mixed, y_mixed)
    }


def demonstrate_baseline_models():
    """Demonstrate baseline model creation and usage."""
    print("\n" + "="*60)
    print("BASELINE MODELS DEMONSTRATION")
    print("="*60)
    
    # Create classification baselines
    print("\nCreating Classification Baselines:")
    cls_models = create_baseline_models(
        task_type="classification",
        include_models=["RandomForest", "LogisticRegression"]
    )
    print(f"  Available models: {list(cls_models.keys())}")
    
    # Create regression baselines
    print("\nCreating Regression Baselines:")
    reg_models = create_baseline_models(
        task_type="regression",
        include_models=["RandomForest", "LinearRegression"]
    )
    print(f"  Available models: {list(reg_models.keys())}")
    
    # Test model training
    generator = SyntheticDataGenerator()
    X, y = generator.generate_mixed_types_dataset(n_samples=200, random_state=42)
    
    print(f"\nTesting model training on dataset: X={X.shape}, y={y.shape}")
    
    for name, model in cls_models.items():
        try:
            model.fit(X, y)
            predictions = model.predict(X[:10])
            print(f"  {name}: Training successful, predictions shape: {predictions.shape}")
        except Exception as e:
            print(f"  {name}: Training failed - {e}")
    
    return cls_models, reg_models


def demonstrate_benchmark_evaluation():
    """Demonstrate benchmark evaluation."""
    print("\n" + "="*60)
    print("BENCHMARK EVALUATION DEMONSTRATION")
    print("="*60)
    
    # Create synthetic datasets
    generator = SyntheticDataGenerator()
    
    # Create benchmark datasets
    datasets = {}
    task_types = {}
    
    # Classification datasets
    for i, (n_samples, n_features) in enumerate([(500, 10), (1000, 15)]):
        X, y = generator.generate_classification_dataset(
            n_samples=n_samples, n_features=n_features, n_classes=2, random_state=42+i
        )
        dataset_name = f"classification_{i+1}"
        datasets[dataset_name] = (X, y)
        task_types[dataset_name] = "classification"
    
    # Regression datasets
    for i, (n_samples, n_features) in enumerate([(500, 8), (1000, 12)]):
        X, y = generator.generate_regression_dataset(
            n_samples=n_samples, n_features=n_features, noise=0.1, random_state=42+i
        )
        dataset_name = f"regression_{i+1}"
        datasets[dataset_name] = (X, y)
        task_types[dataset_name] = "regression"
    
    print(f"Created {len(datasets)} benchmark datasets:")
    for name, (X, y) in datasets.items():
        print(f"  {name}: X={X.shape}, y={y.shape}, task={task_types[name]}")
    
    # Create models to evaluate
    models = {}
    
    # Add TabGPT models
    models["TabGPT_Classification"] = create_mock_tabgpt_model("classification")
    models["TabGPT_Regression"] = create_mock_tabgpt_model("regression")
    
    # Add baseline models
    cls_baselines = create_baseline_models("classification", ["RandomForest"])
    reg_baselines = create_baseline_models("regression", ["RandomForest"])
    models.update(cls_baselines)
    models.update(reg_baselines)
    
    print(f"\nEvaluating {len(models)} models: {list(models.keys())}")
    
    # Run batch evaluation
    config = EvaluationConfig(strategy="holdout", test_size=0.2, verbose=True)
    evaluator = BatchEvaluator(config)
    
    results = evaluator.evaluate_models(models, datasets, task_types)
    
    print(f"\nEvaluation completed! Generated {len(results)} results.")
    
    # Get summary
    summary = evaluator.get_summary()
    print("\nEvaluation Summary:")
    print(summary.to_string(index=False))
    
    return summary, results


def demonstrate_transfer_learning():
    """Demonstrate transfer learning evaluation."""
    print("\n" + "="*60)
    print("TRANSFER LEARNING DEMONSTRATION")
    print("="*60)
    
    generator = SyntheticDataGenerator()
    
    # Create source and target datasets
    print("Creating source and target datasets...")
    
    # Source dataset (larger)
    X_source, y_source = generator.generate_classification_dataset(
        n_samples=2000, n_features=15, n_classes=2, class_sep=1.0, random_state=42
    )
    
    # Target dataset (smaller, similar distribution)
    X_target, y_target = generator.generate_classification_dataset(
        n_samples=500, n_features=15, n_classes=2, class_sep=0.8, random_state=43
    )
    
    print(f"  Source dataset: X={X_source.shape}, y={y_source.shape}")
    print(f"  Target dataset: X={X_target.shape}, y={y_target.shape}")
    
    # Create models
    pretrained_model = create_mock_tabgpt_model("classification")
    baseline_model = create_mock_tabgpt_model("classification")
    
    # Evaluate transfer learning
    config = EvaluationConfig(strategy="holdout", test_size=0.3)
    evaluator = TransferLearningEvaluator(config)
    
    print("\nEvaluating transfer learning...")
    result = evaluator.evaluate_transfer(
        pretrained_model=pretrained_model,
        baseline_model=baseline_model,
        source_dataset=(X_source, y_source),
        target_dataset=(X_target, y_target),
        source_name="large_source",
        target_name="small_target"
    )
    
    print("\nTransfer Learning Results:")
    print(f"  Source → Target: {result.source_dataset} → {result.target_dataset}")
    print(f"  Baseline accuracy: {result.baseline_metrics.get('accuracy', 0):.4f}")
    print(f"  Pre-trained accuracy: {result.pretrained_metrics.get('accuracy', 0):.4f}")
    print(f"  Improvement: {result.improvement.get('accuracy', 0):.4f}")
    print(f"  Relative improvement: {result.relative_improvement.get('accuracy', 0):.2f}%")
    
    return result


def demonstrate_pretraining_benefits():
    """Demonstrate pre-training benefit analysis."""
    print("\n" + "="*60)
    print("PRE-TRAINING BENEFIT ANALYSIS DEMONSTRATION")
    print("="*60)
    
    generator = SyntheticDataGenerator()
    
    # Create source dataset
    X_source, y_source = generator.generate_classification_dataset(
        n_samples=3000, n_features=20, n_classes=2, random_state=42
    )
    
    # Create target dataset
    X_target, y_target = generator.generate_classification_dataset(
        n_samples=1000, n_features=20, n_classes=2, random_state=43
    )
    
    print(f"Source dataset: X={X_source.shape}, y={y_source.shape}")
    print(f"Target dataset: X={X_target.shape}, y={y_target.shape}")
    
    # Create models
    pretrained_model = create_mock_tabgpt_model("classification")
    baseline_model = create_mock_tabgpt_model("classification")
    
    # Analyze pre-training benefits
    analyzer = PretrainingBenefitAnalyzer()
    
    print("\nAnalyzing dataset size effect...")
    size_results = analyzer.analyze_dataset_size_effect(
        pretrained_model=pretrained_model,
        baseline_model=baseline_model,
        source_dataset=(X_source, y_source),
        target_dataset=(X_target, y_target),
        target_sizes=[100, 200, 500, 800]
    )
    
    print(f"Analyzed {len(size_results)} different target sizes")
    
    # Get benefit summary
    benefit_summary = analyzer.get_benefit_summary()
    
    if not benefit_summary.empty:
        print("\nPre-training Benefit Summary:")
        print(benefit_summary.to_string(index=False))
    
    return benefit_summary


def main():
    """Run all evaluation demonstrations."""
    print("TabGPT Evaluation Framework Demonstration")
    print("="*60)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    try:
        # 1. Metrics computation
        demonstrate_metrics_computation()
        
        # 2. Synthetic data generation
        datasets = demonstrate_synthetic_data_generation()
        
        # 3. Baseline models
        cls_models, reg_models = demonstrate_baseline_models()
        
        # 4. Benchmark evaluation
        summary, results = demonstrate_benchmark_evaluation()
        
        # 5. Transfer learning
        transfer_result = demonstrate_transfer_learning()
        
        # 6. Pre-training benefits
        benefit_summary = demonstrate_pretraining_benefits()
        
        print("\n" + "="*60)
        print("DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nKey Results:")
        print(f"  • Generated {len(datasets)} synthetic datasets")
        print(f"  • Created {len(cls_models) + len(reg_models)} baseline models")
        print(f"  • Ran {len(results)} benchmark evaluations")
        print(f"  • Evaluated transfer learning with {transfer_result.relative_improvement.get('accuracy', 0):.2f}% improvement")
        print(f"  • Analyzed pre-training benefits across different dataset sizes")
        
        print("\nThe evaluation framework is ready for use!")
        
    except Exception as e:
        print(f"\nError during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()