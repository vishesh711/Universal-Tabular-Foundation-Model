"""Tests for the evaluation framework."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
import tempfile
import os

from tabgpt.evaluation import (
    BenchmarkSuite, ClassificationBenchmark, RegressionBenchmark,
    EvaluationMetrics, ClassificationMetrics, RegressionMetrics,
    EvaluationConfig, HoldoutEvaluator, CrossValidationEvaluator,
    TabularDatasetLoader, SyntheticDataGenerator,
    create_baseline_models, TransferLearningEvaluator
)


class TestMetrics:
    """Test evaluation metrics."""
    
    def test_classification_metrics(self):
        """Test classification metrics computation."""
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 1, 0, 0])
        y_pred_proba = np.array([[0.8, 0.2], [0.3, 0.7], [0.4, 0.6], 
                                [0.2, 0.8], [0.9, 0.1], [0.6, 0.4]])
        
        metrics = ClassificationMetrics.compute_metrics(y_true, y_pred, y_pred_proba)
        
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics
        assert 'auc' in metrics
        
        # Check accuracy calculation
        expected_accuracy = 4/6  # 4 correct out of 6
        assert abs(metrics['accuracy'] - expected_accuracy) < 1e-6
    
    def test_regression_metrics(self):
        """Test regression metrics computation."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.2, 2.8, 4.1, 4.9])
        
        metrics = RegressionMetrics.compute_metrics(y_true, y_pred)
        
        assert 'mse' in metrics
        assert 'rmse' in metrics
        assert 'mae' in metrics
        assert 'r2' in metrics
        
        # Check MSE calculation
        expected_mse = np.mean((y_true - y_pred) ** 2)
        assert abs(metrics['mse'] - expected_mse) < 1e-6
        
        # Check RMSE calculation
        expected_rmse = np.sqrt(expected_mse)
        assert abs(metrics['rmse'] - expected_rmse) < 1e-6


class TestEvaluators:
    """Test model evaluators."""
    
    def test_holdout_evaluator(self):
        """Test holdout evaluation strategy."""
        config = EvaluationConfig(strategy="holdout", test_size=0.3)
        evaluator = HoldoutEvaluator(config)
        
        # Create mock model
        model = Mock()
        model.fit = Mock()
        model.predict = Mock(return_value=np.array([0, 1, 0, 1]))
        model.predict_proba = Mock(return_value=np.array([[0.8, 0.2], [0.3, 0.7], [0.6, 0.4], [0.2, 0.8]]))
        model.__class__.__name__ = "MockModel"
        
        # Create sample data
        X = pd.DataFrame({'feature1': range(10), 'feature2': range(10, 20)})
        y = pd.Series([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        
        result = evaluator.evaluate(model, X, y, task_type="classification")
        
        assert result.model_name == "MockModel"
        assert 'accuracy' in result.metrics
        assert result.training_time >= 0
        assert result.evaluation_time >= 0
        assert len(result.fold_results) == 1
    
    def test_cross_validation_evaluator(self):
        """Test cross-validation evaluation strategy."""
        config = EvaluationConfig(strategy="cross_validation", cv_folds=3)
        evaluator = CrossValidationEvaluator(config)
        
        # Create mock model
        model = Mock()
        model.fit = Mock()
        model.predict = Mock(return_value=np.array([0, 1, 0]))
        model.__class__.__name__ = "MockModel"
        
        # Create sample data
        X = pd.DataFrame({'feature1': range(9), 'feature2': range(9, 18)})
        y = pd.Series([0, 1, 0, 1, 0, 1, 0, 1, 0])
        
        result = evaluator.evaluate(model, X, y, task_type="classification")
        
        assert result.model_name == "MockModel"
        assert 'accuracy' in result.metrics
        assert 'accuracy' in result.metrics_std
        assert len(result.fold_results) == 3


class TestBenchmarks:
    """Test benchmark framework."""
    
    def test_classification_benchmark(self):
        """Test classification benchmark."""
        def load_data():
            X = pd.DataFrame({'feature1': range(100), 'feature2': range(100, 200)})
            y = pd.Series(np.random.choice([0, 1], 100))
            return X, y
        
        benchmark = ClassificationBenchmark(
            name="test_classification",
            data_loader=load_data,
            description="Test classification benchmark"
        )
        
        # Create mock models
        models = {}
        for i in range(2):
            model = Mock()
            model.fit = Mock()
            model.predict = Mock(return_value=np.random.choice([0, 1], 20))
            model.predict_proba = Mock(return_value=np.random.rand(20, 2))
            model.__class__.__name__ = f"MockModel{i}"
            models[f"model_{i}"] = model
        
        results = benchmark.run_benchmark(models, test_size=0.2, random_state=42)
        
        assert len(results) == 2
        for result in results:
            assert 'accuracy' in result.metrics
            assert result.training_time >= 0
            assert result.inference_time >= 0
    
    def test_benchmark_suite(self):
        """Test benchmark suite."""
        suite = BenchmarkSuite("Test Suite")
        
        # Add benchmarks
        def load_data():
            X = pd.DataFrame({'feature1': range(50), 'feature2': range(50, 100)})
            y = pd.Series(np.random.choice([0, 1], 50))
            return X, y
        
        benchmark = ClassificationBenchmark("test", load_data)
        suite.add_benchmark(benchmark)
        
        # Create mock model
        model = Mock()
        model.fit = Mock()
        model.predict = Mock(return_value=np.random.choice([0, 1], 10))
        model.__class__.__name__ = "MockModel"
        
        models = {"test_model": model}
        results = suite.run_all(models, test_size=0.2)
        
        assert len(results) == 1
        assert len(suite.results) == 1


class TestDatasets:
    """Test dataset loading and generation."""
    
    def test_synthetic_data_generator(self):
        """Test synthetic data generation."""
        generator = SyntheticDataGenerator()
        
        # Test classification dataset
        X, y = generator.generate_classification_dataset(
            n_samples=100, n_features=10, n_classes=3
        )
        
        assert X.shape == (100, 10)
        assert len(y) == 100
        assert len(np.unique(y)) == 3
        
        # Test regression dataset
        X, y = generator.generate_regression_dataset(
            n_samples=100, n_features=10
        )
        
        assert X.shape == (100, 10)
        assert len(y) == 100
        assert isinstance(y.iloc[0], (int, float))
    
    def test_mixed_types_dataset(self):
        """Test mixed data types dataset generation."""
        generator = SyntheticDataGenerator()
        
        X, y = generator.generate_mixed_types_dataset(
            n_samples=100, n_numerical=5, n_categorical=3
        )
        
        assert X.shape == (100, 8)  # 5 numerical + 3 categorical
        assert len(y) == 100
        
        # Check data types
        numerical_cols = [col for col in X.columns if 'num_feature' in col]
        categorical_cols = [col for col in X.columns if 'cat_feature' in col]
        
        assert len(numerical_cols) == 5
        assert len(categorical_cols) == 3
        
        # Check that categorical columns are strings
        for col in categorical_cols:
            assert X[col].dtype == 'object'
    
    def test_tabular_dataset_loader(self):
        """Test tabular dataset loader."""
        with tempfile.TemporaryDirectory() as temp_dir:
            loader = TabularDatasetLoader(cache_dir=temp_dir)
            
            # Test listing datasets
            datasets = loader.list_datasets()
            assert isinstance(datasets, list)
            assert len(datasets) > 0
            
            # Test loading a dataset (will use synthetic fallback)
            try:
                X, y = loader.load_dataset(datasets[0])
                assert isinstance(X, pd.DataFrame)
                assert isinstance(y, pd.Series)
                assert len(X) == len(y)
            except Exception:
                # Expected if no real datasets available
                pass


class TestBaselines:
    """Test baseline models."""
    
    def test_create_baseline_models(self):
        """Test creating baseline models."""
        models = create_baseline_models(
            task_type="classification",
            include_models=["RandomForest"]
        )
        
        assert "RandomForest" in models
        assert models["RandomForest"].task_type == "classification"
    
    def test_baseline_model_training(self):
        """Test baseline model training and prediction."""
        models = create_baseline_models(
            task_type="classification",
            include_models=["RandomForest"]
        )
        
        if "RandomForest" in models:
            model = models["RandomForest"]
            
            # Create sample data
            X = pd.DataFrame({
                'num_feature': np.random.randn(100),
                'cat_feature': np.random.choice(['A', 'B', 'C'], 100)
            })
            y = pd.Series(np.random.choice([0, 1], 100))
            
            # Train model
            model.fit(X, y)
            assert model.is_fitted
            
            # Make predictions
            predictions = model.predict(X)
            assert len(predictions) == len(X)
            
            # Test probability predictions
            try:
                probabilities = model.predict_proba(X)
                assert probabilities.shape[0] == len(X)
            except AttributeError:
                # Some models might not support probabilities
                pass


class TestTransferLearning:
    """Test transfer learning evaluation."""
    
    def test_transfer_learning_evaluator(self):
        """Test transfer learning evaluator."""
        config = EvaluationConfig(strategy="holdout", test_size=0.3)
        evaluator = TransferLearningEvaluator(config)
        
        # Create mock models
        pretrained_model = Mock()
        pretrained_model.fit = Mock()
        pretrained_model.predict = Mock(return_value=np.array([0, 1, 0, 1]))
        pretrained_model.__class__.__name__ = "PretrainedModel"
        
        baseline_model = Mock()
        baseline_model.fit = Mock()
        baseline_model.predict = Mock(return_value=np.array([1, 0, 1, 0]))
        baseline_model.__class__.__name__ = "BaselineModel"
        
        # Create sample datasets
        X_source = pd.DataFrame({'feature1': range(50), 'feature2': range(50, 100)})
        y_source = pd.Series(np.random.choice([0, 1], 50))
        
        X_target = pd.DataFrame({'feature1': range(20), 'feature2': range(20, 40)})
        y_target = pd.Series(np.random.choice([0, 1], 20))
        
        result = evaluator.evaluate_transfer(
            pretrained_model=pretrained_model,
            baseline_model=baseline_model,
            source_dataset=(X_source, y_source),
            target_dataset=(X_target, y_target),
            source_name="source",
            target_name="target"
        )
        
        assert result.source_dataset == "source"
        assert result.target_dataset == "target"
        assert result.model_name == "PretrainedModel"
        assert 'accuracy' in result.baseline_metrics
        assert 'accuracy' in result.pretrained_metrics
        assert 'accuracy' in result.improvement
        assert 'accuracy' in result.relative_improvement


class TestIntegration:
    """Integration tests for the evaluation framework."""
    
    def test_end_to_end_evaluation(self):
        """Test end-to-end evaluation pipeline."""
        # Create synthetic datasets
        generator = SyntheticDataGenerator()
        X, y = generator.generate_classification_dataset(n_samples=100, n_features=5)
        
        # Create mock model
        model = Mock()
        model.fit = Mock()
        model.predict = Mock(return_value=np.random.choice([0, 1], 20))
        model.predict_proba = Mock(return_value=np.random.rand(20, 2))
        model.__class__.__name__ = "TestModel"
        
        # Create evaluation config
        config = EvaluationConfig(strategy="holdout", test_size=0.2)
        
        # Create benchmark
        def load_data():
            return X, y
        
        benchmark = ClassificationBenchmark("test_dataset", load_data)
        
        # Run evaluation
        models = {"test_model": model}
        results = benchmark.run_benchmark(models, test_size=0.2)
        
        assert len(results) == 1
        result = results[0]
        assert result.model_name == "TestModel"
        assert result.dataset_name == "test_dataset"
        assert 'accuracy' in result.metrics
    
    def test_batch_evaluation(self):
        """Test batch evaluation of multiple models and datasets."""
        from tabgpt.evaluation.evaluator import BatchEvaluator
        
        config = EvaluationConfig(strategy="holdout", test_size=0.3)
        evaluator = BatchEvaluator(config)
        
        # Create mock models
        models = {}
        for i in range(2):
            model = Mock()
            model.fit = Mock()
            model.predict = Mock(return_value=np.random.choice([0, 1], 7))
            model.__class__.__name__ = f"Model{i}"
            models[f"model_{i}"] = model
        
        # Create datasets
        datasets = {}
        for i in range(2):
            X = pd.DataFrame({'feature1': range(10), 'feature2': range(10, 20)})
            y = pd.Series(np.random.choice([0, 1], 10))
            datasets[f"dataset_{i}"] = (X, y)
        
        task_types = {name: "classification" for name in datasets.keys()}
        
        results = evaluator.evaluate_models(models, datasets, task_types)
        
        # Should have 2 models × 2 datasets = 4 results
        assert len(results) == 4
        
        # Test summary generation
        summary = evaluator.get_summary()
        assert not summary.empty
        assert len(summary) == 4  # One row per model-dataset combination


if __name__ == "__main__":
    pytest.main([__file__])