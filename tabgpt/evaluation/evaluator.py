"""Model evaluation framework with different evaluation strategies."""

import time
import logging
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import warnings

import torch
import torch.nn as nn
import pandas as pd
import numpy as np

try:
    from sklearn.model_selection import (
        cross_val_score, StratifiedKFold, KFold, 
        TimeSeriesSplit, train_test_split
    )
    from sklearn.metrics import make_scorer
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from .metrics import EvaluationMetrics, compute_all_metrics, MetricTracker

logger = logging.getLogger(__name__)


@dataclass
class EvaluationConfig:
    """Configuration for model evaluation."""
    
    # Evaluation strategy
    strategy: str = "holdout"  # holdout, cross_validation, time_series
    
    # Cross-validation settings
    cv_folds: int = 5
    cv_shuffle: bool = True
    cv_random_state: int = 42
    
    # Holdout settings
    test_size: float = 0.2
    validation_size: float = 0.2
    
    # Time series settings
    n_splits: int = 5
    max_train_size: Optional[int] = None
    
    # Metrics settings
    primary_metric: str = "accuracy"
    compute_all_metrics: bool = True
    
    # Performance settings
    n_jobs: int = 1
    verbose: bool = True
    
    # Statistical testing
    statistical_test: bool = True
    confidence_level: float = 0.95


@dataclass
class EvaluationResult:
    """Result of model evaluation."""
    
    model_name: str
    dataset_name: str
    config: EvaluationConfig
    metrics: Dict[str, float]
    metrics_std: Dict[str, float] = field(default_factory=dict)
    fold_results: List[Dict[str, float]] = field(default_factory=list)
    training_time: float = 0.0
    evaluation_time: float = 0.0
    additional_info: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'model_name': self.model_name,
            'dataset_name': self.dataset_name,
            'config': self.config.__dict__,
            'metrics': self.metrics,
            'metrics_std': self.metrics_std,
            'fold_results': self.fold_results,
            'training_time': self.training_time,
            'evaluation_time': self.evaluation_time,
            'additional_info': self.additional_info
        }


class ModelEvaluator(ABC):
    """Abstract base class for model evaluators."""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.metric_tracker = MetricTracker()
    
    @abstractmethod
    def evaluate(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series,
        **kwargs
    ) -> EvaluationResult:
        """Evaluate a model."""
        pass
    
    def _fit_and_predict(
        self,
        model: Any,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        **kwargs
    ) -> Tuple[np.ndarray, Optional[np.ndarray], float]:
        """Fit model and make predictions."""
        start_time = time.time()
        
        # Fit model
        if hasattr(model, 'fit'):
            model.fit(X_train, y_train)
        
        training_time = time.time() - start_time
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Get prediction probabilities if available
        y_pred_proba = None
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)
        
        return y_pred, y_pred_proba, training_time
    
    def _compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: Optional[np.ndarray] = None,
        task_type: str = "classification"
    ) -> Dict[str, float]:
        """Compute evaluation metrics."""
        if self.config.compute_all_metrics:
            return compute_all_metrics(y_true, y_pred, y_pred_proba, task_type)
        else:
            # Compute only primary metric
            if task_type == "classification":
                return {self.config.primary_metric: float(np.mean(y_true == y_pred))}
            else:
                return {self.config.primary_metric: float(np.mean((y_true - y_pred) ** 2))}


class HoldoutEvaluator(ModelEvaluator):
    """Holdout evaluation strategy."""
    
    def evaluate(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series,
        task_type: str = "classification",
        **kwargs
    ) -> EvaluationResult:
        """Evaluate model using holdout strategy."""
        start_time = time.time()
        
        # Split data
        stratify = y if task_type == "classification" else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.config.test_size,
            random_state=self.config.cv_random_state,
            stratify=stratify
        )
        
        # Fit and predict
        y_pred, y_pred_proba, training_time = self._fit_and_predict(
            model, X_train, y_train, X_test, **kwargs
        )
        
        # Compute metrics
        metrics = self._compute_metrics(y_test, y_pred, y_pred_proba, task_type)
        
        evaluation_time = time.time() - start_time
        
        return EvaluationResult(
            model_name=model.__class__.__name__,
            dataset_name=kwargs.get('dataset_name', 'Unknown'),
            config=self.config,
            metrics=metrics,
            training_time=training_time,
            evaluation_time=evaluation_time,
            fold_results=[metrics]
        )


class CrossValidationEvaluator(ModelEvaluator):
    """Cross-validation evaluation strategy."""
    
    def evaluate(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series,
        task_type: str = "classification",
        **kwargs
    ) -> EvaluationResult:
        """Evaluate model using cross-validation."""
        if not SKLEARN_AVAILABLE:
            raise ImportError("sklearn required for cross-validation")
        
        start_time = time.time()
        
        # Create cross-validation splitter
        if task_type == "classification":
            cv = StratifiedKFold(
                n_splits=self.config.cv_folds,
                shuffle=self.config.cv_shuffle,
                random_state=self.config.cv_random_state
            )
        else:
            cv = KFold(
                n_splits=self.config.cv_folds,
                shuffle=self.config.cv_shuffle,
                random_state=self.config.cv_random_state
            )
        
        # Perform cross-validation
        fold_results = []
        total_training_time = 0.0
        
        for fold, (train_idx, test_idx) in enumerate(cv.split(X, y)):
            if self.config.verbose:
                logger.info(f"Evaluating fold {fold + 1}/{self.config.cv_folds}")
            
            # Split data
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Create a copy of the model for this fold
            import copy
            fold_model = copy.deepcopy(model)
            
            # Fit and predict
            y_pred, y_pred_proba, training_time = self._fit_and_predict(
                fold_model, X_train, y_train, X_test, **kwargs
            )
            
            total_training_time += training_time
            
            # Compute metrics
            fold_metrics = self._compute_metrics(y_test, y_pred, y_pred_proba, task_type)
            fold_results.append(fold_metrics)
        
        # Aggregate results
        metrics = {}
        metrics_std = {}
        
        # Get all metric names
        all_metrics = set()
        for fold_result in fold_results:
            all_metrics.update(fold_result.keys())
        
        # Compute mean and std for each metric
        for metric_name in all_metrics:
            values = [fold_result.get(metric_name, 0.0) for fold_result in fold_results]
            metrics[metric_name] = float(np.mean(values))
            metrics_std[metric_name] = float(np.std(values))
        
        evaluation_time = time.time() - start_time
        
        return EvaluationResult(
            model_name=model.__class__.__name__,
            dataset_name=kwargs.get('dataset_name', 'Unknown'),
            config=self.config,
            metrics=metrics,
            metrics_std=metrics_std,
            fold_results=fold_results,
            training_time=total_training_time,
            evaluation_time=evaluation_time
        )


class TimeSeriesEvaluator(ModelEvaluator):
    """Time series evaluation strategy."""
    
    def evaluate(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series,
        task_type: str = "regression",
        **kwargs
    ) -> EvaluationResult:
        """Evaluate model using time series split."""
        if not SKLEARN_AVAILABLE:
            raise ImportError("sklearn required for time series evaluation")
        
        start_time = time.time()
        
        # Create time series splitter
        tscv = TimeSeriesSplit(
            n_splits=self.config.n_splits,
            max_train_size=self.config.max_train_size
        )
        
        # Perform time series cross-validation
        fold_results = []
        total_training_time = 0.0
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            if self.config.verbose:
                logger.info(f"Evaluating time series fold {fold + 1}/{self.config.n_splits}")
            
            # Split data
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Create a copy of the model for this fold
            import copy
            fold_model = copy.deepcopy(model)
            
            # Fit and predict
            y_pred, y_pred_proba, training_time = self._fit_and_predict(
                fold_model, X_train, y_train, X_test, **kwargs
            )
            
            total_training_time += training_time
            
            # Compute metrics
            fold_metrics = self._compute_metrics(y_test, y_pred, y_pred_proba, task_type)
            fold_results.append(fold_metrics)
        
        # Aggregate results
        metrics = {}
        metrics_std = {}
        
        # Get all metric names
        all_metrics = set()
        for fold_result in fold_results:
            all_metrics.update(fold_result.keys())
        
        # Compute mean and std for each metric
        for metric_name in all_metrics:
            values = [fold_result.get(metric_name, 0.0) for fold_result in fold_results]
            metrics[metric_name] = float(np.mean(values))
            metrics_std[metric_name] = float(np.std(values))
        
        evaluation_time = time.time() - start_time
        
        return EvaluationResult(
            model_name=model.__class__.__name__,
            dataset_name=kwargs.get('dataset_name', 'Unknown'),
            config=self.config,
            metrics=metrics,
            metrics_std=metrics_std,
            fold_results=fold_results,
            training_time=total_training_time,
            evaluation_time=evaluation_time
        )


def create_evaluator(config: EvaluationConfig) -> ModelEvaluator:
    """Create an evaluator based on configuration."""
    if config.strategy == "holdout":
        return HoldoutEvaluator(config)
    elif config.strategy == "cross_validation":
        return CrossValidationEvaluator(config)
    elif config.strategy == "time_series":
        return TimeSeriesEvaluator(config)
    else:
        raise ValueError(f"Unknown evaluation strategy: {config.strategy}")


class BatchEvaluator:
    """Evaluate multiple models on multiple datasets."""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.results: List[EvaluationResult] = []
    
    def evaluate_models(
        self,
        models: Dict[str, Any],
        datasets: Dict[str, Tuple[pd.DataFrame, pd.Series]],
        task_types: Dict[str, str] = None,
        **kwargs
    ) -> List[EvaluationResult]:
        """Evaluate multiple models on multiple datasets."""
        
        if task_types is None:
            task_types = {name: "classification" for name in datasets.keys()}
        
        evaluator = create_evaluator(self.config)
        
        all_results = []
        
        for dataset_name, (X, y) in datasets.items():
            task_type = task_types.get(dataset_name, "classification")
            
            logger.info(f"Evaluating on dataset: {dataset_name}")
            
            for model_name, model in models.items():
                logger.info(f"  Evaluating model: {model_name}")
                
                try:
                    # Create a copy of the model for each evaluation
                    import copy
                    model_copy = copy.deepcopy(model)
                    
                    result = evaluator.evaluate(
                        model_copy, X, y, task_type,
                        dataset_name=dataset_name,
                        **kwargs
                    )
                    result.model_name = model_name  # Override with custom name
                    
                    all_results.append(result)
                    self.results.append(result)
                    
                except Exception as e:
                    logger.error(f"Error evaluating {model_name} on {dataset_name}: {e}")
                    continue
        
        return all_results
    
    def get_summary(self) -> pd.DataFrame:
        """Get summary of all evaluation results."""
        if not self.results:
            return pd.DataFrame()
        
        summary_data = []
        for result in self.results:
            row = {
                'Model': result.model_name,
                'Dataset': result.dataset_name,
                'Strategy': result.config.strategy,
                'Training Time (s)': result.training_time,
                'Evaluation Time (s)': result.evaluation_time
            }
            
            # Add metrics
            for metric, value in result.metrics.items():
                row[metric] = value
                
                # Add standard deviation if available
                if metric in result.metrics_std:
                    row[f'{metric}_std'] = result.metrics_std[metric]
            
            summary_data.append(row)
        
        return pd.DataFrame(summary_data)
    
    def get_model_comparison(self, dataset_name: str, metric: str = None) -> pd.DataFrame:
        """Compare models on a specific dataset."""
        if metric is None:
            metric = self.config.primary_metric
        
        dataset_results = [r for r in self.results if r.dataset_name == dataset_name]
        
        if not dataset_results:
            return pd.DataFrame()
        
        comparison_data = []
        for result in dataset_results:
            row = {
                'Model': result.model_name,
                'Strategy': result.config.strategy
            }
            
            # Add all metrics
            for metric_name, value in result.metrics.items():
                row[metric_name] = value
                if metric_name in result.metrics_std:
                    row[f'{metric_name}_std'] = result.metrics_std[metric_name]
            
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        
        # Sort by primary metric
        if metric in df.columns:
            ascending = metric.lower() in ['loss', 'error', 'mse', 'mae']
            df = df.sort_values(metric, ascending=ascending)
        
        return df