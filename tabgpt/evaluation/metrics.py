"""Comprehensive metrics for evaluating TabGPT models."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod
import warnings

try:
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, average_precision_score, log_loss,
        mean_squared_error, mean_absolute_error, r2_score,
        confusion_matrix, classification_report
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics."""
    
    task_type: str
    metrics: Dict[str, float]
    additional_info: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.additional_info is None:
            self.additional_info = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'task_type': self.task_type,
            'metrics': self.metrics,
            'additional_info': self.additional_info
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EvaluationMetrics':
        """Create from dictionary."""
        return cls(**data)


class ClassificationMetrics:
    """Metrics for classification tasks."""
    
    @staticmethod
    def compute_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: Optional[np.ndarray] = None,
        average: str = 'weighted',
        labels: Optional[List] = None
    ) -> Dict[str, float]:
        """Compute classification metrics."""
        if not SKLEARN_AVAILABLE:
            warnings.warn("sklearn not available, returning basic metrics only")
            return {'accuracy': float(np.mean(y_true == y_pred))}
        
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, average=average, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, average=average, zero_division=0)
        metrics['f1'] = f1_score(y_true, y_pred, average=average, zero_division=0)
        
        # Probability-based metrics
        if y_pred_proba is not None:
            try:
                # For binary classification
                if y_pred_proba.shape[1] == 2:
                    metrics['auc'] = roc_auc_score(y_true, y_pred_proba[:, 1])
                    metrics['average_precision'] = average_precision_score(y_true, y_pred_proba[:, 1])
                # For multiclass
                else:
                    metrics['auc'] = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average=average)
                
                # Log loss
                metrics['log_loss'] = log_loss(y_true, y_pred_proba)
            except Exception as e:
                warnings.warn(f"Could not compute probability-based metrics: {e}")
        
        # Per-class metrics for detailed analysis
        if labels is not None:
            try:
                precision_per_class = precision_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
                recall_per_class = recall_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
                f1_per_class = f1_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
                
                for i, label in enumerate(labels):
                    metrics[f'precision_class_{label}'] = precision_per_class[i]
                    metrics[f'recall_class_{label}'] = recall_per_class[i]
                    metrics[f'f1_class_{label}'] = f1_per_class[i]
            except Exception as e:
                warnings.warn(f"Could not compute per-class metrics: {e}")
        
        return metrics
    
    @staticmethod
    def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Compute confusion matrix."""
        if not SKLEARN_AVAILABLE:
            raise ImportError("sklearn required for confusion matrix")
        return confusion_matrix(y_true, y_pred)
    
    @staticmethod
    def get_classification_report(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        target_names: Optional[List[str]] = None
    ) -> str:
        """Get detailed classification report."""
        if not SKLEARN_AVAILABLE:
            raise ImportError("sklearn required for classification report")
        return classification_report(y_true, y_pred, target_names=target_names)


class RegressionMetrics:
    """Metrics for regression tasks."""
    
    @staticmethod
    def compute_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Compute regression metrics."""
        metrics = {}
        
        # Basic metrics
        metrics['mse'] = float(np.mean((y_true - y_pred) ** 2))
        metrics['rmse'] = float(np.sqrt(metrics['mse']))
        metrics['mae'] = float(np.mean(np.abs(y_true - y_pred)))
        
        # Relative metrics
        if np.std(y_true) > 0:
            metrics['normalized_rmse'] = metrics['rmse'] / np.std(y_true)
        
        # Percentage errors
        non_zero_mask = y_true != 0
        if np.any(non_zero_mask):
            mape = np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100
            metrics['mape'] = float(mape)
        
        # R-squared and adjusted R-squared
        if SKLEARN_AVAILABLE:
            metrics['r2'] = r2_score(y_true, y_pred)
        else:
            # Manual R² calculation
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            metrics['r2'] = float(1 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0
        
        # Additional metrics
        metrics['max_error'] = float(np.max(np.abs(y_true - y_pred)))
        metrics['median_absolute_error'] = float(np.median(np.abs(y_true - y_pred)))
        
        return metrics


class PretrainingMetrics:
    """Metrics for pre-training evaluation."""
    
    @staticmethod
    def compute_masked_lm_metrics(
        predictions: np.ndarray,
        targets: np.ndarray,
        mask: np.ndarray
    ) -> Dict[str, float]:
        """Compute masked language modeling metrics."""
        metrics = {}
        
        # Only consider masked positions
        masked_predictions = predictions[mask]
        masked_targets = targets[mask]
        
        if len(masked_predictions) > 0:
            # Accuracy for categorical predictions
            if masked_predictions.ndim > 1:
                pred_classes = np.argmax(masked_predictions, axis=-1)
                metrics['masked_accuracy'] = float(np.mean(pred_classes == masked_targets))
                
                # Cross-entropy loss
                # Clip predictions to avoid log(0)
                clipped_preds = np.clip(masked_predictions, 1e-7, 1 - 1e-7)
                ce_loss = -np.mean(np.log(clipped_preds[np.arange(len(masked_targets)), masked_targets]))
                metrics['masked_cross_entropy'] = float(ce_loss)
            else:
                # MSE for continuous predictions
                mse = np.mean((masked_predictions - masked_targets) ** 2)
                metrics['masked_mse'] = float(mse)
                metrics['masked_rmse'] = float(np.sqrt(mse))
        
        return metrics
    
    @staticmethod
    def compute_contrastive_metrics(
        similarities: np.ndarray,
        labels: np.ndarray,
        temperature: float = 0.1
    ) -> Dict[str, float]:
        """Compute contrastive learning metrics."""
        metrics = {}
        
        # InfoNCE loss
        scaled_similarities = similarities / temperature
        exp_similarities = np.exp(scaled_similarities)
        
        # Compute loss for positive pairs
        positive_similarities = scaled_similarities[labels == 1]
        if len(positive_similarities) > 0:
            # Approximate InfoNCE loss
            numerator = np.exp(positive_similarities)
            denominator = np.sum(exp_similarities, axis=-1, keepdims=True)
            infonce_loss = -np.mean(np.log(numerator / (denominator + 1e-8)))
            metrics['infonce_loss'] = float(infonce_loss)
        
        # Accuracy (how often positive pairs have highest similarity)
        if similarities.ndim == 2:
            pred_labels = np.argmax(similarities, axis=-1)
            true_labels = np.argmax(labels.reshape(similarities.shape), axis=-1)
            metrics['contrastive_accuracy'] = float(np.mean(pred_labels == true_labels))
        
        return metrics


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: Optional[np.ndarray] = None,
    task_type: str = "classification",
    **kwargs
) -> Dict[str, float]:
    """Compute all relevant metrics for a given task type."""
    
    if task_type == "classification":
        return ClassificationMetrics.compute_metrics(
            y_true, y_pred, y_pred_proba, **kwargs
        )
    elif task_type == "regression":
        return RegressionMetrics.compute_metrics(y_true, y_pred, **kwargs)
    else:
        raise ValueError(f"Unsupported task type: {task_type}")


def compare_models(
    results: List[EvaluationMetrics],
    primary_metric: str = "accuracy"
) -> pd.DataFrame:
    """Compare multiple model results."""
    
    if not results:
        return pd.DataFrame()
    
    # Extract metrics from all results
    comparison_data = []
    for i, result in enumerate(results):
        row = {'Model': f'Model_{i}', 'Task': result.task_type}
        row.update(result.metrics)
        comparison_data.append(row)
    
    df = pd.DataFrame(comparison_data)
    
    # Sort by primary metric if available
    if primary_metric in df.columns:
        ascending = primary_metric.lower() in ['loss', 'error', 'mse', 'mae']
        df = df.sort_values(primary_metric, ascending=ascending)
    
    return df


class MetricTracker:
    """Track metrics over time during training."""
    
    def __init__(self):
        self.metrics_history: Dict[str, List[float]] = {}
        self.steps: List[int] = []
    
    def update(self, metrics: Dict[str, float], step: int):
        """Update metrics for a given step."""
        self.steps.append(step)
        
        for metric_name, value in metrics.items():
            if metric_name not in self.metrics_history:
                self.metrics_history[metric_name] = []
            self.metrics_history[metric_name].append(value)
    
    def get_history(self, metric_name: str) -> Tuple[List[int], List[float]]:
        """Get history for a specific metric."""
        if metric_name not in self.metrics_history:
            return [], []
        return self.steps, self.metrics_history[metric_name]
    
    def get_latest(self, metric_name: str) -> Optional[float]:
        """Get latest value for a metric."""
        if metric_name not in self.metrics_history or not self.metrics_history[metric_name]:
            return None
        return self.metrics_history[metric_name][-1]
    
    def get_best(self, metric_name: str, higher_is_better: bool = True) -> Tuple[int, float]:
        """Get best value and step for a metric."""
        if metric_name not in self.metrics_history:
            return -1, 0.0
        
        values = self.metrics_history[metric_name]
        if higher_is_better:
            best_idx = np.argmax(values)
        else:
            best_idx = np.argmin(values)
        
        return self.steps[best_idx], values[best_idx]
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert metrics history to DataFrame."""
        data = {'step': self.steps}
        
        # Pad shorter metric lists with NaN
        max_length = len(self.steps)
        for metric_name, values in self.metrics_history.items():
            padded_values = values + [np.nan] * (max_length - len(values))
            data[metric_name] = padded_values
        
        return pd.DataFrame(data)


class StatisticalSignificanceTest:
    """Statistical significance testing for model comparisons."""
    
    @staticmethod
    def paired_t_test(
        scores_a: np.ndarray,
        scores_b: np.ndarray,
        alpha: float = 0.05
    ) -> Dict[str, float]:
        """Perform paired t-test between two sets of scores."""
        try:
            from scipy.stats import ttest_rel
            
            statistic, p_value = ttest_rel(scores_a, scores_b)
            
            return {
                'statistic': float(statistic),
                'p_value': float(p_value),
                'significant': p_value < alpha,
                'alpha': alpha
            }
        except ImportError:
            warnings.warn("scipy not available for statistical tests")
            return {}
    
    @staticmethod
    def wilcoxon_test(
        scores_a: np.ndarray,
        scores_b: np.ndarray,
        alpha: float = 0.05
    ) -> Dict[str, float]:
        """Perform Wilcoxon signed-rank test."""
        try:
            from scipy.stats import wilcoxon
            
            statistic, p_value = wilcoxon(scores_a, scores_b)
            
            return {
                'statistic': float(statistic),
                'p_value': float(p_value),
                'significant': p_value < alpha,
                'alpha': alpha
            }
        except ImportError:
            warnings.warn("scipy not available for statistical tests")
            return {}
    
    @staticmethod
    def bootstrap_confidence_interval(
        scores: np.ndarray,
        confidence_level: float = 0.95,
        n_bootstrap: int = 1000
    ) -> Dict[str, float]:
        """Compute bootstrap confidence interval."""
        np.random.seed(42)  # For reproducibility
        
        bootstrap_means = []
        for _ in range(n_bootstrap):
            bootstrap_sample = np.random.choice(scores, size=len(scores), replace=True)
            bootstrap_means.append(np.mean(bootstrap_sample))
        
        bootstrap_means = np.array(bootstrap_means)
        alpha = 1 - confidence_level
        
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        return {
            'mean': float(np.mean(scores)),
            'lower_bound': float(np.percentile(bootstrap_means, lower_percentile)),
            'upper_bound': float(np.percentile(bootstrap_means, upper_percentile)),
            'confidence_level': confidence_level
        }