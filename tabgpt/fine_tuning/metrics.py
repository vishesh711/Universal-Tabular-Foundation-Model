"""Metrics computation for fine-tuning evaluation."""
import numpy as np
import torch
from typing import Dict, List, Optional, Union, Any, Tuple
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score,
    confusion_matrix, classification_report,
    ndcg_score, average_precision_score
)
import warnings

try:
    from transformers.trainer_callback import TrainerCallback
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    TrainerCallback = object


def compute_classification_metrics(eval_pred) -> Dict[str, float]:
    """
    Compute classification metrics for evaluation.
    
    Args:
        eval_pred: Tuple of (predictions, labels)
        
    Returns:
        Dictionary of computed metrics
    """
    predictions, labels = eval_pred
    
    # Handle different prediction formats
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    
    # Convert logits to predictions
    if len(predictions.shape) > 1 and predictions.shape[-1] > 1:
        # Multi-class classification
        pred_classes = np.argmax(predictions, axis=-1)
        pred_probs = predictions
        is_binary = False
    else:
        # Binary classification
        pred_probs = predictions if predictions.max() <= 1.0 else torch.sigmoid(torch.tensor(predictions)).numpy()
        pred_classes = (pred_probs > 0.5).astype(int).flatten()
        is_binary = True
    
    # Ensure labels are integers
    labels = labels.astype(int)
    
    # Basic metrics
    metrics = {}
    
    # Accuracy
    metrics['accuracy'] = accuracy_score(labels, pred_classes)
    
    # Precision, Recall, F1
    try:
        if is_binary:
            precision, recall, f1, _ = precision_recall_fscore_support(
                labels, pred_classes, average='binary', zero_division=0
            )
        else:
            precision, recall, f1, _ = precision_recall_fscore_support(
                labels, pred_classes, average='weighted', zero_division=0
            )
        
        metrics['precision'] = precision
        metrics['recall'] = recall
        metrics['f1'] = f1
    except Exception as e:
        warnings.warn(f"Could not compute precision/recall/f1: {e}")
    
    # AUC
    try:
        if is_binary:
            metrics['auc'] = roc_auc_score(labels, pred_probs.flatten())
        else:
            # Multi-class AUC
            metrics['auc'] = roc_auc_score(labels, pred_probs, multi_class='ovr', average='weighted')
    except Exception as e:
        warnings.warn(f"Could not compute AUC: {e}")
    
    # Average Precision (AP)
    try:
        if is_binary:
            metrics['average_precision'] = average_precision_score(labels, pred_probs.flatten())
        else:
            # Multi-class AP (macro average)
            from sklearn.preprocessing import label_binarize
            n_classes = len(np.unique(labels))
            if n_classes > 2:
                labels_bin = label_binarize(labels, classes=range(n_classes))
                ap_scores = []
                for i in range(n_classes):
                    ap = average_precision_score(labels_bin[:, i], pred_probs[:, i])
                    ap_scores.append(ap)
                metrics['average_precision'] = np.mean(ap_scores)
    except Exception as e:
        warnings.warn(f"Could not compute Average Precision: {e}")
    
    # Class-specific metrics for multi-class
    if not is_binary and len(np.unique(labels)) <= 10:  # Only for reasonable number of classes
        try:
            precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
                labels, pred_classes, average=None, zero_division=0
            )
            
            for i, (p, r, f) in enumerate(zip(precision_per_class, recall_per_class, f1_per_class)):
                metrics[f'precision_class_{i}'] = p
                metrics[f'recall_class_{i}'] = r
                metrics[f'f1_class_{i}'] = f
        except Exception:
            pass
    
    return metrics


def compute_regression_metrics(eval_pred) -> Dict[str, float]:
    """
    Compute regression metrics for evaluation.
    
    Args:
        eval_pred: Tuple of (predictions, labels)
        
    Returns:
        Dictionary of computed metrics
    """
    predictions, labels = eval_pred
    
    # Handle different prediction formats
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    
    # Flatten if needed
    predictions = predictions.flatten()
    labels = labels.flatten()
    
    metrics = {}
    
    # Basic regression metrics
    metrics['mse'] = mean_squared_error(labels, predictions)
    metrics['mae'] = mean_absolute_error(labels, predictions)
    metrics['rmse'] = np.sqrt(metrics['mse'])
    
    # R-squared
    try:
        metrics['r2'] = r2_score(labels, predictions)
    except Exception as e:
        warnings.warn(f"Could not compute R²: {e}")
        metrics['r2'] = 0.0
    
    # Mean Absolute Percentage Error (MAPE)
    try:
        non_zero_mask = labels != 0
        if np.any(non_zero_mask):
            mape = np.mean(np.abs((labels[non_zero_mask] - predictions[non_zero_mask]) / labels[non_zero_mask])) * 100
            metrics['mape'] = mape
    except Exception:
        pass
    
    # Residual statistics
    residuals = labels - predictions
    metrics['mean_residual'] = np.mean(residuals)
    metrics['std_residual'] = np.std(residuals)
    
    # Explained variance
    try:
        from sklearn.metrics import explained_variance_score
        metrics['explained_variance'] = explained_variance_score(labels, predictions)
    except Exception:
        pass
    
    # Max error
    metrics['max_error'] = np.max(np.abs(residuals))
    
    return metrics


def compute_ranking_metrics(eval_pred, k_values: List[int] = [1, 3, 5, 10]) -> Dict[str, float]:
    """
    Compute ranking metrics for evaluation.
    
    Args:
        eval_pred: Tuple of (predictions, labels)
        k_values: List of k values for top-k metrics
        
    Returns:
        Dictionary of computed metrics
    """
    predictions, labels = eval_pred
    
    # Handle different prediction formats
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    
    metrics = {}
    
    # NDCG (Normalized Discounted Cumulative Gain)
    try:
        for k in k_values:
            if k <= predictions.shape[-1]:
                ndcg_k = ndcg_score(labels.reshape(1, -1), predictions.reshape(1, -1), k=k)
                metrics[f'ndcg@{k}'] = ndcg_k
    except Exception as e:
        warnings.warn(f"Could not compute NDCG: {e}")
    
    # Mean Reciprocal Rank (MRR)
    try:
        # Get ranking of true labels
        sorted_indices = np.argsort(predictions, axis=-1)[::-1]
        ranks = []
        
        for i, label in enumerate(labels):
            if label > 0:  # Relevant item
                rank = np.where(sorted_indices == i)[0]
                if len(rank) > 0:
                    ranks.append(1.0 / (rank[0] + 1))
        
        if ranks:
            metrics['mrr'] = np.mean(ranks)
    except Exception as e:
        warnings.warn(f"Could not compute MRR: {e}")
    
    # Precision@K and Recall@K
    try:
        for k in k_values:
            if k <= len(predictions):
                # Get top-k predictions
                top_k_indices = np.argsort(predictions)[-k:]
                
                # Count relevant items in top-k
                relevant_in_top_k = np.sum(labels[top_k_indices] > 0)
                total_relevant = np.sum(labels > 0)
                
                # Precision@K
                metrics[f'precision@{k}'] = relevant_in_top_k / k if k > 0 else 0.0
                
                # Recall@K
                metrics[f'recall@{k}'] = relevant_in_top_k / total_relevant if total_relevant > 0 else 0.0
    except Exception as e:
        warnings.warn(f"Could not compute Precision@K/Recall@K: {e}")
    
    return metrics


class MetricsCallback(TrainerCallback if HF_AVAILABLE else object):
    """Callback for computing and logging custom metrics during training."""
    
    def __init__(
        self,
        task_type: str = "classification",
        compute_metrics_fn: Optional[callable] = None,
        log_predictions: bool = False
    ):
        self.task_type = task_type
        self.compute_metrics_fn = compute_metrics_fn
        self.log_predictions = log_predictions
        
        # Set default metrics function
        if self.compute_metrics_fn is None:
            if task_type == "classification":
                self.compute_metrics_fn = compute_classification_metrics
            elif task_type == "regression":
                self.compute_metrics_fn = compute_regression_metrics
            elif task_type == "ranking":
                self.compute_metrics_fn = compute_ranking_metrics
    
    def on_evaluate(self, args, state, control, model, eval_dataloader, **kwargs):
        """Called after evaluation."""
        if not HF_AVAILABLE:
            return
        
        # Custom evaluation logic can be added here
        pass
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Called when logging."""
        if not HF_AVAILABLE or logs is None:
            return
        
        # Add custom logging logic here
        pass


def create_metrics_function(task_type: str, **kwargs) -> callable:
    """
    Create metrics computation function for a specific task.
    
    Args:
        task_type: Type of task (classification, regression, ranking)
        **kwargs: Additional arguments for metrics computation
        
    Returns:
        Metrics computation function
    """
    if task_type == "classification":
        return compute_classification_metrics
    elif task_type == "regression":
        return compute_regression_metrics
    elif task_type == "ranking":
        def ranking_metrics_fn(eval_pred):
            return compute_ranking_metrics(eval_pred, **kwargs)
        return ranking_metrics_fn
    else:
        raise ValueError(f"Unsupported task type: {task_type}")


def evaluate_predictions(
    predictions: np.ndarray,
    labels: np.ndarray,
    task_type: str,
    **kwargs
) -> Dict[str, float]:
    """
    Evaluate predictions against labels.
    
    Args:
        predictions: Model predictions
        labels: True labels
        task_type: Type of task
        **kwargs: Additional arguments
        
    Returns:
        Dictionary of computed metrics
    """
    eval_pred = (predictions, labels)
    
    if task_type == "classification":
        return compute_classification_metrics(eval_pred)
    elif task_type == "regression":
        return compute_regression_metrics(eval_pred)
    elif task_type == "ranking":
        return compute_ranking_metrics(eval_pred, **kwargs)
    else:
        raise ValueError(f"Unsupported task type: {task_type}")


def print_metrics_summary(metrics: Dict[str, float], task_type: str):
    """
    Print a formatted summary of metrics.
    
    Args:
        metrics: Dictionary of computed metrics
        task_type: Type of task
    """
    print(f"\n{task_type.title()} Metrics Summary:")
    print("=" * 40)
    
    if task_type == "classification":
        # Primary metrics
        primary_metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
        for metric in primary_metrics:
            if metric in metrics:
                print(f"{metric.upper():>15}: {metrics[metric]:.4f}")
        
        # Class-specific metrics
        class_metrics = {k: v for k, v in metrics.items() if 'class_' in k}
        if class_metrics:
            print("\nPer-class metrics:")
            for metric, value in class_metrics.items():
                print(f"{metric:>15}: {value:.4f}")
    
    elif task_type == "regression":
        # Primary metrics
        primary_metrics = ['mse', 'mae', 'rmse', 'r2']
        for metric in primary_metrics:
            if metric in metrics:
                print(f"{metric.upper():>15}: {metrics[metric]:.4f}")
        
        # Additional metrics
        additional_metrics = ['mape', 'explained_variance', 'max_error']
        for metric in additional_metrics:
            if metric in metrics:
                print(f"{metric:>15}: {metrics[metric]:.4f}")
    
    elif task_type == "ranking":
        # NDCG metrics
        ndcg_metrics = {k: v for k, v in metrics.items() if 'ndcg' in k}
        if ndcg_metrics:
            print("NDCG metrics:")
            for metric, value in ndcg_metrics.items():
                print(f"{metric:>15}: {value:.4f}")
        
        # Precision/Recall metrics
        precision_metrics = {k: v for k, v in metrics.items() if 'precision@' in k}
        recall_metrics = {k: v for k, v in metrics.items() if 'recall@' in k}
        
        if precision_metrics:
            print("\nPrecision@K metrics:")
            for metric, value in precision_metrics.items():
                print(f"{metric:>15}: {value:.4f}")
        
        if recall_metrics:
            print("\nRecall@K metrics:")
            for metric, value in recall_metrics.items():
                print(f"{metric:>15}: {value:.4f}")
        
        # MRR
        if 'mrr' in metrics:
            print(f"\n{'MRR':>15}: {metrics['mrr']:.4f}")
    
    print("=" * 40)