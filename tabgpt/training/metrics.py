"""Metrics computation for TabGPT training and evaluation."""
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Union, Any, Tuple
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, mean_squared_error, mean_absolute_error,
    r2_score, confusion_matrix
)
import warnings


class MetricsComputer:
    """Compute various metrics for model evaluation."""
    
    def __init__(self, task_type: str = "pretraining"):
        """
        Initialize metrics computer.
        
        Args:
            task_type: Type of task ('pretraining', 'classification', 'regression')
        """
        self.task_type = task_type
    
    def compute_pretraining_metrics(
        self,
        model_outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """
        Compute metrics for pre-training objectives.
        
        Args:
            model_outputs: Model outputs containing losses and predictions
            targets: Target values for different objectives
            
        Returns:
            Dictionary of computed metrics
        """
        metrics = {}
        
        # Masked Cell Modeling metrics
        if 'mcm_predictions' in model_outputs and 'mcm_targets' in targets:
            mcm_metrics = self._compute_mcm_metrics(
                model_outputs['mcm_predictions'],
                targets['mcm_targets'],
                targets.get('mcm_mask')
            )
            metrics.update({f'mcm_{k}': v for k, v in mcm_metrics.items()})
        
        # Masked Column Modeling metrics
        if 'mcol_predictions' in model_outputs and 'mcol_targets' in targets:
            mcol_metrics = self._compute_mcol_metrics(
                model_outputs['mcol_predictions'],
                targets['mcol_targets'],
                targets.get('mcol_mask')
            )
            metrics.update({f'mcol_{k}': v for k, v in mcol_metrics.items()})
        
        # Contrastive Row Learning metrics
        if 'crl_similarity' in model_outputs:
            crl_metrics = self._compute_crl_metrics(model_outputs['crl_similarity'])
            metrics.update({f'crl_{k}': v for k, v in crl_metrics.items()})
        
        # Next Row Prediction metrics
        if 'nrp_predictions' in model_outputs and 'nrp_targets' in targets:
            nrp_metrics = self._compute_nrp_metrics(
                model_outputs['nrp_predictions'],
                targets['nrp_targets']
            )
            metrics.update({f'nrp_{k}': v for k, v in nrp_metrics.items()})
        
        return metrics
    
    def _compute_mcm_metrics(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """Compute Masked Cell Modeling metrics."""
        metrics = {}
        
        if mask is not None:
            # Only compute metrics for masked positions
            masked_predictions = predictions[mask]
            masked_targets = targets[mask]
        else:
            masked_predictions = predictions
            masked_targets = targets
        
        if len(masked_predictions) == 0:
            return metrics
        
        # Convert to numpy for sklearn metrics
        pred_np = masked_predictions.detach().cpu().numpy()
        target_np = masked_targets.detach().cpu().numpy()
        
        # Check if this is classification or regression
        if len(masked_targets.shape) > 1 and masked_targets.shape[-1] > 1:
            # Multi-class classification
            pred_classes = np.argmax(pred_np, axis=-1)
            target_classes = np.argmax(target_np, axis=-1) if target_np.shape[-1] > 1 else target_np.astype(int)
            
            metrics['accuracy'] = accuracy_score(target_classes, pred_classes)
            
            try:
                precision, recall, f1, _ = precision_recall_fscore_support(
                    target_classes, pred_classes, average='weighted', zero_division=0
                )
                metrics['precision'] = precision
                metrics['recall'] = recall
                metrics['f1'] = f1
            except Exception:
                pass
        else:
            # Regression or binary classification
            if torch.all((masked_targets >= 0) & (masked_targets <= 1)):
                # Binary classification
                pred_probs = torch.sigmoid(masked_predictions).detach().cpu().numpy()
                pred_classes = (pred_probs > 0.5).astype(int)
                target_classes = target_np.astype(int)
                
                metrics['accuracy'] = accuracy_score(target_classes, pred_classes)
                
                try:
                    metrics['auc'] = roc_auc_score(target_classes, pred_probs)
                except Exception:
                    pass
            else:
                # Regression
                metrics['mse'] = mean_squared_error(target_np, pred_np)
                metrics['mae'] = mean_absolute_error(target_np, pred_np)
                metrics['rmse'] = np.sqrt(metrics['mse'])
                
                try:
                    metrics['r2'] = r2_score(target_np, pred_np)
                except Exception:
                    pass
        
        return metrics
    
    def _compute_mcol_metrics(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """Compute Masked Column Modeling metrics."""
        metrics = {}
        
        if mask is not None:
            # Only compute metrics for masked columns
            masked_predictions = predictions[mask]
            masked_targets = targets[mask]
        else:
            masked_predictions = predictions
            masked_targets = targets
        
        if len(masked_predictions) == 0:
            return metrics
        
        # Convert to numpy
        pred_np = masked_predictions.detach().cpu().numpy()
        target_np = masked_targets.detach().cpu().numpy()
        
        # Column-level prediction is typically classification
        if len(pred_np.shape) > 1 and pred_np.shape[-1] > 1:
            pred_classes = np.argmax(pred_np, axis=-1)
            target_classes = target_np.astype(int) if len(target_np.shape) == 1 else np.argmax(target_np, axis=-1)
            
            metrics['accuracy'] = accuracy_score(target_classes, pred_classes)
            
            try:
                precision, recall, f1, _ = precision_recall_fscore_support(
                    target_classes, pred_classes, average='weighted', zero_division=0
                )
                metrics['precision'] = precision
                metrics['recall'] = recall
                metrics['f1'] = f1
            except Exception:
                pass
        
        return metrics
    
    def _compute_crl_metrics(self, similarity_matrix: torch.Tensor) -> Dict[str, float]:
        """Compute Contrastive Row Learning metrics."""
        metrics = {}
        
        # Convert to numpy
        sim_np = similarity_matrix.detach().cpu().numpy()
        
        # Compute statistics of similarity matrix
        metrics['mean_similarity'] = np.mean(sim_np)
        metrics['std_similarity'] = np.std(sim_np)
        
        # Compute positive pair similarity (diagonal elements)
        if sim_np.shape[0] == sim_np.shape[1]:
            positive_similarities = np.diag(sim_np)
            metrics['positive_similarity_mean'] = np.mean(positive_similarities)
            metrics['positive_similarity_std'] = np.std(positive_similarities)
            
            # Compute negative pair similarity (off-diagonal elements)
            mask = np.ones_like(sim_np, dtype=bool)
            np.fill_diagonal(mask, False)
            negative_similarities = sim_np[mask]
            metrics['negative_similarity_mean'] = np.mean(negative_similarities)
            metrics['negative_similarity_std'] = np.std(negative_similarities)
            
            # Compute separation (positive - negative similarity)
            metrics['similarity_separation'] = (
                metrics['positive_similarity_mean'] - metrics['negative_similarity_mean']
            )
        
        return metrics
    
    def _compute_nrp_metrics(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """Compute Next Row Prediction metrics."""
        metrics = {}
        
        for column_name in predictions:
            if column_name not in targets:
                continue
            
            pred = predictions[column_name]
            target = targets[column_name]
            
            # Convert to numpy
            pred_np = pred.detach().cpu().numpy()
            target_np = target.detach().cpu().numpy()
            
            # Determine if this is classification or regression
            if len(pred.shape) > 1 and pred.shape[-1] > 1:
                # Classification
                pred_classes = np.argmax(pred_np, axis=-1)
                target_classes = target_np.astype(int) if len(target_np.shape) == 1 else np.argmax(target_np, axis=-1)
                
                accuracy = accuracy_score(target_classes, pred_classes)
                metrics[f'{column_name}_accuracy'] = accuracy
            else:
                # Regression
                mse = mean_squared_error(target_np, pred_np)
                mae = mean_absolute_error(target_np, pred_np)
                
                metrics[f'{column_name}_mse'] = mse
                metrics[f'{column_name}_mae'] = mae
                metrics[f'{column_name}_rmse'] = np.sqrt(mse)
        
        # Compute average metrics across columns
        if metrics:
            accuracy_metrics = [v for k, v in metrics.items() if 'accuracy' in k]
            mse_metrics = [v for k, v in metrics.items() if 'mse' in k and 'rmse' not in k]
            mae_metrics = [v for k, v in metrics.items() if 'mae' in k]
            
            if accuracy_metrics:
                metrics['avg_accuracy'] = np.mean(accuracy_metrics)
            if mse_metrics:
                metrics['avg_mse'] = np.mean(mse_metrics)
                metrics['avg_rmse'] = np.sqrt(metrics['avg_mse'])
            if mae_metrics:
                metrics['avg_mae'] = np.mean(mae_metrics)
        
        return metrics
    
    def compute_classification_metrics(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        num_classes: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Compute classification metrics.
        
        Args:
            predictions: Model predictions (logits or probabilities)
            targets: True labels
            num_classes: Number of classes (for multi-class)
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Convert to numpy
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()
        
        # Determine task type
        if len(predictions.shape) > 1 and predictions.shape[-1] > 1:
            # Multi-class classification
            pred_classes = np.argmax(predictions, axis=-1)
            pred_probs = predictions
        else:
            # Binary classification
            pred_probs = predictions if predictions.max() <= 1.0 else torch.sigmoid(torch.tensor(predictions)).numpy()
            pred_classes = (pred_probs > 0.5).astype(int)
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(targets, pred_classes)
        
        try:
            precision, recall, f1, _ = precision_recall_fscore_support(
                targets, pred_classes, average='weighted', zero_division=0
            )
            metrics['precision'] = precision
            metrics['recall'] = recall
            metrics['f1'] = f1
        except Exception as e:
            warnings.warn(f"Could not compute precision/recall/f1: {e}")
        
        # AUC for binary classification or multi-class with probabilities
        try:
            if len(pred_probs.shape) == 1 or pred_probs.shape[-1] == 1:
                # Binary classification
                metrics['auc'] = roc_auc_score(targets, pred_probs)
            elif pred_probs.shape[-1] > 2:
                # Multi-class classification
                metrics['auc'] = roc_auc_score(targets, pred_probs, multi_class='ovr', average='weighted')
        except Exception as e:
            warnings.warn(f"Could not compute AUC: {e}")
        
        return metrics
    
    def compute_regression_metrics(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute regression metrics.
        
        Args:
            predictions: Model predictions
            targets: True values
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Convert to numpy
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()
        
        # Basic regression metrics
        metrics['mse'] = mean_squared_error(targets, predictions)
        metrics['mae'] = mean_absolute_error(targets, predictions)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        
        try:
            metrics['r2'] = r2_score(targets, predictions)
        except Exception as e:
            warnings.warn(f"Could not compute R²: {e}")
        
        # Additional metrics
        residuals = targets - predictions
        metrics['mean_residual'] = np.mean(residuals)
        metrics['std_residual'] = np.std(residuals)
        
        # Mean Absolute Percentage Error (MAPE)
        try:
            non_zero_mask = targets != 0
            if np.any(non_zero_mask):
                mape = np.mean(np.abs((targets[non_zero_mask] - predictions[non_zero_mask]) / targets[non_zero_mask])) * 100
                metrics['mape'] = mape
        except Exception:
            pass
        
        return metrics
    
    def compute_perplexity(self, loss: torch.Tensor) -> float:
        """
        Compute perplexity from cross-entropy loss.
        
        Args:
            loss: Cross-entropy loss tensor
            
        Returns:
            Perplexity value
        """
        return torch.exp(loss).item()
    
    def compute_attention_metrics(self, attention_weights: torch.Tensor) -> Dict[str, float]:
        """
        Compute metrics for attention weights.
        
        Args:
            attention_weights: Attention weight tensor [batch, heads, seq_len, seq_len]
            
        Returns:
            Dictionary of attention metrics
        """
        metrics = {}
        
        if attention_weights is None or attention_weights.numel() == 0:
            return metrics
        
        # Convert to numpy
        attn_np = attention_weights.detach().cpu().numpy()
        
        # Attention entropy (measure of attention distribution)
        # Higher entropy means more distributed attention
        attn_entropy = -np.sum(attn_np * np.log(attn_np + 1e-8), axis=-1)
        metrics['attention_entropy_mean'] = np.mean(attn_entropy)
        metrics['attention_entropy_std'] = np.std(attn_entropy)
        
        # Attention concentration (inverse of entropy)
        metrics['attention_concentration'] = 1.0 / (metrics['attention_entropy_mean'] + 1e-8)
        
        # Maximum attention weight (measure of attention sharpness)
        max_attention = np.max(attn_np, axis=-1)
        metrics['max_attention_mean'] = np.mean(max_attention)
        metrics['max_attention_std'] = np.std(max_attention)
        
        # Attention variance across heads
        if len(attn_np.shape) >= 4:  # [batch, heads, seq_len, seq_len]
            head_variance = np.var(attn_np, axis=1)  # Variance across heads
            metrics['attention_head_variance'] = np.mean(head_variance)
        
        return metrics


def compute_model_metrics(
    model,
    dataloader,
    device: str = "cpu",
    task_type: str = "pretraining"
) -> Dict[str, float]:
    """
    Compute comprehensive metrics for a model on a dataset.
    
    Args:
        model: PyTorch model
        dataloader: Data loader
        device: Device to run evaluation on
        task_type: Type of task for metric computation
        
    Returns:
        Dictionary of computed metrics
    """
    model.eval()
    metrics_computer = MetricsComputer(task_type)
    
    all_predictions = []
    all_targets = []
    all_losses = []
    
    with torch.no_grad():
        for batch in dataloader:
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Forward pass
            outputs = model(**batch)
            
            if isinstance(outputs, dict):
                if 'loss' in outputs:
                    all_losses.append(outputs['loss'].item())
                
                # Collect predictions and targets based on task type
                if task_type == "pretraining":
                    # Handle pre-training outputs
                    pass
                elif task_type == "classification":
                    if 'logits' in outputs:
                        all_predictions.append(outputs['logits'])
                    if 'labels' in batch:
                        all_targets.append(batch['labels'])
                elif task_type == "regression":
                    if 'predictions' in outputs:
                        all_predictions.append(outputs['predictions'])
                    if 'labels' in batch:
                        all_targets.append(batch['labels'])
    
    # Compute metrics
    metrics = {}
    
    if all_losses:
        metrics['avg_loss'] = np.mean(all_losses)
        if task_type in ["classification", "pretraining"]:
            metrics['perplexity'] = np.exp(metrics['avg_loss'])
    
    if all_predictions and all_targets:
        predictions = torch.cat(all_predictions, dim=0)
        targets = torch.cat(all_targets, dim=0)
        
        if task_type == "classification":
            class_metrics = metrics_computer.compute_classification_metrics(predictions, targets)
            metrics.update(class_metrics)
        elif task_type == "regression":
            reg_metrics = metrics_computer.compute_regression_metrics(predictions, targets)
            metrics.update(reg_metrics)
    
    return metrics