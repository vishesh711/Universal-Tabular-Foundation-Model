"""Classification heads for binary, multi-class, and multi-label tasks."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Union, Any
import numpy as np

from .base import BaseTaskHead, TaskOutput, TaskType, MLPHead, AttentionPoolingHead


class ClassificationHead(MLPHead):
    """General classification head that can handle binary, multi-class, and multi-label tasks."""
    
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        task_type: TaskType = TaskType.MULTICLASS_CLASSIFICATION,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.1,
        activation: str = "relu",
        use_class_weights: bool = False,
        class_weights: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.0
    ):
        # Determine final activation based on task type
        if task_type == TaskType.BINARY_CLASSIFICATION:
            final_activation = "sigmoid"
            output_dim = 1
        elif task_type == TaskType.MULTICLASS_CLASSIFICATION:
            final_activation = None  # Will use softmax in loss
            output_dim = num_classes
        elif task_type == TaskType.MULTILABEL_CLASSIFICATION:
            final_activation = "sigmoid"
            output_dim = num_classes
        else:
            raise ValueError(f"Unsupported task type for classification: {task_type}")
        
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            task_type=task_type,
            hidden_dims=hidden_dims,
            dropout=dropout,
            activation=activation,
            final_activation=final_activation
        )
        
        self.num_classes = num_classes
        self.use_class_weights = use_class_weights
        self.label_smoothing = label_smoothing
        
        # Class weights for imbalanced datasets
        if class_weights is not None:
            self.register_buffer('class_weights', class_weights)
        else:
            self.class_weights = None
    
    def compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """Compute classification loss with optional class weights and label smoothing."""
        
        if self.task_type == TaskType.BINARY_CLASSIFICATION:
            # Binary classification
            loss_fn = nn.BCELoss(weight=self.class_weights)
            return loss_fn(predictions.squeeze(), targets.float())
        
        elif self.task_type == TaskType.MULTICLASS_CLASSIFICATION:
            # Multi-class classification
            if self.label_smoothing > 0:
                # Label smoothing
                loss_fn = nn.CrossEntropyLoss(
                    weight=self.class_weights,
                    label_smoothing=self.label_smoothing
                )
            else:
                loss_fn = nn.CrossEntropyLoss(weight=self.class_weights)
            
            return loss_fn(predictions, targets.long())
        
        elif self.task_type == TaskType.MULTILABEL_CLASSIFICATION:
            # Multi-label classification
            pos_weight = self.class_weights if self.class_weights is not None else None
            loss_fn = nn.BCELoss(weight=pos_weight)
            return loss_fn(predictions, targets.float())
        
        else:
            raise NotImplementedError(f"Loss not implemented for task type: {self.task_type}")
    
    def compute_metrics(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        **kwargs
    ) -> Dict[str, float]:
        """Compute classification-specific metrics."""
        metrics = super().compute_metrics(predictions, targets, **kwargs)
        
        if self.task_type == TaskType.MULTICLASS_CLASSIFICATION:
            # Top-k accuracy
            for k in [3, 5]:
                if k < self.num_classes:
                    top_k_preds = predictions.topk(k, dim=-1)[1]
                    top_k_correct = (top_k_preds == targets.unsqueeze(-1)).any(dim=-1)
                    metrics[f'top_{k}_accuracy'] = top_k_correct.float().mean().item()
        
        elif self.task_type == TaskType.MULTILABEL_CLASSIFICATION:
            # Multi-label specific metrics
            pred_binary = (predictions > 0.5).float()
            
            # Hamming loss
            hamming_loss = torch.mean(torch.abs(pred_binary - targets.float())).item()
            metrics['hamming_loss'] = hamming_loss
            
            # Subset accuracy (exact match)
            subset_accuracy = torch.all(pred_binary == targets.float(), dim=-1).float().mean().item()
            metrics['subset_accuracy'] = subset_accuracy
        
        return metrics


class BinaryClassificationHead(ClassificationHead):
    """Specialized head for binary classification tasks."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.1,
        activation: str = "relu",
        threshold: float = 0.5,
        use_class_weights: bool = False,
        pos_weight: Optional[float] = None
    ):
        # Set class weights for binary classification
        class_weights = None
        if pos_weight is not None:
            class_weights = torch.tensor([1.0, pos_weight])
        
        super().__init__(
            input_dim=input_dim,
            num_classes=2,
            task_type=TaskType.BINARY_CLASSIFICATION,
            hidden_dims=hidden_dims,
            dropout=dropout,
            activation=activation,
            use_class_weights=use_class_weights,
            class_weights=class_weights
        )
        
        self.threshold = threshold
    
    def get_predictions(self, logits: torch.Tensor) -> torch.Tensor:
        """Get binary predictions with custom threshold."""
        probabilities = torch.sigmoid(logits)
        return (probabilities > self.threshold).float()
    
    def compute_metrics(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        **kwargs
    ) -> Dict[str, float]:
        """Compute binary classification metrics."""
        # Get probabilities for AUC calculation
        probabilities = torch.sigmoid(predictions) if predictions.max() > 1 else predictions
        pred_classes = (probabilities > self.threshold).float()
        
        # Basic metrics
        accuracy = (pred_classes.squeeze() == targets.float()).float().mean().item()
        
        # Confusion matrix components
        tp = ((pred_classes.squeeze() == 1) & (targets == 1)).sum().item()
        fp = ((pred_classes.squeeze() == 1) & (targets == 0)).sum().item()
        tn = ((pred_classes.squeeze() == 0) & (targets == 0)).sum().item()
        fn = ((pred_classes.squeeze() == 0) & (targets == 1)).sum().item()
        
        # Precision, Recall, F1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Balanced accuracy
        balanced_accuracy = (recall + specificity) / 2
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'specificity': specificity,
            'f1_score': f1,
            'balanced_accuracy': balanced_accuracy,
            'true_positives': tp,
            'false_positives': fp,
            'true_negatives': tn,
            'false_negatives': fn
        }


class MultiClassClassificationHead(ClassificationHead):
    """Specialized head for multi-class classification tasks."""
    
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.1,
        activation: str = "relu",
        use_class_weights: bool = False,
        class_weights: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.0,
        temperature: float = 1.0
    ):
        super().__init__(
            input_dim=input_dim,
            num_classes=num_classes,
            task_type=TaskType.MULTICLASS_CLASSIFICATION,
            hidden_dims=hidden_dims,
            dropout=dropout,
            activation=activation,
            use_class_weights=use_class_weights,
            class_weights=class_weights,
            label_smoothing=label_smoothing
        )
        
        self.temperature = temperature
    
    def get_predictions(self, logits: torch.Tensor) -> torch.Tensor:
        """Get class predictions with temperature scaling."""
        scaled_logits = logits / self.temperature
        return torch.softmax(scaled_logits, dim=-1)
    
    def compute_metrics(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        **kwargs
    ) -> Dict[str, float]:
        """Compute multi-class classification metrics."""
        pred_classes = predictions.argmax(dim=-1)
        
        # Overall accuracy
        accuracy = (pred_classes == targets).float().mean().item()
        
        # Per-class metrics
        per_class_precision = []
        per_class_recall = []
        per_class_f1 = []
        
        for class_idx in range(self.num_classes):
            # Binary mask for current class
            true_class = (targets == class_idx)
            pred_class = (pred_classes == class_idx)
            
            if true_class.sum() > 0:  # Only compute if class exists in targets
                tp = (true_class & pred_class).sum().item()
                fp = (~true_class & pred_class).sum().item()
                fn = (true_class & ~pred_class).sum().item()
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
                
                per_class_precision.append(precision)
                per_class_recall.append(recall)
                per_class_f1.append(f1)
        
        # Macro averages
        macro_precision = np.mean(per_class_precision) if per_class_precision else 0.0
        macro_recall = np.mean(per_class_recall) if per_class_recall else 0.0
        macro_f1 = np.mean(per_class_f1) if per_class_f1 else 0.0
        
        # Top-k accuracy
        metrics = {
            'accuracy': accuracy,
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'macro_f1': macro_f1
        }
        
        # Add top-k accuracy
        for k in [3, 5]:
            if k < self.num_classes:
                top_k_preds = predictions.topk(k, dim=-1)[1]
                top_k_correct = (top_k_preds == targets.unsqueeze(-1)).any(dim=-1)
                metrics[f'top_{k}_accuracy'] = top_k_correct.float().mean().item()
        
        return metrics


class MultiLabelClassificationHead(ClassificationHead):
    """Specialized head for multi-label classification tasks."""
    
    def __init__(
        self,
        input_dim: int,
        num_labels: int,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.1,
        activation: str = "relu",
        threshold: float = 0.5,
        use_class_weights: bool = False,
        pos_weights: Optional[torch.Tensor] = None
    ):
        super().__init__(
            input_dim=input_dim,
            num_classes=num_labels,
            task_type=TaskType.MULTILABEL_CLASSIFICATION,
            hidden_dims=hidden_dims,
            dropout=dropout,
            activation=activation,
            use_class_weights=use_class_weights,
            class_weights=pos_weights
        )
        
        self.num_labels = num_labels
        self.threshold = threshold
    
    def get_predictions(self, logits: torch.Tensor) -> torch.Tensor:
        """Get multi-label predictions with custom threshold."""
        probabilities = torch.sigmoid(logits)
        return (probabilities > self.threshold).float()
    
    def compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """Compute multi-label loss with optional positive weights."""
        if self.class_weights is not None:
            # Use BCEWithLogitsLoss for numerical stability with pos_weight
            loss_fn = nn.BCEWithLogitsLoss(pos_weight=self.class_weights)
            # Need to pass logits, not probabilities
            logits = torch.log(predictions / (1 - predictions + 1e-8))  # Inverse sigmoid
            return loss_fn(logits, targets.float())
        else:
            loss_fn = nn.BCELoss()
            return loss_fn(predictions, targets.float())
    
    def compute_metrics(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        **kwargs
    ) -> Dict[str, float]:
        """Compute multi-label classification metrics."""
        # Get binary predictions
        probabilities = torch.sigmoid(predictions) if predictions.max() > 1 else predictions
        pred_binary = (probabilities > self.threshold).float()
        targets_float = targets.float()
        
        # Hamming loss (fraction of wrong labels)
        hamming_loss = torch.mean(torch.abs(pred_binary - targets_float)).item()
        
        # Subset accuracy (exact match ratio)
        subset_accuracy = torch.all(pred_binary == targets_float, dim=-1).float().mean().item()
        
        # Label-wise metrics
        label_precision = []
        label_recall = []
        label_f1 = []
        
        for label_idx in range(self.num_labels):
            true_label = targets_float[:, label_idx]
            pred_label = pred_binary[:, label_idx]
            
            tp = (true_label * pred_label).sum().item()
            fp = ((1 - true_label) * pred_label).sum().item()
            fn = (true_label * (1 - pred_label)).sum().item()
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            label_precision.append(precision)
            label_recall.append(recall)
            label_f1.append(f1)
        
        # Micro averages (aggregate TP, FP, FN across all labels)
        all_tp = (targets_float * pred_binary).sum().item()
        all_fp = ((1 - targets_float) * pred_binary).sum().item()
        all_fn = (targets_float * (1 - pred_binary)).sum().item()
        
        micro_precision = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0.0
        micro_recall = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0.0
        micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0.0
        
        # Macro averages
        macro_precision = np.mean(label_precision)
        macro_recall = np.mean(label_recall)
        macro_f1 = np.mean(label_f1)
        
        return {
            'hamming_loss': hamming_loss,
            'subset_accuracy': subset_accuracy,
            'micro_precision': micro_precision,
            'micro_recall': micro_recall,
            'micro_f1': micro_f1,
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'macro_f1': macro_f1
        }


class HierarchicalClassificationHead(BaseTaskHead):
    """Head for hierarchical classification with multiple levels."""
    
    def __init__(
        self,
        input_dim: int,
        hierarchy_sizes: List[int],  # Number of classes at each level
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.1,
        activation: str = "relu",
        level_weights: Optional[List[float]] = None
    ):
        super().__init__(input_dim, TaskType.MULTICLASS_CLASSIFICATION, dropout, activation)
        
        self.hierarchy_sizes = hierarchy_sizes
        self.num_levels = len(hierarchy_sizes)
        self.level_weights = level_weights or [1.0] * self.num_levels
        
        # Shared feature extraction
        if hidden_dims:
            layers = []
            prev_dim = input_dim
            for hidden_dim in hidden_dims:
                layers.extend([
                    nn.Linear(prev_dim, hidden_dim),
                    self.activation_fn,
                    self.dropout_layer
                ])
                prev_dim = hidden_dim
            self.shared_layers = nn.Sequential(*layers)
            feature_dim = prev_dim
        else:
            self.shared_layers = nn.Identity()
            feature_dim = input_dim
        
        # Level-specific classifiers
        self.level_classifiers = nn.ModuleList([
            nn.Linear(feature_dim, num_classes)
            for num_classes in hierarchy_sizes
        ])
    
    def forward(
        self,
        features: torch.Tensor,
        targets: Optional[List[torch.Tensor]] = None,
        **kwargs
    ) -> TaskOutput:
        """Forward pass through hierarchical classifier."""
        # Shared feature extraction
        shared_features = self.shared_layers(features)
        
        # Level-specific predictions
        level_logits = []
        level_predictions = []
        
        for classifier in self.level_classifiers:
            logits = classifier(shared_features)
            predictions = torch.softmax(logits, dim=-1)
            
            level_logits.append(logits)
            level_predictions.append(predictions)
        
        # Compute loss if targets provided
        loss = None
        if targets is not None:
            loss = self.compute_hierarchical_loss(level_logits, targets)
        
        return TaskOutput(
            predictions=level_predictions,
            loss=loss,
            logits=level_logits,
            features=shared_features,
            metadata={'num_levels': self.num_levels}
        )
    
    def compute_hierarchical_loss(
        self,
        level_logits: List[torch.Tensor],
        targets: List[torch.Tensor]
    ) -> torch.Tensor:
        """Compute weighted loss across hierarchy levels."""
        total_loss = 0.0
        
        for i, (logits, target, weight) in enumerate(zip(level_logits, targets, self.level_weights)):
            level_loss = nn.CrossEntropyLoss()(logits, target.long())
            total_loss += weight * level_loss
        
        return total_loss
    
    def compute_loss(
        self,
        predictions: List[torch.Tensor],
        targets: List[torch.Tensor],
        **kwargs
    ) -> torch.Tensor:
        """Compute loss for hierarchical classification."""
        # Convert predictions back to logits for loss computation
        level_logits = []
        for pred in predictions:
            # Convert probabilities back to logits
            logits = torch.log(pred + 1e-8)
            level_logits.append(logits)
        
        return self.compute_hierarchical_loss(level_logits, targets)