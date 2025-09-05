"""Base classes for task-specific heads."""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass
from enum import Enum


class TaskType(Enum):
    """Types of downstream tasks."""
    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    MULTILABEL_CLASSIFICATION = "multilabel_classification"
    REGRESSION = "regression"
    MULTI_TARGET_REGRESSION = "multi_target_regression"
    QUANTILE_REGRESSION = "quantile_regression"
    ANOMALY_DETECTION = "anomaly_detection"
    SURVIVAL_ANALYSIS = "survival_analysis"
    RANKING = "ranking"
    CLUSTERING = "clustering"


@dataclass
class TaskOutput:
    """Output from a task-specific head."""
    predictions: torch.Tensor
    loss: Optional[torch.Tensor] = None
    probabilities: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None
    features: Optional[torch.Tensor] = None
    attention_weights: Optional[torch.Tensor] = None
    metadata: Optional[Dict[str, Any]] = None


class BaseTaskHead(nn.Module, ABC):
    """Base class for all task-specific heads."""
    
    def __init__(
        self,
        input_dim: int,
        task_type: TaskType,
        dropout: float = 0.1,
        activation: str = "relu",
        use_batch_norm: bool = False,
        use_layer_norm: bool = True
    ):
        super().__init__()
        self.input_dim = input_dim
        self.task_type = task_type
        self.dropout = dropout
        self.activation = activation
        self.use_batch_norm = use_batch_norm
        self.use_layer_norm = use_layer_norm
        
        # Common components
        self.dropout_layer = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Activation function
        if activation == "relu":
            self.activation_fn = nn.ReLU()
        elif activation == "gelu":
            self.activation_fn = nn.GELU()
        elif activation == "tanh":
            self.activation_fn = nn.Tanh()
        elif activation == "sigmoid":
            self.activation_fn = nn.Sigmoid()
        else:
            self.activation_fn = nn.ReLU()  # Default
        
        # Normalization
        if use_layer_norm:
            self.norm_layer = nn.LayerNorm(input_dim)
        elif use_batch_norm:
            self.norm_layer = nn.BatchNorm1d(input_dim)
        else:
            self.norm_layer = nn.Identity()
    
    @abstractmethod
    def forward(
        self,
        features: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        **kwargs
    ) -> TaskOutput:
        """Forward pass through the task head."""
        pass
    
    @abstractmethod
    def compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """Compute task-specific loss."""
        pass
    
    def get_predictions(self, logits: torch.Tensor) -> torch.Tensor:
        """Convert logits to predictions."""
        if self.task_type in [TaskType.BINARY_CLASSIFICATION, TaskType.MULTILABEL_CLASSIFICATION]:
            return torch.sigmoid(logits)
        elif self.task_type == TaskType.MULTICLASS_CLASSIFICATION:
            return torch.softmax(logits, dim=-1)
        else:
            return logits  # For regression tasks
    
    def get_probabilities(self, logits: torch.Tensor) -> Optional[torch.Tensor]:
        """Get class probabilities if applicable."""
        if self.task_type in [
            TaskType.BINARY_CLASSIFICATION,
            TaskType.MULTICLASS_CLASSIFICATION,
            TaskType.MULTILABEL_CLASSIFICATION
        ]:
            return self.get_predictions(logits)
        return None
    
    def compute_metrics(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        **kwargs
    ) -> Dict[str, float]:
        """Compute task-specific metrics."""
        metrics = {}
        
        if self.task_type == TaskType.BINARY_CLASSIFICATION:
            # Binary classification metrics
            pred_classes = (predictions > 0.5).float()
            accuracy = (pred_classes == targets).float().mean().item()
            metrics['accuracy'] = accuracy
            
            # Precision, Recall, F1 (simplified)
            tp = ((pred_classes == 1) & (targets == 1)).sum().item()
            fp = ((pred_classes == 1) & (targets == 0)).sum().item()
            fn = ((pred_classes == 0) & (targets == 1)).sum().item()
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            metrics.update({
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            })
        
        elif self.task_type == TaskType.MULTICLASS_CLASSIFICATION:
            # Multi-class classification metrics
            pred_classes = predictions.argmax(dim=-1)
            accuracy = (pred_classes == targets).float().mean().item()
            metrics['accuracy'] = accuracy
        
        elif self.task_type in [TaskType.REGRESSION, TaskType.MULTI_TARGET_REGRESSION]:
            # Regression metrics
            mse = torch.mean((predictions - targets) ** 2).item()
            mae = torch.mean(torch.abs(predictions - targets)).item()
            
            metrics.update({
                'mse': mse,
                'mae': mae,
                'rmse': mse ** 0.5
            })
        
        return metrics


class MLPHead(BaseTaskHead):
    """Multi-layer perceptron head for various tasks."""
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        task_type: TaskType,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.1,
        activation: str = "relu",
        use_batch_norm: bool = False,
        use_layer_norm: bool = True,
        final_activation: Optional[str] = None
    ):
        super().__init__(input_dim, task_type, dropout, activation, use_batch_norm, use_layer_norm)
        
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims or [input_dim // 2]
        self.final_activation = final_activation
        
        # Build MLP layers
        layers = []
        prev_dim = input_dim
        
        # Hidden layers
        for hidden_dim in self.hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            if self.use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            elif self.use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            
            layers.append(self.activation_fn)
            layers.append(self.dropout_layer)
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        # Final activation
        if final_activation == "sigmoid":
            layers.append(nn.Sigmoid())
        elif final_activation == "softmax":
            layers.append(nn.Softmax(dim=-1))
        elif final_activation == "tanh":
            layers.append(nn.Tanh())
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(
        self,
        features: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        **kwargs
    ) -> TaskOutput:
        """Forward pass through MLP head."""
        # Apply normalization to input features
        features = self.norm_layer(features)
        
        # Forward through MLP
        logits = self.mlp(features)
        
        # Get predictions and probabilities
        predictions = self.get_predictions(logits)
        probabilities = self.get_probabilities(logits)
        
        # Compute loss if targets provided
        loss = None
        if targets is not None:
            loss = self.compute_loss(predictions, targets, **kwargs)
        
        return TaskOutput(
            predictions=predictions,
            loss=loss,
            probabilities=probabilities,
            logits=logits,
            features=features
        )
    
    def compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """Compute loss based on task type."""
        if self.task_type == TaskType.BINARY_CLASSIFICATION:
            return nn.BCELoss()(predictions.squeeze(), targets.float())
        
        elif self.task_type == TaskType.MULTICLASS_CLASSIFICATION:
            return nn.CrossEntropyLoss()(predictions, targets.long())
        
        elif self.task_type == TaskType.MULTILABEL_CLASSIFICATION:
            return nn.BCELoss()(predictions, targets.float())
        
        elif self.task_type in [TaskType.REGRESSION, TaskType.MULTI_TARGET_REGRESSION]:
            return nn.MSELoss()(predictions, targets.float())
        
        else:
            raise NotImplementedError(f"Loss not implemented for task type: {self.task_type}")


class AttentionPoolingHead(BaseTaskHead):
    """Head with attention-based pooling for sequence inputs."""
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        task_type: TaskType,
        attention_dim: Optional[int] = None,
        dropout: float = 0.1,
        activation: str = "relu"
    ):
        super().__init__(input_dim, task_type, dropout, activation)
        
        self.output_dim = output_dim
        self.attention_dim = attention_dim or input_dim // 2
        
        # Attention mechanism
        self.attention_weights = nn.Sequential(
            nn.Linear(input_dim, self.attention_dim),
            self.activation_fn,
            nn.Linear(self.attention_dim, 1)
        )
        
        # Output projection
        self.output_projection = nn.Linear(input_dim, output_dim)
    
    def forward(
        self,
        features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
        **kwargs
    ) -> TaskOutput:
        """Forward pass with attention pooling."""
        # features: [batch_size, seq_len, input_dim]
        batch_size, seq_len, input_dim = features.shape
        
        # Compute attention weights
        attention_scores = self.attention_weights(features)  # [batch_size, seq_len, 1]
        attention_scores = attention_scores.squeeze(-1)  # [batch_size, seq_len]
        
        # Apply attention mask if provided
        if attention_mask is not None:
            attention_scores = attention_scores.masked_fill(~attention_mask, -1e9)
        
        # Softmax to get attention weights
        attention_weights = torch.softmax(attention_scores, dim=-1)  # [batch_size, seq_len]
        
        # Apply attention pooling
        pooled_features = torch.sum(
            features * attention_weights.unsqueeze(-1), dim=1
        )  # [batch_size, input_dim]
        
        # Output projection
        logits = self.output_projection(pooled_features)
        
        # Get predictions and probabilities
        predictions = self.get_predictions(logits)
        probabilities = self.get_probabilities(logits)
        
        # Compute loss if targets provided
        loss = None
        if targets is not None:
            loss = self.compute_loss(predictions, targets, **kwargs)
        
        return TaskOutput(
            predictions=predictions,
            loss=loss,
            probabilities=probabilities,
            logits=logits,
            features=pooled_features,
            attention_weights=attention_weights
        )
    
    def compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """Compute loss based on task type."""
        if self.task_type == TaskType.BINARY_CLASSIFICATION:
            return nn.BCELoss()(predictions.squeeze(), targets.float())
        
        elif self.task_type == TaskType.MULTICLASS_CLASSIFICATION:
            return nn.CrossEntropyLoss()(predictions, targets.long())
        
        elif self.task_type == TaskType.MULTILABEL_CLASSIFICATION:
            return nn.BCELoss()(predictions, targets.float())
        
        elif self.task_type in [TaskType.REGRESSION, TaskType.MULTI_TARGET_REGRESSION]:
            return nn.MSELoss()(predictions, targets.float())
        
        else:
            raise NotImplementedError(f"Loss not implemented for task type: {self.task_type}")


class ResidualHead(BaseTaskHead):
    """Head with residual connections for deep architectures."""
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        task_type: TaskType,
        n_layers: int = 3,
        dropout: float = 0.1,
        activation: str = "relu"
    ):
        super().__init__(input_dim, task_type, dropout, activation)
        
        self.output_dim = output_dim
        self.n_layers = n_layers
        
        # Residual blocks
        self.residual_blocks = nn.ModuleList()
        for _ in range(n_layers):
            block = nn.Sequential(
                nn.Linear(input_dim, input_dim),
                nn.LayerNorm(input_dim),
                self.activation_fn,
                self.dropout_layer,
                nn.Linear(input_dim, input_dim),
                nn.LayerNorm(input_dim)
            )
            self.residual_blocks.append(block)
        
        # Output projection
        self.output_projection = nn.Linear(input_dim, output_dim)
    
    def forward(
        self,
        features: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        **kwargs
    ) -> TaskOutput:
        """Forward pass through residual blocks."""
        x = features
        
        # Apply residual blocks
        for block in self.residual_blocks:
            residual = x
            x = block(x)
            x = x + residual  # Residual connection
            x = self.activation_fn(x)
        
        # Output projection
        logits = self.output_projection(x)
        
        # Get predictions and probabilities
        predictions = self.get_predictions(logits)
        probabilities = self.get_probabilities(logits)
        
        # Compute loss if targets provided
        loss = None
        if targets is not None:
            loss = self.compute_loss(predictions, targets, **kwargs)
        
        return TaskOutput(
            predictions=predictions,
            loss=loss,
            probabilities=probabilities,
            logits=logits,
            features=x
        )
    
    def compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """Compute loss based on task type."""
        if self.task_type == TaskType.BINARY_CLASSIFICATION:
            return nn.BCELoss()(predictions.squeeze(), targets.float())
        
        elif self.task_type == TaskType.MULTICLASS_CLASSIFICATION:
            return nn.CrossEntropyLoss()(predictions, targets.long())
        
        elif self.task_type == TaskType.MULTILABEL_CLASSIFICATION:
            return nn.BCELoss()(predictions, targets.float())
        
        elif self.task_type in [TaskType.REGRESSION, TaskType.MULTI_TARGET_REGRESSION]:
            return nn.MSELoss()(predictions, targets.float())
        
        else:
            raise NotImplementedError(f"Loss not implemented for task type: {self.task_type}")