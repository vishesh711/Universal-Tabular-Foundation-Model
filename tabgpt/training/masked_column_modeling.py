"""Masked Column Modeling (MCM) pre-training objective for TabGPT."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from dataclasses import dataclass

from ..tokenizers.tabular_tokenizer import ColumnMetadata


@dataclass
class MaskedColumnOutput:
    """Output from masked column modeling."""
    loss: torch.Tensor
    column_type_loss: Optional[torch.Tensor]
    column_stats_loss: Optional[torch.Tensor]
    column_correlation_loss: Optional[torch.Tensor]
    predictions: Dict[str, torch.Tensor]
    masked_columns: torch.Tensor
    accuracy: Dict[str, float]


class ColumnTypePredictionHead(nn.Module):
    """Prediction head for column data types."""
    
    def __init__(self, d_model: int, num_types: int = 6, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_types = num_types  # categorical, numerical, boolean, datetime, text, mixed
        
        self.prediction_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_types)
        )
        
    def forward(self, column_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for column type prediction.
        
        Args:
            column_embeddings: Column embeddings [batch_size, n_columns, d_model]
            
        Returns:
            Type logits [batch_size, n_columns, num_types]
        """
        return self.prediction_head(column_embeddings)


class ColumnStatsPredictionHead(nn.Module):
    """Prediction head for column statistics."""
    
    def __init__(self, d_model: int, num_stats: int = 8, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_stats = num_stats  # mean, std, min, max, median, skewness, kurtosis, null_ratio
        
        self.prediction_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_stats)
        )
        
    def forward(self, column_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for column statistics prediction.
        
        Args:
            column_embeddings: Column embeddings [batch_size, n_columns, d_model]
            
        Returns:
            Statistics predictions [batch_size, n_columns, num_stats]
        """
        return self.prediction_head(column_embeddings)


class ColumnCorrelationHead(nn.Module):
    """Prediction head for column correlations."""
    
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        
        # Bilinear layer for computing correlations
        self.correlation_layer = nn.Bilinear(d_model, d_model, 1)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, column_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for column correlation prediction.
        
        Args:
            column_embeddings: Column embeddings [batch_size, n_columns, d_model]
            
        Returns:
            Correlation matrix [batch_size, n_columns, n_columns]
        """
        batch_size, n_columns, d_model = column_embeddings.shape
        
        # Compute pairwise correlations
        correlations = torch.zeros(batch_size, n_columns, n_columns, device=column_embeddings.device)
        
        for i in range(n_columns):
            for j in range(n_columns):
                if i == j:
                    correlations[:, i, j] = 1.0  # Self-correlation is 1
                else:
                    corr = self.correlation_layer(
                        column_embeddings[:, i, :],
                        column_embeddings[:, j, :]
                    ).squeeze(-1)
                    correlations[:, i, j] = torch.tanh(corr)  # Bound between -1 and 1
        
        return correlations


class MaskedColumnModelingHead(nn.Module):
    """
    Complete masked column modeling head with multiple prediction tasks.
    """
    
    def __init__(
        self,
        d_model: int,
        num_column_types: int = 6,
        num_stats: int = 8,
        dropout: float = 0.1,
        type_loss_weight: float = 1.0,
        stats_loss_weight: float = 1.0,
        correlation_loss_weight: float = 0.5
    ):
        super().__init__()
        self.d_model = d_model
        self.num_column_types = num_column_types
        self.num_stats = num_stats
        self.type_loss_weight = type_loss_weight
        self.stats_loss_weight = stats_loss_weight
        self.correlation_loss_weight = correlation_loss_weight
        
        # Prediction heads
        self.type_head = ColumnTypePredictionHead(d_model, num_column_types, dropout)
        self.stats_head = ColumnStatsPredictionHead(d_model, num_stats, dropout)
        self.correlation_head = ColumnCorrelationHead(d_model, dropout)
        
        # Loss functions
        self.type_loss_fn = nn.CrossEntropyLoss(reduction='none')
        self.stats_loss_fn = nn.MSELoss(reduction='none')
        self.correlation_loss_fn = nn.MSELoss(reduction='none')
        
        # Column type mapping
        self.type_mapping = {
            'categorical': 0,
            'numerical': 1,
            'boolean': 2,
            'datetime': 3,
            'text': 4,
            'mixed': 5
        }
        
    def forward(
        self,
        column_embeddings: torch.Tensor,
        masked_columns: torch.Tensor,
        column_metadata: List[ColumnMetadata],
        column_statistics: Optional[torch.Tensor] = None,
        column_correlations: Optional[torch.Tensor] = None
    ) -> MaskedColumnOutput:
        """
        Forward pass for masked column modeling.
        
        Args:
            column_embeddings: Column embeddings [batch_size, n_columns, d_model]
            masked_columns: Boolean mask of masked columns [batch_size, n_columns]
            column_metadata: List of column metadata
            column_statistics: Ground truth statistics [batch_size, n_columns, num_stats]
            column_correlations: Ground truth correlations [batch_size, n_columns, n_columns]
            
        Returns:
            MaskedColumnOutput with loss and predictions
        """
        batch_size, n_columns, _ = column_embeddings.shape
        
        # Get predictions for all columns
        type_logits = self.type_head(column_embeddings)
        stats_preds = self.stats_head(column_embeddings)
        correlation_preds = self.correlation_head(column_embeddings)
        
        # Initialize losses and accuracies
        total_loss = 0.0
        type_loss = None
        stats_loss = None
        correlation_loss = None
        type_correct = 0
        type_total = 0
        stats_mae = 0.0
        correlation_mae = 0.0
        
        predictions = {
            'column_types': type_logits,
            'column_stats': stats_preds,
            'column_correlations': correlation_preds
        }
        
        # Compute losses only for masked columns
        if masked_columns.any():
            # Column type prediction loss
            if len(column_metadata) > 0:
                # Create type targets
                type_targets = torch.zeros(batch_size, n_columns, dtype=torch.long, device=column_embeddings.device)
                for i, metadata in enumerate(column_metadata):
                    if i < n_columns:
                        type_id = self.type_mapping.get(metadata.dtype, 5)  # Default to 'mixed'
                        type_targets[:, i] = type_id
                
                # Compute type loss for masked columns
                masked_type_logits = type_logits[masked_columns]
                masked_type_targets = type_targets[masked_columns]
                
                if len(masked_type_logits) > 0:
                    type_loss_per_token = self.type_loss_fn(masked_type_logits, masked_type_targets)
                    type_loss = type_loss_per_token.mean()
                    total_loss += self.type_loss_weight * type_loss
                    
                    # Compute accuracy
                    type_preds = masked_type_logits.argmax(dim=-1)
                    type_correct = (type_preds == masked_type_targets).sum().item()
                    type_total = len(masked_type_targets)
            
            # Column statistics loss
            if column_statistics is not None:
                masked_stats_preds = stats_preds[masked_columns]
                masked_stats_targets = column_statistics[masked_columns]
                
                if len(masked_stats_preds) > 0:
                    stats_loss_per_token = self.stats_loss_fn(masked_stats_preds, masked_stats_targets)
                    stats_loss = stats_loss_per_token.mean()
                    total_loss += self.stats_loss_weight * stats_loss
                    
                    # Compute MAE
                    stats_mae = F.l1_loss(masked_stats_preds, masked_stats_targets).item()
            
            # Column correlation loss
            if column_correlations is not None:
                # Only compute correlation loss for pairs involving masked columns
                mask_expanded = masked_columns.unsqueeze(-1) | masked_columns.unsqueeze(-2)
                
                masked_corr_preds = correlation_preds[mask_expanded]
                masked_corr_targets = column_correlations[mask_expanded]
                
                if len(masked_corr_preds) > 0:
                    corr_loss_per_token = self.correlation_loss_fn(masked_corr_preds, masked_corr_targets)
                    correlation_loss = corr_loss_per_token.mean()
                    total_loss += self.correlation_loss_weight * correlation_loss
                    
                    # Compute MAE
                    correlation_mae = F.l1_loss(masked_corr_preds, masked_corr_targets).item()
        
        # Compute accuracies
        accuracy = {
            'column_type': type_correct / max(type_total, 1),
            'stats_mae': stats_mae,
            'correlation_mae': correlation_mae,
            'total_masked_columns': masked_columns.sum().item()
        }
        
        return MaskedColumnOutput(
            loss=total_loss,
            column_type_loss=type_loss,
            column_stats_loss=stats_loss,
            column_correlation_loss=correlation_loss,
            predictions=predictions,
            masked_columns=masked_columns,
            accuracy=accuracy
        )


class MaskedColumnModelingObjective(nn.Module):
    """
    Complete Masked Column Modeling objective with column masking strategy.
    """
    
    def __init__(
        self,
        d_model: int,
        mask_probability: float = 0.15,
        num_column_types: int = 6,
        num_stats: int = 8,
        dropout: float = 0.1,
        type_loss_weight: float = 1.0,
        stats_loss_weight: float = 1.0,
        correlation_loss_weight: float = 0.5,
        min_masked_columns: int = 1
    ):
        super().__init__()
        self.mask_probability = mask_probability
        self.min_masked_columns = min_masked_columns
        
        # Prediction head
        self.prediction_head = MaskedColumnModelingHead(
            d_model=d_model,
            num_column_types=num_column_types,
            num_stats=num_stats,
            dropout=dropout,
            type_loss_weight=type_loss_weight,
            stats_loss_weight=stats_loss_weight,
            correlation_loss_weight=correlation_loss_weight
        )
        
    def create_column_mask(
        self,
        batch_size: int,
        n_columns: int,
        device: torch.device
    ) -> torch.Tensor:
        """
        Create column mask for pre-training.
        
        Args:
            batch_size: Batch size
            n_columns: Number of columns
            device: Device to create mask on
            
        Returns:
            Boolean mask of columns to mask [batch_size, n_columns]
        """
        # Create random mask
        mask_probs = torch.rand(batch_size, n_columns, device=device)
        masked_columns = mask_probs < self.mask_probability
        
        # Ensure at least min_masked_columns are masked per sample
        for i in range(batch_size):
            if masked_columns[i].sum() < self.min_masked_columns:
                # Randomly select columns to mask
                available_columns = torch.arange(n_columns, device=device)
                n_to_mask = min(self.min_masked_columns, n_columns)
                selected = torch.randperm(n_columns, device=device)[:n_to_mask]
                masked_columns[i, selected] = True
        
        return masked_columns
    
    def compute_column_statistics(
        self,
        data: torch.Tensor,
        column_metadata: List[ColumnMetadata],
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute column statistics for ground truth.
        
        Args:
            data: Input data [batch_size, n_columns, seq_len] or [batch_size, n_columns]
            column_metadata: Column metadata
            attention_mask: Attention mask
            
        Returns:
            Column statistics [batch_size, n_columns, num_stats]
        """
        batch_size = data.shape[0]
        n_columns = len(column_metadata)
        num_stats = 8
        
        stats = torch.zeros(batch_size, n_columns, num_stats, device=data.device)
        
        for i, metadata in enumerate(column_metadata):
            if i >= data.shape[1]:
                break
                
            if metadata.dtype in ['numerical', 'datetime']:
                # For numerical columns, compute actual statistics
                col_data = data[:, i] if len(data.shape) == 2 else data[:, i].mean(dim=-1)
                
                # Handle attention mask
                if attention_mask is not None and len(data.shape) > 2:
                    mask = attention_mask[:, i] if attention_mask.shape[1] > i else attention_mask.any(dim=1)
                    col_data = col_data * mask.float()
                
                # Compute statistics (normalized)
                stats[:, i, 0] = col_data.mean(dim=0) if len(col_data.shape) > 0 else col_data  # mean
                stats[:, i, 1] = col_data.std(dim=0) if len(col_data.shape) > 0 else 0.0  # std
                stats[:, i, 2] = col_data.min(dim=0)[0] if len(col_data.shape) > 0 else col_data  # min
                stats[:, i, 3] = col_data.max(dim=0)[0] if len(col_data.shape) > 0 else col_data  # max
                stats[:, i, 4] = col_data.median(dim=0)[0] if len(col_data.shape) > 0 else col_data  # median
                # Simplified skewness and kurtosis (set to 0 for now)
                stats[:, i, 5] = 0.0  # skewness
                stats[:, i, 6] = 0.0  # kurtosis
                stats[:, i, 7] = 0.0  # null_ratio (simplified)
            else:
                # For categorical columns, use simplified statistics
                stats[:, i, :] = torch.randn(batch_size, num_stats, device=data.device) * 0.1
        
        return stats
    
    def compute_column_correlations(
        self,
        data: torch.Tensor,
        column_metadata: List[ColumnMetadata],
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute column correlations for ground truth.
        
        Args:
            data: Input data [batch_size, n_columns, seq_len] or [batch_size, n_columns]
            column_metadata: Column metadata
            attention_mask: Attention mask
            
        Returns:
            Column correlations [batch_size, n_columns, n_columns]
        """
        batch_size, n_columns = data.shape[0], len(column_metadata)
        
        correlations = torch.zeros(batch_size, n_columns, n_columns, device=data.device)
        
        # Simplified correlation computation
        for i in range(n_columns):
            for j in range(n_columns):
                if i == j:
                    correlations[:, i, j] = 1.0
                elif i < data.shape[1] and j < data.shape[1]:
                    # Compute correlation between columns i and j
                    col_i = data[:, i] if len(data.shape) == 2 else data[:, i].mean(dim=-1)
                    col_j = data[:, j] if len(data.shape) == 2 else data[:, j].mean(dim=-1)
                    
                    # Simple correlation (can be improved)
                    corr = torch.cosine_similarity(col_i.unsqueeze(0), col_j.unsqueeze(0), dim=1)
                    correlations[:, i, j] = corr
                    correlations[:, j, i] = corr  # Symmetric
        
        return correlations
    
    def forward(
        self,
        column_embeddings: torch.Tensor,
        input_data: torch.Tensor,
        column_metadata: List[ColumnMetadata],
        attention_mask: Optional[torch.Tensor] = None
    ) -> MaskedColumnOutput:
        """
        Forward pass for masked column modeling objective.
        
        Args:
            column_embeddings: Column embeddings [batch_size, n_columns, d_model]
            input_data: Original input data for computing ground truth
            column_metadata: Column metadata
            attention_mask: Attention mask
            
        Returns:
            MaskedColumnOutput with loss and predictions
        """
        batch_size, n_columns, d_model = column_embeddings.shape
        device = column_embeddings.device
        
        # Create column mask
        masked_columns = self.create_column_mask(batch_size, n_columns, device)
        
        # Compute ground truth statistics and correlations
        column_statistics = self.compute_column_statistics(
            input_data, column_metadata, attention_mask
        )
        column_correlations = self.compute_column_correlations(
            input_data, column_metadata, attention_mask
        )
        
        # Forward pass through prediction head
        mcm_output = self.prediction_head(
            column_embeddings=column_embeddings,
            masked_columns=masked_columns,
            column_metadata=column_metadata,
            column_statistics=column_statistics,
            column_correlations=column_correlations
        )
        
        return mcm_output
    
    def compute_metrics(self, outputs: MaskedColumnOutput) -> Dict[str, float]:
        """
        Compute evaluation metrics for masked column modeling.
        
        Args:
            outputs: MaskedColumnOutput from forward pass
            
        Returns:
            Dictionary of metrics
        """
        metrics = {
            'mcm_loss': outputs.loss.item() if outputs.loss is not None else 0.0,
            'column_type_accuracy': outputs.accuracy['column_type'],
            'stats_mae': outputs.accuracy['stats_mae'],
            'correlation_mae': outputs.accuracy['correlation_mae'],
            'total_masked_columns': outputs.accuracy['total_masked_columns'],
            'column_mask_ratio': outputs.masked_columns.float().mean().item()
        }
        
        if outputs.column_type_loss is not None:
            metrics['column_type_loss'] = outputs.column_type_loss.item()
        
        if outputs.column_stats_loss is not None:
            metrics['column_stats_loss'] = outputs.column_stats_loss.item()
        
        if outputs.column_correlation_loss is not None:
            metrics['column_correlation_loss'] = outputs.column_correlation_loss.item()
        
        return metrics