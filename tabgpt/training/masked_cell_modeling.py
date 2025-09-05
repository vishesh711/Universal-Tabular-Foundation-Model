"""Masked Cell Modeling (MCM) pre-training objective for TabGPT."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from dataclasses import dataclass

from ..tokenizers.tabular_tokenizer import ColumnMetadata


@dataclass
class MaskedCellOutput:
    """Output from masked cell modeling."""
    loss: torch.Tensor
    categorical_loss: Optional[torch.Tensor]
    numerical_loss: Optional[torch.Tensor]
    predictions: Dict[str, torch.Tensor]
    masked_positions: torch.Tensor
    accuracy: Dict[str, float]


class CategoricalPredictionHead(nn.Module):
    """Prediction head for categorical features."""
    
    def __init__(self, d_model: int, vocab_size: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        self.prediction_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, vocab_size)
        )
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for categorical prediction.
        
        Args:
            hidden_states: Hidden states [batch_size, seq_len, d_model]
            
        Returns:
            Logits for categorical prediction [batch_size, seq_len, vocab_size]
        """
        return self.prediction_head(hidden_states)


class NumericalPredictionHead(nn.Module):
    """Prediction head for numerical features."""
    
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        
        self.prediction_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1)
        )
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for numerical prediction.
        
        Args:
            hidden_states: Hidden states [batch_size, seq_len, d_model]
            
        Returns:
            Numerical predictions [batch_size, seq_len, 1]
        """
        return self.prediction_head(hidden_states)


class MaskedCellModelingHead(nn.Module):
    """
    Complete masked cell modeling head with support for different data types.
    """
    
    def __init__(
        self,
        d_model: int,
        categorical_vocab_size: int = 10000,
        dropout: float = 0.1,
        numerical_loss_weight: float = 1.0,
        categorical_loss_weight: float = 1.0
    ):
        super().__init__()
        self.d_model = d_model
        self.categorical_vocab_size = categorical_vocab_size
        self.numerical_loss_weight = numerical_loss_weight
        self.categorical_loss_weight = categorical_loss_weight
        
        # Prediction heads for different data types
        self.categorical_head = CategoricalPredictionHead(
            d_model, categorical_vocab_size, dropout
        )
        self.numerical_head = NumericalPredictionHead(d_model, dropout)
        
        # Loss functions
        self.categorical_loss_fn = nn.CrossEntropyLoss(reduction='none')
        self.numerical_loss_fn = nn.MSELoss(reduction='none')
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        masked_positions: torch.Tensor,
        original_values: torch.Tensor,
        feature_types: List[str],
        attention_mask: Optional[torch.Tensor] = None
    ) -> MaskedCellOutput:
        """
        Forward pass for masked cell modeling.
        
        Args:
            hidden_states: Model hidden states [batch_size, seq_len, d_model]
            masked_positions: Boolean mask of masked positions [batch_size, seq_len]
            original_values: Original values before masking [batch_size, seq_len]
            feature_types: List of feature types for each position
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            MaskedCellOutput with loss and predictions
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Get predictions for all positions
        categorical_logits = self.categorical_head(hidden_states)
        numerical_preds = self.numerical_head(hidden_states).squeeze(-1)
        
        # Initialize losses and accuracies
        total_loss = 0.0
        categorical_loss = None
        numerical_loss = None
        categorical_correct = 0
        categorical_total = 0
        numerical_mae = 0.0
        numerical_total = 0
        
        predictions = {
            'categorical': categorical_logits,
            'numerical': numerical_preds
        }
        
        # Compute losses only for masked positions
        if masked_positions.any():
            # Create masks for different feature types
            categorical_mask = torch.zeros_like(masked_positions, dtype=torch.bool)
            numerical_mask = torch.zeros_like(masked_positions, dtype=torch.bool)
            
            for i, feat_type in enumerate(feature_types):
                if feat_type in ['categorical', 'boolean']:
                    categorical_mask[:, i] = True
                elif feat_type in ['numerical', 'datetime']:
                    numerical_mask[:, i] = True
            
            # Combine with masked positions and attention mask
            categorical_positions = masked_positions & categorical_mask
            numerical_positions = masked_positions & numerical_mask
            
            if attention_mask is not None:
                categorical_positions = categorical_positions & attention_mask
                numerical_positions = numerical_positions & attention_mask
            
            # Categorical loss
            if categorical_positions.any():
                cat_logits_masked = categorical_logits[categorical_positions]
                cat_targets_masked = original_values[categorical_positions].long()
                
                # Ensure targets are within vocabulary range
                cat_targets_masked = torch.clamp(cat_targets_masked, 0, self.categorical_vocab_size - 1)
                
                cat_loss_per_token = self.categorical_loss_fn(cat_logits_masked, cat_targets_masked)
                categorical_loss = cat_loss_per_token.mean()
                total_loss += self.categorical_loss_weight * categorical_loss
                
                # Compute accuracy
                cat_preds = cat_logits_masked.argmax(dim=-1)
                categorical_correct = (cat_preds == cat_targets_masked).sum().item()
                categorical_total = categorical_positions.sum().item()
            
            # Numerical loss
            if numerical_positions.any():
                num_preds_masked = numerical_preds[numerical_positions]
                num_targets_masked = original_values[numerical_positions].float()
                
                num_loss_per_token = self.numerical_loss_fn(num_preds_masked, num_targets_masked)
                numerical_loss = num_loss_per_token.mean()
                total_loss += self.numerical_loss_weight * numerical_loss
                
                # Compute MAE
                numerical_mae = F.l1_loss(num_preds_masked, num_targets_masked).item()
                numerical_total = numerical_positions.sum().item()
        
        # Compute accuracies
        accuracy = {
            'categorical': categorical_correct / max(categorical_total, 1),
            'numerical_mae': numerical_mae,
            'total_masked': masked_positions.sum().item()
        }
        
        return MaskedCellOutput(
            loss=total_loss,
            categorical_loss=categorical_loss,
            numerical_loss=numerical_loss,
            predictions=predictions,
            masked_positions=masked_positions,
            accuracy=accuracy
        )


class MaskedCellModelingObjective(nn.Module):
    """
    Complete Masked Cell Modeling objective with masking strategy.
    """
    
    def __init__(
        self,
        d_model: int,
        mask_probability: float = 0.15,
        replace_probability: float = 0.8,
        random_probability: float = 0.1,
        categorical_vocab_size: int = 10000,
        mask_token_id: int = 0,
        pad_token_id: int = 1,
        dropout: float = 0.1,
        numerical_loss_weight: float = 1.0,
        categorical_loss_weight: float = 1.0
    ):
        super().__init__()
        self.mask_probability = mask_probability
        self.replace_probability = replace_probability
        self.random_probability = random_probability
        self.categorical_vocab_size = categorical_vocab_size
        self.mask_token_id = mask_token_id
        self.pad_token_id = pad_token_id
        
        # Prediction head
        self.prediction_head = MaskedCellModelingHead(
            d_model=d_model,
            categorical_vocab_size=categorical_vocab_size,
            dropout=dropout,
            numerical_loss_weight=numerical_loss_weight,
            categorical_loss_weight=categorical_loss_weight
        )
        
    def create_masked_inputs(
        self,
        input_embeddings: torch.Tensor,
        input_values: torch.Tensor,
        feature_types: List[str],
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Create masked inputs for pre-training.
        
        Args:
            input_embeddings: Input embeddings [batch_size, seq_len, d_model]
            input_values: Original input values [batch_size, seq_len]
            feature_types: List of feature types
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Tuple of (masked_embeddings, masked_positions, original_values)
        """
        batch_size, seq_len, d_model = input_embeddings.shape
        device = input_embeddings.device
        
        # Create random mask
        mask_probs = torch.rand(batch_size, seq_len, device=device)
        
        # Don't mask padded positions
        if attention_mask is not None:
            mask_probs = mask_probs * attention_mask.float()
        
        # Create masked positions
        masked_positions = mask_probs < self.mask_probability
        
        # Clone inputs for modification
        masked_embeddings = input_embeddings.clone()
        original_values = input_values.clone()
        
        if masked_positions.any():
            # For each masked position, decide what to do
            replace_probs = torch.rand(batch_size, seq_len, device=device)
            
            # 80% of the time: replace with mask token
            replace_with_mask = masked_positions & (replace_probs < self.replace_probability)
            
            # 10% of the time: replace with random token
            replace_with_random = (
                masked_positions & 
                (replace_probs >= self.replace_probability) & 
                (replace_probs < self.replace_probability + self.random_probability)
            )
            
            # 10% of the time: keep original (no change needed)
            
            # Replace with mask token (zero embedding for simplicity)
            if replace_with_mask.any():
                masked_embeddings[replace_with_mask] = 0.0
            
            # Replace with random embeddings
            if replace_with_random.any():
                random_embeddings = torch.randn_like(masked_embeddings[replace_with_random])
                masked_embeddings[replace_with_random] = random_embeddings
        
        return masked_embeddings, masked_positions, original_values
    
    def forward(
        self,
        input_embeddings: torch.Tensor,
        input_values: torch.Tensor,
        feature_types: List[str],
        model_forward_fn: callable,
        attention_mask: Optional[torch.Tensor] = None,
        **model_kwargs
    ) -> MaskedCellOutput:
        """
        Forward pass for masked cell modeling objective.
        
        Args:
            input_embeddings: Input embeddings [batch_size, seq_len, d_model]
            input_values: Original input values [batch_size, seq_len]
            feature_types: List of feature types
            model_forward_fn: Function to call model forward pass
            attention_mask: Attention mask [batch_size, seq_len]
            **model_kwargs: Additional arguments for model forward
            
        Returns:
            MaskedCellOutput with loss and predictions
        """
        # Create masked inputs
        masked_embeddings, masked_positions, original_values = self.create_masked_inputs(
            input_embeddings, input_values, feature_types, attention_mask
        )
        
        # Forward pass through model with masked inputs
        model_outputs = model_forward_fn(
            input_features=masked_embeddings,
            attention_mask=attention_mask,
            **model_kwargs
        )
        
        # Get hidden states
        hidden_states = model_outputs['last_hidden_state']
        
        # Compute masked cell modeling loss
        mcm_output = self.prediction_head(
            hidden_states=hidden_states,
            masked_positions=masked_positions,
            original_values=original_values,
            feature_types=feature_types,
            attention_mask=attention_mask
        )
        
        return mcm_output
    
    def compute_metrics(self, outputs: MaskedCellOutput) -> Dict[str, float]:
        """
        Compute evaluation metrics for masked cell modeling.
        
        Args:
            outputs: MaskedCellOutput from forward pass
            
        Returns:
            Dictionary of metrics
        """
        metrics = {
            'mcm_loss': outputs.loss.item() if outputs.loss is not None else 0.0,
            'categorical_accuracy': outputs.accuracy['categorical'],
            'numerical_mae': outputs.accuracy['numerical_mae'],
            'total_masked_cells': outputs.accuracy['total_masked'],
            'mask_ratio': outputs.masked_positions.float().mean().item()
        }
        
        if outputs.categorical_loss is not None:
            metrics['categorical_loss'] = outputs.categorical_loss.item()
        
        if outputs.numerical_loss is not None:
            metrics['numerical_loss'] = outputs.numerical_loss.item()
        
        return metrics