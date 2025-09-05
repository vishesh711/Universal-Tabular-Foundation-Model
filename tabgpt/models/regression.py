"""TabGPT model for regression tasks."""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional

from .base import TabGPTPreTrainedModel, TabGPTModel
from ..config import TabGPTConfig


class TabGPTForRegression(TabGPTPreTrainedModel):
    """TabGPT model for regression tasks."""
    
    def __init__(self, config: TabGPTConfig, num_targets: int = 1):
        super().__init__(config)
        self.num_targets = num_targets
        
        # Base TabGPT model
        self.tabgpt = TabGPTModel(config)
        
        # Regression head
        self.regressor = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, config.d_model // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model // 2, num_targets)
        )
        
        # Initialize weights
        self.init_weights()
    
    def forward(
        self,
        input_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        column_metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for regression.
        
        Args:
            input_features: Tokenized tabular features
            attention_mask: Attention mask
            labels: Ground truth targets for training
            column_metadata: Column information
            
        Returns:
            Dictionary containing loss and predictions
        """
        # Get base model outputs
        outputs = self.tabgpt(
            input_features=input_features,
            attention_mask=attention_mask,
            column_metadata=column_metadata,
            **kwargs
        )
        
        # Get pooled representation
        pooled_output = outputs["pooler_output"]
        
        # Regression predictions
        predictions = self.regressor(pooled_output)
        
        loss = None
        if labels is not None:
            loss_fct = nn.MSELoss()
            if self.num_targets == 1:
                loss = loss_fct(predictions.squeeze(), labels.squeeze())
            else:
                loss = loss_fct(predictions, labels)
        
        return {
            "loss": loss,
            "predictions": predictions,
            "hidden_states": outputs["last_hidden_state"],
        }