"""TabGPT model for classification tasks."""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional

from .base import TabGPTPreTrainedModel, TabGPTModel
from ..config import TabGPTConfig


class TabGPTForClassification(TabGPTPreTrainedModel):
    """TabGPT model for classification tasks."""
    
    def __init__(self, config: TabGPTConfig, num_labels: int = 2):
        super().__init__(config)
        self.num_labels = num_labels
        
        # Base TabGPT model
        self.tabgpt = TabGPTModel(config)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, config.d_model // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model // 2, num_labels)
        )
        
        # Initialize weights
        self._init_classifier_weights()
    
    def _init_classifier_weights(self):
        """Initialize classifier weights."""
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self,
        input_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        column_metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for classification.
        
        Args:
            input_features: Tokenized tabular features
            attention_mask: Attention mask
            labels: Ground truth labels for training
            column_metadata: Column information
            
        Returns:
            Dictionary containing loss and logits
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
        
        # Classification logits
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            if self.num_labels == 1:
                # Regression case
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.squeeze(), labels.squeeze())
            else:
                # Classification case
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        
        return {
            "loss": loss,
            "logits": logits,
            "hidden_states": outputs["last_hidden_state"],
        }