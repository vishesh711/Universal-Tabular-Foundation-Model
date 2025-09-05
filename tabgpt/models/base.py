"""Base TabGPT model implementation."""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple
from abc import ABC, abstractmethod

from ..config import TabGPTConfig


class TabGPTPreTrainedModel(nn.Module, ABC):
    """Base class for all TabGPT models."""
    
    config_class = TabGPTConfig
    
    def __init__(self, config: TabGPTConfig):
        super().__init__()
        self.config = config
    
    def save_pretrained(self, save_directory: str) -> None:
        """Save model and config to directory."""
        import os
        os.makedirs(save_directory, exist_ok=True)
        
        # Save config
        self.config.save_pretrained(save_directory)
        
        # Save model weights
        model_path = os.path.join(save_directory, "pytorch_model.bin")
        torch.save(self.state_dict(), model_path)
    
    @classmethod
    def from_pretrained(cls, model_name_or_path: str, **kwargs) -> 'TabGPTPreTrainedModel':
        """Load pretrained model."""
        import os
        
        # Load config
        config = TabGPTConfig.from_pretrained(model_name_or_path)
        
        # Create model
        model = cls(config, **kwargs)
        
        # Load weights
        if os.path.isdir(model_name_or_path):
            model_path = os.path.join(model_name_or_path, "pytorch_model.bin")
            if os.path.exists(model_path):
                state_dict = torch.load(model_path, map_location='cpu')
                model.load_state_dict(state_dict)
        
        return model


class TabGPTModel(TabGPTPreTrainedModel):
    """
    The base TabGPT model for pre-training and feature extraction.
    
    This model combines row-level and column-level representations through
    a hybrid transformer architecture.
    """
    
    def __init__(self, config: TabGPTConfig):
        super().__init__(config)
        
        # Import here to avoid circular imports
        from .row_encoder import RowEncoder
        from .cross_attention import CrossAttentionFusion
        from ..encoders import ColumnEncoder
        
        # Core components
        self.row_encoder = RowEncoder(config)
        self.column_encoder = ColumnEncoder(
            embedding_dim=config.column_embedding_dim,
            statistical_features=config.statistical_features
        )
        
        # Cross-attention fusion
        self.cross_attention = CrossAttentionFusion(
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=getattr(config, 'cross_attention_layers', 2),
            dropout=config.dropout,
            fusion_strategy=getattr(config, 'fusion_strategy', 'gate')
        )
        
        # Initialize weights
        self.init_weights()
    
    def init_weights(self) -> None:
        """Initialize model weights."""
        # Initialize row encoder weights
        if hasattr(self, 'row_encoder'):
            self.row_encoder._init_weights()
        
        # Initialize column encoder weights  
        if hasattr(self, 'column_encoder'):
            self.column_encoder._init_weights()
    
    def forward(
        self,
        input_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        column_metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the TabGPT model.
        
        Args:
            input_features: Tokenized tabular features [batch_size, n_features, embedding_dim]
            attention_mask: Attention mask [batch_size, n_features]
            column_metadata: Dictionary containing column information
            
        Returns:
            Dictionary containing model outputs
        """
        # Process through row encoder
        row_outputs = self.row_encoder(input_features, attention_mask)
        row_embeddings = row_outputs["last_hidden_state"]
        
        # Column encoding (if metadata provided)
        column_embeddings = None
        if column_metadata is not None:
            # Extract column embeddings from metadata
            column_embeddings_list = self.column_encoder.encode_columns(
                column_metadata.get('column_metadata', []),
                column_metadata.get('dataframe', None)
            )
            column_embeddings = torch.stack([emb.combined_embedding for emb in column_embeddings_list])
        
        # Apply cross-attention fusion if column embeddings are available
        if column_embeddings is not None:
            fusion_outputs = self.cross_attention(
                row_embeddings=row_embeddings,
                column_embeddings=column_embeddings,
                row_attention_mask=attention_mask,
                return_attention_weights=kwargs.get('return_attention_weights', False)
            )
            
            # Use fused representations as the final output
            final_hidden_state = fusion_outputs['fused_representations']
            
            # Pool the fused representations
            if attention_mask is not None:
                # Masked mean pooling
                mask = attention_mask.unsqueeze(-1).expand_as(final_hidden_state)
                masked_hidden = final_hidden_state * mask.float()
                sum_hidden = masked_hidden.sum(dim=1)
                sum_mask = mask.float().sum(dim=1)
                pooler_output = sum_hidden / (sum_mask + 1e-8)
            else:
                pooler_output = final_hidden_state.mean(dim=1)
            
            outputs = {
                "last_hidden_state": final_hidden_state,
                "pooler_output": pooler_output,
                "row_embeddings": fusion_outputs['enhanced_row_embeddings'],
                "column_embeddings": fusion_outputs['enhanced_column_embeddings'],
            }
            
            # Add attention weights if requested
            if kwargs.get('return_attention_weights', False):
                outputs["attention_weights"] = fusion_outputs.get('attention_weights', [])
                outputs["row_attention_weights"] = row_outputs.get("attention_weights", [])
            
            return outputs
        else:
            # No column metadata provided, return row encoder outputs only
            return {
                "last_hidden_state": row_embeddings,
                "pooler_output": row_outputs["pooler_output"],
                "attention_weights": row_outputs.get("attention_weights", []),
            }