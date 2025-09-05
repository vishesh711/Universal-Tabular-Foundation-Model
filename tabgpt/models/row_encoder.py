"""FT-Transformer-based row encoder for tabular data."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict, Any

from ..config import TabGPTConfig


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism for tabular features."""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
        
    def forward(
        self, 
        query: torch.Tensor, 
        key: torch.Tensor, 
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of multi-head attention.
        
        Args:
            query: Query tensor [batch_size, seq_len, d_model]
            key: Key tensor [batch_size, seq_len, d_model]
            value: Value tensor [batch_size, seq_len, d_model]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Tuple of (output, attention_weights)
        """
        batch_size, seq_len, _ = query.shape
        
        # Linear projections
        Q = self.w_q(query).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # Apply attention mask if provided
        if attention_mask is not None:
            # Create causal mask: [batch_size, seq_len] -> [batch_size, seq_len, seq_len]
            mask = attention_mask.unsqueeze(1) & attention_mask.unsqueeze(2)
            # Expand for multi-head attention: [batch_size, 1, seq_len, seq_len]
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(~mask, float('-inf'))
        
        attention_weights = F.softmax(scores, dim=-1)
        
        # Handle NaN values that can occur when all values are masked
        if attention_mask is not None:
            # Zero out attention weights where the query is masked
            query_mask = attention_mask.unsqueeze(1).unsqueeze(-1)  # [batch_size, 1, seq_len, 1]
            attention_weights = attention_weights * query_mask.float()
            
            # Normalize attention weights to sum to 1 for valid queries
            attention_sum = attention_weights.sum(dim=-1, keepdim=True)
            attention_weights = torch.where(
                attention_sum > 0,
                attention_weights / (attention_sum + 1e-8),
                attention_weights
            )
        
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        
        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        # Final linear projection
        output = self.w_o(context)
        
        # Return average attention weights across heads for interpretability
        avg_attention = attention_weights.mean(dim=1)
        
        return output, avg_attention


class PositionalEncoding(nn.Module):
    """Positional encoding for feature ordering (optional for tabular data)."""
    
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input embeddings."""
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len]
        return self.dropout(x)


class FeedForward(nn.Module):
    """Position-wise feed-forward network."""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of feed-forward network."""
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class TransformerBlock(nn.Module):
    """Single transformer block with self-attention and feed-forward."""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self, 
        x: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of transformer block.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Tuple of (output, attention_weights)
        """
        # Self-attention with residual connection and layer norm
        attn_output, attention_weights = self.attention(x, x, x, attention_mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection and layer norm
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x, attention_weights


class RowEncoder(nn.Module):
    """
    FT-Transformer-based row encoder for processing tabular features.
    
    This encoder processes row-level patterns using transformer attention
    mechanisms adapted for tabular data.
    """
    
    def __init__(self, config: TabGPTConfig):
        super().__init__()
        self.config = config
        
        # Feature tokenization projection
        self.feature_projection = nn.Linear(config.embedding_dim, config.d_model)
        
        # Positional encoding (optional for tabular data)
        self.use_positional_encoding = getattr(config, 'use_positional_encoding', False)
        if self.use_positional_encoding:
            self.pos_encoding = PositionalEncoding(
                config.d_model, 
                config.max_features, 
                config.dropout
            )
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                d_model=config.d_model,
                n_heads=config.n_heads,
                d_ff=config.d_model * 4,  # Standard transformer ratio
                dropout=config.dropout
            )
            for _ in range(config.n_layers)
        ])
        
        # Output layer norm
        self.layer_norm = nn.LayerNorm(config.d_model)
        
        # Pooling strategies
        self.pooling_strategy = getattr(config, 'pooling_strategy', 'cls')
        if self.pooling_strategy == 'cls':
            # Add CLS token for classification-style pooling
            self.cls_token = nn.Parameter(torch.randn(1, 1, config.d_model))
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(
        self,
        feature_embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the row encoder.
        
        Args:
            feature_embeddings: Tokenized features [batch_size, n_features, embedding_dim]
            attention_mask: Attention mask [batch_size, n_features]
            
        Returns:
            Dictionary containing:
                - last_hidden_state: Final hidden states [batch_size, seq_len, d_model]
                - pooler_output: Pooled representation [batch_size, d_model]
                - attention_weights: List of attention weights from each layer
        """
        batch_size, n_features, _ = feature_embeddings.shape
        
        # Project features to model dimension
        x = self.feature_projection(feature_embeddings)
        
        # Add CLS token if using CLS pooling
        if self.pooling_strategy == 'cls':
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)
            
            # Extend attention mask for CLS token
            if attention_mask is not None:
                cls_mask = torch.ones(batch_size, 1, dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat([cls_mask, attention_mask], dim=1)
        
        # Add positional encoding if enabled
        if self.use_positional_encoding:
            x = self.pos_encoding(x)
        
        # Pass through transformer blocks
        attention_weights = []
        for transformer_block in self.transformer_blocks:
            x, attn_weights = transformer_block(x, attention_mask)
            attention_weights.append(attn_weights)
        
        # Final layer normalization
        x = self.layer_norm(x)
        
        # Pooling for sequence representation
        if self.pooling_strategy == 'cls':
            # Use CLS token representation
            pooler_output = x[:, 0]  # First token is CLS
            last_hidden_state = x[:, 1:]  # Remove CLS from sequence output
        elif self.pooling_strategy == 'mean':
            # Mean pooling over valid positions
            if attention_mask is not None:
                mask_expanded = attention_mask.unsqueeze(-1).expand_as(x)
                sum_embeddings = torch.sum(x * mask_expanded, dim=1)
                sum_mask = torch.sum(attention_mask, dim=1, keepdim=True)
                pooler_output = sum_embeddings / sum_mask.clamp(min=1)
            else:
                pooler_output = torch.mean(x, dim=1)
            last_hidden_state = x
        elif self.pooling_strategy == 'max':
            # Max pooling over valid positions
            if attention_mask is not None:
                mask_expanded = attention_mask.unsqueeze(-1).expand_as(x)
                x_masked = x.masked_fill(~mask_expanded, float('-inf'))
                pooler_output = torch.max(x_masked, dim=1)[0]
            else:
                pooler_output = torch.max(x, dim=1)[0]
            last_hidden_state = x
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")
        
        return {
            'last_hidden_state': last_hidden_state,
            'pooler_output': pooler_output,
            'attention_weights': attention_weights
        }
    
    def get_attention_patterns(
        self,
        feature_embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        layer_idx: int = -1
    ) -> torch.Tensor:
        """
        Extract attention patterns for interpretability.
        
        Args:
            feature_embeddings: Input feature embeddings
            attention_mask: Attention mask
            layer_idx: Layer index to extract attention from (-1 for last layer)
            
        Returns:
            Attention weights [batch_size, n_heads, seq_len, seq_len]
        """
        outputs = self.forward(feature_embeddings, attention_mask)
        attention_weights = outputs['attention_weights']
        
        if layer_idx == -1:
            layer_idx = len(attention_weights) - 1
        
        return attention_weights[layer_idx]
    
    def compute_feature_importance(
        self,
        feature_embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute feature importance scores based on attention patterns.
        
        Args:
            feature_embeddings: Input feature embeddings
            attention_mask: Attention mask
            
        Returns:
            Feature importance scores [batch_size, n_features]
        """
        outputs = self.forward(feature_embeddings, attention_mask)
        attention_weights = outputs['attention_weights']
        
        if not attention_weights:
            # Fallback: uniform importance if no attention weights
            batch_size, n_features, _ = feature_embeddings.shape
            return torch.ones(batch_size, n_features, device=feature_embeddings.device) / n_features
        
        # Average attention weights across all layers
        # Each attention weight is [batch_size, seq_len, seq_len]
        all_attention = torch.stack(attention_weights, dim=0)  # [n_layers, batch_size, seq_len, seq_len]
        avg_attention = all_attention.mean(dim=0)  # [batch_size, seq_len, seq_len]
        
        # Compute importance as average attention received by each position
        if self.pooling_strategy == 'cls':
            # For CLS pooling, use attention from CLS token to features
            # CLS token is at position 0, features start at position 1
            if avg_attention.size(-1) > 1:
                importance = avg_attention[:, 0, 1:]  # [batch_size, n_features]
            else:
                # Fallback if no features after CLS
                batch_size = avg_attention.size(0)
                n_features = feature_embeddings.size(1)
                importance = torch.ones(batch_size, n_features, device=feature_embeddings.device) / n_features
        else:
            # For other pooling, use average attention received
            importance = avg_attention.mean(dim=1)  # [batch_size, seq_len]
        
        return importance