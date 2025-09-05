"""Cross-attention fusion mechanism for TabGPT."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
import math


class CrossAttentionLayer(nn.Module):
    """
    Cross-attention layer for fusing row and column representations.
    
    This layer implements bidirectional attention between row embeddings
    (feature-level representations) and column embeddings (metadata representations).
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        dropout: float = 0.1,
        temperature: float = 1.0
    ):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.temperature = temperature
        
        # Query, Key, Value projections for row-to-column attention
        self.row_to_col_q = nn.Linear(d_model, d_model, bias=False)
        self.row_to_col_k = nn.Linear(d_model, d_model, bias=False)
        self.row_to_col_v = nn.Linear(d_model, d_model, bias=False)
        
        # Query, Key, Value projections for column-to-row attention
        self.col_to_row_q = nn.Linear(d_model, d_model, bias=False)
        self.col_to_row_k = nn.Linear(d_model, d_model, bias=False)
        self.col_to_row_v = nn.Linear(d_model, d_model, bias=False)
        
        # Output projections
        self.row_output_proj = nn.Linear(d_model, d_model)
        self.col_output_proj = nn.Linear(d_model, d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Scale factor for attention scores
        self.scale = math.sqrt(self.d_k) * temperature
        
    def _compute_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute scaled dot-product attention.
        
        Args:
            query: Query tensor [batch_size, n_heads, seq_len_q, d_k]
            key: Key tensor [batch_size, n_heads, seq_len_k, d_k]
            value: Value tensor [batch_size, n_heads, seq_len_k, d_k]
            attention_mask: Mask tensor [batch_size, seq_len_q, seq_len_k]
            
        Returns:
            Tuple of (attention_output, attention_weights)
        """
        # Compute attention scores
        scores = torch.matmul(query, key.transpose(-2, -1)) / self.scale
        
        # Apply attention mask if provided
        if attention_mask is not None:
            # Expand mask for multi-head attention if needed
            if attention_mask.dim() == 3:  # [batch_size, seq_len_q, seq_len_k]
                mask = attention_mask.unsqueeze(1)  # [batch_size, 1, seq_len_q, seq_len_k]
            else:  # Already has the right dimensions
                mask = attention_mask
            scores = scores.masked_fill(~mask, float('-inf'))
        
        # Apply softmax and dropout
        attention_weights = F.softmax(scores, dim=-1)
        
        # Handle NaN values that can occur when all values are masked
        attention_weights = torch.where(
            torch.isnan(attention_weights),
            torch.zeros_like(attention_weights),
            attention_weights
        )
        
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attention_output = torch.matmul(attention_weights, value)
        
        return attention_output, attention_weights
    
    def forward(
        self,
        row_embeddings: torch.Tensor,
        column_embeddings: torch.Tensor,
        row_attention_mask: Optional[torch.Tensor] = None,
        column_attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of cross-attention layer.
        
        Args:
            row_embeddings: Row embeddings [batch_size, n_features, d_model]
            column_embeddings: Column embeddings [n_features, d_model]
            row_attention_mask: Mask for row embeddings [batch_size, n_features]
            column_attention_mask: Mask for column embeddings [n_features]
            
        Returns:
            Dictionary containing fused representations and attention weights
        """
        batch_size, n_features, _ = row_embeddings.shape
        
        # Expand column embeddings to match batch size
        column_embeddings = column_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        
        # === Row-to-Column Attention ===
        # Rows attend to columns to get column-aware row representations
        
        # Project to Q, K, V
        row_q = self.row_to_col_q(row_embeddings)  # [batch_size, n_features, d_model]
        col_k = self.row_to_col_k(column_embeddings)  # [batch_size, n_features, d_model]
        col_v = self.row_to_col_v(column_embeddings)  # [batch_size, n_features, d_model]
        
        # Reshape for multi-head attention
        row_q = row_q.view(batch_size, n_features, self.n_heads, self.d_k).transpose(1, 2)
        col_k = col_k.view(batch_size, n_features, self.n_heads, self.d_k).transpose(1, 2)
        col_v = col_v.view(batch_size, n_features, self.n_heads, self.d_k).transpose(1, 2)
        
        # Create attention mask for row-to-column attention
        row_to_col_mask = None
        if row_attention_mask is not None and column_attention_mask is not None:
            # [batch_size, n_features] x [n_features] -> [batch_size, n_features, n_features]
            row_to_col_mask = row_attention_mask.unsqueeze(-1) & column_attention_mask.unsqueeze(0)
        
        # Compute row-to-column attention
        row_attended, row_to_col_weights = self._compute_attention(
            row_q, col_k, col_v, row_to_col_mask
        )
        
        # Debug: Check shapes before reshape
        # print(f"DEBUG: row_attended shape before transpose: {row_attended.shape}")
        # print(f"DEBUG: row_attended numel: {row_attended.numel()}")
        # print(f"DEBUG: expected shape: [{batch_size}, {n_features}, {self.d_model}]")
        
        # Reshape back
        row_attended = row_attended.transpose(1, 2).contiguous().view(
            batch_size, n_features, self.d_model
        )
        
        # === Column-to-Row Attention ===
        # Columns attend to rows to get row-aware column representations
        
        # Project to Q, K, V
        col_q = self.col_to_row_q(column_embeddings)  # [batch_size, n_features, d_model]
        row_k = self.col_to_row_k(row_embeddings)  # [batch_size, n_features, d_model]
        row_v = self.col_to_row_v(row_embeddings)  # [batch_size, n_features, d_model]
        
        # Reshape for multi-head attention
        col_q = col_q.view(batch_size, n_features, self.n_heads, self.d_k).transpose(1, 2)
        row_k = row_k.view(batch_size, n_features, self.n_heads, self.d_k).transpose(1, 2)
        row_v = row_v.view(batch_size, n_features, self.n_heads, self.d_k).transpose(1, 2)
        
        # Create attention mask for column-to-row attention
        col_to_row_mask = None
        if row_attention_mask is not None and column_attention_mask is not None:
            # [n_features] x [batch_size, n_features] -> [batch_size, n_features, n_features]
            col_to_row_mask = column_attention_mask.unsqueeze(0).unsqueeze(0) & row_attention_mask.unsqueeze(-1)
        
        # Compute column-to-row attention
        col_attended, col_to_row_weights = self._compute_attention(
            col_q, row_k, row_v, col_to_row_mask
        )
        
        # Reshape back
        col_attended = col_attended.transpose(1, 2).contiguous().view(
            batch_size, n_features, self.d_model
        )
        
        # Apply output projections
        fused_row_embeddings = self.row_output_proj(row_attended)
        fused_col_embeddings = self.col_output_proj(col_attended)
        
        return {
            'fused_row_embeddings': fused_row_embeddings,
            'fused_column_embeddings': fused_col_embeddings,
            'row_to_column_attention': row_to_col_weights.mean(dim=1),  # Average across heads
            'column_to_row_attention': col_to_row_weights.mean(dim=1),  # Average across heads
        }


class CrossAttentionFusion(nn.Module):
    """
    Complete cross-attention fusion mechanism with residual connections.
    
    This module combines row and column representations through bidirectional
    cross-attention and applies residual connections to preserve original information.
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        n_layers: int = 2,
        dropout: float = 0.1,
        temperature: float = 1.0,
        use_residual: bool = True,
        fusion_strategy: str = 'concat'  # 'concat', 'add', 'gate'
    ):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.use_residual = use_residual
        self.fusion_strategy = fusion_strategy
        
        # Cross-attention layers
        self.cross_attention_layers = nn.ModuleList([
            CrossAttentionLayer(d_model, n_heads, dropout, temperature)
            for _ in range(n_layers)
        ])
        
        # Layer normalization
        self.row_layer_norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(n_layers)
        ])
        self.col_layer_norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(n_layers)
        ])
        
        # Fusion layers based on strategy
        if fusion_strategy == 'concat':
            self.fusion_proj = nn.Linear(2 * d_model, d_model)
        elif fusion_strategy == 'gate':
            self.gate_proj = nn.Linear(2 * d_model, d_model)
            self.gate_activation = nn.Sigmoid()
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        row_embeddings: torch.Tensor,
        column_embeddings: torch.Tensor,
        row_attention_mask: Optional[torch.Tensor] = None,
        column_attention_mask: Optional[torch.Tensor] = None,
        return_attention_weights: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of cross-attention fusion.
        
        Args:
            row_embeddings: Row embeddings [batch_size, n_features, d_model]
            column_embeddings: Column embeddings [n_features, d_model]
            row_attention_mask: Mask for row embeddings [batch_size, n_features]
            column_attention_mask: Mask for column embeddings [n_features]
            return_attention_weights: Whether to return attention weights
            
        Returns:
            Dictionary containing fused representations
        """
        batch_size, n_features, _ = row_embeddings.shape
        
        # Initialize current embeddings
        current_row_embeddings = row_embeddings
        current_col_embeddings = column_embeddings
        
        # Store attention weights if requested
        all_attention_weights = []
        
        # Apply cross-attention layers
        for i, cross_attn_layer in enumerate(self.cross_attention_layers):
            # Cross-attention
            cross_attn_outputs = cross_attn_layer(
                current_row_embeddings,
                current_col_embeddings,
                row_attention_mask,
                column_attention_mask
            )
            
            fused_row = cross_attn_outputs['fused_row_embeddings']
            fused_col = cross_attn_outputs['fused_column_embeddings']
            
            # Apply residual connections and layer normalization
            if self.use_residual:
                current_row_embeddings = self.row_layer_norms[i](
                    current_row_embeddings + self.dropout(fused_row)
                )
                # Column embeddings need to be expanded for residual connection
                expanded_col = current_col_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
                current_col_embeddings = self.col_layer_norms[i](
                    expanded_col + self.dropout(fused_col)
                ).mean(dim=0)  # Average across batch for column embeddings
            else:
                current_row_embeddings = self.row_layer_norms[i](fused_row)
                current_col_embeddings = self.col_layer_norms[i](fused_col.mean(dim=0))
            
            # Store attention weights
            if return_attention_weights:
                all_attention_weights.append({
                    'row_to_column': cross_attn_outputs['row_to_column_attention'],
                    'column_to_row': cross_attn_outputs['column_to_row_attention']
                })
        
        # Final fusion of row and column information
        fused_representations = self._fuse_representations(
            current_row_embeddings, 
            current_col_embeddings,
            batch_size,
            n_features
        )
        
        outputs = {
            'fused_representations': fused_representations,
            'enhanced_row_embeddings': current_row_embeddings,
            'enhanced_column_embeddings': current_col_embeddings
        }
        
        if return_attention_weights:
            outputs['attention_weights'] = all_attention_weights
        
        return outputs
    
    def _fuse_representations(
        self,
        row_embeddings: torch.Tensor,
        column_embeddings: torch.Tensor,
        batch_size: int,
        n_features: int
    ) -> torch.Tensor:
        """
        Fuse row and column representations using the specified strategy.
        
        Args:
            row_embeddings: Enhanced row embeddings [batch_size, n_features, d_model]
            column_embeddings: Enhanced column embeddings [n_features, d_model]
            batch_size: Batch size
            n_features: Number of features
            
        Returns:
            Fused representations [batch_size, n_features, d_model]
        """
        # Expand column embeddings to match batch size
        expanded_col_embeddings = column_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        
        if self.fusion_strategy == 'add':
            # Simple addition
            return row_embeddings + expanded_col_embeddings
        
        elif self.fusion_strategy == 'concat':
            # Concatenate and project
            concatenated = torch.cat([row_embeddings, expanded_col_embeddings], dim=-1)
            return self.fusion_proj(concatenated)
        
        elif self.fusion_strategy == 'gate':
            # Gated fusion
            concatenated = torch.cat([row_embeddings, expanded_col_embeddings], dim=-1)
            gate = self.gate_activation(self.gate_proj(concatenated))
            return gate * row_embeddings + (1 - gate) * expanded_col_embeddings
        
        else:
            raise ValueError(f"Unknown fusion strategy: {self.fusion_strategy}")
    
    def get_attention_patterns(
        self,
        row_embeddings: torch.Tensor,
        column_embeddings: torch.Tensor,
        row_attention_mask: Optional[torch.Tensor] = None,
        column_attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Get attention patterns for interpretability.
        
        Args:
            row_embeddings: Row embeddings
            column_embeddings: Column embeddings
            row_attention_mask: Row attention mask
            column_attention_mask: Column attention mask
            
        Returns:
            Dictionary of attention patterns
        """
        outputs = self.forward(
            row_embeddings,
            column_embeddings,
            row_attention_mask,
            column_attention_mask,
            return_attention_weights=True
        )
        
        attention_weights = outputs['attention_weights']
        
        # Aggregate attention weights across layers
        row_to_col_attention = torch.stack([
            layer_attn['row_to_column'] for layer_attn in attention_weights
        ]).mean(dim=0)
        
        col_to_row_attention = torch.stack([
            layer_attn['column_to_row'] for layer_attn in attention_weights
        ]).mean(dim=0)
        
        return {
            'row_to_column_attention': row_to_col_attention,
            'column_to_row_attention': col_to_row_attention,
            'layer_attention_weights': attention_weights
        }
    
    def compute_interaction_scores(
        self,
        row_embeddings: torch.Tensor,
        column_embeddings: torch.Tensor,
        row_attention_mask: Optional[torch.Tensor] = None,
        column_attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute feature interaction scores based on cross-attention patterns.
        
        Args:
            row_embeddings: Row embeddings
            column_embeddings: Column embeddings
            row_attention_mask: Row attention mask
            column_attention_mask: Column attention mask
            
        Returns:
            Dictionary of interaction scores
        """
        attention_patterns = self.get_attention_patterns(
            row_embeddings, column_embeddings, row_attention_mask, column_attention_mask
        )
        
        row_to_col = attention_patterns['row_to_column_attention']
        col_to_row = attention_patterns['column_to_row_attention']
        
        # Compute bidirectional interaction strength
        # Average of both directions for symmetric interaction score
        interaction_matrix = (row_to_col + col_to_row.transpose(-2, -1)) / 2
        
        # Compute feature importance based on attention received
        feature_importance = interaction_matrix.sum(dim=-1)  # Sum of attention received
        
        # Normalize importance scores
        if row_attention_mask is not None:
            feature_importance = feature_importance * row_attention_mask.float()
            feature_importance = feature_importance / (feature_importance.sum(dim=-1, keepdim=True) + 1e-8)
        else:
            feature_importance = F.softmax(feature_importance, dim=-1)
        
        return {
            'interaction_matrix': interaction_matrix,
            'feature_importance': feature_importance,
            'row_to_column_strength': row_to_col.sum(dim=-1),
            'column_to_row_strength': col_to_row.sum(dim=-1)
        }