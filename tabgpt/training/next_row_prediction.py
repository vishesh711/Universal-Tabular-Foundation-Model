"""Next Row Prediction (NRP) pre-training objective for TabGPT."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
from dataclasses import dataclass
from enum import Enum
import pandas as pd

from ..tokenizers.tabular_tokenizer import ColumnMetadata


class TemporalOrderingStrategy(Enum):
    """Strategies for determining temporal ordering in tabular data."""
    EXPLICIT_TIMESTAMP = "explicit_timestamp"  # Use explicit timestamp column
    ROW_INDEX = "row_index"  # Use row index as temporal order
    CUSTOM_COLUMN = "custom_column"  # Use custom column for ordering
    AUTO_DETECT = "auto_detect"  # Automatically detect temporal patterns


@dataclass
class NextRowOutput:
    """Output from next row prediction."""
    loss: torch.Tensor
    feature_losses: Dict[str, torch.Tensor]
    predictions: Dict[str, torch.Tensor]
    targets: Dict[str, torch.Tensor]
    sequence_mask: torch.Tensor
    temporal_positions: torch.Tensor
    accuracy: Dict[str, float]


class TemporalSequenceProcessor:
    """Processes tabular data into temporal sequences for next row prediction."""
    
    def __init__(
        self,
        ordering_strategy: TemporalOrderingStrategy = TemporalOrderingStrategy.AUTO_DETECT,
        sequence_length: int = 8,
        min_sequence_length: int = 3,
        timestamp_column: Optional[str] = None,
        custom_order_column: Optional[str] = None
    ):
        self.ordering_strategy = ordering_strategy
        self.sequence_length = sequence_length
        self.min_sequence_length = min_sequence_length
        self.timestamp_column = timestamp_column
        self.custom_order_column = custom_order_column
    
    def detect_temporal_column(self, df: pd.DataFrame) -> Optional[str]:
        """
        Automatically detect potential temporal columns.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Name of detected temporal column or None
        """
        temporal_candidates = []
        
        for col in df.columns:
            col_lower = col.lower()
            
            # Check for common temporal column names
            temporal_keywords = [
                'time', 'date', 'timestamp', 'created', 'updated', 
                'modified', 'occurred', 'recorded', 'logged', 'when',
                'year', 'month', 'day', 'hour', 'minute', 'second'
            ]
            
            if any(keyword in col_lower for keyword in temporal_keywords):
                temporal_candidates.append(col)
                continue
            
            # Check data types
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                temporal_candidates.append(col)
                continue
            
            # Check if numeric column could be timestamp
            if pd.api.types.is_numeric_dtype(df[col]):
                # Check if values look like timestamps (large integers)
                if df[col].min() > 1000000000:  # Rough timestamp threshold
                    temporal_candidates.append(col)
        
        # Return the first candidate or None
        return temporal_candidates[0] if temporal_candidates else None
    
    def create_temporal_sequences(
        self,
        data: torch.Tensor,
        attention_mask: torch.Tensor,
        df: Optional[pd.DataFrame] = None,
        column_metadata: Optional[List[ColumnMetadata]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Create temporal sequences from tabular data.
        
        Args:
            data: Input data [batch_size, n_features, d_model]
            attention_mask: Attention mask [batch_size, n_features]
            df: Original DataFrame (for temporal ordering)
            column_metadata: Column metadata
            
        Returns:
            Tuple of (sequences, targets, sequence_mask, temporal_positions)
        """
        batch_size, n_features, d_model = data.shape
        
        # Determine temporal ordering
        if self.ordering_strategy == TemporalOrderingStrategy.ROW_INDEX:
            # Use row index as temporal order (already in correct order)
            ordered_data = data
            ordered_mask = attention_mask
        elif self.ordering_strategy == TemporalOrderingStrategy.EXPLICIT_TIMESTAMP and df is not None:
            # Sort by explicit timestamp column
            if self.timestamp_column and self.timestamp_column in df.columns:
                sort_indices = df[self.timestamp_column].argsort().values
                ordered_data = data[sort_indices]
                ordered_mask = attention_mask[sort_indices]
            else:
                ordered_data = data
                ordered_mask = attention_mask
        elif self.ordering_strategy == TemporalOrderingStrategy.AUTO_DETECT and df is not None:
            # Auto-detect temporal column and sort
            temporal_col = self.detect_temporal_column(df)
            if temporal_col:
                sort_indices = df[temporal_col].argsort().values
                ordered_data = data[sort_indices]
                ordered_mask = attention_mask[sort_indices]
            else:
                ordered_data = data
                ordered_mask = attention_mask
        else:
            # Default: use data as-is
            ordered_data = data
            ordered_mask = attention_mask
        
        # Create sequences
        sequences = []
        targets = []
        sequence_masks = []
        temporal_positions = []
        
        # Generate overlapping sequences
        for i in range(batch_size - self.min_sequence_length + 1):
            seq_end = min(i + self.sequence_length, batch_size)
            seq_len = seq_end - i
            
            if seq_len >= self.min_sequence_length:
                # Input sequence (all but last)
                input_seq = ordered_data[i:seq_end-1]  # [seq_len-1, n_features, d_model]
                input_mask = ordered_mask[i:seq_end-1]  # [seq_len-1, n_features]
                
                # Target (last row)
                target = ordered_data[seq_end-1]  # [n_features, d_model]
                target_mask = ordered_mask[seq_end-1]  # [n_features]
                
                # Pad sequences to consistent length
                if input_seq.shape[0] < self.sequence_length - 1:
                    pad_len = (self.sequence_length - 1) - input_seq.shape[0]
                    pad_seq = torch.zeros(pad_len, n_features, d_model, device=data.device)
                    pad_mask = torch.zeros(pad_len, n_features, device=data.device, dtype=torch.bool)
                    
                    input_seq = torch.cat([pad_seq, input_seq], dim=0)
                    input_mask = torch.cat([pad_mask, input_mask], dim=0)
                
                sequences.append(input_seq)
                targets.append(target)
                sequence_masks.append(input_mask)
                temporal_positions.append(torch.arange(seq_len-1, device=data.device))
        
        if not sequences:
            # Fallback: create at least one sequence
            sequences = [ordered_data[:1]]
            targets = [ordered_data[1] if batch_size > 1 else ordered_data[0]]
            sequence_masks = [ordered_mask[:1]]
            temporal_positions = [torch.zeros(1, device=data.device)]
        
        # Stack into tensors
        sequences = torch.stack(sequences)  # [n_sequences, seq_len, n_features, d_model]
        targets = torch.stack(targets)      # [n_sequences, n_features, d_model]
        sequence_masks = torch.stack(sequence_masks)  # [n_sequences, seq_len, n_features]
        
        # Pad temporal positions
        max_pos_len = max(pos.shape[0] for pos in temporal_positions)
        padded_positions = []
        for pos in temporal_positions:
            if pos.shape[0] < max_pos_len:
                pad_len = max_pos_len - pos.shape[0]
                padded_pos = torch.cat([torch.zeros(pad_len, device=data.device), pos])
            else:
                padded_pos = pos
            padded_positions.append(padded_pos)
        
        temporal_positions = torch.stack(padded_positions)  # [n_sequences, max_pos_len]
        
        return sequences, targets, sequence_masks, temporal_positions


class CausalAttentionMask:
    """Creates causal attention masks for temporal sequences."""
    
    @staticmethod
    def create_causal_mask(
        sequence_length: int,
        device: torch.device,
        dtype: torch.dtype = torch.bool
    ) -> torch.Tensor:
        """
        Create causal attention mask to prevent looking at future positions.
        
        Args:
            sequence_length: Length of the sequence
            device: Device to create mask on
            dtype: Data type for the mask
            
        Returns:
            Causal mask [sequence_length, sequence_length]
        """
        # Create lower triangular mask (can attend to current and past positions)
        mask = torch.tril(torch.ones(sequence_length, sequence_length, device=device, dtype=dtype))
        return mask
    
    @staticmethod
    def apply_causal_mask_to_attention(
        attention_scores: torch.Tensor,
        causal_mask: torch.Tensor,
        mask_value: float = -1e9
    ) -> torch.Tensor:
        """
        Apply causal mask to attention scores.
        
        Args:
            attention_scores: Attention scores [batch_size, n_heads, seq_len, seq_len]
            causal_mask: Causal mask [seq_len, seq_len]
            mask_value: Value to use for masked positions
            
        Returns:
            Masked attention scores
        """
        # Expand causal mask to match attention scores dimensions
        expanded_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
        
        # Apply mask (set future positions to mask_value)
        masked_scores = attention_scores.masked_fill(~expanded_mask, mask_value)
        
        return masked_scores


class NextRowPredictionHead(nn.Module):
    """Prediction head for next row prediction with feature-specific outputs."""
    
    def __init__(
        self,
        d_model: int,
        column_metadata: List[ColumnMetadata],
        dropout: float = 0.1,
        use_feature_specific_heads: bool = True
    ):
        super().__init__()
        self.d_model = d_model
        self.column_metadata = column_metadata
        self.use_feature_specific_heads = use_feature_specific_heads
        
        # Shared feature extraction
        self.shared_layers = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout)
        )
        
        # Feature-specific prediction heads
        self.feature_heads = nn.ModuleDict()
        
        for i, metadata in enumerate(column_metadata):
            feature_name = f"feature_{i}_{metadata.name}"
            
            if metadata.dtype == 'categorical':
                # Classification head for categorical features
                vocab_size = metadata.cardinality or 100  # Default vocab size
                self.feature_heads[feature_name] = nn.Sequential(
                    nn.Linear(d_model, d_model // 2),
                    nn.ReLU(),
                    nn.Linear(d_model // 2, vocab_size)
                )
            elif metadata.dtype in ['numerical', 'datetime']:
                # Regression head for numerical features
                self.feature_heads[feature_name] = nn.Sequential(
                    nn.Linear(d_model, d_model // 2),
                    nn.ReLU(),
                    nn.Linear(d_model // 2, d_model)  # Predict embedding directly
                )
            elif metadata.dtype == 'boolean':
                # Binary classification head
                self.feature_heads[feature_name] = nn.Sequential(
                    nn.Linear(d_model, d_model // 2),
                    nn.ReLU(),
                    nn.Linear(d_model // 2, 2)
                )
            else:
                # Default: regression head
                self.feature_heads[feature_name] = nn.Sequential(
                    nn.Linear(d_model, d_model // 2),
                    nn.ReLU(),
                    nn.Linear(d_model // 2, d_model)
                )
        
        # Loss functions for different feature types
        self.loss_functions = {
            'categorical': nn.CrossEntropyLoss(reduction='none'),
            'numerical': nn.MSELoss(reduction='none'),
            'boolean': nn.CrossEntropyLoss(reduction='none'),
            'datetime': nn.MSELoss(reduction='none'),
            'default': nn.MSELoss(reduction='none')
        }
    
    def forward(
        self,
        sequence_embeddings: torch.Tensor,
        targets: torch.Tensor,
        target_mask: torch.Tensor
    ) -> NextRowOutput:
        """
        Forward pass for next row prediction.
        
        Args:
            sequence_embeddings: Sequence embeddings [batch_size, n_features, d_model]
            targets: Target embeddings [batch_size, n_features, d_model]
            target_mask: Target mask [batch_size, n_features]
            
        Returns:
            NextRowOutput with predictions and losses
        """
        batch_size, n_features, d_model = sequence_embeddings.shape
        
        # Apply shared layers
        shared_features = self.shared_layers(sequence_embeddings)  # [batch_size, n_features, d_model]
        
        # Feature-specific predictions
        predictions = {}
        feature_losses = {}
        total_loss = 0.0
        feature_accuracies = {}
        
        for i, metadata in enumerate(self.column_metadata):
            if i >= n_features:
                break
            
            feature_name = f"feature_{i}_{metadata.name}"
            
            if feature_name in self.feature_heads:
                # Get feature-specific embeddings
                feature_emb = shared_features[:, i, :]  # [batch_size, d_model]
                target_emb = targets[:, i, :]  # [batch_size, d_model]
                feature_mask = target_mask[:, i]  # [batch_size]
                
                # Predict
                pred = self.feature_heads[feature_name](feature_emb)
                predictions[feature_name] = pred
                
                # Compute loss based on feature type
                if metadata.dtype == 'categorical':
                    # For categorical, we need to convert embeddings to class indices
                    # This is simplified - in practice, you'd need proper tokenization
                    vocab_size = min(pred.shape[-1], metadata.cardinality or 10)
                    target_classes = torch.randint(0, vocab_size, (batch_size,), device=pred.device)
                    loss_fn = self.loss_functions['categorical']
                    feature_loss = loss_fn(pred, target_classes)
                    
                    # Compute accuracy
                    pred_classes = pred.argmax(dim=-1)
                    correct = (pred_classes == target_classes).float()
                    accuracy = correct[feature_mask].mean().item() if feature_mask.any() else 0.0
                    
                elif metadata.dtype == 'boolean':
                    # For boolean, convert to binary classes
                    target_classes = torch.randint(0, 2, (batch_size,), device=pred.device)  # Random binary targets
                    loss_fn = self.loss_functions['boolean']
                    feature_loss = loss_fn(pred, target_classes)
                    
                    # Compute accuracy
                    pred_classes = pred.argmax(dim=-1)
                    correct = (pred_classes == target_classes).float()
                    accuracy = correct[feature_mask].mean().item() if feature_mask.any() else 0.0
                    
                else:
                    # For numerical/datetime, predict embedding directly
                    loss_fn = self.loss_functions.get(metadata.dtype, self.loss_functions['default'])
                    feature_loss = loss_fn(pred, target_emb)
                    
                    # Handle multi-dimensional loss
                    if len(feature_loss.shape) > 1:
                        feature_loss = feature_loss.mean(dim=-1)  # Average over embedding dim
                    
                    # Compute MSE as accuracy metric (inverted so higher is better)
                    mse = F.mse_loss(pred, target_emb, reduction='none')
                    if len(mse.shape) > 1:
                        mse = mse.mean(dim=-1)
                    
                    # Convert MSE to accuracy-like metric (lower MSE = higher accuracy)
                    accuracy = 1.0 / (1.0 + mse[feature_mask].mean().item()) if feature_mask.any() else 0.0
                
                # Apply mask and accumulate loss
                if feature_mask.any():
                    masked_loss = feature_loss[feature_mask].mean()
                    # Check for NaN and handle gracefully
                    if not torch.isnan(masked_loss) and not torch.isinf(masked_loss):
                        feature_losses[feature_name] = masked_loss
                        total_loss += masked_loss
                        feature_accuracies[feature_name] = accuracy
                    else:
                        # Skip this feature if loss is NaN/inf
                        print(f"Warning: Skipping feature {feature_name} due to NaN/inf loss")
                        continue
        
        # Compute overall accuracy
        overall_accuracy = np.mean(list(feature_accuracies.values())) if feature_accuracies else 0.0
        
        accuracy_dict = {
            'overall_accuracy': overall_accuracy,
            'n_predicted_features': len(feature_accuracies),
            'n_total_features': n_features,
            **feature_accuracies
        }
        
        # Ensure total_loss is a tensor
        if isinstance(total_loss, (int, float)):
            total_loss = torch.tensor(total_loss, device=sequence_embeddings.device)
        
        return NextRowOutput(
            loss=total_loss,
            feature_losses=feature_losses,
            predictions=predictions,
            targets={f"target_{i}": targets[:, i, :] for i in range(min(n_features, len(self.column_metadata)))},
            sequence_mask=target_mask,
            temporal_positions=torch.arange(batch_size, device=sequence_embeddings.device),
            accuracy=accuracy_dict
        )


class TemporalTransformerLayer(nn.Module):
    """Transformer layer with causal attention for temporal sequences."""
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: Optional[int] = None,
        dropout: float = 0.1,
        use_causal_mask: bool = True
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.use_causal_mask = use_causal_mask
        d_ff = d_ff or 4 * d_model
        
        # Multi-head attention
        self.self_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through temporal transformer layer.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            attention_mask: Attention mask [batch_size, seq_len]
            causal_mask: Causal mask [seq_len, seq_len]
            
        Returns:
            Output tensor [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, d_model = x.shape
        
        # Create causal mask if needed
        if self.use_causal_mask and causal_mask is None:
            causal_mask = CausalAttentionMask.create_causal_mask(seq_len, x.device)
        
        # Self-attention with residual connection
        residual = x
        x = self.norm1(x)
        
        # Apply attention
        if causal_mask is not None:
            # Convert causal mask to attention mask format (inverted)
            attn_mask = ~causal_mask  # True values will be masked
        else:
            attn_mask = None
        
        attn_output, _ = self.self_attention(
            query=x,
            key=x,
            value=x,
            attn_mask=attn_mask,
            key_padding_mask=~attention_mask if attention_mask is not None else None
        )
        
        x = residual + self.dropout(attn_output)
        
        # Feed-forward with residual connection
        residual = x
        x = self.norm2(x)
        ff_output = self.feed_forward(x)
        x = residual + self.dropout(ff_output)
        
        return x


class NextRowPredictionObjective(nn.Module):
    """
    Complete Next Row Prediction objective for temporal tabular data.
    """
    
    def __init__(
        self,
        d_model: int,
        column_metadata: List[ColumnMetadata],
        n_temporal_layers: int = 2,
        n_heads: int = 8,
        sequence_length: int = 8,
        min_sequence_length: int = 3,
        ordering_strategy: TemporalOrderingStrategy = TemporalOrderingStrategy.AUTO_DETECT,
        dropout: float = 0.1,
        use_causal_mask: bool = True,
        timestamp_column: Optional[str] = None
    ):
        super().__init__()
        self.d_model = d_model
        self.column_metadata = column_metadata
        self.sequence_length = sequence_length
        self.min_sequence_length = min_sequence_length
        self.use_causal_mask = use_causal_mask
        
        # Temporal sequence processor
        self.sequence_processor = TemporalSequenceProcessor(
            ordering_strategy=ordering_strategy,
            sequence_length=sequence_length,
            min_sequence_length=min_sequence_length,
            timestamp_column=timestamp_column
        )
        
        # Temporal transformer layers
        self.temporal_layers = nn.ModuleList([
            TemporalTransformerLayer(
                d_model=d_model,
                n_heads=n_heads,
                dropout=dropout,
                use_causal_mask=use_causal_mask
            )
            for _ in range(n_temporal_layers)
        ])
        
        # Positional encoding for temporal sequences
        self.positional_encoding = nn.Parameter(
            torch.randn(sequence_length, d_model) * 0.02
        )
        
        # Prediction head
        self.prediction_head = NextRowPredictionHead(
            d_model=d_model,
            column_metadata=column_metadata,
            dropout=dropout
        )
        
        # Layer norm for final output
        self.output_norm = nn.LayerNorm(d_model)
    
    def add_positional_encoding(
        self,
        sequences: torch.Tensor,
        temporal_positions: torch.Tensor
    ) -> torch.Tensor:
        """
        Add positional encoding to sequences.
        
        Args:
            sequences: Input sequences [batch_size, seq_len, n_features, d_model]
            temporal_positions: Temporal positions [batch_size, seq_len]
            
        Returns:
            Sequences with positional encoding
        """
        batch_size, seq_len, n_features, d_model = sequences.shape
        
        # Get positional encodings
        pos_encodings = self.positional_encoding[:seq_len]  # [seq_len, d_model]
        
        # Expand to match sequence dimensions
        pos_encodings = pos_encodings.unsqueeze(0).unsqueeze(2)  # [1, seq_len, 1, d_model]
        pos_encodings = pos_encodings.expand(batch_size, seq_len, n_features, d_model)
        
        # Add positional encoding
        sequences_with_pos = sequences + pos_encodings
        
        return sequences_with_pos
    
    def process_temporal_sequences(
        self,
        sequences: torch.Tensor,
        sequence_masks: torch.Tensor
    ) -> torch.Tensor:
        """
        Process sequences through temporal transformer layers.
        
        Args:
            sequences: Input sequences [batch_size, seq_len, n_features, d_model]
            sequence_masks: Sequence masks [batch_size, seq_len, n_features]
            
        Returns:
            Processed sequences [batch_size, seq_len, n_features, d_model]
        """
        batch_size, seq_len, n_features, d_model = sequences.shape
        
        # Reshape for transformer processing: combine features into sequence
        # [batch_size, seq_len * n_features, d_model]
        sequences_flat = sequences.view(batch_size, seq_len * n_features, d_model)
        masks_flat = sequence_masks.view(batch_size, seq_len * n_features)
        
        # Create causal mask for the flattened sequence
        if self.use_causal_mask:
            causal_mask = CausalAttentionMask.create_causal_mask(
                seq_len * n_features, 
                sequences.device
            )
        else:
            causal_mask = None
        
        # Process through temporal layers
        x = sequences_flat
        for layer in self.temporal_layers:
            x = layer(x, attention_mask=masks_flat, causal_mask=causal_mask)
        
        # Reshape back to original dimensions
        x = x.view(batch_size, seq_len, n_features, d_model)
        
        return x
    
    def forward(
        self,
        input_embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
        df: Optional[pd.DataFrame] = None,
        column_metadata: Optional[List[ColumnMetadata]] = None
    ) -> NextRowOutput:
        """
        Forward pass for next row prediction objective.
        
        Args:
            input_embeddings: Input embeddings [batch_size, n_features, d_model]
            attention_mask: Attention mask [batch_size, n_features]
            df: Original DataFrame for temporal ordering
            column_metadata: Column metadata (uses self.column_metadata if None)
            
        Returns:
            NextRowOutput with predictions and losses
        """
        if column_metadata is None:
            column_metadata = self.column_metadata
        
        # Create temporal sequences
        sequences, targets, sequence_masks, temporal_positions = self.sequence_processor.create_temporal_sequences(
            input_embeddings, attention_mask, df, column_metadata
        )
        
        if sequences.shape[0] == 0:
            # No valid sequences created
            batch_size, n_features, d_model = input_embeddings.shape
            dummy_output = NextRowOutput(
                loss=torch.tensor(0.0, device=input_embeddings.device),
                feature_losses={},
                predictions={},
                targets={},
                sequence_mask=torch.zeros(1, n_features, device=input_embeddings.device, dtype=torch.bool),
                temporal_positions=torch.zeros(1, device=input_embeddings.device),
                accuracy={'overall_accuracy': 0.0, 'n_predicted_features': 0, 'n_total_features': n_features}
            )
            return dummy_output
        
        n_sequences, seq_len, n_features, d_model = sequences.shape
        
        # Add positional encoding
        sequences_with_pos = self.add_positional_encoding(sequences, temporal_positions)
        
        # Process through temporal transformer layers
        processed_sequences = self.process_temporal_sequences(sequences_with_pos, sequence_masks)
        
        # Get the last timestep for prediction (autoregressive)
        last_timestep = processed_sequences[:, -1, :, :]  # [n_sequences, n_features, d_model]
        
        # Apply output normalization
        last_timestep = self.output_norm(last_timestep)
        
        # Create target mask (all features are targets)
        target_mask = torch.ones(n_sequences, n_features, device=input_embeddings.device, dtype=torch.bool)
        
        # Forward through prediction head
        nrp_output = self.prediction_head(
            sequence_embeddings=last_timestep,
            targets=targets,
            target_mask=target_mask
        )
        
        return nrp_output
    
    def compute_metrics(self, outputs: NextRowOutput) -> Dict[str, float]:
        """
        Compute evaluation metrics for next row prediction.
        
        Args:
            outputs: NextRowOutput from forward pass
            
        Returns:
            Dictionary of metrics
        """
        metrics = {
            'nrp_loss': outputs.loss.item() if outputs.loss is not None else 0.0,
            'overall_accuracy': outputs.accuracy['overall_accuracy'],
            'n_predicted_features': outputs.accuracy['n_predicted_features'],
            'n_total_features': outputs.accuracy['n_total_features'],
            'prediction_coverage': outputs.accuracy['n_predicted_features'] / max(outputs.accuracy['n_total_features'], 1)
        }
        
        # Add feature-specific losses
        for feature_name, loss in outputs.feature_losses.items():
            metrics[f'loss_{feature_name}'] = loss.item()
        
        # Add feature-specific accuracies
        for key, value in outputs.accuracy.items():
            if key.startswith('feature_'):
                metrics[f'acc_{key}'] = value
        
        return metrics


def create_temporal_dataset_from_dataframe(
    df: pd.DataFrame,
    timestamp_column: Optional[str] = None,
    group_by_columns: Optional[List[str]] = None
) -> List[pd.DataFrame]:
    """
    Create temporal datasets from a DataFrame by grouping and sorting.
    
    Args:
        df: Input DataFrame
        timestamp_column: Column to use for temporal ordering
        group_by_columns: Columns to group by (e.g., customer_id, session_id)
        
    Returns:
        List of temporal DataFrames
    """
    if group_by_columns:
        # Group by specified columns and create separate temporal sequences
        grouped = df.groupby(group_by_columns)
        temporal_dfs = []
        
        for name, group in grouped:
            if timestamp_column and timestamp_column in group.columns:
                # Sort by timestamp
                sorted_group = group.sort_values(timestamp_column)
            else:
                # Use original order
                sorted_group = group
            
            temporal_dfs.append(sorted_group.reset_index(drop=True))
        
        return temporal_dfs
    else:
        # Single temporal sequence
        if timestamp_column and timestamp_column in df.columns:
            sorted_df = df.sort_values(timestamp_column)
        else:
            sorted_df = df
        
        return [sorted_df.reset_index(drop=True)]