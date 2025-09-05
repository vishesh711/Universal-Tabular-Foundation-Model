"""Tests for Next Row Prediction pre-training objective."""

import pytest
import torch
import numpy as np
import pandas as pd

from tabgpt.training.next_row_prediction import (
    NextRowPredictionObjective,
    NextRowPredictionHead,
    NextRowOutput,
    TemporalSequenceProcessor,
    TemporalTransformerLayer,
    CausalAttentionMask,
    TemporalOrderingStrategy,
    create_temporal_dataset_from_dataframe
)
from tabgpt.tokenizers import TabularTokenizer
from tabgpt.models import TabGPTModel
from tabgpt.config import TabGPTConfig


class TestCausalAttentionMask:
    """Test causal attention mask utilities."""
    
    def test_create_causal_mask(self):
        """Test causal mask creation."""
        seq_len = 5
        device = torch.device('cpu')
        
        mask = CausalAttentionMask.create_causal_mask(seq_len, device)
        
        assert mask.shape == (seq_len, seq_len)
        assert mask.dtype == torch.bool
        
        # Check that it's lower triangular
        for i in range(seq_len):
            for j in range(seq_len):
                if j <= i:
                    assert mask[i, j] == True  # Can attend to current and past
                else:
                    assert mask[i, j] == False  # Cannot attend to future
    
    def test_apply_causal_mask_to_attention(self):
        """Test applying causal mask to attention scores."""
        batch_size, n_heads, seq_len = 2, 4, 3
        
        attention_scores = torch.randn(batch_size, n_heads, seq_len, seq_len)
        causal_mask = CausalAttentionMask.create_causal_mask(seq_len, attention_scores.device)
        
        masked_scores = CausalAttentionMask.apply_causal_mask_to_attention(
            attention_scores, causal_mask, mask_value=-1e9
        )
        
        assert masked_scores.shape == attention_scores.shape
        
        # Check that future positions are masked
        for i in range(seq_len):
            for j in range(i + 1, seq_len):
                assert torch.allclose(masked_scores[:, :, i, j], torch.tensor(-1e9))


class TestTemporalSequenceProcessor:
    """Test temporal sequence processing."""
    
    def test_temporal_processor_creation(self):
        """Test temporal sequence processor creation."""
        processor = TemporalSequenceProcessor(
            ordering_strategy=TemporalOrderingStrategy.ROW_INDEX,
            sequence_length=5,
            min_sequence_length=2
        )
        
        assert processor.ordering_strategy == TemporalOrderingStrategy.ROW_INDEX
        assert processor.sequence_length == 5
        assert processor.min_sequence_length == 2
    
    def test_detect_temporal_column(self):
        """Test automatic temporal column detection."""
        processor = TemporalSequenceProcessor()
        
        # DataFrame with obvious temporal column
        df = pd.DataFrame({
            'id': [1, 2, 3],
            'timestamp': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03']),
            'value': [10, 20, 30]
        })
        
        temporal_col = processor.detect_temporal_column(df)
        assert temporal_col == 'timestamp'
        
        # DataFrame without temporal column
        df_no_time = pd.DataFrame({
            'id': [1, 2, 3],
            'value': [10, 20, 30]
        })
        
        temporal_col_none = processor.detect_temporal_column(df_no_time)
        assert temporal_col_none is None
    
    def test_create_temporal_sequences_basic(self):
        """Test basic temporal sequence creation."""
        batch_size, n_features, d_model = 6, 4, 32
        
        processor = TemporalSequenceProcessor(
            ordering_strategy=TemporalOrderingStrategy.ROW_INDEX,
            sequence_length=4,
            min_sequence_length=2
        )
        
        data = torch.randn(batch_size, n_features, d_model)
        attention_mask = torch.ones(batch_size, n_features, dtype=torch.bool)
        
        sequences, targets, sequence_masks, temporal_positions = processor.create_temporal_sequences(
            data, attention_mask
        )
        
        assert sequences.shape[0] > 0  # Should create at least one sequence
        assert sequences.shape[2] == n_features
        assert sequences.shape[3] == d_model
        assert targets.shape[1] == n_features
        assert targets.shape[2] == d_model
        assert sequence_masks.shape[0] == sequences.shape[0]
        assert temporal_positions.shape[0] == sequences.shape[0]
    
    def test_create_temporal_sequences_with_dataframe(self):
        """Test temporal sequence creation with DataFrame ordering."""
        batch_size, n_features, d_model = 5, 3, 16
        
        # Create DataFrame with timestamp
        df = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=batch_size, freq='D'),
            'feature1': range(batch_size),
            'feature2': range(batch_size, 2 * batch_size)
        })
        
        processor = TemporalSequenceProcessor(
            ordering_strategy=TemporalOrderingStrategy.EXPLICIT_TIMESTAMP,
            timestamp_column='timestamp',
            sequence_length=3,
            min_sequence_length=2
        )
        
        data = torch.randn(batch_size, n_features, d_model)
        attention_mask = torch.ones(batch_size, n_features, dtype=torch.bool)
        
        sequences, targets, sequence_masks, temporal_positions = processor.create_temporal_sequences(
            data, attention_mask, df
        )
        
        assert sequences.shape[0] > 0
        assert sequences.shape[2] == n_features
        assert sequences.shape[3] == d_model


class TestTemporalTransformerLayer:
    """Test temporal transformer layer."""
    
    def test_temporal_transformer_creation(self):
        """Test temporal transformer layer creation."""
        d_model, n_heads = 64, 8
        
        layer = TemporalTransformerLayer(
            d_model=d_model,
            n_heads=n_heads,
            use_causal_mask=True
        )
        
        assert layer.d_model == d_model
        assert layer.n_heads == n_heads
        assert layer.use_causal_mask == True
    
    def test_temporal_transformer_forward(self):
        """Test temporal transformer forward pass."""
        batch_size, seq_len, d_model = 3, 5, 32
        n_heads = 4
        
        layer = TemporalTransformerLayer(
            d_model=d_model,
            n_heads=n_heads,
            use_causal_mask=True
        )
        
        x = torch.randn(batch_size, seq_len, d_model)
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        
        output = layer(x, attention_mask=attention_mask)
        
        assert output.shape == x.shape
        assert not torch.isnan(output).any()
    
    def test_temporal_transformer_causal_mask(self):
        """Test temporal transformer with causal mask."""
        batch_size, seq_len, d_model = 2, 4, 16
        n_heads = 2
        
        layer = TemporalTransformerLayer(
            d_model=d_model,
            n_heads=n_heads,
            use_causal_mask=True
        )
        
        x = torch.randn(batch_size, seq_len, d_model)
        causal_mask = CausalAttentionMask.create_causal_mask(seq_len, x.device)
        
        output = layer(x, causal_mask=causal_mask)
        
        assert output.shape == x.shape
        assert not torch.isnan(output).any()


class TestNextRowPredictionHead:
    """Test next row prediction head."""
    
    def test_nrp_head_creation(self):
        """Test NRP head creation."""
        d_model = 64
        
        from tabgpt.tokenizers.tabular_tokenizer import ColumnMetadata
        column_metadata = [
            ColumnMetadata(name='num1', dtype='numerical', unique_values=None),
            ColumnMetadata(name='cat1', dtype='categorical', unique_values=10),
            ColumnMetadata(name='bool1', dtype='boolean', unique_values=2)
        ]
        
        head = NextRowPredictionHead(
            d_model=d_model,
            column_metadata=column_metadata
        )
        
        assert head.d_model == d_model
        assert len(head.feature_heads) == len(column_metadata)
    
    def test_nrp_head_forward(self):
        """Test NRP head forward pass."""
        batch_size, n_features, d_model = 3, 4, 32
        
        from tabgpt.tokenizers.tabular_tokenizer import ColumnMetadata
        column_metadata = [
            ColumnMetadata(name='num1', dtype='numerical', unique_values=None),
            ColumnMetadata(name='cat1', dtype='categorical', unique_values=5),
            ColumnMetadata(name='bool1', dtype='boolean', unique_values=2),
            ColumnMetadata(name='num2', dtype='numerical', unique_values=None)
        ]
        
        head = NextRowPredictionHead(
            d_model=d_model,
            column_metadata=column_metadata
        )
        
        sequence_embeddings = torch.randn(batch_size, n_features, d_model)
        targets = torch.randn(batch_size, n_features, d_model)
        target_mask = torch.ones(batch_size, n_features, dtype=torch.bool)
        
        output = head(sequence_embeddings, targets, target_mask)
        
        assert isinstance(output, NextRowOutput)
        assert output.loss is not None
        assert not torch.isnan(output.loss)
        assert len(output.predictions) > 0
        assert len(output.feature_losses) > 0
        assert 'overall_accuracy' in output.accuracy
    
    def test_nrp_head_different_feature_types(self):
        """Test NRP head with different feature types."""
        batch_size, n_features, d_model = 2, 3, 16
        
        from tabgpt.tokenizers.tabular_tokenizer import ColumnMetadata
        column_metadata = [
            ColumnMetadata(name='categorical_feature', dtype='categorical', unique_values=8),
            ColumnMetadata(name='numerical_feature', dtype='numerical', unique_values=None),
            ColumnMetadata(name='boolean_feature', dtype='boolean', unique_values=2)
        ]
        
        head = NextRowPredictionHead(
            d_model=d_model,
            column_metadata=column_metadata
        )
        
        sequence_embeddings = torch.randn(batch_size, n_features, d_model)
        targets = torch.randn(batch_size, n_features, d_model)
        target_mask = torch.ones(batch_size, n_features, dtype=torch.bool)
        
        output = head(sequence_embeddings, targets, target_mask)
        
        # Check that we have predictions for each feature type
        feature_names = [f"feature_{i}_{meta.name}" for i, meta in enumerate(column_metadata)]
        for feature_name in feature_names:
            assert feature_name in output.predictions or feature_name in head.feature_heads


class TestNextRowPredictionObjective:
    """Test complete next row prediction objective."""
    
    def test_nrp_objective_creation(self):
        """Test NRP objective creation."""
        d_model = 64
        
        from tabgpt.tokenizers.tabular_tokenizer import ColumnMetadata
        column_metadata = [
            ColumnMetadata(name='feature1', dtype='numerical', unique_values=None),
            ColumnMetadata(name='feature2', dtype='categorical', unique_values=10)
        ]
        
        objective = NextRowPredictionObjective(
            d_model=d_model,
            column_metadata=column_metadata,
            n_temporal_layers=2,
            sequence_length=5
        )
        
        assert objective.d_model == d_model
        assert len(objective.temporal_layers) == 2
        assert objective.sequence_length == 5
    
    def test_add_positional_encoding(self):
        """Test positional encoding addition."""
        d_model = 32
        
        from tabgpt.tokenizers.tabular_tokenizer import ColumnMetadata
        column_metadata = [
            ColumnMetadata(name='feature1', dtype='numerical', unique_values=None)
        ]
        
        objective = NextRowPredictionObjective(
            d_model=d_model,
            column_metadata=column_metadata,
            sequence_length=4
        )
        
        batch_size, seq_len, n_features = 2, 3, 2
        sequences = torch.randn(batch_size, seq_len, n_features, d_model)
        temporal_positions = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
        
        sequences_with_pos = objective.add_positional_encoding(sequences, temporal_positions)
        
        assert sequences_with_pos.shape == sequences.shape
        assert not torch.allclose(sequences_with_pos, sequences)  # Should be different due to pos encoding
    
    def test_nrp_objective_forward_basic(self):
        """Test NRP objective basic forward pass."""
        batch_size, n_features, d_model = 6, 3, 32
        
        from tabgpt.tokenizers.tabular_tokenizer import ColumnMetadata
        column_metadata = [
            ColumnMetadata(name='feature1', dtype='numerical', unique_values=None),
            ColumnMetadata(name='feature2', dtype='categorical', unique_values=5),
            ColumnMetadata(name='feature3', dtype='boolean', unique_values=2)
        ]
        
        objective = NextRowPredictionObjective(
            d_model=d_model,
            column_metadata=column_metadata,
            n_temporal_layers=1,
            sequence_length=4,
            min_sequence_length=2
        )
        
        input_embeddings = torch.randn(batch_size, n_features, d_model)
        attention_mask = torch.ones(batch_size, n_features, dtype=torch.bool)
        
        output = objective(input_embeddings, attention_mask)
        
        assert isinstance(output, NextRowOutput)
        assert output.loss is not None
        assert not torch.isnan(output.loss)
        assert 'overall_accuracy' in output.accuracy
    
    def test_nrp_objective_with_dataframe(self):
        """Test NRP objective with DataFrame temporal ordering."""
        batch_size, n_features, d_model = 5, 2, 16
        
        # Create temporal DataFrame
        df = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=batch_size, freq='H'),
            'value': range(batch_size)
        })
        
        from tabgpt.tokenizers.tabular_tokenizer import ColumnMetadata
        column_metadata = [
            ColumnMetadata(name='timestamp', dtype='datetime', unique_values=None),
            ColumnMetadata(name='value', dtype='numerical', unique_values=None)
        ]
        
        objective = NextRowPredictionObjective(
            d_model=d_model,
            column_metadata=column_metadata,
            ordering_strategy=TemporalOrderingStrategy.AUTO_DETECT,
            sequence_length=3,
            min_sequence_length=2
        )
        
        input_embeddings = torch.randn(batch_size, n_features, d_model)
        attention_mask = torch.ones(batch_size, n_features, dtype=torch.bool)
        
        output = objective(input_embeddings, attention_mask, df=df)
        
        assert isinstance(output, NextRowOutput)
        assert output.loss is not None
    
    def test_compute_metrics(self):
        """Test metrics computation."""
        from tabgpt.tokenizers.tabular_tokenizer import ColumnMetadata
        column_metadata = [
            ColumnMetadata(name='feature1', dtype='numerical', unique_values=None)
        ]
        
        objective = NextRowPredictionObjective(
            d_model=32,
            column_metadata=column_metadata
        )
        
        # Create mock output
        output = NextRowOutput(
            loss=torch.tensor(2.5),
            feature_losses={'feature_0_feature1': torch.tensor(1.2)},
            predictions={'feature_0_feature1': torch.randn(2, 32)},
            targets={'target_0': torch.randn(2, 32)},
            sequence_mask=torch.ones(2, 1, dtype=torch.bool),
            temporal_positions=torch.arange(2),
            accuracy={
                'overall_accuracy': 0.75,
                'n_predicted_features': 1,
                'n_total_features': 1,
                'feature_0_feature1': 0.8
            }
        )
        
        metrics = objective.compute_metrics(output)
        
        assert 'nrp_loss' in metrics
        assert 'overall_accuracy' in metrics
        assert 'n_predicted_features' in metrics
        assert 'prediction_coverage' in metrics
        assert 'loss_feature_0_feature1' in metrics
        assert 'acc_feature_0_feature1' in metrics
        
        assert metrics['nrp_loss'] == 2.5
        assert metrics['overall_accuracy'] == 0.75
        assert metrics['prediction_coverage'] == 1.0


class TestNRPIntegration:
    """Test NRP integration with TabGPT model."""
    
    def test_nrp_with_tabgpt_model(self):
        """Test NRP objective with actual TabGPT model."""
        # Create temporal test data
        df = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=8, freq='D'),
            'customer_id': [1, 1, 1, 1, 2, 2, 2, 2],
            'transaction_amount': [100, 150, 200, 120, 80, 90, 110, 95],
            'category': ['A', 'B', 'A', 'C', 'B', 'A', 'C', 'B']
        })
        
        # Tokenize data
        tokenizer = TabularTokenizer(embedding_dim=64)
        tokenized = tokenizer.fit_transform(df)
        
        # Create NRP objective
        nrp_objective = NextRowPredictionObjective(
            d_model=64,
            column_metadata=tokenizer.column_metadata,
            n_temporal_layers=1,
            sequence_length=4,
            min_sequence_length=2,
            ordering_strategy=TemporalOrderingStrategy.AUTO_DETECT
        )
        
        # Forward pass
        output = nrp_objective(
            input_embeddings=tokenized.tokens,
            attention_mask=tokenized.attention_mask,
            df=df
        )
        
        assert isinstance(output, NextRowOutput)
        assert output.loss is not None
        assert not torch.isnan(output.loss)
        
        # Test metrics computation
        metrics = nrp_objective.compute_metrics(output)
        assert isinstance(metrics, dict)
        assert 'nrp_loss' in metrics
    
    def test_nrp_gradient_flow(self):
        """Test gradient flow through NRP objective."""
        batch_size, n_features, d_model = 5, 3, 32
        
        from tabgpt.tokenizers.tabular_tokenizer import ColumnMetadata
        column_metadata = [
            ColumnMetadata(name='feature1', dtype='numerical', unique_values=None),
            ColumnMetadata(name='feature2', dtype='categorical', unique_values=5),
            ColumnMetadata(name='feature3', dtype='boolean', unique_values=2)
        ]
        
        # Create NRP objective
        nrp_objective = NextRowPredictionObjective(
            d_model=d_model,
            column_metadata=column_metadata,
            n_temporal_layers=1,
            sequence_length=3
        )
        
        input_embeddings = torch.randn(batch_size, n_features, d_model, requires_grad=True)
        attention_mask = torch.ones(batch_size, n_features, dtype=torch.bool)
        
        # Forward pass
        output = nrp_objective(input_embeddings, attention_mask)
        
        # Backward pass
        if output.loss.requires_grad:
            output.loss.backward()
        
            # Check gradients
            assert input_embeddings.grad is not None
            assert not torch.isnan(input_embeddings.grad).any()
        
        # Check that NRP parameters have gradients
        nrp_params_with_grad = [p for p in nrp_objective.parameters() if p.grad is not None]
        assert len(nrp_params_with_grad) > 0
        
        for param in nrp_params_with_grad:
            assert not torch.isnan(param.grad).any()
    
    def test_nrp_different_sequence_lengths(self):
        """Test NRP with different sequence lengths."""
        batch_size, n_features, d_model = 8, 2, 16
        
        from tabgpt.tokenizers.tabular_tokenizer import ColumnMetadata
        column_metadata = [
            ColumnMetadata(name='feature1', dtype='numerical', unique_values=None),
            ColumnMetadata(name='feature2', dtype='numerical', unique_values=None)
        ]
        
        sequence_lengths = [3, 5, 7]
        
        for seq_len in sequence_lengths:
            objective = NextRowPredictionObjective(
                d_model=d_model,
                column_metadata=column_metadata,
                sequence_length=seq_len,
                min_sequence_length=2
            )
            
            input_embeddings = torch.randn(batch_size, n_features, d_model)
            attention_mask = torch.ones(batch_size, n_features, dtype=torch.bool)
            
            output = objective(input_embeddings, attention_mask)
            
            assert isinstance(output, NextRowOutput)
            assert output.loss is not None
            assert not torch.isnan(output.loss)


class TestTemporalDatasetCreation:
    """Test temporal dataset creation utilities."""
    
    def test_create_temporal_dataset_basic(self):
        """Test basic temporal dataset creation."""
        df = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=6, freq='D'),
            'value': range(6)
        })
        
        temporal_dfs = create_temporal_dataset_from_dataframe(
            df, timestamp_column='timestamp'
        )
        
        assert len(temporal_dfs) == 1
        assert len(temporal_dfs[0]) == 6
        # Should be sorted by timestamp
        assert temporal_dfs[0]['timestamp'].is_monotonic_increasing
    
    def test_create_temporal_dataset_grouped(self):
        """Test temporal dataset creation with grouping."""
        df = pd.DataFrame({
            'customer_id': [1, 2, 1, 2, 1, 2],
            'timestamp': pd.to_datetime(['2023-01-01', '2023-01-01', '2023-01-02', 
                                       '2023-01-02', '2023-01-03', '2023-01-03']),
            'value': [10, 20, 15, 25, 12, 22]
        })
        
        temporal_dfs = create_temporal_dataset_from_dataframe(
            df, 
            timestamp_column='timestamp',
            group_by_columns=['customer_id']
        )
        
        assert len(temporal_dfs) == 2  # Two customers
        
        # Each group should be sorted by timestamp
        for temporal_df in temporal_dfs:
            assert temporal_df['timestamp'].is_monotonic_increasing
            assert len(temporal_df['customer_id'].unique()) == 1
    
    def test_create_temporal_dataset_no_timestamp(self):
        """Test temporal dataset creation without timestamp column."""
        df = pd.DataFrame({
            'id': [1, 2, 3, 4],
            'value': [10, 20, 30, 40]
        })
        
        temporal_dfs = create_temporal_dataset_from_dataframe(df)
        
        assert len(temporal_dfs) == 1
        assert len(temporal_dfs[0]) == 4
        # Should preserve original order
        assert temporal_dfs[0]['value'].tolist() == [10, 20, 30, 40]


class TestNRPEdgeCases:
    """Test NRP edge cases and error handling."""
    
    def test_nrp_empty_sequences(self):
        """Test NRP with data that creates no valid sequences."""
        batch_size, n_features, d_model = 1, 2, 16  # Too small for sequences
        
        from tabgpt.tokenizers.tabular_tokenizer import ColumnMetadata
        column_metadata = [
            ColumnMetadata(name='feature1', dtype='numerical', unique_values=None),
            ColumnMetadata(name='feature2', dtype='numerical', unique_values=None)
        ]
        
        objective = NextRowPredictionObjective(
            d_model=d_model,
            column_metadata=column_metadata,
            sequence_length=5,  # Longer than available data
            min_sequence_length=3
        )
        
        input_embeddings = torch.randn(batch_size, n_features, d_model)
        attention_mask = torch.ones(batch_size, n_features, dtype=torch.bool)
        
        output = objective(input_embeddings, attention_mask)
        
        # Should handle gracefully
        assert isinstance(output, NextRowOutput)
        assert output.loss == 0.0
        assert output.accuracy['n_predicted_features'] == 0
    
    def test_nrp_single_feature_type(self):
        """Test NRP with single feature type."""
        batch_size, n_features, d_model = 4, 3, 16
        
        from tabgpt.tokenizers.tabular_tokenizer import ColumnMetadata
        column_metadata = [
            ColumnMetadata(name='num1', dtype='numerical', unique_values=None),
            ColumnMetadata(name='num2', dtype='numerical', unique_values=None),
            ColumnMetadata(name='num3', dtype='numerical', unique_values=None)
        ]
        
        objective = NextRowPredictionObjective(
            d_model=d_model,
            column_metadata=column_metadata,
            sequence_length=3,
            min_sequence_length=2
        )
        
        input_embeddings = torch.randn(batch_size, n_features, d_model)
        attention_mask = torch.ones(batch_size, n_features, dtype=torch.bool)
        
        output = objective(input_embeddings, attention_mask)
        
        assert isinstance(output, NextRowOutput)
        assert output.loss is not None
        assert not torch.isnan(output.loss)
    
    def test_nrp_masked_features(self):
        """Test NRP with partially masked features."""
        batch_size, n_features, d_model = 4, 3, 16
        
        from tabgpt.tokenizers.tabular_tokenizer import ColumnMetadata
        column_metadata = [
            ColumnMetadata(name='feature1', dtype='numerical', unique_values=None),
            ColumnMetadata(name='feature2', dtype='categorical', unique_values=5),
            ColumnMetadata(name='feature3', dtype='boolean', unique_values=2)
        ]
        
        objective = NextRowPredictionObjective(
            d_model=d_model,
            column_metadata=column_metadata,
            sequence_length=3
        )
        
        input_embeddings = torch.randn(batch_size, n_features, d_model)
        # Mask some features
        attention_mask = torch.tensor([
            [True, True, False],
            [True, False, True],
            [False, True, True],
            [True, True, True]
        ])
        
        output = objective(input_embeddings, attention_mask)
        
        assert isinstance(output, NextRowOutput)
        assert output.loss is not None