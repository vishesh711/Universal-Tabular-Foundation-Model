"""Tests for TabularTokenizer."""

import pytest
import pandas as pd
import torch
import numpy as np
from tabgpt.tokenizers import TabularTokenizer


def test_tokenizer_creation():
    """Test basic tokenizer creation."""
    tokenizer = TabularTokenizer()
    assert tokenizer.vocab_size == 10000
    assert tokenizer.embedding_dim == 256
    assert not tokenizer.is_fitted


def test_tokenizer_fit():
    """Test tokenizer fitting."""
    # Create sample data
    df = pd.DataFrame({
        'numerical': [1.0, 2.0, 3.0, 4.0],
        'categorical': ['A', 'B', 'A', 'C'],
        'boolean': [True, False, True, False]
    })
    
    tokenizer = TabularTokenizer()
    tokenizer.fit(df)
    
    assert tokenizer.is_fitted
    assert len(tokenizer.feature_names) == 3
    assert len(tokenizer.column_metadata) == 3


def test_tokenizer_transform():
    """Test tokenizer transformation."""
    # Create sample data
    df = pd.DataFrame({
        'numerical': [1.0, 2.0, 3.0],
        'categorical': ['A', 'B', 'A']
    })
    
    tokenizer = TabularTokenizer(embedding_dim=128)
    tokenized = tokenizer.fit_transform(df)
    
    # Check output structure
    assert isinstance(tokenized.tokens, torch.Tensor)
    assert isinstance(tokenized.attention_mask, torch.Tensor)
    assert tokenized.tokens.shape == (3, 2, 128)  # batch_size, n_features, embedding_dim
    assert tokenized.attention_mask.shape == (3, 2)  # batch_size, n_features


def test_column_metadata_creation():
    """Test column metadata creation."""
    df = pd.DataFrame({
        'numerical': [1.0, 2.0, 3.0, 4.0, 5.0],
        'categorical': ['A', 'B', 'A', 'C', 'B'],
        'datetime': pd.date_range('2023-01-01', periods=5),
        'boolean': [True, False, True, False, True]
    })
    
    tokenizer = TabularTokenizer()
    tokenizer.fit(df)
    
    metadata = tokenizer.column_metadata
    assert len(metadata) == 4
    
    # Check numerical column
    num_meta = next(m for m in metadata if m.name == 'numerical')
    assert num_meta.dtype == 'numerical'
    assert num_meta.cardinality == 5
    
    # Check categorical column  
    cat_meta = next(m for m in metadata if m.name == 'categorical')
    assert cat_meta.dtype == 'categorical'
    assert cat_meta.cardinality == 3
    
    # Check datetime column
    dt_meta = next(m for m in metadata if m.name == 'datetime')
    assert dt_meta.dtype == 'datetime'
    
    # Check boolean column
    bool_meta = next(m for m in metadata if m.name == 'boolean')
    assert bool_meta.dtype == 'boolean'


def test_missing_values_handling():
    """Test handling of missing values."""
    df = pd.DataFrame({
        'numerical': [1.0, None, 3.0],
        'categorical': ['A', 'B', None]
    })
    
    tokenizer = TabularTokenizer()
    tokenized = tokenizer.fit_transform(df)
    
    # Should handle missing values gracefully
    assert tokenized.tokens.shape == (3, 2, 256)
    assert not torch.isnan(tokenized.tokens).any()


def test_comprehensive_data_types():
    """Test comprehensive data type handling."""
    df = pd.DataFrame({
        'int_categorical': [1, 2, 1, 3, 2],  # Low cardinality int -> categorical
        'float_numerical': [1.5, 2.7, 3.9, 4.1, 5.3],  # Float -> numerical
        'string_categorical': ['red', 'blue', 'red', 'green', 'blue'],
        'datetime_col': pd.date_range('2023-01-01', periods=5, freq='D'),
        'bool_col': [True, False, True, False, True],
        'mixed_with_nan': [1.0, np.nan, 3.0, None, 5.0]
    })
    
    tokenizer = TabularTokenizer(embedding_dim=64)
    tokenized = tokenizer.fit_transform(df)
    
    # Check shapes
    assert tokenized.tokens.shape == (5, 6, 64)
    assert tokenized.attention_mask.shape == (5, 6)
    
    # Check metadata types
    metadata_dict = {m.name: m.dtype for m in tokenized.column_metadata}
    assert metadata_dict['int_categorical'] == 'categorical'
    assert metadata_dict['float_numerical'] == 'numerical'
    assert metadata_dict['string_categorical'] == 'categorical'
    assert metadata_dict['datetime_col'] == 'datetime'
    assert metadata_dict['bool_col'] == 'boolean'
    assert metadata_dict['mixed_with_nan'] == 'numerical'


def test_statistical_profiles():
    """Test statistical profile computation."""
    df = pd.DataFrame({
        'numerical': [1.0, 2.0, 3.0, 4.0, 5.0, 100.0],  # Has outlier
        'categorical': ['A', 'B', 'A', 'C', 'B', 'A']
    })
    
    tokenizer = TabularTokenizer()
    tokenizer.fit(df)
    
    # Check numerical column statistics
    num_meta = next(m for m in tokenizer.column_metadata if m.name == 'numerical')
    stats = num_meta.statistical_profile
    
    assert 'mean' in stats
    assert 'std' in stats
    assert 'min' in stats
    assert 'max' in stats
    assert stats['min'] == 1.0
    assert stats['max'] == 100.0
    assert abs(stats['mean'] - df['numerical'].mean()) < 1e-6


def test_encoder_consistency():
    """Test that encoders produce consistent outputs."""
    df = pd.DataFrame({
        'categorical': ['A', 'B', 'C', 'A', 'B'],
        'numerical': [1.0, 2.0, 3.0, 1.0, 2.0]  # Repeated values
    })
    
    tokenizer = TabularTokenizer(embedding_dim=32)
    tokenizer.fit(df)
    
    # Transform twice
    tokenized1 = tokenizer.transform(df)
    tokenized2 = tokenizer.transform(df)
    
    # Should be identical
    assert torch.equal(tokenized1.tokens, tokenized2.tokens)
    assert torch.equal(tokenized1.attention_mask, tokenized2.attention_mask)
    
    # Same values should produce same embeddings
    # Row 0 and 3 have same values, row 1 and 4 have same values
    assert torch.equal(tokenized1.tokens[0], tokenized1.tokens[3])
    assert torch.equal(tokenized1.tokens[1], tokenized1.tokens[4])


def test_large_vocabulary():
    """Test handling of large vocabulary."""
    # Create data with many unique categorical values
    n_samples = 1000
    n_unique = 500
    categories = [f'cat_{i % n_unique}' for i in range(n_samples)]
    
    df = pd.DataFrame({
        'large_categorical': categories,
        'numerical': np.random.randn(n_samples)
    })
    
    tokenizer = TabularTokenizer(vocab_size=100, embedding_dim=32)
    tokenized = tokenizer.fit_transform(df)
    
    # Should handle gracefully with vocabulary limiting
    assert tokenized.tokens.shape == (n_samples, 2, 32)
    
    # Check that categorical encoder has limited vocab
    cat_encoder = tokenizer.encoders['large_categorical']
    assert cat_encoder.vocab_size <= 100


def test_empty_dataframe():
    """Test handling of edge cases."""
    # Empty dataframe
    df_empty = pd.DataFrame()
    tokenizer = TabularTokenizer()
    
    with pytest.raises(Exception):  # Should raise some error
        tokenizer.fit(df_empty)
    
    # Single row
    df_single = pd.DataFrame({'col1': [1], 'col2': ['A']})
    tokenizer_single = TabularTokenizer(embedding_dim=16)
    tokenized = tokenizer_single.fit_transform(df_single)
    
    assert tokenized.tokens.shape == (1, 2, 16)


def test_column_order_consistency():
    """Test that column order is preserved."""
    df = pd.DataFrame({
        'z_col': [1, 2, 3],
        'a_col': ['X', 'Y', 'Z'],
        'm_col': [True, False, True]
    })
    
    tokenizer = TabularTokenizer()
    tokenizer.fit(df)
    
    # Feature names should preserve original order
    assert tokenizer.feature_names == ['z_col', 'a_col', 'm_col']
    
    # Metadata should be in same order
    metadata_names = [m.name for m in tokenizer.column_metadata]
    assert metadata_names == ['z_col', 'a_col', 'm_col']