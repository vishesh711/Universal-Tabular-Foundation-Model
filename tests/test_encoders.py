"""Tests for feature encoders."""

import pytest
import pandas as pd
import torch
import numpy as np
from datetime import datetime, date

from tabgpt.tokenizers.encoders import (
    CategoricalEncoder,
    NumericalEncoder, 
    DatetimeEncoder,
    BooleanEncoder
)


class TestCategoricalEncoder:
    """Test categorical encoder."""
    
    def test_basic_encoding(self):
        """Test basic categorical encoding."""
        data = pd.Series(['A', 'B', 'A', 'C', 'B'])
        encoder = CategoricalEncoder(embedding_dim=64)
        
        encoder.fit(data)
        embeddings = encoder.transform(data)
        
        assert embeddings.shape == (5, 64)
        assert encoder.vocab_size >= 3  # At least A, B, C + special tokens
    
    def test_oov_handling(self):
        """Test out-of-vocabulary handling."""
        train_data = pd.Series(['A', 'B', 'A'])
        test_data = pd.Series(['A', 'B', 'C'])  # C is OOV
        
        encoder = CategoricalEncoder(embedding_dim=32)
        encoder.fit(train_data)
        
        # Should handle OOV gracefully
        embeddings = encoder.transform(test_data)
        assert embeddings.shape == (3, 32)
    
    def test_missing_values(self):
        """Test missing value handling."""
        data = pd.Series(['A', None, 'B', np.nan])
        encoder = CategoricalEncoder(embedding_dim=32)
        
        encoder.fit(data)
        embeddings = encoder.transform(data)
        
        assert embeddings.shape == (4, 32)
        assert not torch.isnan(embeddings).any()
    
    def test_vocab_size_limit(self):
        """Test vocabulary size limiting."""
        # Create data with many unique values
        data = pd.Series([f'cat_{i}' for i in range(1000)])
        encoder = CategoricalEncoder(embedding_dim=32, max_vocab_size=100)
        
        encoder.fit(data)
        assert encoder.vocab_size <= 100


class TestNumericalEncoder:
    """Test numerical encoder."""
    
    def test_basic_encoding(self):
        """Test basic numerical encoding."""
        data = pd.Series([1.0, 2.5, 3.7, 4.2, 5.1])
        encoder = NumericalEncoder(embedding_dim=64)
        
        encoder.fit(data)
        embeddings = encoder.transform(data)
        
        assert embeddings.shape == (5, 64)
        assert not torch.isnan(embeddings).any()
    
    def test_missing_values(self):
        """Test missing value handling."""
        data = pd.Series([1.0, None, 3.0, np.nan, 5.0])
        encoder = NumericalEncoder(embedding_dim=32)
        
        encoder.fit(data)
        embeddings = encoder.transform(data)
        
        assert embeddings.shape == (5, 32)
        assert not torch.isnan(embeddings).any()
    
    def test_different_strategies(self):
        """Test different binning strategies."""
        data = pd.Series(np.random.randn(100))
        
        # Test quantile strategy
        encoder_quantile = NumericalEncoder(embedding_dim=32, strategy='quantile')
        encoder_quantile.fit(data)
        emb_quantile = encoder_quantile.transform(data)
        
        # Test uniform strategy
        encoder_uniform = NumericalEncoder(embedding_dim=32, strategy='uniform')
        encoder_uniform.fit(data)
        emb_uniform = encoder_uniform.transform(data)
        
        assert emb_quantile.shape == emb_uniform.shape == (100, 32)
    
    def test_normalization_methods(self):
        """Test different normalization methods."""
        data = pd.Series([1, 10, 100, 1000, 10000])  # Wide range
        
        # Test standard normalization
        encoder_std = NumericalEncoder(embedding_dim=32, normalization='standard')
        encoder_std.fit(data)
        emb_std = encoder_std.transform(data)
        
        # Test robust normalization
        encoder_robust = NumericalEncoder(embedding_dim=32, normalization='robust')
        encoder_robust.fit(data)
        emb_robust = encoder_robust.transform(data)
        
        assert emb_std.shape == emb_robust.shape == (5, 32)


class TestDatetimeEncoder:
    """Test datetime encoder."""
    
    def test_basic_encoding(self):
        """Test basic datetime encoding."""
        dates = pd.date_range('2023-01-01', periods=5, freq='D')
        data = pd.Series(dates)
        encoder = DatetimeEncoder(embedding_dim=64)
        
        encoder.fit(data)
        embeddings = encoder.transform(data)
        
        assert embeddings.shape == (5, 64)
        assert not torch.isnan(embeddings).any()
    
    def test_string_dates(self):
        """Test string date conversion."""
        data = pd.Series(['2023-01-01', '2023-06-15', '2023-12-31'])
        encoder = DatetimeEncoder(embedding_dim=32)
        
        encoder.fit(data)
        embeddings = encoder.transform(data)
        
        assert embeddings.shape == (3, 32)
    
    def test_missing_dates(self):
        """Test missing datetime handling."""
        data = pd.Series([datetime(2023, 1, 1), None, datetime(2023, 12, 31)])
        encoder = DatetimeEncoder(embedding_dim=32)
        
        encoder.fit(data)
        embeddings = encoder.transform(data)
        
        assert embeddings.shape == (3, 32)
        assert not torch.isnan(embeddings).any()
    
    def test_cyclical_encoding(self):
        """Test that cyclical patterns are captured."""
        # Test that December and January are close (month cyclical)
        dates = [datetime(2023, 1, 1), datetime(2023, 12, 31)]
        data = pd.Series(dates)
        encoder = DatetimeEncoder(embedding_dim=64)
        
        encoder.fit(data)
        embeddings = encoder.transform(data)
        
        # Should produce different but related embeddings
        assert embeddings.shape == (2, 64)
        assert not torch.equal(embeddings[0], embeddings[1])


class TestBooleanEncoder:
    """Test boolean encoder."""
    
    def test_basic_encoding(self):
        """Test basic boolean encoding."""
        data = pd.Series([True, False, True, False])
        encoder = BooleanEncoder(embedding_dim=32)
        
        encoder.fit(data)
        embeddings = encoder.transform(data)
        
        assert embeddings.shape == (4, 32)
        assert not torch.isnan(embeddings).any()
    
    def test_missing_booleans(self):
        """Test missing boolean handling."""
        data = pd.Series([True, None, False, np.nan])
        encoder = BooleanEncoder(embedding_dim=16)
        
        encoder.fit(data)
        embeddings = encoder.transform(data)
        
        assert embeddings.shape == (4, 16)
        assert not torch.isnan(embeddings).any()
    
    def test_different_embeddings(self):
        """Test that True/False/NaN produce different embeddings."""
        data = pd.Series([True, False, None])
        encoder = BooleanEncoder(embedding_dim=8)
        
        encoder.fit(data)
        embeddings = encoder.transform(data)
        
        # All three should be different
        assert not torch.equal(embeddings[0], embeddings[1])
        assert not torch.equal(embeddings[0], embeddings[2])
        assert not torch.equal(embeddings[1], embeddings[2])


class TestEncoderIntegration:
    """Test encoder integration."""
    
    def test_mixed_data_types(self):
        """Test handling mixed data types."""
        df = pd.DataFrame({
            'categorical': ['A', 'B', 'A', 'C'],
            'numerical': [1.5, 2.7, 3.1, 4.9],
            'datetime': pd.date_range('2023-01-01', periods=4),
            'boolean': [True, False, True, False]
        })
        
        # Test each encoder individually
        cat_encoder = CategoricalEncoder(embedding_dim=16)
        num_encoder = NumericalEncoder(embedding_dim=16)
        dt_encoder = DatetimeEncoder(embedding_dim=16)
        bool_encoder = BooleanEncoder(embedding_dim=16)
        
        cat_encoder.fit(df['categorical'])
        num_encoder.fit(df['numerical'])
        dt_encoder.fit(df['datetime'])
        bool_encoder.fit(df['boolean'])
        
        cat_emb = cat_encoder.transform(df['categorical'])
        num_emb = num_encoder.transform(df['numerical'])
        dt_emb = dt_encoder.transform(df['datetime'])
        bool_emb = bool_encoder.transform(df['boolean'])
        
        assert cat_emb.shape == (4, 16)
        assert num_emb.shape == (4, 16)
        assert dt_emb.shape == (4, 16)
        assert bool_emb.shape == (4, 16)