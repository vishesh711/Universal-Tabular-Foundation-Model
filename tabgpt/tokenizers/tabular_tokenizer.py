"""Tabular data tokenizer."""

import torch
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class ColumnMetadata:
    """Metadata for a single column."""
    name: str
    dtype: str  # 'categorical', 'numerical', 'datetime', 'boolean'
    unique_values: Optional[int]  # Number of unique values, None for numerical
    missing_rate: float = 0.0
    statistical_profile: Optional[Dict[str, float]] = None


@dataclass
class TokenizedTable:
    """Tokenized tabular data."""
    tokens: torch.Tensor  # [batch_size, n_features, embedding_dim]
    attention_mask: torch.Tensor  # [batch_size, n_features]
    column_metadata: List[ColumnMetadata]
    feature_names: List[str]


class FeatureEncoder(ABC):
    """Abstract base class for feature encoders."""
    
    @abstractmethod
    def fit(self, data: pd.Series) -> None:
        """Fit encoder to data."""
        pass
    
    @abstractmethod
    def transform(self, data: pd.Series) -> torch.Tensor:
        """Transform data to embeddings."""
        pass
    
    @abstractmethod
    def get_embedding_dim(self) -> int:
        """Get embedding dimension."""
        pass


class TabularTokenizer:
    """
    Tokenizer for tabular data that converts heterogeneous features
    into uniform token representations.
    """
    
    def __init__(
        self,
        vocab_size: int = 10000,
        embedding_dim: int = 256,
        max_features: int = 512,
        handle_missing: bool = True
    ):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_features = max_features
        self.handle_missing = handle_missing
        
        # Feature encoders for different data types
        self.encoders: Dict[str, FeatureEncoder] = {}
        self.column_metadata: List[ColumnMetadata] = []
        self.feature_names: List[str] = []
        self.is_fitted = False
    
    def fit(self, dataframe: pd.DataFrame) -> 'TabularTokenizer':
        """
        Fit tokenizer to tabular data.
        
        Args:
            dataframe: Input pandas DataFrame
            
        Returns:
            Self for method chaining
        """
        if dataframe.empty:
            raise ValueError("Cannot fit tokenizer on empty DataFrame")
        
        self.feature_names = list(dataframe.columns)
        self.column_metadata = []
        self.encoders = {}
        
        for col in dataframe.columns:
            # Infer column type and create metadata
            metadata = self._create_column_metadata(dataframe[col], col)
            self.column_metadata.append(metadata)
            
            # Create and fit appropriate encoder
            encoder = self._create_encoder(metadata.dtype)
            encoder.fit(dataframe[col])
            self.encoders[col] = encoder
        
        self.is_fitted = True
        return self
    
    def transform(self, dataframe: pd.DataFrame) -> TokenizedTable:
        """
        Transform tabular data to tokens.
        
        Args:
            dataframe: Input pandas DataFrame
            
        Returns:
            TokenizedTable with embeddings and metadata
        """
        if not self.is_fitted:
            raise ValueError("Tokenizer must be fitted before transform")
        
        batch_size = len(dataframe)
        n_features = len(self.feature_names)
        
        # Initialize token tensor
        tokens = torch.zeros(batch_size, n_features, self.embedding_dim)
        attention_mask = torch.ones(batch_size, n_features, dtype=torch.bool)
        
        # Transform each feature
        for i, col in enumerate(self.feature_names):
            if col in dataframe.columns:
                encoder = self.encoders[col]
                feature_tokens = encoder.transform(dataframe[col])
                tokens[:, i, :] = feature_tokens
            else:
                # Handle missing columns
                attention_mask[:, i] = False
        
        return TokenizedTable(
            tokens=tokens,
            attention_mask=attention_mask,
            column_metadata=self.column_metadata,
            feature_names=self.feature_names
        )
    
    def fit_transform(self, dataframe: pd.DataFrame) -> TokenizedTable:
        """Fit tokenizer and transform data in one step."""
        return self.fit(dataframe).transform(dataframe)
    
    def _create_column_metadata(self, series: pd.Series, name: str) -> ColumnMetadata:
        """Create metadata for a column."""
        # Infer data type - check boolean first before numeric
        if pd.api.types.is_bool_dtype(series):
            dtype = 'boolean'
        elif pd.api.types.is_datetime64_any_dtype(series):
            dtype = 'datetime'
        elif pd.api.types.is_numeric_dtype(series):
            if series.nunique() <= 10 and series.dtype in ['int64', 'int32']:
                dtype = 'categorical'
            else:
                dtype = 'numerical'
        else:
            dtype = 'categorical'
        
        # Compute statistics
        unique_values = series.nunique() if dtype in ['categorical', 'boolean'] else None
        missing_rate = series.isnull().mean()
        
        statistical_profile = None
        if dtype == 'numerical':
            statistical_profile = {
                'mean': float(series.mean()) if not series.empty else 0.0,
                'std': float(series.std()) if not series.empty else 0.0,
                'min': float(series.min()) if not series.empty else 0.0,
                'max': float(series.max()) if not series.empty else 0.0,
                'skew': float(series.skew()) if not series.empty else 0.0,
                'kurtosis': float(series.kurtosis()) if not series.empty else 0.0,
            }
        
        return ColumnMetadata(
            name=name,
            dtype=dtype,
            unique_values=unique_values,
            missing_rate=missing_rate,
            statistical_profile=statistical_profile
        )
    
    def _create_encoder(self, dtype: str) -> FeatureEncoder:
        """Create appropriate encoder for data type."""
        from .encoders import (
            CategoricalEncoder, 
            NumericalEncoder, 
            DatetimeEncoder, 
            BooleanEncoder,
            DummyEncoder
        )
        
        if dtype == 'categorical':
            return CategoricalEncoder(
                embedding_dim=self.embedding_dim,
                max_vocab_size=self.vocab_size
            )
        elif dtype == 'numerical':
            return NumericalEncoder(
                embedding_dim=self.embedding_dim,
                n_bins=100,
                strategy='quantile',
                normalization='standard'
            )
        elif dtype == 'datetime':
            return DatetimeEncoder(embedding_dim=self.embedding_dim)
        elif dtype == 'boolean':
            return BooleanEncoder(embedding_dim=self.embedding_dim)
        else:
            # Fallback to dummy encoder
            return DummyEncoder(self.embedding_dim)