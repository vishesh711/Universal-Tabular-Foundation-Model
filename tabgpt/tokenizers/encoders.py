"""Feature encoders for different data types."""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from datetime import datetime
import math

from .tabular_tokenizer import FeatureEncoder


class CategoricalEncoder(FeatureEncoder):
    """Encoder for categorical features with vocabulary management."""
    
    def __init__(self, embedding_dim: int, max_vocab_size: int = 10000):
        self.embedding_dim = embedding_dim
        self.max_vocab_size = max_vocab_size
        self.label_encoder = LabelEncoder()
        self.vocab_size = 0
        self.embedding = None
        self.oov_token = "<UNK>"
        self.mask_token = "<MASK>"
        self.is_fitted = False
    
    def fit(self, data: pd.Series) -> None:
        """Fit encoder to categorical data."""
        # Handle missing values
        data_clean = data.fillna("NaN")
        
        # Add special tokens
        unique_values = list(data_clean.unique())
        if self.oov_token not in unique_values:
            unique_values.append(self.oov_token)
        if self.mask_token not in unique_values:
            unique_values.append(self.mask_token)
        
        # Limit vocabulary size
        if len(unique_values) > self.max_vocab_size:
            # Keep most frequent values
            value_counts = data_clean.value_counts()
            top_values = value_counts.head(self.max_vocab_size - 2).index.tolist()
            unique_values = top_values + [self.oov_token, self.mask_token]
        
        # Fit label encoder
        self.label_encoder.fit(unique_values)
        self.vocab_size = len(unique_values)
        
        # Create embedding layer
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        nn.init.xavier_uniform_(self.embedding.weight)
        
        self.is_fitted = True
    
    def transform(self, data: pd.Series) -> torch.Tensor:
        """Transform categorical data to embeddings."""
        if not self.is_fitted:
            raise ValueError("Encoder must be fitted before transform")
        
        # Handle missing values
        data_clean = data.fillna("NaN")
        
        # Handle OOV values
        known_classes = set(self.label_encoder.classes_)
        data_processed = []
        for value in data_clean:
            if value in known_classes:
                data_processed.append(value)
            else:
                data_processed.append(self.oov_token)
        
        # Encode to indices
        indices = self.label_encoder.transform(data_processed)
        indices_tensor = torch.tensor(indices, dtype=torch.long)
        
        # Get embeddings
        embeddings = self.embedding(indices_tensor)
        return embeddings
    
    def get_embedding_dim(self) -> int:
        """Get embedding dimension."""
        return self.embedding_dim
    
    def get_mask_token_id(self) -> int:
        """Get mask token ID."""
        if not self.is_fitted:
            raise ValueError("Encoder must be fitted first")
        return self.label_encoder.transform([self.mask_token])[0]


class NumericalEncoder(FeatureEncoder):
    """Encoder for numerical features with binning and normalization."""
    
    def __init__(
        self, 
        embedding_dim: int, 
        n_bins: int = 100,
        strategy: str = 'quantile',
        normalization: str = 'standard'
    ):
        self.embedding_dim = embedding_dim
        self.n_bins = n_bins
        self.strategy = strategy
        self.normalization = normalization
        
        self.scaler = None
        self.bin_edges = None
        self.embedding = None
        self.linear_projection = None
        self.is_fitted = False
        
        # Special tokens
        self.mask_value = -999.0
        self.nan_bin = n_bins  # Special bin for NaN values
    
    def fit(self, data: pd.Series) -> None:
        """Fit encoder to numerical data."""
        # Remove NaN for fitting
        data_clean = data.dropna()
        
        if len(data_clean) == 0:
            # Handle case with all NaN values
            self.bin_edges = np.array([0, 1])
            self.scaler = StandardScaler()
            self.scaler.fit([[0], [1]])
        else:
            # Fit scaler
            if self.normalization == 'standard':
                self.scaler = StandardScaler()
            elif self.normalization == 'robust':
                self.scaler = RobustScaler()
            else:
                raise ValueError(f"Unknown normalization: {self.normalization}")
            
            self.scaler.fit(data_clean.values.reshape(-1, 1))
            
            # Create bins
            if self.strategy == 'quantile':
                self.bin_edges = np.quantile(
                    data_clean, 
                    np.linspace(0, 1, self.n_bins + 1)
                )
            elif self.strategy == 'uniform':
                self.bin_edges = np.linspace(
                    data_clean.min(), 
                    data_clean.max(), 
                    self.n_bins + 1
                )
            else:
                raise ValueError(f"Unknown binning strategy: {self.strategy}")
        
        # Create embedding for bins (including NaN bin)
        self.embedding = nn.Embedding(self.n_bins + 1, self.embedding_dim // 2)
        
        # Linear projection for normalized values
        self.linear_projection = nn.Linear(1, self.embedding_dim // 2)
        
        # Initialize weights
        nn.init.xavier_uniform_(self.embedding.weight)
        nn.init.xavier_uniform_(self.linear_projection.weight)
        
        self.is_fitted = True
    
    def transform(self, data: pd.Series) -> torch.Tensor:
        """Transform numerical data to embeddings."""
        if not self.is_fitted:
            raise ValueError("Encoder must be fitted before transform")
        
        batch_size = len(data)
        
        # Handle NaN values
        is_nan = data.isna()
        data_filled = data.fillna(0.0)  # Temporary fill for processing
        
        # Normalize values
        normalized = self.scaler.transform(data_filled.values.reshape(-1, 1))
        normalized_tensor = torch.tensor(normalized, dtype=torch.float32)
        
        # Bin values
        bin_indices = np.digitize(data_filled.values, self.bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, self.n_bins - 1)
        
        # Set NaN values to special bin
        bin_indices[is_nan] = self.nan_bin
        bin_indices_tensor = torch.tensor(bin_indices, dtype=torch.long)
        
        # Get embeddings
        bin_embeddings = self.embedding(bin_indices_tensor)
        value_embeddings = self.linear_projection(normalized_tensor)
        
        # Handle NaN in value embeddings
        nan_mask = torch.tensor(is_nan.values, dtype=torch.bool)
        value_embeddings[nan_mask] = 0.0
        
        # Concatenate bin and value embeddings
        combined_embeddings = torch.cat([bin_embeddings, value_embeddings], dim=-1)
        
        return combined_embeddings
    
    def get_embedding_dim(self) -> int:
        """Get embedding dimension."""
        return self.embedding_dim


class DatetimeEncoder(FeatureEncoder):
    """Encoder for datetime features with cyclical and temporal embeddings."""
    
    def __init__(self, embedding_dim: int):
        self.embedding_dim = embedding_dim
        self.is_fitted = False
        
        # Temporal components
        self.year_scaler = StandardScaler()
        self.linear_projection = None
        
    def fit(self, data: pd.Series) -> None:
        """Fit encoder to datetime data."""
        # Convert to datetime if not already
        if not pd.api.types.is_datetime64_any_dtype(data):
            data = pd.to_datetime(data, errors='coerce')
        
        # Remove NaN for fitting
        data_clean = data.dropna()
        
        if len(data_clean) > 0:
            # Fit year scaler
            years = data_clean.dt.year.values.reshape(-1, 1)
            self.year_scaler.fit(years)
        else:
            # Handle all NaN case
            self.year_scaler.fit([[2000], [2023]])
        
        # Linear projection for all temporal features
        # Features: year, month_sin, month_cos, day_sin, day_cos, hour_sin, hour_cos, weekday_sin, weekday_cos
        n_temporal_features = 9
        self.linear_projection = nn.Linear(n_temporal_features, self.embedding_dim)
        nn.init.xavier_uniform_(self.linear_projection.weight)
        
        self.is_fitted = True
    
    def transform(self, data: pd.Series) -> torch.Tensor:
        """Transform datetime data to embeddings."""
        if not self.is_fitted:
            raise ValueError("Encoder must be fitted before transform")
        
        # Convert to datetime if not already
        if not pd.api.types.is_datetime64_any_dtype(data):
            data = pd.to_datetime(data, errors='coerce')
        
        batch_size = len(data)
        features = torch.zeros(batch_size, 9)
        
        # Handle NaN values
        is_nan = data.isna()
        
        for i, dt in enumerate(data):
            if pd.isna(dt):
                # Use default values for NaN
                features[i] = torch.zeros(9)
            else:
                # Extract temporal components
                year_norm = self.year_scaler.transform([[dt.year]])[0, 0]
                
                # Cyclical encoding
                month_angle = 2 * math.pi * (dt.month - 1) / 12
                day_angle = 2 * math.pi * (dt.day - 1) / 31
                hour_angle = 2 * math.pi * dt.hour / 24
                weekday_angle = 2 * math.pi * dt.weekday() / 7
                
                features[i] = torch.tensor([
                    year_norm,
                    math.sin(month_angle), math.cos(month_angle),
                    math.sin(day_angle), math.cos(day_angle),
                    math.sin(hour_angle), math.cos(hour_angle),
                    math.sin(weekday_angle), math.cos(weekday_angle)
                ])
        
        # Project to embedding dimension
        embeddings = self.linear_projection(features)
        
        return embeddings
    
    def get_embedding_dim(self) -> int:
        """Get embedding dimension."""
        return self.embedding_dim


class BooleanEncoder(FeatureEncoder):
    """Encoder for boolean features."""
    
    def __init__(self, embedding_dim: int):
        self.embedding_dim = embedding_dim
        self.embedding = None
        self.is_fitted = False
    
    def fit(self, data: pd.Series) -> None:
        """Fit encoder to boolean data."""
        # Boolean has 3 states: True, False, NaN
        vocab_size = 3
        self.embedding = nn.Embedding(vocab_size, self.embedding_dim)
        nn.init.xavier_uniform_(self.embedding.weight)
        self.is_fitted = True
    
    def transform(self, data: pd.Series) -> torch.Tensor:
        """Transform boolean data to embeddings."""
        if not self.is_fitted:
            raise ValueError("Encoder must be fitted before transform")
        
        # Map boolean values to indices
        # True -> 0, False -> 1, NaN -> 2
        indices = []
        for value in data:
            if pd.isna(value):
                indices.append(2)
            elif value:
                indices.append(0)
            else:
                indices.append(1)
        
        indices_tensor = torch.tensor(indices, dtype=torch.long)
        embeddings = self.embedding(indices_tensor)
        
        return embeddings
    
    def get_embedding_dim(self) -> int:
        """Get embedding dimension."""
        return self.embedding_dim


class DummyEncoder(FeatureEncoder):
    """Dummy encoder for fallback cases."""
    
    def __init__(self, embedding_dim: int):
        self.embedding_dim = embedding_dim
    
    def fit(self, data: pd.Series) -> None:
        """Fit encoder to data."""
        pass
    
    def transform(self, data: pd.Series) -> torch.Tensor:
        """Transform data to embeddings."""
        batch_size = len(data)
        return torch.zeros(batch_size, self.embedding_dim)
    
    def get_embedding_dim(self) -> int:
        """Get embedding dimension."""
        return self.embedding_dim