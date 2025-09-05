"""Column encoder for semantic understanding and cross-dataset transfer."""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
import hashlib
import json

from ..tokenizers.tabular_tokenizer import ColumnMetadata


@dataclass
class ColumnEmbedding:
    """Embedding representation of a column."""
    name_embedding: torch.Tensor
    statistical_embedding: torch.Tensor
    type_embedding: torch.Tensor
    distribution_embedding: torch.Tensor
    combined_embedding: torch.Tensor
    metadata: ColumnMetadata


class ColumnEncoder(nn.Module):
    """
    Encoder that generates semantic embeddings for columns to enable
    cross-dataset transfer learning.
    """
    
    def __init__(
        self,
        embedding_dim: int = 128,
        statistical_features: int = 8,
        distribution_bins: int = 32,
        text_encoder_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.statistical_features = statistical_features
        self.distribution_bins = distribution_bins
        self.text_encoder_model = text_encoder_model
        
        # Initialize text encoder for column names
        self._init_text_encoder()
        
        # Type embeddings for different column types
        self.type_vocab = {
            'categorical': 0,
            'numerical': 1, 
            'datetime': 2,
            'boolean': 3,
            'unknown': 4
        }
        self.type_embedding = nn.Embedding(
            len(self.type_vocab), 
            embedding_dim // 4
        )
        
        # Statistical profile encoder
        self.statistical_encoder = nn.Sequential(
            nn.Linear(statistical_features, embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(embedding_dim // 2, embedding_dim // 4)
        )
        
        # Distribution encoder
        self.distribution_encoder = nn.Sequential(
            nn.Linear(distribution_bins, embedding_dim // 2),
            nn.ReLU(), 
            nn.Linear(embedding_dim // 2, embedding_dim // 4)
        )
        
        # Final fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_text_encoder(self):
        """Initialize text encoder for column names."""
        try:
            from sentence_transformers import SentenceTransformer
            self.text_encoder = SentenceTransformer(self.text_encoder_model)
            self.text_embedding_dim = self.text_encoder.get_sentence_embedding_dimension()
            
            # Project text embeddings to our embedding dimension
            self.text_projection = nn.Linear(
                self.text_embedding_dim, 
                self.embedding_dim // 4
            )
        except ImportError:
            # Fallback to simple hash-based encoding if sentence-transformers not available
            print("Warning: sentence-transformers not available, using hash-based encoding")
            self.text_encoder = None
            self.text_projection = nn.Linear(64, self.embedding_dim // 4)  # Hash to 64-dim
    
    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.xavier_uniform_(module.weight)
    
    def encode_column_name(self, column_name: str) -> torch.Tensor:
        """
        Encode column name using semantic text encoder.
        
        Args:
            column_name: Name of the column
            
        Returns:
            Text embedding tensor
        """
        if self.text_encoder is not None:
            # Use sentence transformer
            with torch.no_grad():
                text_emb = self.text_encoder.encode([column_name])
                text_emb = torch.tensor(text_emb, dtype=torch.float32)
        else:
            # Fallback to hash-based encoding
            text_emb = self._hash_encode_text(column_name)
        
        # Project to target dimension
        projected = self.text_projection(text_emb)
        return projected.squeeze(0)
    
    def _hash_encode_text(self, text: str) -> torch.Tensor:
        """Fallback hash-based text encoding."""
        # Create deterministic hash-based encoding
        hash_obj = hashlib.md5(text.encode())
        hash_bytes = hash_obj.digest()
        
        # Convert to float values
        hash_values = np.frombuffer(hash_bytes, dtype=np.uint8).astype(np.float32)
        hash_values = (hash_values - 127.5) / 127.5  # Normalize to [-1, 1]
        
        # Pad or truncate to 64 dimensions
        if len(hash_values) < 64:
            hash_values = np.pad(hash_values, (0, 64 - len(hash_values)))
        else:
            hash_values = hash_values[:64]
        
        return torch.tensor(hash_values, dtype=torch.float32).unsqueeze(0)
    
    def encode_statistical_profile(self, metadata: ColumnMetadata) -> torch.Tensor:
        """
        Encode statistical profile of the column.
        
        Args:
            metadata: Column metadata containing statistics
            
        Returns:
            Statistical embedding tensor
        """
        # Create statistical feature vector
        stats = metadata.statistical_profile
        
        if metadata.dtype == 'numerical' and stats:
            # For numerical columns, use computed statistics
            features = [
                stats.get('mean', 0.0),
                stats.get('std', 1.0),
                stats.get('min', 0.0),
                stats.get('max', 1.0),
                stats.get('skew', 0.0),
                stats.get('kurtosis', 0.0),
                metadata.missing_rate,
                np.log1p(metadata.cardinality)  # Log cardinality
            ]
        else:
            # For non-numerical columns, use basic statistics
            features = [
                0.0,  # mean (not applicable)
                0.0,  # std (not applicable)
                0.0,  # min (not applicable)
                0.0,  # max (not applicable)
                0.0,  # skew (not applicable)
                0.0,  # kurtosis (not applicable)
                metadata.missing_rate,
                np.log1p(metadata.cardinality)
            ]
        
        # Ensure we have exactly the right number of features
        features = features[:self.statistical_features]
        while len(features) < self.statistical_features:
            features.append(0.0)
        
        features_tensor = torch.tensor(features, dtype=torch.float32)
        return self.statistical_encoder(features_tensor)
    
    def encode_distribution(self, column_data: pd.Series) -> torch.Tensor:
        """
        Encode distribution characteristics of the column.
        
        Args:
            column_data: The actual column data
            
        Returns:
            Distribution embedding tensor
        """
        # Remove missing values for distribution analysis
        clean_data = column_data.dropna()
        
        if len(clean_data) == 0:
            # All missing values
            dist_features = torch.zeros(self.distribution_bins)
        elif pd.api.types.is_numeric_dtype(clean_data):
            # For numerical data, create histogram
            try:
                hist, _ = np.histogram(clean_data, bins=self.distribution_bins, density=True)
                dist_features = torch.tensor(hist, dtype=torch.float32)
            except:
                dist_features = torch.zeros(self.distribution_bins)
        else:
            # For categorical data, create frequency distribution
            value_counts = clean_data.value_counts()
            
            # Take top bins and normalize
            top_counts = value_counts.head(self.distribution_bins).values
            if len(top_counts) < self.distribution_bins:
                top_counts = np.pad(top_counts, (0, self.distribution_bins - len(top_counts)))
            
            # Normalize to probabilities
            if top_counts.sum() > 0:
                top_counts = top_counts / top_counts.sum()
            
            dist_features = torch.tensor(top_counts, dtype=torch.float32)
        
        return self.distribution_encoder(dist_features)
    
    def encode_column_type(self, dtype: str) -> torch.Tensor:
        """
        Encode column data type.
        
        Args:
            dtype: Data type string
            
        Returns:
            Type embedding tensor
        """
        type_id = self.type_vocab.get(dtype, self.type_vocab['unknown'])
        type_tensor = torch.tensor([type_id], dtype=torch.long)
        return self.type_embedding(type_tensor).squeeze(0)
    
    def forward(
        self, 
        metadata: ColumnMetadata, 
        column_data: Optional[pd.Series] = None
    ) -> ColumnEmbedding:
        """
        Generate complete column embedding.
        
        Args:
            metadata: Column metadata
            column_data: Optional actual column data for distribution encoding
            
        Returns:
            ColumnEmbedding with all components
        """
        # Encode different aspects
        name_emb = self.encode_column_name(metadata.name)
        statistical_emb = self.encode_statistical_profile(metadata)
        type_emb = self.encode_column_type(metadata.dtype)
        
        if column_data is not None:
            dist_emb = self.encode_distribution(column_data)
        else:
            # Use zero distribution if no data provided
            dist_emb = torch.zeros(self.embedding_dim // 4)
        
        # Concatenate all embeddings
        combined = torch.cat([name_emb, statistical_emb, type_emb, dist_emb], dim=0)
        
        # Ensure combined embedding has correct size for fusion layer
        if combined.size(0) != self.embedding_dim:
            # Pad or truncate to match expected size
            if combined.size(0) < self.embedding_dim:
                padding = torch.zeros(self.embedding_dim - combined.size(0))
                combined = torch.cat([combined, padding], dim=0)
            else:
                combined = combined[:self.embedding_dim]
        
        # Apply fusion layer
        fused_embedding = self.fusion_layer(combined)
        
        return ColumnEmbedding(
            name_embedding=name_emb,
            statistical_embedding=statistical_emb,
            type_embedding=type_emb,
            distribution_embedding=dist_emb,
            combined_embedding=fused_embedding,
            metadata=metadata
        )
    
    def encode_columns(
        self, 
        column_metadata: List[ColumnMetadata],
        dataframe: Optional[pd.DataFrame] = None
    ) -> List[ColumnEmbedding]:
        """
        Encode multiple columns.
        
        Args:
            column_metadata: List of column metadata
            dataframe: Optional dataframe for distribution encoding
            
        Returns:
            List of column embeddings
        """
        embeddings = []
        
        for metadata in column_metadata:
            column_data = None
            if dataframe is not None and metadata.name in dataframe.columns:
                column_data = dataframe[metadata.name]
            
            embedding = self.forward(metadata, column_data)
            embeddings.append(embedding)
        
        return embeddings
    
    def compute_column_similarity(
        self, 
        emb1: ColumnEmbedding, 
        emb2: ColumnEmbedding
    ) -> Dict[str, float]:
        """
        Compute similarity between two column embeddings.
        
        Args:
            emb1: First column embedding
            emb2: Second column embedding
            
        Returns:
            Dictionary of similarity scores
        """
        def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
            sim = torch.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()
            # Handle NaN values (can occur with zero vectors)
            if torch.isnan(torch.tensor(sim)):
                return 0.0
            # Clamp to valid range to handle floating point precision issues
            return max(-1.0, min(1.0, sim))
        
        return {
            'name_similarity': cosine_similarity(emb1.name_embedding, emb2.name_embedding),
            'statistical_similarity': cosine_similarity(emb1.statistical_embedding, emb2.statistical_embedding),
            'type_similarity': cosine_similarity(emb1.type_embedding, emb2.type_embedding),
            'distribution_similarity': cosine_similarity(emb1.distribution_embedding, emb2.distribution_embedding),
            'overall_similarity': cosine_similarity(emb1.combined_embedding, emb2.combined_embedding)
        }
    
    def find_similar_columns(
        self,
        target_embedding: ColumnEmbedding,
        candidate_embeddings: List[ColumnEmbedding],
        top_k: int = 5
    ) -> List[tuple]:
        """
        Find most similar columns to a target column.
        
        Args:
            target_embedding: Target column embedding
            candidate_embeddings: List of candidate embeddings
            top_k: Number of top matches to return
            
        Returns:
            List of (embedding, similarity_score) tuples
        """
        similarities = []
        
        for candidate in candidate_embeddings:
            sim_scores = self.compute_column_similarity(target_embedding, candidate)
            overall_sim = sim_scores['overall_similarity']
            similarities.append((candidate, overall_sim))
        
        # Sort by similarity and return top k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]