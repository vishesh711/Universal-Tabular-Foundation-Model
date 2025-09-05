"""Semantic column encoder with advanced NLP capabilities."""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Set
import re
from collections import Counter

from .column_encoder import ColumnEncoder, ColumnEmbedding
from ..tokenizers.tabular_tokenizer import ColumnMetadata


class SemanticColumnEncoder(ColumnEncoder):
    """
    Advanced column encoder with semantic understanding capabilities.
    
    This encoder extends the base ColumnEncoder with:
    - Domain-specific vocabulary recognition
    - Column name parsing and semantic analysis
    - Cross-domain knowledge transfer
    """
    
    def __init__(
        self,
        embedding_dim: int = 128,
        statistical_features: int = 8,
        distribution_bins: int = 32,
        text_encoder_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        super().__init__(embedding_dim, statistical_features, distribution_bins, text_encoder_model)
        
        # Domain-specific vocabularies
        self.domain_vocabularies = self._init_domain_vocabularies()
        
        # Semantic pattern recognition
        self.semantic_patterns = self._init_semantic_patterns()
        
        # Domain embedding layer
        self.domain_embedding = nn.Embedding(
            len(self.domain_vocabularies) + 1,  # +1 for unknown domain
            embedding_dim // 8
        )
        
        # Update fusion layer to include domain information
        self.semantic_fusion_layer = nn.Sequential(
            nn.Linear(embedding_dim + embedding_dim // 8, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
        # Re-initialize weights
        self._init_weights()
    
    def _init_domain_vocabularies(self) -> Dict[str, Set[str]]:
        """Initialize domain-specific vocabularies."""
        return {
            'finance': {
                'price', 'cost', 'amount', 'revenue', 'profit', 'loss', 'balance',
                'account', 'transaction', 'payment', 'invoice', 'budget', 'expense',
                'income', 'salary', 'wage', 'tax', 'fee', 'rate', 'interest',
                'loan', 'debt', 'credit', 'debit', 'currency', 'dollar', 'euro'
            },
            'healthcare': {
                'patient', 'diagnosis', 'treatment', 'medication', 'dose', 'symptom',
                'disease', 'condition', 'therapy', 'hospital', 'clinic', 'doctor',
                'nurse', 'medical', 'health', 'blood', 'pressure', 'temperature',
                'heart', 'pulse', 'weight', 'height', 'bmi', 'age', 'gender'
            },
            'ecommerce': {
                'product', 'item', 'sku', 'category', 'brand', 'model', 'size',
                'color', 'quantity', 'stock', 'inventory', 'order', 'customer',
                'user', 'rating', 'review', 'cart', 'checkout', 'shipping',
                'delivery', 'return', 'refund', 'discount', 'coupon', 'promotion'
            },
            'temporal': {
                'date', 'time', 'timestamp', 'year', 'month', 'day', 'hour',
                'minute', 'second', 'created', 'updated', 'modified', 'start',
                'end', 'duration', 'period', 'interval', 'schedule', 'deadline',
                'expiry', 'birth', 'anniversary', 'event', 'occurrence'
            },
            'geographic': {
                'address', 'street', 'city', 'state', 'country', 'zip', 'postal',
                'location', 'latitude', 'longitude', 'coordinate', 'region',
                'area', 'district', 'neighborhood', 'place', 'venue', 'site'
            },
            'identity': {
                'id', 'identifier', 'key', 'code', 'number', 'serial', 'reference',
                'name', 'title', 'first', 'last', 'middle', 'full', 'username',
                'email', 'phone', 'contact', 'profile', 'account', 'user'
            }
        }
    
    def _init_semantic_patterns(self) -> Dict[str, str]:
        """Initialize regex patterns for semantic recognition."""
        return {
            'id_pattern': r'.*\b(id|identifier|key|code|number|serial|ref)\b.*',
            'name_pattern': r'.*\b(name|title|label|description)\b.*',
            'date_pattern': r'.*\b(date|time|timestamp|created|updated|modified)\b.*',
            'amount_pattern': r'.*\b(amount|price|cost|value|total|sum|balance)\b.*',
            'count_pattern': r'.*\b(count|quantity|number|num|qty|size|length)\b.*',
            'rate_pattern': r'.*\b(rate|ratio|percentage|percent|score|rating)\b.*',
            'status_pattern': r'.*\b(status|state|condition|flag|active|enabled)\b.*',
            'category_pattern': r'.*\b(category|type|class|group|kind|genre)\b.*'
        }
    
    def detect_domain(self, column_name: str, column_data: Optional[pd.Series] = None) -> str:
        """
        Detect the domain of a column based on name and data.
        
        Args:
            column_name: Name of the column
            column_data: Optional column data for analysis
            
        Returns:
            Detected domain name
        """
        name_lower = column_name.lower()
        
        # Score each domain based on vocabulary matches
        domain_scores = {}
        
        for domain, vocab in self.domain_vocabularies.items():
            score = 0
            for word in vocab:
                if word in name_lower:
                    score += 1
            
            # Normalize by vocabulary size
            domain_scores[domain] = score / len(vocab)
        
        # Additional scoring based on data patterns if available
        if column_data is not None:
            domain_scores = self._score_domains_by_data(domain_scores, column_data)
        
        # Return domain with highest score, or 'unknown' if all scores are 0
        if max(domain_scores.values()) > 0:
            return max(domain_scores, key=domain_scores.get)
        else:
            return 'unknown'
    
    def _score_domains_by_data(self, domain_scores: Dict[str, float], column_data: pd.Series) -> Dict[str, float]:
        """Score domains based on actual data patterns."""
        # Sample some data for analysis
        sample_data = column_data.dropna().head(100)
        
        if len(sample_data) == 0:
            return domain_scores
        
        # Convert to string for pattern matching
        str_data = sample_data.astype(str)
        
        # Check for email patterns (identity domain)
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        email_matches = str_data.str.match(email_pattern).sum()
        if email_matches > len(sample_data) * 0.5:
            domain_scores['identity'] += 0.5
        
        # Check for currency patterns (finance domain)
        currency_pattern = r'^\$?\d+\.?\d*$'
        currency_matches = str_data.str.match(currency_pattern).sum()
        if currency_matches > len(sample_data) * 0.3:
            domain_scores['finance'] += 0.3
        
        # Check for date patterns (temporal domain)
        try:
            pd.to_datetime(sample_data, errors='coerce')
            date_matches = pd.to_datetime(sample_data, errors='coerce').notna().sum()
            if date_matches > len(sample_data) * 0.5:
                domain_scores['temporal'] += 0.4
        except:
            pass
        
        return domain_scores
    
    def extract_semantic_features(self, column_name: str) -> Dict[str, Any]:
        """
        Extract semantic features from column name.
        
        Args:
            column_name: Name of the column
            
        Returns:
            Dictionary of semantic features
        """
        name_lower = column_name.lower()
        
        features = {
            'has_underscore': '_' in column_name,
            'has_camelcase': any(c.isupper() for c in column_name[1:]),
            'word_count': len(re.findall(r'\w+', column_name)),
            'numeric_suffix': bool(re.search(r'\d+$', column_name)),
            'is_abbreviation': len(column_name) <= 5 and column_name.isupper(),
            'semantic_patterns': {}
        }
        
        # Check semantic patterns
        for pattern_name, pattern in self.semantic_patterns.items():
            features['semantic_patterns'][pattern_name] = bool(re.match(pattern, name_lower))
        
        return features
    
    def encode_domain(self, domain: str) -> torch.Tensor:
        """
        Encode domain information.
        
        Args:
            domain: Domain name
            
        Returns:
            Domain embedding tensor
        """
        domain_vocab = list(self.domain_vocabularies.keys()) + ['unknown']
        domain_id = domain_vocab.index(domain) if domain in domain_vocab else len(domain_vocab) - 1
        
        domain_tensor = torch.tensor([domain_id], dtype=torch.long)
        return self.domain_embedding(domain_tensor).squeeze(0)
    
    def forward(
        self, 
        metadata: ColumnMetadata, 
        column_data: Optional[pd.Series] = None
    ) -> ColumnEmbedding:
        """
        Generate enhanced column embedding with semantic information.
        
        Args:
            metadata: Column metadata
            column_data: Optional actual column data
            
        Returns:
            Enhanced ColumnEmbedding
        """
        # Get base embeddings
        base_embedding = super().forward(metadata, column_data)
        
        # Detect domain
        domain = self.detect_domain(metadata.name, column_data)
        domain_emb = self.encode_domain(domain)
        
        # Extract semantic features
        semantic_features = self.extract_semantic_features(metadata.name)
        
        # Combine with domain information
        enhanced_combined = torch.cat([base_embedding.combined_embedding, domain_emb], dim=0)
        
        # Ensure correct size for fusion layer
        expected_size = self.embedding_dim + self.embedding_dim // 8
        if enhanced_combined.size(0) != expected_size:
            if enhanced_combined.size(0) < expected_size:
                padding = torch.zeros(expected_size - enhanced_combined.size(0))
                enhanced_combined = torch.cat([enhanced_combined, padding], dim=0)
            else:
                enhanced_combined = enhanced_combined[:expected_size]
        
        # Apply enhanced fusion
        final_embedding = self.semantic_fusion_layer(enhanced_combined)
        
        # Create enhanced embedding object
        enhanced_embedding = ColumnEmbedding(
            name_embedding=base_embedding.name_embedding,
            statistical_embedding=base_embedding.statistical_embedding,
            type_embedding=base_embedding.type_embedding,
            distribution_embedding=base_embedding.distribution_embedding,
            combined_embedding=final_embedding,
            metadata=metadata
        )
        
        # Add semantic information as attributes
        enhanced_embedding.domain = domain
        enhanced_embedding.semantic_features = semantic_features
        enhanced_embedding.domain_embedding = domain_emb
        
        return enhanced_embedding
    
    def compute_semantic_similarity(
        self,
        emb1: ColumnEmbedding,
        emb2: ColumnEmbedding
    ) -> Dict[str, float]:
        """
        Compute semantic similarity between columns.
        
        Args:
            emb1: First column embedding
            emb2: Second column embedding
            
        Returns:
            Enhanced similarity scores
        """
        # Get base similarity
        base_similarity = self.compute_column_similarity(emb1, emb2)
        
        # Add domain similarity
        domain_match = 1.0 if getattr(emb1, 'domain', None) == getattr(emb2, 'domain', None) else 0.0
        
        # Add semantic pattern similarity
        semantic_sim = 0.0
        if hasattr(emb1, 'semantic_features') and hasattr(emb2, 'semantic_features'):
            patterns1 = emb1.semantic_features.get('semantic_patterns', {})
            patterns2 = emb2.semantic_features.get('semantic_patterns', {})
            
            matching_patterns = sum(1 for k in patterns1 if patterns1.get(k) == patterns2.get(k))
            total_patterns = len(patterns1)
            semantic_sim = matching_patterns / total_patterns if total_patterns > 0 else 0.0
        
        # Enhanced similarity scores
        enhanced_similarity = base_similarity.copy()
        enhanced_similarity.update({
            'domain_similarity': domain_match,
            'semantic_pattern_similarity': semantic_sim,
            'enhanced_overall_similarity': (
                base_similarity['overall_similarity'] * 0.6 +
                domain_match * 0.3 +
                semantic_sim * 0.1
            )
        })
        
        return enhanced_similarity
    
    def analyze_dataset_schema(self, dataframe: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze the overall schema of a dataset.
        
        Args:
            dataframe: Input dataframe
            
        Returns:
            Schema analysis results
        """
        # Encode all columns
        from ..tokenizers import TabularTokenizer
        tokenizer = TabularTokenizer()
        tokenizer.fit(dataframe)
        
        column_embeddings = self.encode_columns(tokenizer.column_metadata, dataframe)
        
        # Analyze domains
        domains = [getattr(emb, 'domain', 'unknown') for emb in column_embeddings]
        domain_distribution = Counter(domains)
        
        # Analyze data types
        type_distribution = Counter([emb.metadata.dtype for emb in column_embeddings])
        
        # Compute schema complexity
        schema_complexity = {
            'num_columns': len(column_embeddings),
            'num_domains': len(set(domains)),
            'num_types': len(set(type_distribution.keys())),
            'avg_cardinality': np.mean([emb.metadata.cardinality for emb in column_embeddings]),
            'avg_missing_rate': np.mean([emb.metadata.missing_rate for emb in column_embeddings])
        }
        
        return {
            'column_embeddings': column_embeddings,
            'domain_distribution': dict(domain_distribution),
            'type_distribution': dict(type_distribution),
            'schema_complexity': schema_complexity,
            'primary_domain': domain_distribution.most_common(1)[0][0] if domain_distribution else 'unknown'
        }