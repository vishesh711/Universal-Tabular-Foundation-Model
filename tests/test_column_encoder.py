"""Tests for column encoder."""

import pytest
import pandas as pd
import torch
import numpy as np

from tabgpt.encoders import ColumnEncoder, SemanticColumnEncoder
from tabgpt.tokenizers import TabularTokenizer


class TestColumnEncoder:
    """Test basic column encoder functionality."""
    
    def test_encoder_creation(self):
        """Test encoder initialization."""
        encoder = ColumnEncoder(embedding_dim=128)
        assert encoder.embedding_dim == 128
        assert encoder.statistical_features == 8
        assert encoder.distribution_bins == 32
    
    def test_column_name_encoding(self):
        """Test column name encoding."""
        encoder = ColumnEncoder(embedding_dim=64)
        
        # Test different column names
        names = ['user_id', 'total_amount', 'created_at', 'is_active']
        embeddings = [encoder.encode_column_name(name) for name in names]
        
        # Check shapes
        for emb in embeddings:
            assert emb.shape == (16,)  # embedding_dim // 4
        
        # Different names should produce different embeddings
        assert not torch.equal(embeddings[0], embeddings[1])
    
    def test_statistical_profile_encoding(self):
        """Test statistical profile encoding."""
        encoder = ColumnEncoder(embedding_dim=64)
        
        # Create test metadata
        from tabgpt.tokenizers.tabular_tokenizer import ColumnMetadata
        
        # Numerical column
        num_metadata = ColumnMetadata(
            name='price',
            dtype='numerical',
            unique_values=None,
            missing_rate=0.1,
            statistical_profile={
                'mean': 50.0,
                'std': 15.0,
                'min': 10.0,
                'max': 100.0,
                'skew': 0.5,
                'kurtosis': -0.2
            }
        )
        
        # Categorical column
        cat_metadata = ColumnMetadata(
            name='category',
            dtype='categorical',
            unique_values=5,
            missing_rate=0.05,
            statistical_profile=None
        )
        
        num_emb = encoder.encode_statistical_profile(num_metadata)
        cat_emb = encoder.encode_statistical_profile(cat_metadata)
        
        assert num_emb.shape == (16,)  # embedding_dim // 4
        assert cat_emb.shape == (16,)
        assert not torch.equal(num_emb, cat_emb)
    
    def test_type_encoding(self):
        """Test column type encoding."""
        encoder = ColumnEncoder(embedding_dim=64)
        
        types = ['categorical', 'numerical', 'datetime', 'boolean', 'unknown']
        embeddings = [encoder.encode_column_type(dtype) for dtype in types]
        
        # Check shapes
        for emb in embeddings:
            assert emb.shape == (16,)  # embedding_dim // 4
        
        # Different types should produce different embeddings
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                assert not torch.equal(embeddings[i], embeddings[j])
    
    def test_distribution_encoding(self):
        """Test distribution encoding."""
        encoder = ColumnEncoder(embedding_dim=64, distribution_bins=16)
        
        # Numerical data
        num_data = pd.Series(np.random.normal(0, 1, 1000))
        num_emb = encoder.encode_distribution(num_data)
        
        # Categorical data
        cat_data = pd.Series(['A'] * 300 + ['B'] * 200 + ['C'] * 100)
        cat_emb = encoder.encode_distribution(cat_data)
        
        # Empty data
        empty_data = pd.Series([np.nan] * 10)
        empty_emb = encoder.encode_distribution(empty_data)
        
        assert num_emb.shape == (16,)  # embedding_dim // 4
        assert cat_emb.shape == (16,)
        assert empty_emb.shape == (16,)
        
        # Different distributions should produce different embeddings
        assert not torch.equal(num_emb, cat_emb)
    
    def test_full_column_encoding(self):
        """Test complete column encoding."""
        encoder = ColumnEncoder(embedding_dim=128)
        
        # Create test data and metadata
        df = pd.DataFrame({
            'price': [10.5, 20.0, 15.5, 30.0, 25.5],
            'category': ['A', 'B', 'A', 'C', 'B']
        })
        
        tokenizer = TabularTokenizer()
        tokenizer.fit(df)
        
        # Encode first column (price)
        price_metadata = tokenizer.column_metadata[0]
        price_embedding = encoder.forward(price_metadata, df['price'])
        
        assert isinstance(price_embedding.name_embedding, torch.Tensor)
        assert isinstance(price_embedding.statistical_embedding, torch.Tensor)
        assert isinstance(price_embedding.type_embedding, torch.Tensor)
        assert isinstance(price_embedding.distribution_embedding, torch.Tensor)
        assert isinstance(price_embedding.combined_embedding, torch.Tensor)
        
        assert price_embedding.combined_embedding.shape == (128,)
    
    def test_multiple_columns_encoding(self):
        """Test encoding multiple columns."""
        encoder = ColumnEncoder(embedding_dim=64)
        
        df = pd.DataFrame({
            'user_id': [1, 2, 3, 4, 5],
            'amount': [100.0, 200.0, 150.0, 300.0, 250.0],
            'status': ['active', 'inactive', 'active', 'pending', 'active'],
            'created_at': pd.date_range('2023-01-01', periods=5)
        })
        
        tokenizer = TabularTokenizer()
        tokenizer.fit(df)
        
        embeddings = encoder.encode_columns(tokenizer.column_metadata, df)
        
        assert len(embeddings) == 4
        for emb in embeddings:
            assert emb.combined_embedding.shape == (64,)
    
    def test_column_similarity(self):
        """Test column similarity computation."""
        encoder = ColumnEncoder(embedding_dim=64)
        
        # Create similar columns
        df1 = pd.DataFrame({'price': [10, 20, 30, 40, 50]})
        df2 = pd.DataFrame({'cost': [15, 25, 35, 45, 55]})
        
        tokenizer1 = TabularTokenizer()
        tokenizer1.fit(df1)
        emb1 = encoder.forward(tokenizer1.column_metadata[0], df1['price'])
        
        tokenizer2 = TabularTokenizer()
        tokenizer2.fit(df2)
        emb2 = encoder.forward(tokenizer2.column_metadata[0], df2['cost'])
        
        similarity = encoder.compute_column_similarity(emb1, emb2)
        
        assert 'name_similarity' in similarity
        assert 'statistical_similarity' in similarity
        assert 'type_similarity' in similarity
        assert 'distribution_similarity' in similarity
        assert 'overall_similarity' in similarity
        
        # All similarities should be between -1 and 1
        for sim_value in similarity.values():
            assert -1 <= sim_value <= 1
    
    def test_find_similar_columns(self):
        """Test finding similar columns."""
        encoder = ColumnEncoder(embedding_dim=32)
        
        # Create multiple columns
        df = pd.DataFrame({
            'price1': [10, 20, 30],
            'price2': [15, 25, 35],
            'category': ['A', 'B', 'C'],
            'count': [1, 2, 3]
        })
        
        tokenizer = TabularTokenizer()
        tokenizer.fit(df)
        
        embeddings = encoder.encode_columns(tokenizer.column_metadata, df)
        
        # Find similar to price1
        similar = encoder.find_similar_columns(embeddings[0], embeddings[1:], top_k=2)
        
        assert len(similar) <= 2
        assert all(isinstance(item, tuple) and len(item) == 2 for item in similar)
        
        # Similarities should be in descending order
        if len(similar) > 1:
            assert similar[0][1] >= similar[1][1]


class TestSemanticColumnEncoder:
    """Test semantic column encoder functionality."""
    
    def test_semantic_encoder_creation(self):
        """Test semantic encoder initialization."""
        encoder = SemanticColumnEncoder(embedding_dim=128)
        assert encoder.embedding_dim == 128
        assert len(encoder.domain_vocabularies) > 0
        assert len(encoder.semantic_patterns) > 0
    
    def test_domain_detection(self):
        """Test domain detection from column names."""
        encoder = SemanticColumnEncoder()
        
        # Test finance domain
        assert encoder.detect_domain('total_price') == 'finance'
        assert encoder.detect_domain('account_balance') == 'finance'
        
        # Test healthcare domain
        assert encoder.detect_domain('patient_age') == 'healthcare'
        assert encoder.detect_domain('blood_pressure') == 'healthcare'
        
        # Test temporal domain
        assert encoder.detect_domain('created_date') == 'temporal'
        assert encoder.detect_domain('timestamp') == 'temporal'
        
        # Test unknown domain
        assert encoder.detect_domain('random_column') == 'unknown'
    
    def test_semantic_features_extraction(self):
        """Test semantic feature extraction."""
        encoder = SemanticColumnEncoder()
        
        # Test different naming patterns
        features1 = encoder.extract_semantic_features('user_id')
        assert features1['has_underscore'] == True
        assert features1['semantic_patterns']['id_pattern'] == True
        
        features2 = encoder.extract_semantic_features('totalAmount')
        assert features2['has_camelcase'] == True
        assert features2['semantic_patterns']['amount_pattern'] == True
        
        features3 = encoder.extract_semantic_features('created_at_2023')
        assert features3['numeric_suffix'] == True
        assert features3['semantic_patterns']['date_pattern'] == True
    
    def test_domain_encoding(self):
        """Test domain embedding encoding."""
        encoder = SemanticColumnEncoder(embedding_dim=64)
        
        domains = ['finance', 'healthcare', 'ecommerce', 'unknown']
        embeddings = [encoder.encode_domain(domain) for domain in domains]
        
        # Check shapes
        for emb in embeddings:
            assert emb.shape == (8,)  # embedding_dim // 8
        
        # Different domains should produce different embeddings
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                assert not torch.equal(embeddings[i], embeddings[j])
    
    def test_enhanced_column_encoding(self):
        """Test enhanced column encoding with semantic information."""
        encoder = SemanticColumnEncoder(embedding_dim=128)
        
        # Create test data with domain-specific columns
        df = pd.DataFrame({
            'customer_id': [1, 2, 3, 4, 5],
            'total_price': [100.0, 200.0, 150.0, 300.0, 250.0],
            'order_date': pd.date_range('2023-01-01', periods=5)
        })
        
        tokenizer = TabularTokenizer()
        tokenizer.fit(df)
        
        # Encode columns
        embeddings = encoder.encode_columns(tokenizer.column_metadata, df)
        
        # Check that semantic information is added
        for emb in embeddings:
            assert hasattr(emb, 'domain')
            assert hasattr(emb, 'semantic_features')
            assert hasattr(emb, 'domain_embedding')
            assert emb.combined_embedding.shape == (128,)
    
    def test_semantic_similarity(self):
        """Test semantic similarity computation."""
        encoder = SemanticColumnEncoder(embedding_dim=64)
        
        # Create columns with similar semantics
        df1 = pd.DataFrame({'user_id': [1, 2, 3]})
        df2 = pd.DataFrame({'customer_id': [4, 5, 6]})
        
        tokenizer1 = TabularTokenizer()
        tokenizer1.fit(df1)
        emb1 = encoder.forward(tokenizer1.column_metadata[0], df1['user_id'])
        
        tokenizer2 = TabularTokenizer()
        tokenizer2.fit(df2)
        emb2 = encoder.forward(tokenizer2.column_metadata[0], df2['customer_id'])
        
        similarity = encoder.compute_semantic_similarity(emb1, emb2)
        
        assert 'domain_similarity' in similarity
        assert 'semantic_pattern_similarity' in similarity
        assert 'enhanced_overall_similarity' in similarity
        
        # Should detect similar ID patterns
        assert similarity['semantic_pattern_similarity'] > 0
    
    def test_dataset_schema_analysis(self):
        """Test dataset schema analysis."""
        encoder = SemanticColumnEncoder()
        
        # Create diverse dataset
        df = pd.DataFrame({
            'customer_id': [1, 2, 3, 4, 5],
            'total_amount': [100.0, 200.0, 150.0, 300.0, 250.0],
            'order_date': pd.date_range('2023-01-01', periods=5),
            'product_category': ['A', 'B', 'A', 'C', 'B'],
            'is_premium': [True, False, True, False, True]
        })
        
        analysis = encoder.analyze_dataset_schema(df)
        
        assert 'column_embeddings' in analysis
        assert 'domain_distribution' in analysis
        assert 'type_distribution' in analysis
        assert 'schema_complexity' in analysis
        assert 'primary_domain' in analysis
        
        assert len(analysis['column_embeddings']) == 5
        assert analysis['schema_complexity']['num_columns'] == 5
        
        # Should detect ecommerce as primary domain
        assert 'ecommerce' in analysis['domain_distribution']