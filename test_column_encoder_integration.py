"""Integration test for column encoder functionality."""

import pandas as pd
import numpy as np
import torch
from tabgpt.encoders import ColumnEncoder, SemanticColumnEncoder
from tabgpt.tokenizers import TabularTokenizer

def test_column_encoder_integration():
    """Test complete column encoder pipeline."""
    
    # Create comprehensive test dataset
    df = pd.DataFrame({
        'customer_id': [1001, 1002, 1003, 1004, 1005],
        'total_amount': [99.99, 149.50, 75.25, 200.00, 125.75],
        'order_date': pd.date_range('2023-01-01', periods=5, freq='D'),
        'product_category': ['Electronics', 'Clothing', 'Electronics', 'Books', 'Clothing'],
        'is_premium_customer': [True, False, True, False, True],
        'customer_age': [25, 34, 28, 45, 31],
        'shipping_address': ['123 Main St', '456 Oak Ave', '789 Pine Rd', '321 Elm St', '654 Maple Dr']
    })
    
    print("Test Dataset:")
    print(df)
    print(f"Shape: {df.shape}")
    
    # Initialize tokenizer to get metadata
    tokenizer = TabularTokenizer()
    tokenizer.fit(df)
    
    print(f"\nColumn Metadata:")
    for meta in tokenizer.column_metadata:
        print(f"  {meta.name}: {meta.dtype}, cardinality={meta.cardinality}")
    
    # Test basic column encoder
    print("\n=== Testing Basic Column Encoder ===")
    basic_encoder = ColumnEncoder(embedding_dim=128)
    
    # Encode all columns
    basic_embeddings = basic_encoder.encode_columns(tokenizer.column_metadata, df)
    
    print(f"Encoded {len(basic_embeddings)} columns")
    for i, emb in enumerate(basic_embeddings):
        print(f"  Column '{emb.metadata.name}': embedding shape {emb.combined_embedding.shape}")
    
    # Test column similarity
    print("\nColumn Similarities (Basic Encoder):")
    for i in range(len(basic_embeddings)):
        for j in range(i + 1, len(basic_embeddings)):
            sim = basic_encoder.compute_column_similarity(basic_embeddings[i], basic_embeddings[j])
            print(f"  {basic_embeddings[i].metadata.name} vs {basic_embeddings[j].metadata.name}: {sim['overall_similarity']:.3f}")
    
    # Test semantic column encoder
    print("\n=== Testing Semantic Column Encoder ===")
    semantic_encoder = SemanticColumnEncoder(embedding_dim=128)
    
    # Encode all columns with semantic information
    semantic_embeddings = semantic_encoder.encode_columns(tokenizer.column_metadata, df)
    
    print(f"Encoded {len(semantic_embeddings)} columns with semantic info")
    for emb in semantic_embeddings:
        domain = getattr(emb, 'domain', 'unknown')
        print(f"  Column '{emb.metadata.name}': domain='{domain}', embedding shape {emb.combined_embedding.shape}")
    
    # Test semantic similarities
    print("\nSemantic Similarities:")
    for i in range(len(semantic_embeddings)):
        for j in range(i + 1, len(semantic_embeddings)):
            sim = semantic_encoder.compute_semantic_similarity(semantic_embeddings[i], semantic_embeddings[j])
            print(f"  {semantic_embeddings[i].metadata.name} vs {semantic_embeddings[j].metadata.name}:")
            print(f"    Overall: {sim['overall_similarity']:.3f}, Enhanced: {sim['enhanced_overall_similarity']:.3f}")
            print(f"    Domain: {sim['domain_similarity']:.3f}, Semantic: {sim['semantic_pattern_similarity']:.3f}")
    
    # Test dataset schema analysis
    print("\n=== Dataset Schema Analysis ===")
    schema_analysis = semantic_encoder.analyze_dataset_schema(df)
    
    print(f"Schema Complexity:")
    for key, value in schema_analysis['schema_complexity'].items():
        print(f"  {key}: {value}")
    
    print(f"\nDomain Distribution:")
    for domain, count in schema_analysis['domain_distribution'].items():
        print(f"  {domain}: {count}")
    
    print(f"\nType Distribution:")
    for dtype, count in schema_analysis['type_distribution'].items():
        print(f"  {dtype}: {count}")
    
    print(f"\nPrimary Domain: {schema_analysis['primary_domain']}")
    
    # Test cross-dataset similarity
    print("\n=== Testing Cross-Dataset Transfer ===")
    
    # Create similar dataset with different column names
    df2 = pd.DataFrame({
        'user_id': [2001, 2002, 2003],
        'purchase_amount': [89.99, 159.50, 95.25],
        'transaction_date': pd.date_range('2023-06-01', periods=3, freq='D'),
        'item_category': ['Tech', 'Fashion', 'Tech'],
        'vip_status': [False, True, False]
    })
    
    tokenizer2 = TabularTokenizer()
    tokenizer2.fit(df2)
    
    embeddings2 = semantic_encoder.encode_columns(tokenizer2.column_metadata, df2)
    
    print("Cross-dataset column similarities:")
    print("Dataset 1 -> Dataset 2 matches:")
    
    for emb1 in semantic_embeddings:
        similar_cols = semantic_encoder.find_similar_columns(emb1, embeddings2, top_k=2)
        if similar_cols:
            best_match = similar_cols[0]
            print(f"  '{emb1.metadata.name}' -> '{best_match[0].metadata.name}' (similarity: {best_match[1]:.3f})")
    
    # Verify no NaN values in embeddings
    for emb in semantic_embeddings:
        assert not torch.isnan(emb.combined_embedding).any(), f"NaN found in {emb.metadata.name} embedding"
    
    print("\n✅ All column encoder tests passed!")
    
    return semantic_embeddings, schema_analysis

if __name__ == "__main__":
    test_column_encoder_integration()