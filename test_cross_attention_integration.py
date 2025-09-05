"""Integration test for cross-attention fusion with tabular data."""

import pandas as pd
import numpy as np
import torch
from tabgpt.tokenizers import TabularTokenizer
from tabgpt.encoders import ColumnEncoder
from tabgpt.models.row_encoder import RowEncoder
from tabgpt.models.cross_attention import CrossAttentionFusion

def test_cross_attention_integration():
    """Test cross-attention fusion with complete tabular pipeline."""
    
    # Create comprehensive test dataset
    np.random.seed(42)
    torch.manual_seed(42)
    
    df = pd.DataFrame({
        'customer_id': [1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008],
        'age': [25, 34, 28, 45, 31, 38, 52, 29],
        'annual_income': [50000, 75000, 60000, 90000, 55000, 80000, 120000, 65000],
        'credit_score': [720, 680, 750, 800, 690, 770, 820, 740],
        'account_balance': [5000.50, 12000.75, 8500.25, 15000.00, 6500.80, 11000.40, 25000.00, 9500.30],
        'product_category': ['Premium', 'Standard', 'Premium', 'Premium', 'Standard', 'Premium', 'VIP', 'Standard'],
        'is_active': [True, False, True, True, False, True, True, True],
        'signup_date': pd.date_range('2023-01-01', periods=8, freq='M'),
        'transaction_count': [45, 23, 67, 89, 34, 56, 123, 78],
        'risk_score': [0.2, 0.6, 0.1, 0.05, 0.7, 0.15, 0.03, 0.25]
    })
    
    print("Test Dataset:")
    print(df)
    print(f"Shape: {df.shape}")
    
    # Step 1: Tokenize the data (row-level representations)
    print("\n=== Step 1: Tokenizing Data (Row Representations) ===")
    tokenizer = TabularTokenizer(embedding_dim=128)
    tokenized = tokenizer.fit_transform(df)
    
    print(f"Tokenized shape: {tokenized.tokens.shape}")
    print(f"Attention mask shape: {tokenized.attention_mask.shape}")
    print(f"Features: {tokenized.feature_names}")
    
    # Step 2: Generate column embeddings (column-level representations)
    print("\n=== Step 2: Generating Column Embeddings ===")
    column_encoder = ColumnEncoder(embedding_dim=128)
    column_embeddings_list = column_encoder.encode_columns(tokenizer.column_metadata, df)
    
    # Convert list of embeddings to tensor
    column_embeddings = torch.stack([emb.combined_embedding for emb in column_embeddings_list])
    
    print(f"Column embeddings shape: {column_embeddings.shape}")
    print(f"Column embedding sample: {column_embeddings[0, :5].tolist()}")
    
    # Step 3: Process through row encoder
    print("\n=== Step 3: Processing Through Row Encoder ===")
    from tabgpt.config import TabGPTConfig
    
    config = TabGPTConfig(
        d_model=128,
        n_heads=8,
        n_layers=3,
        dropout=0.1,
        embedding_dim=128,
        max_features=20
    )
    
    row_encoder = RowEncoder(config)
    
    with torch.no_grad():
        row_outputs = row_encoder(
            feature_embeddings=tokenized.tokens,
            attention_mask=tokenized.attention_mask
        )
    
    row_embeddings = row_outputs['last_hidden_state']
    print(f"Row embeddings shape: {row_embeddings.shape}")
    
    # Step 4: Apply cross-attention fusion
    print("\n=== Step 4: Applying Cross-Attention Fusion ===")
    
    # Test different fusion strategies
    fusion_strategies = ['add', 'concat', 'gate']
    fusion_results = {}
    
    for strategy in fusion_strategies:
        print(f"\n--- Testing {strategy.upper()} fusion strategy ---")
        
        cross_attention = CrossAttentionFusion(
            d_model=128,
            n_heads=8,
            n_layers=2,
            dropout=0.1,
            fusion_strategy=strategy,
            use_residual=True
        )
        
        with torch.no_grad():
            fusion_outputs = cross_attention(
                row_embeddings=row_embeddings,
                column_embeddings=column_embeddings,
                row_attention_mask=tokenized.attention_mask,
                return_attention_weights=True
            )
        
        fusion_results[strategy] = fusion_outputs
        
        print(f"Fused representations shape: {fusion_outputs['fused_representations'].shape}")
        print(f"Enhanced row embeddings shape: {fusion_outputs['enhanced_row_embeddings'].shape}")
        print(f"Enhanced column embeddings shape: {fusion_outputs['enhanced_column_embeddings'].shape}")
        
        # Check for NaN values
        assert not torch.isnan(fusion_outputs['fused_representations']).any()
        print(f"✓ No NaN values in {strategy} fusion")
    
    # Step 5: Analyze attention patterns
    print("\n=== Step 5: Analyzing Attention Patterns ===")
    
    # Use the 'gate' strategy for detailed analysis
    fusion_gate = CrossAttentionFusion(
        d_model=128,
        n_heads=8,
        n_layers=2,
        fusion_strategy='gate'
    )
    
    attention_patterns = fusion_gate.get_attention_patterns(
        row_embeddings=row_embeddings,
        column_embeddings=column_embeddings,
        row_attention_mask=tokenized.attention_mask
    )
    
    row_to_col_attention = attention_patterns['row_to_column_attention']
    col_to_row_attention = attention_patterns['column_to_row_attention']
    
    print(f"Row-to-column attention shape: {row_to_col_attention.shape}")
    print(f"Column-to-row attention shape: {col_to_row_attention.shape}")
    
    # Analyze attention patterns for first sample
    print("\n--- Attention Analysis for First Sample ---")
    sample_row_to_col = row_to_col_attention[0].detach().numpy()
    sample_col_to_row = col_to_row_attention[0].detach().numpy()
    
    print("Top row-to-column attention weights (> 0.15):")
    for i, source_feat in enumerate(tokenized.feature_names):
        for j, target_feat in enumerate(tokenized.feature_names):
            if sample_row_to_col[i, j] > 0.15:
                print(f"  {source_feat} -> {target_feat}: {sample_row_to_col[i, j]:.3f}")
    
    print("\nTop column-to-row attention weights (> 0.15):")
    for i, source_feat in enumerate(tokenized.feature_names):
        for j, target_feat in enumerate(tokenized.feature_names):
            if sample_col_to_row[i, j] > 0.15:
                print(f"  {source_feat} -> {target_feat}: {sample_col_to_row[i, j]:.3f}")
    
    # Step 6: Compute feature interaction scores
    print("\n=== Step 6: Computing Feature Interaction Scores ===")
    
    interaction_scores = fusion_gate.compute_interaction_scores(
        row_embeddings=row_embeddings,
        column_embeddings=column_embeddings,
        row_attention_mask=tokenized.attention_mask
    )
    
    interaction_matrix = interaction_scores['interaction_matrix']
    feature_importance = interaction_scores['feature_importance']
    
    print(f"Interaction matrix shape: {interaction_matrix.shape}")
    print(f"Feature importance shape: {feature_importance.shape}")
    
    # Show feature importance scores (averaged across samples)
    avg_importance = feature_importance.mean(dim=0)
    print("\nFeature importance scores (average across samples):")
    for i, (feature_name, importance) in enumerate(zip(tokenized.feature_names, avg_importance)):
        print(f"  {feature_name}: {importance:.4f}")
    
    # Find top feature interactions
    print("\nTop feature interactions (average across samples):")
    avg_interaction = interaction_matrix.mean(dim=0).detach().numpy()
    
    # Get top interactions (excluding self-interactions)
    interaction_pairs = []
    for i in range(len(tokenized.feature_names)):
        for j in range(i+1, len(tokenized.feature_names)):
            interaction_strength = avg_interaction[i, j]
            interaction_pairs.append((
                tokenized.feature_names[i],
                tokenized.feature_names[j],
                interaction_strength
            ))
    
    # Sort by interaction strength
    interaction_pairs.sort(key=lambda x: x[2], reverse=True)
    
    print("Top 5 feature interactions:")
    for feat1, feat2, strength in interaction_pairs[:5]:
        print(f"  {feat1} <-> {feat2}: {strength:.4f}")
    
    # Step 7: Compare fusion strategies
    print("\n=== Step 7: Comparing Fusion Strategies ===")
    
    # Compute similarities between different fusion strategies
    strategies = list(fusion_results.keys())
    
    for i, strategy1 in enumerate(strategies):
        for j, strategy2 in enumerate(strategies[i+1:], i+1):
            repr1 = fusion_results[strategy1]['fused_representations']
            repr2 = fusion_results[strategy2]['fused_representations']
            
            # Compute cosine similarity
            similarity = torch.cosine_similarity(
                repr1.flatten(1), repr2.flatten(1), dim=1
            ).mean()
            
            print(f"Similarity between {strategy1} and {strategy2}: {similarity:.4f}")
    
    # Step 8: Test with missing values
    print("\n=== Step 8: Testing with Missing Values ===")
    
    # Create dataset with missing values
    df_missing = df.copy()
    df_missing.loc[1, 'annual_income'] = np.nan
    df_missing.loc[3, 'credit_score'] = np.nan
    df_missing.loc[5, 'account_balance'] = np.nan
    
    # Tokenize with missing values
    tokenizer_missing = TabularTokenizer(embedding_dim=128)
    tokenized_missing = tokenizer_missing.fit_transform(df_missing)
    
    # Generate column embeddings for missing data
    column_embeddings_missing_list = column_encoder.encode_columns(tokenizer_missing.column_metadata, df_missing)
    column_embeddings_missing = torch.stack([emb.combined_embedding for emb in column_embeddings_missing_list])
    
    # Process through row encoder
    with torch.no_grad():
        row_outputs_missing = row_encoder(
            feature_embeddings=tokenized_missing.tokens,
            attention_mask=tokenized_missing.attention_mask
        )
    
    # Apply cross-attention fusion
    with torch.no_grad():
        fusion_outputs_missing = fusion_gate(
            row_embeddings=row_outputs_missing['last_hidden_state'],
            column_embeddings=column_embeddings_missing,
            row_attention_mask=tokenized_missing.attention_mask
        )
    
    print(f"Fusion with missing values - output shape: {fusion_outputs_missing['fused_representations'].shape}")
    print(f"No NaN in outputs: {not torch.isnan(fusion_outputs_missing['fused_representations']).any()}")
    
    # Step 9: Performance analysis
    print("\n=== Step 9: Performance Analysis ===")
    
    import time
    
    # Measure inference time
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(10):
            _ = fusion_gate(
                row_embeddings=row_embeddings,
                column_embeddings=column_embeddings,
                row_attention_mask=tokenized.attention_mask
            )
    
    end_time = time.time()
    avg_time = (end_time - start_time) / 10
    
    print(f"Average inference time: {avg_time:.4f} seconds")
    print(f"Throughput: {len(df) / avg_time:.1f} samples/second")
    
    # Memory usage
    total_params = sum(p.numel() for p in fusion_gate.parameters())
    print(f"Cross-attention fusion parameters: {total_params:,}")
    
    # Verify all outputs are valid
    for strategy, outputs in fusion_results.items():
        assert not torch.isnan(outputs['fused_representations']).any()
        assert not torch.isnan(outputs['enhanced_row_embeddings']).any()
        assert not torch.isnan(outputs['enhanced_column_embeddings']).any()
    
    assert not torch.isnan(interaction_matrix).any()
    assert not torch.isnan(feature_importance).any()
    
    print("\n✅ All cross-attention fusion tests passed!")
    
    return {
        'tokenized_data': tokenized,
        'column_embeddings': column_embeddings,
        'row_embeddings': row_embeddings,
        'fusion_results': fusion_results,
        'attention_patterns': attention_patterns,
        'interaction_scores': interaction_scores
    }

if __name__ == "__main__":
    results = test_cross_attention_integration()