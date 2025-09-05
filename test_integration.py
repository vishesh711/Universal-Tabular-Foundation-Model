"""Integration test for the complete feature tokenizer."""

import pandas as pd
import numpy as np
import torch
from tabgpt.tokenizers import TabularTokenizer
from tabgpt.tokenizers.masking import RandomCellMasking, ContrastiveAugmentation

def test_complete_pipeline():
    """Test the complete tokenization and masking pipeline."""
    
    # Create comprehensive test data
    df = pd.DataFrame({
        'categorical': ['red', 'blue', 'green', 'red', 'blue', 'yellow'],
        'numerical': [1.5, 2.7, 3.9, 4.1, 5.3, 6.8],
        'datetime': pd.date_range('2023-01-01', periods=6, freq='D'),
        'boolean': [True, False, True, False, True, False],
        'with_missing': [1.0, np.nan, 3.0, None, 5.0, 6.0],
        'high_cardinality': [f'item_{i}' for i in range(6)]
    })
    
    print("Original DataFrame:")
    print(df)
    print(f"Shape: {df.shape}")
    
    # Initialize tokenizer
    tokenizer = TabularTokenizer(embedding_dim=64, vocab_size=1000)
    
    # Fit and transform
    print("\nFitting tokenizer...")
    tokenized = tokenizer.fit_transform(df)
    
    print(f"Tokenized shape: {tokenized.tokens.shape}")
    print(f"Attention mask shape: {tokenized.attention_mask.shape}")
    print(f"Number of features: {len(tokenized.feature_names)}")
    
    # Check column metadata
    print("\nColumn Metadata:")
    for meta in tokenized.column_metadata:
        print(f"  {meta.name}: {meta.dtype}, cardinality={meta.cardinality}, missing_rate={meta.missing_rate:.2f}")
        if meta.statistical_profile:
            print(f"    Stats: {meta.statistical_profile}")
    
    # Test masking
    print("\nTesting masking...")
    masker = RandomCellMasking(mask_probability=0.3)
    masked_tokens, mask_positions, original_tokens = masker.apply_mask(
        tokenized.tokens, tokenized.attention_mask
    )
    
    print(f"Mask positions shape: {mask_positions.shape}")
    print(f"Number of masked positions: {mask_positions.sum().item()}")
    print(f"Mask percentage: {mask_positions.float().mean().item():.2%}")
    
    # Test contrastive augmentation
    print("\nTesting contrastive augmentation...")
    augmenter = ContrastiveAugmentation()
    anchor, positive = augmenter.create_positive_pairs(
        tokenized.tokens, tokenized.attention_mask
    )
    
    print(f"Anchor shape: {anchor.shape}")
    print(f"Positive shape: {positive.shape}")
    
    # Verify no NaN values
    assert not torch.isnan(tokenized.tokens).any(), "Found NaN in tokenized output"
    assert not torch.isnan(masked_tokens).any(), "Found NaN in masked tokens"
    assert not torch.isnan(anchor).any(), "Found NaN in anchor"
    assert not torch.isnan(positive).any(), "Found NaN in positive"
    
    print("\n✅ All tests passed! Feature tokenizer is working correctly.")
    
    return tokenized

if __name__ == "__main__":
    test_complete_pipeline()