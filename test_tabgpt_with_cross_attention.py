"""Integration test for complete TabGPT pipeline with cross-attention fusion."""

import pandas as pd
import numpy as np
import torch
from tabgpt import TabGPTModel, TabGPTForClassification, TabularTokenizer, TabGPTConfig

def test_tabgpt_with_cross_attention():
    """Test complete TabGPT model with cross-attention fusion."""
    
    # Create comprehensive test dataset
    np.random.seed(42)
    torch.manual_seed(42)
    
    df = pd.DataFrame({
        'customer_id': [1001, 1002, 1003, 1004, 1005, 1006],
        'age': [25, 34, 28, 45, 31, 38],
        'annual_income': [50000, 75000, 60000, 90000, 55000, 80000],
        'credit_score': [720, 680, 750, 800, 690, 770],
        'account_balance': [5000.50, 12000.75, 8500.25, 15000.00, 6500.80, 11000.40],
        'product_category': ['Premium', 'Standard', 'Premium', 'Premium', 'Standard', 'Premium'],
        'is_active': [True, False, True, True, False, True],
        'transaction_count': [45, 23, 67, 89, 34, 56],
        'risk_score': [0.2, 0.6, 0.1, 0.05, 0.7, 0.15]
    })
    
    print("Test Dataset:")
    print(df)
    print(f"Shape: {df.shape}")
    
    # Step 1: Tokenize data
    print("\n=== Step 1: Tokenizing Data ===")
    tokenizer = TabularTokenizer(embedding_dim=128)
    tokenized = tokenizer.fit_transform(df)
    
    print(f"Tokenized shape: {tokenized.tokens.shape}")
    print(f"Attention mask shape: {tokenized.attention_mask.shape}")
    print(f"Features: {tokenized.feature_names}")
    
    # Step 2: Test TabGPT model with cross-attention
    print("\n=== Step 2: Testing TabGPT Model with Cross-Attention ===")
    config = TabGPTConfig(
        d_model=128,
        n_heads=8,
        n_layers=3,
        dropout=0.1,
        max_features=20,
        embedding_dim=128,
        column_embedding_dim=128,
        cross_attention_layers=2,
        fusion_strategy='gate'
    )
    
    model = TabGPTModel(config)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test without column metadata (row encoder only)
    print("\n--- Testing without column metadata ---")
    with torch.no_grad():
        outputs_no_col = model(
            input_features=tokenized.tokens,
            attention_mask=tokenized.attention_mask,
            return_attention_weights=True
        )
    
    print(f"Last hidden state shape: {outputs_no_col['last_hidden_state'].shape}")
    print(f"Pooler output shape: {outputs_no_col['pooler_output'].shape}")
    print(f"Has attention weights: {'attention_weights' in outputs_no_col}")
    
    # Test with column metadata (full cross-attention fusion)
    print("\n--- Testing with column metadata ---")
    column_metadata = {
        'column_metadata': tokenizer.column_metadata,
        'dataframe': df
    }
    
    with torch.no_grad():
        outputs_with_col = model(
            input_features=tokenized.tokens,
            attention_mask=tokenized.attention_mask,
            column_metadata=column_metadata,
            return_attention_weights=True
        )
    
    print(f"Last hidden state shape: {outputs_with_col['last_hidden_state'].shape}")
    print(f"Pooler output shape: {outputs_with_col['pooler_output'].shape}")
    print(f"Row embeddings shape: {outputs_with_col['row_embeddings'].shape}")
    print(f"Column embeddings shape: {outputs_with_col['column_embeddings'].shape}")
    print(f"Has attention weights: {'attention_weights' in outputs_with_col}")
    print(f"Has row attention weights: {'row_attention_weights' in outputs_with_col}")
    
    # Compare outputs with and without cross-attention
    similarity = torch.cosine_similarity(
        outputs_no_col['pooler_output'],
        outputs_with_col['pooler_output'],
        dim=1
    ).mean()
    print(f"Similarity between outputs (with/without cross-attention): {similarity:.4f}")
    
    # Step 3: Test classification model with cross-attention
    print("\n=== Step 3: Testing Classification with Cross-Attention ===")
    classification_model = TabGPTForClassification(config, num_labels=3)
    
    # Create dummy labels
    labels = torch.randint(0, 3, (len(df),))
    
    # Test with cross-attention
    class_outputs = classification_model(
        input_features=tokenized.tokens,
        attention_mask=tokenized.attention_mask,
        column_metadata=column_metadata,
        labels=labels
    )
    
    print(f"Classification logits shape: {class_outputs['logits'].shape}")
    print(f"Loss: {class_outputs['loss'].item():.4f}")
    print(f"Predicted classes: {class_outputs['logits'].argmax(dim=1).tolist()}")
    print(f"True labels: {labels.tolist()}")
    
    # Step 4: Test different fusion strategies
    print("\n=== Step 4: Testing Different Fusion Strategies ===")
    
    fusion_strategies = ['add', 'concat', 'gate']
    strategy_outputs = {}
    
    for strategy in fusion_strategies:
        print(f"\n--- Testing {strategy.upper()} fusion strategy ---")
        
        config_strategy = TabGPTConfig(
            d_model=128,
            n_heads=8,
            n_layers=2,
            dropout=0.1,
            max_features=20,
            embedding_dim=128,
            column_embedding_dim=128,
            cross_attention_layers=1,
            fusion_strategy=strategy
        )
        
        model_strategy = TabGPTModel(config_strategy)
        
        with torch.no_grad():
            outputs_strategy = model_strategy(
                input_features=tokenized.tokens,
                attention_mask=tokenized.attention_mask,
                column_metadata=column_metadata
            )
        
        strategy_outputs[strategy] = outputs_strategy
        print(f"Output shape: {outputs_strategy['pooler_output'].shape}")
        print(f"Sample values: {outputs_strategy['pooler_output'][0, :5].tolist()}")
    
    # Compare fusion strategies
    print("\n--- Comparing Fusion Strategies ---")
    strategies = list(strategy_outputs.keys())
    for i, strategy1 in enumerate(strategies):
        for j, strategy2 in enumerate(strategies[i+1:], i+1):
            sim = torch.cosine_similarity(
                strategy_outputs[strategy1]['pooler_output'],
                strategy_outputs[strategy2]['pooler_output'],
                dim=1
            ).mean()
            print(f"Similarity between {strategy1} and {strategy2}: {sim:.4f}")
    
    # Step 5: Test with missing values
    print("\n=== Step 5: Testing with Missing Values ===")
    
    df_missing = df.copy()
    df_missing.loc[1, 'annual_income'] = np.nan
    df_missing.loc[3, 'credit_score'] = np.nan
    df_missing.loc[5, 'account_balance'] = np.nan
    
    tokenizer_missing = TabularTokenizer(embedding_dim=128)
    tokenized_missing = tokenizer_missing.fit_transform(df_missing)
    
    column_metadata_missing = {
        'column_metadata': tokenizer_missing.column_metadata,
        'dataframe': df_missing
    }
    
    with torch.no_grad():
        outputs_missing = model(
            input_features=tokenized_missing.tokens,
            attention_mask=tokenized_missing.attention_mask,
            column_metadata=column_metadata_missing
        )
    
    print(f"Output with missing values shape: {outputs_missing['pooler_output'].shape}")
    print(f"No NaN in outputs: {not torch.isnan(outputs_missing['pooler_output']).any()}")
    
    # Step 6: Test gradient flow
    print("\n=== Step 6: Testing Gradient Flow ===")
    
    # Enable gradients
    tokenized.tokens.requires_grad_(True)
    
    # Forward pass
    train_outputs = classification_model(
        input_features=tokenized.tokens,
        attention_mask=tokenized.attention_mask,
        column_metadata=column_metadata,
        labels=labels
    )
    
    # Backward pass
    loss = train_outputs['loss']
    loss.backward()
    
    # Check gradients
    input_has_grad = tokenized.tokens.grad is not None
    model_has_grad = any(p.grad is not None for p in classification_model.parameters() if p.requires_grad)
    
    print(f"Input gradients computed: {input_has_grad}")
    print(f"Model gradients computed: {model_has_grad}")
    
    # Step 7: Performance analysis
    print("\n=== Step 7: Performance Analysis ===")
    
    import time
    
    # Measure inference time with cross-attention
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(10):
            _ = model(
                input_features=tokenized.tokens,
                attention_mask=tokenized.attention_mask,
                column_metadata=column_metadata
            )
    
    end_time = time.time()
    avg_time_with_cross_attn = (end_time - start_time) / 10
    
    # Measure inference time without cross-attention
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(10):
            _ = model(
                input_features=tokenized.tokens,
                attention_mask=tokenized.attention_mask
            )
    
    end_time = time.time()
    avg_time_without_cross_attn = (end_time - start_time) / 10
    
    print(f"Average inference time with cross-attention: {avg_time_with_cross_attn:.4f} seconds")
    print(f"Average inference time without cross-attention: {avg_time_without_cross_attn:.4f} seconds")
    print(f"Cross-attention overhead: {((avg_time_with_cross_attn / avg_time_without_cross_attn) - 1) * 100:.1f}%")
    
    # Memory usage
    total_params = sum(p.numel() for p in model.parameters())
    cross_attn_params = sum(p.numel() for p in model.cross_attention.parameters())
    
    print(f"Total model parameters: {total_params:,}")
    print(f"Cross-attention parameters: {cross_attn_params:,}")
    print(f"Cross-attention parameter ratio: {(cross_attn_params / total_params) * 100:.1f}%")
    
    # Step 8: Test model saving and loading
    print("\n=== Step 8: Testing Model Save/Load ===")
    import tempfile
    import os
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save model
        model_path = os.path.join(temp_dir, "test_model_cross_attn")
        model.save_pretrained(model_path)
        
        print(f"Model saved to: {model_path}")
        print(f"Files: {os.listdir(model_path)}")
        
        # Load model
        loaded_model = TabGPTModel.from_pretrained(model_path)
        
        # Test loaded model
        with torch.no_grad():
            loaded_outputs = loaded_model(
                input_features=tokenized.tokens,
                attention_mask=tokenized.attention_mask,
                column_metadata=column_metadata
            )
        
        # Compare outputs
        similarity = torch.cosine_similarity(
            outputs_with_col['pooler_output'],
            loaded_outputs['pooler_output'],
            dim=1
        ).mean()
        
        print(f"Output similarity (original vs loaded): {similarity:.6f}")
        assert similarity > 0.95, "Loaded model outputs should be very similar"
    
    # Verify all outputs are valid
    assert not torch.isnan(outputs_no_col['pooler_output']).any()
    assert not torch.isnan(outputs_with_col['pooler_output']).any()
    assert not torch.isnan(class_outputs['logits']).any()
    assert not torch.isnan(outputs_missing['pooler_output']).any()
    
    for strategy_output in strategy_outputs.values():
        assert not torch.isnan(strategy_output['pooler_output']).any()
    
    print("\n✅ All TabGPT cross-attention integration tests passed!")
    
    return {
        'model': model,
        'classification_model': classification_model,
        'tokenized_data': tokenized,
        'outputs_with_cross_attention': outputs_with_col,
        'outputs_without_cross_attention': outputs_no_col,
        'strategy_outputs': strategy_outputs
    }

if __name__ == "__main__":
    results = test_tabgpt_with_cross_attention()