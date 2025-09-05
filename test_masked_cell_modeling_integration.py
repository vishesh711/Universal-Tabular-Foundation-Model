"""Integration test for Masked Cell Modeling with TabGPT."""

import pandas as pd
import numpy as np
import torch
from tabgpt.tokenizers import TabularTokenizer
from tabgpt.models import TabGPTModel
from tabgpt.config import TabGPTConfig
from tabgpt.training import MaskedCellModelingObjective

def test_masked_cell_modeling_integration():
    """Test complete MCM pipeline with real tabular data."""
    
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
        'risk_score': [0.2, 0.6, 0.1, 0.05, 0.7, 0.15, 0.03, 0.25]
    })
    
    print("Test Dataset:")
    print(df)
    print(f"Shape: {df.shape}")
    
    # Step 1: Tokenize the data
    print("\n=== Step 1: Tokenizing Data ===")
    tokenizer = TabularTokenizer(embedding_dim=128)
    tokenized = tokenizer.fit_transform(df)
    
    print(f"Tokenized shape: {tokenized.tokens.shape}")
    print(f"Attention mask shape: {tokenized.attention_mask.shape}")
    print(f"Features: {tokenized.feature_names}")
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")
    
    # Step 2: Create TabGPT model
    print("\n=== Step 2: Creating TabGPT Model ===")
    config = TabGPTConfig(
        d_model=128,
        n_heads=8,
        n_layers=3,
        dropout=0.1,
        embedding_dim=128,
        max_features=20,
        column_embedding_dim=128
    )
    
    model = TabGPTModel(config)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Step 3: Create MCM objective
    print("\n=== Step 3: Creating MCM Objective ===")
    mcm_objective = MaskedCellModelingObjective(
        d_model=128,
        mask_probability=0.15,
        categorical_vocab_size=tokenizer.vocab_size,
        numerical_loss_weight=1.0,
        categorical_loss_weight=1.0
    )
    
    print(f"MCM parameters: {sum(p.numel() for p in mcm_objective.parameters()):,}")
    print(f"Mask probability: {mcm_objective.mask_probability}")
    
    # Step 4: Extract feature types from tokenizer
    print("\n=== Step 4: Extracting Feature Types ===")
    feature_types = []
    for metadata in tokenizer.column_metadata:
        feature_types.append(metadata.dtype)
    
    print(f"Feature types: {feature_types}")
    
    # Step 5: Test MCM forward pass
    print("\n=== Step 5: Testing MCM Forward Pass ===")
    
    # Create input values (simplified - using token sums as proxy)
    input_values = torch.zeros(tokenized.tokens.shape[:2])
    for i, feat_type in enumerate(feature_types):
        if feat_type == 'categorical':
            # Use categorical indices
            input_values[:, i] = torch.randint(0, tokenizer.vocab_size, (len(df),)).float()
        else:
            # Use normalized numerical values
            input_values[:, i] = torch.randn(len(df))
    
    # Forward pass through MCM
    with torch.no_grad():
        mcm_output = mcm_objective(
            input_embeddings=tokenized.tokens,
            input_values=input_values,
            feature_types=feature_types,
            model_forward_fn=model.forward,
            attention_mask=tokenized.attention_mask
        )
    
    print(f"MCM Loss: {mcm_output.loss.item():.4f}")
    print(f"Categorical Loss: {mcm_output.categorical_loss.item() if mcm_output.categorical_loss else 'N/A'}")
    print(f"Numerical Loss: {mcm_output.numerical_loss.item() if mcm_output.numerical_loss else 'N/A'}")
    print(f"Masked positions: {mcm_output.masked_positions.sum().item()}/{mcm_output.masked_positions.numel()}")
    
    # Step 6: Compute and display metrics
    print("\n=== Step 6: Computing Metrics ===")
    metrics = mcm_objective.compute_metrics(mcm_output)
    
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")
    
    # Step 7: Test different mask probabilities
    print("\n=== Step 7: Testing Different Mask Probabilities ===")
    
    mask_probs = [0.05, 0.15, 0.25, 0.35]
    
    for mask_prob in mask_probs:
        mcm_test = MaskedCellModelingObjective(
            d_model=128,
            mask_probability=mask_prob,
            categorical_vocab_size=tokenizer.vocab_size
        )
        
        with torch.no_grad():
            output = mcm_test(
                input_embeddings=tokenized.tokens,
                input_values=input_values,
                feature_types=feature_types,
                model_forward_fn=model.forward,
                attention_mask=tokenized.attention_mask
            )
        
        actual_mask_ratio = output.masked_positions.float().mean().item()
        print(f"Mask prob {mask_prob:.2f}: Actual ratio {actual_mask_ratio:.3f}, Loss {output.loss.item():.4f}")
    
    # Step 8: Test gradient flow and training step
    print("\n=== Step 8: Testing Gradient Flow ===")
    
    # Enable gradients
    model.train()
    mcm_objective.train()
    
    # Create optimizer
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(mcm_objective.parameters()),
        lr=1e-4
    )
    
    # Training step
    optimizer.zero_grad()
    
    train_output = mcm_objective(
        input_embeddings=tokenized.tokens,
        input_values=input_values,
        feature_types=feature_types,
        model_forward_fn=model.forward,
        attention_mask=tokenized.attention_mask
    )
    
    loss = train_output.loss
    loss.backward()
    
    # Check gradients
    model_params_with_grad = [p for p in model.parameters() if p.grad is not None]
    mcm_params_with_grad = [p for p in mcm_objective.parameters() if p.grad is not None]
    
    print(f"Model parameters with gradients: {len(model_params_with_grad)}/{len(list(model.parameters()))}")
    print(f"MCM parameters with gradients: {len(mcm_params_with_grad)}/{len(list(mcm_objective.parameters()))}")
    
    # Compute gradient norms
    model_grad_norm = torch.norm(torch.stack([p.grad.norm() for p in model_params_with_grad])).item()
    mcm_grad_norm = torch.norm(torch.stack([p.grad.norm() for p in mcm_params_with_grad])).item()
    
    print(f"Model gradient norm: {model_grad_norm:.4f}")
    print(f"MCM gradient norm: {mcm_grad_norm:.4f}")
    
    # Take optimization step
    optimizer.step()
    
    print("✓ Gradient flow and optimization step successful")
    
    # Step 9: Test with missing values
    print("\n=== Step 9: Testing with Missing Values ===")
    
    # Create dataset with missing values
    df_missing = df.copy()
    df_missing.loc[1, 'annual_income'] = np.nan
    df_missing.loc[3, 'credit_score'] = np.nan
    df_missing.loc[5, 'account_balance'] = np.nan
    
    # Tokenize with missing values
    tokenizer_missing = TabularTokenizer(embedding_dim=128)
    tokenized_missing = tokenizer_missing.fit_transform(df_missing)
    
    # Create input values for missing data
    input_values_missing = torch.zeros(tokenized_missing.tokens.shape[:2])
    feature_types_missing = [metadata.dtype for metadata in tokenizer_missing.column_metadata]
    
    for i, feat_type in enumerate(feature_types_missing):
        if feat_type == 'categorical':
            input_values_missing[:, i] = torch.randint(0, tokenizer_missing.vocab_size, (len(df),)).float()
        else:
            input_values_missing[:, i] = torch.randn(len(df))
    
    # Test MCM with missing values
    mcm_missing = MaskedCellModelingObjective(
        d_model=128,
        mask_probability=0.2,
        categorical_vocab_size=tokenizer_missing.vocab_size
    )
    
    with torch.no_grad():
        output_missing = mcm_missing(
            input_embeddings=tokenized_missing.tokens,
            input_values=input_values_missing,
            feature_types=feature_types_missing,
            model_forward_fn=model.forward,
            attention_mask=tokenized_missing.attention_mask
        )
    
    print(f"MCM with missing values - Loss: {output_missing.loss.item():.4f}")
    print(f"No NaN in loss: {not torch.isnan(output_missing.loss)}")
    
    # Step 10: Test masking strategies
    print("\n=== Step 10: Testing Masking Strategies ===")
    
    # Test different replace probabilities
    strategies = [
        {'replace_prob': 0.8, 'random_prob': 0.1},  # Standard BERT-like
        {'replace_prob': 1.0, 'random_prob': 0.0},  # Always mask
        {'replace_prob': 0.5, 'random_prob': 0.3},  # More random replacement
    ]
    
    for i, strategy in enumerate(strategies):
        mcm_strategy = MaskedCellModelingObjective(
            d_model=128,
            mask_probability=0.15,
            replace_probability=strategy['replace_prob'],
            random_probability=strategy['random_prob'],
            categorical_vocab_size=tokenizer.vocab_size
        )
        
        with torch.no_grad():
            output_strategy = mcm_strategy(
                input_embeddings=tokenized.tokens,
                input_values=input_values,
                feature_types=feature_types,
                model_forward_fn=model.forward,
                attention_mask=tokenized.attention_mask
            )
        
        print(f"Strategy {i+1} (replace={strategy['replace_prob']}, random={strategy['random_prob']}): "
              f"Loss {output_strategy.loss.item():.4f}")
    
    # Step 11: Performance analysis
    print("\n=== Step 11: Performance Analysis ===")
    
    import time
    
    # Measure inference time
    model.eval()
    mcm_objective.eval()
    
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(10):
            _ = mcm_objective(
                input_embeddings=tokenized.tokens,
                input_values=input_values,
                feature_types=feature_types,
                model_forward_fn=model.forward,
                attention_mask=tokenized.attention_mask
            )
    
    end_time = time.time()
    avg_time = (end_time - start_time) / 10
    
    print(f"Average MCM inference time: {avg_time:.4f} seconds")
    print(f"Throughput: {len(df) / avg_time:.1f} samples/second")
    
    # Memory usage
    total_params = sum(p.numel() for p in model.parameters()) + sum(p.numel() for p in mcm_objective.parameters())
    mcm_params = sum(p.numel() for p in mcm_objective.parameters())
    
    print(f"Total parameters (Model + MCM): {total_params:,}")
    print(f"MCM parameters: {mcm_params:,}")
    print(f"MCM parameter ratio: {(mcm_params / total_params) * 100:.1f}%")
    
    # Verify all outputs are valid
    assert not torch.isnan(mcm_output.loss)
    assert mcm_output.masked_positions.dtype == torch.bool
    assert 'categorical' in mcm_output.predictions
    assert 'numerical' in mcm_output.predictions
    
    print("\n✅ All Masked Cell Modeling tests passed!")
    
    return {
        'tokenized_data': tokenized,
        'model': model,
        'mcm_objective': mcm_objective,
        'mcm_output': mcm_output,
        'metrics': metrics
    }

if __name__ == "__main__":
    results = test_masked_cell_modeling_integration()