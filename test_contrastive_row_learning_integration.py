"""Integration test for Contrastive Row Learning with TabGPT."""

import pandas as pd
import numpy as np
import torch
from tabgpt.tokenizers import TabularTokenizer
from tabgpt.models import TabGPTModel
from tabgpt.config import TabGPTConfig
from tabgpt.training import (
    ContrastiveRowLearningObjective,
    create_default_augmentation_strategies,
    NoiseInjectionAugmentation,
    FeatureDropoutAugmentation,
    ValuePerturbationAugmentation
)
from tabgpt.encoders import ColumnEncoder

def test_contrastive_row_learning_integration():
    """Test complete CRL pipeline with real tabular data."""
    
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
        'risk_score': [0.2, 0.6, 0.1, 0.05, 0.7, 0.15, 0.03, 0.25],
        'signup_month': [1, 3, 5, 2, 8, 4, 6, 7],
        'transaction_count': [45, 23, 67, 89, 34, 56, 123, 78]
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
    print(f"Column metadata count: {len(tokenizer.column_metadata)}")
    
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
    
    # Step 3: Get model outputs and row embeddings
    print("\n=== Step 3: Generating Row Embeddings ===")
    with torch.no_grad():
        model_outputs = model(
            input_features=tokenized.tokens,
            attention_mask=tokenized.attention_mask
        )
    
    print(f"Available model output keys: {list(model_outputs.keys())}")
    
    # Use the last hidden state and pool it
    if 'last_hidden_state' in model_outputs:
        # Pool the sequence dimension to get row embeddings
        row_embeddings = model_outputs['last_hidden_state'].mean(dim=1)  # [batch_size, d_model]
    else:
        # Fallback: use the first available output
        first_key = list(model_outputs.keys())[0]
        output_tensor = model_outputs[first_key]
        if len(output_tensor.shape) == 3:  # [batch_size, seq_len, d_model]
            row_embeddings = output_tensor.mean(dim=1)
        else:
            row_embeddings = output_tensor
    
    print(f"Row embeddings shape: {row_embeddings.shape}")
    print(f"Row embedding sample: {row_embeddings[0, :5].tolist()}")
    
    # Step 4: Test individual augmentation strategies
    print("\n=== Step 4: Testing Augmentation Strategies ===")
    
    # Test noise injection
    noise_aug = NoiseInjectionAugmentation(
        augmentation_probability=1.0,
        noise_std=0.1,
        numerical_only=True
    )
    
    augmented_noise = noise_aug(
        tokenized.tokens,
        attention_mask=tokenized.attention_mask,
        column_metadata=tokenizer.column_metadata
    )
    
    print(f"Original features sample: {tokenized.tokens[0, 0, :5].tolist()}")
    print(f"Noise augmented sample: {augmented_noise[0, 0, :5].tolist()}")
    print(f"Noise difference: {torch.norm(augmented_noise - tokenized.tokens).item():.4f}")
    
    # Test feature dropout
    dropout_aug = FeatureDropoutAugmentation(
        augmentation_probability=1.0,
        dropout_probability=0.3,
        min_features=2
    )
    
    augmented_dropout = dropout_aug(
        tokenized.tokens,
        attention_mask=tokenized.attention_mask
    )
    
    # Count zero features
    zero_features = (augmented_dropout == 0).all(dim=-1).sum(dim=-1)
    print(f"Features dropped per sample: {zero_features.tolist()}")
    
    # Test value perturbation
    perturb_aug = ValuePerturbationAugmentation(
        augmentation_probability=1.0,
        numerical_perturbation_std=0.05
    )
    
    augmented_perturb = perturb_aug(
        tokenized.tokens,
        column_metadata=tokenizer.column_metadata
    )
    
    perturbation_diff = torch.norm(augmented_perturb - tokenized.tokens).item()
    print(f"Perturbation difference: {perturbation_diff:.4f}")
    
    # Step 5: Create CRL objective with custom augmentations
    print("\n=== Step 5: Creating CRL Objective ===")
    
    # Create custom augmentation strategies
    custom_augmentations = [
        NoiseInjectionAugmentation(
            augmentation_probability=0.6,
            noise_std=0.1,
            numerical_only=True
        ),
        FeatureDropoutAugmentation(
            augmentation_probability=0.5,
            dropout_probability=0.2,
            min_features=2
        ),
        ValuePerturbationAugmentation(
            augmentation_probability=0.4,
            numerical_perturbation_std=0.05,
            categorical_swap_probability=0.1
        )
    ]
    
    crl_objective = ContrastiveRowLearningObjective(
        d_model=128,
        augmentation_strategies=custom_augmentations,
        temperature=0.07,
        normalize_embeddings=True,
        projection_dim=64,
        use_projection_head=True
    )
    
    print(f"CRL parameters: {sum(p.numel() for p in crl_objective.parameters()):,}")
    print(f"Temperature: {crl_objective.temperature}")
    print(f"Number of augmentation strategies: {len(crl_objective.augmentation_strategy.augmentation_strategies)}")
    
    # Step 6: Test augmented pair creation
    print("\n=== Step 6: Testing Augmented Pair Creation ===")
    
    aug_v1, aug_v2 = crl_objective.create_augmented_pairs(
        tokenized.tokens,
        attention_mask=tokenized.attention_mask,
        column_metadata=tokenizer.column_metadata
    )
    
    print(f"Augmented v1 shape: {aug_v1.shape}")
    print(f"Augmented v2 shape: {aug_v2.shape}")
    
    # Check that augmentations are different
    v1_diff = torch.norm(aug_v1 - tokenized.tokens).item()
    v2_diff = torch.norm(aug_v2 - tokenized.tokens).item()
    v1_v2_diff = torch.norm(aug_v1 - aug_v2).item()
    
    print(f"V1 vs Original difference: {v1_diff:.4f}")
    print(f"V2 vs Original difference: {v2_diff:.4f}")
    print(f"V1 vs V2 difference: {v1_v2_diff:.4f}")
    
    # Step 7: Create model forward function for CRL
    print("\n=== Step 7: Creating Model Forward Function ===")
    
    def model_forward_fn(features, mask):
        """Forward function for computing embeddings from augmented features."""
        with torch.no_grad():
            outputs = model(input_features=features, attention_mask=mask)
            # Use the same pooling strategy as above
            if 'last_hidden_state' in outputs:
                return outputs['last_hidden_state'].mean(dim=1)
            else:
                first_key = list(outputs.keys())[0]
                output_tensor = outputs[first_key]
                if len(output_tensor.shape) == 3:
                    return output_tensor.mean(dim=1)
                else:
                    return output_tensor
    
    # Test the forward function
    test_embeddings = model_forward_fn(aug_v1, tokenized.attention_mask)
    print(f"Test embeddings shape: {test_embeddings.shape}")
    print(f"Test embeddings sample: {test_embeddings[0, :5].tolist()}")
    
    # Step 8: Test CRL forward pass
    print("\n=== Step 8: Testing CRL Forward Pass ===")
    
    crl_output = crl_objective(
        row_embeddings=row_embeddings,
        input_features=tokenized.tokens,
        attention_mask=tokenized.attention_mask,
        column_metadata=tokenizer.column_metadata,
        model_forward_fn=model_forward_fn
    )
    
    print(f"CRL Loss: {crl_output.loss.item():.4f}")
    print(f"Logits shape: {crl_output.logits.shape}")
    print(f"Labels shape: {crl_output.labels.shape}")
    print(f"Temperature: {crl_output.temperature}")
    print(f"Positive pairs shape: {crl_output.positive_pairs.shape}")
    
    # Step 9: Compute and display metrics
    print("\n=== Step 9: Computing Metrics ===")
    metrics = crl_objective.compute_metrics(crl_output)
    
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")
    
    # Step 10: Analyze contrastive learning effectiveness
    print("\n=== Step 10: Analyzing Contrastive Learning ===")
    
    # Check similarity patterns
    logits = crl_output.logits
    positive_similarities = logits[:, 0]  # First column is positive similarities
    negative_similarities = logits[:, 1:] if logits.shape[1] > 1 else torch.zeros_like(positive_similarities)
    
    print(f"Positive similarities: {positive_similarities.tolist()}")
    if negative_similarities.numel() > 0:
        print(f"Negative similarities (mean): {negative_similarities.mean(dim=1).tolist()}")
        print(f"Negative similarities (max): {negative_similarities.max(dim=1)[0].tolist()}")
    
    # Compute accuracy
    predictions = logits.argmax(dim=-1)
    accuracy = (predictions == crl_output.labels).float().mean().item()
    print(f"Contrastive accuracy: {accuracy:.4f}")
    
    # Step 11: Test gradient flow
    print("\n=== Step 11: Testing Gradient Flow ===")
    
    # Enable gradients
    row_embeddings_grad = row_embeddings.clone().requires_grad_(True)
    input_features_grad = tokenized.tokens.clone().requires_grad_(True)
    
    # Forward pass with gradients (without model_forward_fn to avoid gradient issues)
    crl_output_grad = crl_objective(
        row_embeddings=row_embeddings_grad,
        input_features=input_features_grad,
        attention_mask=tokenized.attention_mask,
        column_metadata=tokenizer.column_metadata
        # Don't use model_forward_fn here to avoid gradient complications
    )
    
    # Backward pass
    crl_output_grad.loss.backward()
    
    print(f"Row embeddings gradient norm: {row_embeddings_grad.grad.norm().item():.6f}")
    
    # Check if input features have gradients (they might not due to augmentation)
    if input_features_grad.grad is not None:
        print(f"Input features gradient norm: {input_features_grad.grad.norm().item():.6f}")
    else:
        print("Input features gradient: None (expected due to augmentation process)")
    
    # Check CRL parameter gradients
    crl_param_grads = []
    for name, param in crl_objective.named_parameters():
        if param.grad is not None:
            crl_param_grads.append((name, param.grad.norm().item()))
    
    print("CRL parameter gradients:")
    for name, grad_norm in crl_param_grads[:5]:  # Show first 5
        print(f"  {name}: {grad_norm:.6f}")
    
    # Step 12: Test different temperatures
    print("\n=== Step 12: Testing Different Temperatures ===")
    
    temperatures = [0.01, 0.05, 0.1, 0.2, 0.5]
    temp_results = []
    
    for temp in temperatures:
        temp_objective = ContrastiveRowLearningObjective(
            d_model=128,
            temperature=temp,
            use_projection_head=False  # Faster for testing
        )
        
        with torch.no_grad():
            temp_output = temp_objective(
                row_embeddings=row_embeddings,
                input_features=tokenized.tokens,
                attention_mask=tokenized.attention_mask
            )
        
        temp_results.append({
            'temperature': temp,
            'loss': temp_output.loss.item(),
            'accuracy': temp_output.accuracy['contrastive_accuracy'],
            'pos_sim': temp_output.accuracy['positive_similarity'],
            'neg_sim': temp_output.accuracy['negative_similarity']
        })
    
    print("Temperature analysis:")
    for result in temp_results:
        print(f"  T={result['temperature']:.2f}: loss={result['loss']:.3f}, "
              f"acc={result['accuracy']:.3f}, pos_sim={result['pos_sim']:.3f}, "
              f"neg_sim={result['neg_sim']:.3f}")
    
    # Step 13: Test with default augmentation strategies
    print("\n=== Step 13: Testing Default Augmentation Strategies ===")
    
    default_augs = create_default_augmentation_strategies(
        noise_std=0.1,
        dropout_prob=0.15,
        perturbation_std=0.05,
        shuffle_ratio=0.2,
        mix_ratio=0.3
    )
    
    default_crl = ContrastiveRowLearningObjective(
        d_model=128,
        augmentation_strategies=default_augs,
        temperature=0.07
    )
    
    with torch.no_grad():
        default_output = default_crl(
            row_embeddings=row_embeddings,
            input_features=tokenized.tokens,
            attention_mask=tokenized.attention_mask,
            column_metadata=tokenizer.column_metadata
        )
    
    default_metrics = default_crl.compute_metrics(default_output)
    
    print("Default augmentation results:")
    for metric_name, metric_value in default_metrics.items():
        print(f"  {metric_name}: {metric_value:.4f}")
    
    # Step 14: Performance comparison
    print("\n=== Step 14: Performance Summary ===")
    
    print(f"Dataset: {df.shape[0]} samples, {df.shape[1]} features")
    print(f"Model size: {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"CRL size: {sum(p.numel() for p in crl_objective.parameters()):,} parameters")
    print(f"Final CRL loss: {crl_output.loss.item():.4f}")
    print(f"Final contrastive accuracy: {metrics['contrastive_accuracy']:.4f}")
    print(f"Similarity gap: {metrics['similarity_gap']:.4f}")
    
    # Verify that contrastive learning is working
    assert crl_output.loss.item() > 0, "CRL loss should be positive"
    assert not torch.isnan(crl_output.loss), "CRL loss should not be NaN"
    assert metrics['contrastive_accuracy'] >= 0.0, "Accuracy should be non-negative"
    assert metrics['similarity_gap'] != 0.0, "Should have similarity gap between pos/neg"
    
    print("\n✅ Contrastive Row Learning integration test completed successfully!")
    
    return {
        'crl_loss': crl_output.loss.item(),
        'contrastive_accuracy': metrics['contrastive_accuracy'],
        'similarity_gap': metrics['similarity_gap'],
        'model_params': sum(p.numel() for p in model.parameters()),
        'crl_params': sum(p.numel() for p in crl_objective.parameters())
    }


if __name__ == "__main__":
    results = test_contrastive_row_learning_integration()
    print(f"\nFinal Results: {results}")