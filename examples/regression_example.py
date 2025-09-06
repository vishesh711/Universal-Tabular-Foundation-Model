#!/usr/bin/env python3
"""
Regression Example with TabGPT

This example demonstrates how to use TabGPT for regression tasks using
the California Housing dataset. It covers data loading, preprocessing,
model training, and evaluation with uncertainty estimation.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.datasets import fetch_california_housing
import torch
import matplotlib.pyplot as plt

# TabGPT imports
from tabgpt import TabGPTForRegression, TabGPTTokenizer
from tabgpt.heads import RegressionHead
from tabgpt.utils import RobustNormalizer, DataValidator
from tabgpt.fine_tuning import TabGPTFineTuningTrainer, FineTuningConfig


def load_california_housing():
    """Load and prepare the California Housing dataset."""
    # Load the dataset
    housing = fetch_california_housing()
    
    # Create DataFrame
    df = pd.DataFrame(housing.data, columns=housing.feature_names)
    df['target'] = housing.target
    
    # Add some feature engineering
    df['rooms_per_household'] = df['AveRooms'] / df['AveOccup']
    df['bedrooms_per_room'] = df['AveBedrms'] / df['AveRooms']
    df['population_per_household'] = df['Population'] / df['HouseAge']
    
    return df


def plot_predictions(y_true, y_pred, uncertainties=None, title="Predictions vs Actual"):
    """Plot predictions against actual values."""
    plt.figure(figsize=(10, 6))
    
    if uncertainties is not None:
        # Plot with error bars
        plt.errorbar(y_true, y_pred, yerr=uncertainties, fmt='o', alpha=0.6, capsize=3)
    else:
        plt.scatter(y_true, y_pred, alpha=0.6)
    
    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('regression_predictions.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """Main function demonstrating TabGPT regression."""
    print("TabGPT Regression Example")
    print("=" * 30)
    
    # 1. Load and explore data
    print("1. Loading California Housing dataset...")
    df = load_california_housing()
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Target statistics:")
    print(df['target'].describe())
    print()
    
    # 2. Validate data quality
    print("2. Validating data quality...")
    validator = DataValidator(
        missing_threshold=0.1,
        min_samples=1000,
        min_features=5
    )
    
    validation_result = validator.validate_dataframe(df)
    print(f"Data validation passed: {validation_result['valid']}")
    if validation_result['warnings']:
        print(f"Warnings: {validation_result['warnings']}")
    print()
    
    # 3. Prepare features and target
    print("3. Preparing features and target...")
    X = df.drop('target', axis=1)
    y = df['target']
    
    print(f"Features: {list(X.columns)}")
    print(f"Target range: [{y.min():.2f}, {y.max():.2f}]")
    print()
    
    # 4. Robust preprocessing
    print("4. Applying robust preprocessing...")
    normalizer = RobustNormalizer(
        numerical_strategy="robust",
        categorical_strategy="frequency",
        outlier_action="clip",
        missing_strategy="median"
    )
    
    X_processed, norm_log = normalizer.fit_transform(X)
    print(f"Preprocessing completed: {norm_log['success']}")
    print()
    
    # 5. Split data
    print("5. Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.2, random_state=42
    )
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print()
    
    # 6. Initialize TabGPT model and tokenizer
    print("6. Initializing TabGPT regression model...")
    tokenizer = TabGPTTokenizer(
        vocab_size=15000,
        max_length=512
    )
    
    # Create model with uncertainty estimation
    model = TabGPTForRegression(output_dim=1)
    
    # Replace regression head with uncertainty-aware head
    model.regression_head = RegressionHead(
        input_dim=model.config.hidden_size,
        output_dim=1,
        estimate_uncertainty=True,
        hidden_dims=[512, 256]
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()
    
    # 7. Prepare datasets
    print("7. Preparing datasets...")
    train_dataset = tokenizer.create_dataset(X_train, y_train)
    test_dataset = tokenizer.create_dataset(X_test, y_test)
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    print()
    
    # 8. Configure training
    print("8. Configuring training...")
    config = FineTuningConfig(
        task_type="regression",
        output_dim=1,
        learning_rate=3e-5,
        num_epochs=5,
        batch_size=64,
        warmup_steps=200,
        weight_decay=0.01,
        eval_steps=100,
        logging_steps=50,
        save_steps=200,
        load_best_model_at_end=True,
        metric_for_best_model="eval_mse",
        greater_is_better=False
    )
    
    # 9. Create trainer with regression metrics
    print("9. Creating trainer...")
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        # Handle uncertainty if present
        if predictions.shape[-1] == 2:  # Mean and uncertainty
            pred_mean = predictions[:, 0]
        else:
            pred_mean = predictions.flatten()
        
        labels = labels.flatten()
        
        mse = mean_squared_error(labels, pred_mean)
        mae = mean_absolute_error(labels, pred_mean)
        r2 = r2_score(labels, pred_mean)
        
        return {
            'mse': mse,
            'rmse': np.sqrt(mse),
            'mae': mae,
            'r2': r2
        }
    
    trainer = TabGPTFineTuningTrainer(
        model=model,
        config=config,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    
    # 10. Train model
    print("10. Training model...")
    print("This may take several minutes...")
    trainer.train()
    print("Training completed!")
    print()
    
    # 11. Evaluate model
    print("11. Evaluating model...")
    eval_results = trainer.evaluate()
    print("Evaluation Results:")
    for metric, value in eval_results.items():
        print(f"  {metric}: {value:.4f}")
    print()
    
    # 12. Make predictions with uncertainty
    print("12. Making predictions on test set...")
    model.eval()
    test_tokens = tokenizer.encode_batch(X_test)
    
    with torch.no_grad():
        outputs = model(**test_tokens)
        
        # Handle uncertainty estimation
        if hasattr(outputs, 'uncertainty') and outputs.uncertainty is not None:
            predictions = outputs.predictions.cpu().numpy().flatten()
            uncertainties = outputs.uncertainty.cpu().numpy().flatten()
            print(f"Predictions with uncertainty available")
        else:
            predictions = outputs.predictions.cpu().numpy().flatten()
            uncertainties = None
            print(f"Predictions without uncertainty")
    
    # 13. Compute detailed metrics
    print("13. Computing detailed metrics...")
    y_test_np = y_test.values
    
    mse = mean_squared_error(y_test_np, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_np, predictions)
    r2 = r2_score(y_test_np, predictions)
    
    print(f"Test MSE: {mse:.4f}")
    print(f"Test RMSE: {rmse:.4f}")
    print(f"Test MAE: {mae:.4f}")
    print(f"Test R²: {r2:.4f}")
    
    if uncertainties is not None:
        mean_uncertainty = np.mean(uncertainties)
        print(f"Mean Uncertainty: {mean_uncertainty:.4f}")
    print()
    
    # 14. Analyze predictions by price range
    print("14. Analyzing predictions by price range...")
    price_ranges = [(0, 2), (2, 4), (4, 6), (6, float('inf'))]
    
    for low, high in price_ranges:
        mask = (y_test_np >= low) & (y_test_np < high)
        if mask.sum() > 0:
            range_mse = mean_squared_error(y_test_np[mask], predictions[mask])
            range_r2 = r2_score(y_test_np[mask], predictions[mask])
            print(f"Price range [{low}, {high}): MSE={range_mse:.4f}, R²={range_r2:.4f}, n={mask.sum()}")
    print()
    
    # 15. Show example predictions
    print("15. Example predictions:")
    print("-" * 70)
    print("Actual    Predicted   Error     Uncertainty")
    print("-" * 70)
    
    for i in range(min(15, len(y_test))):
        actual = y_test_np[i]
        pred = predictions[i]
        error = abs(actual - pred)
        unc = uncertainties[i] if uncertainties is not None else 0.0
        
        print(f"{actual:6.2f}    {pred:8.2f}    {error:6.2f}    {unc:8.2f}")
    print()
    
    # 16. Plot results
    print("16. Creating visualization...")
    try:
        plot_predictions(
            y_test_np, 
            predictions, 
            uncertainties,
            "TabGPT Regression: Predictions vs Actual (California Housing)"
        )
        print("Visualization saved as 'regression_predictions.png'")
    except Exception as e:
        print(f"Could not create plot: {e}")
    print()
    
    # 17. Feature importance analysis (simplified)
    print("17. Analyzing feature importance...")
    feature_names = X.columns.tolist()
    
    # Simple gradient-based importance
    model.train()
    sample_tokens = tokenizer.encode_batch(X_test.iloc[:100])
    sample_tokens['input_ids'].requires_grad_(True)
    
    outputs = model(**sample_tokens)
    loss = outputs.loss if outputs.loss is not None else outputs.predictions.sum()
    loss.backward()
    
    # Get gradient magnitudes (simplified importance)
    grad_importance = sample_tokens['input_ids'].grad.abs().mean(dim=0)
    
    print("Top 5 most important features (by gradient magnitude):")
    for i, importance in enumerate(grad_importance[:5]):
        print(f"  Feature {i}: {importance:.4f}")
    print()
    
    # 18. Save model
    print("18. Saving model...")
    model_path = "./trained_housing_model"
    trainer.save_model(model_path)
    print(f"Model saved to: {model_path}")
    print()
    
    # 19. Demonstrate loading and inference
    print("19. Demonstrating model loading...")
    # In practice, you would load like this:
    # loaded_model = TabGPTForRegression.from_pretrained(model_path)
    # loaded_tokenizer = TabGPTTokenizer.from_pretrained(model_path)
    
    print("Model can be loaded for inference using:")
    print("  model = TabGPTForRegression.from_pretrained('./trained_housing_model')")
    print("  tokenizer = TabGPTTokenizer.from_pretrained('./trained_housing_model')")
    print()
    
    print("Regression example completed successfully!")
    print(f"Final test R² score: {r2:.4f}")


if __name__ == "__main__":
    main()