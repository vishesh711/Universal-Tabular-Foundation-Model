#!/usr/bin/env python3
"""
Basic Classification Example with TabGPT

This example demonstrates how to use TabGPT for a simple binary classification task
using the Titanic dataset. It covers data loading, preprocessing, model training,
and evaluation.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import torch

# TabGPT imports
from tabgpt import TabGPTForSequenceClassification, TabGPTTokenizer
from tabgpt.utils import RobustNormalizer, DataValidator
from tabgpt.fine_tuning import TabGPTFineTuningTrainer, FineTuningConfig


def load_titanic_data():
    """Load and prepare the Titanic dataset."""
    # Create synthetic Titanic-like data for demonstration
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'age': np.random.normal(30, 12, n_samples),
        'fare': np.random.lognormal(3, 1, n_samples),
        'pclass': np.random.choice([1, 2, 3], n_samples, p=[0.2, 0.3, 0.5]),
        'sex': np.random.choice(['male', 'female'], n_samples, p=[0.6, 0.4]),
        'embarked': np.random.choice(['C', 'Q', 'S'], n_samples, p=[0.2, 0.1, 0.7]),
        'sibsp': np.random.poisson(0.5, n_samples),
        'parch': np.random.poisson(0.4, n_samples)
    }
    
    # Create target variable with some logic
    survived_prob = (
        0.1 +  # Base survival rate
        0.4 * (data['sex'] == 'female').astype(int) +  # Women more likely to survive
        0.2 * (data['pclass'] == 1).astype(int) +  # First class more likely
        0.1 * (data['age'] < 18).astype(int) -  # Children more likely
        0.1 * (data['age'] > 60).astype(int)  # Elderly less likely
    )
    survived_prob = np.clip(survived_prob, 0, 1)
    data['survived'] = np.random.binomial(1, survived_prob, n_samples)
    
    df = pd.DataFrame(data)
    
    # Add some missing values to make it realistic
    missing_indices = np.random.choice(n_samples, size=int(0.1 * n_samples), replace=False)
    df.loc[missing_indices, 'age'] = np.nan
    
    return df


def main():
    """Main function demonstrating TabGPT classification."""
    print("TabGPT Basic Classification Example")
    print("=" * 40)
    
    # 1. Load and explore data
    print("1. Loading Titanic dataset...")
    df = load_titanic_data()
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Target distribution: {df['survived'].value_counts().to_dict()}")
    print()
    
    # 2. Validate data quality
    print("2. Validating data quality...")
    validator = DataValidator(
        missing_threshold=0.3,
        min_samples=100,
        min_features=2
    )
    
    validation_result = validator.validate_dataframe(df)
    print(f"Data validation passed: {validation_result['valid']}")
    if validation_result['warnings']:
        print(f"Warnings: {validation_result['warnings']}")
    print()
    
    # 3. Prepare features and target
    print("3. Preparing features and target...")
    X = df.drop('survived', axis=1)
    y = df['survived']
    
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
    print(f"Actions taken: {len(norm_log.get('actions_taken', []))}\")\n    
    # 5. Split data
    print("5. Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print()
    
    # 6. Initialize TabGPT model and tokenizer
    print("6. Initializing TabGPT model...")
    tokenizer = TabGPTTokenizer(
        vocab_size=10000,
        max_length=256
    )
    
    model = TabGPTForSequenceClassification(
        num_labels=2  # Binary classification
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
        task_type="classification",
        num_labels=2,
        learning_rate=5e-5,
        num_epochs=3,
        batch_size=32,
        warmup_steps=100,
        weight_decay=0.01,
        eval_steps=50,
        logging_steps=25,
        save_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="eval_accuracy",
        greater_is_better=True
    )
    
    # 9. Create trainer
    print("9. Creating trainer...")
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return {
            'accuracy': accuracy_score(labels, predictions),
            'f1': f1_score(labels, predictions, average='weighted')
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
    print("This may take a few minutes...")
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
    
    # 12. Make predictions
    print("12. Making predictions on test set...")
    model.eval()
    test_tokens = tokenizer.encode_batch(X_test)
    
    with torch.no_grad():
        outputs = model(**test_tokens)
        predictions = outputs.logits.argmax(dim=-1).cpu().numpy()
        probabilities = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
    
    # 13. Compute detailed metrics
    print("13. Computing detailed metrics...")
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)
    cm = confusion_matrix(y_test, predictions)
    
    print(f"Test Accuracy: {accuracy:.4f}")
    print("\\nClassification Report:")
    print(report)
    print("\\nConfusion Matrix:")
    print(cm)
    print()
    
    # 14. Show some example predictions
    print("14. Example predictions:")
    print("-" * 50)
    for i in range(min(10, len(X_test))):
        actual = y_test.iloc[i]
        pred = predictions[i]
        prob = probabilities[i][1]  # Probability of survival
        status = "✓" if actual == pred else "✗"
        print(f"{status} Actual: {actual}, Predicted: {pred}, Prob(Survive): {prob:.3f}")
    print()
    
    # 15. Save model
    print("15. Saving model...")
    model_path = "./trained_titanic_model"
    trainer.save_model(model_path)
    print(f"Model saved to: {model_path}")
    print()
    
    print("Example completed successfully!")
    print("You can now use the trained model for inference on new data.")


if __name__ == "__main__":
    # Import additional dependencies
    from sklearn.metrics import f1_score
    
    main()