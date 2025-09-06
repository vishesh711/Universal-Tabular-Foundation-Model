#!/usr/bin/env python3
"""
Transfer Learning Example with TabGPT

This example demonstrates how to use pre-trained TabGPT models for transfer learning.
It shows how to fine-tune a model pre-trained on a large dataset for a smaller,
domain-specific task, and compares the performance with training from scratch.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import make_classification
import torch

# TabGPT imports
from tabgpt import TabGPTForSequenceClassification, TabGPTTokenizer
from tabgpt.adapters import LoRAConfig, apply_lora_to_model
from tabgpt.fine_tuning import TabGPTFineTuningTrainer, FineTuningConfig
from tabgpt.evaluation import TransferLearningEvaluator, EvaluationConfig
from tabgpt.utils import RobustNormalizer


def create_source_dataset(n_samples=10000):
    """Create a large source dataset for pre-training."""
    print("Creating large source dataset...")
    
    # Generate a complex classification dataset
    X, y = make_classification(
        n_samples=n_samples,
        n_features=20,
        n_informative=15,
        n_redundant=3,
        n_clusters_per_class=2,
        class_sep=0.8,
        random_state=42
    )
    
    # Convert to DataFrame with meaningful column names
    feature_names = [f'feature_{i:02d}' for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    # Add some categorical features
    df['category_A'] = np.random.choice(['type1', 'type2', 'type3'], n_samples, p=[0.5, 0.3, 0.2])
    df['category_B'] = np.random.choice(['group_x', 'group_y'], n_samples, p=[0.6, 0.4])
    
    # Add some noise and missing values
    noise_cols = np.random.choice(feature_names, 3, replace=False)
    for col in noise_cols:
        missing_idx = np.random.choice(n_samples, int(0.05 * n_samples), replace=False)
        df.loc[missing_idx, col] = np.nan
    
    return df


def create_target_dataset(n_samples=500):
    """Create a smaller target dataset with different characteristics."""
    print("Creating small target dataset...")
    
    # Generate a related but different classification task
    X, y = make_classification(
        n_samples=n_samples,
        n_features=15,  # Fewer features
        n_informative=10,
        n_redundant=2,
        n_clusters_per_class=1,
        class_sep=0.6,  # More challenging
        random_state=123
    )
    
    # Convert to DataFrame
    feature_names = [f'target_feature_{i:02d}' for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    # Add domain-specific categorical features
    df['domain_category'] = np.random.choice(['A', 'B', 'C'], n_samples, p=[0.4, 0.4, 0.2])
    df['specialized_type'] = np.random.choice(['alpha', 'beta'], n_samples, p=[0.7, 0.3])
    
    return df


def train_baseline_model(X_train, y_train, X_val, y_val, tokenizer):
    """Train a baseline model from scratch."""
    print("Training baseline model from scratch...")
    
    # Create fresh model
    baseline_model = TabGPTForSequenceClassification(num_labels=2)
    
    # Prepare datasets
    train_dataset = tokenizer.create_dataset(X_train, y_train)
    val_dataset = tokenizer.create_dataset(X_val, y_val)
    
    # Configure training
    config = FineTuningConfig(
        task_type="classification",
        num_labels=2,
        learning_rate=5e-5,
        num_epochs=5,
        batch_size=32,
        eval_steps=50,
        logging_steps=25,
        load_best_model_at_end=True,
        metric_for_best_model="eval_accuracy",
        greater_is_better=True
    )
    
    # Train
    trainer = TabGPTFineTuningTrainer(
        model=baseline_model,
        config=config,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=lambda x: {"accuracy": accuracy_score(x.label_ids, x.predictions.argmax(-1))}
    )
    
    trainer.train()
    return baseline_model, trainer


def pretrain_on_source_data(source_df):
    """Pre-train a model on the large source dataset."""
    print("Pre-training model on source dataset...")
    
    # Prepare source data
    X_source = source_df.drop('target', axis=1)
    y_source = source_df['target']
    
    # Split source data
    X_source_train, X_source_val, y_source_train, y_source_val = train_test_split(
        X_source, y_source, test_size=0.2, random_state=42, stratify=y_source
    )
    
    # Preprocess
    normalizer = RobustNormalizer()
    X_source_train_processed, _ = normalizer.fit_transform(X_source_train)
    X_source_val_processed, _ = normalizer.transform(X_source_val)
    
    # Initialize model and tokenizer
    tokenizer = TabGPTTokenizer(vocab_size=15000, max_length=512)
    model = TabGPTForSequenceClassification(num_labels=2)
    
    # Prepare datasets
    train_dataset = tokenizer.create_dataset(X_source_train_processed, y_source_train)
    val_dataset = tokenizer.create_dataset(X_source_val_processed, y_source_val)
    
    # Configure pre-training
    config = FineTuningConfig(
        task_type="classification",
        num_labels=2,
        learning_rate=3e-5,
        num_epochs=3,
        batch_size=64,
        eval_steps=100,
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="eval_accuracy",
        greater_is_better=True
    )
    
    # Train on source data
    trainer = TabGPTFineTuningTrainer(
        model=model,
        config=config,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=lambda x: {"accuracy": accuracy_score(x.label_ids, x.predictions.argmax(-1))}
    )
    
    trainer.train()
    
    # Evaluate on source validation set
    source_results = trainer.evaluate()
    print(f"Source dataset performance: {source_results['eval_accuracy']:.4f}")
    
    return model, tokenizer, normalizer


def fine_tune_with_lora(pretrained_model, X_train, y_train, X_val, y_val, tokenizer):
    """Fine-tune pre-trained model using LoRA."""
    print("Fine-tuning with LoRA...")
    
    # Configure LoRA
    lora_config = LoRAConfig(
        r=8,
        alpha=16,
        dropout=0.1,
        target_modules=["query", "key", "value", "dense"]
    )
    
    # Apply LoRA to model
    lora_model = apply_lora_to_model(pretrained_model, lora_config)
    
    # Check parameter efficiency
    total_params = sum(p.numel() for p in lora_model.parameters())
    trainable_params = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
    print(f"LoRA efficiency: {trainable_params:,} / {total_params:,} ({trainable_params/total_params:.2%} trainable)")
    
    # Prepare datasets
    train_dataset = tokenizer.create_dataset(X_train, y_train)
    val_dataset = tokenizer.create_dataset(X_val, y_val)
    
    # Configure fine-tuning with higher learning rate for LoRA
    config = FineTuningConfig(
        task_type="classification",
        num_labels=2,
        learning_rate=5e-4,  # Higher LR for LoRA
        num_epochs=3,
        batch_size=32,
        eval_steps=25,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="eval_accuracy",
        greater_is_better=True
    )
    
    # Fine-tune
    trainer = TabGPTFineTuningTrainer(
        model=lora_model,
        config=config,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=lambda x: {"accuracy": accuracy_score(x.label_ids, x.predictions.argmax(-1))}
    )
    
    trainer.train()
    return lora_model, trainer


def evaluate_models(models, X_test, y_test, tokenizer):
    """Evaluate all models on the test set."""
    results = {}
    
    for name, model in models.items():
        print(f"Evaluating {name}...")
        
        model.eval()
        test_tokens = tokenizer.encode_batch(X_test)
        
        with torch.no_grad():
            outputs = model(**test_tokens)
            predictions = outputs.logits.argmax(dim=-1).cpu().numpy()
        
        accuracy = accuracy_score(y_test, predictions)
        report = classification_report(y_test, predictions, output_dict=True)
        
        results[name] = {
            'accuracy': accuracy,
            'precision': report['weighted avg']['precision'],
            'recall': report['weighted avg']['recall'],
            'f1': report['weighted avg']['f1-score']
        }
        
        print(f"  Accuracy: {accuracy:.4f}")
    
    return results


def main():
    """Main function demonstrating transfer learning with TabGPT."""
    print("TabGPT Transfer Learning Example")
    print("=" * 40)
    
    # 1. Create datasets
    print("1. Creating datasets...")
    source_df = create_source_dataset(n_samples=5000)  # Smaller for demo
    target_df = create_target_dataset(n_samples=300)   # Small target dataset
    
    print(f"Source dataset: {source_df.shape}")
    print(f"Target dataset: {target_df.shape}")
    print()
    
    # 2. Pre-train on source data
    print("2. Pre-training on source dataset...")
    pretrained_model, tokenizer, source_normalizer = pretrain_on_source_data(source_df)
    print("Pre-training completed!")
    print()
    
    # 3. Prepare target dataset
    print("3. Preparing target dataset...")
    X_target = target_df.drop('target', axis=1)
    y_target = target_df['target']
    
    # Split target data (small dataset, so careful splitting)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_target, y_target, test_size=0.4, random_state=42, stratify=y_target
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    print(f"Target train: {len(X_train)} samples")
    print(f"Target val: {len(X_val)} samples")
    print(f"Target test: {len(X_test)} samples")
    
    # Preprocess target data
    target_normalizer = RobustNormalizer()
    X_train_processed, _ = target_normalizer.fit_transform(X_train)
    X_val_processed, _ = target_normalizer.transform(X_val)
    X_test_processed, _ = target_normalizer.transform(X_test)
    print()
    
    # 4. Train baseline model (from scratch)
    print("4. Training baseline model from scratch...")
    baseline_model, baseline_trainer = train_baseline_model(
        X_train_processed, y_train, X_val_processed, y_val, tokenizer
    )
    baseline_results = baseline_trainer.evaluate()
    print(f"Baseline validation accuracy: {baseline_results['eval_accuracy']:.4f}")
    print()
    
    # 5. Fine-tune pre-trained model with full parameters
    print("5. Fine-tuning pre-trained model (full parameters)...")
    
    # Create a copy of the pre-trained model for full fine-tuning
    full_finetune_model = TabGPTForSequenceClassification(num_labels=2)
    full_finetune_model.load_state_dict(pretrained_model.state_dict())
    
    train_dataset = tokenizer.create_dataset(X_train_processed, y_train)
    val_dataset = tokenizer.create_dataset(X_val_processed, y_val)
    
    config = FineTuningConfig(
        task_type="classification",
        num_labels=2,
        learning_rate=1e-5,  # Lower LR for full fine-tuning
        num_epochs=3,
        batch_size=16,  # Smaller batch for small dataset
        eval_steps=20,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="eval_accuracy",
        greater_is_better=True
    )
    
    full_trainer = TabGPTFineTuningTrainer(
        model=full_finetune_model,
        config=config,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=lambda x: {"accuracy": accuracy_score(x.label_ids, x.predictions.argmax(-1))}
    )
    
    full_trainer.train()
    full_results = full_trainer.evaluate()
    print(f"Full fine-tuning validation accuracy: {full_results['eval_accuracy']:.4f}")
    print()
    
    # 6. Fine-tune with LoRA
    print("6. Fine-tuning with LoRA...")
    
    # Create another copy for LoRA
    lora_base_model = TabGPTForSequenceClassification(num_labels=2)
    lora_base_model.load_state_dict(pretrained_model.state_dict())
    
    lora_model, lora_trainer = fine_tune_with_lora(
        lora_base_model, X_train_processed, y_train, X_val_processed, y_val, tokenizer
    )
    lora_results = lora_trainer.evaluate()
    print(f"LoRA fine-tuning validation accuracy: {lora_results['eval_accuracy']:.4f}")
    print()
    
    # 7. Evaluate all models on test set
    print("7. Evaluating all models on test set...")
    models = {
        'Baseline (from scratch)': baseline_model,
        'Full fine-tuning': full_finetune_model,
        'LoRA fine-tuning': lora_model
    }
    
    test_results = evaluate_models(models, X_test_processed, y_test, tokenizer)
    print()
    
    # 8. Compare results
    print("8. Transfer Learning Results Summary:")
    print("=" * 60)
    print(f"{'Method':<25} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10}")
    print("-" * 60)
    
    for method, metrics in test_results.items():
        print(f"{method:<25} {metrics['accuracy']:<10.4f} {metrics['precision']:<10.4f} "
              f"{metrics['recall']:<10.4f} {metrics['f1']:<10.4f}")
    print()
    
    # 9. Analyze improvements
    print("9. Transfer Learning Analysis:")
    baseline_acc = test_results['Baseline (from scratch)']['accuracy']
    full_acc = test_results['Full fine-tuning']['accuracy']
    lora_acc = test_results['LoRA fine-tuning']['accuracy']
    
    full_improvement = ((full_acc - baseline_acc) / baseline_acc) * 100
    lora_improvement = ((lora_acc - baseline_acc) / baseline_acc) * 100
    
    print(f"Full fine-tuning improvement: {full_improvement:+.1f}%")
    print(f"LoRA fine-tuning improvement: {lora_improvement:+.1f}%")
    
    if full_acc > baseline_acc:
        print("✓ Transfer learning shows positive benefit!")
    else:
        print("⚠ Transfer learning did not improve performance (may need more data or different approach)")
    
    if lora_acc >= full_acc * 0.95:  # Within 95% of full fine-tuning
        print("✓ LoRA achieves comparable performance with much fewer parameters!")
    print()
    
    # 10. Sample efficiency analysis
    print("10. Sample Efficiency Analysis:")
    sample_sizes = [50, 100, 150, 200, len(X_train)]
    
    print("Training with different sample sizes...")
    efficiency_results = {}
    
    for size in sample_sizes:
        if size > len(X_train):
            continue
            
        print(f"  Training with {size} samples...")
        
        # Sample subset
        X_subset = X_train_processed.iloc[:size]
        y_subset = y_train.iloc[:size]
        
        # Train baseline
        baseline_subset = TabGPTForSequenceClassification(num_labels=2)
        subset_dataset = tokenizer.create_dataset(X_subset, y_subset)
        
        config_subset = FineTuningConfig(
            task_type="classification",
            num_labels=2,
            learning_rate=5e-5,
            num_epochs=2,  # Fewer epochs for speed
            batch_size=min(16, size),
            eval_steps=max(5, size//4),
            logging_steps=max(5, size//4)
        )
        
        trainer_subset = TabGPTFineTuningTrainer(
            model=baseline_subset,
            config=config_subset,
            train_dataset=subset_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
            compute_metrics=lambda x: {"accuracy": accuracy_score(x.label_ids, x.predictions.argmax(-1))}
        )
        
        trainer_subset.train()
        
        # Evaluate
        baseline_subset.eval()
        with torch.no_grad():
            test_tokens = tokenizer.encode_batch(X_test_processed)
            outputs = baseline_subset(**test_tokens)
            predictions = outputs.logits.argmax(dim=-1).cpu().numpy()
        
        subset_acc = accuracy_score(y_test, predictions)
        efficiency_results[size] = subset_acc
    
    print("\\nSample Efficiency Results:")
    print(f"{'Sample Size':<12} {'Accuracy':<10}")
    print("-" * 22)
    for size, acc in efficiency_results.items():
        print(f"{size:<12} {acc:<10.4f}")
    print()
    
    # 11. Save models
    print("11. Saving models...")
    baseline_trainer.save_model("./baseline_model")
    full_trainer.save_model("./full_finetune_model")
    lora_trainer.save_model("./lora_model")
    
    print("Models saved:")
    print("  - ./baseline_model")
    print("  - ./full_finetune_model")
    print("  - ./lora_model")
    print()
    
    print("Transfer learning example completed!")
    print("Key takeaways:")
    print("1. Pre-training on large datasets can improve performance on small target tasks")
    print("2. LoRA provides parameter-efficient fine-tuning with competitive performance")
    print("3. Transfer learning is especially beneficial when target data is limited")


if __name__ == "__main__":
    main()