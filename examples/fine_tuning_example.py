"""
Example: Fine-tuning TabGPT with LoRA adapters

This example demonstrates how to fine-tune a pre-trained TabGPT model
on a downstream classification task using LoRA (Low-Rank Adaptation)
for efficient parameter updates.
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tabgpt.models import TabGPTForSequenceClassification
from tabgpt.tokenizers import TabGPTTokenizer
from tabgpt.adapters import LoRAConfig, apply_lora_to_model, get_lora_model_info
from tabgpt.fine_tuning import (
    TabGPTFineTuningTrainer,
    FineTuningConfig,
    prepare_classification_data,
    create_default_callbacks
)


def create_synthetic_dataset(n_samples=1000, n_features=20, n_classes=3, random_state=42):
    """Create a synthetic tabular dataset for demonstration."""
    print(f"Creating synthetic dataset with {n_samples} samples, {n_features} features, {n_classes} classes")
    
    # Generate synthetic data
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_features // 2,
        n_redundant=n_features // 4,
        n_classes=n_classes,
        n_clusters_per_class=1,
        random_state=random_state
    )
    
    # Create feature names
    feature_names = [f'feature_{i}' for i in range(n_features)]
    
    # Create DataFrame
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    # Add some categorical features
    df['category_1'] = np.random.choice(['A', 'B', 'C'], size=n_samples)
    df['category_2'] = np.random.choice(['X', 'Y'], size=n_samples)
    
    print(f"Dataset created with shape: {df.shape}")
    print(f"Target distribution: {df['target'].value_counts().to_dict()}")
    
    return df


def demonstrate_lora_efficiency():
    """Demonstrate LoRA parameter efficiency."""
    print("\n" + "="*50)
    print("LoRA Parameter Efficiency Demonstration")
    print("="*50)
    
    # Create a simple model for demonstration
    model = nn.Sequential(
        nn.Linear(100, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 3)
    )
    
    # Count original parameters
    original_params = sum(p.numel() for p in model.parameters())
    print(f"Original model parameters: {original_params:,}")
    
    # Apply LoRA with different ranks
    for rank in [4, 8, 16]:
        lora_config = LoRAConfig(
            r=rank,
            alpha=rank * 2,
            target_modules=["0", "2", "4", "6"]  # Target all linear layers
        )
        
        # Apply LoRA (create a copy for each test)
        import copy
        lora_model = apply_lora_to_model(copy.deepcopy(model), lora_config)
        
        # Get LoRA info
        info = get_lora_model_info(lora_model)
        
        print(f"\nLoRA rank {rank}:")
        print(f"  Total parameters: {info['total_parameters']:,}")
        print(f"  LoRA parameters: {info['lora_parameters']:,}")
        print(f"  Parameter efficiency: {info['parameter_efficiency']:.2%}")
        print(f"  LoRA modules: {len(info['lora_modules'])}")


def fine_tune_example():
    """Complete fine-tuning example."""
    print("\n" + "="*50)
    print("TabGPT Fine-tuning Example")
    print("="*50)
    
    # Create synthetic dataset
    df = create_synthetic_dataset(n_samples=1000, n_features=15, n_classes=3)
    
    # Split data
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['target'])
    train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42, stratify=train_df['target'])
    
    print(f"\nData splits:")
    print(f"  Train: {len(train_df)} samples")
    print(f"  Validation: {len(val_df)} samples")
    print(f"  Test: {len(test_df)} samples")
    
    # For this example, we'll create a mock model and tokenizer
    # In practice, you would load a pre-trained TabGPT model
    print("\nNote: This example uses mock components for demonstration.")
    print("In practice, you would load a pre-trained TabGPT model:")
    print("  model = TabGPTForSequenceClassification.from_pretrained('path/to/pretrained')")
    print("  tokenizer = TabGPTTokenizer.from_pretrained('path/to/pretrained')")
    
    # Create mock model for demonstration
    class MockTabGPTModel(nn.Module):
        def __init__(self, input_dim=128, num_classes=3):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(128, 64)
            )
            self.classifier = nn.Linear(64, num_classes)
            
        def forward(self, input_ids, attention_mask=None, labels=None):
            # Mock forward pass
            batch_size = input_ids.size(0)
            # Simulate feature extraction
            features = torch.randn(batch_size, 64)
            logits = self.classifier(features)
            
            loss = None
            if labels is not None:
                loss_fn = nn.CrossEntropyLoss()
                loss = loss_fn(logits, labels)
                
            return type('Output', (), {
                'loss': loss,
                'logits': logits
            })()
        
        def save_pretrained(self, path):
            os.makedirs(path, exist_ok=True)
            torch.save(self.state_dict(), os.path.join(path, 'pytorch_model.bin'))
    
    # Create mock tokenizer
    class MockTokenizer:
        def encode_batch(self, df):
            batch_size = len(df)
            seq_length = 32
            return {
                'input_ids': torch.randint(0, 1000, (batch_size, seq_length)),
                'attention_mask': torch.ones(batch_size, seq_length)
            }
    
    model = MockTabGPTModel(num_classes=3)
    tokenizer = MockTokenizer()
    
    print(f"\nModel created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Configure LoRA
    lora_config = LoRAConfig(
        r=8,
        alpha=16,
        dropout=0.1,
        target_modules=["encoder.0", "encoder.3", "classifier"]
    )
    
    # Apply LoRA
    print("\nApplying LoRA adapters...")
    lora_model = apply_lora_to_model(model, lora_config)
    
    # Show parameter efficiency
    info = get_lora_model_info(lora_model)
    print(f"LoRA applied:")
    print(f"  Total parameters: {info['total_parameters']:,}")
    print(f"  Trainable parameters: {info['lora_parameters']:,}")
    print(f"  Parameter efficiency: {info['parameter_efficiency']:.2%}")
    
    # Prepare datasets (mock implementation)
    class MockDataset:
        def __init__(self, df, target_col, tokenizer):
            self.df = df
            self.target_col = target_col
            self.tokenizer = tokenizer
            
        def __len__(self):
            return len(self.df)
            
        def __getitem__(self, idx):
            row = self.df.iloc[idx]
            target = row[self.target_col]
            
            # Mock tokenization
            return {
                'input_ids': torch.randint(0, 1000, (32,)),
                'attention_mask': torch.ones(32),
                'labels': torch.tensor(target, dtype=torch.long)
            }
    
    train_dataset = MockDataset(train_df, 'target', tokenizer)
    val_dataset = MockDataset(val_df, 'target', tokenizer)
    
    print(f"\nDatasets prepared:")
    print(f"  Train dataset: {len(train_dataset)} samples")
    print(f"  Validation dataset: {len(val_dataset)} samples")
    
    # Configure training
    training_config = FineTuningConfig(
        task_type="classification",
        num_labels=3,
        learning_rate=5e-4,  # Higher LR for LoRA
        num_epochs=2,  # Short for demo
        batch_size=16
    )
    
    # Create callbacks
    callbacks = create_default_callbacks(
        early_stopping_patience=3,
        save_best_model=True,
        metric_for_best_model="eval_accuracy",
        greater_is_better=True
    )
    
    print(f"\nTraining configuration:")
    print(f"  Epochs: {training_config.num_epochs}")
    print(f"  Batch size: {training_config.batch_size}")
    print(f"  Learning rate: {training_config.learning_rate}")
    print(f"  Task type: {training_config.task_type}")
    
    # Note: In a real scenario, you would create and run the trainer
    print(f"\nTraining setup complete!")
    print("In a real scenario, you would now run:")
    print("  trainer = TabGPTFineTuningTrainer(...)")
    print("  trainer.train()")
    
    # Demonstrate saving LoRA weights
    print(f"\nSaving LoRA weights...")
    from tabgpt.adapters import save_lora_weights
    
    os.makedirs("./lora_weights", exist_ok=True)
    save_lora_weights(lora_model, "./lora_weights", lora_config)
    
    print("LoRA weights saved to ./lora_weights")
    
    # Clean up
    import shutil
    if os.path.exists("./lora_weights"):
        shutil.rmtree("./lora_weights")
    if os.path.exists("./fine_tuned_model"):
        shutil.rmtree("./fine_tuned_model")


def main():
    """Run all examples."""
    print("TabGPT Fine-tuning Examples")
    print("="*50)
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Run examples
    demonstrate_lora_efficiency()
    fine_tune_example()
    
    print("\n" + "="*50)
    print("Examples completed successfully!")
    print("="*50)


if __name__ == "__main__":
    main()