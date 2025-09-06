# TabGPT Fine-tuning Guide

This guide explains how to fine-tune pre-trained TabGPT models on downstream tasks using efficient adaptation techniques like LoRA (Low-Rank Adaptation).

## Overview

TabGPT supports efficient fine-tuning through adapter modules that add a small number of trainable parameters while keeping the pre-trained model frozen. This approach provides several benefits:

- **Parameter Efficiency**: Only 1-10% of parameters need to be trained
- **Memory Efficiency**: Reduced memory usage during training
- **Fast Training**: Faster convergence due to fewer parameters
- **Modularity**: Easy to switch between different task adaptations

## Quick Start

### 1. Basic Fine-tuning with LoRA

```python
from tabgpt.models import TabGPTForSequenceClassification
from tabgpt.tokenizers import TabGPTTokenizer
from tabgpt.adapters import LoRAConfig, apply_lora_to_model
from tabgpt.fine_tuning import FineTuningConfig, TabGPTFineTuningTrainer

# Load pre-trained model
model = TabGPTForSequenceClassification.from_pretrained("path/to/pretrained")
tokenizer = TabGPTTokenizer.from_pretrained("path/to/pretrained")

# Configure LoRA
lora_config = LoRAConfig(
    r=8,                    # Rank of adaptation
    alpha=16,               # LoRA scaling parameter
    dropout=0.1,            # Dropout probability
    target_modules=["query", "key", "value", "dense"]
)

# Apply LoRA to model
model = apply_lora_to_model(model, lora_config)

# Configure training
config = FineTuningConfig(
    task_type="classification",
    num_labels=3,
    learning_rate=5e-4,
    num_epochs=3,
    batch_size=32
)

# Create trainer and train
trainer = TabGPTFineTuningTrainer(
    model=model,
    config=config,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer
)

trainer.train()
```

### 2. Using the Fine-tuning Script

For convenience, you can use the provided fine-tuning script:

```bash
python scripts/fine_tune_tabgpt.py \
    --model_name_or_path path/to/pretrained \
    --task_type classification \
    --train_file data/train.csv \
    --validation_file data/val.csv \
    --target_column target \
    --output_dir ./fine_tuned_model \
    --use_lora \
    --lora_r 8 \
    --lora_alpha 16 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 32 \
    --learning_rate 5e-4
```

## LoRA Configuration

### Key Parameters

- **r (rank)**: Controls the bottleneck dimension. Higher values = more parameters but potentially better performance
  - Typical values: 4, 8, 16, 32
  - Start with 8 for most tasks

- **alpha**: Scaling parameter that controls the magnitude of LoRA updates
  - Typical values: 16, 32 (often 2x the rank)
  - Higher values = stronger adaptation

- **dropout**: Dropout probability for LoRA layers
  - Typical values: 0.1, 0.2
  - Helps prevent overfitting

- **target_modules**: Which modules to apply LoRA to
  - Common choices: `["query", "key", "value"]` (attention only)
  - Or: `["query", "key", "value", "dense"]` (attention + FFN)

### Parameter Efficiency Examples

| Rank | Parameters | Efficiency | Use Case |
|------|------------|------------|----------|
| 4    | ~1-3%      | Highest    | Small datasets, simple tasks |
| 8    | ~3-6%      | High       | Most tasks (recommended) |
| 16   | ~6-12%     | Medium     | Complex tasks, large datasets |
| 32   | ~12-25%    | Lower      | Very complex tasks |

## Task Types

### Classification

```python
config = FineTuningConfig(
    task_type="classification",
    num_labels=3,  # Number of classes
    learning_rate=5e-4
)
```

### Regression

```python
config = FineTuningConfig(
    task_type="regression",
    output_dim=1,  # Number of target values
    learning_rate=1e-4
)
```

## Data Preparation

### CSV Format

Your data should be in CSV format with:
- Feature columns (numerical, categorical, text)
- Target column (labels for classification, values for regression)

Example:
```csv
feature1,feature2,category,target
1.5,2.3,A,0
2.1,1.8,B,1
0.9,3.2,A,0
```

### Data Loading

```python
from tabgpt.fine_tuning import prepare_classification_data

# Prepare dataset
dataset = prepare_classification_data(
    df=train_df,
    target_column="target",
    tokenizer=tokenizer
)
```

## Training Configuration

### Basic Configuration

```python
config = FineTuningConfig(
    task_type="classification",
    learning_rate=5e-4,      # Higher LR for LoRA
    num_epochs=3,
    batch_size=32,
    gradient_accumulation_steps=1
)
```

### Advanced Configuration

```python
config = FineTuningConfig(
    task_type="classification",
    num_labels=5,
    
    # LoRA settings
    use_adapters=True,
    adapter_type="lora",
    lora_config=lora_config,
    
    # Training settings
    learning_rate=5e-4,
    num_epochs=5,
    batch_size=16,
    gradient_accumulation_steps=2,
    
    # Optimization
    weight_decay=0.01,
    warmup_ratio=0.1,
    
    # Evaluation
    eval_strategy="steps",
    eval_steps=100,
    
    # Saving
    save_strategy="steps",
    save_steps=500,
    save_total_limit=3
)
```

## Callbacks and Monitoring

### Early Stopping

```python
from tabgpt.fine_tuning import EarlyStoppingCallback

callback = EarlyStoppingCallback(
    early_stopping_patience=3,
    metric_for_best_model="eval_accuracy",
    greater_is_better=True
)
```

### Model Checkpointing

```python
from tabgpt.fine_tuning import ModelCheckpointCallback

callback = ModelCheckpointCallback(
    save_best_model=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False
)
```

### Progress Logging

```python
from tabgpt.fine_tuning import ProgressCallback

callback = ProgressCallback(
    log_every_n_steps=50
)
```

## Saving and Loading

### Save LoRA Weights

```python
from tabgpt.adapters import save_lora_weights

save_lora_weights(
    model=model,
    save_directory="./lora_weights",
    config=lora_config
)
```

### Load LoRA Weights

```python
from tabgpt.adapters import load_lora_weights

config = load_lora_weights(
    model=model,
    load_directory="./lora_weights"
)
```

### Save Full Model

```python
# Save model with merged weights
model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")
```

## Best Practices

### 1. Learning Rate

- Use higher learning rates for LoRA (1e-4 to 5e-4)
- Start with 5e-4 and adjust based on convergence
- Use learning rate scheduling for longer training

### 2. Batch Size

- Start with smaller batches (16-32) for LoRA
- Use gradient accumulation for effective larger batches
- Monitor memory usage

### 3. Regularization

- Use dropout in LoRA layers (0.1-0.2)
- Apply weight decay (0.01-0.1)
- Use early stopping to prevent overfitting

### 4. Target Modules

- Start with attention modules: `["query", "key", "value"]`
- Add FFN modules if needed: `["dense"]`
- Experiment with different combinations

### 5. Rank Selection

- Start with rank 8 for most tasks
- Increase rank for complex tasks or large datasets
- Monitor parameter efficiency vs. performance trade-off

## Troubleshooting

### Common Issues

1. **Out of Memory**
   - Reduce batch size
   - Use gradient accumulation
   - Reduce LoRA rank

2. **Poor Convergence**
   - Increase learning rate
   - Increase LoRA rank
   - Add more target modules

3. **Overfitting**
   - Increase dropout
   - Use early stopping
   - Reduce LoRA rank

4. **Slow Training**
   - Increase batch size
   - Use mixed precision (fp16)
   - Optimize data loading

### Performance Tips

- Use mixed precision training (`fp16=True`)
- Enable gradient checkpointing for large models
- Use multiple GPUs with data parallelism
- Optimize data loading with multiple workers

## Examples

See the `examples/` directory for complete examples:

- `fine_tuning_example.py`: Basic fine-tuning with LoRA
- `classification_example.py`: Classification task example
- `regression_example.py`: Regression task example

## API Reference

For detailed API documentation, see:

- `tabgpt.adapters`: LoRA and adapter utilities
- `tabgpt.fine_tuning`: Fine-tuning trainer and configuration
- `tabgpt.fine_tuning.callbacks`: Training callbacks