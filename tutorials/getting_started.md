# Getting Started with TabGPT

Welcome to TabGPT! This tutorial will guide you through your first steps with the universal tabular foundation model. By the end of this tutorial, you'll know how to load data, train a model, and make predictions.

## What is TabGPT?

TabGPT is a transformer-based foundation model specifically designed for tabular data. Unlike traditional machine learning models that require extensive feature engineering, TabGPT can automatically learn representations from raw tabular data and achieve state-of-the-art performance across various tasks.

### Key Benefits

- **Universal**: One model architecture for classification, regression, and specialized tasks
- **Automatic Feature Learning**: No manual feature engineering required
- **Transfer Learning**: Pre-trained models can be fine-tuned for your specific tasks
- **Robust**: Handles missing values, outliers, and mixed data types automatically
- **Efficient**: Parameter-efficient fine-tuning with LoRA adapters

## Installation

First, let's install TabGPT and its dependencies:

```bash
# Clone the repository
git clone https://github.com/your-org/tabgpt.git
cd tabgpt

# Create a virtual environment
python -m venv tabgpt_env
source tabgpt_env/bin/activate  # On Windows: tabgpt_env\\Scripts\\activate

# Install TabGPT
pip install -e .

# Install optional dependencies for examples
pip install matplotlib seaborn jupyter
```

## Your First TabGPT Model

Let's start with a simple classification example using synthetic data.

### Step 1: Import Libraries

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# TabGPT imports
from tabgpt import TabGPTForSequenceClassification, TabGPTTokenizer
from tabgpt.utils import RobustNormalizer
from tabgpt.fine_tuning import TabGPTFineTuningTrainer, FineTuningConfig
```

### Step 2: Create Sample Data

```python
# Create a simple dataset
np.random.seed(42)
n_samples = 1000

# Generate features
data = {
    'age': np.random.normal(35, 10, n_samples),
    'income': np.random.lognormal(10, 0.5, n_samples),
    'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples),
    'experience': np.random.exponential(5, n_samples),
    'city_size': np.random.choice(['Small', 'Medium', 'Large'], n_samples)
}

# Create target based on features (with some logic)
target_prob = (
    0.3 +  # Base probability
    0.2 * (data['education'] == 'PhD').astype(int) +
    0.1 * (data['education'] == 'Master').astype(int) +
    0.15 * (data['city_size'] == 'Large').astype(int) +
    0.1 * (data['income'] > np.median(data['income'])).astype(int)
)
target_prob = np.clip(target_prob, 0, 1)
data['target'] = np.random.binomial(1, target_prob, n_samples)

# Create DataFrame
df = pd.DataFrame(data)
print(f"Dataset shape: {df.shape}")
print(df.head())
```

### Step 3: Prepare Data

```python
# Separate features and target
X = df.drop('target', axis=1)
y = df['target']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
```

### Step 4: Preprocess Data (Optional)

TabGPT handles preprocessing automatically, but you can use robust preprocessing for better results:

```python
# Apply robust preprocessing
normalizer = RobustNormalizer(
    numerical_strategy="robust",
    categorical_strategy="frequency",
    missing_strategy="median"
)

X_train_processed, _ = normalizer.fit_transform(X_train)
X_test_processed, _ = normalizer.transform(X_test)

print("Data preprocessing completed!")
```

### Step 5: Initialize Model and Tokenizer

```python
# Initialize tokenizer
tokenizer = TabGPTTokenizer(
    vocab_size=5000,  # Smaller vocab for simple data
    max_length=128    # Shorter sequences for tabular data
)

# Initialize model for binary classification
model = TabGPTForSequenceClassification(num_labels=2)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
```

### Step 6: Prepare Datasets

```python
# Create PyTorch datasets
train_dataset = tokenizer.create_dataset(X_train_processed, y_train)
test_dataset = tokenizer.create_dataset(X_test_processed, y_test)

print(f"Training dataset size: {len(train_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")
```

### Step 7: Configure Training

```python
# Configure fine-tuning
config = FineTuningConfig(
    task_type="classification",
    num_labels=2,
    learning_rate=5e-5,
    num_epochs=3,
    batch_size=32,
    eval_steps=50,
    logging_steps=25,
    load_best_model_at_end=True,
    metric_for_best_model="eval_accuracy",
    greater_is_better=True
)

print("Training configuration set!")
```

### Step 8: Train the Model

```python
# Create trainer
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {'accuracy': accuracy_score(labels, predictions)}

trainer = TabGPTFineTuningTrainer(
    model=model,
    config=config,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# Train the model
print("Starting training...")
trainer.train()
print("Training completed!")
```

### Step 9: Evaluate the Model

```python
# Evaluate on test set
eval_results = trainer.evaluate()
print("Evaluation Results:")
for metric, value in eval_results.items():
    print(f"  {metric}: {value:.4f}")
```

### Step 10: Make Predictions

```python
# Make predictions on new data
model.eval()
test_tokens = tokenizer.encode_batch(X_test_processed)

with torch.no_grad():
    outputs = model(**test_tokens)
    predictions = outputs.logits.argmax(dim=-1).cpu().numpy()
    probabilities = torch.softmax(outputs.logits, dim=-1).cpu().numpy()

# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)
print(f"Test Accuracy: {accuracy:.4f}")

# Show some example predictions
print("\\nExample Predictions:")
for i in range(5):
    actual = y_test.iloc[i]
    pred = predictions[i]
    prob = probabilities[i][1]  # Probability of class 1
    print(f"Actual: {actual}, Predicted: {pred}, Probability: {prob:.3f}")
```

### Step 11: Save the Model

```python
# Save the trained model
model_path = "./my_first_tabgpt_model"
trainer.save_model(model_path)
print(f"Model saved to: {model_path}")

# Later, you can load it like this:
# loaded_model = TabGPTForSequenceClassification.from_pretrained(model_path)
# loaded_tokenizer = TabGPTTokenizer.from_pretrained(model_path)
```

## Complete Example Script

Here's the complete script you can run:

```python
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from tabgpt import TabGPTForSequenceClassification, TabGPTTokenizer
from tabgpt.utils import RobustNormalizer
from tabgpt.fine_tuning import TabGPTFineTuningTrainer, FineTuningConfig

def main():
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'age': np.random.normal(35, 10, n_samples),
        'income': np.random.lognormal(10, 0.5, n_samples),
        'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples),
        'experience': np.random.exponential(5, n_samples),
        'city_size': np.random.choice(['Small', 'Medium', 'Large'], n_samples)
    }
    
    target_prob = (
        0.3 + 
        0.2 * (np.array(data['education']) == 'PhD').astype(int) +
        0.1 * (np.array(data['education']) == 'Master').astype(int) +
        0.15 * (np.array(data['city_size']) == 'Large').astype(int) +
        0.1 * (data['income'] > np.median(data['income'])).astype(int)
    )
    target_prob = np.clip(target_prob, 0, 1)
    data['target'] = np.random.binomial(1, target_prob, n_samples)
    
    df = pd.DataFrame(data)
    
    # Prepare data
    X = df.drop('target', axis=1)
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Preprocess
    normalizer = RobustNormalizer()
    X_train_processed, _ = normalizer.fit_transform(X_train)
    X_test_processed, _ = normalizer.transform(X_test)
    
    # Initialize model
    tokenizer = TabGPTTokenizer(vocab_size=5000, max_length=128)
    model = TabGPTForSequenceClassification(num_labels=2)
    
    # Prepare datasets
    train_dataset = tokenizer.create_dataset(X_train_processed, y_train)
    test_dataset = tokenizer.create_dataset(X_test_processed, y_test)
    
    # Configure and train
    config = FineTuningConfig(
        task_type="classification",
        num_labels=2,
        learning_rate=5e-5,
        num_epochs=3,
        batch_size=32
    )
    
    trainer = TabGPTFineTuningTrainer(
        model=model,
        config=config,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        compute_metrics=lambda x: {'accuracy': accuracy_score(x.label_ids, x.predictions.argmax(-1))}
    )
    
    trainer.train()
    
    # Evaluate
    results = trainer.evaluate()
    print(f"Final accuracy: {results['eval_accuracy']:.4f}")
    
    # Save model
    trainer.save_model("./my_tabgpt_model")
    print("Model saved successfully!")

if __name__ == "__main__":
    main()
```

## Next Steps

Congratulations! You've successfully trained your first TabGPT model. Here are some next steps to explore:

### 1. Try Different Tasks
- **Regression**: Use `TabGPTForRegression` for continuous targets
- **Multi-class**: Increase `num_labels` for multi-class classification
- **Multi-label**: Use `MultiLabelClassificationHead` for multi-label tasks

### 2. Use Pre-trained Models
```python
# Load a pre-trained model
model = TabGPTForSequenceClassification.from_pretrained('tabgpt-base')
tokenizer = TabGPTTokenizer.from_pretrained('tabgpt-base')
```

### 3. Efficient Fine-tuning with LoRA
```python
from tabgpt.adapters import LoRAConfig, apply_lora_to_model

# Apply LoRA for efficient fine-tuning
lora_config = LoRAConfig(r=8, alpha=16)
model = apply_lora_to_model(model, lora_config)
```

### 4. Advanced Features
- **Uncertainty Estimation**: Use `RegressionHead` with `estimate_uncertainty=True`
- **Survival Analysis**: Use `SurvivalHead` for time-to-event modeling
- **Anomaly Detection**: Use `AnomalyDetectionHead` for outlier detection

### 5. Evaluation and Benchmarking
```python
from tabgpt.evaluation import ClassificationBenchmark, create_baseline_models

# Compare with traditional ML models
baselines = create_baseline_models("classification")
benchmark = ClassificationBenchmark("my_dataset", load_data_func)
results = benchmark.run_benchmark({"TabGPT": model, **baselines})
```

## Common Issues and Solutions

### Out of Memory
- Reduce `batch_size` in `FineTuningConfig`
- Use gradient accumulation: `gradient_accumulation_steps=4`
- Enable mixed precision: `fp16=True`

### Poor Performance
- Check data quality with `DataValidator`
- Try different learning rates: `[1e-5, 5e-5, 1e-4]`
- Increase model size or training epochs
- Use pre-trained models for transfer learning

### Slow Training
- Increase `batch_size` if memory allows
- Use multiple workers: `dataloader_num_workers=4`
- Enable mixed precision: `fp16=True`

## Resources

- **Examples**: Check the `examples/` directory for more complex scenarios
- **API Reference**: See `docs/api_reference.md` for detailed documentation
- **User Guide**: Read `docs/user_guide.md` for comprehensive usage information
- **GitHub Issues**: Report bugs or ask questions on GitHub

Happy modeling with TabGPT! 🚀