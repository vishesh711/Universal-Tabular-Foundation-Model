# TabGPT Examples

This directory contains practical examples demonstrating how to use TabGPT for various tabular data tasks. Each example is self-contained and includes detailed explanations.

## 📁 Example Files

### 🎯 Basic Examples

#### `basic_classification.py`
**What it demonstrates:**
- Binary classification on synthetic Titanic-like dataset
- Data validation and preprocessing
- Model training and evaluation
- Basic prediction workflow

**Key concepts:**
- `TabGPTForSequenceClassification`
- `TabGPTTokenizer`
- `RobustNormalizer`
- `TabGPTFineTuningTrainer`

**Run time:** ~5-10 minutes

```bash
python examples/basic_classification.py
```

#### `regression_example.py`
**What it demonstrates:**
- Regression with uncertainty estimation
- California Housing dataset
- Advanced preprocessing
- Prediction visualization

**Key concepts:**
- `TabGPTForRegression`
- `RegressionHead` with uncertainty
- Performance metrics (MSE, RMSE, R²)
- Error analysis by price range

**Run time:** ~10-15 minutes

```bash
python examples/regression_example.py
```

### 🚀 Advanced Examples

#### `transfer_learning_example.py`
**What it demonstrates:**
- Pre-training on large source dataset
- Fine-tuning on small target dataset
- LoRA parameter-efficient adaptation
- Performance comparison with baselines

**Key concepts:**
- Transfer learning workflow
- `LoRAConfig` and `apply_lora_to_model`
- Sample efficiency analysis
- Statistical significance testing

**Run time:** ~20-30 minutes

```bash
python examples/transfer_learning_example.py
```

#### `comprehensive_example.py`
**What it demonstrates:**
- Complete TabGPT workflow
- Multiple task types (classification, regression, survival)
- Benchmarking against traditional ML
- Cross-validation evaluation
- Results visualization

**Key concepts:**
- End-to-end pipeline
- `ClassificationBenchmark`
- `CrossValidationEvaluator`
- `SurvivalHead` setup
- Performance visualization

**Run time:** ~30-45 minutes

```bash
python examples/comprehensive_example.py
```

## 🛠️ Prerequisites

### Required Dependencies
```bash
pip install torch>=2.0.0
pip install transformers>=4.20.0
pip install pandas>=1.5.0
pip install numpy>=1.21.0
pip install scikit-learn>=1.1.0
```

### Optional Dependencies (for full functionality)
```bash
pip install matplotlib>=3.5.0      # For visualizations
pip install seaborn>=0.11.0        # For advanced plots
pip install xgboost>=1.6.0         # For baseline comparisons
pip install lightgbm>=3.3.0        # For baseline comparisons
pip install jupyter>=1.0.0         # For notebook examples
```

### Install All Dependencies
```bash
pip install -e ".[examples]"
```

## 📊 Example Datasets

The examples use various datasets to demonstrate different scenarios:

### Synthetic Datasets
- **Titanic-like**: Binary classification with mixed data types
- **California Housing**: Regression with real-world characteristics
- **Financial Risk**: Multi-class classification with categorical features
- **Medical Survival**: Time-to-event analysis with censoring

### Real Datasets (when available)
- **OpenML datasets**: Standardized benchmarks
- **Scikit-learn datasets**: Built-in datasets for quick testing
- **Custom datasets**: Domain-specific examples

## 🎯 Learning Path

### Beginner (New to TabGPT)
1. Start with `basic_classification.py`
2. Try `regression_example.py`
3. Read the [Getting Started Tutorial](../tutorials/getting_started.md)

### Intermediate (Familiar with ML)
1. Run `transfer_learning_example.py`
2. Explore different model configurations
3. Try your own datasets with the basic examples

### Advanced (ML Practitioners)
1. Study `comprehensive_example.py`
2. Implement custom task heads
3. Develop domain-specific applications
4. Contribute new examples

## 🔧 Customization Guide

### Adapting Examples to Your Data

#### 1. Data Loading
Replace the synthetic data generation with your data loading:

```python
# Instead of synthetic data
df = create_synthetic_data()

# Use your data
df = pd.read_csv('your_data.csv')
# or
df = pd.read_parquet('your_data.parquet')
# or
df = load_from_database()
```

#### 2. Task Configuration
Modify the task type and parameters:

```python
# For your classification task
config = FineTuningConfig(
    task_type="classification",
    num_labels=YOUR_NUM_CLASSES,  # Change this
    learning_rate=5e-5,
    num_epochs=5,  # Adjust based on your data size
    batch_size=32  # Adjust based on your memory
)

# For your regression task
config = FineTuningConfig(
    task_type="regression",
    output_dim=YOUR_OUTPUT_DIM,  # Change this
    learning_rate=3e-5,
    num_epochs=10
)
```

#### 3. Preprocessing
Adjust preprocessing for your data characteristics:

```python
normalizer = RobustNormalizer(
    numerical_strategy="robust",     # or "standard", "minmax"
    categorical_strategy="frequency", # or "hash", "learned"
    outlier_action="clip",           # or "remove", "transform"
    missing_strategy="median"        # or "mean", "mode", "drop"
)
```

#### 4. Model Architecture
Customize model size for your task:

```python
# For small datasets
config = TabGPTConfig(
    hidden_size=256,
    num_hidden_layers=6,
    num_attention_heads=8
)

# For large datasets
config = TabGPTConfig(
    hidden_size=1024,
    num_hidden_layers=16,
    num_attention_heads=16
)

model = TabGPTForSequenceClassification(config=config, num_labels=num_classes)
```

### Adding New Examples

To contribute a new example:

1. **Create the example file**: `examples/your_example.py`
2. **Follow the template structure**:
   ```python
   #!/usr/bin/env python3
   """
   Your Example Title
   
   Description of what this example demonstrates.
   """
   
   # Imports
   # Helper functions
   # Main demonstration function
   # if __name__ == "__main__": main()
   ```
3. **Add documentation**: Update this README
4. **Test thoroughly**: Ensure it runs in different environments
5. **Submit a pull request**: Follow contribution guidelines

## 🐛 Troubleshooting

### Common Issues

#### Out of Memory Errors
```python
# Reduce batch size
config.batch_size = 16  # or smaller

# Use gradient accumulation
config.gradient_accumulation_steps = 4

# Enable mixed precision
config.fp16 = True
```

#### Slow Training
```python
# Increase batch size (if memory allows)
config.batch_size = 64

# Use multiple workers
config.dataloader_num_workers = 4

# Reduce logging frequency
config.logging_steps = 100
```

#### Poor Performance
```python
# Try different learning rates
for lr in [1e-5, 5e-5, 1e-4]:
    config.learning_rate = lr
    # train and evaluate

# Increase model capacity
config = TabGPTConfig(hidden_size=1024, num_hidden_layers=16)

# Use pre-trained model
model = TabGPTForSequenceClassification.from_pretrained('tabgpt-base')
```

#### Import Errors
```bash
# Install missing dependencies
pip install missing_package

# Update TabGPT
pip install -e . --upgrade

# Check Python version (requires 3.8+)
python --version
```

### Getting Help

1. **Check the logs**: Look for error messages in the console output
2. **Verify data**: Use `DataValidator` to check your data quality
3. **Start simple**: Begin with smaller datasets and basic configurations
4. **Read documentation**: Check the [User Guide](../docs/user_guide.md) and [API Reference](../docs/api_reference.md)
5. **Ask for help**: Open an issue on GitHub with:
   - Your code
   - Error messages
   - System information
   - Data characteristics (size, types, etc.)

## 📈 Performance Expectations

### Training Times (approximate)
- **Basic Classification** (1K samples): 2-5 minutes
- **Regression** (5K samples): 5-10 minutes
- **Transfer Learning** (10K + 1K samples): 15-30 minutes
- **Comprehensive Demo**: 30-60 minutes

### Memory Requirements
- **Minimum**: 4GB RAM, no GPU required
- **Recommended**: 8GB RAM, GPU with 4GB VRAM
- **Optimal**: 16GB RAM, GPU with 8GB+ VRAM

### Accuracy Expectations
- **Simple datasets**: 85-95% accuracy
- **Complex datasets**: 70-85% accuracy
- **Transfer learning**: 5-15% improvement over baselines
- **LoRA fine-tuning**: 95-100% of full fine-tuning performance

## 🎓 Educational Value

Each example is designed to teach specific concepts:

### Machine Learning Concepts
- **Supervised Learning**: Classification and regression
- **Transfer Learning**: Leveraging pre-trained models
- **Parameter Efficiency**: LoRA and adapter methods
- **Model Evaluation**: Cross-validation, benchmarking
- **Uncertainty Quantification**: Prediction confidence

### Software Engineering Concepts
- **Modular Design**: Reusable components
- **Configuration Management**: Flexible parameter settings
- **Error Handling**: Robust data processing
- **Testing**: Validation and verification
- **Documentation**: Clear code and comments

### Data Science Concepts
- **Data Quality**: Validation and cleaning
- **Feature Engineering**: Automatic representation learning
- **Model Selection**: Architecture and hyperparameter choices
- **Performance Analysis**: Metrics and visualization
- **Reproducibility**: Consistent results across runs

## 🤝 Contributing

We welcome contributions of new examples! Please:

1. **Follow the existing style**: Consistent formatting and documentation
2. **Include comprehensive comments**: Explain what each section does
3. **Add error handling**: Make examples robust
4. **Test thoroughly**: Verify examples work in different environments
5. **Update documentation**: Add your example to this README

### Example Contribution Checklist
- [ ] Code follows PEP 8 style guidelines
- [ ] Includes docstrings and comments
- [ ] Handles common errors gracefully
- [ ] Runs successfully on clean environment
- [ ] Demonstrates clear learning objectives
- [ ] Updates README.md with description
- [ ] Includes expected runtime and requirements

## 📚 Additional Resources

- **[User Guide](../docs/user_guide.md)**: Comprehensive usage documentation
- **[API Reference](../docs/api_reference.md)**: Detailed API documentation
- **[Getting Started Tutorial](../tutorials/getting_started.md)**: Step-by-step introduction
- **[GitHub Repository](https://github.com/your-org/tabgpt)**: Source code and issues
- **[Research Papers](../docs/references.md)**: Academic background and citations

---

**Happy learning with TabGPT!** 🚀

If you have questions or suggestions for new examples, please open an issue or start a discussion on GitHub.