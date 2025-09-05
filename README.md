# TabGPT: A Foundation Model for Tabular Data

TabGPT is a general-purpose pre-trained model for tabular datasets that can adapt to any downstream task including classification, regression, anomaly detection, and survival analysis. Similar to how BERT and GPT revolutionized natural language processing, TabGPT aims to bring foundation model capabilities to structured tabular data.

## 🚀 Key Features

- **Universal Pre-training**: Self-supervised learning on diverse tabular datasets
- **Cross-Schema Transfer**: Adapts to datasets with different column structures
- **Multi-Task Support**: Classification, regression, anomaly detection, survival analysis
- **HuggingFace Compatible**: Familiar API for easy integration
- **Transformer Architecture**: Based on proven FT-Transformer design with novel enhancements

## 🏗️ Architecture

TabGPT combines:
- **Feature Tokenizer**: Converts heterogeneous tabular data into uniform embeddings
- **Column Encoder**: Learns semantic representations of column metadata
- **Row Encoder**: FT-Transformer-based processing of row-level patterns  
- **Cross-Attention Fusion**: Combines row and column information

## 📦 Installation

```bash
pip install tabgpt
```

For development:
```bash
git clone https://github.com/tabgpt/tabgpt.git
cd tabgpt
pip install -e ".[dev]"
```

## 🔥 Quick Start

```python
import pandas as pd
from tabgpt import TabGPTForClassification, TabularTokenizer

# Load your tabular data
df = pd.read_csv("your_dataset.csv")

# Initialize tokenizer and model
tokenizer = TabularTokenizer()
model = TabGPTForClassification.from_pretrained("tabgpt-base")

# Tokenize data
tokenized = tokenizer.fit_transform(df)

# Make predictions
outputs = model(tokenized.tokens, attention_mask=tokenized.attention_mask)
predictions = outputs.logits.argmax(dim=-1)
```

## 🎯 Pre-training Objectives

TabGPT uses multiple self-supervised objectives:

1. **Masked Cell Modeling**: Predict randomly masked cell values
2. **Masked Column Modeling**: Predict entire columns from remaining features  
3. **Contrastive Row Learning**: Learn consistent representations under augmentation
4. **Next Row Prediction**: Temporal modeling for time-series data

## 🔧 Model Variants

- `tabgpt-base`: 6 layers, 256 hidden size
- `tabgpt-large`: 12 layers, 512 hidden size  
- `tabgpt-xl`: 24 layers, 1024 hidden size

## 📊 Benchmarks

TabGPT achieves state-of-the-art results on:
- OpenML CC18 benchmark suite
- UCI repository datasets
- Real-world Kaggle competitions

## 🤝 Contributing

We welcome contributions! Please see our [contributing guidelines](CONTRIBUTING.md).

## 📄 License

MIT License - see [LICENSE](LICENSE) file.

## 📚 Citation

```bibtex
@article{tabgpt2024,
  title={TabGPT: A Foundation Model for Tabular Data},
  author={TabGPT Team},
  journal={arXiv preprint},
  year={2024}
}
```

## 🔗 Links

- [Documentation](https://tabgpt.readthedocs.io)
- [Paper](https://arxiv.org/abs/xxxx.xxxxx)
- [HuggingFace Hub](https://huggingface.co/tabgpt)
- [Tutorials](https://github.com/tabgpt/tutorials)