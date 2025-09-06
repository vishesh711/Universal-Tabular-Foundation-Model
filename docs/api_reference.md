# TabGPT API Reference

Complete API documentation for TabGPT components.

## Table of Contents

1. [Models](#models)
2. [Tokenizers](#tokenizers)
3. [Task Heads](#task-heads)
4. [Training](#training)
5. [Fine-tuning](#fine-tuning)
6. [Adapters](#adapters)
7. [Evaluation](#evaluation)
8. [Utilities](#utilities)
9. [Data Processing](#data-processing)

## Models

### TabGPTConfig

Configuration class for TabGPT models.

```python
class TabGPTConfig:
    def __init__(
        self,
        vocab_size: int = 30000,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        max_position_embeddings: int = 512,
        type_vocab_size: int = 2,
        initializer_range: float = 0.02,
        layer_norm_eps: float = 1e-12,
        pad_token_id: int = 0,
        **kwargs
    )
```

**Parameters:**
- `vocab_size`: Size of the vocabulary
- `hidden_size`: Hidden dimension of the model
- `num_hidden_layers`: Number of transformer layers
- `num_attention_heads`: Number of attention heads
- `intermediate_size`: Dimension of feed-forward layers
- `hidden_dropout_prob`: Dropout probability for hidden layers
- `attention_probs_dropout_prob`: Dropout probability for attention
- `max_position_embeddings`: Maximum sequence length
- `type_vocab_size`: Number of token types
- `initializer_range`: Standard deviation for weight initialization
- `layer_norm_eps`: Epsilon for layer normalization
- `pad_token_id`: ID of padding token

### TabGPTModel

Base TabGPT model without task-specific head.

```python
class TabGPTModel(nn.Module):
    def __init__(self, config: TabGPTConfig)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True
    ) -> Union[Tuple, BaseModelOutput]
```

**Parameters:**
- `input_ids`: Token IDs of shape `(batch_size, sequence_length)`
- `attention_mask`: Attention mask of shape `(batch_size, sequence_length)`
- `token_type_ids`: Token type IDs of shape `(batch_size, sequence_length)`
- `position_ids`: Position IDs of shape `(batch_size, sequence_length)`
- `output_attentions`: Whether to return attention weights
- `output_hidden_states`: Whether to return hidden states
- `return_dict`: Whether to return a dictionary

**Returns:**
- `BaseModelOutput` with `last_hidden_state`, `hidden_states`, `attentions`

### TabGPTForSequenceClassification

TabGPT model for sequence classification tasks.

```python
class TabGPTForSequenceClassification(nn.Module):
    def __init__(
        self,
        config: Optional[TabGPTConfig] = None,
        num_labels: int = 2
    )
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Union[Tuple, SequenceClassifierOutput]
    
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        **kwargs
    ) -> 'TabGPTForSequenceClassification'
    
    def save_pretrained(self, save_directory: str)
```

**Parameters:**
- `config`: Model configuration
- `num_labels`: Number of classification labels
- `input_ids`: Token IDs
- `attention_mask`: Attention mask
- `labels`: Ground truth labels for training

**Returns:**
- `SequenceClassifierOutput` with `loss`, `logits`, `hidden_states`, `attentions`

### TabGPTForRegression

TabGPT model for regression tasks.

```python
class TabGPTForRegression(nn.Module):
    def __init__(
        self,
        config: Optional[TabGPTConfig] = None,
        output_dim: int = 1
    )
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Union[Tuple, RegressionOutput]
```

**Parameters:**
- `config`: Model configuration
- `output_dim`: Dimension of regression output
- `input_ids`: Token IDs
- `attention_mask`: Attention mask
- `labels`: Ground truth values for training

**Returns:**
- `RegressionOutput` with `loss`, `predictions`, `hidden_states`, `attentions`

## Tokenizers

### TabGPTTokenizer

Main tokenizer for TabGPT models.

```python
class TabGPTTokenizer:
    def __init__(
        self,
        vocab_size: int = 30000,
        max_length: int = 512,
        numerical_encoder: Optional[NumericalEncoder] = None,
        categorical_encoder: Optional[CategoricalEncoder] = None,
        column_encoder: Optional[ColumnEncoder] = None,
        **kwargs
    )
    
    def encode_batch(
        self,
        df: pd.DataFrame,
        return_tensors: str = "pt"
    ) -> Dict[str, torch.Tensor]
    
    def encode_single(
        self,
        row: pd.Series,
        return_tensors: str = "pt"
    ) -> Dict[str, torch.Tensor]
    
    def decode(
        self,
        token_ids: torch.Tensor
    ) -> List[str]
    
    def create_dataset(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None
    ) -> TabularDataset
    
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str
    ) -> 'TabGPTTokenizer'
    
    def save_pretrained(self, save_directory: str)
```

**Methods:**
- `encode_batch()`: Tokenize a batch of rows
- `encode_single()`: Tokenize a single row
- `decode()`: Convert token IDs back to strings
- `create_dataset()`: Create a PyTorch dataset
- `from_pretrained()`: Load pre-trained tokenizer
- `save_pretrained()`: Save tokenizer

### NumericalEncoder

Encoder for numerical features.

```python
class NumericalEncoder:
    def __init__(
        self,
        strategy: str = "binning",
        num_bins: int = 100,
        embedding_dim: int = 64,
        handle_missing: str = "special_token"
    )
    
    def fit(self, values: pd.Series, column_info: ColumnInfo)
    
    def encode(
        self,
        values: pd.Series,
        column_info: ColumnInfo
    ) -> torch.Tensor
    
    def decode(
        self,
        tokens: torch.Tensor,
        column_info: ColumnInfo
    ) -> pd.Series
```

**Parameters:**
- `strategy`: Encoding strategy ("binning", "normalization", "embedding")
- `num_bins`: Number of bins for binning strategy
- `embedding_dim`: Dimension of embeddings
- `handle_missing`: How to handle missing values

### CategoricalEncoder

Encoder for categorical features.

```python
class CategoricalEncoder:
    def __init__(
        self,
        strategy: str = "frequency",
        max_vocab_size: int = 10000,
        min_frequency: int = 1,
        handle_unknown: str = "unk_token"
    )
    
    def fit(self, values: pd.Series, column_info: ColumnInfo)
    
    def encode(
        self,
        values: pd.Series,
        column_info: ColumnInfo
    ) -> torch.Tensor
```

**Parameters:**
- `strategy`: Encoding strategy ("frequency", "hash", "learned")
- `max_vocab_size`: Maximum vocabulary size
- `min_frequency`: Minimum frequency for inclusion
- `handle_unknown`: How to handle unknown categories

## Task Heads

### BaseTaskHead

Abstract base class for task-specific heads.

```python
class BaseTaskHead(nn.Module):
    def __init__(self)
    
    @abstractmethod
    def forward(
        self,
        features: torch.Tensor,
        targets: Optional[torch.Tensor] = None
    ) -> TaskOutput
    
    def compute_loss(
        self,
        outputs: TaskOutput,
        targets: torch.Tensor
    ) -> torch.Tensor
    
    def compute_metrics(
        self,
        outputs: TaskOutput,
        targets: torch.Tensor
    ) -> Dict[str, float]
```

### BinaryClassificationHead

Head for binary classification tasks.

```python
class BinaryClassificationHead(BaseTaskHead):
    def __init__(
        self,
        input_dim: int,
        dropout: float = 0.1,
        hidden_dims: Optional[List[int]] = None
    )
    
    def forward(
        self,
        features: torch.Tensor,
        targets: Optional[torch.Tensor] = None
    ) -> TaskOutput
```

**Parameters:**
- `input_dim`: Input feature dimension
- `dropout`: Dropout probability
- `hidden_dims`: Hidden layer dimensions

### MultiClassClassificationHead

Head for multi-class classification tasks.

```python
class MultiClassClassificationHead(BaseTaskHead):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        dropout: float = 0.1,
        hidden_dims: Optional[List[int]] = None
    )
```

### RegressionHead

Head for regression tasks.

```python
class RegressionHead(BaseTaskHead):
    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,
        dropout: float = 0.1,
        hidden_dims: Optional[List[int]] = None,
        estimate_uncertainty: bool = False
    )
```

**Parameters:**
- `output_dim`: Number of regression outputs
- `estimate_uncertainty`: Whether to estimate prediction uncertainty

### MultiLabelClassificationHead

Head for multi-label classification tasks.

```python
class MultiLabelClassificationHead(BaseTaskHead):
    def __init__(
        self,
        input_dim: int,
        num_labels: int,
        dropout: float = 0.1,
        threshold: float = 0.5,
        hidden_dims: Optional[List[int]] = None
    )
```

**Parameters:**
- `num_labels`: Number of labels
- `threshold`: Classification threshold

### SurvivalHead

Head for survival analysis tasks.

```python
class SurvivalHead(BaseTaskHead):
    def __init__(
        self,
        input_dim: int,
        risk_estimation_method: str = "cox",
        num_time_bins: Optional[int] = None,
        dropout: float = 0.1
    )
    
    def predict_survival_function(
        self,
        metadata: Dict[str, torch.Tensor],
        time_points: torch.Tensor
    ) -> torch.Tensor
    
    def predict_hazard_ratio(
        self,
        metadata: Dict[str, torch.Tensor]
    ) -> torch.Tensor
```

**Parameters:**
- `risk_estimation_method`: Method for risk estimation ("cox", "discrete_time", "parametric")
- `num_time_bins`: Number of time bins for discrete-time models

### AnomalyDetectionHead

Head for anomaly detection tasks.

```python
class AnomalyDetectionHead(BaseTaskHead):
    def __init__(
        self,
        input_dim: int,
        reconstruction_dim: int = 128,
        detection_method: str = "reconstruction",
        threshold: Optional[float] = None
    )
    
    def compute_anomaly_score(
        self,
        features: torch.Tensor
    ) -> torch.Tensor
```

**Parameters:**
- `reconstruction_dim`: Dimension for reconstruction
- `detection_method`: Detection method ("reconstruction", "one_class_svm")
- `threshold`: Anomaly threshold

## Training

### MaskedCellModelingTrainer

Trainer for masked cell modeling pre-training.

```python
class MaskedCellModelingTrainer:
    def __init__(
        self,
        model: TabGPTModel,
        tokenizer: TabGPTTokenizer,
        masking_prob: float = 0.15,
        mask_strategy: str = "random",
        optimizer: Optional[torch.optim.Optimizer] = None
    )
    
    def train(
        self,
        dataloader: DataLoader,
        num_epochs: int = 1,
        log_interval: int = 100
    )
    
    def evaluate(
        self,
        dataloader: DataLoader
    ) -> Dict[str, float]
```

**Parameters:**
- `masking_prob`: Probability of masking each cell
- `mask_strategy`: Masking strategy ("random", "column_wise", "row_wise")

### ContrastiveRowLearningTrainer

Trainer for contrastive row learning.

```python
class ContrastiveRowLearningTrainer:
    def __init__(
        self,
        model: TabGPTModel,
        temperature: float = 0.1,
        augmentation_strategy: str = "noise",
        optimizer: Optional[torch.optim.Optimizer] = None
    )
    
    def train(
        self,
        dataloader: DataLoader,
        num_epochs: int = 1
    )
```

**Parameters:**
- `temperature`: Temperature for contrastive loss
- `augmentation_strategy`: Data augmentation strategy

### NextRowPredictionTrainer

Trainer for next row prediction.

```python
class NextRowPredictionTrainer:
    def __init__(
        self,
        model: TabGPTModel,
        tokenizer: TabGPTTokenizer,
        sequence_length: int = 10,
        optimizer: Optional[torch.optim.Optimizer] = None
    )
```

## Fine-tuning

### FineTuningConfig

Configuration for fine-tuning.

```python
@dataclass
class FineTuningConfig:
    task_type: str
    num_labels: Optional[int] = None
    output_dim: Optional[int] = None
    learning_rate: float = 5e-5
    num_epochs: int = 3
    batch_size: int = 32
    warmup_steps: int = 0
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    fp16: bool = False
    dataloader_num_workers: int = 0
    save_steps: int = 500
    eval_steps: int = 500
    logging_steps: int = 50
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
```

### TabGPTFineTuningTrainer

Main trainer for fine-tuning TabGPT models.

```python
class TabGPTFineTuningTrainer:
    def __init__(
        self,
        model: nn.Module,
        config: FineTuningConfig,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        tokenizer: Optional[TabGPTTokenizer] = None,
        compute_metrics: Optional[Callable] = None,
        callbacks: Optional[List[TrainerCallback]] = None
    )
    
    def train(self) -> TrainerState
    
    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None
    ) -> Dict[str, float]
    
    def predict(
        self,
        test_dataset: Dataset
    ) -> PredictionOutput
    
    def save_model(self, output_dir: str)
    
    def load_model(self, model_path: str)
```

### TrainerCallback

Base class for training callbacks.

```python
class TrainerCallback:
    def on_train_begin(self, trainer: TabGPTFineTuningTrainer)
    def on_train_end(self, trainer: TabGPTFineTuningTrainer)
    def on_epoch_begin(self, trainer: TabGPTFineTuningTrainer)
    def on_epoch_end(self, trainer: TabGPTFineTuningTrainer)
    def on_step_begin(self, trainer: TabGPTFineTuningTrainer)
    def on_step_end(self, trainer: TabGPTFineTuningTrainer)
```

### EarlyStoppingCallback

Callback for early stopping.

```python
class EarlyStoppingCallback(TrainerCallback):
    def __init__(
        self,
        early_stopping_patience: int = 3,
        metric_for_best_model: str = "eval_loss",
        greater_is_better: bool = False,
        early_stopping_threshold: float = 0.0
    )
```

## Adapters

### LoRAConfig

Configuration for LoRA (Low-Rank Adaptation).

```python
@dataclass
class LoRAConfig:
    r: int = 8
    alpha: int = 16
    dropout: float = 0.1
    target_modules: List[str] = field(default_factory=lambda: ["query", "key", "value", "dense"])
    bias: str = "none"
    task_type: str = "FEATURE_EXTRACTION"
```

**Parameters:**
- `r`: Rank of adaptation
- `alpha`: LoRA scaling parameter
- `dropout`: LoRA dropout
- `target_modules`: Modules to apply LoRA to
- `bias`: Bias handling ("none", "all", "lora_only")

### apply_lora_to_model

Function to apply LoRA to a model.

```python
def apply_lora_to_model(
    model: nn.Module,
    config: LoRAConfig
) -> nn.Module
```

### AdaLoRAConfig

Configuration for AdaLoRA (Adaptive LoRA).

```python
@dataclass
class AdaLoRAConfig(LoRAConfig):
    target_r: int = 8
    init_r: int = 12
    tinit: int = 0
    tfinal: int = 1000
    deltaT: int = 10
    beta1: float = 0.85
    beta2: float = 0.85
    orth_reg_weight: float = 0.5
```

## Evaluation

### EvaluationConfig

Configuration for evaluation.

```python
@dataclass
class EvaluationConfig:
    strategy: str = "holdout"
    test_size: float = 0.2
    cv_folds: int = 5
    random_state: int = 42
    stratify: bool = True
    primary_metric: str = "accuracy"
    compute_all_metrics: bool = True
    baseline_models: List[str] = field(default_factory=list)
```

### CrossValidationEvaluator

Evaluator using cross-validation.

```python
class CrossValidationEvaluator:
    def __init__(self, config: EvaluationConfig)
    
    def evaluate(
        self,
        model: nn.Module,
        X: pd.DataFrame,
        y: pd.Series,
        task_type: str
    ) -> EvaluationResult
```

### ClassificationBenchmark

Benchmark for classification tasks.

```python
class ClassificationBenchmark:
    def __init__(
        self,
        name: str,
        data_loader: Callable,
        description: str = "",
        task_type: str = "binary"
    )
    
    def run_benchmark(
        self,
        models: Dict[str, nn.Module],
        test_size: float = 0.2,
        cv_folds: int = 5
    ) -> BenchmarkResult
    
    def get_leaderboard(
        self,
        metric: str = "accuracy"
    ) -> pd.DataFrame
```

### TransferLearningEvaluator

Evaluator for transfer learning experiments.

```python
class TransferLearningEvaluator:
    def __init__(self, config: EvaluationConfig)
    
    def evaluate_transfer(
        self,
        pretrained_model: nn.Module,
        baseline_model: nn.Module,
        source_dataset: Tuple[pd.DataFrame, pd.Series],
        target_dataset: Tuple[pd.DataFrame, pd.Series]
    ) -> TransferLearningResult
```

### create_baseline_models

Function to create baseline models for comparison.

```python
def create_baseline_models(
    task_type: str,
    include_models: Optional[List[str]] = None
) -> Dict[str, BaseEstimator]
```

**Parameters:**
- `task_type`: Type of task ("classification", "regression")
- `include_models`: List of models to include

**Returns:**
- Dictionary mapping model names to sklearn estimators

## Utilities

### DataValidator

Validator for data quality checks.

```python
class DataValidator:
    def __init__(
        self,
        missing_threshold: float = 0.3,
        min_samples: int = 100,
        min_features: int = 2,
        max_cardinality: int = 1000
    )
    
    def validate_dataframe(
        self,
        df: pd.DataFrame
    ) -> ValidationResult
    
    def validate_target(
        self,
        y: pd.Series,
        task_type: str
    ) -> ValidationResult
```

### DataRecovery

Automatic data recovery and cleaning.

```python
class DataRecovery:
    def __init__(
        self,
        auto_fix: bool = True,
        missing_strategy: str = "median",
        outlier_strategy: str = "clip",
        dtype_coercion: bool = True
    )
    
    def recover_dataframe(
        self,
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, RecoveryLog]
```

### RobustNormalizer

Robust data normalization.

```python
class RobustNormalizer:
    def __init__(
        self,
        numerical_strategy: str = "robust",
        categorical_strategy: str = "frequency",
        outlier_action: str = "clip",
        missing_strategy: str = "median"
    )
    
    def fit(self, df: pd.DataFrame) -> 'RobustNormalizer'
    
    def transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, NormalizationLog]
    
    def fit_transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, NormalizationLog]
    
    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame
```

## Data Processing

### TabularDataset

PyTorch dataset for tabular data.

```python
class TabularDataset(Dataset):
    def __init__(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        tokenizer: Optional[TabGPTTokenizer] = None
    )
    
    def __len__(self) -> int
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]
```

### ColumnInfo

Information about a column.

```python
@dataclass
class ColumnInfo:
    name: str
    dtype: str
    is_categorical: bool
    is_numerical: bool
    unique_values: Optional[List] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    missing_count: int = 0
    cardinality: int = 0
```

### TaskOutput

Output from task-specific heads.

```python
@dataclass
class TaskOutput:
    predictions: torch.Tensor
    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None
    hidden_states: Optional[Tuple[torch.Tensor]] = None
    attentions: Optional[Tuple[torch.Tensor]] = None
    metadata: Optional[Dict[str, Any]] = None
    uncertainty: Optional[torch.Tensor] = None
```

### EvaluationResult

Result from model evaluation.

```python
@dataclass
class EvaluationResult:
    metrics: Dict[str, float]
    metrics_std: Dict[str, float]
    predictions: Optional[np.ndarray] = None
    probabilities: Optional[np.ndarray] = None
    fold_results: Optional[List[Dict[str, float]]] = None
    confusion_matrix: Optional[np.ndarray] = None
    classification_report: Optional[str] = None
```

### BenchmarkResult

Result from benchmark comparison.

```python
@dataclass
class BenchmarkResult:
    model_results: Dict[str, EvaluationResult]
    leaderboard: pd.DataFrame
    statistical_tests: Dict[str, Dict[str, float]]
    runtime_comparison: Dict[str, float]
```

### TransferLearningResult

Result from transfer learning evaluation.

```python
@dataclass
class TransferLearningResult:
    baseline_metrics: Dict[str, float]
    pretrained_metrics: Dict[str, float]
    relative_improvement: Dict[str, float]
    statistical_significance: Dict[str, float]
    sample_efficiency: Dict[str, List[float]]
```

## Exceptions

### TabGPTError

Base exception for TabGPT.

```python
class TabGPTError(Exception):
    pass
```

### TokenizationError

Exception for tokenization errors.

```python
class TokenizationError(TabGPTError):
    pass
```

### ValidationError

Exception for validation errors.

```python
class ValidationError(TabGPTError):
    pass
```

### TrainingError

Exception for training errors.

```python
class TrainingError(TabGPTError):
    pass
```

### ModelError

Exception for model errors.

```python
class ModelError(TabGPTError):
    pass
```

## Constants

### Task Types

```python
CLASSIFICATION_TASKS = ["binary", "multiclass", "multilabel"]
REGRESSION_TASKS = ["regression", "multi_output_regression"]
SPECIALIZED_TASKS = ["survival", "anomaly_detection", "time_series"]
ALL_TASK_TYPES = CLASSIFICATION_TASKS + REGRESSION_TASKS + SPECIALIZED_TASKS
```

### Encoding Strategies

```python
NUMERICAL_STRATEGIES = ["binning", "normalization", "embedding"]
CATEGORICAL_STRATEGIES = ["frequency", "hash", "learned", "one_hot"]
MISSING_STRATEGIES = ["special_token", "median", "mode", "drop"]
```

### Model Sizes

```python
MODEL_CONFIGS = {
    "tabgpt-tiny": {"hidden_size": 256, "num_hidden_layers": 6, "num_attention_heads": 8},
    "tabgpt-small": {"hidden_size": 512, "num_hidden_layers": 8, "num_attention_heads": 8},
    "tabgpt-base": {"hidden_size": 768, "num_hidden_layers": 12, "num_attention_heads": 12},
    "tabgpt-large": {"hidden_size": 1024, "num_hidden_layers": 16, "num_attention_heads": 16},
    "tabgpt-xl": {"hidden_size": 1280, "num_hidden_layers": 20, "num_attention_heads": 20}
}
```

This API reference provides comprehensive documentation for all TabGPT components. For usage examples and tutorials, see the examples directory and user guide.