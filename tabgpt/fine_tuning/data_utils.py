"""Data utilities for fine-tuning TabGPT models."""
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
from dataclasses import dataclass
import logging

from ..tokenizers import TabGPTTokenizer, TabularTokenizer

logger = logging.getLogger(__name__)


class TabularDataset(Dataset):
    """Dataset for tabular data fine-tuning."""
    
    def __init__(
        self,
        dataframe: pd.DataFrame,
        target_column: str,
        tokenizer: Optional[Union[TabGPTTokenizer, TabularTokenizer]] = None,
        max_length: Optional[int] = None,
        task_type: str = "classification"
    ):
        self.dataframe = dataframe.copy()
        self.target_column = target_column
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.task_type = task_type
        
        # Separate features and targets
        self.features = self.dataframe.drop(columns=[target_column])
        self.targets = self.dataframe[target_column]
        
        # Validate data
        self._validate_data()
    
    def _validate_data(self):
        """Validate dataset."""
        if len(self.features) == 0:
            raise ValueError("No feature columns found")
        
        if len(self.targets) == 0:
            raise ValueError("No target values found")
        
        if len(self.features) != len(self.targets):
            raise ValueError("Mismatch between features and targets length")
        
        # Check for missing targets
        if self.targets.isnull().any():
            logger.warning(f"Found {self.targets.isnull().sum()} missing target values")
    
    def __len__(self) -> int:
        return len(self.dataframe)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # Get row data
        row_features = self.features.iloc[idx]
        target = self.targets.iloc[idx]
        
        # Convert to dictionary
        item = {
            "features": row_features.to_dict(),
            "labels": target
        }
        
        # Tokenize if tokenizer is provided
        if self.tokenizer is not None:
            # Create single-row DataFrame for tokenization
            row_df = pd.DataFrame([row_features])
            
            if isinstance(self.tokenizer, TabGPTTokenizer):
                # Use HuggingFace-compatible tokenizer
                tokenized = self.tokenizer(
                    row_df,
                    max_length=self.max_length,
                    padding=True,
                    truncation=True,
                    return_tensors="pt"
                )
                
                # Squeeze batch dimension
                for key, value in tokenized.items():
                    if isinstance(value, torch.Tensor):
                        item[key] = value.squeeze(0)
            else:
                # Use TabularTokenizer
                tokenized = self.tokenizer.tokenize_dataframe(row_df)
                if tokenized and len(tokenized) > 0:
                    # Convert to tensor format expected by model
                    item["input_ids"] = torch.tensor(tokenized[0], dtype=torch.long)
        
        # Convert target to appropriate format
        if self.task_type == "classification":
            if isinstance(target, str):
                # Handle string labels (will need label encoding)
                item["labels"] = target
            else:
                item["labels"] = torch.tensor(target, dtype=torch.long)
        elif self.task_type == "regression":
            item["labels"] = torch.tensor(target, dtype=torch.float)
        
        return item


@dataclass
class TabularDataCollator:
    """Data collator for tabular data."""
    
    tokenizer: Optional[Union[TabGPTTokenizer, TabularTokenizer]] = None
    padding: bool = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate batch of features."""
        batch = {}
        
        # Handle different input formats
        if "input_ids" in features[0]:
            # Tokenized input
            batch["input_ids"] = self._pad_sequence([f["input_ids"] for f in features])
            
            if "attention_mask" in features[0]:
                batch["attention_mask"] = self._pad_sequence([f["attention_mask"] for f in features])
            
            if "token_type_ids" in features[0]:
                batch["token_type_ids"] = self._pad_sequence([f["token_type_ids"] for f in features])
        
        elif "features" in features[0]:
            # Raw features - convert to tensor
            feature_dicts = [f["features"] for f in features]
            batch["features"] = self._dict_to_tensor(feature_dicts)
        
        # Handle labels
        if "labels" in features[0]:
            labels = [f["labels"] for f in features]
            
            # Check if labels are strings (need encoding)
            if isinstance(labels[0], str):
                # Create label mapping
                unique_labels = sorted(list(set(labels)))
                label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
                labels = [label_to_id[label] for label in labels]
            
            batch["labels"] = torch.tensor(labels)
        
        return batch
    
    def _pad_sequence(self, sequences: List[torch.Tensor]) -> torch.Tensor:
        """Pad sequences to same length."""
        if not sequences:
            return torch.empty(0)
        
        # Find max length
        max_len = max(len(seq) for seq in sequences)
        
        if self.max_length is not None:
            max_len = min(max_len, self.max_length)
        
        # Pad sequences
        padded = []
        for seq in sequences:
            if len(seq) > max_len:
                # Truncate
                padded_seq = seq[:max_len]
            else:
                # Pad
                pad_length = max_len - len(seq)
                padded_seq = torch.cat([seq, torch.zeros(pad_length, dtype=seq.dtype)])
            
            padded.append(padded_seq)
        
        return torch.stack(padded)
    
    def _dict_to_tensor(self, feature_dicts: List[Dict[str, Any]]) -> torch.Tensor:
        """Convert list of feature dictionaries to tensor."""
        # Get all unique keys
        all_keys = set()
        for d in feature_dicts:
            all_keys.update(d.keys())
        
        all_keys = sorted(list(all_keys))
        
        # Convert to tensor
        batch_features = []
        for d in feature_dicts:
            row_features = []
            for key in all_keys:
                value = d.get(key, 0.0)  # Default to 0 for missing values
                if isinstance(value, str):
                    # Simple string encoding (hash-based)
                    value = float(hash(value) % 10000) / 10000.0
                elif pd.isna(value):
                    value = 0.0
                row_features.append(float(value))
            batch_features.append(row_features)
        
        return torch.tensor(batch_features, dtype=torch.float)


def prepare_classification_data(
    dataframe: pd.DataFrame,
    target_column: str,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True
) -> Tuple[TabularDataset, TabularDataset]:
    """
    Prepare data for classification fine-tuning.
    
    Args:
        dataframe: Input DataFrame
        target_column: Name of target column
        test_size: Fraction of data for testing
        random_state: Random seed
        stratify: Whether to stratify split
        
    Returns:
        Tuple of (train_dataset, eval_dataset)
    """
    from sklearn.model_selection import train_test_split
    
    # Prepare features and targets
    X = dataframe.drop(columns=[target_column])
    y = dataframe[target_column]
    
    # Split data
    stratify_param = y if stratify else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify_param
    )
    
    # Create datasets
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    
    train_dataset = TabularDataset(train_df, target_column, task_type="classification")
    eval_dataset = TabularDataset(test_df, target_column, task_type="classification")
    
    logger.info(f"Created classification datasets: train={len(train_dataset)}, eval={len(eval_dataset)}")
    logger.info(f"Number of classes: {y.nunique()}")
    
    return train_dataset, eval_dataset


def prepare_regression_data(
    dataframe: pd.DataFrame,
    target_column: str,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[TabularDataset, TabularDataset]:
    """
    Prepare data for regression fine-tuning.
    
    Args:
        dataframe: Input DataFrame
        target_column: Name of target column
        test_size: Fraction of data for testing
        random_state: Random seed
        
    Returns:
        Tuple of (train_dataset, eval_dataset)
    """
    from sklearn.model_selection import train_test_split
    
    # Prepare features and targets
    X = dataframe.drop(columns=[target_column])
    y = dataframe[target_column]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Create datasets
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    
    train_dataset = TabularDataset(train_df, target_column, task_type="regression")
    eval_dataset = TabularDataset(test_df, target_column, task_type="regression")
    
    logger.info(f"Created regression datasets: train={len(train_dataset)}, eval={len(eval_dataset)}")
    logger.info(f"Target statistics: mean={y.mean():.3f}, std={y.std():.3f}")
    
    return train_dataset, eval_dataset


def create_data_collator(
    tokenizer: Optional[Union[TabGPTTokenizer, TabularTokenizer]] = None,
    padding: bool = True,
    max_length: Optional[int] = None
) -> TabularDataCollator:
    """
    Create data collator for tabular data.
    
    Args:
        tokenizer: Tokenizer for processing data
        padding: Whether to pad sequences
        max_length: Maximum sequence length
        
    Returns:
        Configured data collator
    """
    return TabularDataCollator(
        tokenizer=tokenizer,
        padding=padding,
        max_length=max_length
    )


def prepare_data_for_task(
    dataframe: pd.DataFrame,
    target_column: str,
    task_type: str,
    tokenizer: Optional[Union[TabGPTTokenizer, TabularTokenizer]] = None,
    test_size: float = 0.2,
    max_length: Optional[int] = None,
    random_state: int = 42
) -> Tuple[TabularDataset, TabularDataset, TabularDataCollator]:
    """
    Prepare data for any task type.
    
    Args:
        dataframe: Input DataFrame
        target_column: Name of target column
        task_type: Type of task (classification, regression)
        tokenizer: Optional tokenizer
        test_size: Fraction of data for testing
        max_length: Maximum sequence length
        random_state: Random seed
        
    Returns:
        Tuple of (train_dataset, eval_dataset, data_collator)
    """
    # Prepare datasets based on task type
    if task_type == "classification":
        train_dataset, eval_dataset = prepare_classification_data(
            dataframe, target_column, test_size, random_state
        )
    elif task_type == "regression":
        train_dataset, eval_dataset = prepare_regression_data(
            dataframe, target_column, test_size, random_state
        )
    else:
        raise ValueError(f"Unsupported task type: {task_type}")
    
    # Set tokenizer if provided
    if tokenizer is not None:
        train_dataset.tokenizer = tokenizer
        train_dataset.max_length = max_length
        eval_dataset.tokenizer = tokenizer
        eval_dataset.max_length = max_length
    
    # Create data collator
    data_collator = create_data_collator(tokenizer, max_length=max_length)
    
    return train_dataset, eval_dataset, data_collator


def load_dataset_from_file(
    file_path: str,
    target_column: str,
    task_type: str = "classification",
    **kwargs
) -> pd.DataFrame:
    """
    Load dataset from file.
    
    Args:
        file_path: Path to data file
        target_column: Name of target column
        task_type: Type of task
        **kwargs: Additional arguments for pandas read functions
        
    Returns:
        Loaded DataFrame
    """
    file_path = str(file_path).lower()
    
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path, **kwargs)
    elif file_path.endswith('.parquet'):
        df = pd.read_parquet(file_path, **kwargs)
    elif file_path.endswith('.json'):
        df = pd.read_json(file_path, **kwargs)
    elif file_path.endswith(('.xlsx', '.xls')):
        df = pd.read_excel(file_path, **kwargs)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")
    
    # Validate target column
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset")
    
    # Basic data validation
    logger.info(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    logger.info(f"Target column '{target_column}' statistics:")
    
    if task_type == "classification":
        logger.info(f"  Classes: {df[target_column].nunique()}")
        logger.info(f"  Class distribution:\n{df[target_column].value_counts()}")
    elif task_type == "regression":
        logger.info(f"  Mean: {df[target_column].mean():.3f}")
        logger.info(f"  Std: {df[target_column].std():.3f}")
        logger.info(f"  Range: [{df[target_column].min():.3f}, {df[target_column].max():.3f}]")
    
    return df


def create_sample_dataset(
    n_samples: int = 1000,
    n_features: int = 10,
    task_type: str = "classification",
    n_classes: int = 3,
    noise: float = 0.1,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Create a sample dataset for testing.
    
    Args:
        n_samples: Number of samples
        n_features: Number of features
        task_type: Type of task
        n_classes: Number of classes (for classification)
        noise: Noise level
        random_state: Random seed
        
    Returns:
        Sample DataFrame
    """
    np.random.seed(random_state)
    
    # Generate features
    feature_names = [f"feature_{i}" for i in range(n_features)]
    
    # Mix of numerical and categorical features
    data = {}
    
    # Numerical features
    for i in range(n_features // 2):
        data[feature_names[i]] = np.random.normal(0, 1, n_samples)
    
    # Categorical features
    for i in range(n_features // 2, n_features):
        n_categories = np.random.randint(3, 8)
        categories = [f"cat_{j}" for j in range(n_categories)]
        data[feature_names[i]] = np.random.choice(categories, n_samples)
    
    # Generate target
    if task_type == "classification":
        # Create target based on features with some noise
        target_weights = np.random.normal(0, 1, n_features // 2)
        target_score = sum(data[feature_names[i]] * target_weights[i] 
                          for i in range(n_features // 2))
        
        # Convert to classes
        target_score += np.random.normal(0, noise, n_samples)
        percentiles = np.percentile(target_score, np.linspace(0, 100, n_classes + 1))
        target = np.digitize(target_score, percentiles[1:-1])
        
    elif task_type == "regression":
        # Create continuous target
        target_weights = np.random.normal(0, 1, n_features // 2)
        target = sum(data[feature_names[i]] * target_weights[i] 
                    for i in range(n_features // 2))
        target += np.random.normal(0, noise, n_samples)
    
    data["target"] = target
    
    df = pd.DataFrame(data)
    logger.info(f"Created sample {task_type} dataset: {df.shape}")
    
    return df