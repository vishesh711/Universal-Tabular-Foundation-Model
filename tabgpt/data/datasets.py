"""Dataset classes for tabular data loading and processing."""

import os
import pickle
import warnings
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Tuple, Any, Iterator, Callable
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
from dataclasses import dataclass

from .loaders import TabularDataLoader, DatasetInfo
from .preprocessing import TabularPreprocessor, PreprocessingConfig, DataType
from ..tokenizers import TabularTokenizer, TokenizedTable


@dataclass
class DataSplit:
    """Data split configuration."""
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    random_state: int = 42
    stratify_column: Optional[str] = None


class TabularDataset(Dataset):
    """PyTorch Dataset for tabular data."""
    
    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: Optional[TabularTokenizer] = None,
        preprocessor: Optional[TabularPreprocessor] = None,
        target_column: Optional[str] = None,
        transform: Optional[Callable] = None,
        cache_tokenized: bool = True
    ):
        self.df = df.copy()
        self.tokenizer = tokenizer
        self.preprocessor = preprocessor
        self.target_column = target_column
        self.transform = transform
        self.cache_tokenized = cache_tokenized
        
        self._tokenized_cache = None
        self._preprocessed_df = None
        
        # Preprocess data if preprocessor is provided
        if self.preprocessor:
            if not self.preprocessor.is_fitted:
                self.preprocessor.fit(self.df, target_column)
            self._preprocessed_df = self.preprocessor.transform(self.df)
        else:
            self._preprocessed_df = self.df
        
        # Extract targets if target column is specified
        self.targets = None
        if self.target_column and self.target_column in self._preprocessed_df.columns:
            self.targets = self._preprocessed_df[self.target_column].values
            # Remove target from features
            self._preprocessed_df = self._preprocessed_df.drop(columns=[self.target_column])
        
        # Tokenize data if tokenizer is provided and caching is enabled
        if self.tokenizer and self.cache_tokenized:
            self._tokenize_data()
    
    def _tokenize_data(self):
        """Tokenize the entire dataset and cache results."""
        if not self.tokenizer.is_fitted:
            self.tokenizer.fit(self._preprocessed_df)
        
        self._tokenized_cache = self.tokenizer.transform(self._preprocessed_df)
    
    def __len__(self) -> int:
        return len(self._preprocessed_df)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample."""
        if self._tokenized_cache is not None:
            # Use cached tokenized data
            sample = {
                'input_features': self._tokenized_cache.tokens[idx],
                'attention_mask': self._tokenized_cache.attention_mask[idx]
            }
        else:
            # Tokenize on-the-fly
            row_df = self._preprocessed_df.iloc[[idx]]
            if not self.tokenizer.is_fitted:
                self.tokenizer.fit(self._preprocessed_df)
            
            tokenized = self.tokenizer.transform(row_df)
            sample = {
                'input_features': tokenized.tokens[0],
                'attention_mask': tokenized.attention_mask[0]
            }
        
        # Add target if available
        if self.targets is not None:
            target_value = self.targets[idx]
            if isinstance(target_value, str):
                # For string targets, we'll need to encode them
                # For now, just store as string and let the user handle encoding
                sample['target'] = target_value
            else:
                sample['target'] = torch.tensor(target_value)
        
        # Add raw data
        sample['raw_data'] = {col: self._preprocessed_df.iloc[idx][col] for col in self._preprocessed_df.columns}
        
        # Apply transform if provided
        if self.transform:
            sample = self.transform(sample)
        
        return sample
    
    def get_batch(self, indices: List[int]) -> Dict[str, torch.Tensor]:
        """Get a batch of samples efficiently."""
        if self._tokenized_cache is not None:
            # Use cached tokenized data
            batch = {
                'input_features': self._tokenized_cache.tokens[indices],
                'attention_mask': self._tokenized_cache.attention_mask[indices]
            }
        else:
            # Tokenize batch
            batch_df = self._preprocessed_df.iloc[indices]
            if not self.tokenizer.is_fitted:
                self.tokenizer.fit(self._preprocessed_df)
            
            tokenized = self.tokenizer.transform(batch_df)
            batch = {
                'input_features': tokenized.tokens,
                'attention_mask': tokenized.attention_mask
            }
        
        # Add targets if available
        if self.targets is not None:
            target_values = self.targets[indices]
            if isinstance(target_values[0], str):
                batch['target'] = target_values.tolist()
            else:
                batch['target'] = torch.tensor(target_values)
        
        return batch
    
    def get_feature_names(self) -> List[str]:
        """Get feature names."""
        return list(self._preprocessed_df.columns)
    
    def get_data_info(self) -> Dict[str, Any]:
        """Get information about the dataset."""
        info = {
            'n_samples': len(self),
            'n_features': len(self._preprocessed_df.columns),
            'feature_names': self.get_feature_names(),
            'has_targets': self.targets is not None,
            'is_tokenized': self._tokenized_cache is not None
        }
        
        if self.preprocessor:
            info.update(self.preprocessor.get_feature_info())
        
        return info


class StreamingTabularDataset(IterableDataset):
    """Streaming dataset for large tabular data that doesn't fit in memory."""
    
    def __init__(
        self,
        file_path: str,
        tokenizer: Optional[TabularTokenizer] = None,
        preprocessor: Optional[TabularPreprocessor] = None,
        target_column: Optional[str] = None,
        chunk_size: int = 10000,
        transform: Optional[Callable] = None,
        file_format: str = "csv"
    ):
        self.file_path = Path(file_path)
        self.tokenizer = tokenizer
        self.preprocessor = preprocessor
        self.target_column = target_column
        self.chunk_size = chunk_size
        self.transform = transform
        self.file_format = file_format.lower()
        
        # Validate file exists
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Get basic info about the file
        self._get_file_info()
    
    def _get_file_info(self):
        """Get basic information about the file."""
        if self.file_format == "csv":
            # Read first chunk to get column info
            first_chunk = pd.read_csv(self.file_path, nrows=1000)
            self.columns = list(first_chunk.columns)
            self.n_columns = len(self.columns)
            
            # Estimate total rows (rough estimate)
            file_size = self.file_path.stat().st_size
            avg_row_size = file_size / len(first_chunk) if len(first_chunk) > 0 else 1000
            self.estimated_n_rows = int(file_size / avg_row_size)
        else:
            raise NotImplementedError(f"File format {self.file_format} not supported for streaming")
    
    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Iterate over the dataset in chunks."""
        if self.file_format == "csv":
            chunk_reader = pd.read_csv(self.file_path, chunksize=self.chunk_size)
        else:
            raise NotImplementedError(f"File format {self.file_format} not supported")
        
        for chunk_df in chunk_reader:
            # Preprocess chunk if preprocessor is provided
            if self.preprocessor:
                if not self.preprocessor.is_fitted:
                    # Fit on first chunk (not ideal, but necessary for streaming)
                    self.preprocessor.fit(chunk_df, self.target_column)
                
                try:
                    chunk_df = self.preprocessor.transform(chunk_df)
                except Exception as e:
                    warnings.warn(f"Preprocessing failed for chunk: {e}")
                    continue
            
            # Extract targets if specified
            targets = None
            if self.target_column and self.target_column in chunk_df.columns:
                targets = chunk_df[self.target_column].values
                chunk_df = chunk_df.drop(columns=[self.target_column])
            
            # Tokenize chunk
            if self.tokenizer:
                if not self.tokenizer.is_fitted:
                    self.tokenizer.fit(chunk_df)
                
                try:
                    tokenized = self.tokenizer.transform(chunk_df)
                except Exception as e:
                    warnings.warn(f"Tokenization failed for chunk: {e}")
                    continue
                
                # Yield individual samples from chunk
                for i in range(len(tokenized.tokens)):
                    sample = {
                        'input_features': tokenized.tokens[i],
                        'attention_mask': tokenized.attention_mask[i]
                    }
                    
                    if targets is not None:
                        sample['target'] = torch.tensor(targets[i])
                    
                    if self.transform:
                        sample = self.transform(sample)
                    
                    yield sample
            else:
                # Yield raw data without tokenization
                for idx, row in chunk_df.iterrows():
                    sample = {
                        'raw_data': row.to_dict()
                    }
                    
                    if targets is not None:
                        sample['target'] = torch.tensor(targets[idx])
                    
                    if self.transform:
                        sample = self.transform(sample)
                    
                    yield sample


class CachedTabularDataset(TabularDataset):
    """Dataset with persistent caching to disk."""
    
    def __init__(
        self,
        df: pd.DataFrame,
        cache_dir: str,
        dataset_name: str,
        tokenizer: Optional[TabularTokenizer] = None,
        preprocessor: Optional[TabularPreprocessor] = None,
        target_column: Optional[str] = None,
        transform: Optional[Callable] = None,
        force_refresh: bool = False
    ):
        self.cache_dir = Path(cache_dir)
        self.dataset_name = dataset_name
        self.force_refresh = force_refresh
        
        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache file paths
        self.cache_file = self.cache_dir / f"{dataset_name}_processed.pkl"
        self.tokenized_cache_file = self.cache_dir / f"{dataset_name}_tokenized.pkl"
        
        # Try to load from cache first
        if not force_refresh and self._load_from_cache():
            # Successfully loaded from cache
            self.tokenizer = tokenizer
            self.transform = transform
        else:
            # Initialize normally and save to cache
            super().__init__(df, tokenizer, preprocessor, target_column, transform, cache_tokenized=True)
            self._save_to_cache()
    
    def _load_from_cache(self) -> bool:
        """Load dataset from cache."""
        try:
            if self.cache_file.exists():
                with open(self.cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                
                self.df = cache_data['df']
                self._preprocessed_df = cache_data['preprocessed_df']
                self.targets = cache_data['targets']
                self.target_column = cache_data['target_column']
                self.preprocessor = cache_data['preprocessor']
                
                # Load tokenized cache if available
                if self.tokenized_cache_file.exists():
                    with open(self.tokenized_cache_file, 'rb') as f:
                        self._tokenized_cache = pickle.load(f)
                else:
                    self._tokenized_cache = None
                
                return True
        except Exception as e:
            warnings.warn(f"Failed to load from cache: {e}")
        
        return False
    
    def _save_to_cache(self):
        """Save dataset to cache."""
        try:
            # Save main data
            cache_data = {
                'df': self.df,
                'preprocessed_df': self._preprocessed_df,
                'targets': self.targets,
                'target_column': self.target_column,
                'preprocessor': self.preprocessor
            }
            
            with open(self.cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            
            # Save tokenized data separately (can be large)
            if self._tokenized_cache is not None:
                with open(self.tokenized_cache_file, 'wb') as f:
                    pickle.dump(self._tokenized_cache, f)
        
        except Exception as e:
            warnings.warn(f"Failed to save to cache: {e}")


class MultiTableDataset(Dataset):
    """Dataset for handling multiple related tables."""
    
    def __init__(
        self,
        tables: Dict[str, pd.DataFrame],
        relationships: Optional[Dict[str, Dict[str, str]]] = None,
        tokenizer: Optional[TabularTokenizer] = None,
        preprocessor: Optional[TabularPreprocessor] = None,
        primary_table: Optional[str] = None,
        transform: Optional[Callable] = None
    ):
        self.tables = {name: df.copy() for name, df in tables.items()}
        self.relationships = relationships or {}
        self.tokenizer = tokenizer
        self.preprocessor = preprocessor
        self.primary_table = primary_table or list(tables.keys())[0]
        self.transform = transform
        
        # Preprocess all tables
        self.preprocessed_tables = {}
        for table_name, df in self.tables.items():
            if self.preprocessor:
                if not self.preprocessor.is_fitted:
                    self.preprocessor.fit(df)
                self.preprocessed_tables[table_name] = self.preprocessor.transform(df)
            else:
                self.preprocessed_tables[table_name] = df
        
        # Tokenize all tables
        self.tokenized_tables = {}
        if self.tokenizer:
            for table_name, df in self.preprocessed_tables.items():
                if not self.tokenizer.is_fitted:
                    self.tokenizer.fit(df)
                self.tokenized_tables[table_name] = self.tokenizer.transform(df)
    
    def __len__(self) -> int:
        return len(self.preprocessed_tables[self.primary_table])
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a sample with data from all related tables."""
        sample = {}
        
        # Get primary table data
        if self.primary_table in self.tokenized_tables:
            tokenized = self.tokenized_tables[self.primary_table]
            sample[self.primary_table] = {
                'input_features': tokenized.tokens[idx],
                'attention_mask': tokenized.attention_mask[idx]
            }
        
        # Get related table data based on relationships
        primary_row = self.preprocessed_tables[self.primary_table].iloc[idx]
        
        for table_name, relationship in self.relationships.items():
            if table_name in self.tokenized_tables:
                # Find related rows based on foreign key relationship
                foreign_key = relationship.get('foreign_key')
                primary_key = relationship.get('primary_key', foreign_key)
                
                if foreign_key and primary_key:
                    related_rows = self.preprocessed_tables[table_name][
                        self.preprocessed_tables[table_name][foreign_key] == primary_row[primary_key]
                    ]
                    
                    if len(related_rows) > 0:
                        # Use first related row (could be extended to handle multiple)
                        related_idx = related_rows.index[0]
                        tokenized = self.tokenized_tables[table_name]
                        
                        # Find the position in the tokenized data
                        original_idx = self.preprocessed_tables[table_name].index.get_loc(related_idx)
                        
                        sample[table_name] = {
                            'input_features': tokenized.tokens[original_idx],
                            'attention_mask': tokenized.attention_mask[original_idx]
                        }
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample


class TemporalTabularDataset(TabularDataset):
    """Dataset for temporal tabular data with sequence support."""
    
    def __init__(
        self,
        df: pd.DataFrame,
        timestamp_column: str,
        sequence_length: int = 10,
        prediction_horizon: int = 1,
        group_by_columns: Optional[List[str]] = None,
        tokenizer: Optional[TabularTokenizer] = None,
        preprocessor: Optional[TabularPreprocessor] = None,
        target_column: Optional[str] = None,
        transform: Optional[Callable] = None
    ):
        self.timestamp_column = timestamp_column
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.group_by_columns = group_by_columns or []
        
        # Sort by timestamp and group if needed
        df_sorted = df.sort_values([*self.group_by_columns, timestamp_column])
        
        super().__init__(
            df_sorted, tokenizer, preprocessor, target_column, transform, cache_tokenized=False
        )
        
        # Create sequence indices
        self.sequence_indices = self._create_sequence_indices()
    
    def _create_sequence_indices(self) -> List[Tuple[int, int]]:
        """Create indices for temporal sequences."""
        indices = []
        
        if self.group_by_columns:
            # Group by specified columns
            grouped = self._preprocessed_df.groupby(self.group_by_columns)
            
            for name, group in grouped:
                group_indices = group.index.tolist()
                
                # Create sequences within each group
                for i in range(len(group_indices) - self.sequence_length - self.prediction_horizon + 1):
                    start_idx = i
                    end_idx = i + self.sequence_length
                    target_idx = i + self.sequence_length + self.prediction_horizon - 1
                    
                    if target_idx < len(group_indices):
                        indices.append((
                            group_indices[start_idx:end_idx],
                            group_indices[target_idx]
                        ))
        else:
            # Single sequence
            for i in range(len(self._preprocessed_df) - self.sequence_length - self.prediction_horizon + 1):
                start_idx = i
                end_idx = i + self.sequence_length
                target_idx = i + self.sequence_length + self.prediction_horizon - 1
                
                indices.append((
                    list(range(start_idx, end_idx)),
                    target_idx
                ))
        
        return indices
    
    def __len__(self) -> int:
        return len(self.sequence_indices)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a temporal sequence sample."""
        sequence_indices, target_idx = self.sequence_indices[idx]
        
        # Get sequence data
        sequence_df = self._preprocessed_df.iloc[sequence_indices]
        
        # Tokenize sequence
        if not self.tokenizer.is_fitted:
            self.tokenizer.fit(self._preprocessed_df)
        
        tokenized_sequence = self.tokenizer.transform(sequence_df)
        
        sample = {
            'input_features': tokenized_sequence.tokens,  # [seq_len, n_features, d_model]
            'attention_mask': tokenized_sequence.attention_mask,  # [seq_len, n_features]
            'sequence_length': torch.tensor(len(sequence_indices))
        }
        
        # Add target if available
        if self.targets is not None:
            sample['target'] = torch.tensor(self.targets[target_idx])
        
        # Add temporal information
        if self.timestamp_column in self._preprocessed_df.columns:
            timestamps = self._preprocessed_df.iloc[sequence_indices][self.timestamp_column].values
            sample['timestamps'] = torch.tensor(pd.to_datetime(timestamps).astype(np.int64))
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample


def create_data_splits(
    dataset: TabularDataset,
    split_config: DataSplit
) -> Tuple[TabularDataset, TabularDataset, TabularDataset]:
    """Create train/validation/test splits from a dataset."""
    n_samples = len(dataset)
    
    # Calculate split sizes
    n_train = int(n_samples * split_config.train_ratio)
    n_val = int(n_samples * split_config.val_ratio)
    n_test = n_samples - n_train - n_val
    
    # Create random indices
    np.random.seed(split_config.random_state)
    indices = np.random.permutation(n_samples)
    
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]
    
    # Create subset datasets
    train_df = dataset._preprocessed_df.iloc[train_indices].reset_index(drop=True)
    val_df = dataset._preprocessed_df.iloc[val_indices].reset_index(drop=True)
    test_df = dataset._preprocessed_df.iloc[test_indices].reset_index(drop=True)
    
    # Create new datasets
    train_dataset = TabularDataset(
        train_df, dataset.tokenizer, dataset.preprocessor,
        dataset.target_column, dataset.transform
    )
    
    val_dataset = TabularDataset(
        val_df, dataset.tokenizer, dataset.preprocessor,
        dataset.target_column, dataset.transform
    )
    
    test_dataset = TabularDataset(
        test_df, dataset.tokenizer, dataset.preprocessor,
        dataset.target_column, dataset.transform
    )
    
    return train_dataset, val_dataset, test_dataset


def create_dataloader(
    dataset: Union[TabularDataset, StreamingTabularDataset],
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    **kwargs
) -> DataLoader:
    """Create a PyTorch DataLoader for tabular dataset."""
    
    def collate_fn(batch):
        """Custom collate function for tabular data."""
        if not batch:
            return {}
        
        # Handle different sample structures
        keys = batch[0].keys()
        collated = {}
        
        for key in keys:
            if key == 'raw_data':
                # Keep raw data as list of dicts
                collated[key] = [sample[key] for sample in batch]
            elif key in ['input_features', 'attention_mask']:
                # Stack tensor data
                tensors = [sample[key] for sample in batch]
                if len(tensors[0].shape) == 2:  # [n_features, d_model]
                    collated[key] = torch.stack(tensors)
                elif len(tensors[0].shape) == 3:  # [seq_len, n_features, d_model]
                    collated[key] = torch.stack(tensors)
                else:
                    collated[key] = torch.stack(tensors)
            elif key == 'target':
                # Handle targets
                targets = [sample[key] for sample in batch]
                collated[key] = torch.stack(targets) if isinstance(targets[0], torch.Tensor) else torch.tensor(targets)
            else:
                # Handle other keys
                values = [sample[key] for sample in batch]
                if isinstance(values[0], torch.Tensor):
                    collated[key] = torch.stack(values)
                else:
                    collated[key] = values
        
        return collated
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        **kwargs
    )