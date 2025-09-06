"""Dataset loading and management for benchmarking."""

import os
import logging
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
from dataclasses import dataclass
from pathlib import Path
import warnings

import pandas as pd
import numpy as np
from sklearn.datasets import make_classification, make_regression

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkDataset:
    """Container for benchmark dataset information."""
    
    name: str
    description: str
    task_type: str  # classification, regression, anomaly_detection
    n_samples: int
    n_features: int
    n_classes: Optional[int] = None
    target_names: Optional[List[str]] = None
    feature_names: Optional[List[str]] = None
    data_url: Optional[str] = None
    citation: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'description': self.description,
            'task_type': self.task_type,
            'n_samples': self.n_samples,
            'n_features': self.n_features,
            'n_classes': self.n_classes,
            'target_names': self.target_names,
            'feature_names': self.feature_names,
            'data_url': self.data_url,
            'citation': self.citation
        }


class TabularDatasetLoader:
    """Loader for tabular datasets from various sources."""
    
    def __init__(self, cache_dir: str = "./data_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._dataset_registry = {}
        self._register_builtin_datasets()
    
    def _register_builtin_datasets(self):
        """Register built-in datasets."""
        
        # OpenML CC18 datasets (subset)
        openml_datasets = [
            {
                'name': 'adult',
                'description': 'Adult income prediction dataset',
                'task_type': 'classification',
                'openml_id': 1590,
                'n_samples': 48842,
                'n_features': 14,
                'n_classes': 2
            },
            {
                'name': 'credit-g',
                'description': 'German credit risk dataset',
                'task_type': 'classification',
                'openml_id': 31,
                'n_samples': 1000,
                'n_features': 20,
                'n_classes': 2
            },
            {
                'name': 'diabetes',
                'description': 'Pima Indians diabetes dataset',
                'task_type': 'classification',
                'openml_id': 37,
                'n_samples': 768,
                'n_features': 8,
                'n_classes': 2
            },
            {
                'name': 'titanic',
                'description': 'Titanic survival prediction',
                'task_type': 'classification',
                'openml_id': 40945,
                'n_samples': 1309,
                'n_features': 13,
                'n_classes': 2
            },
            {
                'name': 'wine-quality-red',
                'description': 'Red wine quality dataset',
                'task_type': 'regression',
                'openml_id': 287,
                'n_samples': 1599,
                'n_features': 11,
                'n_classes': None
            }
        ]
        
        for dataset_info in openml_datasets:
            self._dataset_registry[dataset_info['name']] = dataset_info
    
    def list_datasets(self, task_type: Optional[str] = None) -> List[str]:
        """List available datasets."""
        if task_type is None:
            return list(self._dataset_registry.keys())
        else:
            return [
                name for name, info in self._dataset_registry.items()
                if info['task_type'] == task_type
            ]
    
    def get_dataset_info(self, name: str) -> BenchmarkDataset:
        """Get information about a dataset."""
        if name not in self._dataset_registry:
            raise ValueError(f"Dataset '{name}' not found")
        
        info = self._dataset_registry[name]
        return BenchmarkDataset(**info)
    
    def load_dataset(
        self,
        name: str,
        return_X_y: bool = True,
        cache: bool = True
    ) -> Union[Tuple[pd.DataFrame, pd.Series], pd.DataFrame]:
        """Load a dataset."""
        
        if name not in self._dataset_registry:
            raise ValueError(f"Dataset '{name}' not found")
        
        # Check cache first
        cache_file = self.cache_dir / f"{name}.csv"
        if cache and cache_file.exists():
            logger.info(f"Loading {name} from cache")
            df = pd.read_csv(cache_file)
            
            if return_X_y:
                # Assume last column is target
                X = df.iloc[:, :-1]
                y = df.iloc[:, -1]
                return X, y
            else:
                return df
        
        # Load from source
        dataset_info = self._dataset_registry[name]
        
        try:
            # Try OpenML first
            if 'openml_id' in dataset_info:
                df = self._load_from_openml(dataset_info['openml_id'])
            else:
                # Try other sources
                df = self._load_from_url(dataset_info.get('data_url'))
            
            # Cache the dataset
            if cache:
                df.to_csv(cache_file, index=False)
                logger.info(f"Cached {name} to {cache_file}")
            
            if return_X_y:
                # Assume last column is target
                X = df.iloc[:, :-1]
                y = df.iloc[:, -1]
                return X, y
            else:
                return df
                
        except Exception as e:
            logger.error(f"Failed to load dataset {name}: {e}")
            # Return synthetic data as fallback
            logger.info(f"Generating synthetic data for {name}")
            return self._generate_synthetic_fallback(dataset_info, return_X_y)
    
    def _load_from_openml(self, dataset_id: int) -> pd.DataFrame:
        """Load dataset from OpenML."""
        try:
            import openml
            
            dataset = openml.datasets.get_dataset(dataset_id)
            X, y, categorical_indicator, attribute_names = dataset.get_data(
                dataset_format="dataframe",
                target=dataset.default_target_attribute
            )
            
            # Combine features and target
            df = X.copy()
            df['target'] = y
            
            return df
            
        except ImportError:
            raise ImportError("openml package required for OpenML datasets")
    
    def _load_from_url(self, url: str) -> pd.DataFrame:
        """Load dataset from URL."""
        if url is None:
            raise ValueError("No URL provided")
        
        return pd.read_csv(url)
    
    def _generate_synthetic_fallback(
        self,
        dataset_info: Dict[str, Any],
        return_X_y: bool = True
    ) -> Union[Tuple[pd.DataFrame, pd.Series], pd.DataFrame]:
        """Generate synthetic data as fallback."""
        
        n_samples = dataset_info.get('n_samples', 1000)
        n_features = dataset_info.get('n_features', 10)
        task_type = dataset_info.get('task_type', 'classification')
        
        if task_type == 'classification':
            n_classes = dataset_info.get('n_classes', 2)
            X, y = make_classification(
                n_samples=n_samples,
                n_features=n_features,
                n_classes=n_classes,
                n_informative=min(n_features, n_features // 2),
                n_redundant=0,
                random_state=42
            )
        else:
            X, y = make_regression(
                n_samples=n_samples,
                n_features=n_features,
                noise=0.1,
                random_state=42
            )
        
        # Convert to DataFrame
        feature_names = [f'feature_{i}' for i in range(n_features)]
        X_df = pd.DataFrame(X, columns=feature_names)
        y_series = pd.Series(y, name='target')
        
        if return_X_y:
            return X_df, y_series
        else:
            df = X_df.copy()
            df['target'] = y_series
            return df
    
    def register_dataset(
        self,
        name: str,
        loader_func: Callable[[], Tuple[pd.DataFrame, pd.Series]],
        description: str = "",
        task_type: str = "classification",
        **kwargs
    ):
        """Register a custom dataset."""
        
        # Get dataset info by calling the loader
        try:
            X, y = loader_func()
            n_samples, n_features = X.shape
            n_classes = len(np.unique(y)) if task_type == 'classification' else None
        except Exception as e:
            logger.warning(f"Could not get dataset info for {name}: {e}")
            n_samples = n_features = n_classes = None
        
        dataset_info = {
            'name': name,
            'description': description,
            'task_type': task_type,
            'n_samples': n_samples,
            'n_features': n_features,
            'n_classes': n_classes,
            'loader_func': loader_func,
            **kwargs
        }
        
        self._dataset_registry[name] = dataset_info
        logger.info(f"Registered custom dataset: {name}")


class SyntheticDataGenerator:
    """Generate synthetic datasets for testing and benchmarking."""
    
    @staticmethod
    def generate_classification_dataset(
        n_samples: int = 1000,
        n_features: int = 20,
        n_classes: int = 2,
        n_informative: Optional[int] = None,
        n_redundant: int = 0,
        n_clusters_per_class: int = 1,
        class_sep: float = 1.0,
        flip_y: float = 0.01,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Generate synthetic classification dataset."""
        
        if n_informative is None:
            n_informative = min(n_features, n_features // 2)
        
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_informative,
            n_redundant=n_redundant,
            n_classes=n_classes,
            n_clusters_per_class=n_clusters_per_class,
            class_sep=class_sep,
            flip_y=flip_y,
            random_state=random_state
        )
        
        # Convert to DataFrame
        feature_names = [f'feature_{i}' for i in range(n_features)]
        X_df = pd.DataFrame(X, columns=feature_names)
        y_series = pd.Series(y, name='target')
        
        return X_df, y_series
    
    @staticmethod
    def generate_regression_dataset(
        n_samples: int = 1000,
        n_features: int = 20,
        n_informative: Optional[int] = None,
        n_targets: int = 1,
        noise: float = 0.1,
        bias: float = 0.0,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Generate synthetic regression dataset."""
        
        if n_informative is None:
            n_informative = min(n_features, n_features // 2)
        
        X, y = make_regression(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_informative,
            n_targets=n_targets,
            noise=noise,
            bias=bias,
            random_state=random_state
        )
        
        # Convert to DataFrame
        feature_names = [f'feature_{i}' for i in range(n_features)]
        X_df = pd.DataFrame(X, columns=feature_names)
        
        if n_targets == 1:
            y_series = pd.Series(y, name='target')
        else:
            y_series = pd.DataFrame(y, columns=[f'target_{i}' for i in range(n_targets)])
        
        return X_df, y_series
    
    @staticmethod
    def generate_mixed_types_dataset(
        n_samples: int = 1000,
        n_numerical: int = 10,
        n_categorical: int = 5,
        n_classes: int = 2,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Generate dataset with mixed data types."""
        
        np.random.seed(random_state)
        
        # Generate numerical features
        X_num = np.random.randn(n_samples, n_numerical)
        
        # Generate categorical features
        X_cat = np.random.randint(0, 5, size=(n_samples, n_categorical))
        
        # Combine features
        X = np.column_stack([X_num, X_cat])
        
        # Generate target based on features
        weights = np.random.randn(X.shape[1])
        logits = X @ weights
        probabilities = 1 / (1 + np.exp(-logits))
        y = (probabilities > 0.5).astype(int)
        
        # Convert to DataFrame with appropriate types
        feature_names = (
            [f'num_feature_{i}' for i in range(n_numerical)] +
            [f'cat_feature_{i}' for i in range(n_categorical)]
        )
        
        X_df = pd.DataFrame(X, columns=feature_names)
        
        # Convert categorical columns to strings
        for i in range(n_numerical, n_numerical + n_categorical):
            col_name = feature_names[i]
            X_df[col_name] = X_df[col_name].astype(str)
        
        y_series = pd.Series(y, name='target')
        
        return X_df, y_series


def load_benchmark_datasets(
    dataset_names: Optional[List[str]] = None,
    task_type: Optional[str] = None,
    cache_dir: str = "./data_cache"
) -> Dict[str, Tuple[pd.DataFrame, pd.Series]]:
    """Load multiple benchmark datasets."""
    
    loader = TabularDatasetLoader(cache_dir)
    
    if dataset_names is None:
        dataset_names = loader.list_datasets(task_type)
    
    datasets = {}
    for name in dataset_names:
        try:
            logger.info(f"Loading dataset: {name}")
            X, y = loader.load_dataset(name)
            datasets[name] = (X, y)
        except Exception as e:
            logger.error(f"Failed to load dataset {name}: {e}")
            continue
    
    return datasets


def create_synthetic_dataset(
    task_type: str = "classification",
    n_samples: int = 1000,
    n_features: int = 20,
    **kwargs
) -> Tuple[pd.DataFrame, pd.Series]:
    """Create a synthetic dataset."""
    
    generator = SyntheticDataGenerator()
    
    if task_type == "classification":
        return generator.generate_classification_dataset(
            n_samples=n_samples,
            n_features=n_features,
            **kwargs
        )
    elif task_type == "regression":
        return generator.generate_regression_dataset(
            n_samples=n_samples,
            n_features=n_features,
            **kwargs
        )
    elif task_type == "mixed":
        return generator.generate_mixed_types_dataset(
            n_samples=n_samples,
            **kwargs
        )
    else:
        raise ValueError(f"Unsupported task type: {task_type}")


# Predefined dataset collections
OPENML_CC18_CLASSIFICATION = [
    'adult', 'credit-g', 'diabetes', 'titanic'
]

OPENML_CC18_REGRESSION = [
    'wine-quality-red'
]

UCI_DATASETS = [
    'adult', 'diabetes', 'titanic'
]