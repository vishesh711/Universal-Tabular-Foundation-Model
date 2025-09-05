"""Data loaders for various tabular data formats and sources."""

import os
import json
import pickle
import warnings
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Tuple, Any, Iterator
from pathlib import Path
import pandas as pd
import numpy as np
from dataclasses import dataclass
from enum import Enum

try:
    import openml
    OPENML_AVAILABLE = True
except ImportError:
    OPENML_AVAILABLE = False
    warnings.warn("OpenML not available. Install with: pip install openml")

try:
    import kaggle
    KAGGLE_AVAILABLE = True
except ImportError:
    KAGGLE_AVAILABLE = False
    warnings.warn("Kaggle API not available. Install with: pip install kaggle")

try:
    import pyarrow.parquet as pq
    PARQUET_AVAILABLE = True
except ImportError:
    PARQUET_AVAILABLE = False
    warnings.warn("Parquet support not available. Install with: pip install pyarrow")


class DataFormat(Enum):
    """Supported data formats."""
    CSV = "csv"
    PARQUET = "parquet"
    JSON = "json"
    EXCEL = "excel"
    PICKLE = "pickle"
    FEATHER = "feather"
    HDF5 = "hdf5"
    SQL = "sql"


@dataclass
class DatasetInfo:
    """Information about a dataset."""
    name: str
    source: str
    format: DataFormat
    size: int
    n_rows: int
    n_columns: int
    target_column: Optional[str] = None
    task_type: Optional[str] = None
    description: Optional[str] = None
    url: Optional[str] = None
    local_path: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class BaseDataLoader(ABC):
    """Base class for data loaders."""
    
    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = Path(cache_dir) if cache_dir else Path.home() / ".tabgpt" / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    @abstractmethod
    def load(self, identifier: str, **kwargs) -> Tuple[pd.DataFrame, DatasetInfo]:
        """Load dataset by identifier."""
        pass
    
    @abstractmethod
    def list_datasets(self, **kwargs) -> List[DatasetInfo]:
        """List available datasets."""
        pass
    
    def _get_cache_path(self, identifier: str, format: str = "parquet") -> Path:
        """Get cache file path for dataset."""
        safe_name = "".join(c for c in identifier if c.isalnum() or c in ('-', '_'))
        return self.cache_dir / f"{safe_name}.{format}"
    
    def _load_from_cache(self, cache_path: Path) -> Optional[Tuple[pd.DataFrame, DatasetInfo]]:
        """Load dataset from cache if available."""
        if cache_path.exists():
            try:
                if cache_path.suffix == ".parquet":
                    df = pd.read_parquet(cache_path)
                elif cache_path.suffix == ".pickle":
                    with open(cache_path, 'rb') as f:
                        data = pickle.load(f)
                        df, info = data['dataframe'], data['info']
                        return df, info
                else:
                    return None
                
                # Load metadata
                metadata_path = cache_path.with_suffix('.json')
                if metadata_path.exists():
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                        info = DatasetInfo(**metadata)
                        return df, info
                
                return df, None
            except Exception as e:
                warnings.warn(f"Failed to load from cache: {e}")
                return None
        return None
    
    def _save_to_cache(self, df: pd.DataFrame, info: DatasetInfo, cache_path: Path):
        """Save dataset to cache."""
        try:
            # Save dataframe
            if cache_path.suffix == ".parquet" and PARQUET_AVAILABLE:
                df.to_parquet(cache_path, index=False)
            else:
                cache_path = cache_path.with_suffix('.pickle')
                with open(cache_path, 'wb') as f:
                    pickle.dump({'dataframe': df, 'info': info}, f)
            
            # Save metadata
            metadata_path = cache_path.with_suffix('.json')
            with open(metadata_path, 'w') as f:
                # Convert dataclass to dict, handling non-serializable fields
                metadata = {
                    'name': info.name,
                    'source': info.source,
                    'format': info.format.value if isinstance(info.format, DataFormat) else str(info.format),
                    'size': info.size,
                    'n_rows': info.n_rows,
                    'n_columns': info.n_columns,
                    'target_column': info.target_column,
                    'task_type': info.task_type,
                    'description': info.description,
                    'url': info.url,
                    'local_path': str(info.local_path) if info.local_path else None,
                    'metadata': info.metadata
                }
                json.dump(metadata, f, indent=2)
                
        except Exception as e:
            warnings.warn(f"Failed to save to cache: {e}")


class CSVLoader(BaseDataLoader):
    """Loader for CSV files."""
    
    def load(self, file_path: str, **kwargs) -> Tuple[pd.DataFrame, DatasetInfo]:
        """Load CSV file."""
        path = Path(file_path)
        
        # Check cache first
        cache_path = self._get_cache_path(f"csv_{path.stem}")
        cached = self._load_from_cache(cache_path)
        if cached:
            return cached
        
        # Load CSV
        df = pd.read_csv(file_path, **kwargs)
        
        # Create dataset info
        info = DatasetInfo(
            name=path.stem,
            source="local_csv",
            format=DataFormat.CSV,
            size=path.stat().st_size if path.exists() else 0,
            n_rows=len(df),
            n_columns=len(df.columns),
            local_path=str(path.absolute()),
            metadata={'columns': list(df.columns), 'dtypes': df.dtypes.to_dict()}
        )
        
        # Cache the result
        self._save_to_cache(df, info, cache_path)
        
        return df, info
    
    def list_datasets(self, directory: str = ".") -> List[DatasetInfo]:
        """List CSV files in directory."""
        csv_files = []
        for file_path in Path(directory).glob("*.csv"):
            try:
                # Quick scan for basic info
                df_sample = pd.read_csv(file_path, nrows=1)
                info = DatasetInfo(
                    name=file_path.stem,
                    source="local_csv",
                    format=DataFormat.CSV,
                    size=file_path.stat().st_size,
                    n_rows=0,  # Would need full scan
                    n_columns=len(df_sample.columns),
                    local_path=str(file_path.absolute())
                )
                csv_files.append(info)
            except Exception as e:
                warnings.warn(f"Failed to scan {file_path}: {e}")
        
        return csv_files


class ParquetLoader(BaseDataLoader):
    """Loader for Parquet files."""
    
    def load(self, file_path: str, **kwargs) -> Tuple[pd.DataFrame, DatasetInfo]:
        """Load Parquet file."""
        if not PARQUET_AVAILABLE:
            raise ImportError("Parquet support requires pyarrow: pip install pyarrow")
        
        path = Path(file_path)
        
        # Load Parquet
        df = pd.read_parquet(file_path, **kwargs)
        
        # Create dataset info
        info = DatasetInfo(
            name=path.stem,
            source="local_parquet",
            format=DataFormat.PARQUET,
            size=path.stat().st_size if path.exists() else 0,
            n_rows=len(df),
            n_columns=len(df.columns),
            local_path=str(path.absolute()),
            metadata={'columns': list(df.columns), 'dtypes': df.dtypes.to_dict()}
        )
        
        return df, info
    
    def list_datasets(self, directory: str = ".") -> List[DatasetInfo]:
        """List Parquet files in directory."""
        parquet_files = []
        for file_path in Path(directory).glob("*.parquet"):
            try:
                # Get metadata without loading full file
                if PARQUET_AVAILABLE:
                    parquet_file = pq.ParquetFile(file_path)
                    n_rows = parquet_file.metadata.num_rows
                    n_columns = parquet_file.metadata.num_columns
                else:
                    n_rows, n_columns = 0, 0
                
                info = DatasetInfo(
                    name=file_path.stem,
                    source="local_parquet",
                    format=DataFormat.PARQUET,
                    size=file_path.stat().st_size,
                    n_rows=n_rows,
                    n_columns=n_columns,
                    local_path=str(file_path.absolute())
                )
                parquet_files.append(info)
            except Exception as e:
                warnings.warn(f"Failed to scan {file_path}: {e}")
        
        return parquet_files


class JSONLoader(BaseDataLoader):
    """Loader for JSON files."""
    
    def load(self, file_path: str, **kwargs) -> Tuple[pd.DataFrame, DatasetInfo]:
        """Load JSON file."""
        path = Path(file_path)
        
        # Load JSON
        df = pd.read_json(file_path, **kwargs)
        
        # Create dataset info
        info = DatasetInfo(
            name=path.stem,
            source="local_json",
            format=DataFormat.JSON,
            size=path.stat().st_size if path.exists() else 0,
            n_rows=len(df),
            n_columns=len(df.columns),
            local_path=str(path.absolute()),
            metadata={'columns': list(df.columns), 'dtypes': df.dtypes.to_dict()}
        )
        
        return df, info
    
    def list_datasets(self, directory: str = ".") -> List[DatasetInfo]:
        """List JSON files in directory."""
        json_files = []
        for file_path in Path(directory).glob("*.json"):
            try:
                info = DatasetInfo(
                    name=file_path.stem,
                    source="local_json",
                    format=DataFormat.JSON,
                    size=file_path.stat().st_size,
                    n_rows=0,  # Would need full scan
                    n_columns=0,
                    local_path=str(file_path.absolute())
                )
                json_files.append(info)
            except Exception as e:
                warnings.warn(f"Failed to scan {file_path}: {e}")
        
        return json_files


class ExcelLoader(BaseDataLoader):
    """Loader for Excel files."""
    
    def load(self, file_path: str, sheet_name: Union[str, int] = 0, **kwargs) -> Tuple[pd.DataFrame, DatasetInfo]:
        """Load Excel file."""
        path = Path(file_path)
        
        # Load Excel
        df = pd.read_excel(file_path, sheet_name=sheet_name, **kwargs)
        
        # Create dataset info
        info = DatasetInfo(
            name=f"{path.stem}_{sheet_name}",
            source="local_excel",
            format=DataFormat.EXCEL,
            size=path.stat().st_size if path.exists() else 0,
            n_rows=len(df),
            n_columns=len(df.columns),
            local_path=str(path.absolute()),
            metadata={
                'sheet_name': sheet_name,
                'columns': list(df.columns),
                'dtypes': df.dtypes.to_dict()
            }
        )
        
        return df, info
    
    def list_datasets(self, directory: str = ".") -> List[DatasetInfo]:
        """List Excel files in directory."""
        excel_files = []
        for file_path in Path(directory).glob("*.xlsx"):
            try:
                info = DatasetInfo(
                    name=file_path.stem,
                    source="local_excel",
                    format=DataFormat.EXCEL,
                    size=file_path.stat().st_size,
                    n_rows=0,  # Would need full scan
                    n_columns=0,
                    local_path=str(file_path.absolute())
                )
                excel_files.append(info)
            except Exception as e:
                warnings.warn(f"Failed to scan {file_path}: {e}")
        
        return excel_files


class OpenMLLoader(BaseDataLoader):
    """Loader for OpenML datasets."""
    
    def __init__(self, cache_dir: Optional[str] = None, api_key: Optional[str] = None):
        super().__init__(cache_dir)
        if not OPENML_AVAILABLE:
            raise ImportError("OpenML support requires openml: pip install openml")
        
        if api_key:
            openml.config.apikey = api_key
    
    def load(self, dataset_id: Union[int, str], **kwargs) -> Tuple[pd.DataFrame, DatasetInfo]:
        """Load OpenML dataset."""
        # Check cache first
        cache_path = self._get_cache_path(f"openml_{dataset_id}")
        cached = self._load_from_cache(cache_path)
        if cached:
            return cached
        
        # Load from OpenML
        dataset = openml.datasets.get_dataset(dataset_id)
        X, y, categorical_indicator, attribute_names = dataset.get_data(
            dataset_format="dataframe",
            target=dataset.default_target_attribute
        )
        
        # Combine features and target
        if y is not None:
            df = pd.concat([X, y], axis=1)
        else:
            df = X
        
        # Create dataset info
        info = DatasetInfo(
            name=dataset.name,
            source="openml",
            format=DataFormat.CSV,  # OpenML provides CSV-like data
            size=0,  # Not available from OpenML API
            n_rows=len(df),
            n_columns=len(df.columns),
            target_column=dataset.default_target_attribute,
            task_type=self._infer_task_type(dataset),
            description=dataset.description,
            url=dataset.url,
            metadata={
                'dataset_id': dataset_id,
                'version': dataset.version,
                'categorical_indicator': categorical_indicator,
                'attribute_names': attribute_names,
                'qualities': dataset.qualities
            }
        )
        
        # Cache the result
        self._save_to_cache(df, info, cache_path)
        
        return df, info
    
    def list_datasets(self, limit: int = 100, **filters) -> List[DatasetInfo]:
        """List OpenML datasets."""
        datasets = openml.datasets.list_datasets(output_format='dataframe', **filters)
        
        dataset_infos = []
        for idx, (dataset_id, row) in enumerate(datasets.head(limit).iterrows()):
            info = DatasetInfo(
                name=row['name'],
                source="openml",
                format=DataFormat.CSV,
                size=0,
                n_rows=row.get('NumberOfInstances', 0),
                n_columns=row.get('NumberOfFeatures', 0),
                target_column=row.get('DefaultTargetAttribute'),
                description=row.get('description', ''),
                metadata={'dataset_id': dataset_id}
            )
            dataset_infos.append(info)
        
        return dataset_infos
    
    def _infer_task_type(self, dataset) -> str:
        """Infer task type from OpenML dataset."""
        if hasattr(dataset, 'qualities') and dataset.qualities:
            if 'NumberOfClasses' in dataset.qualities:
                n_classes = dataset.qualities['NumberOfClasses']
                if n_classes == 2:
                    return 'binary_classification'
                elif n_classes > 2:
                    return 'multiclass_classification'
            elif 'NumberOfNumericFeatures' in dataset.qualities:
                return 'regression'
        return 'unknown'


class UCILoader(BaseDataLoader):
    """Loader for UCI ML Repository datasets."""
    
    def __init__(self, cache_dir: Optional[str] = None):
        super().__init__(cache_dir)
        # UCI dataset registry (subset of popular datasets)
        self.uci_datasets = {
            'iris': {
                'url': 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',
                'columns': ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class'],
                'target': 'class',
                'task_type': 'multiclass_classification'
            },
            'wine': {
                'url': 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data',
                'columns': ['class'] + [f'feature_{i}' for i in range(1, 14)],
                'target': 'class',
                'task_type': 'multiclass_classification'
            },
            'boston': {
                'url': 'https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data',
                'columns': ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV'],
                'target': 'MEDV',
                'task_type': 'regression',
                'sep': r'\s+'
            }
        }
    
    def load(self, dataset_name: str, **kwargs) -> Tuple[pd.DataFrame, DatasetInfo]:
        """Load UCI dataset."""
        if dataset_name not in self.uci_datasets:
            raise ValueError(f"Dataset {dataset_name} not available. Available: {list(self.uci_datasets.keys())}")
        
        # Check cache first
        cache_path = self._get_cache_path(f"uci_{dataset_name}")
        cached = self._load_from_cache(cache_path)
        if cached:
            return cached
        
        dataset_config = self.uci_datasets[dataset_name]
        
        # Load from URL
        df = pd.read_csv(
            dataset_config['url'],
            names=dataset_config['columns'],
            sep=dataset_config.get('sep', ','),
            **kwargs
        )
        
        # Create dataset info
        info = DatasetInfo(
            name=dataset_name,
            source="uci",
            format=DataFormat.CSV,
            size=0,  # Not available
            n_rows=len(df),
            n_columns=len(df.columns),
            target_column=dataset_config['target'],
            task_type=dataset_config['task_type'],
            description=f"UCI ML Repository - {dataset_name}",
            url=dataset_config['url'],
            metadata={'uci_config': dataset_config}
        )
        
        # Cache the result
        self._save_to_cache(df, info, cache_path)
        
        return df, info
    
    def list_datasets(self) -> List[DatasetInfo]:
        """List available UCI datasets."""
        dataset_infos = []
        for name, config in self.uci_datasets.items():
            info = DatasetInfo(
                name=name,
                source="uci",
                format=DataFormat.CSV,
                size=0,
                n_rows=0,  # Would need to load to get exact count
                n_columns=len(config['columns']),
                target_column=config['target'],
                task_type=config['task_type'],
                description=f"UCI ML Repository - {name}",
                url=config['url']
            )
            dataset_infos.append(info)
        
        return dataset_infos


class KaggleLoader(BaseDataLoader):
    """Loader for Kaggle datasets."""
    
    def __init__(self, cache_dir: Optional[str] = None):
        super().__init__(cache_dir)
        if not KAGGLE_AVAILABLE:
            raise ImportError("Kaggle support requires kaggle: pip install kaggle")
    
    def load(self, dataset_name: str, file_name: Optional[str] = None, **kwargs) -> Tuple[pd.DataFrame, DatasetInfo]:
        """Load Kaggle dataset."""
        # Check cache first
        cache_path = self._get_cache_path(f"kaggle_{dataset_name.replace('/', '_')}")
        cached = self._load_from_cache(cache_path)
        if cached:
            return cached
        
        # Download from Kaggle
        download_path = self.cache_dir / "kaggle_downloads" / dataset_name
        download_path.mkdir(parents=True, exist_ok=True)
        
        try:
            kaggle.api.dataset_download_files(dataset_name, path=download_path, unzip=True)
        except Exception as e:
            raise RuntimeError(f"Failed to download Kaggle dataset {dataset_name}: {e}")
        
        # Find CSV files in download
        csv_files = list(download_path.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in Kaggle dataset {dataset_name}")
        
        # Use specified file or first CSV
        if file_name:
            csv_file = download_path / file_name
            if not csv_file.exists():
                raise FileNotFoundError(f"File {file_name} not found in dataset")
        else:
            csv_file = csv_files[0]
        
        # Load CSV
        df = pd.read_csv(csv_file, **kwargs)
        
        # Create dataset info
        info = DatasetInfo(
            name=f"{dataset_name}_{csv_file.stem}",
            source="kaggle",
            format=DataFormat.CSV,
            size=csv_file.stat().st_size,
            n_rows=len(df),
            n_columns=len(df.columns),
            description=f"Kaggle dataset - {dataset_name}",
            local_path=str(csv_file),
            metadata={
                'kaggle_dataset': dataset_name,
                'file_name': csv_file.name,
                'available_files': [f.name for f in download_path.glob("*")]
            }
        )
        
        # Cache the result
        self._save_to_cache(df, info, cache_path)
        
        return df, info
    
    def list_datasets(self, search: Optional[str] = None, limit: int = 20) -> List[DatasetInfo]:
        """List Kaggle datasets."""
        try:
            datasets = kaggle.api.dataset_list(search=search, max_size=limit)
            
            dataset_infos = []
            for dataset in datasets:
                info = DatasetInfo(
                    name=dataset.ref,
                    source="kaggle",
                    format=DataFormat.CSV,  # Assume CSV for simplicity
                    size=dataset.totalBytes,
                    n_rows=0,  # Not available from API
                    n_columns=0,
                    description=dataset.title,
                    url=f"https://www.kaggle.com/datasets/{dataset.ref}",
                    metadata={
                        'owner': dataset.ownerName,
                        'license': dataset.licenseName,
                        'download_count': dataset.downloadCount,
                        'vote_count': dataset.voteCount
                    }
                )
                dataset_infos.append(info)
            
            return dataset_infos
        except Exception as e:
            warnings.warn(f"Failed to list Kaggle datasets: {e}")
            return []


class TabularDataLoader:
    """Unified data loader for multiple formats and sources."""
    
    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = cache_dir
        
        # Initialize loaders
        self.loaders = {
            'csv': CSVLoader(cache_dir),
            'parquet': ParquetLoader(cache_dir),
            'json': JSONLoader(cache_dir),
            'excel': ExcelLoader(cache_dir),
        }
        
        # Optional loaders (require additional dependencies)
        try:
            self.loaders['openml'] = OpenMLLoader(cache_dir)
        except ImportError:
            pass
        
        try:
            self.loaders['uci'] = UCILoader(cache_dir)
        except ImportError:
            pass
        
        try:
            self.loaders['kaggle'] = KaggleLoader(cache_dir)
        except ImportError:
            pass
    
    def load(self, source: str, identifier: str, **kwargs) -> Tuple[pd.DataFrame, DatasetInfo]:
        """Load dataset from specified source."""
        if source not in self.loaders:
            raise ValueError(f"Source {source} not available. Available: {list(self.loaders.keys())}")
        
        return self.loaders[source].load(identifier, **kwargs)
    
    def load_auto(self, path_or_identifier: str, **kwargs) -> Tuple[pd.DataFrame, DatasetInfo]:
        """Auto-detect format and load dataset."""
        path = Path(path_or_identifier)
        
        # Check if it's a local file
        if path.exists():
            suffix = path.suffix.lower()
            if suffix == '.csv':
                return self.loaders['csv'].load(str(path), **kwargs)
            elif suffix == '.parquet':
                return self.loaders['parquet'].load(str(path), **kwargs)
            elif suffix == '.json':
                return self.loaders['json'].load(str(path), **kwargs)
            elif suffix in ['.xlsx', '.xls']:
                return self.loaders['excel'].load(str(path), **kwargs)
            else:
                raise ValueError(f"Unsupported file format: {suffix}")
        
        # Try to parse as dataset identifier
        if path_or_identifier.isdigit() and 'openml' in self.loaders:
            # Looks like OpenML dataset ID
            return self.loaders['openml'].load(int(path_or_identifier), **kwargs)
        elif '/' in path_or_identifier and 'kaggle' in self.loaders:
            # Looks like Kaggle dataset
            return self.loaders['kaggle'].load(path_or_identifier, **kwargs)
        elif path_or_identifier in ['iris', 'wine', 'boston'] and 'uci' in self.loaders:
            # Known UCI dataset
            return self.loaders['uci'].load(path_or_identifier, **kwargs)
        
        raise ValueError(f"Could not auto-detect format for: {path_or_identifier}")
    
    def list_datasets(self, source: Optional[str] = None, **kwargs) -> Dict[str, List[DatasetInfo]]:
        """List available datasets from all or specified sources."""
        if source:
            if source not in self.loaders:
                raise ValueError(f"Source {source} not available")
            return {source: self.loaders[source].list_datasets(**kwargs)}
        
        all_datasets = {}
        for source_name, loader in self.loaders.items():
            try:
                all_datasets[source_name] = loader.list_datasets(**kwargs)
            except Exception as e:
                warnings.warn(f"Failed to list datasets from {source_name}: {e}")
                all_datasets[source_name] = []
        
        return all_datasets


class DatasetRegistry:
    """Registry for managing and discovering datasets."""
    
    def __init__(self, registry_file: Optional[str] = None):
        self.registry_file = Path(registry_file) if registry_file else Path.home() / ".tabgpt" / "registry.json"
        self.registry_file.parent.mkdir(parents=True, exist_ok=True)
        
        self.datasets = self._load_registry()
        self.loader = TabularDataLoader()
    
    def _load_registry(self) -> Dict[str, DatasetInfo]:
        """Load dataset registry from file."""
        if self.registry_file.exists():
            try:
                with open(self.registry_file, 'r') as f:
                    data = json.load(f)
                    return {
                        name: DatasetInfo(**info) for name, info in data.items()
                    }
            except Exception as e:
                warnings.warn(f"Failed to load registry: {e}")
        
        return {}
    
    def _save_registry(self):
        """Save dataset registry to file."""
        try:
            data = {}
            for name, info in self.datasets.items():
                data[name] = {
                    'name': info.name,
                    'source': info.source,
                    'format': info.format.value if isinstance(info.format, DataFormat) else str(info.format),
                    'size': info.size,
                    'n_rows': info.n_rows,
                    'n_columns': info.n_columns,
                    'target_column': info.target_column,
                    'task_type': info.task_type,
                    'description': info.description,
                    'url': info.url,
                    'local_path': info.local_path,
                    'metadata': info.metadata
                }
            
            with open(self.registry_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            warnings.warn(f"Failed to save registry: {e}")
    
    def register_dataset(self, name: str, source: str, identifier: str, **metadata):
        """Register a dataset in the registry."""
        try:
            df, info = self.loader.load(source, identifier)
            
            # Update with provided metadata
            for key, value in metadata.items():
                if hasattr(info, key):
                    setattr(info, key, value)
            
            self.datasets[name] = info
            self._save_registry()
            
            return info
        except Exception as e:
            raise RuntimeError(f"Failed to register dataset {name}: {e}")
    
    def get_dataset(self, name: str) -> Tuple[pd.DataFrame, DatasetInfo]:
        """Get dataset by registered name."""
        if name not in self.datasets:
            raise KeyError(f"Dataset {name} not found in registry")
        
        info = self.datasets[name]
        
        # Load dataset
        if info.local_path and Path(info.local_path).exists():
            return self.loader.load_auto(info.local_path)
        elif info.source and info.metadata:
            # Try to reload from original source
            if info.source == 'openml' and 'dataset_id' in info.metadata:
                return self.loader.load('openml', info.metadata['dataset_id'])
            elif info.source == 'kaggle' and 'kaggle_dataset' in info.metadata:
                return self.loader.load('kaggle', info.metadata['kaggle_dataset'])
            elif info.source == 'uci':
                return self.loader.load('uci', info.name)
        
        raise RuntimeError(f"Could not load dataset {name}")
    
    def list_registered(self) -> List[DatasetInfo]:
        """List all registered datasets."""
        return list(self.datasets.values())
    
    def remove_dataset(self, name: str):
        """Remove dataset from registry."""
        if name in self.datasets:
            del self.datasets[name]
            self._save_registry()
    
    def search_datasets(self, query: str) -> List[DatasetInfo]:
        """Search registered datasets by name or description."""
        query_lower = query.lower()
        results = []
        
        for info in self.datasets.values():
            if (query_lower in info.name.lower() or 
                (info.description and query_lower in info.description.lower())):
                results.append(info)
        
        return results