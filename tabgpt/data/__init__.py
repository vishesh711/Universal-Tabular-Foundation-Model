"""Data loading and preprocessing utilities for TabGPT."""

from .loaders import (
    TabularDataLoader,
    OpenMLLoader,
    UCILoader,
    KaggleLoader,
    CSVLoader,
    ParquetLoader,
    JSONLoader,
    ExcelLoader,
    DatasetRegistry
)
from .preprocessing import (
    TabularPreprocessor,
    DataQualityChecker,
    SchemaValidator,
    TypeInferencer,
    DataNormalizer,
    MissingValueHandler,
    OutlierDetector
)
from .datasets import (
    TabularDataset,
    StreamingTabularDataset,
    CachedTabularDataset,
    MultiTableDataset,
    TemporalTabularDataset,
    DataSplit,
    create_data_splits,
    create_dataloader
)
from .transforms import (
    TabularTransform,
    NormalizationTransform,
    EncodingTransform,
    ImputationTransform,
    AugmentationTransform,
    CompositeTransform
)
from .utils import (
    infer_data_types,
    validate_schema,
    compute_statistics,
    detect_outliers,
    sample_dataset,
    split_dataset
)

__all__ = [
    # Loaders
    "TabularDataLoader",
    "OpenMLLoader",
    "UCILoader", 
    "KaggleLoader",
    "CSVLoader",
    "ParquetLoader",
    "JSONLoader",
    "ExcelLoader",
    "DatasetRegistry",
    
    # Preprocessing
    "TabularPreprocessor",
    "DataQualityChecker",
    "SchemaValidator",
    "TypeInferencer",
    "DataNormalizer",
    "MissingValueHandler",
    "OutlierDetector",
    
    # Datasets
    "TabularDataset",
    "StreamingTabularDataset",
    "CachedTabularDataset",
    "MultiTableDataset",
    "TemporalTabularDataset",
    "DataSplit",
    "create_data_splits",
    "create_dataloader",
    
    # Transforms
    "TabularTransform",
    "NormalizationTransform",
    "EncodingTransform",
    "ImputationTransform",
    "AugmentationTransform",
    "CompositeTransform",
    
    # Utils
    "infer_data_types",
    "validate_schema",
    "compute_statistics",
    "detect_outliers",
    "sample_dataset",
    "split_dataset",
]