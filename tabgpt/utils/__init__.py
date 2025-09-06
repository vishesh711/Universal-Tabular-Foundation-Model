"""Utility functions and classes."""

from .data_utils import infer_column_types, normalize_numerical_features
from .tensor_utils import create_attention_mask, pad_sequences
from .exceptions import (
    TabGPTError, DataQualityError, ValidationError, SchemaError,
    TokenizationError, ModelError, ConfigurationError, InferenceError,
    TrainingError, MissingColumnsError, ExtraColumnsError, DataTypeError,
    EmptyDataError, ExcessiveMissingValuesError, OutlierError,
    SchemaMismatchError, VocabularyError, ModelNotTrainedError,
    IncompatibleModelError, ResourceError, MemoryError, TimeoutError,
    handle_exception
)
from .validation import (
    DataValidator, ConfigValidator, SchemaValidator,
    validate_input_data, validate_model_config
)
from .recovery import (
    DataRecovery, ModelRecovery, robust_operation, graceful_degradation
)
from .normalization import (
    RobustNormalizer
)

__all__ = [
    # Original utilities
    "infer_column_types",
    "normalize_numerical_features", 
    "create_attention_mask",
    "pad_sequences",
    
    # Exceptions
    "TabGPTError", "DataQualityError", "ValidationError", "SchemaError",
    "TokenizationError", "ModelError", "ConfigurationError", "InferenceError",
    "TrainingError", "MissingColumnsError", "ExtraColumnsError", "DataTypeError",
    "EmptyDataError", "ExcessiveMissingValuesError", "OutlierError",
    "SchemaMismatchError", "VocabularyError", "ModelNotTrainedError",
    "IncompatibleModelError", "ResourceError", "MemoryError", "TimeoutError",
    "handle_exception",
    
    # Validation
    "DataValidator", "ConfigValidator", "SchemaValidator",
    "validate_input_data", "validate_model_config",
    
    # Recovery
    "DataRecovery", "ModelRecovery", "robust_operation", "graceful_degradation",
    
    # Normalization
    "RobustNormalizer"
]