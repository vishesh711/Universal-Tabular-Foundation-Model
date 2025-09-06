"""Custom exceptions for TabGPT."""

from typing import Optional, Dict, Any, List


class TabGPTError(Exception):
    """Base exception class for TabGPT."""
    
    def __init__(self, message: str, error_code: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.details = details or {}
    
    def __str__(self):
        base_msg = f"[{self.error_code}] {self.message}"
        if self.details:
            details_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            return f"{base_msg} (Details: {details_str})"
        return base_msg


class DataQualityError(TabGPTError):
    """Raised when data quality issues are detected."""
    pass


class SchemaError(TabGPTError):
    """Raised when schema-related issues occur."""
    pass


class TokenizationError(TabGPTError):
    """Raised when tokenization fails."""
    pass


class ModelError(TabGPTError):
    """Raised when model-related issues occur."""
    pass


class ConfigurationError(TabGPTError):
    """Raised when configuration is invalid."""
    pass


class InferenceError(TabGPTError):
    """Raised during inference when issues occur."""
    pass


class TrainingError(TabGPTError):
    """Raised during training when issues occur."""
    pass


class ValidationError(TabGPTError):
    """Raised when input validation fails."""
    pass


# Specific data quality errors
class MissingColumnsError(DataQualityError):
    """Raised when expected columns are missing."""
    
    def __init__(self, missing_columns: List[str], expected_columns: List[str]):
        self.missing_columns = missing_columns
        self.expected_columns = expected_columns
        message = f"Missing columns: {missing_columns}. Expected: {expected_columns}"
        super().__init__(message, details={
            'missing_columns': missing_columns,
            'expected_columns': expected_columns
        })


class ExtraColumnsError(DataQualityError):
    """Raised when unexpected columns are present."""
    
    def __init__(self, extra_columns: List[str], expected_columns: List[str]):
        self.extra_columns = extra_columns
        self.expected_columns = expected_columns
        message = f"Unexpected columns: {extra_columns}. Expected: {expected_columns}"
        super().__init__(message, details={
            'extra_columns': extra_columns,
            'expected_columns': expected_columns
        })


class DataTypeError(DataQualityError):
    """Raised when data types don't match expectations."""
    
    def __init__(self, column: str, expected_type: str, actual_type: str):
        self.column = column
        self.expected_type = expected_type
        self.actual_type = actual_type
        message = f"Column '{column}' has type '{actual_type}', expected '{expected_type}'"
        super().__init__(message, details={
            'column': column,
            'expected_type': expected_type,
            'actual_type': actual_type
        })


class EmptyDataError(DataQualityError):
    """Raised when dataset is empty."""
    
    def __init__(self, dataset_name: Optional[str] = None):
        self.dataset_name = dataset_name
        message = f"Dataset is empty"
        if dataset_name:
            message = f"Dataset '{dataset_name}' is empty"
        super().__init__(message, details={'dataset_name': dataset_name})


class ExcessiveMissingValuesError(DataQualityError):
    """Raised when too many missing values are detected."""
    
    def __init__(self, column: str, missing_ratio: float, threshold: float):
        self.column = column
        self.missing_ratio = missing_ratio
        self.threshold = threshold
        message = f"Column '{column}' has {missing_ratio:.2%} missing values, exceeds threshold {threshold:.2%}"
        super().__init__(message, details={
            'column': column,
            'missing_ratio': missing_ratio,
            'threshold': threshold
        })


class OutlierError(DataQualityError):
    """Raised when excessive outliers are detected."""
    
    def __init__(self, column: str, outlier_count: int, total_count: int, threshold: float):
        self.column = column
        self.outlier_count = outlier_count
        self.total_count = total_count
        self.threshold = threshold
        outlier_ratio = outlier_count / total_count
        message = f"Column '{column}' has {outlier_count}/{total_count} ({outlier_ratio:.2%}) outliers, exceeds threshold {threshold:.2%}"
        super().__init__(message, details={
            'column': column,
            'outlier_count': outlier_count,
            'total_count': total_count,
            'outlier_ratio': outlier_ratio,
            'threshold': threshold
        })


class SchemaMismatchError(SchemaError):
    """Raised when schema doesn't match expectations."""
    
    def __init__(self, expected_schema: Dict[str, str], actual_schema: Dict[str, str]):
        self.expected_schema = expected_schema
        self.actual_schema = actual_schema
        
        mismatches = []
        for col, expected_type in expected_schema.items():
            if col not in actual_schema:
                mismatches.append(f"Missing column '{col}'")
            elif actual_schema[col] != expected_type:
                mismatches.append(f"Column '{col}': expected {expected_type}, got {actual_schema[col]}")
        
        for col in actual_schema:
            if col not in expected_schema:
                mismatches.append(f"Unexpected column '{col}'")
        
        message = f"Schema mismatch: {'; '.join(mismatches)}"
        super().__init__(message, details={
            'expected_schema': expected_schema,
            'actual_schema': actual_schema,
            'mismatches': mismatches
        })


class VocabularyError(TokenizationError):
    """Raised when vocabulary-related issues occur."""
    
    def __init__(self, unknown_tokens: List[str], column: Optional[str] = None):
        self.unknown_tokens = unknown_tokens
        self.column = column
        message = f"Unknown tokens: {unknown_tokens}"
        if column:
            message = f"Unknown tokens in column '{column}': {unknown_tokens}"
        super().__init__(message, details={
            'unknown_tokens': unknown_tokens,
            'column': column
        })


class ModelNotTrainedError(ModelError):
    """Raised when trying to use an untrained model."""
    
    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name
        message = "Model has not been trained"
        if model_name:
            message = f"Model '{model_name}' has not been trained"
        super().__init__(message, details={'model_name': model_name})


class IncompatibleModelError(ModelError):
    """Raised when model is incompatible with data or task."""
    
    def __init__(self, reason: str, model_info: Optional[Dict[str, Any]] = None):
        self.reason = reason
        self.model_info = model_info or {}
        message = f"Model incompatibility: {reason}"
        super().__init__(message, details={'reason': reason, 'model_info': model_info})


class ResourceError(TabGPTError):
    """Raised when resource constraints are exceeded."""
    
    def __init__(self, resource_type: str, required: float, available: float, unit: str = ""):
        self.resource_type = resource_type
        self.required = required
        self.available = available
        self.unit = unit
        
        unit_str = f" {unit}" if unit else ""
        message = f"Insufficient {resource_type}: required {required}{unit_str}, available {available}{unit_str}"
        super().__init__(message, details={
            'resource_type': resource_type,
            'required': required,
            'available': available,
            'unit': unit
        })


class MemoryError(ResourceError):
    """Raised when memory requirements exceed available memory."""
    
    def __init__(self, required_mb: float, available_mb: float):
        super().__init__("memory", required_mb, available_mb, "MB")


class TimeoutError(TabGPTError):
    """Raised when operations timeout."""
    
    def __init__(self, operation: str, timeout_seconds: float):
        self.operation = operation
        self.timeout_seconds = timeout_seconds
        message = f"Operation '{operation}' timed out after {timeout_seconds} seconds"
        super().__init__(message, details={
            'operation': operation,
            'timeout_seconds': timeout_seconds
        })


def handle_exception(func):
    """Decorator to handle exceptions and provide better error messages."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except TabGPTError:
            # Re-raise TabGPT errors as-is
            raise
        except Exception as e:
            # Wrap other exceptions in TabGPTError
            raise TabGPTError(
                f"Unexpected error in {func.__name__}: {str(e)}",
                error_code="UnexpectedError",
                details={'function': func.__name__, 'original_error': str(e)}
            ) from e
    return wrapper