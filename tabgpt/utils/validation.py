"""Input validation and data quality checks for TabGPT."""

import logging
from typing import Dict, List, Optional, Union, Any, Tuple, Set
import warnings

import pandas as pd
import numpy as np

from .exceptions import (
    DataQualityError, ValidationError, MissingColumnsError, ExtraColumnsError,
    DataTypeError, EmptyDataError, ExcessiveMissingValuesError, OutlierError,
    SchemaMismatchError, ConfigurationError
)

logger = logging.getLogger(__name__)


class DataValidator:
    """Comprehensive data validation for TabGPT."""
    
    def __init__(
        self,
        missing_threshold: float = 0.5,
        outlier_threshold: float = 0.1,
        min_samples: int = 10,
        max_samples: Optional[int] = None,
        min_features: int = 1,
        max_features: Optional[int] = None,
        strict_mode: bool = False
    ):
        """
        Initialize data validator.
        
        Args:
            missing_threshold: Maximum allowed ratio of missing values per column
            outlier_threshold: Maximum allowed ratio of outliers per column
            min_samples: Minimum number of samples required
            max_samples: Maximum number of samples allowed
            min_features: Minimum number of features required
            max_features: Maximum number of features allowed
            strict_mode: If True, raise errors instead of warnings
        """
        self.missing_threshold = missing_threshold
        self.outlier_threshold = outlier_threshold
        self.min_samples = min_samples
        self.max_samples = max_samples
        self.min_features = min_features
        self.max_features = max_features
        self.strict_mode = strict_mode
    
    def validate_dataframe(
        self,
        df: pd.DataFrame,
        expected_columns: Optional[List[str]] = None,
        expected_dtypes: Optional[Dict[str, str]] = None,
        allow_extra_columns: bool = True,
        dataset_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive validation of a pandas DataFrame.
        
        Args:
            df: DataFrame to validate
            expected_columns: List of expected column names
            expected_dtypes: Dictionary mapping column names to expected data types
            allow_extra_columns: Whether to allow extra columns beyond expected
            dataset_name: Name of dataset for error messages
            
        Returns:
            Dictionary with validation results and warnings
        """
        results = {
            'valid': True,
            'warnings': [],
            'errors': [],
            'stats': {}
        }
        
        try:
            # Basic structure validation
            self._validate_basic_structure(df, dataset_name, results)
            
            # Column validation
            if expected_columns is not None:
                self._validate_columns(df, expected_columns, allow_extra_columns, results)
            
            # Data type validation
            if expected_dtypes is not None:
                self._validate_dtypes(df, expected_dtypes, results)
            
            # Data quality validation
            self._validate_data_quality(df, results)
            
            # Compute statistics
            results['stats'] = self._compute_data_stats(df)
            
        except Exception as e:
            results['valid'] = False
            results['errors'].append(str(e))
            if self.strict_mode:
                raise
        
        # Handle warnings and errors
        if results['errors'] and self.strict_mode:
            raise ValidationError(f"Data validation failed: {'; '.join(results['errors'])}")
        
        if results['warnings']:
            for warning in results['warnings']:
                warnings.warn(warning, UserWarning)
        
        return results
    
    def _validate_basic_structure(self, df: pd.DataFrame, dataset_name: Optional[str], results: Dict[str, Any]):
        """Validate basic DataFrame structure."""
        # Check if empty
        if df.empty:
            error = EmptyDataError(dataset_name)
            results['errors'].append(str(error))
            if self.strict_mode:
                raise error
            return
        
        n_samples, n_features = df.shape
        
        # Check sample count
        if n_samples < self.min_samples:
            error = f"Dataset has {n_samples} samples, minimum required: {self.min_samples}"
            results['errors'].append(error)
            if self.strict_mode:
                raise ValidationError(error)
        
        if self.max_samples and n_samples > self.max_samples:
            warning = f"Dataset has {n_samples} samples, maximum recommended: {self.max_samples}"
            results['warnings'].append(warning)
        
        # Check feature count
        if n_features < self.min_features:
            error = f"Dataset has {n_features} features, minimum required: {self.min_features}"
            results['errors'].append(error)
            if self.strict_mode:
                raise ValidationError(error)
        
        if self.max_features and n_features > self.max_features:
            warning = f"Dataset has {n_features} features, maximum recommended: {self.max_features}"
            results['warnings'].append(warning)
    
    def _validate_columns(self, df: pd.DataFrame, expected_columns: List[str], allow_extra: bool, results: Dict[str, Any]):
        """Validate column names."""
        actual_columns = set(df.columns)
        expected_set = set(expected_columns)
        
        # Check for missing columns
        missing_columns = expected_set - actual_columns
        if missing_columns:
            error = MissingColumnsError(list(missing_columns), expected_columns)
            results['errors'].append(str(error))
            if self.strict_mode:
                raise error
        
        # Check for extra columns
        extra_columns = actual_columns - expected_set
        if extra_columns and not allow_extra:
            error = ExtraColumnsError(list(extra_columns), expected_columns)
            results['errors'].append(str(error))
            if self.strict_mode:
                raise error
        elif extra_columns:
            warning = f"Extra columns found: {list(extra_columns)}"
            results['warnings'].append(warning)
    
    def _validate_dtypes(self, df: pd.DataFrame, expected_dtypes: Dict[str, str], results: Dict[str, Any]):
        """Validate data types."""
        for column, expected_dtype in expected_dtypes.items():
            if column not in df.columns:
                continue  # Already handled in column validation
            
            actual_dtype = str(df[column].dtype)
            
            # Normalize dtype names for comparison
            expected_normalized = self._normalize_dtype(expected_dtype)
            actual_normalized = self._normalize_dtype(actual_dtype)
            
            if expected_normalized != actual_normalized:
                # Try to see if conversion is possible
                if self._can_convert_dtype(df[column], expected_dtype):
                    warning = f"Column '{column}' has type '{actual_dtype}' but can be converted to '{expected_dtype}'"
                    results['warnings'].append(warning)
                else:
                    error = DataTypeError(column, expected_dtype, actual_dtype)
                    results['errors'].append(str(error))
                    if self.strict_mode:
                        raise error
    
    def _validate_data_quality(self, df: pd.DataFrame, results: Dict[str, Any]):
        """Validate data quality metrics."""
        for column in df.columns:
            # Check missing values
            missing_ratio = df[column].isnull().sum() / len(df)
            if missing_ratio > self.missing_threshold:
                error = ExcessiveMissingValuesError(column, missing_ratio, self.missing_threshold)
                results['errors'].append(str(error))
                if self.strict_mode:
                    raise error
            elif missing_ratio > 0.1:  # Warn if > 10% missing
                warning = f"Column '{column}' has {missing_ratio:.2%} missing values"
                results['warnings'].append(warning)
            
            # Check for outliers in numeric columns
            if pd.api.types.is_numeric_dtype(df[column]):
                outlier_count = self._count_outliers(df[column])
                if outlier_count > 0:
                    outlier_ratio = outlier_count / len(df)
                    if outlier_ratio > self.outlier_threshold:
                        error = OutlierError(column, outlier_count, len(df), self.outlier_threshold)
                        results['errors'].append(str(error))
                        if self.strict_mode:
                            raise error
                    elif outlier_ratio > 0.05:  # Warn if > 5% outliers
                        warning = f"Column '{column}' has {outlier_count} ({outlier_ratio:.2%}) potential outliers"
                        results['warnings'].append(warning)
            
            # Check for constant columns
            if df[column].nunique() <= 1:
                warning = f"Column '{column}' has constant values"
                results['warnings'].append(warning)
    
    def _normalize_dtype(self, dtype_str: str) -> str:
        """Normalize dtype string for comparison."""
        dtype_str = dtype_str.lower()
        
        # Map common dtype variations
        if 'int' in dtype_str:
            return 'integer'
        elif 'float' in dtype_str:
            return 'float'
        elif 'object' in dtype_str or 'string' in dtype_str:
            return 'object'
        elif 'bool' in dtype_str:
            return 'boolean'
        elif 'datetime' in dtype_str:
            return 'datetime'
        elif 'category' in dtype_str:
            return 'category'
        else:
            return dtype_str
    
    def _can_convert_dtype(self, series: pd.Series, target_dtype: str) -> bool:
        """Check if series can be converted to target dtype."""
        try:
            if target_dtype.lower() in ['int', 'integer']:
                pd.to_numeric(series, errors='raise').astype(int)
            elif target_dtype.lower() in ['float', 'numeric']:
                pd.to_numeric(series, errors='raise')
            elif target_dtype.lower() in ['datetime', 'timestamp']:
                pd.to_datetime(series, errors='raise')
            elif target_dtype.lower() in ['bool', 'boolean']:
                series.astype(bool)
            return True
        except (ValueError, TypeError):
            return False
    
    def _count_outliers(self, series: pd.Series) -> int:
        """Count outliers using IQR method."""
        if not pd.api.types.is_numeric_dtype(series):
            return 0
        
        # Remove missing values
        clean_series = series.dropna()
        if len(clean_series) < 4:  # Need at least 4 values for IQR
            return 0
        
        Q1 = clean_series.quantile(0.25)
        Q3 = clean_series.quantile(0.75)
        IQR = Q3 - Q1
        
        if IQR == 0:  # All values are the same
            return 0
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = (clean_series < lower_bound) | (clean_series > upper_bound)
        return outliers.sum()
    
    def _compute_data_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Compute comprehensive data statistics."""
        stats = {
            'n_samples': len(df),
            'n_features': len(df.columns),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024 * 1024),
            'columns': {}
        }
        
        for column in df.columns:
            col_stats = {
                'dtype': str(df[column].dtype),
                'missing_count': df[column].isnull().sum(),
                'missing_ratio': df[column].isnull().sum() / len(df),
                'unique_count': df[column].nunique(),
                'unique_ratio': df[column].nunique() / len(df)
            }
            
            if pd.api.types.is_numeric_dtype(df[column]):
                col_stats.update({
                    'mean': df[column].mean(),
                    'std': df[column].std(),
                    'min': df[column].min(),
                    'max': df[column].max(),
                    'outlier_count': self._count_outliers(df[column])
                })
            
            stats['columns'][column] = col_stats
        
        return stats


class ConfigValidator:
    """Validator for configuration objects."""
    
    @staticmethod
    def validate_config(config: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate configuration against schema.
        
        Args:
            config: Configuration dictionary to validate
            schema: Schema dictionary with validation rules
            
        Returns:
            Validation results
        """
        results = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        # Check required fields
        required_fields = schema.get('required', [])
        for field in required_fields:
            if field not in config:
                results['errors'].append(f"Required field '{field}' is missing")
                results['valid'] = False
        
        # Check field types and constraints
        field_specs = schema.get('fields', {})
        for field, spec in field_specs.items():
            if field not in config:
                continue
            
            value = config[field]
            
            # Type checking
            expected_type = spec.get('type')
            if expected_type and not isinstance(value, expected_type):
                results['errors'].append(f"Field '{field}' should be {expected_type.__name__}, got {type(value).__name__}")
                results['valid'] = False
                continue
            
            # Range checking for numeric values
            if isinstance(value, (int, float)):
                min_val = spec.get('min')
                max_val = spec.get('max')
                
                if min_val is not None and value < min_val:
                    results['errors'].append(f"Field '{field}' value {value} is below minimum {min_val}")
                    results['valid'] = False
                
                if max_val is not None and value > max_val:
                    results['errors'].append(f"Field '{field}' value {value} is above maximum {max_val}")
                    results['valid'] = False
            
            # Choice validation
            choices = spec.get('choices')
            if choices and value not in choices:
                results['errors'].append(f"Field '{field}' value '{value}' not in allowed choices: {choices}")
                results['valid'] = False
        
        # Check for unknown fields
        known_fields = set(field_specs.keys())
        config_fields = set(config.keys())
        unknown_fields = config_fields - known_fields
        
        if unknown_fields:
            results['warnings'].append(f"Unknown configuration fields: {list(unknown_fields)}")
        
        return results


class SchemaValidator:
    """Validator for data schemas."""
    
    def __init__(self):
        self.known_schemas = {}
    
    def register_schema(self, name: str, schema: Dict[str, str]):
        """Register a data schema."""
        self.known_schemas[name] = schema
    
    def validate_against_schema(self, df: pd.DataFrame, schema_name: str) -> Dict[str, Any]:
        """Validate DataFrame against a registered schema."""
        if schema_name not in self.known_schemas:
            raise ValidationError(f"Unknown schema: {schema_name}")
        
        expected_schema = self.known_schemas[schema_name]
        actual_schema = {col: str(df[col].dtype) for col in df.columns}
        
        # Check for schema mismatch
        mismatches = []
        for col, expected_type in expected_schema.items():
            if col not in actual_schema:
                mismatches.append(f"Missing column '{col}'")
            elif self._normalize_dtype(actual_schema[col]) != self._normalize_dtype(expected_type):
                mismatches.append(f"Column '{col}': expected {expected_type}, got {actual_schema[col]}")
        
        for col in actual_schema:
            if col not in expected_schema:
                mismatches.append(f"Unexpected column '{col}'")
        
        if mismatches:
            raise SchemaMismatchError(expected_schema, actual_schema)
        
        return {'valid': True, 'schema_name': schema_name}
    
    def _normalize_dtype(self, dtype_str: str) -> str:
        """Normalize dtype string for comparison."""
        dtype_str = dtype_str.lower()
        
        if 'int' in dtype_str:
            return 'integer'
        elif 'float' in dtype_str:
            return 'float'
        elif 'object' in dtype_str or 'string' in dtype_str:
            return 'object'
        elif 'bool' in dtype_str:
            return 'boolean'
        elif 'datetime' in dtype_str:
            return 'datetime'
        elif 'category' in dtype_str:
            return 'category'
        else:
            return dtype_str


def validate_input_data(
    df: pd.DataFrame,
    expected_columns: Optional[List[str]] = None,
    expected_dtypes: Optional[Dict[str, str]] = None,
    missing_threshold: float = 0.5,
    strict_mode: bool = False,
    dataset_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Convenience function for input data validation.
    
    Args:
        df: DataFrame to validate
        expected_columns: List of expected column names
        expected_dtypes: Dictionary mapping column names to expected data types
        missing_threshold: Maximum allowed ratio of missing values per column
        strict_mode: If True, raise errors instead of warnings
        dataset_name: Name of dataset for error messages
        
    Returns:
        Dictionary with validation results
    """
    validator = DataValidator(
        missing_threshold=missing_threshold,
        strict_mode=strict_mode
    )
    
    return validator.validate_dataframe(
        df=df,
        expected_columns=expected_columns,
        expected_dtypes=expected_dtypes,
        dataset_name=dataset_name
    )


def validate_model_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate model configuration.
    
    Args:
        config: Model configuration dictionary
        
    Returns:
        Validation results
    """
    schema = {
        'required': ['model_type', 'hidden_size'],
        'fields': {
            'model_type': {'type': str, 'choices': ['classification', 'regression', 'pretraining']},
            'hidden_size': {'type': int, 'min': 64, 'max': 4096},
            'num_layers': {'type': int, 'min': 1, 'max': 24},
            'num_attention_heads': {'type': int, 'min': 1, 'max': 32},
            'dropout': {'type': float, 'min': 0.0, 'max': 1.0},
            'learning_rate': {'type': float, 'min': 1e-6, 'max': 1e-1},
            'batch_size': {'type': int, 'min': 1, 'max': 1024}
        }
    }
    
    return ConfigValidator.validate_config(config, schema)