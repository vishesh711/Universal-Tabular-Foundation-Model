"""Error recovery and graceful degradation utilities for TabGPT."""

import logging
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
import warnings
from functools import wraps

import pandas as pd
import numpy as np

from .exceptions import (
    DataQualityError, SchemaError, TokenizationError, ModelError,
    MissingColumnsError, ExtraColumnsError, DataTypeError,
    ExcessiveMissingValuesError, OutlierError, VocabularyError
)
from .validation import DataValidator

logger = logging.getLogger(__name__)


class DataRecovery:
    """Data recovery and preprocessing utilities."""
    
    def __init__(
        self,
        auto_fix: bool = True,
        missing_strategy: str = "median",
        outlier_strategy: str = "clip",
        dtype_coercion: bool = True,
        verbose: bool = True
    ):
        """
        Initialize data recovery system.
        
        Args:
            auto_fix: Whether to automatically fix data issues
            missing_strategy: Strategy for handling missing values ('median', 'mean', 'mode', 'drop')
            outlier_strategy: Strategy for handling outliers ('clip', 'remove', 'transform')
            dtype_coercion: Whether to attempt automatic dtype conversion
            verbose: Whether to log recovery actions
        """
        self.auto_fix = auto_fix
        self.missing_strategy = missing_strategy
        self.outlier_strategy = outlier_strategy
        self.dtype_coercion = dtype_coercion
        self.verbose = verbose
    
    def recover_dataframe(
        self,
        df: pd.DataFrame,
        expected_columns: Optional[List[str]] = None,
        expected_dtypes: Optional[Dict[str, str]] = None,
        target_column: Optional[str] = None
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Attempt to recover a DataFrame from various data quality issues.
        
        Args:
            df: Input DataFrame
            expected_columns: Expected column names
            expected_dtypes: Expected data types
            target_column: Target column name (won't be modified)
            
        Returns:
            Tuple of (recovered_dataframe, recovery_log)
        """
        recovery_log = {
            'actions_taken': [],
            'warnings': [],
            'original_shape': df.shape,
            'columns_added': [],
            'columns_removed': [],
            'columns_modified': [],
            'rows_removed': 0
        }
        
        if not self.auto_fix:
            return df.copy(), recovery_log
        
        recovered_df = df.copy()
        
        try:
            # 1. Handle missing columns
            if expected_columns:
                recovered_df, log = self._handle_missing_columns(recovered_df, expected_columns)
                recovery_log['actions_taken'].extend(log['actions_taken'])
                recovery_log['columns_added'].extend(log['columns_added'])
            
            # 2. Handle extra columns
            if expected_columns:
                recovered_df, log = self._handle_extra_columns(recovered_df, expected_columns, target_column)
                recovery_log['actions_taken'].extend(log['actions_taken'])
                recovery_log['columns_removed'].extend(log['columns_removed'])
            
            # 3. Handle data type issues
            if expected_dtypes:
                recovered_df, log = self._handle_dtype_issues(recovered_df, expected_dtypes)
                recovery_log['actions_taken'].extend(log['actions_taken'])
                recovery_log['columns_modified'].extend(log['columns_modified'])
            
            # 4. Handle missing values
            recovered_df, log = self._handle_missing_values(recovered_df, target_column)
            recovery_log['actions_taken'].extend(log['actions_taken'])
            recovery_log['columns_modified'].extend(log['columns_modified'])
            recovery_log['rows_removed'] += log['rows_removed']
            
            # 5. Handle outliers
            recovered_df, log = self._handle_outliers(recovered_df, target_column)
            recovery_log['actions_taken'].extend(log['actions_taken'])
            recovery_log['columns_modified'].extend(log['columns_modified'])
            recovery_log['rows_removed'] += log['rows_removed']
            
            # 6. Final validation
            recovered_df = self._final_cleanup(recovered_df)
            
            recovery_log['final_shape'] = recovered_df.shape
            recovery_log['success'] = True
            
            if self.verbose and recovery_log['actions_taken']:
                logger.info(f"Data recovery completed. Actions taken: {len(recovery_log['actions_taken'])}")
                for action in recovery_log['actions_taken']:
                    logger.info(f"  - {action}")
        
        except Exception as e:
            recovery_log['success'] = False
            recovery_log['error'] = str(e)
            logger.error(f"Data recovery failed: {e}")
            
            # Return original data if recovery fails
            return df.copy(), recovery_log
        
        return recovered_df, recovery_log
    
    def _handle_missing_columns(self, df: pd.DataFrame, expected_columns: List[str]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Handle missing columns by adding them with default values."""
        log = {'actions_taken': [], 'columns_added': []}
        
        missing_columns = set(expected_columns) - set(df.columns)
        
        for col in missing_columns:
            # Add column with NaN values
            df[col] = np.nan
            log['actions_taken'].append(f"Added missing column '{col}' with NaN values")
            log['columns_added'].append(col)
        
        return df, log
    
    def _handle_extra_columns(self, df: pd.DataFrame, expected_columns: List[str], target_column: Optional[str]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Handle extra columns by removing them (except target column)."""
        log = {'actions_taken': [], 'columns_removed': []}
        
        expected_set = set(expected_columns)
        if target_column:
            expected_set.add(target_column)
        
        extra_columns = set(df.columns) - expected_set
        
        for col in extra_columns:
            df = df.drop(columns=[col])
            log['actions_taken'].append(f"Removed extra column '{col}'")
            log['columns_removed'].append(col)
        
        return df, log
    
    def _handle_dtype_issues(self, df: pd.DataFrame, expected_dtypes: Dict[str, str]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Handle data type conversion issues."""
        log = {'actions_taken': [], 'columns_modified': []}
        
        for col, expected_dtype in expected_dtypes.items():
            if col not in df.columns:
                continue
            
            current_dtype = str(df[col].dtype)
            
            if self._normalize_dtype(current_dtype) != self._normalize_dtype(expected_dtype):
                try:
                    df[col] = self._convert_dtype(df[col], expected_dtype)
                    log['actions_taken'].append(f"Converted column '{col}' from {current_dtype} to {expected_dtype}")
                    log['columns_modified'].append(col)
                except Exception as e:
                    log['actions_taken'].append(f"Failed to convert column '{col}' to {expected_dtype}: {e}")
        
        return df, log
    
    def _handle_missing_values(self, df: pd.DataFrame, target_column: Optional[str]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Handle missing values using specified strategy."""
        log = {'actions_taken': [], 'columns_modified': [], 'rows_removed': 0}
        
        for col in df.columns:
            if col == target_column:
                # Handle target column missing values by dropping rows
                missing_count = df[col].isnull().sum()
                if missing_count > 0:
                    df = df.dropna(subset=[col])
                    log['actions_taken'].append(f"Dropped {missing_count} rows with missing target values")
                    log['rows_removed'] += missing_count
                continue
            
            missing_count = df[col].isnull().sum()
            if missing_count == 0:
                continue
            
            if self.missing_strategy == "drop":
                df = df.dropna(subset=[col])
                log['actions_taken'].append(f"Dropped {missing_count} rows with missing values in '{col}'")
                log['rows_removed'] += missing_count
            else:
                fill_value = self._get_fill_value(df[col], self.missing_strategy)
                df[col] = df[col].fillna(fill_value)
                log['actions_taken'].append(f"Filled {missing_count} missing values in '{col}' with {self.missing_strategy} ({fill_value})")
                log['columns_modified'].append(col)
        
        return df, log
    
    def _handle_outliers(self, df: pd.DataFrame, target_column: Optional[str]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Handle outliers using specified strategy."""
        log = {'actions_taken': [], 'columns_modified': [], 'rows_removed': 0}
        
        for col in df.columns:
            if col == target_column or not pd.api.types.is_numeric_dtype(df[col]):
                continue
            
            outlier_mask = self._detect_outliers(df[col])
            outlier_count = outlier_mask.sum()
            
            if outlier_count == 0:
                continue
            
            if self.outlier_strategy == "remove":
                df = df[~outlier_mask]
                log['actions_taken'].append(f"Removed {outlier_count} outliers from '{col}'")
                log['rows_removed'] += outlier_count
            elif self.outlier_strategy == "clip":
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
                log['actions_taken'].append(f"Clipped {outlier_count} outliers in '{col}' to [{lower_bound:.2f}, {upper_bound:.2f}]")
                log['columns_modified'].append(col)
            elif self.outlier_strategy == "transform":
                # Log transform for positive values
                if (df[col] > 0).all():
                    df[col] = np.log1p(df[col])
                    log['actions_taken'].append(f"Applied log transform to '{col}' to reduce outlier impact")
                    log['columns_modified'].append(col)
        
        return df, log
    
    def _final_cleanup(self, df: pd.DataFrame) -> pd.DataFrame:
        """Final cleanup of the DataFrame."""
        # Remove completely empty rows
        initial_rows = len(df)
        df = df.dropna(how='all')
        removed_rows = initial_rows - len(df)
        
        if removed_rows > 0 and self.verbose:
            logger.info(f"Removed {removed_rows} completely empty rows")
        
        # Reset index
        df = df.reset_index(drop=True)
        
        return df
    
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
    
    def _convert_dtype(self, series: pd.Series, target_dtype: str) -> pd.Series:
        """Convert series to target dtype."""
        target_dtype = target_dtype.lower()
        
        if target_dtype in ['int', 'integer']:
            return pd.to_numeric(series, errors='coerce').astype('Int64')  # Nullable integer
        elif target_dtype in ['float', 'numeric']:
            return pd.to_numeric(series, errors='coerce')
        elif target_dtype in ['datetime', 'timestamp']:
            return pd.to_datetime(series, errors='coerce')
        elif target_dtype in ['bool', 'boolean']:
            return series.astype('boolean')  # Nullable boolean
        elif target_dtype in ['str', 'string', 'object']:
            return series.astype(str)
        elif target_dtype == 'category':
            return series.astype('category')
        else:
            raise ValueError(f"Unknown target dtype: {target_dtype}")
    
    def _get_fill_value(self, series: pd.Series, strategy: str):
        """Get fill value for missing data based on strategy."""
        if strategy == "median":
            if pd.api.types.is_numeric_dtype(series):
                return series.median()
            else:
                return series.mode().iloc[0] if len(series.mode()) > 0 else "unknown"
        elif strategy == "mean":
            if pd.api.types.is_numeric_dtype(series):
                return series.mean()
            else:
                return series.mode().iloc[0] if len(series.mode()) > 0 else "unknown"
        elif strategy == "mode":
            mode_values = series.mode()
            return mode_values.iloc[0] if len(mode_values) > 0 else "unknown"
        else:
            return "unknown"
    
    def _detect_outliers(self, series: pd.Series) -> pd.Series:
        """Detect outliers using IQR method."""
        if not pd.api.types.is_numeric_dtype(series):
            return pd.Series([False] * len(series), index=series.index)
        
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        
        if IQR == 0:
            return pd.Series([False] * len(series), index=series.index)
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        return (series < lower_bound) | (series > upper_bound)


class ModelRecovery:
    """Model recovery and fallback utilities."""
    
    def __init__(self, enable_fallbacks: bool = True, verbose: bool = True):
        self.enable_fallbacks = enable_fallbacks
        self.verbose = verbose
        self.fallback_models = {}
    
    def register_fallback_model(self, task_type: str, model: Any):
        """Register a fallback model for a specific task type."""
        self.fallback_models[task_type] = model
        if self.verbose:
            logger.info(f"Registered fallback model for task type: {task_type}")
    
    def safe_predict(self, model: Any, X: pd.DataFrame, task_type: str = "classification") -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Safely make predictions with fallback options.
        
        Args:
            model: Primary model to use
            X: Input data
            task_type: Type of task for fallback selection
            
        Returns:
            Tuple of (predictions, recovery_info)
        """
        recovery_info = {
            'primary_model_used': True,
            'fallback_used': False,
            'errors': [],
            'warnings': []
        }
        
        try:
            # Try primary model
            predictions = model.predict(X)
            return predictions, recovery_info
        
        except Exception as e:
            recovery_info['primary_model_used'] = False
            recovery_info['errors'].append(f"Primary model failed: {str(e)}")
            
            if self.verbose:
                logger.warning(f"Primary model prediction failed: {e}")
            
            # Try fallback model
            if self.enable_fallbacks and task_type in self.fallback_models:
                try:
                    fallback_model = self.fallback_models[task_type]
                    
                    # Fit fallback model if needed
                    if not hasattr(fallback_model, '_is_fitted') or not fallback_model._is_fitted:
                        # Use simple heuristics for emergency fitting
                        y_dummy = self._generate_dummy_target(X, task_type)
                        fallback_model.fit(X, y_dummy)
                        fallback_model._is_fitted = True
                    
                    predictions = fallback_model.predict(X)
                    recovery_info['fallback_used'] = True
                    recovery_info['warnings'].append("Used fallback model for predictions")
                    
                    if self.verbose:
                        logger.info("Successfully used fallback model for predictions")
                    
                    return predictions, recovery_info
                
                except Exception as fallback_error:
                    recovery_info['errors'].append(f"Fallback model failed: {str(fallback_error)}")
                    if self.verbose:
                        logger.error(f"Fallback model also failed: {fallback_error}")
            
            # Last resort: return dummy predictions
            predictions = self._generate_dummy_predictions(X, task_type)
            recovery_info['warnings'].append("Used dummy predictions as last resort")
            
            if self.verbose:
                logger.warning("Using dummy predictions as last resort")
            
            return predictions, recovery_info
    
    def _generate_dummy_target(self, X: pd.DataFrame, task_type: str) -> np.ndarray:
        """Generate dummy target values for emergency model fitting."""
        n_samples = len(X)
        
        if task_type == "classification":
            # Binary classification dummy targets
            return np.random.choice([0, 1], size=n_samples)
        elif task_type == "regression":
            # Regression dummy targets
            return np.random.randn(n_samples)
        else:
            return np.zeros(n_samples)
    
    def _generate_dummy_predictions(self, X: pd.DataFrame, task_type: str) -> np.ndarray:
        """Generate dummy predictions as last resort."""
        n_samples = len(X)
        
        if task_type == "classification":
            # Return majority class (0)
            return np.zeros(n_samples, dtype=int)
        elif task_type == "regression":
            # Return zeros
            return np.zeros(n_samples)
        else:
            return np.zeros(n_samples)


def robust_operation(max_retries: int = 3, backoff_factor: float = 1.0):
    """
    Decorator for robust operations with retry logic.
    
    Args:
        max_retries: Maximum number of retry attempts
        backoff_factor: Exponential backoff factor for delays
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    
                    if attempt < max_retries:
                        delay = backoff_factor * (2 ** attempt)
                        logger.warning(f"Operation {func.__name__} failed (attempt {attempt + 1}/{max_retries + 1}): {e}")
                        logger.info(f"Retrying in {delay} seconds...")
                        
                        import time
                        time.sleep(delay)
                    else:
                        logger.error(f"Operation {func.__name__} failed after {max_retries + 1} attempts")
            
            # Re-raise the last exception if all retries failed
            raise last_exception
        
        return wrapper
    return decorator


def graceful_degradation(fallback_value: Any = None, log_error: bool = True):
    """
    Decorator for graceful degradation - return fallback value on error.
    
    Args:
        fallback_value: Value to return on error
        log_error: Whether to log the error
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if log_error:
                    logger.error(f"Function {func.__name__} failed, using fallback: {e}")
                return fallback_value
        
        return wrapper
    return decorator