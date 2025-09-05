"""Data processing utilities."""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
from sklearn.preprocessing import StandardScaler, RobustScaler


def infer_column_types(dataframe: pd.DataFrame) -> Dict[str, str]:
    """
    Infer column types from pandas DataFrame.
    
    Args:
        dataframe: Input DataFrame
        
    Returns:
        Dictionary mapping column names to inferred types
    """
    column_types = {}
    
    for col in dataframe.columns:
        series = dataframe[col]
        
        if pd.api.types.is_numeric_dtype(series):
            # Check if it's actually categorical (low cardinality integers)
            if series.nunique() <= 10 and series.dtype in ['int64', 'int32']:
                column_types[col] = 'categorical'
            else:
                column_types[col] = 'numerical'
        elif pd.api.types.is_datetime64_any_dtype(series):
            column_types[col] = 'datetime'
        elif pd.api.types.is_bool_dtype(series):
            column_types[col] = 'boolean'
        else:
            column_types[col] = 'categorical'
    
    return column_types


def normalize_numerical_features(
    dataframe: pd.DataFrame,
    method: str = 'standard',
    columns: List[str] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Normalize numerical features in DataFrame.
    
    Args:
        dataframe: Input DataFrame
        method: Normalization method ('standard' or 'robust')
        columns: Specific columns to normalize (None for all numerical)
        
    Returns:
        Tuple of (normalized_dataframe, scaler_info)
    """
    df_normalized = dataframe.copy()
    scaler_info = {}
    
    if columns is None:
        # Auto-detect numerical columns
        columns = []
        for col in dataframe.columns:
            if pd.api.types.is_numeric_dtype(dataframe[col]):
                columns.append(col)
    
    for col in columns:
        if col not in dataframe.columns:
            continue
            
        series = dataframe[col].dropna()
        if len(series) == 0:
            continue
        
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        # Fit and transform
        values = series.values.reshape(-1, 1)
        scaler.fit(values)
        
        # Apply to full column (handling NaN)
        mask = dataframe[col].notna()
        normalized_values = dataframe[col].copy()
        if mask.any():
            normalized_values[mask] = scaler.transform(
                dataframe[col][mask].values.reshape(-1, 1)
            ).flatten()
        
        df_normalized[col] = normalized_values
        scaler_info[col] = {
            'scaler': scaler,
            'method': method
        }
    
    return df_normalized, scaler_info


def handle_missing_values(
    dataframe: pd.DataFrame,
    strategy: str = 'mean',
    columns: List[str] = None
) -> pd.DataFrame:
    """
    Handle missing values in DataFrame.
    
    Args:
        dataframe: Input DataFrame
        strategy: Strategy for handling missing values
        columns: Specific columns to process
        
    Returns:
        DataFrame with missing values handled
    """
    df_filled = dataframe.copy()
    
    if columns is None:
        columns = dataframe.columns
    
    for col in columns:
        if col not in dataframe.columns:
            continue
        
        series = dataframe[col]
        if not series.isnull().any():
            continue
        
        if strategy == 'mean' and pd.api.types.is_numeric_dtype(series):
            df_filled[col] = series.fillna(series.mean())
        elif strategy == 'median' and pd.api.types.is_numeric_dtype(series):
            df_filled[col] = series.fillna(series.median())
        elif strategy == 'mode':
            df_filled[col] = series.fillna(series.mode().iloc[0] if not series.mode().empty else 'unknown')
        elif strategy == 'forward_fill':
            df_filled[col] = series.fillna(method='ffill')
        elif strategy == 'backward_fill':
            df_filled[col] = series.fillna(method='bfill')
        else:
            # Default to mode for categorical, mean for numerical
            if pd.api.types.is_numeric_dtype(series):
                df_filled[col] = series.fillna(series.mean())
            else:
                df_filled[col] = series.fillna(series.mode().iloc[0] if not series.mode().empty else 'unknown')
    
    return df_filled