"""Robust normalization and preprocessing utilities for TabGPT."""

import logging
from typing import Dict, List, Optional, Union, Any, Tuple
import warnings

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, QuantileTransformer

from .exceptions import DataQualityError, ValidationError
from .validation import DataValidator

logger = logging.getLogger(__name__)


class RobustNormalizer:
    """Robust data normalization with outlier handling."""
    
    def __init__(
        self,
        numerical_strategy: str = "robust",
        categorical_strategy: str = "frequency",
        outlier_method: str = "iqr",
        outlier_action: str = "clip",
        missing_strategy: str = "median",
        handle_new_categories: str = "ignore",
        preserve_dtypes: bool = True
    ):
        """
        Initialize robust normalizer.
        
        Args:
            numerical_strategy: Strategy for numerical features ('standard', 'minmax', 'robust', 'quantile')
            categorical_strategy: Strategy for categorical features ('frequency', 'target', 'onehot')
            outlier_method: Method for outlier detection ('iqr', 'zscore', 'isolation')
            outlier_action: Action for outliers ('clip', 'remove', 'transform', 'ignore')
            missing_strategy: Strategy for missing values ('median', 'mean', 'mode', 'constant')
            handle_new_categories: How to handle new categories ('ignore', 'error', 'most_frequent')
            preserve_dtypes: Whether to preserve original dtypes where possible
        """
        self.numerical_strategy = numerical_strategy
        self.categorical_strategy = categorical_strategy
        self.outlier_method = outlier_method
        self.outlier_action = outlier_action
        self.missing_strategy = missing_strategy
        self.handle_new_categories = handle_new_categories
        self.preserve_dtypes = preserve_dtypes
        
        # Fitted components
        self.is_fitted = False
        self.numerical_columns = []
        self.categorical_columns = []
        self.datetime_columns = []
        self.scalers = {}
        self.encoders = {}
        self.missing_values = {}
        self.outlier_bounds = {}
        self.column_stats = {}
    
    def fit(self, df: pd.DataFrame, target_column: Optional[str] = None) -> 'RobustNormalizer':
        """
        Fit the normalizer on training data.
        
        Args:
            df: Training DataFrame
            target_column: Name of target column (will be excluded from normalization)
            
        Returns:
            Self for method chaining
        """
        logger.info("Fitting robust normalizer...")
        
        # Identify column types
        self._identify_column_types(df, target_column)
        
        # Fit numerical transformations
        self._fit_numerical_transformations(df)
        
        # Fit categorical transformations
        self._fit_categorical_transformations(df, target_column)
        
        # Fit missing value strategies
        self._fit_missing_value_strategies(df)
        
        # Compute outlier bounds
        self._compute_outlier_bounds(df)
        
        # Store column statistics
        self._compute_column_statistics(df)
        
        self.is_fitted = True
        logger.info(f"Normalizer fitted on {len(df)} samples with {len(df.columns)} features")
        
        return self
    
    def transform(self, df: pd.DataFrame, handle_errors: bool = True) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Transform data using fitted normalizer.
        
        Args:
            df: DataFrame to transform
            handle_errors: Whether to handle errors gracefully
            
        Returns:
            Tuple of (transformed_dataframe, transformation_log)
        """
        if not self.is_fitted:
            raise ValidationError("Normalizer must be fitted before transform")
        
        transformation_log = {
            'original_shape': df.shape,
            'actions_taken': [],
            'warnings': [],
            'errors': []
        }
        
        try:
            # Create copy to avoid modifying original
            transformed_df = df.copy()
            
            # Handle missing values first
            transformed_df = self._handle_missing_values(transformed_df, transformation_log)
            
            # Handle outliers
            transformed_df = self._handle_outliers(transformed_df, transformation_log)
            
            # Transform numerical features
            transformed_df = self._transform_numerical_features(transformed_df, transformation_log)
            
            # Transform categorical features
            transformed_df = self._transform_categorical_features(transformed_df, transformation_log, handle_errors)
            
            # Transform datetime features
            transformed_df = self._transform_datetime_features(transformed_df, transformation_log)
            
            transformation_log['final_shape'] = transformed_df.shape
            transformation_log['success'] = True
            
            logger.info(f"Data transformation completed. Shape: {df.shape} -> {transformed_df.shape}")
            
        except Exception as e:
            transformation_log['success'] = False
            transformation_log['errors'].append(str(e))
            
            if handle_errors:
                logger.error(f"Transformation failed: {e}")
                # Return original data with minimal processing
                transformed_df = self._minimal_transform(df)
                transformation_log['actions_taken'].append("Applied minimal transformation due to errors")
            else:
                raise
        
        return transformed_df, transformation_log
    
    def fit_transform(self, df: pd.DataFrame, target_column: Optional[str] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Fit normalizer and transform data in one step."""
        self.fit(df, target_column)
        return self.transform(df)
    
    def _identify_column_types(self, df: pd.DataFrame, target_column: Optional[str]):
        """Identify column types for appropriate processing."""
        self.numerical_columns = []
        self.categorical_columns = []
        self.datetime_columns = []
        
        for col in df.columns:
            if col == target_column:
                continue
            
            if pd.api.types.is_numeric_dtype(df[col]):
                # Check if it's actually categorical (few unique values)
                unique_ratio = df[col].nunique() / len(df)
                if unique_ratio < 0.05 and df[col].nunique() < 20:
                    self.categorical_columns.append(col)
                else:
                    self.numerical_columns.append(col)
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                self.datetime_columns.append(col)
            else:
                self.categorical_columns.append(col)
        
        logger.info(f"Identified {len(self.numerical_columns)} numerical, "
                   f"{len(self.categorical_columns)} categorical, "
                   f"{len(self.datetime_columns)} datetime columns")
    
    def _fit_numerical_transformations(self, df: pd.DataFrame):
        """Fit numerical feature transformations."""
        for col in self.numerical_columns:
            if self.numerical_strategy == "standard":
                scaler = StandardScaler()
            elif self.numerical_strategy == "minmax":
                scaler = MinMaxScaler()
            elif self.numerical_strategy == "robust":
                scaler = RobustScaler()
            elif self.numerical_strategy == "quantile":
                scaler = QuantileTransformer(output_distribution='normal', random_state=42)
            else:
                continue
            
            # Fit on non-null values
            valid_data = df[col].dropna().values.reshape(-1, 1)
            if len(valid_data) > 0:
                scaler.fit(valid_data)
                self.scalers[col] = scaler
    
    def _fit_categorical_transformations(self, df: pd.DataFrame, target_column: Optional[str]):
        """Fit categorical feature transformations."""
        for col in self.categorical_columns:
            if self.categorical_strategy == "frequency":
                # Frequency encoding
                value_counts = df[col].value_counts()
                self.encoders[col] = {'type': 'frequency', 'mapping': value_counts.to_dict()}
            
            elif self.categorical_strategy == "target" and target_column and target_column in df.columns:
                # Target encoding (mean of target for each category)
                target_means = df.groupby(col)[target_column].mean()
                self.encoders[col] = {'type': 'target', 'mapping': target_means.to_dict()}
            
            elif self.categorical_strategy == "onehot":
                # One-hot encoding (store unique values)
                unique_values = df[col].dropna().unique()
                self.encoders[col] = {'type': 'onehot', 'categories': list(unique_values)}
            
            else:
                # Label encoding (ordinal)
                unique_values = df[col].dropna().unique()
                label_mapping = {val: idx for idx, val in enumerate(unique_values)}
                self.encoders[col] = {'type': 'label', 'mapping': label_mapping}
    
    def _fit_missing_value_strategies(self, df: pd.DataFrame):
        """Fit missing value imputation strategies."""
        for col in df.columns:
            if df[col].isnull().sum() == 0:
                continue
            
            if col in self.numerical_columns:
                if self.missing_strategy == "median":
                    fill_value = df[col].median()
                elif self.missing_strategy == "mean":
                    fill_value = df[col].mean()
                else:
                    fill_value = 0
            else:
                if self.missing_strategy == "mode":
                    mode_values = df[col].mode()
                    fill_value = mode_values.iloc[0] if len(mode_values) > 0 else "unknown"
                else:
                    fill_value = "unknown"
            
            self.missing_values[col] = fill_value
    
    def _compute_outlier_bounds(self, df: pd.DataFrame):
        """Compute outlier bounds for numerical columns."""
        for col in self.numerical_columns:
            if self.outlier_method == "iqr":
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
            elif self.outlier_method == "zscore":
                mean = df[col].mean()
                std = df[col].std()
                lower_bound = mean - 3 * std
                upper_bound = mean + 3 * std
            else:
                # No bounds
                lower_bound = df[col].min()
                upper_bound = df[col].max()
            
            self.outlier_bounds[col] = {'lower': lower_bound, 'upper': upper_bound}
    
    def _compute_column_statistics(self, df: pd.DataFrame):
        """Compute and store column statistics."""
        for col in df.columns:
            stats = {
                'dtype': str(df[col].dtype),
                'missing_count': df[col].isnull().sum(),
                'unique_count': df[col].nunique()
            }
            
            if col in self.numerical_columns:
                stats.update({
                    'mean': df[col].mean(),
                    'std': df[col].std(),
                    'min': df[col].min(),
                    'max': df[col].max(),
                    'median': df[col].median()
                })
            
            self.column_stats[col] = stats
    
    def _handle_missing_values(self, df: pd.DataFrame, log: Dict[str, Any]) -> pd.DataFrame:
        """Handle missing values using fitted strategies."""
        for col, fill_value in self.missing_values.items():
            if col not in df.columns:
                continue
            
            missing_count = df[col].isnull().sum()
            if missing_count > 0:
                df[col] = df[col].fillna(fill_value)
                log['actions_taken'].append(f"Filled {missing_count} missing values in '{col}' with {fill_value}")
        
        return df
    
    def _handle_outliers(self, df: pd.DataFrame, log: Dict[str, Any]) -> pd.DataFrame:
        """Handle outliers using specified strategy."""
        if self.outlier_action == "ignore":
            return df
        
        for col in self.numerical_columns:
            if col not in df.columns or col not in self.outlier_bounds:
                continue
            
            bounds = self.outlier_bounds[col]
            outlier_mask = (df[col] < bounds['lower']) | (df[col] > bounds['upper'])
            outlier_count = outlier_mask.sum()
            
            if outlier_count == 0:
                continue
            
            if self.outlier_action == "clip":
                df[col] = df[col].clip(lower=bounds['lower'], upper=bounds['upper'])
                log['actions_taken'].append(f"Clipped {outlier_count} outliers in '{col}'")
            
            elif self.outlier_action == "remove":
                df = df[~outlier_mask]
                log['actions_taken'].append(f"Removed {outlier_count} outlier rows based on '{col}'")
            
            elif self.outlier_action == "transform":
                # Log transform for positive values
                if (df[col] > 0).all():
                    df[col] = np.log1p(df[col])
                    log['actions_taken'].append(f"Applied log transform to '{col}' to reduce outlier impact")
        
        return df
    
    def _transform_numerical_features(self, df: pd.DataFrame, log: Dict[str, Any]) -> pd.DataFrame:
        """Transform numerical features using fitted scalers."""
        for col in self.numerical_columns:
            if col not in df.columns or col not in self.scalers:
                continue
            
            scaler = self.scalers[col]
            
            # Handle case where all values are NaN
            if df[col].isnull().all():
                continue
            
            # Transform non-null values
            mask = ~df[col].isnull()
            if mask.sum() > 0:
                df.loc[mask, col] = scaler.transform(df.loc[mask, col].values.reshape(-1, 1)).flatten()
                log['actions_taken'].append(f"Scaled numerical column '{col}' using {self.numerical_strategy}")
        
        return df
    
    def _transform_categorical_features(self, df: pd.DataFrame, log: Dict[str, Any], handle_errors: bool) -> pd.DataFrame:
        """Transform categorical features using fitted encoders."""
        for col in self.categorical_columns:
            if col not in df.columns or col not in self.encoders:
                continue
            
            encoder_info = self.encoders[col]
            encoder_type = encoder_info['type']
            
            try:
                if encoder_type == "frequency":
                    mapping = encoder_info['mapping']
                    df[col] = df[col].map(mapping)
                    
                    # Handle unknown categories
                    unknown_mask = df[col].isnull()
                    if unknown_mask.sum() > 0:
                        if self.handle_new_categories == "most_frequent":
                            most_frequent_value = max(mapping.values())
                            df.loc[unknown_mask, col] = most_frequent_value
                        elif self.handle_new_categories == "ignore":
                            df.loc[unknown_mask, col] = 0  # Default frequency
                        # 'error' case handled by exception
                
                elif encoder_type == "target":
                    mapping = encoder_info['mapping']
                    df[col] = df[col].map(mapping)
                    
                    # Handle unknown categories
                    unknown_mask = df[col].isnull()
                    if unknown_mask.sum() > 0:
                        global_mean = np.mean(list(mapping.values()))
                        df.loc[unknown_mask, col] = global_mean
                
                elif encoder_type == "label":
                    mapping = encoder_info['mapping']
                    df[col] = df[col].map(mapping)
                    
                    # Handle unknown categories
                    unknown_mask = df[col].isnull()
                    if unknown_mask.sum() > 0:
                        if self.handle_new_categories == "ignore":
                            df.loc[unknown_mask, col] = -1  # Unknown category marker
                        elif self.handle_new_categories == "most_frequent":
                            most_frequent_idx = 0  # First category
                            df.loc[unknown_mask, col] = most_frequent_idx
                
                elif encoder_type == "onehot":
                    categories = encoder_info['categories']
                    # Create one-hot encoded columns
                    for category in categories:
                        new_col = f"{col}_{category}"
                        df[new_col] = (df[col] == category).astype(int)
                    
                    # Remove original column
                    df = df.drop(columns=[col])
                
                log['actions_taken'].append(f"Encoded categorical column '{col}' using {encoder_type}")
                
            except Exception as e:
                if handle_errors:
                    log['warnings'].append(f"Failed to encode column '{col}': {e}")
                    # Keep original column as-is or convert to string
                    df[col] = df[col].astype(str)
                else:
                    raise
        
        return df
    
    def _transform_datetime_features(self, df: pd.DataFrame, log: Dict[str, Any]) -> pd.DataFrame:
        """Transform datetime features into numerical representations."""
        for col in self.datetime_columns:
            if col not in df.columns:
                continue
            
            # Convert to datetime if not already
            if not pd.api.types.is_datetime64_any_dtype(df[col]):
                df[col] = pd.to_datetime(df[col], errors='coerce')
            
            # Extract datetime components
            df[f"{col}_year"] = df[col].dt.year
            df[f"{col}_month"] = df[col].dt.month
            df[f"{col}_day"] = df[col].dt.day
            df[f"{col}_dayofweek"] = df[col].dt.dayofweek
            df[f"{col}_hour"] = df[col].dt.hour
            
            # Cyclical encoding for periodic features
            df[f"{col}_month_sin"] = np.sin(2 * np.pi * df[col].dt.month / 12)
            df[f"{col}_month_cos"] = np.cos(2 * np.pi * df[col].dt.month / 12)
            df[f"{col}_day_sin"] = np.sin(2 * np.pi * df[col].dt.day / 31)
            df[f"{col}_day_cos"] = np.cos(2 * np.pi * df[col].dt.day / 31)
            
            # Remove original datetime column
            df = df.drop(columns=[col])
            
            log['actions_taken'].append(f"Extracted datetime features from '{col}'")
        
        return df
    
    def _minimal_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply minimal transformation when full transformation fails."""
        # Just handle missing values with simple strategies
        transformed_df = df.copy()
        
        for col in transformed_df.columns:
            if transformed_df[col].isnull().sum() > 0:
                if pd.api.types.is_numeric_dtype(transformed_df[col]):
                    transformed_df[col] = transformed_df[col].fillna(0)
                else:
                    transformed_df[col] = transformed_df[col].fillna("unknown")
        
        return transformed_df
    
    def get_feature_names(self) -> List[str]:
        """Get names of features after transformation."""
        if not self.is_fitted:
            raise ValidationError("Normalizer must be fitted to get feature names")
        
        feature_names = []
        
        # Numerical features (keep original names)
        feature_names.extend(self.numerical_columns)
        
        # Categorical features
        for col in self.categorical_columns:
            if col in self.encoders:
                encoder_info = self.encoders[col]
                if encoder_info['type'] == "onehot":
                    categories = encoder_info['categories']
                    feature_names.extend([f"{col}_{cat}" for cat in categories])
                else:
                    feature_names.append(col)
        
        # Datetime features (expanded)
        for col in self.datetime_columns:
            feature_names.extend([
                f"{col}_year", f"{col}_month", f"{col}_day", f"{col}_dayofweek", f"{col}_hour",
                f"{col}_month_sin", f"{col}_month_cos", f"{col}_day_sin", f"{col}_day_cos"
            ])
        
        return feature_names
    
    def get_transformation_summary(self) -> Dict[str, Any]:
        """Get summary of fitted transformations."""
        if not self.is_fitted:
            raise ValidationError("Normalizer must be fitted to get summary")
        
        return {
            'numerical_columns': len(self.numerical_columns),
            'categorical_columns': len(self.categorical_columns),
            'datetime_columns': len(self.datetime_columns),
            'numerical_strategy': self.numerical_strategy,
            'categorical_strategy': self.categorical_strategy,
            'outlier_method': self.outlier_method,
            'outlier_action': self.outlier_action,
            'missing_strategy': self.missing_strategy,
            'column_stats': self.column_stats
        }