"""Preprocessing utilities for tabular data."""

import warnings
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Tuple, Any, Set
from dataclasses import dataclass
from enum import Enum
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope


class DataType(Enum):
    """Data types for tabular columns."""
    CATEGORICAL = "categorical"
    NUMERICAL = "numerical"
    BOOLEAN = "boolean"
    DATETIME = "datetime"
    TEXT = "text"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class MissingValueStrategy(Enum):
    """Strategies for handling missing values."""
    DROP = "drop"
    MEAN = "mean"
    MEDIAN = "median"
    MODE = "mode"
    CONSTANT = "constant"
    KNN = "knn"
    FORWARD_FILL = "ffill"
    BACKWARD_FILL = "bfill"
    INTERPOLATE = "interpolate"


class OutlierMethod(Enum):
    """Methods for outlier detection."""
    IQR = "iqr"
    Z_SCORE = "z_score"
    ISOLATION_FOREST = "isolation_forest"
    ELLIPTIC_ENVELOPE = "elliptic_envelope"
    PERCENTILE = "percentile"


@dataclass
class DataQualityReport:
    """Report on data quality issues."""
    n_rows: int
    n_columns: int
    missing_values: Dict[str, int]
    missing_percentage: Dict[str, float]
    duplicate_rows: int
    outliers: Dict[str, List[int]]
    data_types: Dict[str, DataType]
    unique_values: Dict[str, int]
    constant_columns: List[str]
    high_cardinality_columns: List[str]
    recommendations: List[str]


@dataclass
class PreprocessingConfig:
    """Configuration for preprocessing pipeline."""
    handle_missing: bool = True
    missing_strategy: MissingValueStrategy = MissingValueStrategy.MEDIAN
    handle_outliers: bool = True
    outlier_method: OutlierMethod = OutlierMethod.IQR
    normalize_numerical: bool = True
    normalization_method: str = "standard"  # standard, minmax, robust
    encode_categorical: bool = True
    encoding_method: str = "label"  # label, onehot, target
    handle_datetime: bool = True
    extract_datetime_features: bool = True
    remove_duplicates: bool = True
    remove_constant_columns: bool = True
    max_cardinality: int = 100
    outlier_threshold: float = 0.05


class TypeInferencer:
    """Automatic data type inference for tabular columns."""
    
    def __init__(
        self,
        categorical_threshold: float = 0.05,
        datetime_formats: Optional[List[str]] = None,
        boolean_values: Optional[Set[str]] = None
    ):
        self.categorical_threshold = categorical_threshold
        self.datetime_formats = datetime_formats or [
            '%Y-%m-%d', '%Y-%m-%d %H:%M:%S', '%d/%m/%Y', '%m/%d/%Y',
            '%Y-%m-%d %H:%M:%S.%f', '%Y%m%d', '%d-%m-%Y'
        ]
        self.boolean_values = boolean_values or {
            'true', 'false', 'yes', 'no', '1', '0', 't', 'f', 'y', 'n'
        }
    
    def infer_column_type(self, series: pd.Series) -> DataType:
        """Infer data type for a single column."""
        # Remove missing values for analysis
        non_null_series = series.dropna()
        
        if len(non_null_series) == 0:
            return DataType.UNKNOWN
        
        # Check if already datetime
        if pd.api.types.is_datetime64_any_dtype(series):
            return DataType.DATETIME
        
        # Check if numeric
        if pd.api.types.is_numeric_dtype(series):
            # Check if boolean (0/1 only)
            unique_values = set(non_null_series.unique())
            if unique_values.issubset({0, 1, 0.0, 1.0}):
                return DataType.BOOLEAN
            return DataType.NUMERICAL
        
        # Check if boolean strings
        if isinstance(non_null_series.iloc[0], str):
            unique_str_values = {str(v).lower().strip() for v in non_null_series.unique()}
            if unique_str_values.issubset(self.boolean_values):
                return DataType.BOOLEAN
        
        # Try to parse as datetime
        if self._is_datetime_column(non_null_series):
            return DataType.DATETIME
        
        # Check if categorical based on cardinality
        n_unique = non_null_series.nunique()
        n_total = len(non_null_series)
        
        if n_unique / n_total <= self.categorical_threshold or n_unique <= 20:
            return DataType.CATEGORICAL
        
        # Check if text (long strings)
        if isinstance(non_null_series.iloc[0], str):
            avg_length = non_null_series.str.len().mean()
            if avg_length > 50:  # Arbitrary threshold for text
                return DataType.TEXT
        
        # Default to categorical for string data
        if non_null_series.dtype == 'object':
            return DataType.CATEGORICAL
        
        return DataType.UNKNOWN
    
    def _is_datetime_column(self, series: pd.Series) -> bool:
        """Check if column can be parsed as datetime."""
        sample_size = min(100, len(series))
        sample = series.sample(sample_size) if len(series) > sample_size else series
        
        for fmt in self.datetime_formats:
            try:
                pd.to_datetime(sample, format=fmt, errors='raise')
                return True
            except (ValueError, TypeError):
                continue
        
        # Try automatic parsing
        try:
            pd.to_datetime(sample, errors='raise', infer_datetime_format=True)
            return True
        except (ValueError, TypeError):
            return False
    
    def infer_types(self, df: pd.DataFrame) -> Dict[str, DataType]:
        """Infer data types for all columns in DataFrame."""
        return {col: self.infer_column_type(df[col]) for col in df.columns}


class DataQualityChecker:
    """Comprehensive data quality assessment."""
    
    def __init__(self, high_cardinality_threshold: int = 100):
        self.high_cardinality_threshold = high_cardinality_threshold
        self.type_inferencer = TypeInferencer()
    
    def check_quality(self, df: pd.DataFrame) -> DataQualityReport:
        """Perform comprehensive data quality check."""
        # Basic info
        n_rows, n_columns = df.shape
        
        # Missing values
        missing_values = df.isnull().sum().to_dict()
        missing_percentage = {
            col: (count / n_rows) * 100 
            for col, count in missing_values.items()
        }
        
        # Duplicate rows
        duplicate_rows = df.duplicated().sum()
        
        # Data types
        data_types = self.type_inferencer.infer_types(df)
        
        # Unique values
        unique_values = {col: df[col].nunique() for col in df.columns}
        
        # Constant columns
        constant_columns = [
            col for col, n_unique in unique_values.items() 
            if n_unique <= 1
        ]
        
        # High cardinality columns
        high_cardinality_columns = [
            col for col, n_unique in unique_values.items()
            if n_unique > self.high_cardinality_threshold
        ]
        
        # Outliers (for numerical columns only)
        outliers = {}
        for col in df.columns:
            if data_types[col] == DataType.NUMERICAL:
                outlier_indices = self._detect_outliers_iqr(df[col])
                if outlier_indices:
                    outliers[col] = outlier_indices
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            missing_percentage, duplicate_rows, constant_columns,
            high_cardinality_columns, outliers, data_types
        )
        
        return DataQualityReport(
            n_rows=n_rows,
            n_columns=n_columns,
            missing_values=missing_values,
            missing_percentage=missing_percentage,
            duplicate_rows=duplicate_rows,
            outliers=outliers,
            data_types=data_types,
            unique_values=unique_values,
            constant_columns=constant_columns,
            high_cardinality_columns=high_cardinality_columns,
            recommendations=recommendations
        )
    
    def _detect_outliers_iqr(self, series: pd.Series) -> List[int]:
        """Detect outliers using IQR method."""
        if not pd.api.types.is_numeric_dtype(series):
            return []
        
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outlier_mask = (series < lower_bound) | (series > upper_bound)
        return series[outlier_mask].index.tolist()
    
    def _generate_recommendations(
        self,
        missing_percentage: Dict[str, float],
        duplicate_rows: int,
        constant_columns: List[str],
        high_cardinality_columns: List[str],
        outliers: Dict[str, List[int]],
        data_types: Dict[str, DataType]
    ) -> List[str]:
        """Generate data quality recommendations."""
        recommendations = []
        
        # Missing values
        high_missing = [col for col, pct in missing_percentage.items() if pct > 50]
        if high_missing:
            recommendations.append(
                f"Consider dropping columns with >50% missing values: {high_missing}"
            )
        
        moderate_missing = [col for col, pct in missing_percentage.items() if 10 < pct <= 50]
        if moderate_missing:
            recommendations.append(
                f"Consider imputation for columns with moderate missing values: {moderate_missing}"
            )
        
        # Duplicates
        if duplicate_rows > 0:
            recommendations.append(f"Remove {duplicate_rows} duplicate rows")
        
        # Constant columns
        if constant_columns:
            recommendations.append(f"Remove constant columns: {constant_columns}")
        
        # High cardinality
        if high_cardinality_columns:
            recommendations.append(
                f"Consider feature engineering for high cardinality columns: {high_cardinality_columns}"
            )
        
        # Outliers
        if outliers:
            recommendations.append(
                f"Review outliers in columns: {list(outliers.keys())}"
            )
        
        # Data types
        unknown_types = [col for col, dtype in data_types.items() if dtype == DataType.UNKNOWN]
        if unknown_types:
            recommendations.append(f"Review data types for: {unknown_types}")
        
        return recommendations


class MissingValueHandler:
    """Handle missing values in tabular data."""
    
    def __init__(self, strategy: MissingValueStrategy = MissingValueStrategy.MEDIAN):
        self.strategy = strategy
        self.imputers = {}
    
    def fit(self, df: pd.DataFrame, data_types: Dict[str, DataType]) -> 'MissingValueHandler':
        """Fit missing value imputers."""
        for col in df.columns:
            if df[col].isnull().any():
                dtype = data_types.get(col, DataType.UNKNOWN)
                
                if self.strategy == MissingValueStrategy.KNN:
                    # KNN imputer for all types
                    self.imputers[col] = KNNImputer(n_neighbors=5)
                    # Convert categorical to numeric for KNN
                    if dtype == DataType.CATEGORICAL:
                        le = LabelEncoder()
                        non_null_mask = df[col].notna()
                        if non_null_mask.any():
                            encoded_values = le.fit_transform(df.loc[non_null_mask, col])
                            temp_series = df[col].copy()
                            temp_series.loc[non_null_mask] = encoded_values
                            self.imputers[col].fit(temp_series.values.reshape(-1, 1))
                            self.imputers[f"{col}_encoder"] = le
                    else:
                        self.imputers[col].fit(df[[col]])
                
                elif dtype == DataType.NUMERICAL:
                    if self.strategy == MissingValueStrategy.MEAN:
                        self.imputers[col] = SimpleImputer(strategy='mean')
                    elif self.strategy == MissingValueStrategy.MEDIAN:
                        self.imputers[col] = SimpleImputer(strategy='median')
                    else:
                        self.imputers[col] = SimpleImputer(strategy='constant', fill_value=0)
                    
                    self.imputers[col].fit(df[[col]])
                
                elif dtype == DataType.CATEGORICAL:
                    if self.strategy == MissingValueStrategy.MODE:
                        self.imputers[col] = SimpleImputer(strategy='most_frequent')
                    else:
                        self.imputers[col] = SimpleImputer(strategy='constant', fill_value='Unknown')
                    
                    self.imputers[col].fit(df[[col]])
                
                elif dtype == DataType.BOOLEAN:
                    # Use mode for boolean
                    self.imputers[col] = SimpleImputer(strategy='most_frequent')
                    self.imputers[col].fit(df[[col]])
        
        return self
    
    def transform(self, df: pd.DataFrame, data_types: Dict[str, DataType]) -> pd.DataFrame:
        """Transform DataFrame by imputing missing values."""
        df_imputed = df.copy()
        
        for col in df.columns:
            if col in self.imputers and df[col].isnull().any():
                if self.strategy == MissingValueStrategy.FORWARD_FILL:
                    df_imputed[col] = df_imputed[col].fillna(method='ffill')
                elif self.strategy == MissingValueStrategy.BACKWARD_FILL:
                    df_imputed[col] = df_imputed[col].fillna(method='bfill')
                elif self.strategy == MissingValueStrategy.INTERPOLATE:
                    if data_types.get(col) == DataType.NUMERICAL:
                        df_imputed[col] = df_imputed[col].interpolate()
                    else:
                        df_imputed[col] = df_imputed[col].fillna(method='ffill')
                else:
                    # Use fitted imputer
                    imputed_values = self.imputers[col].transform(df[[col]])
                    df_imputed[col] = imputed_values.flatten()
        
        return df_imputed


class OutlierDetector:
    """Detect and handle outliers in numerical data."""
    
    def __init__(
        self,
        method: OutlierMethod = OutlierMethod.IQR,
        threshold: float = 0.05,
        contamination: float = 0.1
    ):
        self.method = method
        self.threshold = threshold
        self.contamination = contamination
        self.detectors = {}
    
    def fit(self, df: pd.DataFrame, data_types: Dict[str, DataType]) -> 'OutlierDetector':
        """Fit outlier detectors."""
        numerical_cols = [
            col for col, dtype in data_types.items() 
            if dtype == DataType.NUMERICAL and col in df.columns
        ]
        
        if self.method == OutlierMethod.ISOLATION_FOREST:
            if numerical_cols:
                self.detectors['isolation_forest'] = IsolationForest(
                    contamination=self.contamination,
                    random_state=42
                )
                self.detectors['isolation_forest'].fit(df[numerical_cols].fillna(0))
        
        elif self.method == OutlierMethod.ELLIPTIC_ENVELOPE:
            if numerical_cols:
                self.detectors['elliptic_envelope'] = EllipticEnvelope(
                    contamination=self.contamination,
                    random_state=42
                )
                self.detectors['elliptic_envelope'].fit(df[numerical_cols].fillna(0))
        
        # Store column-specific statistics for IQR and Z-score methods
        for col in numerical_cols:
            series = df[col].dropna()
            if len(series) > 0:
                self.detectors[f"{col}_mean"] = series.mean()
                self.detectors[f"{col}_std"] = series.std()
                self.detectors[f"{col}_q1"] = series.quantile(0.25)
                self.detectors[f"{col}_q3"] = series.quantile(0.75)
                self.detectors[f"{col}_iqr"] = self.detectors[f"{col}_q3"] - self.detectors[f"{col}_q1"]
        
        return self
    
    def detect_outliers(self, df: pd.DataFrame, data_types: Dict[str, DataType]) -> Dict[str, np.ndarray]:
        """Detect outliers and return boolean masks."""
        outlier_masks = {}
        
        numerical_cols = [
            col for col, dtype in data_types.items() 
            if dtype == DataType.NUMERICAL and col in df.columns
        ]
        
        if self.method == OutlierMethod.IQR:
            for col in numerical_cols:
                if f"{col}_iqr" in self.detectors:
                    q1 = self.detectors[f"{col}_q1"]
                    q3 = self.detectors[f"{col}_q3"]
                    iqr = self.detectors[f"{col}_iqr"]
                    
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    
                    outlier_masks[col] = (df[col] < lower_bound) | (df[col] > upper_bound)
        
        elif self.method == OutlierMethod.Z_SCORE:
            for col in numerical_cols:
                if f"{col}_mean" in self.detectors:
                    mean = self.detectors[f"{col}_mean"]
                    std = self.detectors[f"{col}_std"]
                    
                    if std > 0:
                        z_scores = np.abs((df[col] - mean) / std)
                        outlier_masks[col] = z_scores > 3  # 3 standard deviations
        
        elif self.method == OutlierMethod.ISOLATION_FOREST:
            if 'isolation_forest' in self.detectors and numerical_cols:
                predictions = self.detectors['isolation_forest'].predict(df[numerical_cols].fillna(0))
                outlier_mask = predictions == -1
                
                for col in numerical_cols:
                    outlier_masks[col] = outlier_mask
        
        elif self.method == OutlierMethod.ELLIPTIC_ENVELOPE:
            if 'elliptic_envelope' in self.detectors and numerical_cols:
                predictions = self.detectors['elliptic_envelope'].predict(df[numerical_cols].fillna(0))
                outlier_mask = predictions == -1
                
                for col in numerical_cols:
                    outlier_masks[col] = outlier_mask
        
        return outlier_masks
    
    def remove_outliers(self, df: pd.DataFrame, data_types: Dict[str, DataType]) -> pd.DataFrame:
        """Remove outliers from DataFrame."""
        outlier_masks = self.detect_outliers(df, data_types)
        
        if not outlier_masks:
            return df
        
        # Count outliers per row instead of removing rows with ANY outlier
        outlier_count = np.zeros(len(df), dtype=int)
        for mask in outlier_masks.values():
            if hasattr(mask, 'fillna'):
                # Pandas Series
                outlier_count += mask.fillna(False).values.astype(int)
            else:
                # Numpy array
                mask_array = np.array(mask) if not isinstance(mask, np.ndarray) else mask
                outlier_count += np.nan_to_num(mask_array, nan=False).astype(int)
        
        # Only remove rows with outliers in multiple columns (more than half)
        threshold = max(1, len(outlier_masks) // 2)
        rows_to_remove = outlier_count > threshold
        
        # Don't remove more than 20% of the data
        max_remove = int(0.2 * len(df))
        if rows_to_remove.sum() > max_remove:
            # Keep only the most extreme outliers
            outlier_scores = outlier_count * rows_to_remove
            top_outliers = np.argsort(outlier_scores)[-max_remove:]
            rows_to_remove = np.zeros(len(df), dtype=bool)
            rows_to_remove[top_outliers] = True
        
        return df[~rows_to_remove].reset_index(drop=True)


class DataNormalizer:
    """Normalize numerical features."""
    
    def __init__(self, method: str = "standard"):
        self.method = method
        self.scalers = {}
        
        if method == "standard":
            self.scaler_class = StandardScaler
        elif method == "minmax":
            self.scaler_class = MinMaxScaler
        elif method == "robust":
            self.scaler_class = RobustScaler
        else:
            raise ValueError(f"Unknown normalization method: {method}")
    
    def fit(self, df: pd.DataFrame, data_types: Dict[str, DataType]) -> 'DataNormalizer':
        """Fit normalizers for numerical columns."""
        for col, dtype in data_types.items():
            if dtype == DataType.NUMERICAL and col in df.columns:
                self.scalers[col] = self.scaler_class()
                self.scalers[col].fit(df[[col]].fillna(0))
        
        return self
    
    def transform(self, df: pd.DataFrame, data_types: Dict[str, DataType]) -> pd.DataFrame:
        """Transform numerical columns."""
        df_normalized = df.copy()
        
        for col in self.scalers:
            if col in df.columns and len(df) > 0:
                normalized_values = self.scalers[col].transform(df[[col]].fillna(0))
                df_normalized[col] = normalized_values.flatten()
        
        return df_normalized


class SchemaValidator:
    """Validate and normalize data schemas."""
    
    def __init__(self, expected_schema: Optional[Dict[str, DataType]] = None):
        self.expected_schema = expected_schema or {}
    
    def validate_schema(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate DataFrame against expected schema."""
        issues = []
        
        if not self.expected_schema:
            return True, issues
        
        # Check for missing columns
        missing_cols = set(self.expected_schema.keys()) - set(df.columns)
        if missing_cols:
            issues.append(f"Missing columns: {missing_cols}")
        
        # Check for extra columns
        extra_cols = set(df.columns) - set(self.expected_schema.keys())
        if extra_cols:
            issues.append(f"Extra columns: {extra_cols}")
        
        # Check data types
        type_inferencer = TypeInferencer()
        actual_types = type_inferencer.infer_types(df)
        
        for col, expected_type in self.expected_schema.items():
            if col in actual_types:
                actual_type = actual_types[col]
                if actual_type != expected_type:
                    issues.append(f"Column {col}: expected {expected_type}, got {actual_type}")
        
        return len(issues) == 0, issues
    
    def normalize_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize DataFrame to match expected schema."""
        df_normalized = df.copy()
        
        if not self.expected_schema:
            return df_normalized
        
        # Add missing columns with default values
        for col, dtype in self.expected_schema.items():
            if col not in df_normalized.columns:
                if dtype == DataType.NUMERICAL:
                    df_normalized[col] = 0.0
                elif dtype == DataType.BOOLEAN:
                    df_normalized[col] = False
                else:
                    df_normalized[col] = "Unknown"
        
        # Remove extra columns
        expected_cols = list(self.expected_schema.keys())
        df_normalized = df_normalized[expected_cols]
        
        # Convert data types
        for col, expected_type in self.expected_schema.items():
            if col in df_normalized.columns:
                try:
                    if expected_type == DataType.NUMERICAL:
                        df_normalized[col] = pd.to_numeric(df_normalized[col], errors='coerce')
                    elif expected_type == DataType.BOOLEAN:
                        df_normalized[col] = df_normalized[col].astype(bool)
                    elif expected_type == DataType.DATETIME:
                        df_normalized[col] = pd.to_datetime(df_normalized[col], errors='coerce')
                    elif expected_type == DataType.CATEGORICAL:
                        df_normalized[col] = df_normalized[col].astype(str)
                except Exception as e:
                    warnings.warn(f"Failed to convert column {col} to {expected_type}: {e}")
        
        return df_normalized


class TabularPreprocessor:
    """Complete preprocessing pipeline for tabular data."""
    
    def __init__(self, config: Optional[PreprocessingConfig] = None):
        self.config = config or PreprocessingConfig()
        self.type_inferencer = TypeInferencer()
        self.quality_checker = DataQualityChecker()
        self.missing_handler = None
        self.outlier_detector = None
        self.normalizer = None
        self.schema_validator = None
        
        self.data_types = {}
        self.quality_report = None
        self.is_fitted = False
    
    def fit(self, df: pd.DataFrame, target_column: Optional[str] = None) -> 'TabularPreprocessor':
        """Fit preprocessing pipeline."""
        # Infer data types
        self.data_types = self.type_inferencer.infer_types(df)
        
        # Generate quality report
        self.quality_report = self.quality_checker.check_quality(df)
        
        # Prepare data for fitting (remove duplicates, constant columns)
        df_clean = df.copy()
        
        if self.config.remove_duplicates:
            df_clean = df_clean.drop_duplicates()
        
        if self.config.remove_constant_columns:
            constant_cols = self.quality_report.constant_columns
            df_clean = df_clean.drop(columns=constant_cols, errors='ignore')
            # Update data types
            for col in constant_cols:
                if col in self.data_types:
                    del self.data_types[col]
        
        # Fit missing value handler
        if self.config.handle_missing:
            self.missing_handler = MissingValueHandler(self.config.missing_strategy)
            self.missing_handler.fit(df_clean, self.data_types)
        
        # Fit outlier detector
        if self.config.handle_outliers:
            self.outlier_detector = OutlierDetector(
                method=self.config.outlier_method,
                threshold=self.config.outlier_threshold
            )
            self.outlier_detector.fit(df_clean, self.data_types)
        
        # Fit normalizer
        if self.config.normalize_numerical:
            self.normalizer = DataNormalizer(self.config.normalization_method)
            self.normalizer.fit(df_clean, self.data_types)
        
        self.is_fitted = True
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform DataFrame using fitted preprocessing pipeline."""
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")
        
        df_processed = df.copy()
        
        # Remove duplicates
        if self.config.remove_duplicates:
            df_processed = df_processed.drop_duplicates()
        
        # Remove constant columns
        if self.config.remove_constant_columns and self.quality_report:
            constant_cols = [col for col in self.quality_report.constant_columns if col in df_processed.columns]
            df_processed = df_processed.drop(columns=constant_cols, errors='ignore')
        
        # Handle missing values
        if self.config.handle_missing and self.missing_handler:
            df_processed = self.missing_handler.transform(df_processed, self.data_types)
        
        # Handle outliers
        if self.config.handle_outliers and self.outlier_detector:
            df_processed = self.outlier_detector.remove_outliers(df_processed, self.data_types)
        
        # Normalize numerical features
        if self.config.normalize_numerical and self.normalizer:
            df_processed = self.normalizer.transform(df_processed, self.data_types)
        
        return df_processed
    
    def fit_transform(self, df: pd.DataFrame, target_column: Optional[str] = None) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(df, target_column).transform(df)
    
    def get_feature_info(self) -> Dict[str, Any]:
        """Get information about processed features."""
        return {
            'data_types': self.data_types,
            'quality_report': self.quality_report,
            'config': self.config,
            'n_features': len(self.data_types),
            'numerical_features': [col for col, dtype in self.data_types.items() if dtype == DataType.NUMERICAL],
            'categorical_features': [col for col, dtype in self.data_types.items() if dtype == DataType.CATEGORICAL],
            'boolean_features': [col for col, dtype in self.data_types.items() if dtype == DataType.BOOLEAN],
            'datetime_features': [col for col, dtype in self.data_types.items() if dtype == DataType.DATETIME]
        }