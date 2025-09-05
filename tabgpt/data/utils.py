"""Utility functions for data processing and analysis."""

import warnings
from typing import Dict, List, Optional, Union, Tuple, Any
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.metrics import mutual_info_score
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    warnings.warn("Plotting libraries not available. Install with: pip install matplotlib seaborn")

from .preprocessing import DataType, TypeInferencer


def infer_data_types(df: pd.DataFrame, **kwargs) -> Dict[str, DataType]:
    """Infer data types for all columns in DataFrame."""
    inferencer = TypeInferencer(**kwargs)
    return inferencer.infer_types(df)


def validate_schema(
    df: pd.DataFrame,
    expected_schema: Dict[str, DataType]
) -> Tuple[bool, List[str]]:
    """Validate DataFrame schema against expected types."""
    issues = []
    actual_types = infer_data_types(df)
    
    # Check for missing columns
    missing_cols = set(expected_schema.keys()) - set(df.columns)
    if missing_cols:
        issues.append(f"Missing columns: {missing_cols}")
    
    # Check for extra columns
    extra_cols = set(df.columns) - set(expected_schema.keys())
    if extra_cols:
        issues.append(f"Extra columns: {extra_cols}")
    
    # Check data types
    for col, expected_type in expected_schema.items():
        if col in actual_types:
            actual_type = actual_types[col]
            if actual_type != expected_type:
                issues.append(f"Column {col}: expected {expected_type.value}, got {actual_type.value}")
    
    return len(issues) == 0, issues


def compute_statistics(df: pd.DataFrame, include_correlations: bool = True) -> Dict[str, Any]:
    """Compute comprehensive statistics for DataFrame."""
    stats = {
        'shape': df.shape,
        'memory_usage': df.memory_usage(deep=True).sum(),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
        'unique_values': df.nunique().to_dict(),
        'duplicate_rows': df.duplicated().sum()
    }
    
    # Numerical statistics
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    if len(numerical_cols) > 0:
        stats['numerical_stats'] = df[numerical_cols].describe().to_dict()
        
        # Skewness and kurtosis
        stats['skewness'] = df[numerical_cols].skew().to_dict()
        stats['kurtosis'] = df[numerical_cols].kurtosis().to_dict()
    
    # Categorical statistics
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    if len(categorical_cols) > 0:
        stats['categorical_stats'] = {}
        for col in categorical_cols:
            stats['categorical_stats'][col] = {
                'unique_count': df[col].nunique(),
                'most_frequent': df[col].mode().iloc[0] if not df[col].mode().empty else None,
                'frequency': df[col].value_counts().head().to_dict()
            }
    
    # Correlations
    if include_correlations and len(numerical_cols) > 1:
        stats['correlations'] = df[numerical_cols].corr().to_dict()
    
    return stats


def detect_outliers(
    df: pd.DataFrame,
    method: str = "iqr",
    threshold: float = 1.5
) -> Dict[str, List[int]]:
    """Detect outliers in numerical columns."""
    outliers = {}
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numerical_cols:
        series = df[col].dropna()
        
        if method == "iqr":
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
            outliers[col] = df[outlier_mask].index.tolist()
        
        elif method == "z_score":
            z_scores = np.abs((series - series.mean()) / series.std())
            outlier_mask = z_scores > threshold
            outliers[col] = series[outlier_mask].index.tolist()
        
        elif method == "percentile":
            lower_percentile = (1 - threshold) / 2 * 100
            upper_percentile = (1 + threshold) / 2 * 100
            
            lower_bound = series.quantile(lower_percentile / 100)
            upper_bound = series.quantile(upper_percentile / 100)
            
            outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
            outliers[col] = df[outlier_mask].index.tolist()
    
    return outliers


def sample_dataset(
    df: pd.DataFrame,
    n_samples: Optional[int] = None,
    fraction: Optional[float] = None,
    stratify_column: Optional[str] = None,
    random_state: int = 42
) -> pd.DataFrame:
    """Sample a subset of the dataset."""
    if n_samples is None and fraction is None:
        raise ValueError("Either n_samples or fraction must be specified")
    
    if n_samples is not None and fraction is not None:
        raise ValueError("Only one of n_samples or fraction should be specified")
    
    if fraction is not None:
        n_samples = int(len(df) * fraction)
    
    n_samples = min(n_samples, len(df))
    
    if stratify_column and stratify_column in df.columns:
        # Stratified sampling
        try:
            sampled_df, _ = train_test_split(
                df,
                train_size=n_samples,
                stratify=df[stratify_column],
                random_state=random_state
            )
            return sampled_df.reset_index(drop=True)
        except ValueError:
            # Fall back to random sampling if stratification fails
            warnings.warn(f"Stratification failed for column {stratify_column}, using random sampling")
    
    # Random sampling
    return df.sample(n=n_samples, random_state=random_state).reset_index(drop=True)


def split_dataset(
    df: pd.DataFrame,
    train_size: float = 0.7,
    val_size: float = 0.15,
    test_size: float = 0.15,
    stratify_column: Optional[str] = None,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split dataset into train/validation/test sets."""
    # Validate split sizes
    total_size = train_size + val_size + test_size
    if not np.isclose(total_size, 1.0):
        raise ValueError(f"Split sizes must sum to 1.0, got {total_size}")
    
    # First split: train vs (val + test)
    stratify = df[stratify_column] if stratify_column and stratify_column in df.columns else None
    
    train_df, temp_df = train_test_split(
        df,
        train_size=train_size,
        stratify=stratify,
        random_state=random_state
    )
    
    # Second split: val vs test
    val_ratio = val_size / (val_size + test_size)
    stratify_temp = temp_df[stratify_column] if stratify_column and stratify_column in temp_df.columns else None
    
    val_df, test_df = train_test_split(
        temp_df,
        train_size=val_ratio,
        stratify=stratify_temp,
        random_state=random_state
    )
    
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True)
    )


def analyze_feature_importance(
    df: pd.DataFrame,
    target_column: str,
    method: str = "mutual_info"
) -> Dict[str, float]:
    """Analyze feature importance using various methods."""
    if target_column not in df.columns:
        raise ValueError(f"Target column {target_column} not found in DataFrame")
    
    feature_cols = [col for col in df.columns if col != target_column]
    target = df[target_column]
    
    importance_scores = {}
    
    if method == "mutual_info":
        # Mutual information
        for col in feature_cols:
            if df[col].dtype in ['object', 'category']:
                # Categorical feature
                score = mutual_info_score(df[col].fillna('missing'), target)
            else:
                # Numerical feature - discretize for mutual info
                discretized = pd.cut(df[col].fillna(df[col].median()), bins=10, labels=False)
                score = mutual_info_score(discretized, target)
            
            importance_scores[col] = score
    
    elif method == "correlation":
        # Correlation with target (for numerical features only)
        numerical_cols = df[feature_cols].select_dtypes(include=[np.number]).columns
        correlations = df[numerical_cols].corrwith(target).abs()
        importance_scores = correlations.to_dict()
    
    elif method == "variance":
        # Feature variance (higher variance = potentially more informative)
        numerical_cols = df[feature_cols].select_dtypes(include=[np.number]).columns
        variances = df[numerical_cols].var()
        importance_scores = variances.to_dict()
    
    else:
        raise ValueError(f"Unknown importance method: {method}")
    
    # Sort by importance
    return dict(sorted(importance_scores.items(), key=lambda x: x[1], reverse=True))


def detect_data_drift(
    df_reference: pd.DataFrame,
    df_current: pd.DataFrame,
    threshold: float = 0.1
) -> Dict[str, Any]:
    """Detect data drift between reference and current datasets."""
    drift_report = {
        'columns_with_drift': [],
        'drift_scores': {},
        'summary': {}
    }
    
    common_columns = set(df_reference.columns) & set(df_current.columns)
    
    for col in common_columns:
        ref_series = df_reference[col].dropna()
        cur_series = df_current[col].dropna()
        
        if len(ref_series) == 0 or len(cur_series) == 0:
            continue
        
        if df_reference[col].dtype in ['object', 'category']:
            # Categorical drift using distribution comparison
            ref_dist = ref_series.value_counts(normalize=True)
            cur_dist = cur_series.value_counts(normalize=True)
            
            # Align distributions
            all_categories = set(ref_dist.index) | set(cur_dist.index)
            ref_aligned = ref_dist.reindex(all_categories, fill_value=0)
            cur_aligned = cur_dist.reindex(all_categories, fill_value=0)
            
            # Jensen-Shannon divergence
            try:
                from scipy.spatial.distance import jensenshannon
                drift_score = jensenshannon(ref_aligned.values, cur_aligned.values)
            except ImportError:
                # Fallback to simple L1 distance
                drift_score = np.sum(np.abs(ref_aligned.values - cur_aligned.values)) / 2
        
        else:
            # Numerical drift using Kolmogorov-Smirnov test
            try:
                from scipy.stats import ks_2samp
                statistic, p_value = ks_2samp(ref_series, cur_series)
                drift_score = statistic
            except ImportError:
                # Fallback to simple mean difference
                drift_score = abs(ref_series.mean() - cur_series.mean()) / (ref_series.std() + 1e-8)
        
        drift_report['drift_scores'][col] = drift_score
        
        if drift_score > threshold:
            drift_report['columns_with_drift'].append(col)
    
    # Summary statistics
    drift_report['summary'] = {
        'total_columns': len(common_columns),
        'columns_with_drift': len(drift_report['columns_with_drift']),
        'drift_percentage': len(drift_report['columns_with_drift']) / len(common_columns) * 100,
        'max_drift_score': max(drift_report['drift_scores'].values()) if drift_report['drift_scores'] else 0,
        'avg_drift_score': np.mean(list(drift_report['drift_scores'].values())) if drift_report['drift_scores'] else 0
    }
    
    return drift_report


def profile_dataset(df: pd.DataFrame, output_file: Optional[str] = None) -> Dict[str, Any]:
    """Create a comprehensive dataset profile."""
    profile = {
        'basic_info': {
            'shape': df.shape,
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
            'duplicate_rows': df.duplicated().sum(),
            'duplicate_percentage': df.duplicated().sum() / len(df) * 100
        },
        'column_info': {},
        'data_quality': {},
        'statistics': compute_statistics(df),
        'data_types': infer_data_types(df),
        'outliers': detect_outliers(df)
    }
    
    # Per-column analysis
    for col in df.columns:
        col_info = {
            'dtype': str(df[col].dtype),
            'non_null_count': df[col].count(),
            'null_count': df[col].isnull().sum(),
            'null_percentage': df[col].isnull().sum() / len(df) * 100,
            'unique_count': df[col].nunique(),
            'unique_percentage': df[col].nunique() / len(df) * 100
        }
        
        if df[col].dtype in ['object', 'category']:
            # Categorical column
            col_info.update({
                'most_frequent': df[col].mode().iloc[0] if not df[col].mode().empty else None,
                'least_frequent': df[col].value_counts().index[-1] if len(df[col].value_counts()) > 0 else None,
                'cardinality': df[col].nunique()
            })
        else:
            # Numerical column
            col_info.update({
                'mean': df[col].mean(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max(),
                'median': df[col].median(),
                'skewness': df[col].skew(),
                'kurtosis': df[col].kurtosis()
            })
        
        profile['column_info'][col] = col_info
    
    # Data quality assessment
    profile['data_quality'] = {
        'completeness': (1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100,
        'high_missing_columns': [
            col for col in df.columns 
            if df[col].isnull().sum() / len(df) > 0.5
        ],
        'constant_columns': [
            col for col in df.columns 
            if df[col].nunique() <= 1
        ],
        'high_cardinality_columns': [
            col for col in df.columns 
            if df[col].nunique() > len(df) * 0.9
        ]
    }
    
    # Save to file if requested
    if output_file:
        output_path = Path(output_file)
        if output_path.suffix == '.json':
            import json
            with open(output_path, 'w') as f:
                # Convert numpy types to native Python types for JSON serialization
                def convert_types(obj):
                    if isinstance(obj, np.integer):
                        return int(obj)
                    elif isinstance(obj, np.floating):
                        return float(obj)
                    elif isinstance(obj, np.ndarray):
                        return obj.tolist()
                    elif isinstance(obj, DataType):
                        return obj.value
                    return obj
                
                json.dump(profile, f, indent=2, default=convert_types)
        else:
            # Save as pickle
            import pickle
            with open(output_path, 'wb') as f:
                pickle.dump(profile, f)
    
    return profile


def visualize_dataset(
    df: pd.DataFrame,
    output_dir: Optional[str] = None,
    max_categorical_levels: int = 20
) -> None:
    """Create visualizations for dataset exploration."""
    if not PLOTTING_AVAILABLE:
        warnings.warn("Plotting libraries not available. Skipping visualization.")
        return
    
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
    
    # Set up plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Missing values heatmap
    if df.isnull().sum().sum() > 0:
        plt.figure(figsize=(12, 8))
        sns.heatmap(df.isnull(), cbar=True, yticklabels=False, cmap='viridis')
        plt.title('Missing Values Heatmap')
        plt.tight_layout()
        
        if output_dir:
            plt.savefig(output_path / 'missing_values_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    # 2. Correlation matrix for numerical features
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    if len(numerical_cols) > 1:
        plt.figure(figsize=(10, 8))
        correlation_matrix = df[numerical_cols].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, fmt='.2f')
        plt.title('Correlation Matrix - Numerical Features')
        plt.tight_layout()
        
        if output_dir:
            plt.savefig(output_path / 'correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    # 3. Distribution plots for numerical features
    if len(numerical_cols) > 0:
        n_cols = min(4, len(numerical_cols))
        n_rows = (len(numerical_cols) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 else []
        
        for i, col in enumerate(numerical_cols):
            if i < len(axes):
                df[col].hist(bins=30, ax=axes[i], alpha=0.7)
                axes[i].set_title(f'Distribution of {col}')
                axes[i].set_xlabel(col)
                axes[i].set_ylabel('Frequency')
        
        # Hide unused subplots
        for i in range(len(numerical_cols), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if output_dir:
            plt.savefig(output_path / 'numerical_distributions.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    # 4. Categorical feature analysis
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        if df[col].nunique() <= max_categorical_levels:
            plt.figure(figsize=(10, 6))
            value_counts = df[col].value_counts()
            
            if len(value_counts) > 10:
                # Show top 10 categories
                value_counts = value_counts.head(10)
            
            value_counts.plot(kind='bar')
            plt.title(f'Distribution of {col}')
            plt.xlabel(col)
            plt.ylabel('Count')
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            if output_dir:
                safe_col_name = "".join(c for c in col if c.isalnum() or c in ('-', '_'))
                plt.savefig(output_path / f'categorical_{safe_col_name}.png', 
                           dpi=300, bbox_inches='tight')
            plt.show()


def compare_datasets(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    name1: str = "Dataset 1",
    name2: str = "Dataset 2"
) -> Dict[str, Any]:
    """Compare two datasets and identify differences."""
    comparison = {
        'basic_comparison': {},
        'schema_comparison': {},
        'statistical_comparison': {},
        'drift_analysis': {}
    }
    
    # Basic comparison
    comparison['basic_comparison'] = {
        f'{name1}_shape': df1.shape,
        f'{name2}_shape': df2.shape,
        f'{name1}_memory_mb': df1.memory_usage(deep=True).sum() / 1024 / 1024,
        f'{name2}_memory_mb': df2.memory_usage(deep=True).sum() / 1024 / 1024,
        'shape_difference': (df2.shape[0] - df1.shape[0], df2.shape[1] - df1.shape[1])
    }
    
    # Schema comparison
    cols1 = set(df1.columns)
    cols2 = set(df2.columns)
    
    comparison['schema_comparison'] = {
        'common_columns': list(cols1 & cols2),
        f'only_in_{name1}': list(cols1 - cols2),
        f'only_in_{name2}': list(cols2 - cols1),
        'column_count_difference': len(cols2) - len(cols1)
    }
    
    # Statistical comparison for common columns
    common_cols = cols1 & cols2
    comparison['statistical_comparison'] = {}
    
    for col in common_cols:
        if df1[col].dtype in ['object', 'category'] or df2[col].dtype in ['object', 'category']:
            # Categorical comparison
            unique1 = df1[col].nunique()
            unique2 = df2[col].nunique()
            
            comparison['statistical_comparison'][col] = {
                'type': 'categorical',
                f'{name1}_unique_count': unique1,
                f'{name2}_unique_count': unique2,
                'unique_count_difference': unique2 - unique1
            }
        else:
            # Numerical comparison
            stats1 = df1[col].describe()
            stats2 = df2[col].describe()
            
            comparison['statistical_comparison'][col] = {
                'type': 'numerical',
                f'{name1}_mean': stats1['mean'],
                f'{name2}_mean': stats2['mean'],
                'mean_difference': stats2['mean'] - stats1['mean'],
                f'{name1}_std': stats1['std'],
                f'{name2}_std': stats2['std'],
                'std_difference': stats2['std'] - stats1['std']
            }
    
    # Data drift analysis
    if len(common_cols) > 0:
        comparison['drift_analysis'] = detect_data_drift(df1, df2)
    
    return comparison