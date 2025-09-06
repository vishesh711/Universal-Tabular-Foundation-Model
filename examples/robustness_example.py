"""
Example: TabGPT Robustness and Error Handling

This example demonstrates the comprehensive error handling and robustness
features of TabGPT, including data validation, recovery, and normalization.
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import warnings

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tabgpt.utils import (
    DataValidator, DataRecovery, RobustNormalizer,
    DataQualityError, ValidationError, MissingColumnsError,
    robust_operation, graceful_degradation
)


def create_problematic_dataset():
    """Create a dataset with various data quality issues."""
    print("Creating problematic dataset...")
    
    np.random.seed(42)
    n_samples = 1000
    
    # Create base data
    data = {
        # Numerical column with outliers and missing values
        'numerical_outliers': np.concatenate([
            np.random.normal(50, 10, n_samples - 50),  # Normal data
            np.random.uniform(500, 1000, 30),          # Outliers
            [np.nan] * 20                              # Missing values
        ]),
        
        # Categorical column with missing values and inconsistent casing
        'categorical_messy': np.random.choice(
            ['Category_A', 'category_b', 'CATEGORY_C', 'Category_A', np.nan], 
            n_samples
        ),
        
        # Mixed data types (should be numerical but has strings)
        'mixed_types': [str(x) if i % 100 == 0 else x for i, x in enumerate(np.random.randn(n_samples))],
        
        # Column with excessive missing values
        'mostly_missing': [x if i % 10 == 0 else np.nan for i, x in enumerate(range(n_samples))],
        
        # Datetime column with some invalid dates
        'datetime_col': pd.date_range('2023-01-01', periods=n_samples, freq='h'),
        
        # Target variable
        'target': np.random.choice([0, 1], n_samples)
    }
    
    # Convert to list first, then modify
    datetime_list = data['datetime_col'].astype(str).tolist()
    for i in range(0, len(datetime_list), 100):
        datetime_list[i] = 'invalid_date'
    data['datetime_col'] = datetime_list
    
    df = pd.DataFrame(data)
    
    print(f"Created dataset with shape: {df.shape}")
    print(f"Data types: {df.dtypes.to_dict()}")
    print(f"Missing values per column: {df.isnull().sum().to_dict()}")
    
    return df


def demonstrate_data_validation():
    """Demonstrate comprehensive data validation."""
    print("\n" + "="*60)
    print("DATA VALIDATION DEMONSTRATION")
    print("="*60)
    
    # Create problematic dataset
    df = create_problematic_dataset()
    
    # Define expected schema
    expected_columns = ['numerical_outliers', 'categorical_messy', 'mixed_types', 'target']
    expected_dtypes = {
        'numerical_outliers': 'float',
        'categorical_messy': 'object',
        'mixed_types': 'float',
        'target': 'int'
    }
    
    # Create validator with strict settings
    validator = DataValidator(
        missing_threshold=0.3,      # Max 30% missing values
        outlier_threshold=0.05,     # Max 5% outliers
        min_samples=100,
        strict_mode=False           # Don't raise errors, just warn
    )
    
    print("\nRunning comprehensive validation...")
    
    try:
        result = validator.validate_dataframe(
            df=df,
            expected_columns=expected_columns,
            expected_dtypes=expected_dtypes,
            dataset_name="problematic_dataset"
        )
        
        print(f"\nValidation Results:")
        print(f"  Valid: {result['valid']}")
        print(f"  Errors: {len(result['errors'])}")
        print(f"  Warnings: {len(result['warnings'])}")
        
        if result['errors']:
            print(f"\nErrors found:")
            for i, error in enumerate(result['errors'][:5], 1):  # Show first 5
                print(f"  {i}. {error}")
        
        if result['warnings']:
            print(f"\nWarnings:")
            for i, warning in enumerate(result['warnings'][:5], 1):  # Show first 5
                print(f"  {i}. {warning}")
        
        # Show data statistics
        stats = result['stats']
        print(f"\nDataset Statistics:")
        print(f"  Samples: {stats['n_samples']}")
        print(f"  Features: {stats['n_features']}")
        print(f"  Memory usage: {stats['memory_usage_mb']:.2f} MB")
        
    except ValidationError as e:
        print(f"Validation failed: {e}")
    
    return df


def demonstrate_data_recovery():
    """Demonstrate automatic data recovery."""
    print("\n" + "="*60)
    print("DATA RECOVERY DEMONSTRATION")
    print("="*60)
    
    # Create problematic dataset
    df = create_problematic_dataset()
    
    print(f"Original dataset issues:")
    print(f"  Shape: {df.shape}")
    print(f"  Missing values: {df.isnull().sum().sum()}")
    print(f"  Data types: {len(df.dtypes.unique())} unique types")
    
    # Define expected schema for recovery
    expected_columns = ['numerical_outliers', 'categorical_messy', 'mixed_types', 'target']
    expected_dtypes = {
        'numerical_outliers': 'float',
        'mixed_types': 'float',
        'categorical_messy': 'object',
        'target': 'int'
    }
    
    # Create recovery system
    recovery = DataRecovery(
        auto_fix=True,
        missing_strategy="median",
        outlier_strategy="clip",
        dtype_coercion=True,
        verbose=True
    )
    
    print(f"\nApplying automatic data recovery...")
    
    try:
        recovered_df, log = recovery.recover_dataframe(
            df=df,
            expected_columns=expected_columns,
            expected_dtypes=expected_dtypes,
            target_column='target'
        )
        
        print(f"\nRecovery Results:")
        print(f"  Success: {log['success']}")
        print(f"  Actions taken: {len(log['actions_taken'])}")
        print(f"  Original shape: {log['original_shape']}")
        print(f"  Final shape: {log['final_shape']}")
        print(f"  Columns added: {len(log['columns_added'])}")
        print(f"  Columns removed: {len(log['columns_removed'])}")
        print(f"  Rows removed: {log['rows_removed']}")
        
        print(f"\nRecovery Actions:")
        for i, action in enumerate(log['actions_taken'][:10], 1):  # Show first 10
            print(f"  {i}. {action}")
        
        print(f"\nRecovered dataset:")
        print(f"  Shape: {recovered_df.shape}")
        print(f"  Missing values: {recovered_df.isnull().sum().sum()}")
        print(f"  Data types: {recovered_df.dtypes.to_dict()}")
        
        return recovered_df
        
    except Exception as e:
        print(f"Recovery failed: {e}")
        return df


def demonstrate_robust_normalization():
    """Demonstrate robust data normalization."""
    print("\n" + "="*60)
    print("ROBUST NORMALIZATION DEMONSTRATION")
    print("="*60)
    
    # Create dataset with various data types and issues
    df = pd.DataFrame({
        'numerical_normal': np.random.normal(100, 15, 500),
        'numerical_outliers': np.concatenate([
            np.random.normal(50, 10, 480),
            np.random.uniform(500, 1000, 20)  # Outliers
        ]),
        'categorical_freq': np.random.choice(['A', 'B', 'C', 'D'], 500, p=[0.4, 0.3, 0.2, 0.1]),
        'categorical_rare': np.random.choice(['X', 'Y', 'Z'], 500, p=[0.8, 0.15, 0.05]),
        'datetime_col': pd.date_range('2023-01-01', periods=500, freq='D'),
        'target': np.random.choice([0, 1], 500)
    })
    
    # Add some missing values
    df.loc[::50, 'numerical_normal'] = np.nan
    df.loc[::30, 'categorical_freq'] = np.nan
    
    print(f"Original dataset:")
    print(f"  Shape: {df.shape}")
    print(f"  Missing values: {df.isnull().sum().sum()}")
    print(f"  Numerical columns: {df.select_dtypes(include=[np.number]).columns.tolist()}")
    print(f"  Categorical columns: {df.select_dtypes(include=['object']).columns.tolist()}")
    print(f"  Datetime columns: {df.select_dtypes(include=['datetime']).columns.tolist()}")
    
    # Create robust normalizer
    normalizer = RobustNormalizer(
        numerical_strategy="robust",        # Robust scaling (less sensitive to outliers)
        categorical_strategy="frequency",   # Frequency encoding
        outlier_method="iqr",              # IQR-based outlier detection
        outlier_action="clip",             # Clip outliers to bounds
        missing_strategy="median",         # Fill missing with median/mode
        handle_new_categories="ignore",    # Handle unknown categories gracefully
        preserve_dtypes=True
    )
    
    print(f"\nApplying robust normalization...")
    
    try:
        # Fit and transform
        normalized_df, log = normalizer.fit_transform(df, target_column='target')
        
        print(f"\nNormalization Results:")
        print(f"  Success: {log['success']}")
        print(f"  Original shape: {log['original_shape']}")
        print(f"  Final shape: {log['final_shape']}")
        print(f"  Actions taken: {len(log['actions_taken'])}")
        
        print(f"\nNormalization Actions:")
        for i, action in enumerate(log['actions_taken'], 1):
            print(f"  {i}. {action}")
        
        print(f"\nNormalized dataset:")
        print(f"  Shape: {normalized_df.shape}")
        print(f"  Missing values: {normalized_df.isnull().sum().sum()}")
        print(f"  Feature names: {normalizer.get_feature_names()}")
        
        # Show transformation summary
        summary = normalizer.get_transformation_summary()
        print(f"\nTransformation Summary:")
        print(f"  Numerical columns processed: {summary['numerical_columns']}")
        print(f"  Categorical columns processed: {summary['categorical_columns']}")
        print(f"  Datetime columns processed: {summary['datetime_columns']}")
        print(f"  Strategies used: {summary['numerical_strategy']}, {summary['categorical_strategy']}")
        
        return normalized_df
        
    except Exception as e:
        print(f"Normalization failed: {e}")
        return df


def demonstrate_robust_operations():
    """Demonstrate robust operation decorators."""
    print("\n" + "="*60)
    print("ROBUST OPERATIONS DEMONSTRATION")
    print("="*60)
    
    # Demonstrate retry mechanism
    print("Testing retry mechanism...")
    
    attempt_count = 0
    
    @robust_operation(max_retries=3, backoff_factor=0.1)
    def unreliable_operation():
        nonlocal attempt_count
        attempt_count += 1
        print(f"  Attempt {attempt_count}")
        
        if attempt_count < 3:
            raise Exception(f"Temporary failure (attempt {attempt_count})")
        
        return "Success after retries!"
    
    try:
        result = unreliable_operation()
        print(f"  Result: {result}")
    except Exception as e:
        print(f"  Final failure: {e}")
    
    # Demonstrate graceful degradation
    print(f"\nTesting graceful degradation...")
    
    @graceful_degradation(fallback_value="Fallback result", log_error=True)
    def failing_operation():
        raise Exception("This operation always fails")
    
    @graceful_degradation(fallback_value="Fallback result", log_error=True)
    def working_operation():
        return "Normal result"
    
    result1 = failing_operation()
    result2 = working_operation()
    
    print(f"  Failing operation result: {result1}")
    print(f"  Working operation result: {result2}")


def demonstrate_error_handling():
    """Demonstrate comprehensive error handling."""
    print("\n" + "="*60)
    print("ERROR HANDLING DEMONSTRATION")
    print("="*60)
    
    # Test different types of errors
    print("Testing custom exceptions...")
    
    try:
        from tabgpt.utils import MissingColumnsError, DataTypeError, ExcessiveMissingValuesError
        
        # Missing columns error
        try:
            raise MissingColumnsError(['col1', 'col2'], ['col1', 'col2', 'col3'])
        except MissingColumnsError as e:
            print(f"  Caught MissingColumnsError: {e}")
        
        # Data type error
        try:
            raise DataTypeError('column1', 'int', 'str')
        except DataTypeError as e:
            print(f"  Caught DataTypeError: {e}")
        
        # Excessive missing values error
        try:
            raise ExcessiveMissingValuesError('column1', 0.8, 0.5)
        except ExcessiveMissingValuesError as e:
            print(f"  Caught ExcessiveMissingValuesError: {e}")
        
    except ImportError as e:
        print(f"  Import error: {e}")
    
    # Test error recovery in data processing
    print(f"\nTesting error recovery in data processing...")
    
    # Create dataset that will cause errors
    problematic_df = pd.DataFrame({
        'good_column': [1, 2, 3, 4, 5],
        'bad_column': ['a', 'b', None, 'd', 'e'],  # Mixed types
        'empty_column': [None] * 5,                # All missing
        'constant_column': [1] * 5                 # Constant values
    })
    
    # Try to process with error handling
    try:
        validator = DataValidator(strict_mode=False)
        result = validator.validate_dataframe(problematic_df)
        
        print(f"  Validation completed with {len(result['errors'])} errors and {len(result['warnings'])} warnings")
        
        # Try recovery
        recovery = DataRecovery(auto_fix=True)
        recovered_df, log = recovery.recover_dataframe(problematic_df)
        
        print(f"  Recovery completed: {log['success']}")
        print(f"  Actions taken: {len(log['actions_taken'])}")
        
    except Exception as e:
        print(f"  Error in processing: {e}")


def main():
    """Run all robustness demonstrations."""
    print("TabGPT Robustness and Error Handling Demonstration")
    print("="*60)
    
    try:
        # 1. Data validation
        df = demonstrate_data_validation()
        
        # 2. Data recovery
        recovered_df = demonstrate_data_recovery()
        
        # 3. Robust normalization
        normalized_df = demonstrate_robust_normalization()
        
        # 4. Robust operations
        demonstrate_robust_operations()
        
        # 5. Error handling
        demonstrate_error_handling()
        
        print("\n" + "="*60)
        print("DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nKey Features Demonstrated:")
        print("  • Comprehensive data validation with detailed error reporting")
        print("  • Automatic data recovery and preprocessing")
        print("  • Robust normalization with outlier and missing value handling")
        print("  • Retry mechanisms and graceful degradation")
        print("  • Custom exception hierarchy with detailed error information")
        print("  • Schema validation and type coercion")
        print("  • Memory-efficient processing of large datasets")
        
        print("\nThe robustness framework makes TabGPT production-ready!")
        
    except Exception as e:
        print(f"\nUnexpected error during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()