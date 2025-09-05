"""Simple integration test for TabGPT data pipeline."""

import pandas as pd
import numpy as np
import torch

from tabgpt.data import (
    TabularDataLoader, CSVLoader, TabularPreprocessor, TabularDataset,
    create_data_splits, DataSplit
)
from tabgpt.data.preprocessing import PreprocessingConfig, MissingValueStrategy
from tabgpt.data.transforms import create_train_transforms
from tabgpt.data.utils import infer_data_types, compute_statistics
from tabgpt.tokenizers import TabularTokenizer


def test_simple_data_pipeline():
    """Test core data pipeline functionality."""
    
    print("=== Simple TabGPT Data Pipeline Test ===\n")
    
    # Step 1: Create test data
    print("Step 1: Creating Test Data")
    np.random.seed(42)
    
    df = pd.DataFrame({
        'numerical_1': np.random.normal(100, 15, 200),
        'numerical_2': np.random.exponential(2, 200),
        'categorical_1': np.random.choice(['A', 'B', 'C'], 200),
        'categorical_2': np.random.choice(['X', 'Y', 'Z'], 200, p=[0.5, 0.3, 0.2]),
        'boolean_1': np.random.choice([True, False], 200),
        'target': np.random.choice(['Class1', 'Class2', 'Class3'], 200)
    })
    
    # Add some missing values
    missing_idx = np.random.choice(200, 20, replace=False)
    df.loc[missing_idx, 'numerical_1'] = np.nan
    
    print(f"Dataset shape: {df.shape}")
    print(f"Missing values: {df.isnull().sum().sum()}")
    print(f"Columns: {list(df.columns)}")
    
    # Step 2: Data analysis
    print("\nStep 2: Data Analysis")
    data_types = infer_data_types(df)
    print(f"Inferred types: {[(k, v.value) for k, v in data_types.items()]}")
    
    stats = compute_statistics(df, include_correlations=False)
    print(f"Memory usage: {stats['memory_usage'] / 1024:.2f} KB")
    
    # Step 3: Preprocessing
    print("\nStep 3: Preprocessing")
    config = PreprocessingConfig(
        handle_missing=True,
        missing_strategy=MissingValueStrategy.MEDIAN,
        normalize_numerical=True,
        remove_duplicates=True
    )
    
    preprocessor = TabularPreprocessor(config)
    df_processed = preprocessor.fit_transform(df)
    
    print(f"Processed shape: {df_processed.shape}")
    print(f"Missing values after: {df_processed.isnull().sum().sum()}")
    
    # Step 4: Tokenization
    print("\nStep 4: Tokenization")
    tokenizer = TabularTokenizer(embedding_dim=64)
    
    # Separate features and target
    feature_cols = [col for col in df_processed.columns if col != 'target']
    df_features = df_processed[feature_cols]
    
    tokenized = tokenizer.fit_transform(df_features)
    
    print(f"Tokenized shape: {tokenized.tokens.shape}")
    print(f"Feature names: {tokenized.feature_names}")
    
    # Step 5: Dataset creation
    print("\nStep 5: Dataset Creation")
    dataset = TabularDataset(
        df_processed,
        tokenizer=tokenizer,
        target_column='target',
        cache_tokenized=True
    )
    
    print(f"Dataset length: {len(dataset)}")
    
    # Test sample
    sample = dataset[0]
    print(f"Sample keys: {list(sample.keys())}")
    print(f"Input shape: {sample['input_features'].shape}")
    print(f"Has target: {'target' in sample}")
    
    # Step 6: Data splits
    print("\nStep 6: Data Splits")
    train_dataset, val_dataset, test_dataset = create_data_splits(
        dataset,
        DataSplit(train_ratio=0.7, val_ratio=0.2, test_ratio=0.1)
    )
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # Step 7: Transforms
    print("\nStep 7: Data Transforms")
    transforms = create_train_transforms("light")
    
    sample_transformed = transforms(sample.copy())
    
    original_norm = torch.norm(sample['input_features']).item()
    transformed_norm = torch.norm(sample_transformed['input_features']).item()
    
    print(f"Original norm: {original_norm:.4f}")
    print(f"Transformed norm: {transformed_norm:.4f}")
    
    # Step 8: Validation
    print("\nStep 8: Validation")
    
    # Check data integrity
    assert len(dataset) > 0, "Dataset should not be empty"
    assert sample['input_features'].shape[0] > 0, "Should have features"
    assert 'target' in sample, "Should have target"
    assert len(train_dataset) + len(val_dataset) + len(test_dataset) == len(dataset), "Split sizes should match"
    
    print("✅ All validations passed!")
    
    # Step 9: Performance summary
    print("\nStep 9: Performance Summary")
    
    results = {
        'original_shape': df.shape,
        'processed_shape': df_processed.shape,
        'tokenized_shape': tokenized.tokens.shape,
        'dataset_length': len(dataset),
        'train_length': len(train_dataset),
        'val_length': len(val_dataset),
        'test_length': len(test_dataset),
        'feature_count': len(feature_cols),
        'memory_usage_kb': stats['memory_usage'] / 1024
    }
    
    for key, value in results.items():
        print(f"{key}: {value}")
    
    print("\n✅ Simple Data Pipeline Test Completed Successfully!")
    
    return results


if __name__ == "__main__":
    results = test_simple_data_pipeline()
    print(f"\nFinal Results: {results}")