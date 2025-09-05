"""Integration test for TabGPT data loading and preprocessing pipeline."""

import pandas as pd
import numpy as np
import torch
from pathlib import Path
import tempfile
import os

from tabgpt.data import (
    TabularDataLoader, CSVLoader, TabularPreprocessor, TabularDataset,
    StreamingTabularDataset, CachedTabularDataset, TemporalTabularDataset,
    create_data_splits, create_dataloader, DataSplit
)
from tabgpt.data.preprocessing import PreprocessingConfig, MissingValueStrategy, OutlierMethod
from tabgpt.data.transforms import (
    create_train_transforms, create_val_transforms, AugmentationTransform
)
from tabgpt.data.utils import (
    infer_data_types, compute_statistics, detect_outliers, 
    sample_dataset, split_dataset, profile_dataset
)
from tabgpt.tokenizers import TabularTokenizer


def test_data_pipeline_integration():
    """Test complete data pipeline with various scenarios."""
    
    print("=== TabGPT Data Pipeline Integration Test ===\n")
    
    # Step 1: Create comprehensive test datasets
    print("Step 1: Creating Test Datasets")
    
    # Dataset 1: Mixed data types with missing values
    np.random.seed(42)
    n_samples = 1000
    
    df_mixed = pd.DataFrame({
        'customer_id': range(1, n_samples + 1),
        'age': np.random.randint(18, 80, n_samples),
        'income': np.random.lognormal(10, 1, n_samples),
        'credit_score': np.random.normal(700, 100, n_samples),
        'account_balance': np.random.exponential(5000, n_samples),
        'product_type': np.random.choice(['Premium', 'Standard', 'Basic'], n_samples),
        'is_active': np.random.choice([True, False], n_samples),
        'signup_date': pd.date_range('2020-01-01', periods=n_samples, freq='D'),
        'last_transaction': np.random.uniform(0, 365, n_samples),  # days ago
        'risk_category': np.random.choice(['Low', 'Medium', 'High'], n_samples, p=[0.6, 0.3, 0.1])
    })
    
    # Introduce missing values
    missing_indices = np.random.choice(n_samples, size=int(n_samples * 0.1), replace=False)
    df_mixed.loc[missing_indices, 'income'] = np.nan
    
    missing_indices = np.random.choice(n_samples, size=int(n_samples * 0.05), replace=False)
    df_mixed.loc[missing_indices, 'credit_score'] = np.nan
    
    print(f"Mixed dataset shape: {df_mixed.shape}")
    print(f"Missing values: {df_mixed.isnull().sum().sum()}")
    print(f"Data types: {df_mixed.dtypes.to_dict()}")
    
    # Dataset 2: Temporal data
    df_temporal = pd.DataFrame({
        'timestamp': pd.date_range('2023-01-01', periods=500, freq='H'),
        'sensor_1': np.random.normal(25, 5, 500) + np.sin(np.arange(500) * 0.1) * 3,
        'sensor_2': np.random.normal(50, 10, 500) + np.cos(np.arange(500) * 0.05) * 5,
        'sensor_3': np.random.exponential(2, 500),
        'status': np.random.choice(['Normal', 'Warning', 'Alert'], 500, p=[0.8, 0.15, 0.05]),
        'machine_id': np.random.choice(['M001', 'M002', 'M003'], 500)
    })
    
    print(f"Temporal dataset shape: {df_temporal.shape}")
    
    # Step 2: Test data type inference and statistics
    print("\nStep 2: Data Analysis and Profiling")
    
    data_types = infer_data_types(df_mixed)
    print(f"Inferred data types: {data_types}")
    
    statistics = compute_statistics(df_mixed)
    print(f"Dataset statistics computed: {len(statistics)} categories")
    print(f"Memory usage: {statistics['memory_usage'] / 1024 / 1024:.2f} MB")
    
    outliers = detect_outliers(df_mixed)
    print(f"Outliers detected in {len(outliers)} columns")
    
    # Step 3: Test preprocessing pipeline
    print("\nStep 3: Testing Preprocessing Pipeline")
    
    # Basic preprocessing
    config_basic = PreprocessingConfig(
        handle_missing=True,
        missing_strategy=MissingValueStrategy.MEDIAN,
        handle_outliers=True,
        outlier_method=OutlierMethod.IQR,
        normalize_numerical=True,
        remove_duplicates=True
    )
    
    preprocessor_basic = TabularPreprocessor(config_basic)
    df_processed_basic = preprocessor_basic.fit_transform(df_mixed)
    
    print(f"Basic preprocessing: {df_mixed.shape} -> {df_processed_basic.shape}")
    print(f"Missing values after preprocessing: {df_processed_basic.isnull().sum().sum()}")
    
    # Advanced preprocessing
    config_advanced = PreprocessingConfig(
        handle_missing=True,
        missing_strategy=MissingValueStrategy.KNN,
        handle_outliers=True,
        outlier_method=OutlierMethod.ISOLATION_FOREST,
        normalize_numerical=True,
        normalization_method="robust",
        remove_constant_columns=True
    )
    
    preprocessor_advanced = TabularPreprocessor(config_advanced)
    df_processed_advanced = preprocessor_advanced.fit_transform(df_mixed.copy())
    
    print(f"Advanced preprocessing: {df_mixed.shape} -> {df_processed_advanced.shape}")
    
    # Step 4: Test tokenization
    print("\nStep 4: Testing Tokenization")
    
    tokenizer = TabularTokenizer(embedding_dim=128)
    tokenized_basic = tokenizer.fit_transform(df_processed_basic)
    
    print(f"Tokenized shape: {tokenized_basic.tokens.shape}")
    print(f"Attention mask shape: {tokenized_basic.attention_mask.shape}")
    print(f"Feature names: {tokenized_basic.feature_names[:5]}...")
    
    # Step 5: Test dataset classes
    print("\nStep 5: Testing Dataset Classes")
    
    # Basic dataset
    dataset_basic = TabularDataset(
        df_processed_basic,
        tokenizer=tokenizer,
        preprocessor=preprocessor_basic,
        target_column='risk_category'
    )
    
    print(f"Basic dataset length: {len(dataset_basic)}")
    
    sample = dataset_basic[0]
    print(f"Sample keys: {list(sample.keys())}")
    print(f"Input features shape: {sample['input_features'].shape}")
    print(f"Has target: {'target' in sample}")
    
    # Test data splits
    train_dataset, val_dataset, test_dataset = create_data_splits(
        dataset_basic,
        DataSplit(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
    )
    
    print(f"Data splits - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # Step 6: Test temporal dataset
    print("\nStep 6: Testing Temporal Dataset")
    
    temporal_dataset = TemporalTabularDataset(
        df_temporal,
        timestamp_column='timestamp',
        sequence_length=24,  # 24 hours
        prediction_horizon=1,
        group_by_columns=['machine_id'],
        tokenizer=tokenizer,
        target_column='sensor_1'
    )
    
    print(f"Temporal dataset length: {len(temporal_dataset)}")
    
    if len(temporal_dataset) > 0:
        temporal_sample = temporal_dataset[0]
        print(f"Temporal sample keys: {list(temporal_sample.keys())}")
        print(f"Sequence shape: {temporal_sample['input_features'].shape}")
        print(f"Sequence length: {temporal_sample['sequence_length']}")
    
    # Step 7: Test caching
    print("\nStep 7: Testing Dataset Caching")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        cached_dataset = CachedTabularDataset(
            df_processed_basic,
            cache_dir=temp_dir,
            dataset_name="test_dataset",
            tokenizer=tokenizer,
            preprocessor=preprocessor_basic,
            target_column='risk_category'
        )
        
        print(f"Cached dataset length: {len(cached_dataset)}")
        
        # Test loading from cache
        cached_dataset_2 = CachedTabularDataset(
            df_processed_basic,  # This should be ignored due to cache
            cache_dir=temp_dir,
            dataset_name="test_dataset",
            tokenizer=tokenizer
        )
        
        print(f"Loaded from cache: {len(cached_dataset_2)}")
    
    # Step 8: Test data loaders
    print("\nStep 8: Testing Data Loaders")
    
    # Create temporary CSV file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        df_mixed.to_csv(f.name, index=False)
        csv_file = f.name
    
    try:
        # Test CSV loader
        csv_loader = CSVLoader()
        df_loaded, dataset_info = csv_loader.load(csv_file)
        
        print(f"CSV loaded: {df_loaded.shape}")
        print(f"Dataset info: {dataset_info.name}, {dataset_info.source}")
        
        # Test unified loader
        unified_loader = TabularDataLoader()
        df_auto, info_auto = unified_loader.load_auto(csv_file)
        
        print(f"Auto-loaded: {df_auto.shape}")
        
    finally:
        os.unlink(csv_file)
    
    # Step 9: Test transforms
    print("\nStep 9: Testing Data Transforms")
    
    # Create transforms
    train_transforms = create_train_transforms("medium")
    val_transforms = create_val_transforms()
    
    # Apply to sample
    sample_transformed = train_transforms(sample.copy())
    print(f"Transformed sample keys: {list(sample_transformed.keys())}")
    
    original_features = sample['input_features']
    transformed_features = sample_transformed['input_features']
    
    transform_diff = torch.norm(transformed_features - original_features).item()
    print(f"Transform difference norm: {transform_diff:.4f}")
    
    # Step 10: Test PyTorch DataLoader
    print("\nStep 10: Testing PyTorch DataLoader")
    
    dataloader = create_dataloader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0
    )
    
    print(f"DataLoader created with batch size 32")
    
    # Test one batch
    for batch in dataloader:
        print(f"Batch keys: {list(batch.keys())}")
        print(f"Batch input features shape: {batch['input_features'].shape}")
        if 'target' in batch:
            print(f"Batch targets shape: {batch['target'].shape}")
        else:
            print("No targets in batch (unsupervised mode)")
        break
    
    # Step 11: Test streaming dataset
    print("\nStep 11: Testing Streaming Dataset")
    
    # Create larger CSV for streaming test
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        # Create larger dataset
        df_large = pd.concat([df_mixed] * 5, ignore_index=True)  # 5x larger
        df_large.to_csv(f.name, index=False)
        large_csv_file = f.name
    
    try:
        streaming_dataset = StreamingTabularDataset(
            large_csv_file,
            tokenizer=tokenizer,
            preprocessor=preprocessor_basic,
            chunk_size=100,
            target_column='risk_category'
        )
        
        print(f"Streaming dataset created")
        print(f"Estimated rows: {streaming_dataset.estimated_n_rows}")
        
        # Test iteration (just a few samples)
        sample_count = 0
        for sample in streaming_dataset:
            sample_count += 1
            if sample_count >= 5:  # Just test first 5 samples
                break
        
        print(f"Streamed {sample_count} samples successfully")
        
    finally:
        os.unlink(large_csv_file)
    
    # Step 12: Test utilities
    print("\nStep 12: Testing Utility Functions")
    
    # Test sampling
    sampled_df = sample_dataset(df_mixed, fraction=0.1, random_state=42)
    print(f"Sampled dataset: {df_mixed.shape} -> {sampled_df.shape}")
    
    # Test splitting
    train_df, val_df, test_df = split_dataset(
        df_mixed,
        train_size=0.7,
        val_size=0.15,
        test_size=0.15,
        stratify_column='risk_category',
        random_state=42
    )
    
    print(f"Split dataset: Train {train_df.shape}, Val {val_df.shape}, Test {test_df.shape}")
    
    # Test profiling
    with tempfile.TemporaryDirectory() as temp_dir:
        profile_file = Path(temp_dir) / "profile.json"
        profile = profile_dataset(df_mixed, output_file=str(profile_file))
        
        print(f"Dataset profile created with {len(profile)} sections")
        print(f"Data quality completeness: {profile['data_quality']['completeness']:.2f}%")
        print(f"Profile saved to: {profile_file.exists()}")
    
    # Step 13: Performance summary
    print("\nStep 13: Performance Summary")
    
    print(f"Original dataset: {df_mixed.shape}")
    print(f"Processed dataset: {df_processed_basic.shape}")
    print(f"Tokenized shape: {tokenized_basic.tokens.shape}")
    print(f"Dataset samples: {len(dataset_basic)}")
    print(f"Memory usage: {statistics['memory_usage'] / 1024 / 1024:.2f} MB")
    
    # Verify key functionality
    assert len(dataset_basic) > 0, "Dataset should have samples"
    assert sample['input_features'].shape[0] > 0, "Should have features"
    assert 'target' in sample, "Should have target"
    assert len(train_dataset) + len(val_dataset) + len(test_dataset) == len(dataset_basic), "Splits should sum to total"
    
    print("\n✅ Data Pipeline Integration Test Completed Successfully!")
    
    return {
        'original_shape': df_mixed.shape,
        'processed_shape': df_processed_basic.shape,
        'tokenized_shape': tokenized_basic.tokens.shape,
        'dataset_length': len(dataset_basic),
        'train_length': len(train_dataset),
        'val_length': len(val_dataset),
        'test_length': len(test_dataset),
        'memory_usage_mb': statistics['memory_usage'] / 1024 / 1024,
        'data_quality_completeness': profile['data_quality']['completeness']
    }


if __name__ == "__main__":
    results = test_data_pipeline_integration()
    print(f"\nFinal Results: {results}")