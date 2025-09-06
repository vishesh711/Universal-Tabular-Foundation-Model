"""Tests for error handling and robustness features."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
import warnings

from tabgpt.utils.exceptions import (
    TabGPTError, DataQualityError, ValidationError, MissingColumnsError,
    ExtraColumnsError, DataTypeError, EmptyDataError, ExcessiveMissingValuesError,
    OutlierError, SchemaMismatchError, VocabularyError, ModelNotTrainedError
)
from tabgpt.utils.validation import DataValidator, ConfigValidator, validate_input_data
from tabgpt.utils.recovery import DataRecovery, ModelRecovery, robust_operation, graceful_degradation
from tabgpt.utils.normalization import RobustNormalizer


class TestExceptions:
    """Test custom exception classes."""
    
    def test_base_exception(self):
        """Test base TabGPTError exception."""
        error = TabGPTError("Test message", "TEST_CODE", {"key": "value"})
        
        assert str(error) == "[TEST_CODE] Test message (Details: key=value)"
        assert error.message == "Test message"
        assert error.error_code == "TEST_CODE"
        assert error.details == {"key": "value"}
    
    def test_missing_columns_error(self):
        """Test MissingColumnsError."""
        missing = ["col1", "col2"]
        expected = ["col1", "col2", "col3"]
        
        error = MissingColumnsError(missing, expected)
        
        assert error.missing_columns == missing
        assert error.expected_columns == expected
        assert "Missing columns: ['col1', 'col2']" in str(error)
    
    def test_data_type_error(self):
        """Test DataTypeError."""
        error = DataTypeError("column1", "int", "str")
        
        assert error.column == "column1"
        assert error.expected_type == "int"
        assert error.actual_type == "str"
        assert "Column 'column1' has type 'str', expected 'int'" in str(error)
    
    def test_empty_data_error(self):
        """Test EmptyDataError."""
        error = EmptyDataError("test_dataset")
        
        assert error.dataset_name == "test_dataset"
        assert "Dataset 'test_dataset' is empty" in str(error)


class TestDataValidator:
    """Test data validation functionality."""
    
    def test_basic_validation_success(self):
        """Test successful basic validation."""
        df = pd.DataFrame({
            'col1': [1, 2, 3, 4, 5],
            'col2': ['a', 'b', 'c', 'd', 'e'],
            'col3': [1.1, 2.2, 3.3, 4.4, 5.5]
        })
        
        validator = DataValidator(min_samples=3, min_features=2)
        result = validator.validate_dataframe(df)
        
        assert result['valid'] is True
        assert len(result['errors']) == 0
        assert 'stats' in result
        assert result['stats']['n_samples'] == 5
        assert result['stats']['n_features'] == 3
    
    def test_empty_dataframe_validation(self):
        """Test validation of empty DataFrame."""
        df = pd.DataFrame()
        
        validator = DataValidator(strict_mode=False)
        result = validator.validate_dataframe(df)
        
        assert result['valid'] is True  # Not strict mode
        assert len(result['errors']) > 0
        assert any("empty" in error.lower() for error in result['errors'])
    
    def test_missing_columns_validation(self):
        """Test validation with missing columns."""
        df = pd.DataFrame({'col1': [1, 2, 3]})
        expected_columns = ['col1', 'col2', 'col3']
        
        validator = DataValidator(strict_mode=False)
        result = validator.validate_dataframe(df, expected_columns=expected_columns)
        
        assert len(result['errors']) > 0
        assert any("missing" in error.lower() for error in result['errors'])
    
    def test_excessive_missing_values(self):
        """Test validation with excessive missing values."""
        df = pd.DataFrame({
            'col1': [1, 2, np.nan, np.nan, np.nan],
            'col2': ['a', 'b', 'c', 'd', 'e']
        })
        
        validator = DataValidator(missing_threshold=0.3, strict_mode=False)
        result = validator.validate_dataframe(df)
        
        assert len(result['errors']) > 0
        assert any("missing values" in error for error in result['errors'])
    
    def test_outlier_detection(self):
        """Test outlier detection in validation."""
        # Create data with obvious outliers
        df = pd.DataFrame({
            'normal_col': [1, 2, 3, 4, 5],
            'outlier_col': [1, 2, 3, 4, 1000]  # 1000 is an outlier
        })
        
        validator = DataValidator(outlier_threshold=0.1, strict_mode=False)
        result = validator.validate_dataframe(df)
        
        # Should detect outliers in outlier_col
        outlier_warnings = [w for w in result['warnings'] if 'outlier' in w.lower()]
        assert len(outlier_warnings) > 0
    
    def test_strict_mode_validation(self):
        """Test validation in strict mode."""
        df = pd.DataFrame({'col1': [1, 2, 3]})
        expected_columns = ['col1', 'col2']
        
        validator = DataValidator(strict_mode=True)
        
        with pytest.raises(ValidationError):
            validator.validate_dataframe(df, expected_columns=expected_columns)


class TestDataRecovery:
    """Test data recovery functionality."""
    
    def test_missing_columns_recovery(self):
        """Test recovery of missing columns."""
        df = pd.DataFrame({'col1': [1, 2, 3]})
        expected_columns = ['col1', 'col2', 'col3']
        
        recovery = DataRecovery(auto_fix=True)
        recovered_df, log = recovery.recover_dataframe(df, expected_columns=expected_columns)
        
        assert 'col2' in recovered_df.columns
        assert 'col3' in recovered_df.columns
        assert len(log['columns_added']) == 2
        assert log['success'] is True
    
    def test_extra_columns_recovery(self):
        """Test recovery from extra columns."""
        df = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': [4, 5, 6],
            'extra_col': [7, 8, 9]
        })
        expected_columns = ['col1', 'col2']
        
        recovery = DataRecovery(auto_fix=True)
        recovered_df, log = recovery.recover_dataframe(df, expected_columns=expected_columns)
        
        assert 'extra_col' not in recovered_df.columns
        assert len(log['columns_removed']) == 1
        assert log['success'] is True
    
    def test_missing_values_recovery(self):
        """Test recovery of missing values."""
        df = pd.DataFrame({
            'numerical': [1, 2, np.nan, 4, 5],
            'categorical': ['a', 'b', np.nan, 'd', 'e']
        })
        
        recovery = DataRecovery(auto_fix=True, missing_strategy="median")
        recovered_df, log = recovery.recover_dataframe(df)
        
        assert recovered_df['numerical'].isnull().sum() == 0
        assert recovered_df['categorical'].isnull().sum() == 0
        assert log['success'] is True
    
    def test_outlier_recovery(self):
        """Test outlier recovery."""
        df = pd.DataFrame({
            'normal_data': [1, 2, 3, 4, 5],
            'with_outliers': [1, 2, 3, 4, 1000]
        })
        
        recovery = DataRecovery(auto_fix=True, outlier_strategy="clip")
        recovered_df, log = recovery.recover_dataframe(df)
        
        # Outlier should be clipped
        assert recovered_df['with_outliers'].max() < 1000
        assert log['success'] is True
    
    def test_dtype_conversion_recovery(self):
        """Test data type conversion recovery."""
        df = pd.DataFrame({
            'should_be_int': ['1', '2', '3', '4', '5'],
            'should_be_float': ['1.1', '2.2', '3.3', '4.4', '5.5']
        })
        expected_dtypes = {
            'should_be_int': 'int',
            'should_be_float': 'float'
        }
        
        recovery = DataRecovery(auto_fix=True, dtype_coercion=True)
        recovered_df, log = recovery.recover_dataframe(df, expected_dtypes=expected_dtypes)
        
        assert pd.api.types.is_integer_dtype(recovered_df['should_be_int'])
        assert pd.api.types.is_float_dtype(recovered_df['should_be_float'])
        assert log['success'] is True


class TestModelRecovery:
    """Test model recovery functionality."""
    
    def test_safe_predict_success(self):
        """Test successful prediction with primary model."""
        # Mock successful model
        model = Mock()
        model.predict.return_value = np.array([0, 1, 0, 1])
        
        X = pd.DataFrame({'feature1': [1, 2, 3, 4], 'feature2': [5, 6, 7, 8]})
        
        recovery = ModelRecovery()
        predictions, info = recovery.safe_predict(model, X)
        
        assert len(predictions) == 4
        assert info['primary_model_used'] is True
        assert info['fallback_used'] is False
    
    def test_safe_predict_with_fallback(self):
        """Test prediction with fallback when primary model fails."""
        # Mock failing primary model
        primary_model = Mock()
        primary_model.predict.side_effect = Exception("Primary model failed")
        
        # Mock successful fallback model
        fallback_model = Mock()
        fallback_model.predict.return_value = np.array([0, 1, 0, 1])
        fallback_model._is_fitted = True
        
        X = pd.DataFrame({'feature1': [1, 2, 3, 4], 'feature2': [5, 6, 7, 8]})
        
        recovery = ModelRecovery(enable_fallbacks=True)
        recovery.register_fallback_model("classification", fallback_model)
        
        predictions, info = recovery.safe_predict(primary_model, X, "classification")
        
        assert len(predictions) == 4
        assert info['primary_model_used'] is False
        assert info['fallback_used'] is True
        assert len(info['errors']) == 1
    
    def test_safe_predict_dummy_fallback(self):
        """Test prediction with dummy fallback when all models fail."""
        # Mock failing primary model
        primary_model = Mock()
        primary_model.predict.side_effect = Exception("Primary model failed")
        
        X = pd.DataFrame({'feature1': [1, 2, 3, 4], 'feature2': [5, 6, 7, 8]})
        
        recovery = ModelRecovery(enable_fallbacks=True)
        # No fallback model registered
        
        predictions, info = recovery.safe_predict(primary_model, X, "classification")
        
        assert len(predictions) == 4
        assert info['primary_model_used'] is False
        assert info['fallback_used'] is False
        assert len(info['warnings']) > 0


class TestRobustNormalizer:
    """Test robust normalization functionality."""
    
    def test_basic_normalization(self):
        """Test basic normalization functionality."""
        df = pd.DataFrame({
            'numerical': [1, 2, 3, 4, 5],
            'categorical': ['a', 'b', 'c', 'a', 'b'],
            'target': [0, 1, 0, 1, 0]
        })
        
        normalizer = RobustNormalizer()
        normalizer.fit(df, target_column='target')
        
        transformed_df, log = normalizer.transform(df)
        
        assert log['success'] is True
        assert 'numerical' in transformed_df.columns
        assert 'categorical' in transformed_df.columns
        assert normalizer.is_fitted
    
    def test_missing_values_normalization(self):
        """Test normalization with missing values."""
        df = pd.DataFrame({
            'numerical': [1, 2, np.nan, 4, 5],
            'categorical': ['a', 'b', np.nan, 'a', 'b']
        })
        
        normalizer = RobustNormalizer(missing_strategy="median")
        transformed_df, log = normalizer.fit_transform(df)
        
        assert transformed_df['numerical'].isnull().sum() == 0
        assert transformed_df['categorical'].isnull().sum() == 0
        assert log['success'] is True
    
    def test_outlier_handling_normalization(self):
        """Test normalization with outlier handling."""
        df = pd.DataFrame({
            'with_outliers': [1, 2, 3, 4, 1000],
            'normal': [1, 2, 3, 4, 5]
        })
        
        normalizer = RobustNormalizer(outlier_action="clip")
        transformed_df, log = normalizer.fit_transform(df)
        
        # Outlier should be handled
        assert transformed_df['with_outliers'].max() < 1000
        assert log['success'] is True
    
    def test_categorical_encoding(self):
        """Test categorical feature encoding."""
        df = pd.DataFrame({
            'category': ['a', 'b', 'c', 'a', 'b', 'c'],
            'target': [1, 2, 3, 1, 2, 3]
        })
        
        normalizer = RobustNormalizer(categorical_strategy="frequency")
        transformed_df, log = normalizer.fit_transform(df, target_column='target')
        
        # Categorical column should be encoded
        assert pd.api.types.is_numeric_dtype(transformed_df['category'])
        assert log['success'] is True
    
    def test_datetime_feature_extraction(self):
        """Test datetime feature extraction."""
        df = pd.DataFrame({
            'datetime_col': pd.date_range('2023-01-01', periods=5, freq='D'),
            'value': [1, 2, 3, 4, 5]
        })
        
        normalizer = RobustNormalizer()
        transformed_df, log = normalizer.fit_transform(df)
        
        # Should have extracted datetime features
        datetime_features = [col for col in transformed_df.columns if 'datetime_col' in col]
        assert len(datetime_features) > 0
        assert 'datetime_col_year' in transformed_df.columns
        assert 'datetime_col_month' in transformed_df.columns
        assert log['success'] is True
    
    def test_error_handling_in_normalization(self):
        """Test error handling in normalization."""
        df = pd.DataFrame({
            'problematic_col': [1, 2, 3, 4, 5]
        })
        
        normalizer = RobustNormalizer()
        normalizer.fit(df)
        
        # Simulate error by corrupting internal state
        normalizer.scalers['problematic_col'] = Mock()
        normalizer.scalers['problematic_col'].transform.side_effect = Exception("Scaler failed")
        
        # Should handle error gracefully
        transformed_df, log = normalizer.transform(df, handle_errors=True)
        
        assert log['success'] is False
        assert len(log['errors']) > 0
        assert transformed_df is not None  # Should return something


class TestRobustDecorators:
    """Test robust operation decorators."""
    
    def test_robust_operation_success(self):
        """Test robust operation decorator with successful operation."""
        @robust_operation(max_retries=2)
        def successful_operation():
            return "success"
        
        result = successful_operation()
        assert result == "success"
    
    def test_robust_operation_retry(self):
        """Test robust operation decorator with retries."""
        call_count = 0
        
        @robust_operation(max_retries=2, backoff_factor=0.01)  # Fast backoff for testing
        def failing_then_success():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("Temporary failure")
            return "success"
        
        result = failing_then_success()
        assert result == "success"
        assert call_count == 3
    
    def test_robust_operation_final_failure(self):
        """Test robust operation decorator with final failure."""
        @robust_operation(max_retries=1, backoff_factor=0.01)
        def always_failing():
            raise Exception("Always fails")
        
        with pytest.raises(Exception, match="Always fails"):
            always_failing()
    
    def test_graceful_degradation_success(self):
        """Test graceful degradation decorator with successful operation."""
        @graceful_degradation(fallback_value="fallback")
        def successful_operation():
            return "success"
        
        result = successful_operation()
        assert result == "success"
    
    def test_graceful_degradation_fallback(self):
        """Test graceful degradation decorator with fallback."""
        @graceful_degradation(fallback_value="fallback", log_error=False)
        def failing_operation():
            raise Exception("Operation failed")
        
        result = failing_operation()
        assert result == "fallback"


class TestConfigValidator:
    """Test configuration validation."""
    
    def test_valid_config(self):
        """Test validation of valid configuration."""
        config = {
            'required_field': 'value',
            'optional_field': 42
        }
        
        schema = {
            'required': ['required_field'],
            'fields': {
                'required_field': {'type': str},
                'optional_field': {'type': int, 'min': 0, 'max': 100}
            }
        }
        
        result = ConfigValidator.validate_config(config, schema)
        
        assert result['valid'] is True
        assert len(result['errors']) == 0
    
    def test_invalid_config(self):
        """Test validation of invalid configuration."""
        config = {
            'wrong_type': 'should_be_int',
            'out_of_range': 150
        }
        
        schema = {
            'required': ['missing_field'],
            'fields': {
                'wrong_type': {'type': int},
                'out_of_range': {'type': int, 'min': 0, 'max': 100}
            }
        }
        
        result = ConfigValidator.validate_config(config, schema)
        
        assert result['valid'] is False
        assert len(result['errors']) >= 3  # Missing field, wrong type, out of range


class TestIntegration:
    """Integration tests for robustness features."""
    
    def test_end_to_end_data_processing(self):
        """Test end-to-end robust data processing."""
        # Create problematic dataset
        df = pd.DataFrame({
            'numerical_with_outliers': [1, 2, 3, 4, 1000, np.nan],
            'categorical_with_missing': ['a', 'b', np.nan, 'c', 'a', 'b'],
            'mixed_types': ['1', '2', 'invalid', '4', '5', '6'],
            'target': [0, 1, 0, 1, 0, 1]
        })
        
        # Step 1: Validate data
        validator = DataValidator(strict_mode=False)
        validation_result = validator.validate_dataframe(df)
        
        # Should detect issues but not fail
        assert validation_result['valid'] is True
        assert len(validation_result['warnings']) > 0
        
        # Step 2: Recover data
        recovery = DataRecovery(auto_fix=True)
        recovered_df, recovery_log = recovery.recover_dataframe(df, target_column='target')
        
        assert recovery_log['success'] is True
        assert len(recovery_log['actions_taken']) > 0
        
        # Step 3: Normalize data
        normalizer = RobustNormalizer(
            missing_strategy="median",
            outlier_action="clip",
            handle_new_categories="ignore"
        )
        
        normalized_df, norm_log = normalizer.fit_transform(recovered_df, target_column='target')
        
        assert norm_log['success'] is True
        assert normalized_df.isnull().sum().sum() == 0  # No missing values
        
        # Final result should be clean and usable
        assert len(normalized_df) > 0
        assert len(normalized_df.columns) >= len(df.columns) - 1  # Might have more due to encoding


if __name__ == "__main__":
    pytest.main([__file__])