"""Simple tests for task-specific heads."""
import pytest
import torch
import torch.nn as nn
import numpy as np
from tabgpt.heads.base import BaseTaskHead, TaskType, TaskOutput, MLPHead
from tabgpt.heads.survival import SurvivalHead


class TestTaskHeadBasics:
    """Test basic task head functionality."""
    
    def test_task_type_enum(self):
        """Test task type enumeration."""
        assert TaskType.BINARY_CLASSIFICATION.value == "binary_classification"
        assert TaskType.MULTICLASS_CLASSIFICATION.value == "multiclass_classification"
        assert TaskType.REGRESSION.value == "regression"
        assert TaskType.SURVIVAL_ANALYSIS.value == "survival_analysis"
    
    def test_task_output_creation(self):
        """Test creating task output."""
        predictions = torch.randn(4, 1)
        output = TaskOutput(predictions=predictions)
        assert torch.equal(output.predictions, predictions)
        assert output.loss is None
        assert output.probabilities is None
    
    def test_mlp_head_binary_classification(self):
        """Test MLP head for binary classification."""
        head = MLPHead(
            input_dim=128,
            output_dim=1,
            task_type=TaskType.BINARY_CLASSIFICATION,
            hidden_dims=[64]
        )
        
        batch_size = 4
        x = torch.randn(batch_size, 128)
        
        output = head(x)
        assert isinstance(output, TaskOutput)
        assert output.predictions.shape == (batch_size, 1)
        assert output.probabilities is not None
        assert output.logits.shape == (batch_size, 1)
        
        # Test probabilities are in valid range
        probs = output.probabilities
        assert torch.all(probs >= 0) and torch.all(probs <= 1)
    
    def test_mlp_head_multiclass_classification(self):
        """Test MLP head for multi-class classification."""
        num_classes = 5
        head = MLPHead(
            input_dim=128,
            output_dim=num_classes,
            task_type=TaskType.MULTICLASS_CLASSIFICATION,
            hidden_dims=[64]
        )
        
        batch_size = 4
        x = torch.randn(batch_size, 128)
        
        output = head(x)
        assert output.predictions.shape == (batch_size, num_classes)
        assert output.probabilities.shape == (batch_size, num_classes)
        
        # Test probabilities sum to 1
        probs = output.probabilities
        prob_sums = torch.sum(probs, dim=1)
        assert torch.allclose(prob_sums, torch.ones(batch_size), atol=1e-6)
    
    def test_mlp_head_regression(self):
        """Test MLP head for regression."""
        head = MLPHead(
            input_dim=128,
            output_dim=2,
            task_type=TaskType.REGRESSION,
            hidden_dims=[64]
        )
        
        batch_size = 4
        x = torch.randn(batch_size, 128)
        
        output = head(x)
        assert output.predictions.shape == (batch_size, 2)
        assert output.probabilities is None  # No probabilities for regression
    
    def test_mlp_head_loss_computation(self):
        """Test loss computation for MLP head."""
        # Binary classification
        head = MLPHead(
            input_dim=128,
            output_dim=1,
            task_type=TaskType.BINARY_CLASSIFICATION
        )
        
        x = torch.randn(4, 128)
        targets = torch.randint(0, 2, (4,)).float()
        
        output = head(x, targets)
        assert output.loss is not None
        assert isinstance(output.loss, torch.Tensor)
        assert output.loss.dim() == 0
        
        # Regression
        reg_head = MLPHead(
            input_dim=128,
            output_dim=1,
            task_type=TaskType.REGRESSION
        )
        
        reg_targets = torch.randn(4, 1)
        reg_output = reg_head(x, reg_targets)
        assert reg_output.loss is not None
        assert reg_output.loss.item() >= 0
    
    def test_mlp_head_metrics(self):
        """Test metrics computation for MLP head."""
        head = MLPHead(
            input_dim=128,
            output_dim=1,
            task_type=TaskType.BINARY_CLASSIFICATION
        )
        
        predictions = torch.sigmoid(torch.randn(4, 1))
        targets = torch.randint(0, 2, (4,)).float()
        
        metrics = head.compute_metrics(predictions, targets)
        assert isinstance(metrics, dict)
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1_score" in metrics


class TestSurvivalHead:
    """Test survival analysis head."""
    
    def test_survival_head_creation(self):
        """Test creating survival head."""
        # Cox model
        head = SurvivalHead(input_dim=128, risk_estimation_method="cox")
        assert head.risk_estimation_method == "cox"
        assert head.input_dim == 128
        
        # Discrete-time model
        head = SurvivalHead(input_dim=128, risk_estimation_method="discrete_time")
        assert head.risk_estimation_method == "discrete_time"
        
        # Parametric model
        head = SurvivalHead(input_dim=128, risk_estimation_method="parametric", distribution="weibull")
        assert head.risk_estimation_method == "parametric"
        assert head.distribution == "weibull"
    
    def test_cox_survival_forward(self):
        """Test Cox survival model forward pass."""
        head = SurvivalHead(input_dim=128, risk_estimation_method="cox")
        
        batch_size = 4
        x = torch.randn(batch_size, 128)
        
        output = head(x)
        assert isinstance(output, TaskOutput)
        assert output.predictions is not None
        assert output.metadata is not None
        assert "risk_score" in output.metadata
        assert "hazard_ratio" in output.metadata
        
        # Hazard ratios should be positive
        hazard_ratios = output.metadata["hazard_ratio"]
        assert torch.all(hazard_ratios > 0)
    
    def test_discrete_time_survival_forward(self):
        """Test discrete-time survival model forward pass."""
        head = SurvivalHead(input_dim=128, risk_estimation_method="discrete_time", num_time_bins=50)
        
        batch_size = 4
        x = torch.randn(batch_size, 128)
        
        output = head(x)
        assert output.metadata is not None
        assert "hazard_probs" in output.metadata
        assert "survival_probs" in output.metadata
        
        hazard_probs = output.metadata["hazard_probs"]
        survival_probs = output.metadata["survival_probs"]
        
        assert hazard_probs.shape == (batch_size, 50)
        assert survival_probs.shape == (batch_size, 50)
        
        # Probabilities should be in valid range
        assert torch.all(hazard_probs >= 0) and torch.all(hazard_probs <= 1)
        assert torch.all(survival_probs >= 0) and torch.all(survival_probs <= 1)
    
    def test_parametric_survival_forward(self):
        """Test parametric survival model forward pass."""
        # Weibull distribution
        head = SurvivalHead(input_dim=128, risk_estimation_method="parametric", distribution="weibull")
        
        batch_size = 4
        x = torch.randn(batch_size, 128)
        
        output = head(x)
        assert output.metadata is not None
        assert "shape" in output.metadata
        assert "scale" in output.metadata
        
        shape = output.metadata["shape"]
        scale = output.metadata["scale"]
        
        assert shape.shape == (batch_size, 1)
        assert scale.shape == (batch_size, 1)
        
        # Parameters should be positive
        assert torch.all(shape > 0)
        assert torch.all(scale > 0)
    
    def test_survival_loss_computation(self):
        """Test survival loss computation."""
        head = SurvivalHead(input_dim=128, risk_estimation_method="cox")
        
        batch_size = 4
        x = torch.randn(batch_size, 128)
        targets = torch.randn(batch_size, 1)  # Simplified targets
        
        output = head(x, targets)
        assert output.loss is not None
        assert isinstance(output.loss, torch.Tensor)
        assert output.loss.dim() == 0
    
    def test_survival_gradient_flow(self):
        """Test gradient flow through survival head."""
        head = SurvivalHead(input_dim=128, risk_estimation_method="cox")
        
        x = torch.randn(4, 128, requires_grad=True)
        targets = torch.randn(4, 1)
        
        output = head(x, targets)
        loss = output.loss
        
        loss.backward()
        
        # Check gradients exist
        assert x.grad is not None
        assert not torch.allclose(x.grad, torch.zeros_like(x.grad))


class TestHeadEdgeCases:
    """Test edge cases for task heads."""
    
    def test_empty_batch(self):
        """Test heads with empty batch."""
        head = MLPHead(
            input_dim=128,
            output_dim=1,
            task_type=TaskType.BINARY_CLASSIFICATION
        )
        
        x = torch.empty(0, 128)
        output = head(x)
        
        # Should handle empty batch gracefully
        assert output.predictions.shape[0] == 0
        assert output.logits.shape[0] == 0
    
    def test_single_sample(self):
        """Test heads with single sample."""
        head = MLPHead(
            input_dim=128,
            output_dim=1,
            task_type=TaskType.BINARY_CLASSIFICATION
        )
        
        x = torch.randn(1, 128)
        output = head(x)
        
        # Should handle single sample
        assert output.predictions.shape[0] == 1
        assert output.logits.shape[0] == 1
    
    def test_large_batch(self):
        """Test heads with large batch size."""
        batch_size = 1000
        head = MLPHead(
            input_dim=128,
            output_dim=1,
            task_type=TaskType.BINARY_CLASSIFICATION
        )
        
        x = torch.randn(batch_size, 128)
        output = head(x)
        
        assert output.predictions.shape == (batch_size, 1)
        assert output.logits.shape == (batch_size, 1)


if __name__ == "__main__":
    pytest.main([__file__])