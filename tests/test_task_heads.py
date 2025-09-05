"""Tests for task-specific heads."""
import pytest
import torch
import torch.nn as nn
import numpy as np
from tabgpt.heads import (
    BaseTaskHead,
    TaskType,
    TaskOutput,
    ClassificationHead,
    BinaryClassificationHead,
    MultiClassClassificationHead,
    MultiLabelClassificationHead,
    RegressionHead,
    AnomalyDetectionHead,
    SurvivalHead
)


class TestBaseTaskHead:
    """Test base task head functionality."""
    
    def test_task_output_creation(self):
        """Test creating task output."""
        predictions = torch.randn(4, 1)
        output = TaskOutput(predictions=predictions)
        assert torch.equal(output.predictions, predictions)
        assert output.loss is None
        assert output.probabilities is None
    
    def test_task_type_enum(self):
        """Test task type enumeration."""
        assert TaskType.BINARY_CLASSIFICATION == "binary_classification"
        assert TaskType.MULTICLASS_CLASSIFICATION == "multiclass_classification"
        assert TaskType.REGRESSION == "regression"
        assert TaskType.SURVIVAL_ANALYSIS == "survival_analysis"


class TestClassificationHeads:
    """Test classification heads."""
    
    def test_binary_classification_head(self):
        """Test binary classification head."""
        head = BinaryClassificationHead(input_dim=128)
        
        batch_size = 4
        x = torch.randn(batch_size, 128)
        
        outputs = head(x)
        assert "logits" in outputs
        assert "probabilities" in outputs
        assert outputs["logits"].shape == (batch_size, 1)
        assert outputs["probabilities"].shape == (batch_size, 1)
        
        # Test probabilities are in valid range
        probs = outputs["probabilities"]
        assert torch.all(probs >= 0) and torch.all(probs <= 1)
    
    def test_multiclass_classification_head(self):
        """Test multi-class classification head."""
        num_classes = 5
        head = MultiClassClassificationHead(input_dim=128, num_classes=num_classes)
        
        batch_size = 4
        x = torch.randn(batch_size, 128)
        
        outputs = head(x)
        assert "logits" in outputs
        assert "probabilities" in outputs
        assert outputs["logits"].shape == (batch_size, num_classes)
        assert outputs["probabilities"].shape == (batch_size, num_classes)
        
        # Test probabilities sum to 1
        probs = outputs["probabilities"]
        prob_sums = torch.sum(probs, dim=1)
        assert torch.allclose(prob_sums, torch.ones(batch_size), atol=1e-6)
    
    def test_multilabel_classification_head(self):
        """Test multi-label classification head."""
        num_labels = 3
        head = MultiLabelClassificationHead(input_dim=128, num_labels=num_labels)
        
        batch_size = 4
        x = torch.randn(batch_size, 128)
        
        outputs = head(x)
        assert "logits" in outputs
        assert "probabilities" in outputs
        assert outputs["logits"].shape == (batch_size, num_labels)
        assert outputs["probabilities"].shape == (batch_size, num_labels)
        
        # Test probabilities are in valid range for each label
        probs = outputs["probabilities"]
        assert torch.all(probs >= 0) and torch.all(probs <= 1)
    
    def test_classification_loss_computation(self):
        """Test classification loss computation."""
        # Binary classification
        binary_head = BinaryClassificationHead(input_dim=128)
        x = torch.randn(4, 128)
        outputs = binary_head(x)
        targets = {"labels": torch.randint(0, 2, (4,)).float()}
        loss = binary_head.compute_loss(outputs, targets)
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        
        # Multi-class classification
        multiclass_head = MultiClassClassificationHead(input_dim=128, num_classes=5)
        outputs = multiclass_head(x)
        targets = {"labels": torch.randint(0, 5, (4,))}
        loss = multiclass_head.compute_loss(outputs, targets)
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        
        # Multi-label classification
        multilabel_head = MultiLabelClassificationHead(input_dim=128, num_labels=3)
        outputs = multilabel_head(x)
        targets = {"labels": torch.randint(0, 2, (4, 3)).float()}
        loss = multilabel_head.compute_loss(outputs, targets)
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
    
    def test_classification_metrics(self):
        """Test classification metrics computation."""
        # Binary classification
        binary_head = BinaryClassificationHead(input_dim=128)
        x = torch.randn(4, 128)
        outputs = binary_head(x)
        targets = {"labels": torch.randint(0, 2, (4,)).float()}
        metrics = binary_head.compute_metrics(outputs, targets)
        
        assert isinstance(metrics, dict)
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1_score" in metrics


class TestRegressionHead:
    """Test regression head."""
    
    def test_regression_head_creation(self):
        """Test creating regression head."""
        head = RegressionHead(input_dim=128, output_dim=1)
        assert head.output_dim == 1
        
        # Multi-target regression
        head = RegressionHead(input_dim=128, output_dim=3)
        assert head.output_dim == 3
    
    def test_regression_forward(self):
        """Test regression forward pass."""
        head = RegressionHead(input_dim=128, output_dim=2)
        
        batch_size = 4
        x = torch.randn(batch_size, 128)
        
        outputs = head(x)
        assert "predictions" in outputs
        assert outputs["predictions"].shape == (batch_size, 2)
        
        # Test with uncertainty estimation
        config = HeadConfig(estimate_uncertainty=True)
        head = RegressionHead(input_dim=128, output_dim=1, config=config)
        outputs = head(x)
        assert "predictions" in outputs
        assert "uncertainty" in outputs
        assert outputs["predictions"].shape == (batch_size, 1)
        assert outputs["uncertainty"].shape == (batch_size, 1)
    
    def test_regression_loss(self):
        """Test regression loss computation."""
        head = RegressionHead(input_dim=128, output_dim=2)
        
        x = torch.randn(4, 128)
        outputs = head(x)
        targets = {"values": torch.randn(4, 2)}
        
        loss = head.compute_loss(outputs, targets)
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert loss.item() >= 0
    
    def test_regression_metrics(self):
        """Test regression metrics computation."""
        head = RegressionHead(input_dim=128, output_dim=1)
        
        x = torch.randn(4, 128)
        outputs = head(x)
        targets = {"values": torch.randn(4, 1)}
        
        metrics = head.compute_metrics(outputs, targets)
        assert isinstance(metrics, dict)
        assert "mse" in metrics
        assert "mae" in metrics
        assert "r2_score" in metrics


class TestAnomalyDetectionHead:
    """Test anomaly detection head."""
    
    def test_anomaly_detection_head_creation(self):
        """Test creating anomaly detection head."""
        head = AnomalyDetectionHead(input_dim=128)
        assert head.input_dim == 128
    
    def test_anomaly_detection_forward(self):
        """Test anomaly detection forward pass."""
        head = AnomalyDetectionHead(input_dim=128)
        
        batch_size = 4
        x = torch.randn(batch_size, 128)
        
        outputs = head(x)
        assert "anomaly_scores" in outputs
        assert "reconstructed" in outputs
        assert outputs["anomaly_scores"].shape == (batch_size, 1)
        assert outputs["reconstructed"].shape == (batch_size, 128)
    
    def test_anomaly_detection_loss(self):
        """Test anomaly detection loss computation."""
        head = AnomalyDetectionHead(input_dim=128)
        
        x = torch.randn(4, 128)
        outputs = head(x)
        targets = {"input": x, "labels": torch.randint(0, 2, (4,)).float()}
        
        loss = head.compute_loss(outputs, targets)
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert loss.item() >= 0
    
    def test_anomaly_detection_metrics(self):
        """Test anomaly detection metrics computation."""
        head = AnomalyDetectionHead(input_dim=128)
        
        x = torch.randn(4, 128)
        outputs = head(x)
        targets = {"labels": torch.randint(0, 2, (4,)).float()}
        
        metrics = head.compute_metrics(outputs, targets)
        assert isinstance(metrics, dict)
        assert "reconstruction_loss" in metrics
        assert "mean_anomaly_score" in metrics


class TestSurvivalHead:
    """Test survival analysis head."""
    
    def test_survival_head_creation(self):
        """Test creating survival head."""
        # Cox model
        head = SurvivalHead(input_dim=128, risk_estimation_method="cox")
        assert head.risk_estimation_method == "cox"
        
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
        
        outputs = head(x)
        assert "risk_score" in outputs
        assert "hazard_ratio" in outputs
        assert outputs["risk_score"].shape == (batch_size, 1)
        assert outputs["hazard_ratio"].shape == (batch_size, 1)
        
        # Hazard ratios should be positive
        assert torch.all(outputs["hazard_ratio"] > 0)
    
    def test_discrete_time_survival_forward(self):
        """Test discrete-time survival model forward pass."""
        head = SurvivalHead(input_dim=128, risk_estimation_method="discrete_time", num_time_bins=50)
        
        batch_size = 4
        x = torch.randn(batch_size, 128)
        
        outputs = head(x)
        assert "hazard_probs" in outputs
        assert "survival_probs" in outputs
        assert outputs["hazard_probs"].shape == (batch_size, 50)
        assert outputs["survival_probs"].shape == (batch_size, 50)
        
        # Probabilities should be in valid range
        assert torch.all(outputs["hazard_probs"] >= 0) and torch.all(outputs["hazard_probs"] <= 1)
        assert torch.all(outputs["survival_probs"] >= 0) and torch.all(outputs["survival_probs"] <= 1)
    
    def test_parametric_survival_forward(self):
        """Test parametric survival model forward pass."""
        # Weibull distribution
        head = SurvivalHead(input_dim=128, risk_estimation_method="parametric", distribution="weibull")
        
        batch_size = 4
        x = torch.randn(batch_size, 128)
        
        outputs = head(x)
        assert "shape" in outputs
        assert "scale" in outputs
        assert outputs["shape"].shape == (batch_size, 1)
        assert outputs["scale"].shape == (batch_size, 1)
        
        # Parameters should be positive
        assert torch.all(outputs["shape"] > 0)
        assert torch.all(outputs["scale"] > 0)
        
        # Exponential distribution
        head = SurvivalHead(input_dim=128, risk_estimation_method="parametric", distribution="exponential")
        outputs = head(x)
        assert "rate" in outputs
        assert torch.all(outputs["rate"] > 0)
        
        # Log-normal distribution
        head = SurvivalHead(input_dim=128, risk_estimation_method="parametric", distribution="log_normal")
        outputs = head(x)
        assert "mu" in outputs
        assert "sigma" in outputs
        assert torch.all(outputs["sigma"] > 0)
    
    def test_survival_loss_computation(self):
        """Test survival loss computation."""
        batch_size = 4
        x = torch.randn(batch_size, 128)
        
        # Create survival data
        time = torch.rand(batch_size) * 10 + 1  # Times between 1 and 11
        event = torch.randint(0, 2, (batch_size,)).float()  # Censoring indicator
        targets = {"time": time, "event": event}
        
        # Test Cox model
        cox_head = SurvivalHead(input_dim=128, risk_estimation_method="cox")
        outputs = cox_head(x)
        loss = cox_head.compute_loss(outputs, targets)
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        
        # Test discrete-time model
        discrete_head = SurvivalHead(input_dim=128, risk_estimation_method="discrete_time", max_time=12.0)
        outputs = discrete_head(x)
        loss = discrete_head.compute_loss(outputs, targets)
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        
        # Test parametric model
        weibull_head = SurvivalHead(input_dim=128, risk_estimation_method="parametric", distribution="weibull")
        outputs = weibull_head(x)
        loss = weibull_head.compute_loss(outputs, targets)
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
    
    def test_survival_metrics(self):
        """Test survival metrics computation."""
        batch_size = 4
        x = torch.randn(batch_size, 128)
        
        time = torch.rand(batch_size) * 10 + 1
        event = torch.randint(0, 2, (batch_size,)).float()
        targets = {"time": time, "event": event}
        
        # Test Cox model metrics
        cox_head = SurvivalHead(input_dim=128, risk_estimation_method="cox")
        outputs = cox_head(x)
        metrics = cox_head.compute_metrics(outputs, targets)
        
        assert isinstance(metrics, dict)
        assert "c_index" in metrics
        assert "mean_risk_score" in metrics
        assert "std_risk_score" in metrics
        
        # C-index should be between 0 and 1
        assert 0 <= metrics["c_index"] <= 1
    
    def test_survival_function_prediction(self):
        """Test survival function prediction."""
        batch_size = 4
        x = torch.randn(batch_size, 128)
        
        # Test discrete-time model
        head = SurvivalHead(input_dim=128, risk_estimation_method="discrete_time", num_time_bins=20)
        outputs = head(x)
        
        survival_func = head.predict_survival_function(outputs)
        assert survival_func.shape == (batch_size, 20)
        assert torch.all(survival_func >= 0) and torch.all(survival_func <= 1)
        
        # Test parametric model
        head = SurvivalHead(input_dim=128, risk_estimation_method="parametric", distribution="weibull")
        outputs = head(x)
        
        time_points = torch.linspace(0.1, 10, 50)
        survival_func = head.predict_survival_function(outputs, time_points)
        assert survival_func.shape == (batch_size, 50)
        assert torch.all(survival_func >= 0) and torch.all(survival_func <= 1)


class TestHeadIntegration:
    """Test integration between different heads."""
    
    def test_head_gradient_flow(self):
        """Test gradient flow through heads."""
        heads = [
            BinaryClassificationHead(input_dim=128),
            MultiClassClassificationHead(input_dim=128, num_classes=5),
            RegressionHead(input_dim=128, output_dim=2),
            AnomalyDetectionHead(input_dim=128),
            SurvivalHead(input_dim=128, risk_estimation_method="cox")
        ]
        
        for head in heads:
            x = torch.randn(4, 128, requires_grad=True)
            outputs = head(x)
            
            # Compute a simple loss
            if isinstance(head, (BinaryClassificationHead, MultiClassClassificationHead)):
                loss = outputs["logits"].sum()
            elif isinstance(head, RegressionHead):
                loss = outputs["predictions"].sum()
            elif isinstance(head, AnomalyDetectionHead):
                loss = outputs["anomaly_scores"].sum()
            elif isinstance(head, SurvivalHead):
                loss = outputs["risk_score"].sum()
            
            loss.backward()
            
            # Check gradients exist
            assert x.grad is not None
            assert not torch.allclose(x.grad, torch.zeros_like(x.grad))
    
    def test_head_configurations(self):
        """Test different head configurations."""
        configs = [
            HeadConfig(hidden_dims=[64], dropout=0.1, activation="relu"),
            HeadConfig(hidden_dims=[128, 64], dropout=0.2, activation="gelu"),
            HeadConfig(hidden_dims=[256, 128, 64], dropout=0.3, activation="tanh"),
        ]
        
        for config in configs:
            head = BinaryClassificationHead(input_dim=128, config=config)
            x = torch.randn(4, 128)
            outputs = head(x)
            
            assert "logits" in outputs
            assert outputs["logits"].shape == (4, 1)
    
    def test_head_device_compatibility(self):
        """Test head device compatibility."""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            
            head = BinaryClassificationHead(input_dim=128)
            head = head.to(device)
            
            x = torch.randn(4, 128, device=device)
            outputs = head(x)
            
            assert outputs["logits"].device == device
            assert outputs["probabilities"].device == device


class TestHeadEdgeCases:
    """Test edge cases for task heads."""
    
    def test_empty_batch(self):
        """Test heads with empty batch."""
        heads = [
            BinaryClassificationHead(input_dim=128),
            RegressionHead(input_dim=128, output_dim=1),
            AnomalyDetectionHead(input_dim=128)
        ]
        
        for head in heads:
            x = torch.empty(0, 128)
            outputs = head(x)
            
            # Should handle empty batch gracefully
            for key, value in outputs.items():
                assert value.shape[0] == 0
    
    def test_single_sample(self):
        """Test heads with single sample."""
        heads = [
            BinaryClassificationHead(input_dim=128),
            MultiClassClassificationHead(input_dim=128, num_classes=3),
            RegressionHead(input_dim=128, output_dim=2),
            AnomalyDetectionHead(input_dim=128),
            SurvivalHead(input_dim=128, risk_estimation_method="cox")
        ]
        
        for head in heads:
            x = torch.randn(1, 128)
            outputs = head(x)
            
            # Should handle single sample
            for key, value in outputs.items():
                assert value.shape[0] == 1
    
    def test_large_batch(self):
        """Test heads with large batch size."""
        batch_size = 1000
        head = BinaryClassificationHead(input_dim=128)
        
        x = torch.randn(batch_size, 128)
        outputs = head(x)
        
        assert outputs["logits"].shape == (batch_size, 1)
        assert outputs["probabilities"].shape == (batch_size, 1)


if __name__ == "__main__":
    pytest.main([__file__])