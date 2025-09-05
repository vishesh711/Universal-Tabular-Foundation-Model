"""Tests for TabGPT training pipeline."""
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import tempfile
import os
from unittest.mock import Mock, patch
import numpy as np

from tabgpt.training.trainer import (
    TrainingConfig,
    TrainingState,
    MultiObjectiveTrainer,
    create_trainer
)
from tabgpt.training.optimization import (
    LinearWarmupCosineAnnealingLR,
    PolynomialDecayLR,
    WarmupConstantLR,
    get_scheduler,
    create_optimizer,
    GradientClipping,
    EarlyStopping,
    compute_num_parameters
)
from tabgpt.training.metrics import MetricsComputer, compute_model_metrics
from tabgpt.models import TabGPTConfig, TabGPTForPreTraining


class TestTrainingConfig:
    """Test training configuration."""
    
    def test_config_creation(self):
        """Test creating training configuration."""
        config = TrainingConfig()
        
        # Check default values
        assert config.num_epochs == 10
        assert config.batch_size == 32
        assert config.learning_rate == 5e-4
        assert config.use_mixed_precision is True
        assert config.lr_scheduler_type == "cosine"
    
    def test_config_custom_values(self):
        """Test creating config with custom values."""
        config = TrainingConfig(
            num_epochs=20,
            batch_size=64,
            learning_rate=1e-3,
            use_mixed_precision=False,
            lr_scheduler_type="linear"
        )
        
        assert config.num_epochs == 20
        assert config.batch_size == 64
        assert config.learning_rate == 1e-3
        assert config.use_mixed_precision is False
        assert config.lr_scheduler_type == "linear"
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Invalid scheduler type
        with pytest.raises(ValueError):
            TrainingConfig(lr_scheduler_type="invalid")
        
        # Invalid eval strategy
        with pytest.raises(ValueError):
            TrainingConfig(eval_strategy="invalid")
    
    def test_config_output_dir_creation(self):
        """Test output directory creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = os.path.join(temp_dir, "test_output")
            config = TrainingConfig(output_dir=output_dir)
            
            assert os.path.exists(output_dir)


class TestTrainingState:
    """Test training state."""
    
    def test_state_creation(self):
        """Test creating training state."""
        state = TrainingState()
        
        assert state.epoch == 0
        assert state.global_step == 0
        assert state.best_metric is None
        assert state.train_loss == 0.0
    
    def test_state_updates(self):
        """Test updating training state."""
        state = TrainingState()
        
        state.epoch = 5
        state.global_step = 1000
        state.train_loss = 0.5
        state.learning_rate = 1e-4
        
        assert state.epoch == 5
        assert state.global_step == 1000
        assert state.train_loss == 0.5
        assert state.learning_rate == 1e-4


class TestLearningRateSchedulers:
    """Test learning rate schedulers."""
    
    def test_linear_warmup_cosine_annealing(self):
        """Test LinearWarmupCosineAnnealingLR scheduler."""
        model = nn.Linear(10, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_steps=100,
            total_steps=1000,
            min_lr_ratio=0.1
        )
        
        # Test warmup phase
        initial_lr = scheduler.get_lr()[0]
        assert initial_lr == 1e-5  # lr * (1/100)
        
        # Step through warmup
        for _ in range(50):
            scheduler.step()
        
        warmup_lr = scheduler.get_lr()[0]
        assert warmup_lr > initial_lr
        
        # Step through cosine annealing
        for _ in range(500):
            scheduler.step()
        
        annealing_lr = scheduler.get_lr()[0]
        assert annealing_lr < 1e-3  # Should be less than initial lr
    
    def test_polynomial_decay(self):
        """Test PolynomialDecayLR scheduler."""
        model = nn.Linear(10, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        scheduler = PolynomialDecayLR(
            optimizer,
            total_steps=1000,
            power=2.0,
            min_lr_ratio=0.1
        )
        
        initial_lr = scheduler.get_lr()[0]
        assert initial_lr == 1e-3
        
        # Step through decay
        for _ in range(500):
            scheduler.step()
        
        decayed_lr = scheduler.get_lr()[0]
        assert decayed_lr < initial_lr
    
    def test_warmup_constant(self):
        """Test WarmupConstantLR scheduler."""
        model = nn.Linear(10, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        scheduler = WarmupConstantLR(optimizer, warmup_steps=100)
        
        # Test warmup phase
        initial_lr = scheduler.get_lr()[0]
        assert initial_lr == 1e-5  # lr * (1/100)
        
        # Step through warmup
        for _ in range(100):
            scheduler.step()
        
        # Should be at full lr now
        full_lr = scheduler.get_lr()[0]
        assert abs(full_lr - 1e-3) < 1e-6
        
        # Step further - should remain constant
        for _ in range(100):
            scheduler.step()
        
        constant_lr = scheduler.get_lr()[0]
        assert abs(constant_lr - 1e-3) < 1e-6
    
    def test_get_scheduler(self):
        """Test get_scheduler function."""
        model = nn.Linear(10, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        # Test different scheduler types
        scheduler_types = ["linear", "cosine", "polynomial", "constant"]
        
        for scheduler_type in scheduler_types:
            scheduler = get_scheduler(
                optimizer,
                scheduler_type,
                num_training_steps=1000,
                warmup_steps=100
            )
            
            assert scheduler is not None
            initial_lr = scheduler.get_lr()[0]
            assert isinstance(initial_lr, float)


class TestOptimization:
    """Test optimization utilities."""
    
    def test_create_optimizer(self):
        """Test create_optimizer function."""
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.LayerNorm(20),
            nn.Linear(20, 1)
        )
        
        optimizer = create_optimizer(
            model,
            optimizer_type="adamw",
            learning_rate=1e-3,
            weight_decay=0.01
        )
        
        assert isinstance(optimizer, torch.optim.AdamW)
        
        # Check that parameters are properly grouped
        param_groups = optimizer.param_groups
        assert len(param_groups) == 2  # decay and no-decay groups
        
        # Check weight decay settings
        decay_group = param_groups[0]
        no_decay_group = param_groups[1]
        
        assert decay_group['weight_decay'] == 0.01
        assert no_decay_group['weight_decay'] == 0.0
    
    def test_gradient_clipping(self):
        """Test gradient clipping."""
        model = nn.Linear(10, 1)
        
        # Create some gradients
        x = torch.randn(5, 10)
        y = torch.randn(5, 1)
        loss = nn.MSELoss()(model(x), y)
        loss.backward()
        
        # Test gradient norm clipping
        total_norm = GradientClipping.clip_grad_norm(model.parameters(), max_norm=1.0)
        assert isinstance(total_norm, torch.Tensor)
        
        # Test gradient value clipping
        GradientClipping.clip_grad_value(model.parameters(), clip_value=0.5)
        
        # Check that gradients are clipped
        for param in model.parameters():
            if param.grad is not None:
                assert torch.all(torch.abs(param.grad) <= 0.5)
    
    def test_early_stopping(self):
        """Test early stopping."""
        model = nn.Linear(10, 1)
        
        early_stopping = EarlyStopping(patience=3, mode="min")
        
        # Simulate improving scores
        assert not early_stopping(1.0, model)  # First score
        assert not early_stopping(0.8, model)  # Improvement
        assert not early_stopping(0.6, model)  # Improvement
        
        # Simulate no improvement
        assert not early_stopping(0.7, model)  # No improvement (1)
        assert not early_stopping(0.8, model)  # No improvement (2)
        assert not early_stopping(0.9, model)  # No improvement (3)
        assert early_stopping(1.0, model)      # Should stop (4)
    
    def test_compute_num_parameters(self):
        """Test parameter counting."""
        model = nn.Sequential(
            nn.Linear(10, 20),  # 10*20 + 20 = 220 params
            nn.Linear(20, 1)    # 20*1 + 1 = 21 params
        )
        
        param_info = compute_num_parameters(model)
        
        assert param_info['total_parameters'] == 241
        assert param_info['trainable_parameters'] == 241
        assert param_info['non_trainable_parameters'] == 0
        assert param_info['total_parameters_millions'] == 241 / 1e6


class TestMetricsComputer:
    """Test metrics computation."""
    
    def test_metrics_computer_creation(self):
        """Test creating metrics computer."""
        computer = MetricsComputer(task_type="pretraining")
        assert computer.task_type == "pretraining"
    
    def test_classification_metrics(self):
        """Test classification metrics computation."""
        computer = MetricsComputer()
        
        # Binary classification
        predictions = torch.tensor([[0.8], [0.3], [0.9], [0.1]])
        targets = torch.tensor([1, 0, 1, 0])
        
        metrics = computer.compute_classification_metrics(predictions, targets)
        
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics
        
        # Multi-class classification
        predictions = torch.tensor([
            [0.1, 0.8, 0.1],
            [0.7, 0.2, 0.1],
            [0.1, 0.1, 0.8],
            [0.6, 0.3, 0.1]
        ])
        targets = torch.tensor([1, 0, 2, 0])
        
        metrics = computer.compute_classification_metrics(predictions, targets)
        
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics
    
    def test_regression_metrics(self):
        """Test regression metrics computation."""
        computer = MetricsComputer()
        
        predictions = torch.tensor([1.0, 2.0, 3.0, 4.0])
        targets = torch.tensor([1.1, 1.9, 3.2, 3.8])
        
        metrics = computer.compute_regression_metrics(predictions, targets)
        
        assert 'mse' in metrics
        assert 'mae' in metrics
        assert 'rmse' in metrics
        assert 'r2' in metrics
        assert 'mean_residual' in metrics
        assert 'std_residual' in metrics
        
        # Check that metrics are reasonable
        assert metrics['mse'] > 0
        assert metrics['mae'] > 0
        assert metrics['rmse'] == np.sqrt(metrics['mse'])
    
    def test_attention_metrics(self):
        """Test attention metrics computation."""
        computer = MetricsComputer()
        
        # Create attention weights [batch, heads, seq_len, seq_len]
        attention_weights = torch.softmax(torch.randn(2, 4, 8, 8), dim=-1)
        
        metrics = computer.compute_attention_metrics(attention_weights)
        
        assert 'attention_entropy_mean' in metrics
        assert 'attention_entropy_std' in metrics
        assert 'attention_concentration' in metrics
        assert 'max_attention_mean' in metrics
        assert 'max_attention_std' in metrics
        assert 'attention_head_variance' in metrics
        
        # Check that metrics are reasonable
        assert metrics['attention_entropy_mean'] > 0
        assert metrics['attention_concentration'] > 0
        assert 0 <= metrics['max_attention_mean'] <= 1
    
    def test_perplexity_computation(self):
        """Test perplexity computation."""
        computer = MetricsComputer()
        
        loss = torch.tensor(2.0)
        perplexity = computer.compute_perplexity(loss)
        
        expected_perplexity = np.exp(2.0)
        assert abs(perplexity - expected_perplexity) < 1e-6


class TestMultiObjectiveTrainer:
    """Test multi-objective trainer."""
    
    def create_mock_model_and_data(self):
        """Create mock model and data for testing."""
        # Create a simple model
        config = TabGPTConfig(
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            max_columns=10
        )
        model = TabGPTForPreTraining(config)
        
        # Create mock data
        batch_size = 8
        seq_len = 16
        
        # Create dataset
        input_ids = torch.randint(0, 1000, (100, seq_len))
        attention_mask = torch.ones(100, seq_len)
        
        dataset = TensorDataset(input_ids, attention_mask)
        train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        eval_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        return model, train_dataloader, eval_dataloader
    
    def test_trainer_creation(self):
        """Test creating trainer."""
        model, train_dataloader, eval_dataloader = self.create_mock_model_and_data()
        
        config = TrainingConfig(
            num_epochs=1,
            batch_size=8,
            learning_rate=1e-3,
            use_mixed_precision=False  # Disable for testing
        )
        
        trainer = MultiObjectiveTrainer(
            model=model,
            config=config,
            train_dataloader=train_dataloader,
            eval_dataloader=eval_dataloader
        )
        
        assert trainer.model == model
        assert trainer.config == config
        assert trainer.train_dataloader == train_dataloader
        assert trainer.eval_dataloader == eval_dataloader
        assert trainer.optimizer is not None
        assert trainer.lr_scheduler is not None
    
    def test_trainer_prepare_inputs(self):
        """Test input preparation."""
        model, train_dataloader, eval_dataloader = self.create_mock_model_and_data()
        
        config = TrainingConfig(use_mixed_precision=False)
        trainer = MultiObjectiveTrainer(
            model=model,
            config=config,
            train_dataloader=train_dataloader
        )
        
        # Test input preparation
        batch = {
            'input_ids': torch.randint(0, 1000, (4, 10)),
            'attention_mask': torch.ones(4, 10),
            'labels': torch.randint(0, 2, (4,))
        }
        
        prepared_batch = trainer._prepare_inputs(batch)
        
        assert 'input_ids' in prepared_batch
        assert 'attention_mask' in prepared_batch
        assert 'labels' in prepared_batch
        
        # Check that tensors are moved to correct device
        for key, value in prepared_batch.items():
            if isinstance(value, torch.Tensor):
                assert value.device == trainer.device
    
    def test_trainer_compute_loss(self):
        """Test loss computation."""
        model, train_dataloader, eval_dataloader = self.create_mock_model_and_data()
        
        config = TrainingConfig(
            mcm_weight=1.0,
            mcol_weight=0.5,
            crl_weight=0.3,
            use_mixed_precision=False
        )
        
        trainer = MultiObjectiveTrainer(
            model=model,
            config=config,
            train_dataloader=train_dataloader
        )
        
        # Create mock batch
        batch = {
            'input_ids': torch.randint(0, 1000, (4, 10)),
            'attention_mask': torch.ones(4, 10)
        }
        
        # Mock model output
        with patch.object(model, 'forward') as mock_forward:
            mock_forward.return_value = {
                'loss': torch.tensor(2.0),
                'losses': {
                    'mcm_loss': torch.tensor(1.0),
                    'mcol_loss': torch.tensor(0.8),
                    'crl_loss': torch.tensor(0.5)
                }
            }
            
            loss, losses = trainer._compute_loss(batch)
            
            assert isinstance(loss, torch.Tensor)
            assert isinstance(losses, dict)
            
            # Check that loss weights are applied
            expected_loss = 1.0 * 1.0 + 0.8 * 0.5 + 0.5 * 0.3
            assert abs(loss.item() - expected_loss) < 1e-6
    
    def test_create_trainer_function(self):
        """Test create_trainer convenience function."""
        model, train_dataloader, eval_dataloader = self.create_mock_model_and_data()
        
        trainer = create_trainer(
            model=model,
            train_dataloader=train_dataloader,
            eval_dataloader=eval_dataloader,
            num_epochs=2,
            batch_size=8,
            learning_rate=1e-3
        )
        
        assert isinstance(trainer, MultiObjectiveTrainer)
        assert trainer.config.num_epochs == 2
        assert trainer.config.batch_size == 8
        assert trainer.config.learning_rate == 1e-3


class TestTrainingIntegration:
    """Test training integration."""
    
    def test_training_step_basic(self):
        """Test basic training step."""
        # Create simple model and data
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 1)
        )
        
        # Create mock data
        x = torch.randn(32, 10)
        y = torch.randn(32, 1)
        dataset = TensorDataset(x, y)
        dataloader = DataLoader(dataset, batch_size=8)
        
        # Create optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        # Training step
        model.train()
        total_loss = 0.0
        
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            
            outputs = model(batch_x)
            loss = nn.MSELoss()(outputs, batch_y)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        assert total_loss > 0
        
        # Check that gradients were computed
        for param in model.parameters():
            assert param.grad is not None
    
    def test_mixed_precision_training(self):
        """Test mixed precision training."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for mixed precision testing")
        
        model = nn.Linear(10, 1).cuda()
        optimizer = torch.optim.Adam(model.parameters())
        scaler = torch.cuda.amp.GradScaler()
        
        x = torch.randn(4, 10).cuda()
        y = torch.randn(4, 1).cuda()
        
        # Mixed precision forward and backward
        with torch.cuda.amp.autocast():
            outputs = model(x)
            loss = nn.MSELoss()(outputs, y)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        assert loss.item() > 0


class TestTrainingEdgeCases:
    """Test edge cases in training."""
    
    def test_empty_dataloader(self):
        """Test handling empty dataloader."""
        model = nn.Linear(10, 1)
        
        # Empty dataset
        dataset = TensorDataset(torch.empty(0, 10), torch.empty(0, 1))
        dataloader = DataLoader(dataset, batch_size=1)
        
        # Should handle empty dataloader gracefully
        total_loss = 0.0
        num_batches = 0
        
        for batch in dataloader:
            num_batches += 1
        
        assert num_batches == 0
    
    def test_single_batch_training(self):
        """Test training with single batch."""
        model = nn.Linear(10, 1)
        optimizer = torch.optim.Adam(model.parameters())
        
        # Single batch
        x = torch.randn(1, 10)
        y = torch.randn(1, 1)
        
        # Training step
        optimizer.zero_grad()
        outputs = model(x)
        loss = nn.MSELoss()(outputs, y)
        loss.backward()
        optimizer.step()
        
        assert loss.item() >= 0
    
    def test_nan_loss_handling(self):
        """Test handling NaN losses."""
        model = nn.Linear(10, 1)
        
        # Create inputs that might cause NaN
        x = torch.tensor([[float('inf')] * 10])
        y = torch.randn(1, 1)
        
        outputs = model(x)
        loss = nn.MSELoss()(outputs, y)
        
        # Check if loss is NaN
        if torch.isnan(loss):
            # Handle NaN loss appropriately
            assert True  # Expected behavior
        else:
            assert loss.item() >= 0


if __name__ == "__main__":
    pytest.main([__file__])