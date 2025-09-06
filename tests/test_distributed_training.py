"""Tests for distributed training functionality."""

import pytest
import os
import tempfile
import time
import threading
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

import torch
import torch.nn as nn
import torch.distributed as dist
import pandas as pd
import numpy as np

from tabgpt.training.distributed import (
    DistributedConfig, DistributedManager, DataSharding,
    GradientSynchronization, ModelParallelism, DistributedTrainer,
    setup_distributed_training, get_world_size, get_rank, is_main_process
)
from tabgpt.training.distributed_monitoring import (
    DistributedMetrics, ResourceMonitor, CommunicationProfiler,
    DistributedLogger, DistributedTrainingMonitor
)
from tabgpt.training.trainer import TrainingConfig


class MockModel(nn.Module):
    """Mock model for testing."""
    
    def __init__(self, input_size=10, hidden_size=20, output_size=2):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
        self.config = type('Config', (), {'hidden_size': hidden_size})()
    
    def forward(self, **kwargs):
        # Simple forward pass for testing
        batch_size = 4  # Mock batch size
        return {
            'loss': torch.tensor(0.5, requires_grad=True),
            'losses': {
                'mcm_loss': torch.tensor(0.2),
                'mcol_loss': torch.tensor(0.1),
                'crl_loss': torch.tensor(0.1),
                'nrp_loss': torch.tensor(0.1)
            }
        }
    
    def save_pretrained(self, path):
        """Mock save method."""
        Path(path).mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), Path(path) / "pytorch_model.bin")


class MockDataset:
    """Mock dataset for testing."""
    
    def __init__(self, size=100):
        self.size = size
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return {
            'input_ids': torch.randn(10),
            'attention_mask': torch.ones(10),
            'labels': torch.randint(0, 2, (1,))
        }


class TestDistributedConfig:
    """Test distributed configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = DistributedConfig()
        
        assert config.backend == "nccl"
        assert config.world_size == 1
        assert config.rank == 0
        assert config.local_rank == 0
        assert config.use_data_parallel is True
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Valid configuration
        config = DistributedConfig(world_size=4, rank=2)
        assert config.world_size == 4
        assert config.rank == 2
        
        # Invalid world size
        with pytest.raises(ValueError):
            DistributedConfig(world_size=0)
        
        # Invalid rank
        with pytest.raises(ValueError):
            DistributedConfig(world_size=2, rank=2)
        
        # Invalid local rank
        with pytest.raises(ValueError):
            DistributedConfig(local_rank=-1)


class TestDistributedManager:
    """Test distributed manager functionality."""
    
    def test_manager_initialization(self):
        """Test manager initialization."""
        config = DistributedConfig(world_size=1, rank=0)
        manager = DistributedManager(config)
        
        assert manager.config == config
        assert not manager.is_initialized
        assert manager.comm_stats['allreduce_calls'] == 0
    
    @patch('torch.distributed.is_initialized')
    @patch('torch.distributed.init_process_group')
    def test_initialize_single_process(self, mock_init_pg, mock_is_init):
        """Test initialization for single process."""
        mock_is_init.return_value = False
        
        config = DistributedConfig(world_size=1, rank=0)
        manager = DistributedManager(config)
        
        # Should not initialize distributed for single process
        manager.initialize()
        assert manager.is_initialized
    
    def test_communication_stats(self):
        """Test communication statistics tracking."""
        config = DistributedConfig(world_size=1, rank=0)
        manager = DistributedManager(config)
        
        # Test tensor operations without actual distributed setup
        tensor = torch.tensor([1.0, 2.0, 3.0])
        
        # For single process, operations should return original tensor
        result = manager.all_reduce(tensor)
        assert torch.equal(result, tensor)
        
        # Stats should be updated
        stats = manager.get_communication_stats()
        assert 'allreduce_calls' in stats
        assert 'total_communication_time' in stats


class TestDataSharding:
    """Test data sharding functionality."""
    
    def test_distributed_sampler_creation(self):
        """Test creation of distributed sampler."""
        config = DistributedConfig(world_size=2, rank=0)
        sharding = DataSharding(config)
        
        dataset = MockDataset(100)
        sampler = sharding.create_distributed_sampler(dataset)
        
        assert sampler.num_replicas == 2
        assert sampler.rank == 0
        assert len(sampler) <= len(dataset)
    
    def test_dataset_sharding_by_size(self):
        """Test dataset sharding by size."""
        config = DistributedConfig(world_size=2, rank=0)
        sharding = DataSharding(config)
        
        dataset = MockDataset(100)
        shards = sharding.shard_dataset_by_size(dataset, shard_size=30)
        
        # Should create 4 shards (100 / 30 = 3.33, rounded up to 4)
        assert len(shards) == 4
        
        # Check shard sizes
        assert len(shards[0]) == 30
        assert len(shards[1]) == 30
        assert len(shards[2]) == 30
        assert len(shards[3]) == 10  # Remaining samples
    
    def test_shard_balancing(self):
        """Test shard balancing across processes."""
        config = DistributedConfig(world_size=2, rank=0)
        sharding = DataSharding(config)
        
        dataset = MockDataset(100)
        shards = sharding.shard_dataset_by_size(dataset, shard_size=25)
        balanced_shards = sharding.balance_shards(shards)
        
        # Rank 0 should get shards 0, 2, 4, ...
        # For 4 shards and 2 processes, rank 0 gets shards 0 and 2
        assert len(balanced_shards) == 2


class TestGradientSynchronization:
    """Test gradient synchronization."""
    
    def test_gradient_sync_initialization(self):
        """Test gradient synchronization initialization."""
        config = DistributedConfig(world_size=2, use_gradient_compression=True)
        grad_sync = GradientSynchronization(config)
        
        assert grad_sync.config == config
        assert grad_sync.compression_enabled is True
    
    def test_gradient_accumulation(self):
        """Test gradient accumulation."""
        config = DistributedConfig(world_size=1)
        grad_sync = GradientSynchronization(config)
        
        model = MockModel()
        
        # Create some gradients
        loss = torch.tensor(2.0, requires_grad=True)
        loss.backward()
        
        # Test accumulation
        grad_sync.accumulate_gradients(model, accumulation_steps=4)
        
        # Gradients should be scaled
        for param in model.parameters():
            if param.grad is not None:
                # Gradient should be divided by accumulation steps
                assert param.grad.abs().max() <= 1.0  # Should be scaled down


class TestModelParallelism:
    """Test model parallelism functionality."""
    
    def test_pipeline_parallel_setup(self):
        """Test pipeline parallelism setup."""
        config = DistributedConfig(world_size=2, rank=0)
        model_parallel = ModelParallelism(config)
        
        model = MockModel()
        
        # Test pipeline parallel setup
        pipeline_model = model_parallel.setup_pipeline_parallel(model, num_stages=2)
        
        # Should return a model (even if simplified)
        assert isinstance(pipeline_model, nn.Module)
    
    def test_tensor_parallel_setup(self):
        """Test tensor parallelism setup."""
        config = DistributedConfig(world_size=2, rank=0)
        model_parallel = ModelParallelism(config)
        
        model = MockModel()
        
        # Test tensor parallel setup (simplified)
        tensor_model = model_parallel.setup_tensor_parallel(model, tensor_parallel_size=2)
        
        # Should return the model (implementation is simplified)
        assert isinstance(tensor_model, nn.Module)


class TestDistributedMetrics:
    """Test distributed metrics."""
    
    def test_metrics_creation(self):
        """Test metrics object creation."""
        metrics = DistributedMetrics(
            train_loss=0.5,
            eval_loss=0.3,
            learning_rate=1e-4,
            epoch=1,
            global_step=100
        )
        
        assert metrics.train_loss == 0.5
        assert metrics.eval_loss == 0.3
        assert metrics.learning_rate == 1e-4
        assert metrics.epoch == 1
        assert metrics.global_step == 100
    
    def test_metrics_to_dict(self):
        """Test metrics conversion to dictionary."""
        metrics = DistributedMetrics(
            train_loss=0.5,
            samples_per_second=100.0,
            world_size=4,
            rank=1
        )
        
        metrics_dict = metrics.to_dict()
        
        assert isinstance(metrics_dict, dict)
        assert metrics_dict['train_loss'] == 0.5
        assert metrics_dict['samples_per_second'] == 100.0
        assert metrics_dict['world_size'] == 4
        assert metrics_dict['rank'] == 1
        assert 'timestamp' in metrics_dict


class TestResourceMonitor:
    """Test resource monitoring."""
    
    def test_monitor_initialization(self):
        """Test resource monitor initialization."""
        monitor = ResourceMonitor(monitoring_interval=0.1)
        
        assert monitor.monitoring_interval == 0.1
        assert not monitor.is_monitoring
        assert len(monitor.metrics_history) == 0
    
    def test_resource_metrics_collection(self):
        """Test resource metrics collection."""
        monitor = ResourceMonitor()
        
        metrics = monitor.get_current_metrics()
        
        assert isinstance(metrics, dict)
        assert 'timestamp' in metrics
        
        # Should have some metrics (depending on system)
        if torch.cuda.is_available():
            assert 'gpu_memory_used_mb' in metrics
    
    def test_monitoring_lifecycle(self):
        """Test monitoring start/stop lifecycle."""
        monitor = ResourceMonitor(monitoring_interval=0.01)
        
        # Start monitoring
        monitor.start_monitoring()
        assert monitor.is_monitoring
        assert monitor.monitor_thread is not None
        
        # Let it collect some metrics
        time.sleep(0.05)
        
        # Stop monitoring
        monitor.stop_monitoring()
        assert not monitor.is_monitoring
        
        # Should have collected some metrics
        history = monitor.get_metrics_history()
        assert len(history) > 0


class TestCommunicationProfiler:
    """Test communication profiling."""
    
    def test_profiler_initialization(self):
        """Test communication profiler initialization."""
        profiler = CommunicationProfiler()
        
        assert len(profiler.communication_events) == 0
        assert len(profiler.active_operations) == 0
    
    def test_operation_tracking(self):
        """Test communication operation tracking."""
        profiler = CommunicationProfiler()
        
        # Start operation
        op_id = profiler.start_operation("allreduce", data_size_bytes=1024)
        assert op_id in profiler.active_operations
        
        # Simulate some work
        time.sleep(0.01)
        
        # End operation
        profiler.end_operation(op_id)
        assert op_id not in profiler.active_operations
        assert len(profiler.communication_events) == 1
        
        # Check event details
        event = profiler.communication_events[0]
        assert event['type'] == "allreduce"
        assert event['data_size_bytes'] == 1024
        assert event['duration'] > 0
        assert event['bandwidth_mbps'] >= 0
    
    def test_communication_stats(self):
        """Test communication statistics."""
        profiler = CommunicationProfiler()
        
        # Track multiple operations
        for i in range(5):
            op_id = profiler.start_operation("allreduce", data_size_bytes=1024 * (i + 1))
            time.sleep(0.001)
            profiler.end_operation(op_id)
        
        stats = profiler.get_communication_stats()
        
        assert 'allreduce' in stats
        assert stats['allreduce']['count'] == 5
        assert stats['allreduce']['total_duration'] > 0
        assert stats['allreduce']['avg_duration'] > 0
        
        assert 'overall' in stats
        assert stats['overall']['total_operations'] == 5


class TestDistributedLogger:
    """Test distributed logging."""
    
    def test_logger_initialization(self):
        """Test distributed logger initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = DistributedLogger(temp_dir, rank=0, world_size=2)
            
            assert logger.rank == 0
            assert logger.world_size == 2
            assert logger.is_main_process is True
            assert logger.log_dir.exists()
    
    def test_metrics_logging(self):
        """Test metrics logging."""
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = DistributedLogger(temp_dir, rank=0, world_size=1)
            
            # Log some metrics
            metrics = DistributedMetrics(
                train_loss=0.5,
                epoch=1,
                global_step=100
            )
            
            logger.log_metrics(metrics)
            logger.flush_metrics()
            
            # Check if metrics file exists and has content
            metrics_file = logger.metrics_file
            assert metrics_file.exists()
            
            # Read and verify content
            with open(metrics_file, 'r') as f:
                content = f.read().strip()
                assert len(content) > 0
                
                # Should be valid JSON
                import json
                metrics_data = json.loads(content)
                assert metrics_data['train_loss'] == 0.5
                assert metrics_data['epoch'] == 1
    
    def test_event_logging(self):
        """Test event logging."""
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = DistributedLogger(temp_dir, rank=0, world_size=1)
            
            # Log an event
            logger.log_event("test_event", "This is a test message", extra_data="test")
            
            # Check if events file exists
            events_file = logger.events_file
            assert events_file.exists()
            
            # Read and verify content
            with open(events_file, 'r') as f:
                content = f.read().strip()
                assert "test_event" in content
                assert "This is a test message" in content


class TestDistributedTrainingMonitor:
    """Test distributed training monitor."""
    
    def test_monitor_initialization(self):
        """Test training monitor initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            monitor = DistributedTrainingMonitor(
                log_dir=temp_dir,
                rank=0,
                world_size=2,
                monitoring_interval=0.01
            )
            
            assert monitor.rank == 0
            assert monitor.world_size == 2
            assert monitor.is_main_process is True
            assert isinstance(monitor.logger, DistributedLogger)
            assert isinstance(monitor.resource_monitor, ResourceMonitor)
    
    def test_training_step_logging(self):
        """Test training step logging."""
        with tempfile.TemporaryDirectory() as temp_dir:
            monitor = DistributedTrainingMonitor(
                log_dir=temp_dir,
                rank=0,
                world_size=1,
                monitoring_interval=0.01
            )
            
            # Start monitoring
            monitor.start_monitoring()
            
            # Log a training step
            monitor.log_training_step(
                step=100,
                epoch=1,
                loss=0.5,
                learning_rate=1e-4,
                batch_size=32,
                step_time=0.1
            )
            
            # Stop monitoring
            monitor.stop_monitoring()
            
            # Check if metrics were logged
            assert len(monitor.step_times) == 1
            assert monitor.step_times[0] == 0.1
    
    def test_communication_profiling(self):
        """Test communication profiling integration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            monitor = DistributedTrainingMonitor(
                log_dir=temp_dir,
                rank=0,
                world_size=1
            )
            
            # Profile a communication operation
            op_id = monitor.start_communication_profiling("allreduce", 1024)
            time.sleep(0.001)
            monitor.end_communication_profiling(op_id)
            
            # Get statistics
            stats = monitor.get_training_statistics()
            
            assert 'communication_stats' in stats
            comm_stats = stats['communication_stats']
            
            if 'allreduce' in comm_stats:
                assert comm_stats['allreduce']['count'] == 1


class TestDistributedTrainingIntegration:
    """Integration tests for distributed training."""
    
    @patch('torch.distributed.is_initialized')
    def test_single_process_training(self, mock_is_init):
        """Test distributed training with single process."""
        mock_is_init.return_value = False
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create mock model and dataset
            model = MockModel()
            dataset = MockDataset(50)
            
            # Create configurations
            training_config = TrainingConfig(
                num_epochs=1,
                batch_size=4,
                output_dir=temp_dir,
                logging_steps=10
            )
            
            distributed_config = DistributedConfig(
                world_size=1,
                rank=0,
                local_rank=0
            )
            
            # Setup distributed training
            trainer = setup_distributed_training(
                model=model,
                train_dataset=dataset,
                training_config=training_config,
                distributed_config=distributed_config
            )
            
            assert isinstance(trainer, DistributedTrainer)
            assert trainer.distributed_config.world_size == 1
    
    def test_utility_functions(self):
        """Test utility functions."""
        # Test without distributed initialization
        assert get_world_size() >= 1
        assert get_rank() >= 0
        
        # is_main_process should work
        main_process = is_main_process()
        assert isinstance(main_process, bool)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])