#!/usr/bin/env python3
"""
Comprehensive example demonstrating TabGPT distributed training.

This example shows how to:
1. Setup distributed training across multiple GPUs/nodes
2. Configure data and model parallelism
3. Monitor training performance and communication
4. Handle fault tolerance and checkpointing
5. Optimize for large-scale training scenarios
"""

import os
import sys
import argparse
import time
import json
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tabgpt.training.distributed import (
    DistributedConfig, DistributedTrainer, setup_distributed_training,
    launch_distributed_training, get_world_size, get_rank, is_main_process
)
from tabgpt.training.distributed_monitoring import DistributedTrainingMonitor
from tabgpt.training.trainer import TrainingConfig
from tabgpt.models.modeling_tabgpt import TabGPTForPreTraining
from tabgpt.models.configuration_tabgpt import TabGPTConfig


class SyntheticTabularDataset(Dataset):
    """Synthetic tabular dataset for distributed training demonstration."""
    
    def __init__(
        self,
        num_samples: int = 10000,
        num_features: int = 50,
        num_categorical: int = 10,
        vocab_size: int = 1000,
        max_length: int = 128
    ):
        self.num_samples = num_samples
        self.num_features = num_features
        self.num_categorical = num_categorical
        self.vocab_size = vocab_size
        self.max_length = max_length
        
        # Generate synthetic data
        self._generate_data()
    
    def _generate_data(self):
        """Generate synthetic tabular data."""
        np.random.seed(42)  # For reproducibility
        
        # Generate numerical features
        self.numerical_data = np.random.randn(self.num_samples, self.num_features - self.num_categorical)
        
        # Generate categorical features
        self.categorical_data = np.random.randint(
            1, self.vocab_size, 
            size=(self.num_samples, self.num_categorical)
        )
        
        # Generate sequence lengths
        self.sequence_lengths = np.random.randint(
            self.max_length // 2, self.max_length, 
            size=self.num_samples
        )
        
        # Generate labels for different objectives
        self.mcm_labels = np.random.randint(0, self.vocab_size, size=(self.num_samples, self.max_length))
        self.classification_labels = np.random.randint(0, 2, size=self.num_samples)
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Create input sequence (simplified tokenization)
        seq_len = self.sequence_lengths[idx]
        
        # Combine numerical and categorical features into sequence
        input_ids = np.zeros(self.max_length, dtype=np.int64)
        attention_mask = np.zeros(self.max_length, dtype=np.int64)
        
        # Fill with data up to sequence length
        input_ids[:seq_len] = np.random.randint(1, self.vocab_size, seq_len)
        attention_mask[:seq_len] = 1
        
        # Create labels for different objectives
        mcm_labels = self.mcm_labels[idx].copy()
        mcm_labels[seq_len:] = -100  # Ignore padding in loss
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'mcm_labels': torch.tensor(mcm_labels, dtype=torch.long),
            'classification_labels': torch.tensor(self.classification_labels[idx], dtype=torch.long),
            'numerical_features': torch.tensor(self.numerical_data[idx], dtype=torch.float32),
            'categorical_features': torch.tensor(self.categorical_data[idx], dtype=torch.long)
        }


class DistributedTrainingDemo:
    """Comprehensive distributed training demonstration."""
    
    def __init__(self, args):
        self.args = args
        self.rank = get_rank()
        self.world_size = get_world_size()
        self.is_main_process = is_main_process()
        
        # Setup logging
        self.setup_logging()
        
        # Initialize monitoring
        if args.enable_monitoring:
            self.monitor = DistributedTrainingMonitor(
                log_dir=args.log_dir,
                rank=self.rank,
                world_size=self.world_size,
                monitoring_interval=args.monitoring_interval
            )
        else:
            self.monitor = None
    
    def setup_logging(self):
        """Setup logging for distributed training."""
        import logging
        
        # Create log directory
        log_dir = Path(self.args.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format=f'[Rank {self.rank}] %(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / f'training_rank_{self.rank}.log'),
                logging.StreamHandler() if self.is_main_process else logging.NullHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
    
    def create_model_and_config(self) -> tuple:
        """Create TabGPT model and configuration."""
        self.logger.info("Creating TabGPT model and configuration...")
        
        # Model configuration
        config = TabGPTConfig(
            vocab_size=self.args.vocab_size,
            hidden_size=self.args.hidden_size,
            num_hidden_layers=self.args.num_layers,
            num_attention_heads=self.args.num_attention_heads,
            intermediate_size=self.args.intermediate_size,
            max_position_embeddings=self.args.max_length,
            dropout=self.args.dropout,
            attention_dropout=self.args.attention_dropout
        )
        
        # Create model
        model = TabGPTForPreTraining(config)
        
        # Log model info
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        self.logger.info(f"Model created with {total_params:,} total parameters")
        self.logger.info(f"Trainable parameters: {trainable_params:,}")
        
        return model, config
    
    def create_datasets(self) -> tuple:
        """Create training and evaluation datasets."""
        self.logger.info("Creating datasets...")
        
        # Training dataset
        train_dataset = SyntheticTabularDataset(
            num_samples=self.args.train_samples,
            num_features=self.args.num_features,
            num_categorical=self.args.num_categorical,
            vocab_size=self.args.vocab_size,
            max_length=self.args.max_length
        )
        
        # Evaluation dataset
        eval_dataset = SyntheticTabularDataset(
            num_samples=self.args.eval_samples,
            num_features=self.args.num_features,
            num_categorical=self.args.num_categorical,
            vocab_size=self.args.vocab_size,
            max_length=self.args.max_length
        )
        
        self.logger.info(f"Training dataset: {len(train_dataset)} samples")
        self.logger.info(f"Evaluation dataset: {len(eval_dataset)} samples")
        
        return train_dataset, eval_dataset
    
    def create_training_configs(self) -> tuple:
        """Create training and distributed configurations."""
        self.logger.info("Creating training configurations...")
        
        # Training configuration
        training_config = TrainingConfig(
            num_epochs=self.args.num_epochs,
            batch_size=self.args.batch_size,
            learning_rate=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
            warmup_steps=self.args.warmup_steps,
            max_grad_norm=self.args.max_grad_norm,
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            use_mixed_precision=self.args.use_mixed_precision,
            lr_scheduler_type=self.args.lr_scheduler_type,
            save_steps=self.args.save_steps,
            eval_steps=self.args.eval_steps,
            logging_steps=self.args.logging_steps,
            output_dir=self.args.output_dir,
            dataloader_num_workers=self.args.num_workers
        )
        
        # Distributed configuration
        distributed_config = DistributedConfig(
            backend=self.args.backend,
            world_size=self.world_size,
            rank=self.rank,
            local_rank=int(os.environ.get("LOCAL_RANK", 0)),
            use_data_parallel=self.args.use_data_parallel,
            use_model_parallel=self.args.use_model_parallel,
            pipeline_parallel_size=self.args.pipeline_parallel_size,
            tensor_parallel_size=self.args.tensor_parallel_size,
            gradient_sync_frequency=self.args.gradient_sync_frequency,
            use_gradient_compression=self.args.use_gradient_compression,
            bucket_size_mb=self.args.bucket_size_mb,
            enable_fault_tolerance=self.args.enable_fault_tolerance,
            monitor_communication=self.args.monitor_communication
        )
        
        return training_config, distributed_config
    
    def run_training(self):
        """Run distributed training."""
        try:
            # Start monitoring
            if self.monitor:
                self.monitor.start_monitoring()
            
            self.logger.info("Starting distributed training...")
            self.logger.info(f"World size: {self.world_size}")
            self.logger.info(f"Rank: {self.rank}")
            self.logger.info(f"Local rank: {os.environ.get('LOCAL_RANK', 0)}")
            
            # Create model and datasets
            model, model_config = self.create_model_and_config()
            train_dataset, eval_dataset = self.create_datasets()
            training_config, distributed_config = self.create_training_configs()
            
            # Setup distributed training
            trainer = setup_distributed_training(
                model=model,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                training_config=training_config,
                distributed_config=distributed_config
            )
            
            # Log training setup
            if self.is_main_process:
                self.logger.info("Training setup completed")
                self.logger.info(f"Effective batch size: {self.args.batch_size * self.world_size}")
                self.logger.info(f"Gradient accumulation steps: {self.args.gradient_accumulation_steps}")
                self.logger.info(f"Total optimization steps: {trainer._get_num_training_steps()}")
            
            # Custom training loop with monitoring
            if self.monitor:
                trainer = self._wrap_trainer_with_monitoring(trainer)
            
            # Run training
            start_time = time.time()
            training_result = trainer.train()
            training_time = time.time() - start_time
            
            # Log results
            if self.is_main_process:
                self.logger.info(f"Training completed in {training_time:.2f} seconds")
                self.logger.info(f"Final training loss: {training_result.get('train_loss', 'N/A')}")
                
                # Save training results
                results_file = Path(self.args.output_dir) / "training_results.json"
                with open(results_file, 'w') as f:
                    json.dump({
                        'training_time': training_time,
                        'world_size': self.world_size,
                        'effective_batch_size': self.args.batch_size * self.world_size,
                        **training_result
                    }, f, indent=2, default=str)
                
                self.logger.info(f"Training results saved to {results_file}")
            
            return training_result
            
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            raise
        finally:
            # Stop monitoring
            if self.monitor:
                self.monitor.stop_monitoring()
            
            # Cleanup
            trainer.cleanup()
    
    def _wrap_trainer_with_monitoring(self, trainer):
        """Wrap trainer with monitoring capabilities."""
        original_train_epoch = trainer._train_epoch
        
        def monitored_train_epoch():
            epoch_start_time = time.time()
            
            # Log epoch start
            if self.monitor:
                self.monitor.log_event(
                    "epoch_start",
                    f"Starting epoch {trainer.state.epoch}",
                    epoch=trainer.state.epoch
                )
            
            # Run original epoch training
            epoch_loss = original_train_epoch()
            
            # Log epoch end
            epoch_time = time.time() - epoch_start_time
            if self.monitor:
                self.monitor.log_event(
                    "epoch_end",
                    f"Completed epoch {trainer.state.epoch} in {epoch_time:.2f}s",
                    epoch=trainer.state.epoch,
                    epoch_time=epoch_time,
                    epoch_loss=epoch_loss
                )
            
            return epoch_loss
        
        # Replace method
        trainer._train_epoch = monitored_train_epoch
        
        # Wrap step logging
        original_log_step = trainer._log_training_step
        
        def monitored_log_step():
            if self.monitor:
                self.monitor.log_training_step(
                    step=trainer.state.global_step,
                    epoch=trainer.state.epoch,
                    loss=trainer.state.train_loss,
                    learning_rate=trainer.state.learning_rate,
                    batch_size=self.args.batch_size,
                    step_time=time.time() - trainer.state.step_start_time
                )
            
            # Call original logging
            original_log_step()
        
        trainer._log_training_step = monitored_log_step
        
        return trainer
    
    def run_performance_analysis(self):
        """Run performance analysis and scaling tests."""
        if not self.is_main_process:
            return
        
        self.logger.info("Running performance analysis...")
        
        # Collect training statistics
        if self.monitor:
            stats = self.monitor.get_training_statistics()
            
            # Save performance analysis
            analysis_file = Path(self.args.output_dir) / "performance_analysis.json"
            with open(analysis_file, 'w') as f:
                json.dump(stats, f, indent=2, default=str)
            
            self.logger.info(f"Performance analysis saved to {analysis_file}")
            
            # Log key metrics
            if 'step_timing' in stats:
                step_stats = stats['step_timing']
                self.logger.info(f"Average step time: {step_stats.get('avg_step_time', 0):.4f}s")
                self.logger.info(f"Steps per second: {1.0 / step_stats.get('avg_step_time', 1):.2f}")
            
            if 'communication_stats' in stats:
                comm_stats = stats['communication_stats']
                if 'overall' in comm_stats:
                    overall = comm_stats['overall']
                    self.logger.info(f"Total communication time: {overall.get('total_communication_time', 0):.2f}s")
                    self.logger.info(f"Communication efficiency: {overall.get('communication_efficiency', 0):.2f} MB/s")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="TabGPT Distributed Training Example")
    
    # Model configuration
    parser.add_argument("--vocab-size", type=int, default=30000, help="Vocabulary size")
    parser.add_argument("--hidden-size", type=int, default=768, help="Hidden size")
    parser.add_argument("--num-layers", type=int, default=12, help="Number of layers")
    parser.add_argument("--num-attention-heads", type=int, default=12, help="Number of attention heads")
    parser.add_argument("--intermediate-size", type=int, default=3072, help="Intermediate size")
    parser.add_argument("--max-length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout probability")
    parser.add_argument("--attention-dropout", type=float, default=0.1, help="Attention dropout")
    
    # Dataset configuration
    parser.add_argument("--train-samples", type=int, default=10000, help="Number of training samples")
    parser.add_argument("--eval-samples", type=int, default=1000, help="Number of evaluation samples")
    parser.add_argument("--num-features", type=int, default=50, help="Number of features")
    parser.add_argument("--num-categorical", type=int, default=10, help="Number of categorical features")
    
    # Training configuration
    parser.add_argument("--num-epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size per device")
    parser.add_argument("--learning-rate", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--warmup-steps", type=int, default=1000, help="Warmup steps")
    parser.add_argument("--max-grad-norm", type=float, default=1.0, help="Max gradient norm")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--use-mixed-precision", action="store_true", help="Use mixed precision training")
    parser.add_argument("--lr-scheduler-type", type=str, default="cosine", help="Learning rate scheduler type")
    
    # Distributed configuration
    parser.add_argument("--backend", type=str, default="nccl", help="Distributed backend")
    parser.add_argument("--use-data-parallel", action="store_true", default=True, help="Use data parallelism")
    parser.add_argument("--use-model-parallel", action="store_true", help="Use model parallelism")
    parser.add_argument("--pipeline-parallel-size", type=int, default=1, help="Pipeline parallel size")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--gradient-sync-frequency", type=int, default=1, help="Gradient sync frequency")
    parser.add_argument("--use-gradient-compression", action="store_true", help="Use gradient compression")
    parser.add_argument("--bucket-size-mb", type=int, default=25, help="DDP bucket size in MB")
    parser.add_argument("--enable-fault-tolerance", action="store_true", help="Enable fault tolerance")
    parser.add_argument("--monitor-communication", action="store_true", default=True, help="Monitor communication")
    
    # Logging and monitoring
    parser.add_argument("--save-steps", type=int, default=1000, help="Save checkpoint every N steps")
    parser.add_argument("--eval-steps", type=int, default=500, help="Evaluate every N steps")
    parser.add_argument("--logging-steps", type=int, default=100, help="Log every N steps")
    parser.add_argument("--output-dir", type=str, default="./distributed_training_output", help="Output directory")
    parser.add_argument("--log-dir", type=str, default="./distributed_training_logs", help="Log directory")
    parser.add_argument("--enable-monitoring", action="store_true", default=True, help="Enable monitoring")
    parser.add_argument("--monitoring-interval", type=float, default=1.0, help="Monitoring interval in seconds")
    
    # System configuration
    parser.add_argument("--num-workers", type=int, default=4, help="Number of data loader workers")
    
    return parser.parse_args()


def main():
    """Main function for distributed training example."""
    args = parse_arguments()
    
    # Create output directories
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize demo
    demo = DistributedTrainingDemo(args)
    
    try:
        # Run training
        training_result = demo.run_training()
        
        # Run performance analysis
        demo.run_performance_analysis()
        
        if demo.is_main_process:
            print("\\n" + "="*60)
            print("DISTRIBUTED TRAINING COMPLETED SUCCESSFULLY!")
            print("="*60)
            print(f"World size: {demo.world_size}")
            print(f"Training samples: {args.train_samples}")
            print(f"Effective batch size: {args.batch_size * demo.world_size}")
            print(f"Final loss: {training_result.get('train_loss', 'N/A')}")
            print(f"Results saved to: {args.output_dir}")
            print("="*60)
        
    except Exception as e:
        demo.logger.error(f"Training failed: {e}")
        raise


def launch_multi_gpu_training():
    """Launch multi-GPU training using torch.distributed.launch."""
    import subprocess
    import sys
    
    # Get number of available GPUs
    num_gpus = torch.cuda.device_count()
    
    if num_gpus < 2:
        print("Multi-GPU training requires at least 2 GPUs")
        print(f"Found {num_gpus} GPU(s)")
        print("Running single GPU training instead...")
        main()
        return
    
    # Launch distributed training
    cmd = [
        sys.executable, "-m", "torch.distributed.launch",
        f"--nproc_per_node={num_gpus}",
        "--use_env",
        __file__,
        "--use-data-parallel",
        "--enable-monitoring",
        "--monitor-communication",
        "--num-epochs=2",
        "--batch-size=16",
        "--train-samples=5000",
        "--eval-samples=500"
    ]
    
    print(f"Launching distributed training with {num_gpus} GPUs...")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True)
        print("Distributed training completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Distributed training failed: {e}")
        raise


if __name__ == "__main__":
    # Check if this is being run as a distributed training script
    if "LOCAL_RANK" in os.environ:
        # This is a distributed worker process
        main()
    else:
        # This is the main launcher process
        import argparse
        
        launcher_parser = argparse.ArgumentParser(description="Launch distributed training")
        launcher_parser.add_argument("--launch-multi-gpu", action="store_true", 
                                   help="Launch multi-GPU training automatically")
        launcher_args, remaining_args = launcher_parser.parse_known_args()
        
        if launcher_args.launch_multi_gpu:
            launch_multi_gpu_training()
        else:
            # Single process training
            main()