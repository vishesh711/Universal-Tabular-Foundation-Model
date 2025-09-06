"""Distributed training support for TabGPT."""

import os
import logging
import time
import socket
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
from dataclasses import dataclass, field
from pathlib import Path
import warnings

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.cuda.amp import GradScaler, autocast
import numpy as np

from .trainer import TrainingConfig, TrainingState, MultiObjectiveTrainer
from ..models.modeling_tabgpt import TabGPTForPreTraining
from ..utils.exceptions import TrainingError

logger = logging.getLogger(__name__)


@dataclass
class DistributedConfig:
    """Configuration for distributed training."""
    
    # Distributed setup
    backend: str = "nccl"  # nccl, gloo, mpi
    init_method: str = "env://"  # env://, tcp://, file://
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0
    
    # Multi-GPU settings
    device_ids: Optional[List[int]] = None
    output_device: Optional[int] = None
    
    # Data parallelism
    use_data_parallel: bool = True
    find_unused_parameters: bool = False
    
    # Model parallelism
    use_model_parallel: bool = False
    pipeline_parallel_size: int = 1
    tensor_parallel_size: int = 1
    
    # Gradient synchronization
    gradient_sync_frequency: int = 1
    use_gradient_compression: bool = False
    compression_ratio: float = 0.1
    
    # Communication optimization
    bucket_size_mb: int = 25
    use_static_graph: bool = False
    
    # Fault tolerance
    enable_fault_tolerance: bool = False
    checkpoint_frequency: int = 1000
    
    # Monitoring
    monitor_communication: bool = True
    log_communication_stats: bool = False
    
    def __post_init__(self):
        """Post-initialization validation."""
        if self.world_size <= 0:
            raise ValueError("world_size must be positive")
        
        if self.rank < 0 or self.rank >= self.world_size:
            raise ValueError(f"rank must be in [0, {self.world_size})")
        
        if self.local_rank < 0:
            raise ValueError("local_rank must be non-negative")
        
        # Set device IDs if not provided
        if self.device_ids is None and torch.cuda.is_available():
            self.device_ids = [self.local_rank]
        
        # Set output device
        if self.output_device is None and self.device_ids:
            self.output_device = self.device_ids[0]


class DistributedManager:
    """Manager for distributed training setup and coordination."""
    
    def __init__(self, config: DistributedConfig):
        self.config = config
        self.is_initialized = False
        self.process_group = None
        
        # Communication statistics
        self.comm_stats = {
            'allreduce_calls': 0,
            'allreduce_time': 0.0,
            'broadcast_calls': 0,
            'broadcast_time': 0.0,
            'total_bytes_communicated': 0
        }
    
    def initialize(self):
        """Initialize distributed training environment."""
        if self.is_initialized:
            logger.warning("Distributed training already initialized")
            return
        
        try:
            # Initialize process group
            if not dist.is_initialized():
                logger.info(f"Initializing distributed training with backend: {self.config.backend}")
                
                # Set environment variables if not set
                if "RANK" not in os.environ:
                    os.environ["RANK"] = str(self.config.rank)
                if "WORLD_SIZE" not in os.environ:
                    os.environ["WORLD_SIZE"] = str(self.config.world_size)
                if "LOCAL_RANK" not in os.environ:
                    os.environ["LOCAL_RANK"] = str(self.config.local_rank)
                
                # Initialize process group
                dist.init_process_group(
                    backend=self.config.backend,
                    init_method=self.config.init_method,
                    world_size=self.config.world_size,
                    rank=self.config.rank
                )
                
                self.process_group = dist.group.WORLD
            
            # Set CUDA device
            if torch.cuda.is_available() and self.config.device_ids:
                torch.cuda.set_device(self.config.local_rank)
                logger.info(f"Set CUDA device to: {self.config.local_rank}")
            
            self.is_initialized = True
            
            logger.info(
                f"Distributed training initialized: "
                f"rank={self.config.rank}, "
                f"world_size={self.config.world_size}, "
                f"local_rank={self.config.local_rank}"
            )
            
        except Exception as e:
            logger.error(f"Failed to initialize distributed training: {e}")
            raise TrainingError(f"Distributed initialization failed: {e}")
    
    def cleanup(self):
        """Cleanup distributed training environment."""
        if self.is_initialized and dist.is_initialized():
            try:
                dist.destroy_process_group()
                logger.info("Distributed training cleanup completed")
            except Exception as e:
                logger.warning(f"Error during distributed cleanup: {e}")
        
        self.is_initialized = False
    
    def barrier(self):
        """Synchronize all processes."""
        if self.is_initialized:
            dist.barrier()
    
    def all_reduce(self, tensor: torch.Tensor, op=dist.ReduceOp.SUM) -> torch.Tensor:
        """All-reduce operation across all processes."""
        if not self.is_initialized or self.config.world_size == 1:
            return tensor
        
        start_time = time.time()
        
        # Clone tensor to avoid in-place modification
        reduced_tensor = tensor.clone()
        
        # Perform all-reduce
        dist.all_reduce(reduced_tensor, op=op)
        
        # Update statistics
        if self.config.monitor_communication:
            self.comm_stats['allreduce_calls'] += 1
            self.comm_stats['allreduce_time'] += time.time() - start_time
            self.comm_stats['total_bytes_communicated'] += tensor.numel() * tensor.element_size()
        
        return reduced_tensor
    
    def broadcast(self, tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
        """Broadcast tensor from source rank to all processes."""
        if not self.is_initialized or self.config.world_size == 1:
            return tensor
        
        start_time = time.time()
        
        # Perform broadcast
        dist.broadcast(tensor, src=src)
        
        # Update statistics
        if self.config.monitor_communication:
            self.comm_stats['broadcast_calls'] += 1
            self.comm_stats['broadcast_time'] += time.time() - start_time
            if self.config.rank == src:
                self.comm_stats['total_bytes_communicated'] += tensor.numel() * tensor.element_size()
        
        return tensor
    
    def all_gather(self, tensor: torch.Tensor) -> List[torch.Tensor]:
        """Gather tensors from all processes."""
        if not self.is_initialized or self.config.world_size == 1:
            return [tensor]
        
        # Prepare output list
        tensor_list = [torch.zeros_like(tensor) for _ in range(self.config.world_size)]
        
        # Perform all-gather
        dist.all_gather(tensor_list, tensor)
        
        return tensor_list
    
    def reduce_dict(self, input_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Reduce dictionary of tensors across all processes."""
        if not self.is_initialized or self.config.world_size == 1:
            return input_dict
        
        reduced_dict = {}
        for key, tensor in input_dict.items():
            if isinstance(tensor, torch.Tensor):
                reduced_dict[key] = self.all_reduce(tensor) / self.config.world_size
            else:
                reduced_dict[key] = tensor
        
        return reduced_dict
    
    def get_communication_stats(self) -> Dict[str, Any]:
        """Get communication statistics."""
        stats = self.comm_stats.copy()
        
        if stats['allreduce_calls'] > 0:
            stats['avg_allreduce_time'] = stats['allreduce_time'] / stats['allreduce_calls']
        
        if stats['broadcast_calls'] > 0:
            stats['avg_broadcast_time'] = stats['broadcast_time'] / stats['broadcast_calls']
        
        stats['total_communication_time'] = stats['allreduce_time'] + stats['broadcast_time']
        stats['bandwidth_mbps'] = (
            stats['total_bytes_communicated'] / (1024 * 1024) / 
            max(stats['total_communication_time'], 1e-6)
        )
        
        return stats
    
    def is_main_process(self) -> bool:
        """Check if this is the main process (rank 0)."""
        return self.config.rank == 0
    
    def is_local_main_process(self) -> bool:
        """Check if this is the local main process (local_rank 0)."""
        return self.config.local_rank == 0


class DataSharding:
    """Efficient data sharding for distributed training."""
    
    def __init__(self, distributed_config: DistributedConfig):
        self.config = distributed_config
    
    def create_distributed_sampler(
        self,
        dataset,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int = 0
    ) -> DistributedSampler:
        """Create distributed sampler for dataset."""
        return DistributedSampler(
            dataset,
            num_replicas=self.config.world_size,
            rank=self.config.rank,
            shuffle=shuffle,
            drop_last=drop_last,
            seed=seed
        )
    
    def create_distributed_dataloader(
        self,
        dataset,
        batch_size: int,
        shuffle: bool = True,
        num_workers: int = 4,
        pin_memory: bool = True,
        drop_last: bool = False,
        **kwargs
    ) -> DataLoader:
        """Create distributed data loader."""
        sampler = self.create_distributed_sampler(
            dataset, shuffle=shuffle, drop_last=drop_last
        )
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            **kwargs
        )
    
    def shard_dataset_by_size(
        self,
        dataset,
        shard_size: int
    ) -> List[Any]:
        """Shard dataset into chunks of specified size."""
        total_size = len(dataset)
        num_shards = (total_size + shard_size - 1) // shard_size
        
        shards = []
        for i in range(num_shards):
            start_idx = i * shard_size
            end_idx = min((i + 1) * shard_size, total_size)
            
            # Create subset
            indices = list(range(start_idx, end_idx))
            shard = torch.utils.data.Subset(dataset, indices)
            shards.append(shard)
        
        return shards
    
    def balance_shards(self, shards: List[Any]) -> List[Any]:
        """Balance shards across processes to ensure even distribution."""
        if len(shards) < self.config.world_size:
            # Pad with empty shards if needed
            while len(shards) < self.config.world_size:
                shards.append(torch.utils.data.Subset(shards[0].dataset, []))
        
        # Distribute shards round-robin
        balanced_shards = []
        for i in range(self.config.rank, len(shards), self.config.world_size):
            balanced_shards.append(shards[i])
        
        return balanced_shards


class GradientSynchronization:
    """Advanced gradient synchronization for distributed training."""
    
    def __init__(self, distributed_config: DistributedConfig):
        self.config = distributed_config
        self.gradient_buffer = {}
        self.compression_enabled = distributed_config.use_gradient_compression
        
    def synchronize_gradients(
        self,
        model: nn.Module,
        distributed_manager: DistributedManager
    ):
        """Synchronize gradients across all processes."""
        if not distributed_manager.is_initialized or self.config.world_size == 1:
            return
        
        # Collect gradients
        gradients = []
        for param in model.parameters():
            if param.grad is not None:
                gradients.append(param.grad.data)
        
        if not gradients:
            return
        
        # Apply gradient compression if enabled
        if self.compression_enabled:
            gradients = self._compress_gradients(gradients)
        
        # Synchronize gradients
        for grad in gradients:
            distributed_manager.all_reduce(grad)
            grad.div_(self.config.world_size)
        
        # Decompress gradients if needed
        if self.compression_enabled:
            self._decompress_gradients(gradients, model)
    
    def _compress_gradients(self, gradients: List[torch.Tensor]) -> List[torch.Tensor]:
        """Compress gradients to reduce communication overhead."""
        compressed_gradients = []
        
        for grad in gradients:
            if grad.numel() > 1000:  # Only compress large gradients
                # Top-k compression
                k = max(1, int(grad.numel() * self.config.compression_ratio))
                
                # Flatten and get top-k values
                flat_grad = grad.flatten()
                _, top_k_indices = torch.topk(torch.abs(flat_grad), k)
                
                # Create sparse representation
                compressed_grad = torch.zeros_like(flat_grad)
                compressed_grad[top_k_indices] = flat_grad[top_k_indices]
                compressed_grad = compressed_grad.view_as(grad)
                
                compressed_gradients.append(compressed_grad)
            else:
                compressed_gradients.append(grad)
        
        return compressed_gradients
    
    def _decompress_gradients(
        self,
        compressed_gradients: List[torch.Tensor],
        model: nn.Module
    ):
        """Decompress gradients and update model parameters."""
        param_iter = iter(model.parameters())
        
        for compressed_grad in compressed_gradients:
            param = next(param_iter)
            if param.grad is not None:
                param.grad.data.copy_(compressed_grad)
    
    def accumulate_gradients(
        self,
        model: nn.Module,
        accumulation_steps: int
    ):
        """Accumulate gradients over multiple steps before synchronization."""
        # Scale gradients by accumulation steps
        for param in model.parameters():
            if param.grad is not None:
                param.grad.data.div_(accumulation_steps)


class ModelParallelism:
    """Model parallelism support for large models."""
    
    def __init__(self, distributed_config: DistributedConfig):
        self.config = distributed_config
        self.pipeline_stages = []
        self.tensor_parallel_groups = []
    
    def setup_pipeline_parallel(
        self,
        model: nn.Module,
        num_stages: int
    ) -> nn.Module:
        """Setup pipeline parallelism for model."""
        if num_stages <= 1:
            return model
        
        # Split model into pipeline stages
        layers = list(model.children())
        layers_per_stage = len(layers) // num_stages
        
        stages = []
        for i in range(num_stages):
            start_idx = i * layers_per_stage
            end_idx = (i + 1) * layers_per_stage if i < num_stages - 1 else len(layers)
            
            stage_layers = layers[start_idx:end_idx]
            stage = nn.Sequential(*stage_layers)
            stages.append(stage)
        
        # Assign stages to devices
        current_stage = self.config.rank % num_stages
        model_stage = stages[current_stage]
        
        # Move to appropriate device
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{self.config.local_rank}")
            model_stage = model_stage.to(device)
        
        self.pipeline_stages = stages
        return model_stage
    
    def setup_tensor_parallel(
        self,
        model: nn.Module,
        tensor_parallel_size: int
    ) -> nn.Module:
        """Setup tensor parallelism for model."""
        if tensor_parallel_size <= 1:
            return model
        
        # This is a simplified implementation
        # In practice, you would need to split specific layers (e.g., attention, MLP)
        # across multiple devices and handle communication between them
        
        logger.warning("Tensor parallelism is not fully implemented in this example")
        return model


class DistributedTrainer(MultiObjectiveTrainer):
    """Distributed trainer extending the base trainer."""
    
    def __init__(
        self,
        model: TabGPTForPreTraining,
        config: TrainingConfig,
        distributed_config: DistributedConfig,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
        **kwargs
    ):
        # Initialize distributed manager
        self.distributed_manager = DistributedManager(distributed_config)
        self.distributed_config = distributed_config
        
        # Initialize distributed training
        self.distributed_manager.initialize()
        
        # Setup data sharding
        self.data_sharding = DataSharding(distributed_config)
        
        # Setup gradient synchronization
        self.gradient_sync = GradientSynchronization(distributed_config)
        
        # Setup model parallelism
        self.model_parallelism = ModelParallelism(distributed_config)
        
        # Apply model parallelism if configured
        if distributed_config.use_model_parallel:
            if distributed_config.pipeline_parallel_size > 1:
                model = self.model_parallelism.setup_pipeline_parallel(
                    model, distributed_config.pipeline_parallel_size
                )
            
            if distributed_config.tensor_parallel_size > 1:
                model = self.model_parallelism.setup_tensor_parallel(
                    model, distributed_config.tensor_parallel_size
                )
        
        # Initialize base trainer
        super().__init__(
            model=model,
            config=config,
            train_dataloader=train_dataloader,
            eval_dataloader=eval_dataloader,
            **kwargs
        )
        
        # Wrap model with DistributedDataParallel
        if distributed_config.use_data_parallel and distributed_config.world_size > 1:
            self.model = self._wrap_model_ddp()
        
        # Synchronize model parameters across processes
        self._sync_model_parameters()
    
    def _wrap_model_ddp(self) -> nn.Module:
        """Wrap model with DistributedDataParallel."""
        ddp_kwargs = {
            'device_ids': self.distributed_config.device_ids,
            'output_device': self.distributed_config.output_device,
            'find_unused_parameters': self.distributed_config.find_unused_parameters,
        }
        
        # Set bucket size for gradient communication
        if self.distributed_config.bucket_size_mb > 0:
            ddp_kwargs['bucket_cap_mb'] = self.distributed_config.bucket_size_mb
        
        # Enable static graph optimization if configured
        if self.distributed_config.use_static_graph:
            ddp_kwargs['static_graph'] = True
        
        try:
            wrapped_model = DDP(self.model, **ddp_kwargs)
            logger.info("Model wrapped with DistributedDataParallel")
            return wrapped_model
        except Exception as e:
            logger.error(f"Failed to wrap model with DDP: {e}")
            raise TrainingError(f"DDP wrapping failed: {e}")
    
    def _sync_model_parameters(self):
        """Synchronize model parameters across all processes."""
        if self.distributed_manager.is_initialized:
            for param in self.model.parameters():
                self.distributed_manager.broadcast(param.data, src=0)
            
            logger.info("Model parameters synchronized across processes")
    
    def train(self) -> Dict[str, Any]:
        """Distributed training loop."""
        if self.distributed_manager.is_main_process():
            logger.info("***** Running distributed training *****")
            logger.info(f"  World size = {self.distributed_config.world_size}")
            logger.info(f"  Num processes = {self.distributed_config.world_size}")
            logger.info(f"  Distributed backend = {self.distributed_config.backend}")
        
        try:
            # Run training
            result = super().train()
            
            # Gather results from all processes
            if self.distributed_manager.is_initialized:
                result = self._gather_training_results(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Distributed training failed: {e}")
            raise
        finally:
            # Cleanup distributed training
            self.cleanup()
    
    def _train_epoch(self) -> float:
        """Distributed training epoch."""
        # Set epoch for distributed sampler
        if hasattr(self.train_dataloader.sampler, 'set_epoch'):
            self.train_dataloader.sampler.set_epoch(self.state.epoch)
        
        # Run epoch training
        epoch_loss = super()._train_epoch()
        
        # Synchronize epoch loss across processes
        if self.distributed_manager.is_initialized:
            loss_tensor = torch.tensor(epoch_loss, device=self.device)
            loss_tensor = self.distributed_manager.all_reduce(loss_tensor)
            epoch_loss = (loss_tensor / self.distributed_config.world_size).item()
        
        return epoch_loss
    
    def _compute_loss(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute loss with distributed considerations."""
        loss, losses = super()._compute_loss(batch)
        
        # Synchronize losses across processes if needed
        if (self.distributed_manager.is_initialized and 
            self.state.global_step % self.distributed_config.gradient_sync_frequency == 0):
            
            # Convert losses to tensors for synchronization
            loss_dict = {}
            for key, value in losses.items():
                if isinstance(value, (int, float)):
                    loss_dict[key] = torch.tensor(value, device=self.device)
            
            # Synchronize loss dictionary
            if loss_dict:
                synchronized_losses = self.distributed_manager.reduce_dict(loss_dict)
                losses = {key: tensor.item() for key, tensor in synchronized_losses.items()}
        
        return loss, losses
    
    def evaluate(self) -> Dict[str, float]:
        """Distributed evaluation."""
        # Run evaluation on each process
        local_metrics = super().evaluate()
        
        # Synchronize metrics across processes
        if self.distributed_manager.is_initialized and local_metrics:
            # Convert metrics to tensors
            metric_tensors = {}
            for key, value in local_metrics.items():
                if isinstance(value, (int, float)):
                    metric_tensors[key] = torch.tensor(value, device=self.device)
            
            # Synchronize metrics
            if metric_tensors:
                synchronized_metrics = self.distributed_manager.reduce_dict(metric_tensors)
                local_metrics = {key: tensor.item() for key, tensor in synchronized_metrics.items()}
        
        return local_metrics
    
    def _save_checkpoint(self):
        """Save checkpoint only from main process."""
        if self.distributed_manager.is_main_process():
            super()._save_checkpoint()
        
        # Synchronize all processes after checkpoint save
        self.distributed_manager.barrier()
    
    def _save_model(self):
        """Save final model only from main process."""
        if self.distributed_manager.is_main_process():
            super()._save_model()
        
        # Synchronize all processes after model save
        self.distributed_manager.barrier()
    
    def _gather_training_results(self, local_result: Dict[str, Any]) -> Dict[str, Any]:
        """Gather training results from all processes."""
        if not self.distributed_manager.is_initialized:
            return local_result
        
        # Convert results to tensors for gathering
        result_tensors = {}
        for key, value in local_result.items():
            if isinstance(value, (int, float)):
                result_tensors[key] = torch.tensor(value, device=self.device)
        
        # Gather results
        gathered_results = {}
        for key, tensor in result_tensors.items():
            gathered_tensors = self.distributed_manager.all_gather(tensor)
            
            if self.distributed_manager.is_main_process():
                # Compute statistics across processes
                values = [t.item() for t in gathered_tensors]
                gathered_results[key] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'values': values
                }
            else:
                gathered_results[key] = tensor.item()
        
        # Add communication statistics
        if self.distributed_manager.is_main_process():
            comm_stats = self.distributed_manager.get_communication_stats()
            gathered_results['communication_stats'] = comm_stats
        
        return gathered_results
    
    def cleanup(self):
        """Cleanup distributed training resources."""
        try:
            self.distributed_manager.cleanup()
        except Exception as e:
            logger.warning(f"Error during distributed cleanup: {e}")


def setup_distributed_training(
    model: TabGPTForPreTraining,
    train_dataset,
    eval_dataset=None,
    training_config: Optional[TrainingConfig] = None,
    distributed_config: Optional[DistributedConfig] = None,
    **kwargs
) -> DistributedTrainer:
    """
    Setup distributed training with automatic configuration.
    
    Args:
        model: TabGPT model for training
        train_dataset: Training dataset
        eval_dataset: Optional evaluation dataset
        training_config: Training configuration
        distributed_config: Distributed training configuration
        **kwargs: Additional arguments
        
    Returns:
        Configured distributed trainer
    """
    # Create default configurations if not provided
    if training_config is None:
        training_config = TrainingConfig()
    
    if distributed_config is None:
        # Auto-detect distributed configuration from environment
        distributed_config = DistributedConfig(
            world_size=int(os.environ.get("WORLD_SIZE", 1)),
            rank=int(os.environ.get("RANK", 0)),
            local_rank=int(os.environ.get("LOCAL_RANK", 0))
        )
    
    # Setup data sharding
    data_sharding = DataSharding(distributed_config)
    
    # Create distributed data loaders
    train_dataloader = data_sharding.create_distributed_dataloader(
        train_dataset,
        batch_size=training_config.batch_size,
        num_workers=training_config.dataloader_num_workers,
        pin_memory=training_config.dataloader_pin_memory
    )
    
    eval_dataloader = None
    if eval_dataset is not None:
        eval_dataloader = data_sharding.create_distributed_dataloader(
            eval_dataset,
            batch_size=training_config.batch_size,
            shuffle=False,
            num_workers=training_config.dataloader_num_workers,
            pin_memory=training_config.dataloader_pin_memory
        )
    
    # Create distributed trainer
    trainer = DistributedTrainer(
        model=model,
        config=training_config,
        distributed_config=distributed_config,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        **kwargs
    )
    
    return trainer


def launch_distributed_training(
    training_function: Callable,
    num_processes: int,
    num_nodes: int = 1,
    node_rank: int = 0,
    master_addr: str = "localhost",
    master_port: str = "12355",
    backend: str = "nccl",
    **kwargs
):
    """
    Launch distributed training across multiple processes/nodes.
    
    Args:
        training_function: Function to run on each process
        num_processes: Number of processes per node
        num_nodes: Number of nodes
        node_rank: Rank of current node
        master_addr: Master node address
        master_port: Master node port
        backend: Distributed backend
        **kwargs: Additional arguments for training function
    """
    import torch.multiprocessing as mp
    
    # Set environment variables
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port
    os.environ["WORLD_SIZE"] = str(num_processes * num_nodes)
    
    def worker_process(local_rank, *args, **kwargs):
        """Worker process function."""
        # Calculate global rank
        global_rank = node_rank * num_processes + local_rank
        
        # Set environment variables for this process
        os.environ["RANK"] = str(global_rank)
        os.environ["LOCAL_RANK"] = str(local_rank)
        
        # Create distributed configuration
        distributed_config = DistributedConfig(
            backend=backend,
            world_size=num_processes * num_nodes,
            rank=global_rank,
            local_rank=local_rank
        )
        
        # Run training function
        training_function(distributed_config, *args, **kwargs)
    
    # Launch processes
    if num_processes > 1:
        mp.spawn(
            worker_process,
            args=(kwargs,),
            nprocs=num_processes,
            join=True
        )
    else:
        # Single process
        worker_process(0, **kwargs)


# Utility functions for distributed training
def get_world_size() -> int:
    """Get world size from environment or distributed state."""
    if dist.is_initialized():
        return dist.get_world_size()
    return int(os.environ.get("WORLD_SIZE", 1))


def get_rank() -> int:
    """Get rank from environment or distributed state."""
    if dist.is_initialized():
        return dist.get_rank()
    return int(os.environ.get("RANK", 0))


def get_local_rank() -> int:
    """Get local rank from environment."""
    return int(os.environ.get("LOCAL_RANK", 0))


def is_main_process() -> bool:
    """Check if this is the main process."""
    return get_rank() == 0


def barrier():
    """Synchronize all processes."""
    if dist.is_initialized():
        dist.barrier()


def cleanup_distributed():
    """Cleanup distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()