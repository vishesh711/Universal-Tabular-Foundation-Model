# TabGPT Distributed Training Guide

This guide covers everything you need to know about distributed training with TabGPT, from single-node multi-GPU setups to large-scale multi-node training across clusters.

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Configuration](#configuration)
4. [Single-Node Multi-GPU Training](#single-node-multi-gpu-training)
5. [Multi-Node Distributed Training](#multi-node-distributed-training)
6. [Performance Optimization](#performance-optimization)
7. [Monitoring and Debugging](#monitoring-and-debugging)
8. [Fault Tolerance](#fault-tolerance)
9. [Best Practices](#best-practices)
10. [Troubleshooting](#troubleshooting)

## Overview

TabGPT supports comprehensive distributed training capabilities:

- **Data Parallelism**: Distribute data across multiple GPUs/nodes
- **Model Parallelism**: Split large models across devices (pipeline and tensor parallelism)
- **Gradient Synchronization**: Efficient gradient communication with compression
- **Dynamic Batching**: Intelligent batch size adjustment for optimal throughput
- **Fault Tolerance**: Automatic recovery from node failures
- **Performance Monitoring**: Real-time tracking of training and communication metrics

### Supported Backends

- **NCCL**: Optimized for NVIDIA GPUs (recommended for GPU training)
- **Gloo**: CPU and GPU support, good for mixed environments
- **MPI**: For HPC environments with MPI support

## Quick Start

### Single-Node Multi-GPU Training

```bash
# Automatic multi-GPU training (detects available GPUs)
python examples/distributed_training_example.py --launch-multi-gpu

# Manual specification
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    examples/distributed_training_example.py \
    --use-data-parallel \
    --batch-size=32 \
    --num-epochs=3
```

### Multi-Node Training with SLURM

```bash
# Submit SLURM job
sbatch --nodes=4 --ntasks-per-node=8 --gres=gpu:8 \
    scripts/launch_distributed_training.py \
    examples/distributed_training_example.py \
    --model-size=large \
    --batch-size=16
```

### Using the Launcher Script

```bash
# Performance scaling test
python scripts/launch_distributed_training.py \
    examples/distributed_training_example.py \
    --performance-test

# Create launch script
python scripts/launch_distributed_training.py \
    examples/distributed_training_example.py \
    --create-script \
    --model-size=medium \
    --batch-size=32
```

## Configuration

### DistributedConfig

```python
from tabgpt.training import DistributedConfig

config = DistributedConfig(
    # Basic distributed setup
    backend="nccl",              # nccl, gloo, mpi
    world_size=8,                # Total number of processes
    rank=0,                      # Current process rank
    local_rank=0,                # Local rank on current node
    
    # Data parallelism
    use_data_parallel=True,      # Enable data parallelism
    find_unused_parameters=False, # DDP optimization
    
    # Model parallelism
    use_model_parallel=False,    # Enable model parallelism
    pipeline_parallel_size=1,    # Pipeline stages
    tensor_parallel_size=1,      # Tensor parallel groups
    
    # Gradient synchronization
    gradient_sync_frequency=1,   # Sync every N steps
    use_gradient_compression=False, # Compress gradients
    compression_ratio=0.1,       # Compression ratio
    
    # Communication optimization
    bucket_size_mb=25,           # DDP bucket size
    use_static_graph=False,      # Static graph optimization
    
    # Fault tolerance
    enable_fault_tolerance=False, # Enable fault tolerance
    checkpoint_frequency=1000,   # Checkpoint every N steps
    
    # Monitoring
    monitor_communication=True,  # Monitor communication
    log_communication_stats=False # Log detailed stats
)
```

### TrainingConfig for Distributed Training

```python
from tabgpt.training import TrainingConfig

config = TrainingConfig(
    # Adjust batch size for distributed training
    batch_size=32,  # Per-device batch size
    gradient_accumulation_steps=2,  # Effective batch size = batch_size * world_size * accumulation_steps
    
    # Learning rate scaling
    learning_rate=5e-4 * world_size,  # Scale with world size
    
    # Checkpointing (only on main process)
    save_steps=1000,
    save_total_limit=3,
    
    # Evaluation
    eval_steps=500,
    
    # Mixed precision for better performance
    use_mixed_precision=True,
    
    # Data loading
    dataloader_num_workers=4,  # Per-process workers
    dataloader_pin_memory=True
)
```

## Single-Node Multi-GPU Training

### Automatic Setup

```python
from tabgpt.training import setup_distributed_training
from tabgpt.models import TabGPTForPreTraining

# Model and data
model = TabGPTForPreTraining.from_pretrained("tabgpt-base")
train_dataset = YourDataset()
eval_dataset = YourEvalDataset()

# Automatic distributed setup
trainer = setup_distributed_training(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    training_config=training_config,
    distributed_config=distributed_config
)

# Train
trainer.train()
```

### Manual Setup

```python
import torch.distributed as dist
from tabgpt.training import DistributedTrainer, DistributedManager

# Initialize distributed environment
dist.init_process_group(backend="nccl")

# Create distributed manager
distributed_manager = DistributedManager(distributed_config)
distributed_manager.initialize()

# Create trainer
trainer = DistributedTrainer(
    model=model,
    config=training_config,
    distributed_config=distributed_config,
    train_dataloader=train_dataloader,
    eval_dataloader=eval_dataloader
)

# Train
trainer.train()

# Cleanup
trainer.cleanup()
```

### Performance Optimization for Single-Node

```python
# Optimize for single-node multi-GPU
distributed_config = DistributedConfig(
    backend="nccl",
    use_data_parallel=True,
    bucket_size_mb=25,           # Larger buckets for better bandwidth
    find_unused_parameters=False, # Disable if all parameters are used
    use_static_graph=True,       # Enable if model structure is static
    gradient_sync_frequency=1,   # Sync every step for consistency
    monitor_communication=True
)

training_config = TrainingConfig(
    batch_size=64,               # Larger batch size per GPU
    use_mixed_precision=True,    # Enable for V100/A100 GPUs
    gradient_accumulation_steps=1, # No accumulation needed with large batch
    dataloader_num_workers=8,    # More workers for data loading
    dataloader_pin_memory=True
)
```

## Multi-Node Distributed Training

### SLURM Environment

```bash
#!/bin/bash
#SBATCH --job-name=tabgpt-distributed
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:8
#SBATCH --time=24:00:00

# Load modules
module load cuda/11.8
module load nccl/2.15

# Set environment variables
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=12355
export WORLD_SIZE=$SLURM_NTASKS
export NCCL_DEBUG=INFO

# Launch training
srun python examples/distributed_training_example.py \
    --use-data-parallel \
    --backend=nccl \
    --model-size=large \
    --batch-size=16 \
    --num-epochs=10 \
    --enable-monitoring \
    --output-dir=/scratch/tabgpt-output
```

### Manual Multi-Node Setup

```python
# On each node, set environment variables:
import os
os.environ["MASTER_ADDR"] = "node1.cluster.com"
os.environ["MASTER_PORT"] = "12355"
os.environ["WORLD_SIZE"] = "32"  # 4 nodes * 8 GPUs
os.environ["RANK"] = str(node_rank * gpus_per_node + local_rank)
os.environ["LOCAL_RANK"] = str(local_gpu_id)

# Initialize distributed training
from tabgpt.training import setup_distributed_training

trainer = setup_distributed_training(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    training_config=training_config,
    distributed_config=DistributedConfig(
        backend="nccl",
        world_size=32,
        rank=int(os.environ["RANK"]),
        local_rank=int(os.environ["LOCAL_RANK"])
    )
)

trainer.train()
```

### Network Configuration

```python
# For multi-node training, optimize network settings
distributed_config = DistributedConfig(
    backend="nccl",
    bucket_size_mb=50,           # Larger buckets for network efficiency
    gradient_sync_frequency=2,   # Less frequent sync to reduce network overhead
    use_gradient_compression=True, # Compress gradients for network efficiency
    compression_ratio=0.1,       # 10x compression
    monitor_communication=True,
    enable_fault_tolerance=True  # Important for multi-node stability
)

# Set NCCL environment variables for network optimization
os.environ["NCCL_IB_DISABLE"] = "0"      # Enable InfiniBand if available
os.environ["NCCL_NET_GDR_LEVEL"] = "2"   # Enable GPU Direct RDMA
os.environ["NCCL_TREE_THRESHOLD"] = "0"  # Use tree algorithm for large clusters
```

## Performance Optimization

### Gradient Synchronization

```python
# Optimize gradient synchronization
from tabgpt.training import GradientSynchronization

grad_sync = GradientSynchronization(distributed_config)

# Custom synchronization in training loop
for batch in dataloader:
    # Forward pass
    outputs = model(batch)
    loss = outputs.loss / gradient_accumulation_steps
    
    # Backward pass
    loss.backward()
    
    # Synchronize gradients every N steps
    if step % gradient_accumulation_steps == 0:
        grad_sync.synchronize_gradients(model, distributed_manager)
        optimizer.step()
        optimizer.zero_grad()
```

### Dynamic Batching

```python
from tabgpt.serving import DynamicBatcher

# Use dynamic batching for variable-length sequences
batcher = DynamicBatcher(
    max_batch_size=64,
    timeout_ms=100,
    padding_strategy="longest"
)

# In training loop
for samples in data_stream:
    ready = batcher.add_request(samples)
    if ready:
        batch = batcher.get_batch()
        # Process batch
        outputs = model(batch)
```

### Memory Optimization

```python
# Optimize memory usage for large models
training_config = TrainingConfig(
    use_mixed_precision=True,     # Reduce memory by 50%
    gradient_accumulation_steps=4, # Simulate larger batch with less memory
    dataloader_pin_memory=True,   # Faster GPU transfer
    max_grad_norm=1.0            # Prevent gradient explosion
)

distributed_config = DistributedConfig(
    bucket_size_mb=25,           # Balance memory and communication
    find_unused_parameters=False, # Reduce memory overhead
    use_gradient_compression=True # Reduce communication memory
)

# Enable gradient checkpointing for very large models
model.gradient_checkpointing_enable()
```

## Monitoring and Debugging

### Comprehensive Monitoring

```python
from tabgpt.training import DistributedTrainingMonitor

# Initialize monitoring
monitor = DistributedTrainingMonitor(
    log_dir="./distributed_logs",
    rank=get_rank(),
    world_size=get_world_size(),
    monitoring_interval=1.0  # Monitor every second
)

# Start monitoring
monitor.start_monitoring()

# In training loop
monitor.log_training_step(
    step=global_step,
    epoch=epoch,
    loss=loss.item(),
    learning_rate=scheduler.get_last_lr()[0],
    batch_size=batch_size,
    step_time=step_time
)

# Log communication operations
op_id = monitor.start_communication_profiling("allreduce", data_size_bytes=1024)
# ... perform communication ...
monitor.end_communication_profiling(op_id)

# Stop monitoring
monitor.stop_monitoring()
```

### Performance Analysis

```python
# Get comprehensive statistics
stats = monitor.get_training_statistics()

print(f"Average step time: {stats['step_timing']['avg_step_time']:.4f}s")
print(f"Communication efficiency: {stats['communication_stats']['overall']['communication_efficiency']:.2f} MB/s")
print(f"GPU utilization: {stats['resource_stats']['avg_gpu_utilization']:.2%}")

# Analyze scaling efficiency
baseline_throughput = 100  # samples/sec on single GPU
current_throughput = stats['samples_per_second']
scaling_efficiency = current_throughput / (baseline_throughput * world_size)
print(f"Scaling efficiency: {scaling_efficiency:.2%}")
```

### Debugging Communication Issues

```python
# Enable detailed NCCL debugging
os.environ["NCCL_DEBUG"] = "INFO"
os.environ["NCCL_DEBUG_SUBSYS"] = "ALL"

# Monitor communication patterns
from tabgpt.training import CommunicationProfiler

profiler = CommunicationProfiler()

# Profile specific operations
op_id = profiler.start_operation("allreduce", data_size_bytes=model_size)
dist.all_reduce(tensor)
profiler.end_operation(op_id)

# Get detailed statistics
comm_stats = profiler.get_communication_stats()
print(f"AllReduce bandwidth: {comm_stats['allreduce']['avg_bandwidth_mbps']:.2f} MB/s")
```

## Fault Tolerance

### Automatic Checkpointing

```python
# Enable fault tolerance
distributed_config = DistributedConfig(
    enable_fault_tolerance=True,
    checkpoint_frequency=1000  # Checkpoint every 1000 steps
)

training_config = TrainingConfig(
    save_steps=1000,
    save_total_limit=5,  # Keep last 5 checkpoints
    resume_from_checkpoint="./checkpoints/checkpoint-5000"  # Resume from checkpoint
)

# Automatic recovery
trainer = DistributedTrainer(
    model=model,
    config=training_config,
    distributed_config=distributed_config,
    train_dataloader=train_dataloader
)

# Training will automatically resume from last checkpoint if interrupted
trainer.train()
```

### Manual Fault Handling

```python
import signal
import sys

def signal_handler(signum, frame):
    """Handle interruption signals."""
    print(f"Received signal {signum}, saving checkpoint...")
    trainer.save_checkpoint()
    trainer.cleanup()
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

try:
    trainer.train()
except Exception as e:
    print(f"Training failed: {e}")
    trainer.save_checkpoint()  # Save checkpoint on failure
    raise
```

## Best Practices

### Data Loading

```python
# Optimize data loading for distributed training
from tabgpt.training import DataSharding

data_sharding = DataSharding(distributed_config)

# Use distributed sampler
train_sampler = data_sharding.create_distributed_sampler(
    train_dataset,
    shuffle=True,
    drop_last=True  # Ensure consistent batch sizes across processes
)

train_dataloader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    sampler=train_sampler,
    num_workers=8,           # More workers for better throughput
    pin_memory=True,         # Faster GPU transfer
    persistent_workers=True, # Keep workers alive between epochs
    prefetch_factor=2       # Prefetch batches
)

# Set epoch for distributed sampler
for epoch in range(num_epochs):
    train_sampler.set_epoch(epoch)  # Ensure different shuffling each epoch
    # ... training loop ...
```

### Learning Rate Scaling

```python
# Scale learning rate with world size
base_lr = 5e-4
world_size = get_world_size()

# Linear scaling (common approach)
scaled_lr = base_lr * world_size

# Square root scaling (for very large batch sizes)
# scaled_lr = base_lr * math.sqrt(world_size)

# Gradual warmup for large learning rates
warmup_steps = 1000
training_config = TrainingConfig(
    learning_rate=scaled_lr,
    warmup_steps=warmup_steps,
    lr_scheduler_type="cosine"
)
```

### Synchronization Points

```python
# Minimize synchronization points
from tabgpt.training import barrier, is_main_process

# Only synchronize when necessary
if step % eval_steps == 0:
    barrier()  # Synchronize before evaluation
    
    if is_main_process():
        # Only main process does evaluation logging
        eval_metrics = evaluate_model()
        log_metrics(eval_metrics)
    
    barrier()  # Synchronize after evaluation

# Avoid frequent synchronization in training loop
for batch in dataloader:
    # No synchronization needed here
    outputs = model(batch)
    loss = outputs.loss
    loss.backward()
    
    if step % gradient_accumulation_steps == 0:
        # Gradient synchronization happens automatically in DDP
        optimizer.step()
        optimizer.zero_grad()
```

## Troubleshooting

### Common Issues

#### NCCL Initialization Failures

```bash
# Check NCCL installation
python -c "import torch; print(torch.cuda.nccl.version())"

# Set NCCL environment variables
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1  # Disable InfiniBand if causing issues
export NCCL_P2P_DISABLE=1  # Disable P2P if causing issues

# For Docker containers
export NCCL_SOCKET_IFNAME=eth0  # Specify network interface
```

#### Out of Memory Errors

```python
# Reduce memory usage
training_config = TrainingConfig(
    batch_size=16,              # Reduce batch size
    gradient_accumulation_steps=4, # Maintain effective batch size
    use_mixed_precision=True,   # Use FP16
    dataloader_num_workers=2    # Reduce workers
)

# Enable gradient checkpointing
model.gradient_checkpointing_enable()

# Clear cache periodically
if step % 100 == 0:
    torch.cuda.empty_cache()
```

#### Slow Training

```python
# Profile training to identify bottlenecks
import torch.profiler

with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
    on_trace_ready=torch.profiler.tensorboard_trace_handler('./profiler_logs'),
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as prof:
    for step, batch in enumerate(dataloader):
        outputs = model(batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        prof.step()
        
        if step >= (1 + 1 + 3) * 2:
            break
```

#### Communication Bottlenecks

```python
# Monitor communication patterns
monitor = DistributedTrainingMonitor(log_dir="./logs")
monitor.start_monitoring()

# Check communication statistics
stats = monitor.get_training_statistics()
comm_stats = stats['communication_stats']

if comm_stats['overall']['communication_efficiency'] < 100:  # MB/s
    print("Communication bottleneck detected!")
    
    # Potential solutions:
    # 1. Increase bucket size
    distributed_config.bucket_size_mb = 50
    
    # 2. Reduce sync frequency
    distributed_config.gradient_sync_frequency = 2
    
    # 3. Enable compression
    distributed_config.use_gradient_compression = True
```

### Performance Debugging

```python
# Measure different components
import time

# Data loading time
data_start = time.time()
batch = next(iter(dataloader))
data_time = time.time() - data_start

# Forward pass time
forward_start = time.time()
outputs = model(batch)
forward_time = time.time() - forward_start

# Backward pass time
backward_start = time.time()
outputs.loss.backward()
backward_time = time.time() - backward_start

# Communication time (in DDP, happens during backward)
print(f"Data loading: {data_time*1000:.2f}ms")
print(f"Forward pass: {forward_time*1000:.2f}ms")
print(f"Backward pass: {backward_time*1000:.2f}ms")

# Check if data loading is the bottleneck
if data_time > forward_time + backward_time:
    print("Data loading is the bottleneck!")
    print("Consider: more workers, pin_memory=True, prefetch_factor")
```

### Scaling Analysis

```python
# Analyze scaling efficiency
def analyze_scaling_efficiency(single_gpu_throughput, current_throughput, world_size):
    """Analyze distributed training scaling efficiency."""
    
    ideal_throughput = single_gpu_throughput * world_size
    efficiency = current_throughput / ideal_throughput
    
    print(f"Single GPU throughput: {single_gpu_throughput:.1f} samples/sec")
    print(f"Current throughput: {current_throughput:.1f} samples/sec")
    print(f"Ideal throughput: {ideal_throughput:.1f} samples/sec")
    print(f"Scaling efficiency: {efficiency:.2%}")
    
    if efficiency < 0.8:
        print("Poor scaling efficiency detected!")
        print("Potential causes:")
        print("- Communication overhead")
        print("- Load imbalance")
        print("- Memory bandwidth limitations")
        print("- Synchronization bottlenecks")
    
    return efficiency

# Usage
efficiency = analyze_scaling_efficiency(
    single_gpu_throughput=100,
    current_throughput=stats['samples_per_second'],
    world_size=get_world_size()
)
```

This comprehensive guide covers all aspects of distributed training with TabGPT. For additional help, refer to the examples in the `examples/` directory and the API documentation.