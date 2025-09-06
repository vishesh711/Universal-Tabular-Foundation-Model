"""Monitoring and logging utilities for distributed training."""

import os
import time
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import threading
from collections import defaultdict, deque
import socket

import torch
import torch.distributed as dist
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DistributedMetrics:
    """Metrics for distributed training monitoring."""
    
    # Training metrics
    train_loss: float = 0.0
    eval_loss: float = 0.0
    learning_rate: float = 0.0
    epoch: int = 0
    global_step: int = 0
    
    # Performance metrics
    samples_per_second: float = 0.0
    tokens_per_second: float = 0.0
    batch_processing_time: float = 0.0
    gradient_sync_time: float = 0.0
    
    # Communication metrics
    communication_time: float = 0.0
    communication_volume_mb: float = 0.0
    allreduce_calls: int = 0
    broadcast_calls: int = 0
    
    # Resource metrics
    gpu_memory_used_mb: float = 0.0
    gpu_memory_total_mb: float = 0.0
    gpu_utilization: float = 0.0
    cpu_utilization: float = 0.0
    
    # Distributed metrics
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0
    
    # Timing
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            'train_loss': self.train_loss,
            'eval_loss': self.eval_loss,
            'learning_rate': self.learning_rate,
            'epoch': self.epoch,
            'global_step': self.global_step,
            'samples_per_second': self.samples_per_second,
            'tokens_per_second': self.tokens_per_second,
            'batch_processing_time': self.batch_processing_time,
            'gradient_sync_time': self.gradient_sync_time,
            'communication_time': self.communication_time,
            'communication_volume_mb': self.communication_volume_mb,
            'allreduce_calls': self.allreduce_calls,
            'broadcast_calls': self.broadcast_calls,
            'gpu_memory_used_mb': self.gpu_memory_used_mb,
            'gpu_memory_total_mb': self.gpu_memory_total_mb,
            'gpu_utilization': self.gpu_utilization,
            'cpu_utilization': self.cpu_utilization,
            'world_size': self.world_size,
            'rank': self.rank,
            'local_rank': self.local_rank,
            'timestamp': self.timestamp
        }


class ResourceMonitor:
    """Monitor system resources during distributed training."""
    
    def __init__(self, monitoring_interval: float = 1.0):
        self.monitoring_interval = monitoring_interval
        self.is_monitoring = False
        self.monitor_thread = None
        self.metrics_history = deque(maxlen=1000)
        self.lock = threading.Lock()
    
    def start_monitoring(self):
        """Start resource monitoring."""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self.monitor_thread.start()
        logger.info("Resource monitoring started")
    
    def stop_monitoring(self):
        """Stop resource monitoring."""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        logger.info("Resource monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                metrics = self._collect_resource_metrics()
                
                with self.lock:
                    self.metrics_history.append(metrics)
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.warning(f"Error in resource monitoring: {e}")
    
    def _collect_resource_metrics(self) -> Dict[str, float]:
        """Collect current resource metrics."""
        metrics = {}
        
        # GPU metrics
        if torch.cuda.is_available():
            try:
                # Memory usage
                memory_used = torch.cuda.memory_allocated() / (1024 ** 2)  # MB
                memory_total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)  # MB
                
                metrics['gpu_memory_used_mb'] = memory_used
                metrics['gpu_memory_total_mb'] = memory_total
                metrics['gpu_memory_utilization'] = memory_used / memory_total if memory_total > 0 else 0
                
                # GPU utilization (simplified - would need nvidia-ml-py for accurate measurement)
                metrics['gpu_utilization'] = min(memory_used / memory_total, 1.0) if memory_total > 0 else 0
                
            except Exception as e:
                logger.warning(f"Failed to collect GPU metrics: {e}")
        
        # CPU metrics
        try:
            import psutil
            
            # CPU utilization
            metrics['cpu_utilization'] = psutil.cpu_percent(interval=None)
            
            # Memory usage
            memory = psutil.virtual_memory()
            metrics['system_memory_used_mb'] = memory.used / (1024 ** 2)
            metrics['system_memory_total_mb'] = memory.total / (1024 ** 2)
            metrics['system_memory_utilization'] = memory.percent / 100.0
            
            # Network I/O
            network = psutil.net_io_counters()
            metrics['network_bytes_sent'] = network.bytes_sent
            metrics['network_bytes_recv'] = network.bytes_recv
            
        except ImportError:
            logger.warning("psutil not available, skipping system metrics")
        except Exception as e:
            logger.warning(f"Failed to collect system metrics: {e}")
        
        metrics['timestamp'] = time.time()
        return metrics
    
    def get_current_metrics(self) -> Dict[str, float]:
        """Get current resource metrics."""
        return self._collect_resource_metrics()
    
    def get_metrics_history(self) -> List[Dict[str, float]]:
        """Get metrics history."""
        with self.lock:
            return list(self.metrics_history)
    
    def get_average_metrics(self, window_size: int = 60) -> Dict[str, float]:
        """Get average metrics over a time window."""
        with self.lock:
            recent_metrics = list(self.metrics_history)[-window_size:]
        
        if not recent_metrics:
            return {}
        
        # Calculate averages
        avg_metrics = {}
        for key in recent_metrics[0].keys():
            if key != 'timestamp':
                values = [m[key] for m in recent_metrics if key in m]
                if values:
                    avg_metrics[f'avg_{key}'] = np.mean(values)
                    avg_metrics[f'max_{key}'] = np.max(values)
                    avg_metrics[f'min_{key}'] = np.min(values)
        
        return avg_metrics


class CommunicationProfiler:
    """Profile communication patterns in distributed training."""
    
    def __init__(self):
        self.communication_events = []
        self.active_operations = {}
        self.lock = threading.Lock()
    
    def start_operation(self, operation_type: str, data_size_bytes: int = 0) -> str:
        """Start tracking a communication operation."""
        operation_id = f"{operation_type}_{time.time()}_{id(threading.current_thread())}"
        
        with self.lock:
            self.active_operations[operation_id] = {
                'type': operation_type,
                'start_time': time.time(),
                'data_size_bytes': data_size_bytes,
                'rank': dist.get_rank() if dist.is_initialized() else 0
            }
        
        return operation_id
    
    def end_operation(self, operation_id: str):
        """End tracking a communication operation."""
        end_time = time.time()
        
        with self.lock:
            if operation_id in self.active_operations:
                operation = self.active_operations.pop(operation_id)
                
                event = {
                    'type': operation['type'],
                    'start_time': operation['start_time'],
                    'end_time': end_time,
                    'duration': end_time - operation['start_time'],
                    'data_size_bytes': operation['data_size_bytes'],
                    'rank': operation['rank'],
                    'bandwidth_mbps': (
                        operation['data_size_bytes'] / (1024 * 1024) / 
                        max(end_time - operation['start_time'], 1e-6)
                    )
                }
                
                self.communication_events.append(event)
    
    def get_communication_stats(self) -> Dict[str, Any]:
        """Get communication statistics."""
        with self.lock:
            events = self.communication_events.copy()
        
        if not events:
            return {}
        
        # Group by operation type
        stats_by_type = defaultdict(list)
        for event in events:
            stats_by_type[event['type']].append(event)
        
        # Calculate statistics
        stats = {}
        for op_type, type_events in stats_by_type.items():
            durations = [e['duration'] for e in type_events]
            data_sizes = [e['data_size_bytes'] for e in type_events]
            bandwidths = [e['bandwidth_mbps'] for e in type_events]
            
            stats[op_type] = {
                'count': len(type_events),
                'total_duration': sum(durations),
                'avg_duration': np.mean(durations),
                'max_duration': np.max(durations),
                'min_duration': np.min(durations),
                'total_data_mb': sum(data_sizes) / (1024 * 1024),
                'avg_bandwidth_mbps': np.mean(bandwidths) if bandwidths else 0,
                'max_bandwidth_mbps': np.max(bandwidths) if bandwidths else 0
            }
        
        # Overall statistics
        all_durations = [e['duration'] for e in events]
        all_data_sizes = [e['data_size_bytes'] for e in events]
        
        stats['overall'] = {
            'total_operations': len(events),
            'total_communication_time': sum(all_durations),
            'total_data_transferred_mb': sum(all_data_sizes) / (1024 * 1024),
            'avg_operation_duration': np.mean(all_durations),
            'communication_efficiency': (
                sum(all_data_sizes) / (1024 * 1024) / 
                max(sum(all_durations), 1e-6)
            )
        }
        
        return stats
    
    def clear_history(self):
        """Clear communication history."""
        with self.lock:
            self.communication_events.clear()
            self.active_operations.clear()


class DistributedLogger:
    """Centralized logging for distributed training."""
    
    def __init__(
        self,
        log_dir: str,
        rank: int = 0,
        world_size: int = 1,
        log_level: str = "INFO"
    ):
        self.log_dir = Path(log_dir)
        self.rank = rank
        self.world_size = world_size
        self.is_main_process = (rank == 0)
        
        # Create log directory
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self._setup_logging(log_level)
        
        # Metrics storage
        self.metrics_buffer = []
        self.buffer_size = 100
        
        # Log files
        self.metrics_file = self.log_dir / f"metrics_rank_{rank}.jsonl"
        self.events_file = self.log_dir / f"events_rank_{rank}.log"
    
    def _setup_logging(self, log_level: str):
        """Setup logging configuration."""
        # Create formatter
        formatter = logging.Formatter(
            f'[Rank {self.rank}] %(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # File handler for this rank
        file_handler = logging.FileHandler(self.log_dir / f"training_rank_{self.rank}.log")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(getattr(logging, log_level.upper()))
        
        # Console handler (only for main process to avoid spam)
        if self.is_main_process:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            console_handler.setLevel(getattr(logging, log_level.upper()))
            
            # Add handlers to root logger
            root_logger = logging.getLogger()
            root_logger.addHandler(console_handler)
        
        # Add file handler to root logger
        root_logger = logging.getLogger()
        root_logger.addHandler(file_handler)
        root_logger.setLevel(getattr(logging, log_level.upper()))
    
    def log_metrics(self, metrics: DistributedMetrics):
        """Log training metrics."""
        # Add to buffer
        self.metrics_buffer.append(metrics.to_dict())
        
        # Flush buffer if full
        if len(self.metrics_buffer) >= self.buffer_size:
            self.flush_metrics()
    
    def flush_metrics(self):
        """Flush metrics buffer to file."""
        if not self.metrics_buffer:
            return
        
        try:
            with open(self.metrics_file, 'a') as f:
                for metrics in self.metrics_buffer:
                    f.write(json.dumps(metrics) + '\\n')
            
            self.metrics_buffer.clear()
            
        except Exception as e:
            logger.error(f"Failed to flush metrics: {e}")
    
    def log_event(self, event_type: str, message: str, **kwargs):
        """Log training event."""
        event = {
            'timestamp': time.time(),
            'rank': self.rank,
            'event_type': event_type,
            'message': message,
            **kwargs
        }
        
        try:
            with open(self.events_file, 'a') as f:
                f.write(json.dumps(event) + '\\n')
        except Exception as e:
            logger.error(f"Failed to log event: {e}")
        
        # Also log to standard logger
        logger.info(f"[{event_type}] {message}")
    
    def log_communication_stats(self, stats: Dict[str, Any]):
        """Log communication statistics."""
        if self.is_main_process:
            stats_file = self.log_dir / "communication_stats.json"
            
            try:
                with open(stats_file, 'w') as f:
                    json.dump(stats, f, indent=2)
            except Exception as e:
                logger.error(f"Failed to log communication stats: {e}")
    
    def aggregate_metrics_across_ranks(self) -> Dict[str, Any]:
        """Aggregate metrics from all ranks (main process only)."""
        if not self.is_main_process:
            return {}
        
        aggregated_metrics = defaultdict(list)
        
        # Read metrics from all ranks
        for rank in range(self.world_size):
            metrics_file = self.log_dir / f"metrics_rank_{rank}.jsonl"
            
            if not metrics_file.exists():
                continue
            
            try:
                with open(metrics_file, 'r') as f:
                    for line in f:
                        metrics = json.loads(line.strip())
                        
                        for key, value in metrics.items():
                            if isinstance(value, (int, float)):
                                aggregated_metrics[key].append(value)
            
            except Exception as e:
                logger.warning(f"Failed to read metrics from rank {rank}: {e}")
        
        # Calculate statistics
        stats = {}
        for key, values in aggregated_metrics.items():
            if values:
                stats[key] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'count': len(values)
                }
        
        return stats
    
    def create_training_report(self) -> Dict[str, Any]:
        """Create comprehensive training report (main process only)."""
        if not self.is_main_process:
            return {}
        
        report = {
            'training_info': {
                'world_size': self.world_size,
                'log_directory': str(self.log_dir),
                'report_timestamp': time.time()
            },
            'aggregated_metrics': self.aggregate_metrics_across_ranks()
        }
        
        # Save report
        report_file = self.log_dir / "training_report.json"
        try:
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"Training report saved to {report_file}")
        except Exception as e:
            logger.error(f"Failed to save training report: {e}")
        
        return report
    
    def cleanup(self):
        """Cleanup logging resources."""
        # Flush any remaining metrics
        self.flush_metrics()
        
        # Create final report if main process
        if self.is_main_process:
            self.create_training_report()


class DistributedTrainingMonitor:
    """Comprehensive monitoring for distributed training."""
    
    def __init__(
        self,
        log_dir: str,
        rank: int = 0,
        world_size: int = 1,
        monitoring_interval: float = 1.0
    ):
        self.rank = rank
        self.world_size = world_size
        self.is_main_process = (rank == 0)
        
        # Initialize components
        self.logger = DistributedLogger(log_dir, rank, world_size)
        self.resource_monitor = ResourceMonitor(monitoring_interval)
        self.comm_profiler = CommunicationProfiler()
        
        # Training state
        self.training_start_time = None
        self.last_log_time = time.time()
        self.step_times = deque(maxlen=100)
        
    def start_monitoring(self):
        """Start all monitoring components."""
        self.training_start_time = time.time()
        self.resource_monitor.start_monitoring()
        
        self.logger.log_event(
            "training_start",
            f"Started distributed training monitoring on rank {self.rank}"
        )
    
    def stop_monitoring(self):
        """Stop all monitoring components."""
        self.resource_monitor.stop_monitoring()
        
        # Log final statistics
        if self.training_start_time:
            total_time = time.time() - self.training_start_time
            self.logger.log_event(
                "training_end",
                f"Training completed in {total_time:.2f} seconds"
            )
        
        # Log communication statistics
        comm_stats = self.comm_profiler.get_communication_stats()
        self.logger.log_communication_stats(comm_stats)
        
        # Cleanup
        self.logger.cleanup()
    
    def log_training_step(
        self,
        step: int,
        epoch: int,
        loss: float,
        learning_rate: float,
        batch_size: int,
        step_time: float
    ):
        """Log training step metrics."""
        current_time = time.time()
        
        # Calculate performance metrics
        samples_per_second = batch_size / step_time if step_time > 0 else 0
        
        # Get resource metrics
        resource_metrics = self.resource_monitor.get_current_metrics()
        
        # Create metrics object
        metrics = DistributedMetrics(
            train_loss=loss,
            learning_rate=learning_rate,
            epoch=epoch,
            global_step=step,
            samples_per_second=samples_per_second,
            batch_processing_time=step_time,
            world_size=self.world_size,
            rank=self.rank,
            local_rank=int(os.environ.get("LOCAL_RANK", 0)),
            **resource_metrics
        )
        
        # Log metrics
        self.logger.log_metrics(metrics)
        
        # Track step times
        self.step_times.append(step_time)
        
        # Log periodically
        if current_time - self.last_log_time > 60:  # Every minute
            self._log_periodic_summary(step, epoch)
            self.last_log_time = current_time
    
    def log_evaluation_metrics(self, eval_metrics: Dict[str, float]):
        """Log evaluation metrics."""
        self.logger.log_event(
            "evaluation",
            f"Evaluation completed: {eval_metrics}"
        )
    
    def log_checkpoint_save(self, checkpoint_path: str, step: int):
        """Log checkpoint save event."""
        self.logger.log_event(
            "checkpoint_save",
            f"Checkpoint saved at step {step}",
            checkpoint_path=checkpoint_path,
            step=step
        )
    
    def start_communication_profiling(self, operation_type: str, data_size_bytes: int = 0) -> str:
        """Start profiling a communication operation."""
        return self.comm_profiler.start_operation(operation_type, data_size_bytes)
    
    def end_communication_profiling(self, operation_id: str):
        """End profiling a communication operation."""
        self.comm_profiler.end_operation(operation_id)
    
    def _log_periodic_summary(self, step: int, epoch: int):
        """Log periodic training summary."""
        if not self.step_times:
            return
        
        # Calculate averages
        avg_step_time = np.mean(self.step_times)
        
        # Get resource averages
        resource_avg = self.resource_monitor.get_average_metrics(window_size=60)
        
        # Get communication stats
        comm_stats = self.comm_profiler.get_communication_stats()
        
        summary = {
            'step': step,
            'epoch': epoch,
            'avg_step_time': avg_step_time,
            'steps_per_second': 1.0 / avg_step_time if avg_step_time > 0 else 0,
            **resource_avg,
            'communication_stats': comm_stats.get('overall', {})
        }
        
        self.logger.log_event(
            "periodic_summary",
            f"Training summary at step {step}",
            **summary
        )
    
    def get_training_statistics(self) -> Dict[str, Any]:
        """Get comprehensive training statistics."""
        stats = {
            'rank': self.rank,
            'world_size': self.world_size,
            'resource_stats': self.resource_monitor.get_average_metrics(),
            'communication_stats': self.comm_profiler.get_communication_stats(),
            'step_timing': {
                'avg_step_time': np.mean(self.step_times) if self.step_times else 0,
                'min_step_time': np.min(self.step_times) if self.step_times else 0,
                'max_step_time': np.max(self.step_times) if self.step_times else 0,
                'total_steps': len(self.step_times)
            }
        }
        
        if self.training_start_time:
            stats['total_training_time'] = time.time() - self.training_start_time
        
        return stats