#!/usr/bin/env python3
"""
Script for launching distributed TabGPT training across multiple nodes.

This script provides utilities for:
1. Single-node multi-GPU training
2. Multi-node distributed training
3. Automatic resource detection and configuration
4. Fault tolerance and recovery
5. Performance monitoring and optimization
"""

import os
import sys
import argparse
import subprocess
import json
import time
import socket
from pathlib import Path
from typing import List, Dict, Any, Optional

import torch


def get_available_gpus() -> List[int]:
    """Get list of available GPU devices."""
    if not torch.cuda.is_available():
        return []
    
    return list(range(torch.cuda.device_count()))


def get_free_port() -> int:
    """Get a free port for distributed training."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port


def detect_slurm_environment() -> Optional[Dict[str, Any]]:
    """Detect SLURM environment and extract configuration."""
    if "SLURM_JOB_ID" not in os.environ:
        return None
    
    return {
        'job_id': os.environ.get('SLURM_JOB_ID'),
        'num_nodes': int(os.environ.get('SLURM_NNODES', 1)),
        'node_rank': int(os.environ.get('SLURM_NODEID', 0)),
        'num_tasks': int(os.environ.get('SLURM_NTASKS', 1)),
        'tasks_per_node': int(os.environ.get('SLURM_NTASKS_PER_NODE', 1)),
        'cpus_per_task': int(os.environ.get('SLURM_CPUS_PER_TASK', 1)),
        'node_list': os.environ.get('SLURM_JOB_NODELIST', ''),
        'master_addr': os.environ.get('SLURM_LAUNCH_NODE_IPADDR', 'localhost')
    }


def create_hostfile(nodes: List[str], gpus_per_node: int) -> str:
    """Create hostfile for distributed training."""
    hostfile_content = []
    
    for node in nodes:
        for gpu_id in range(gpus_per_node):
            hostfile_content.append(f"{node} slots={gpus_per_node}")
    
    hostfile_path = "hostfile.txt"
    with open(hostfile_path, 'w') as f:
        f.write('\\n'.join(hostfile_content))
    
    return hostfile_path


class DistributedLauncher:
    """Launcher for distributed training."""
    
    def __init__(self, args):
        self.args = args
        self.slurm_env = detect_slurm_environment()
        self.available_gpus = get_available_gpus()
        
        # Setup configuration
        self.setup_configuration()
    
    def setup_configuration(self):
        """Setup distributed training configuration."""
        # Determine number of nodes and processes
        if self.slurm_env:
            self.num_nodes = self.slurm_env['num_nodes']
            self.node_rank = self.slurm_env['node_rank']
            self.master_addr = self.slurm_env['master_addr']
            self.gpus_per_node = len(self.available_gpus)
        else:
            self.num_nodes = self.args.num_nodes
            self.node_rank = self.args.node_rank
            self.master_addr = self.args.master_addr
            self.gpus_per_node = self.args.gpus_per_node or len(self.available_gpus)
        
        # Calculate world size
        self.world_size = self.num_nodes * self.gpus_per_node
        
        # Setup master port
        self.master_port = self.args.master_port or get_free_port()
        
        print(f"Distributed Training Configuration:")
        print(f"  Number of nodes: {self.num_nodes}")
        print(f"  GPUs per node: {self.gpus_per_node}")
        print(f"  World size: {self.world_size}")
        print(f"  Master address: {self.master_addr}")
        print(f"  Master port: {self.master_port}")
        print(f"  Node rank: {self.node_rank}")
        
        if self.slurm_env:
            print(f"  SLURM Job ID: {self.slurm_env['job_id']}")
    
    def launch_single_node_training(self):
        """Launch single-node multi-GPU training."""
        print(f"Launching single-node training with {self.gpus_per_node} GPUs...")
        
        # Prepare environment variables
        env = os.environ.copy()
        env.update({
            'MASTER_ADDR': self.master_addr,
            'MASTER_PORT': str(self.master_port),
            'WORLD_SIZE': str(self.world_size),
            'NODE_RANK': str(self.node_rank)
        })
        
        # Prepare command
        cmd = [
            sys.executable, "-m", "torch.distributed.launch",
            f"--nproc_per_node={self.gpus_per_node}",
            f"--nnodes={self.num_nodes}",
            f"--node_rank={self.node_rank}",
            f"--master_addr={self.master_addr}",
            f"--master_port={self.master_port}",
            "--use_env"
        ]
        
        # Add training script and arguments
        cmd.append(self.args.training_script)
        cmd.extend(self._get_training_arguments())
        
        print(f"Command: {' '.join(cmd)}")
        
        try:
            # Launch training
            result = subprocess.run(cmd, env=env, check=True)
            print("Single-node training completed successfully!")
            return result.returncode
        except subprocess.CalledProcessError as e:
            print(f"Single-node training failed: {e}")
            return e.returncode
    
    def launch_multi_node_training(self):
        """Launch multi-node distributed training."""
        print(f"Launching multi-node training across {self.num_nodes} nodes...")
        
        if self.slurm_env:
            return self._launch_slurm_training()
        else:
            return self._launch_manual_multi_node_training()
    
    def _launch_slurm_training(self):
        """Launch training using SLURM."""
        print("Using SLURM for multi-node training...")
        
        # Prepare SLURM command
        cmd = [
            "srun",
            "--ntasks-per-node", str(self.gpus_per_node),
            "--cpus-per-task", str(self.args.cpus_per_task),
            "--gres", f"gpu:{self.gpus_per_node}",
            sys.executable, self.args.training_script
        ]
        
        # Add training arguments
        cmd.extend(self._get_training_arguments())
        
        # Set environment variables
        env = os.environ.copy()
        env.update({
            'MASTER_ADDR': self.master_addr,
            'MASTER_PORT': str(self.master_port),
            'WORLD_SIZE': str(self.world_size)
        })
        
        print(f"SLURM command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, env=env, check=True)
            print("SLURM training completed successfully!")
            return result.returncode
        except subprocess.CalledProcessError as e:
            print(f"SLURM training failed: {e}")
            return e.returncode
    
    def _launch_manual_multi_node_training(self):
        """Launch multi-node training manually."""
        print("Manual multi-node training not fully implemented.")
        print("Please use SLURM or implement SSH-based launching.")
        return 1
    
    def _get_training_arguments(self) -> List[str]:
        """Get arguments to pass to training script."""
        training_args = []
        
        # Add distributed configuration
        training_args.extend([
            "--use-data-parallel",
            "--backend", self.args.backend,
            "--monitor-communication"
        ])
        
        # Add model configuration
        if self.args.model_size:
            size_configs = {
                'small': {
                    '--hidden-size': '256',
                    '--num-layers': '6',
                    '--num-attention-heads': '8',
                    '--intermediate-size': '1024'
                },
                'medium': {
                    '--hidden-size': '512',
                    '--num-layers': '8',
                    '--num-attention-heads': '8',
                    '--intermediate-size': '2048'
                },
                'large': {
                    '--hidden-size': '768',
                    '--num-layers': '12',
                    '--num-attention-heads': '12',
                    '--intermediate-size': '3072'
                },
                'xl': {
                    '--hidden-size': '1024',
                    '--num-layers': '16',
                    '--num-attention-heads': '16',
                    '--intermediate-size': '4096'
                }
            }
            
            if self.args.model_size in size_configs:
                for key, value in size_configs[self.args.model_size].items():
                    training_args.extend([key, value])
        
        # Add training configuration
        if self.args.batch_size:
            training_args.extend(['--batch-size', str(self.args.batch_size)])
        
        if self.args.learning_rate:
            training_args.extend(['--learning-rate', str(self.args.learning_rate)])
        
        if self.args.num_epochs:
            training_args.extend(['--num-epochs', str(self.args.num_epochs)])
        
        if self.args.output_dir:
            training_args.extend(['--output-dir', self.args.output_dir])
        
        if self.args.log_dir:
            training_args.extend(['--log-dir', self.args.log_dir])
        
        # Add optimization flags
        if self.args.use_mixed_precision:
            training_args.append('--use-mixed-precision')
        
        if self.args.gradient_compression:
            training_args.append('--use-gradient-compression')
        
        if self.args.enable_monitoring:
            training_args.append('--enable-monitoring')
        
        # Add dataset configuration
        if self.args.train_samples:
            training_args.extend(['--train-samples', str(self.args.train_samples)])
        
        if self.args.eval_samples:
            training_args.extend(['--eval-samples', str(self.args.eval_samples)])
        
        return training_args
    
    def create_launch_script(self) -> str:
        """Create a launch script for the training job."""
        script_content = f"""#!/bin/bash
# Generated launch script for TabGPT distributed training

# Job configuration
export MASTER_ADDR={self.master_addr}
export MASTER_PORT={self.master_port}
export WORLD_SIZE={self.world_size}
export NCCL_DEBUG=INFO

# CUDA configuration
export CUDA_VISIBLE_DEVICES={','.join(map(str, range(self.gpus_per_node)))}

# Launch training
"""
        
        if self.num_nodes == 1:
            # Single node command
            cmd = [
                "python", "-m", "torch.distributed.launch",
                f"--nproc_per_node={self.gpus_per_node}",
                "--use_env",
                self.args.training_script
            ]
            cmd.extend(self._get_training_arguments())
            
            script_content += ' '.join(cmd) + '\\n'
        
        else:
            # Multi-node command
            if self.slurm_env:
                cmd = [
                    "srun",
                    f"--ntasks-per-node={self.gpus_per_node}",
                    f"--cpus-per-task={self.args.cpus_per_task}",
                    f"--gres=gpu:{self.gpus_per_node}",
                    "python", self.args.training_script
                ]
                cmd.extend(self._get_training_arguments())
                
                script_content += ' '.join(cmd) + '\\n'
        
        # Write script
        script_path = "launch_training.sh"
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        # Make executable
        os.chmod(script_path, 0o755)
        
        print(f"Launch script created: {script_path}")
        return script_path
    
    def run_performance_test(self):
        """Run performance scaling test."""
        print("Running performance scaling test...")
        
        results = {}
        
        # Test different configurations
        test_configs = [
            {'gpus': 1, 'batch_size': 32},
            {'gpus': 2, 'batch_size': 32},
            {'gpus': 4, 'batch_size': 32},
        ]
        
        if len(self.available_gpus) >= 4:
            test_configs.append({'gpus': len(self.available_gpus), 'batch_size': 32})
        
        for config in test_configs:
            if config['gpus'] <= len(self.available_gpus):
                print(f"Testing with {config['gpus']} GPUs, batch size {config['batch_size']}...")
                
                # Run short training
                result = self._run_test_training(config)
                results[f"{config['gpus']}_gpus"] = result
        
        # Save results
        results_file = "scaling_test_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Scaling test results saved to {results_file}")
        
        # Print summary
        print("\\nScaling Test Summary:")
        print("-" * 40)
        for config_name, result in results.items():
            if result.get('success'):
                print(f"{config_name}: {result.get('samples_per_second', 0):.1f} samples/sec")
            else:
                print(f"{config_name}: FAILED")
    
    def _run_test_training(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run a short training for performance testing."""
        # Prepare test command
        cmd = [
            sys.executable, "-m", "torch.distributed.launch",
            f"--nproc_per_node={config['gpus']}",
            "--use_env",
            self.args.training_script,
            "--num-epochs", "1",
            "--batch-size", str(config['batch_size']),
            "--train-samples", "1000",
            "--eval-samples", "100",
            "--logging-steps", "10",
            "--save-steps", "1000000",  # Don't save during test
            "--eval-steps", "1000000"   # Don't eval during test
        ]
        
        # Set environment
        env = os.environ.copy()
        env.update({
            'MASTER_ADDR': 'localhost',
            'MASTER_PORT': str(get_free_port()),
            'WORLD_SIZE': str(config['gpus']),
            'CUDA_VISIBLE_DEVICES': ','.join(map(str, range(config['gpus'])))
        })
        
        try:
            start_time = time.time()
            result = subprocess.run(
                cmd, env=env, check=True, 
                capture_output=True, text=True, timeout=300  # 5 minute timeout
            )
            end_time = time.time()
            
            # Parse performance from output (simplified)
            training_time = end_time - start_time
            samples_processed = 1000  # Known from test config
            samples_per_second = samples_processed / training_time
            
            return {
                'success': True,
                'training_time': training_time,
                'samples_per_second': samples_per_second,
                'gpus': config['gpus'],
                'batch_size': config['batch_size']
            }
            
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            return {
                'success': False,
                'error': str(e),
                'gpus': config['gpus'],
                'batch_size': config['batch_size']
            }


def main():
    """Main function for distributed training launcher."""
    parser = argparse.ArgumentParser(description="Launch TabGPT Distributed Training")
    
    # Training script
    parser.add_argument("training_script", help="Path to training script")
    
    # Distributed configuration
    parser.add_argument("--num-nodes", type=int, default=1, help="Number of nodes")
    parser.add_argument("--node-rank", type=int, default=0, help="Rank of current node")
    parser.add_argument("--gpus-per-node", type=int, help="GPUs per node (auto-detect if not specified)")
    parser.add_argument("--master-addr", type=str, default="localhost", help="Master node address")
    parser.add_argument("--master-port", type=int, help="Master port (auto-assign if not specified)")
    parser.add_argument("--backend", type=str, default="nccl", help="Distributed backend")
    
    # SLURM configuration
    parser.add_argument("--cpus-per-task", type=int, default=4, help="CPUs per task for SLURM")
    
    # Model configuration
    parser.add_argument("--model-size", choices=["small", "medium", "large", "xl"], 
                       help="Predefined model size")
    
    # Training configuration
    parser.add_argument("--batch-size", type=int, help="Batch size per GPU")
    parser.add_argument("--learning-rate", type=float, help="Learning rate")
    parser.add_argument("--num-epochs", type=int, help="Number of epochs")
    parser.add_argument("--train-samples", type=int, help="Number of training samples")
    parser.add_argument("--eval-samples", type=int, help="Number of evaluation samples")
    
    # Optimization
    parser.add_argument("--use-mixed-precision", action="store_true", help="Use mixed precision")
    parser.add_argument("--gradient-compression", action="store_true", help="Use gradient compression")
    
    # Monitoring
    parser.add_argument("--enable-monitoring", action="store_true", default=True, help="Enable monitoring")
    
    # Output
    parser.add_argument("--output-dir", type=str, help="Output directory")
    parser.add_argument("--log-dir", type=str, help="Log directory")
    
    # Actions
    parser.add_argument("--create-script", action="store_true", help="Create launch script only")
    parser.add_argument("--performance-test", action="store_true", help="Run performance scaling test")
    parser.add_argument("--dry-run", action="store_true", help="Show configuration without launching")
    
    args = parser.parse_args()
    
    # Validate training script exists
    if not Path(args.training_script).exists():
        print(f"Error: Training script not found: {args.training_script}")
        return 1
    
    # Initialize launcher
    launcher = DistributedLauncher(args)
    
    # Handle different actions
    if args.dry_run:
        print("Dry run completed. Configuration shown above.")
        return 0
    
    if args.create_script:
        launcher.create_launch_script()
        return 0
    
    if args.performance_test:
        launcher.run_performance_test()
        return 0
    
    # Launch training
    try:
        if launcher.num_nodes == 1:
            return launcher.launch_single_node_training()
        else:
            return launcher.launch_multi_node_training()
    
    except KeyboardInterrupt:
        print("\\nTraining interrupted by user")
        return 1
    except Exception as e:
        print(f"\\nTraining failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())