"""Training pipeline for TabGPT with multiple objectives."""
import os
import json
import time
import logging
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
from dataclasses import dataclass, field
from pathlib import Path
import warnings

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import numpy as np

from ..models.configuration_tabgpt import TabGPTConfig
from ..models.modeling_tabgpt import TabGPTForPreTraining


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for training pipeline."""
    
    # Basic training parameters
    num_epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 5e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    max_steps: Optional[int] = None
    
    # Multi-objective loss weights
    mcm_weight: float = 1.0  # Masked Cell Modeling
    mcol_weight: float = 0.5  # Masked Column Modeling
    crl_weight: float = 0.3  # Contrastive Row Learning
    nrp_weight: float = 0.2  # Next Row Prediction
    
    # Gradient and optimization
    max_grad_norm: float = 1.0
    gradient_accumulation_steps: int = 1
    
    # Mixed precision training
    use_mixed_precision: bool = True
    
    # Learning rate scheduling
    lr_scheduler_type: str = "cosine"  # linear, cosine, polynomial, constant
    lr_scheduler_warmup_ratio: float = 0.1
    
    # Checkpointing
    save_steps: int = 1000
    save_total_limit: int = 3
    output_dir: str = "./checkpoints"
    resume_from_checkpoint: Optional[str] = None
    
    # Evaluation
    eval_steps: int = 500
    eval_strategy: str = "steps"  # steps, epoch, no
    
    # Logging
    logging_steps: int = 100
    log_level: str = "info"
    
    # Early stopping
    early_stopping_patience: Optional[int] = None
    early_stopping_threshold: float = 0.001
    
    # Data loading
    dataloader_num_workers: int = 4
    dataloader_pin_memory: bool = True
    
    # Device and distributed training
    device: Optional[str] = None
    local_rank: int = -1
    
    def __post_init__(self):
        """Post-initialization validation."""
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Create output directory
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Validate scheduler type
        valid_schedulers = ["linear", "cosine", "polynomial", "constant"]
        if self.lr_scheduler_type not in valid_schedulers:
            raise ValueError(f"lr_scheduler_type must be one of {valid_schedulers}")
        
        # Validate eval strategy
        valid_eval_strategies = ["steps", "epoch", "no"]
        if self.eval_strategy not in valid_eval_strategies:
            raise ValueError(f"eval_strategy must be one of {valid_eval_strategies}")


@dataclass
class TrainingState:
    """State of the training process."""
    epoch: int = 0
    global_step: int = 0
    best_metric: Optional[float] = None
    best_model_checkpoint: Optional[str] = None
    is_local_process_zero: bool = True
    is_world_process_zero: bool = True
    
    # Loss tracking
    train_loss: float = 0.0
    eval_loss: Optional[float] = None
    
    # Objective-specific losses
    mcm_loss: float = 0.0
    mcol_loss: float = 0.0
    crl_loss: float = 0.0
    nrp_loss: float = 0.0
    
    # Learning rate
    learning_rate: float = 0.0
    
    # Timing
    epoch_start_time: float = 0.0
    step_start_time: float = 0.0


class MultiObjectiveTrainer:
    """Trainer for TabGPT with multiple pre-training objectives."""
    
    def __init__(
        self,
        model: TabGPTForPreTraining,
        config: TrainingConfig,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
        optimizer: Optional[Optimizer] = None,
        lr_scheduler: Optional[_LRScheduler] = None,
        compute_metrics: Optional[Callable] = None,
    ):
        self.model = model
        self.config = config
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.compute_metrics = compute_metrics
        
        # Initialize training state
        self.state = TrainingState()
        
        # Setup device
        self.device = torch.device(config.device)
        self.model.to(self.device)
        
        # Setup optimizer
        if optimizer is None:
            self.optimizer = self._create_optimizer()
        else:
            self.optimizer = optimizer
        
        # Setup learning rate scheduler
        if lr_scheduler is None:
            self.lr_scheduler = self._create_lr_scheduler()
        else:
            self.lr_scheduler = lr_scheduler
        
        # Setup mixed precision
        self.scaler = GradScaler() if config.use_mixed_precision else None
        
        # Setup logging
        self._setup_logging()
        
        # Load from checkpoint if specified
        if config.resume_from_checkpoint:
            self._load_checkpoint(config.resume_from_checkpoint)
    
    def _create_optimizer(self) -> Optimizer:
        """Create optimizer with weight decay."""
        # Separate parameters for weight decay
        decay_parameters = []
        no_decay_parameters = []
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if any(nd in name for nd in ["bias", "LayerNorm.weight", "layer_norm.weight"]):
                    no_decay_parameters.append(param)
                else:
                    decay_parameters.append(param)
        
        optimizer_grouped_parameters = [
            {
                "params": decay_parameters,
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": no_decay_parameters,
                "weight_decay": 0.0,
            },
        ]
        
        return torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.config.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
    
    def _create_lr_scheduler(self) -> _LRScheduler:
        """Create learning rate scheduler."""
        num_training_steps = self._get_num_training_steps()
        num_warmup_steps = int(self.config.lr_scheduler_warmup_ratio * num_training_steps)
        
        if self.config.lr_scheduler_type == "linear":
            from torch.optim.lr_scheduler import LinearLR
            return LinearLR(
                self.optimizer,
                start_factor=0.1,
                total_iters=num_warmup_steps
            )
        elif self.config.lr_scheduler_type == "cosine":
            from torch.optim.lr_scheduler import CosineAnnealingLR
            return CosineAnnealingLR(
                self.optimizer,
                T_max=num_training_steps - num_warmup_steps,
                eta_min=0
            )
        elif self.config.lr_scheduler_type == "constant":
            from torch.optim.lr_scheduler import ConstantLR
            return ConstantLR(self.optimizer, factor=1.0)
        else:
            # Default to cosine
            from torch.optim.lr_scheduler import CosineAnnealingLR
            return CosineAnnealingLR(
                self.optimizer,
                T_max=num_training_steps - num_warmup_steps,
                eta_min=0
            )
    
    def _get_num_training_steps(self) -> int:
        """Calculate total number of training steps."""
        if self.config.max_steps is not None:
            return self.config.max_steps
        
        num_update_steps_per_epoch = len(self.train_dataloader) // self.config.gradient_accumulation_steps
        return num_update_steps_per_epoch * self.config.num_epochs
    
    def _setup_logging(self):
        """Setup logging configuration."""
        log_level = getattr(logging, self.config.log_level.upper())
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=log_level,
        )
    
    def train(self) -> Dict[str, Any]:
        """Main training loop."""
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(self.train_dataloader.dataset)}")
        logger.info(f"  Num Epochs = {self.config.num_epochs}")
        logger.info(f"  Batch size = {self.config.batch_size}")
        logger.info(f"  Gradient Accumulation steps = {self.config.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {self._get_num_training_steps()}")
        
        # Training loop
        self.model.train()
        total_loss = 0.0
        
        for epoch in range(self.state.epoch, self.config.num_epochs):
            self.state.epoch = epoch
            self.state.epoch_start_time = time.time()
            
            epoch_loss = self._train_epoch()
            total_loss += epoch_loss
            
            # Evaluation
            if self.config.eval_strategy == "epoch" and self.eval_dataloader is not None:
                eval_metrics = self.evaluate()
                self._log_metrics(eval_metrics, "eval")
            
            # Save checkpoint
            if self.config.save_steps > 0 and (epoch + 1) % (self.config.save_steps // len(self.train_dataloader)) == 0:
                self._save_checkpoint()
            
            logger.info(f"Epoch {epoch + 1}/{self.config.num_epochs} completed. Average loss: {epoch_loss:.4f}")
        
        # Final evaluation
        if self.eval_dataloader is not None:
            final_metrics = self.evaluate()
            logger.info(f"Final evaluation metrics: {final_metrics}")
        
        # Save final model
        self._save_model()
        
        return {
            "train_loss": total_loss / self.config.num_epochs,
            "global_step": self.state.global_step,
            "epoch": self.state.epoch
        }
    
    def _train_epoch(self) -> float:
        """Train for one epoch."""
        epoch_loss = 0.0
        self.optimizer.zero_grad()
        
        for step, batch in enumerate(self.train_dataloader):
            self.state.step_start_time = time.time()
            
            # Move batch to device
            batch = self._prepare_inputs(batch)
            
            # Forward pass with mixed precision
            if self.config.use_mixed_precision and self.scaler is not None:
                with autocast():
                    loss, losses = self._compute_loss(batch)
            else:
                loss, losses = self._compute_loss(batch)
            
            # Scale loss for gradient accumulation
            if self.config.gradient_accumulation_steps > 1:
                loss = loss / self.config.gradient_accumulation_steps
            
            # Backward pass
            if self.config.use_mixed_precision and self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Update weights
            if (step + 1) % self.config.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.config.max_grad_norm > 0:
                    if self.config.use_mixed_precision and self.scaler is not None:
                        self.scaler.unscale_(self.optimizer)
                    
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm
                    )
                
                # Optimizer step
                if self.config.use_mixed_precision and self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                # Learning rate scheduler step
                self.lr_scheduler.step()
                
                # Zero gradients
                self.optimizer.zero_grad()
                
                # Update global step
                self.state.global_step += 1
                self.state.learning_rate = self.lr_scheduler.get_last_lr()[0]
            
            # Update loss tracking
            epoch_loss += loss.item()
            self.state.train_loss = loss.item()
            
            # Update objective-specific losses
            if losses:
                self.state.mcm_loss = losses.get('mcm_loss', 0.0)
                self.state.mcol_loss = losses.get('mcol_loss', 0.0)
                self.state.crl_loss = losses.get('crl_loss', 0.0)
                self.state.nrp_loss = losses.get('nrp_loss', 0.0)
            
            # Logging
            if self.config.logging_steps > 0 and self.state.global_step % self.config.logging_steps == 0:
                self._log_training_step()
            
            # Evaluation during training
            if (self.config.eval_strategy == "steps" and 
                self.config.eval_steps > 0 and 
                self.state.global_step % self.config.eval_steps == 0 and
                self.eval_dataloader is not None):
                eval_metrics = self.evaluate()
                self._log_metrics(eval_metrics, "eval")
                self.model.train()  # Return to training mode
            
            # Save checkpoint
            if (self.config.save_steps > 0 and 
                self.state.global_step % self.config.save_steps == 0):
                self._save_checkpoint()
            
            # Check if max steps reached
            if self.config.max_steps is not None and self.state.global_step >= self.config.max_steps:
                break
        
        return epoch_loss / len(self.train_dataloader)
    
    def _compute_loss(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute multi-objective loss."""
        # Forward pass through model
        outputs = self.model(**batch)
        
        if isinstance(outputs, dict):
            total_loss = outputs.get('loss', torch.tensor(0.0))
            losses = outputs.get('losses', {})
        else:
            # Handle tuple output
            total_loss = outputs[0] if len(outputs) > 0 else torch.tensor(0.0)
            losses = {}
        
        # Apply loss weights
        weighted_loss = 0.0
        loss_dict = {}
        
        if 'mcm_loss' in losses:
            mcm_loss = losses['mcm_loss'] * self.config.mcm_weight
            weighted_loss += mcm_loss
            loss_dict['mcm_loss'] = mcm_loss.item()
        
        if 'mcol_loss' in losses:
            mcol_loss = losses['mcol_loss'] * self.config.mcol_weight
            weighted_loss += mcol_loss
            loss_dict['mcol_loss'] = mcol_loss.item()
        
        if 'crl_loss' in losses:
            crl_loss = losses['crl_loss'] * self.config.crl_weight
            weighted_loss += crl_loss
            loss_dict['crl_loss'] = crl_loss.item()
        
        if 'nrp_loss' in losses:
            nrp_loss = losses['nrp_loss'] * self.config.nrp_weight
            weighted_loss += nrp_loss
            loss_dict['nrp_loss'] = nrp_loss.item()
        
        # Use weighted loss if we have individual losses, otherwise use total loss
        final_loss = weighted_loss if loss_dict else total_loss
        
        return final_loss, loss_dict
    
    def _prepare_inputs(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Prepare inputs for model."""
        prepared_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                prepared_batch[key] = value.to(self.device)
            else:
                prepared_batch[key] = value
        return prepared_batch
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate model on evaluation dataset."""
        if self.eval_dataloader is None:
            return {}
        
        logger.info("***** Running evaluation *****")
        
        self.model.eval()
        total_eval_loss = 0.0
        total_steps = 0
        
        eval_losses = {
            'mcm_loss': 0.0,
            'mcol_loss': 0.0,
            'crl_loss': 0.0,
            'nrp_loss': 0.0
        }
        
        with torch.no_grad():
            for batch in self.eval_dataloader:
                batch = self._prepare_inputs(batch)
                
                if self.config.use_mixed_precision:
                    with autocast():
                        loss, losses = self._compute_loss(batch)
                else:
                    loss, losses = self._compute_loss(batch)
                
                total_eval_loss += loss.item()
                total_steps += 1
                
                # Accumulate objective-specific losses
                for key in eval_losses:
                    if key in losses:
                        eval_losses[key] += losses[key]
        
        # Average losses
        avg_eval_loss = total_eval_loss / total_steps
        for key in eval_losses:
            eval_losses[key] = eval_losses[key] / total_steps
        
        metrics = {
            'eval_loss': avg_eval_loss,
            **{f'eval_{key}': value for key, value in eval_losses.items()}
        }
        
        # Compute additional metrics if provided
        if self.compute_metrics is not None:
            additional_metrics = self.compute_metrics(self.model, self.eval_dataloader)
            metrics.update(additional_metrics)
        
        self.state.eval_loss = avg_eval_loss
        
        return metrics
    
    def _log_training_step(self):
        """Log training step information."""
        step_time = time.time() - self.state.step_start_time
        
        logger.info(
            f"Step {self.state.global_step} | "
            f"Loss: {self.state.train_loss:.4f} | "
            f"LR: {self.state.learning_rate:.2e} | "
            f"Time: {step_time:.2f}s"
        )
        
        # Log objective-specific losses
        if any([self.state.mcm_loss, self.state.mcol_loss, self.state.crl_loss, self.state.nrp_loss]):
            logger.info(
                f"  MCM: {self.state.mcm_loss:.4f} | "
                f"MCOL: {self.state.mcol_loss:.4f} | "
                f"CRL: {self.state.crl_loss:.4f} | "
                f"NRP: {self.state.nrp_loss:.4f}"
            )
    
    def _log_metrics(self, metrics: Dict[str, float], prefix: str = ""):
        """Log metrics."""
        if prefix:
            prefix = f"{prefix}_"
        
        for key, value in metrics.items():
            logger.info(f"{prefix}{key}: {value:.4f}")
    
    def _save_checkpoint(self):
        """Save training checkpoint."""
        checkpoint_dir = Path(self.config.output_dir) / f"checkpoint-{self.state.global_step}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model and tokenizer
        self.model.save_pretrained(checkpoint_dir)
        
        # Save training state
        training_state = {
            'epoch': self.state.epoch,
            'global_step': self.state.global_step,
            'best_metric': self.state.best_metric,
            'train_loss': self.state.train_loss,
            'eval_loss': self.state.eval_loss,
            'learning_rate': self.state.learning_rate,
        }
        
        with open(checkpoint_dir / "training_state.json", "w") as f:
            json.dump(training_state, f, indent=2)
        
        # Save optimizer and scheduler state
        torch.save({
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lr_scheduler_state_dict': self.lr_scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
        }, checkpoint_dir / "optimizer.pt")
        
        logger.info(f"Checkpoint saved to {checkpoint_dir}")
        
        # Clean up old checkpoints
        self._cleanup_checkpoints()
    
    def _cleanup_checkpoints(self):
        """Remove old checkpoints to save disk space."""
        if self.config.save_total_limit <= 0:
            return
        
        checkpoint_dirs = []
        for path in Path(self.config.output_dir).iterdir():
            if path.is_dir() and path.name.startswith("checkpoint-"):
                try:
                    step = int(path.name.split("-")[1])
                    checkpoint_dirs.append((step, path))
                except (ValueError, IndexError):
                    continue
        
        # Sort by step number and keep only the most recent ones
        checkpoint_dirs.sort(key=lambda x: x[0])
        
        while len(checkpoint_dirs) > self.config.save_total_limit:
            _, old_checkpoint = checkpoint_dirs.pop(0)
            logger.info(f"Removing old checkpoint: {old_checkpoint}")
            import shutil
            shutil.rmtree(old_checkpoint)
    
    def _load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint."""
        checkpoint_dir = Path(checkpoint_path)
        
        if not checkpoint_dir.exists():
            logger.warning(f"Checkpoint directory {checkpoint_dir} does not exist")
            return
        
        # Load model
        self.model = TabGPTForPreTraining.from_pretrained(checkpoint_dir)
        self.model.to(self.device)
        
        # Load training state
        training_state_path = checkpoint_dir / "training_state.json"
        if training_state_path.exists():
            with open(training_state_path, "r") as f:
                training_state = json.load(f)
            
            self.state.epoch = training_state.get('epoch', 0)
            self.state.global_step = training_state.get('global_step', 0)
            self.state.best_metric = training_state.get('best_metric')
            self.state.train_loss = training_state.get('train_loss', 0.0)
            self.state.eval_loss = training_state.get('eval_loss')
            self.state.learning_rate = training_state.get('learning_rate', 0.0)
        
        # Load optimizer and scheduler state
        optimizer_path = checkpoint_dir / "optimizer.pt"
        if optimizer_path.exists():
            checkpoint = torch.load(optimizer_path, map_location=self.device)
            
            if 'optimizer_state_dict' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            if 'lr_scheduler_state_dict' in checkpoint:
                self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
            
            if 'scaler_state_dict' in checkpoint and self.scaler is not None:
                self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        logger.info(f"Checkpoint loaded from {checkpoint_dir}")
    
    def _save_model(self):
        """Save final model."""
        output_dir = Path(self.config.output_dir) / "final_model"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        self.model.save_pretrained(output_dir)
        logger.info(f"Final model saved to {output_dir}")


def create_trainer(
    model: TabGPTForPreTraining,
    train_dataloader: DataLoader,
    eval_dataloader: Optional[DataLoader] = None,
    config: Optional[TrainingConfig] = None,
    **kwargs
) -> MultiObjectiveTrainer:
    """
    Create a trainer instance with default configuration.
    
    Args:
        model: TabGPT model for pre-training
        train_dataloader: Training data loader
        eval_dataloader: Optional evaluation data loader
        config: Training configuration
        **kwargs: Additional arguments for TrainingConfig
        
    Returns:
        Configured trainer instance
    """
    if config is None:
        config = TrainingConfig(**kwargs)
    
    return MultiObjectiveTrainer(
        model=model,
        config=config,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader
    )