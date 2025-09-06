"""Training callbacks for fine-tuning."""

import os
import json
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
import torch
import torch.nn as nn
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl

logger = logging.getLogger(__name__)


class EarlyStoppingCallback(TrainerCallback):
    """Early stopping callback to prevent overfitting."""
    
    def __init__(
        self,
        early_stopping_patience: int = 3,
        early_stopping_threshold: float = 0.0,
        metric_for_best_model: str = "eval_loss",
        greater_is_better: bool = False
    ):
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_threshold = early_stopping_threshold
        self.metric_for_best_model = metric_for_best_model
        self.greater_is_better = greater_is_better
        
        self.best_metric = None
        self.patience_counter = 0
        
    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: nn.Module,
        logs: Optional[Dict[str, float]] = None,
        **kwargs
    ):
        """Check for early stopping after evaluation."""
        if logs is None:
            return
            
        current_metric = logs.get(self.metric_for_best_model)
        if current_metric is None:
            return
            
        if self.best_metric is None:
            self.best_metric = current_metric
            self.patience_counter = 0
        else:
            if self.greater_is_better:
                improved = current_metric > self.best_metric + self.early_stopping_threshold
            else:
                improved = current_metric < self.best_metric - self.early_stopping_threshold
                
            if improved:
                self.best_metric = current_metric
                self.patience_counter = 0
            else:
                self.patience_counter += 1
                
        if self.patience_counter >= self.early_stopping_patience:
            logger.info(f"Early stopping triggered after {self.patience_counter} evaluations without improvement")
            control.should_training_stop = True


class ModelCheckpointCallback(TrainerCallback):
    """Save model checkpoints based on metrics."""
    
    def __init__(
        self,
        save_best_model: bool = True,
        metric_for_best_model: str = "eval_loss",
        greater_is_better: bool = False,
        save_total_limit: Optional[int] = None
    ):
        self.save_best_model = save_best_model
        self.metric_for_best_model = metric_for_best_model
        self.greater_is_better = greater_is_better
        self.save_total_limit = save_total_limit
        
        self.best_metric = None
        self.saved_checkpoints: List[str] = []
        
    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: nn.Module,
        logs: Optional[Dict[str, float]] = None,
        **kwargs
    ):
        """Save checkpoint if model improved."""
        if not self.save_best_model or logs is None:
            return
            
        current_metric = logs.get(self.metric_for_best_model)
        if current_metric is None:
            return
            
        should_save = False
        if self.best_metric is None:
            should_save = True
            self.best_metric = current_metric
        else:
            if self.greater_is_better:
                if current_metric > self.best_metric:
                    should_save = True
                    self.best_metric = current_metric
            else:
                if current_metric < self.best_metric:
                    should_save = True
                    self.best_metric = current_metric
                    
        if should_save:
            # Save the model
            checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-best-{state.global_step}")
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            # Save model state
            model.save_pretrained(checkpoint_dir)
            
            # Save metrics
            metrics_file = os.path.join(checkpoint_dir, "eval_results.json")
            with open(metrics_file, 'w') as f:
                json.dump(logs, f, indent=2)
                
            self.saved_checkpoints.append(checkpoint_dir)
            logger.info(f"Saved best model checkpoint to {checkpoint_dir}")
            
            # Clean up old checkpoints if limit is set
            if self.save_total_limit is not None and len(self.saved_checkpoints) > self.save_total_limit:
                old_checkpoint = self.saved_checkpoints.pop(0)
                if os.path.exists(old_checkpoint):
                    import shutil
                    shutil.rmtree(old_checkpoint)
                    logger.info(f"Removed old checkpoint: {old_checkpoint}")


class ProgressCallback(TrainerCallback):
    """Enhanced progress logging callback."""
    
    def __init__(
        self,
        log_every_n_steps: int = 50,
        log_metrics: List[str] = None
    ):
        self.log_every_n_steps = log_every_n_steps
        self.log_metrics = log_metrics or ["loss", "learning_rate"]
        
    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: Optional[Dict[str, float]] = None,
        **kwargs
    ):
        """Enhanced logging with custom metrics."""
        if logs is None or state.global_step % self.log_every_n_steps != 0:
            return
            
        # Log selected metrics
        log_items = []
        for metric in self.log_metrics:
            if metric in logs:
                value = logs[metric]
                if isinstance(value, float):
                    log_items.append(f"{metric}: {value:.6f}")
                else:
                    log_items.append(f"{metric}: {value}")
                    
        if log_items:
            logger.info(f"Step {state.global_step}: {', '.join(log_items)}")
            
    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: Optional[Dict[str, float]] = None,
        **kwargs
    ):
        """Log evaluation results."""
        if logs is None:
            return
            
        eval_items = []
        for key, value in logs.items():
            if key.startswith("eval_"):
                if isinstance(value, float):
                    eval_items.append(f"{key}: {value:.6f}")
                else:
                    eval_items.append(f"{key}: {value}")
                    
        if eval_items:
            logger.info(f"Evaluation at step {state.global_step}: {', '.join(eval_items)}")


class AdapterCallback(TrainerCallback):
    """Callback for adapter-specific functionality."""
    
    def __init__(self, adapter_config: Optional[Dict[str, Any]] = None):
        self.adapter_config = adapter_config or {}
        
    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: nn.Module,
        **kwargs
    ):
        """Log adapter information at training start."""
        if hasattr(model, 'get_trainable_parameters'):
            trainable_params = model.get_trainable_parameters()
            total_params = sum(p.numel() for p in model.parameters())
            
            logger.info(f"Training with adapters:")
            logger.info(f"  Trainable parameters: {trainable_params:,}")
            logger.info(f"  Total parameters: {total_params:,}")
            logger.info(f"  Trainable ratio: {trainable_params/total_params:.2%}")
            
    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: nn.Module,
        **kwargs
    ):
        """Save adapter weights separately."""
        if hasattr(model, 'save_adapter_weights'):
            adapter_path = os.path.join(args.output_dir, f"adapter_weights_step_{state.global_step}.pt")
            model.save_adapter_weights(adapter_path)
            logger.info(f"Saved adapter weights to {adapter_path}")


class MetricsCallback(TrainerCallback):
    """Callback for computing and logging custom metrics."""
    
    def __init__(
        self,
        compute_metrics_fn: Optional[callable] = None,
        log_predictions: bool = False,
        prediction_log_file: Optional[str] = None
    ):
        self.compute_metrics_fn = compute_metrics_fn
        self.log_predictions = log_predictions
        self.prediction_log_file = prediction_log_file
        
    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: Optional[Dict[str, float]] = None,
        **kwargs
    ):
        """Compute and log additional metrics."""
        if self.compute_metrics_fn is not None and logs is not None:
            # Get predictions and labels from the evaluation
            eval_dataset = kwargs.get('eval_dataset')
            model = kwargs.get('model')
            
            if eval_dataset is not None and model is not None:
                try:
                    # This would need to be implemented based on the specific model
                    additional_metrics = self.compute_metrics_fn(model, eval_dataset)
                    logs.update(additional_metrics)
                    
                    logger.info("Additional metrics computed:")
                    for key, value in additional_metrics.items():
                        logger.info(f"  {key}: {value}")
                        
                except Exception as e:
                    logger.warning(f"Failed to compute additional metrics: {e}")


def create_default_callbacks(
    early_stopping_patience: int = 3,
    save_best_model: bool = True,
    metric_for_best_model: str = "eval_loss",
    greater_is_better: bool = False,
    log_every_n_steps: int = 50
) -> List[TrainerCallback]:
    """Create a default set of callbacks for fine-tuning."""
    callbacks = [
        EarlyStoppingCallback(
            early_stopping_patience=early_stopping_patience,
            metric_for_best_model=metric_for_best_model,
            greater_is_better=greater_is_better
        ),
        ModelCheckpointCallback(
            save_best_model=save_best_model,
            metric_for_best_model=metric_for_best_model,
            greater_is_better=greater_is_better
        ),
        ProgressCallback(log_every_n_steps=log_every_n_steps)
    ]
    
    return callbacks