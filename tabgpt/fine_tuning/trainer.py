"""Fine-tuning trainer for TabGPT with adapter support."""
import os
import json
import logging
from typing import Dict, List, Optional, Union, Any, Callable
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np

try:
    from transformers import Trainer, TrainingArguments
    from transformers.trainer_callback import TrainerCallback
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    Trainer = object
    TrainingArguments = object
    TrainerCallback = object

from ..models import TabGPTForSequenceClassification, TabGPTForRegression, TabGPTConfig
from ..adapters import LoRAConfig, apply_lora_to_model, freeze_base_model, get_trainable_parameters
from ..training.metrics import MetricsComputer

logger = logging.getLogger(__name__)


@dataclass
class FineTuningConfig:
    """Configuration for fine-tuning TabGPT models."""
    
    # Task configuration
    task_type: str = "classification"  # classification, regression, ranking
    num_labels: Optional[int] = None
    output_dim: Optional[int] = None
    
    # Adapter configuration
    use_adapters: bool = True
    adapter_type: str = "lora"  # lora, prefix, prompt
    lora_config: Optional[LoRAConfig] = None
    
    # Training configuration
    learning_rate: float = 1e-4
    num_epochs: int = 3
    batch_size: int = 16
    gradient_accumulation_steps: int = 1
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    
    # Evaluation configuration
    eval_strategy: str = "epoch"  # no, steps, epoch
    eval_steps: Optional[int] = None
    save_strategy: str = "epoch"
    save_steps: Optional[int] = None
    
    # Early stopping
    early_stopping_patience: Optional[int] = None
    early_stopping_threshold: float = 0.001
    
    # Logging and output
    logging_steps: int = 100
    output_dir: str = "./fine_tuned_model"
    run_name: Optional[str] = None
    
    # Data preprocessing
    max_length: Optional[int] = None
    preprocessing_num_workers: int = 4
    
    def __post_init__(self):
        """Post-initialization validation and setup."""
        # Validate task configuration
        if self.task_type == "classification" and self.num_labels is None:
            raise ValueError("num_labels must be specified for classification tasks")
        
        if self.task_type == "regression" and self.output_dim is None:
            self.output_dim = 1  # Default to single output regression
        
        # Create default LoRA config if using adapters
        if self.use_adapters and self.adapter_type == "lora" and self.lora_config is None:
            self.lora_config = LoRAConfig(
                r=8,
                alpha=16.0,
                dropout=0.1,
                target_modules=["query", "key", "value", "dense"]
            )
        
        # Create output directory
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
    
    def to_training_arguments(self) -> "TrainingArguments":
        """Convert to HuggingFace TrainingArguments."""
        if not HF_AVAILABLE:
            raise ImportError("transformers library is required for HuggingFace integration")
        
        return TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=self.num_epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            warmup_ratio=self.warmup_ratio,
            logging_steps=self.logging_steps,
            evaluation_strategy=self.eval_strategy,
            eval_steps=self.eval_steps,
            save_strategy=self.save_strategy,
            save_steps=self.save_steps,
            load_best_model_at_end=True if self.early_stopping_patience else False,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            run_name=self.run_name,
            report_to=None,  # Disable wandb/tensorboard by default
        )


class TabGPTFineTuningTrainer:
    """Fine-tuning trainer for TabGPT models with adapter support."""
    
    def __init__(
        self,
        model: nn.Module,
        config: FineTuningConfig,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        tokenizer: Optional[Any] = None,
        data_collator: Optional[Callable] = None,
        compute_metrics: Optional[Callable] = None,
    ):
        self.config = config
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.data_collator = data_collator
        self.compute_metrics = compute_metrics
        
        # Setup model with adapters if specified
        self.model = self._setup_model(model)
        
        # Initialize metrics computer
        self.metrics_computer = MetricsComputer(task_type=config.task_type)
        
        # Setup HuggingFace trainer if available
        self.hf_trainer = None
        if HF_AVAILABLE:
            self._setup_hf_trainer()
    
    def _setup_model(self, model: nn.Module) -> nn.Module:
        """Setup model with adapters and task-specific heads."""
        # Apply adapters if specified
        if self.config.use_adapters:
            if self.config.adapter_type == "lora":
                model = apply_lora_to_model(model, self.config.lora_config)
                freeze_base_model(model)
            else:
                raise ValueError(f"Unsupported adapter type: {self.config.adapter_type}")
        
        # Ensure model has correct task head
        if self.config.task_type == "classification":
            if not hasattr(model, 'classifier') and not isinstance(model, TabGPTForSequenceClassification):
                # Add classification head if needed
                logger.warning("Model doesn't have classification head. Consider using TabGPTForSequenceClassification.")
        elif self.config.task_type == "regression":
            if not hasattr(model, 'regressor') and not isinstance(model, TabGPTForRegression):
                # Add regression head if needed
                logger.warning("Model doesn't have regression head. Consider using TabGPTForRegression.")
        
        return model
    
    def _setup_hf_trainer(self):
        """Setup HuggingFace Trainer."""
        training_args = self.config.to_training_arguments()
        
        # Setup compute metrics function
        compute_metrics_fn = None
        if self.compute_metrics is not None:
            compute_metrics_fn = self.compute_metrics
        elif self.eval_dataset is not None:
            compute_metrics_fn = self._default_compute_metrics
        
        self.hf_trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            compute_metrics=compute_metrics_fn,
        )
    
    def _default_compute_metrics(self, eval_pred):
        """Default metrics computation for evaluation."""
        predictions, labels = eval_pred
        
        if self.config.task_type == "classification":
            # Convert logits to predictions
            if predictions.shape[-1] > 1:
                # Multi-class
                pred_classes = np.argmax(predictions, axis=-1)
            else:
                # Binary
                pred_classes = (predictions > 0).astype(int).flatten()
                labels = labels.astype(int)
            
            # Compute metrics
            from sklearn.metrics import accuracy_score, precision_recall_fscore_support
            
            accuracy = accuracy_score(labels, pred_classes)
            precision, recall, f1, _ = precision_recall_fscore_support(
                labels, pred_classes, average='weighted', zero_division=0
            )
            
            return {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1
            }
        
        elif self.config.task_type == "regression":
            # Compute regression metrics
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            
            mse = mean_squared_error(labels, predictions)
            mae = mean_absolute_error(labels, predictions)
            
            try:
                r2 = r2_score(labels, predictions)
            except:
                r2 = 0.0
            
            return {
                "mse": mse,
                "mae": mae,
                "rmse": np.sqrt(mse),
                "r2": r2
            }
        
        return {}
    
    def train(self) -> Dict[str, Any]:
        """Train the model."""
        logger.info("Starting fine-tuning...")
        
        # Print model information
        trainable_params, total_params, trainable_ratio = get_trainable_parameters(self.model)
        logger.info(f"Trainable parameters: {trainable_params:,} ({trainable_ratio:.2%})")
        logger.info(f"Total parameters: {total_params:,}")
        
        if self.hf_trainer is not None:
            # Use HuggingFace Trainer
            train_result = self.hf_trainer.train()
            
            # Save model and tokenizer
            self.hf_trainer.save_model()
            if self.tokenizer is not None:
                self.tokenizer.save_pretrained(self.config.output_dir)
            
            return train_result.metrics
        else:
            # Fallback to custom training loop
            return self._custom_training_loop()
    
    def _custom_training_loop(self) -> Dict[str, Any]:
        """Custom training loop when HuggingFace is not available."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.model.train()
        
        # Setup optimizer
        optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # Setup data loader
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=self.data_collator
        )
        
        # Training loop
        total_loss = 0.0
        num_steps = 0
        
        for epoch in range(self.config.num_epochs):
            epoch_loss = 0.0
            
            for batch in train_loader:
                # Move batch to device
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(**batch)
                loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
                
                # Backward pass
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
                # Update metrics
                epoch_loss += loss.item()
                total_loss += loss.item()
                num_steps += 1
                
                # Logging
                if num_steps % self.config.logging_steps == 0:
                    logger.info(f"Step {num_steps}, Loss: {loss.item():.4f}")
            
            logger.info(f"Epoch {epoch + 1}/{self.config.num_epochs}, "
                       f"Average Loss: {epoch_loss / len(train_loader):.4f}")
            
            # Evaluation
            if self.eval_dataset is not None:
                eval_metrics = self.evaluate()
                logger.info(f"Evaluation metrics: {eval_metrics}")
        
        # Save model
        self._save_model()
        
        return {
            "train_loss": total_loss / num_steps,
            "num_steps": num_steps
        }
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate the model."""
        if self.hf_trainer is not None:
            return self.hf_trainer.evaluate()
        else:
            return self._custom_evaluation()
    
    def _custom_evaluation(self) -> Dict[str, float]:
        """Custom evaluation when HuggingFace is not available."""
        if self.eval_dataset is None:
            return {}
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.eval()
        
        eval_loader = DataLoader(
            self.eval_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=self.data_collator
        )
        
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in eval_loader:
                # Move batch to device
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(**batch)
                loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
                
                total_loss += loss.item()
                
                # Collect predictions and labels
                if hasattr(outputs, 'logits'):
                    predictions = outputs.logits
                elif hasattr(outputs, 'predictions'):
                    predictions = outputs.predictions
                else:
                    predictions = outputs[1] if len(outputs) > 1 else outputs[0]
                
                all_predictions.append(predictions.cpu())
                if 'labels' in batch:
                    all_labels.append(batch['labels'].cpu())
        
        # Compute metrics
        metrics = {"eval_loss": total_loss / len(eval_loader)}
        
        if all_labels:
            predictions = torch.cat(all_predictions, dim=0)
            labels = torch.cat(all_labels, dim=0)
            
            # Use default metrics computation
            eval_pred = (predictions.numpy(), labels.numpy())
            computed_metrics = self._default_compute_metrics(eval_pred)
            metrics.update({f"eval_{k}": v for k, v in computed_metrics.items()})
        
        return metrics
    
    def _save_model(self):
        """Save the fine-tuned model."""
        output_path = Path(self.config.output_dir)
        
        # Save model state dict
        torch.save(self.model.state_dict(), output_path / "pytorch_model.bin")
        
        # Save config
        if hasattr(self.model, 'config'):
            self.model.config.save_pretrained(output_path)
        
        # Save adapter weights if using adapters
        if self.config.use_adapters and self.config.adapter_type == "lora":
            from ..adapters import save_lora_weights
            save_lora_weights(self.model, output_path / "lora", self.config.lora_config)
        
        logger.info(f"Model saved to {output_path}")
    
    def predict(self, dataset: Dataset) -> np.ndarray:
        """Make predictions on a dataset."""
        if self.hf_trainer is not None:
            predictions = self.hf_trainer.predict(dataset)
            return predictions.predictions
        else:
            return self._custom_predict(dataset)
    
    def _custom_predict(self, dataset: Dataset) -> np.ndarray:
        """Custom prediction when HuggingFace is not available."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.eval()
        
        data_loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=self.data_collator
        )
        
        all_predictions = []
        
        with torch.no_grad():
            for batch in data_loader:
                # Move batch to device
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(**batch)
                
                # Extract predictions
                if hasattr(outputs, 'logits'):
                    predictions = outputs.logits
                elif hasattr(outputs, 'predictions'):
                    predictions = outputs.predictions
                else:
                    predictions = outputs[1] if len(outputs) > 1 else outputs[0]
                
                all_predictions.append(predictions.cpu())
        
        return torch.cat(all_predictions, dim=0).numpy()


def create_fine_tuning_trainer(
    model: nn.Module,
    train_dataset: Dataset,
    eval_dataset: Optional[Dataset] = None,
    config: Optional[FineTuningConfig] = None,
    **kwargs
) -> TabGPTFineTuningTrainer:
    """
    Create a fine-tuning trainer with default configuration.
    
    Args:
        model: Base model to fine-tune
        train_dataset: Training dataset
        eval_dataset: Optional evaluation dataset
        config: Fine-tuning configuration
        **kwargs: Additional arguments for FineTuningConfig
        
    Returns:
        Configured fine-tuning trainer
    """
    if config is None:
        config = FineTuningConfig(**kwargs)
    
    return TabGPTFineTuningTrainer(
        model=model,
        config=config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset
    )