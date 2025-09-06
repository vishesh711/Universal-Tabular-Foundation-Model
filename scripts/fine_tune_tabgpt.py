#!/usr/bin/env python3
"""
Fine-tuning script for TabGPT models.

This script provides a comprehensive interface for fine-tuning TabGPT models
on downstream tasks with support for LoRA adapters, custom datasets, and
various training configurations.
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import warnings

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tabgpt.models import TabGPTForSequenceClassification, TabGPTForRegression
from tabgpt.tokenizers import TabGPTTokenizer
from tabgpt.adapters import LoRAConfig, apply_lora_to_model
from tabgpt.fine_tuning import (
    TabGPTFineTuningTrainer,
    FineTuningConfig,
    prepare_classification_data,
    prepare_regression_data,
    create_default_callbacks
)
from tabgpt.data import TabularDataset, TabularDataLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Fine-tune TabGPT models")
    
    # Model arguments
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="Path to pre-trained model or model identifier"
    )
    parser.add_argument(
        "--task_type",
        type=str,
        choices=["classification", "regression"],
        required=True,
        help="Type of downstream task"
    )
    
    # Data arguments
    parser.add_argument(
        "--train_file",
        type=str,
        required=True,
        help="Path to training data file (CSV)"
    )
    parser.add_argument(
        "--validation_file",
        type=str,
        help="Path to validation data file (CSV)"
    )
    parser.add_argument(
        "--test_file",
        type=str,
        help="Path to test data file (CSV)"
    )
    parser.add_argument(
        "--target_column",
        type=str,
        required=True,
        help="Name of target column"
    )
    parser.add_argument(
        "--validation_split",
        type=float,
        default=0.2,
        help="Validation split ratio if no validation file provided"
    )
    
    # Training arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for model and logs"
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=32,
        help="Training batch size per device"
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=64,
        help="Evaluation batch size per device"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Learning rate"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay"
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=500,
        help="Number of warmup steps"
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=50,
        help="Log every X steps"
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=500,
        help="Evaluate every X steps"
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=1000,
        help="Save checkpoint every X steps"
    )
    
    # LoRA arguments
    parser.add_argument(
        "--use_lora",
        action="store_true",
        help="Use LoRA for efficient fine-tuning"
    )
    parser.add_argument(
        "--lora_r",
        type=int,
        default=8,
        help="LoRA rank"
    )
    parser.add_argument(
        "--lora_alpha",
        type=float,
        default=16.0,
        help="LoRA alpha parameter"
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.1,
        help="LoRA dropout"
    )
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        nargs="+",
        default=["query", "key", "value", "dense"],
        help="Target modules for LoRA"
    )
    
    # Other arguments
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use mixed precision training"
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=4,
        help="Number of dataloader workers"
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=3,
        help="Early stopping patience"
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=512,
        help="Maximum sequence length"
    )
    
    return parser.parse_args()


def load_data(
    train_file: str,
    target_column: str,
    validation_file: Optional[str] = None,
    test_file: Optional[str] = None,
    validation_split: float = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    """Load and split data."""
    logger.info(f"Loading training data from {train_file}")
    train_df = pd.read_csv(train_file)
    
    if target_column not in train_df.columns:
        raise ValueError(f"Target column '{target_column}' not found in training data")
    
    # Load validation data
    if validation_file:
        logger.info(f"Loading validation data from {validation_file}")
        val_df = pd.read_csv(validation_file)
    else:
        logger.info(f"Splitting training data with validation_split={validation_split}")
        train_df, val_df = train_test_split(
            train_df, 
            test_size=validation_split, 
            random_state=42,
            stratify=train_df[target_column] if train_df[target_column].dtype == 'object' else None
        )
    
    # Load test data
    test_df = None
    if test_file:
        logger.info(f"Loading test data from {test_file}")
        test_df = pd.read_csv(test_file)
    
    logger.info(f"Data loaded - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df) if test_df is not None else 0}")
    
    return train_df, val_df, test_df


def prepare_model_and_tokenizer(
    model_name_or_path: str,
    task_type: str,
    num_labels: Optional[int] = None,
    use_lora: bool = False,
    lora_config: Optional[LoRAConfig] = None
) -> Tuple[nn.Module, TabGPTTokenizer]:
    """Prepare model and tokenizer."""
    logger.info(f"Loading model from {model_name_or_path}")
    
    # Load tokenizer
    tokenizer = TabGPTTokenizer.from_pretrained(model_name_or_path)
    
    # Load model based on task type
    if task_type == "classification":
        model = TabGPTForSequenceClassification.from_pretrained(
            model_name_or_path,
            num_labels=num_labels
        )
    elif task_type == "regression":
        model = TabGPTForRegression.from_pretrained(model_name_or_path)
    else:
        raise ValueError(f"Unsupported task type: {task_type}")
    
    # Apply LoRA if requested
    if use_lora and lora_config:
        logger.info("Applying LoRA adapters")
        model = apply_lora_to_model(model, lora_config)
        
        # Print parameter efficiency
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        logger.info(f"Trainable ratio: {trainable_params/total_params:.2%}")
    
    return model, tokenizer


def compute_metrics(eval_pred, task_type: str):
    """Compute metrics for evaluation."""
    predictions, labels = eval_pred
    
    if task_type == "classification":
        predictions = np.argmax(predictions, axis=1)
        accuracy = accuracy_score(labels, predictions)
        return {"accuracy": accuracy}
    elif task_type == "regression":
        mse = mean_squared_error(labels, predictions)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(labels - predictions))
        return {"mse": mse, "rmse": rmse, "mae": mae}
    else:
        return {}


def main():
    """Main fine-tuning function."""
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save arguments
    with open(os.path.join(args.output_dir, "training_args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    
    # Load data
    train_df, val_df, test_df = load_data(
        args.train_file,
        args.target_column,
        args.validation_file,
        args.test_file,
        args.validation_split
    )
    
    # Prepare labels for classification
    num_labels = None
    label_encoder = None
    if args.task_type == "classification":
        label_encoder = LabelEncoder()
        train_df[args.target_column] = label_encoder.fit_transform(train_df[args.target_column])
        val_df[args.target_column] = label_encoder.transform(val_df[args.target_column])
        if test_df is not None:
            test_df[args.target_column] = label_encoder.transform(test_df[args.target_column])
        num_labels = len(label_encoder.classes_)
        logger.info(f"Number of classes: {num_labels}")
    
    # Prepare LoRA config
    lora_config = None
    if args.use_lora:
        lora_config = LoRAConfig(
            r=args.lora_r,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
            target_modules=args.lora_target_modules
        )
    
    # Prepare model and tokenizer
    model, tokenizer = prepare_model_and_tokenizer(
        args.model_name_or_path,
        args.task_type,
        num_labels,
        args.use_lora,
        lora_config
    )
    
    # Prepare datasets
    if args.task_type == "classification":
        train_dataset = prepare_classification_data(train_df, args.target_column, tokenizer)
        val_dataset = prepare_classification_data(val_df, args.target_column, tokenizer)
        test_dataset = prepare_classification_data(test_df, args.target_column, tokenizer) if test_df is not None else None
    else:
        train_dataset = prepare_regression_data(train_df, args.target_column, tokenizer)
        val_dataset = prepare_regression_data(val_df, args.target_column, tokenizer)
        test_dataset = prepare_regression_data(test_df, args.target_column, tokenizer) if test_df is not None else None
    
    # Create training configuration
    training_config = FineTuningConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        fp16=args.fp16,
        dataloader_num_workers=args.dataloader_num_workers,
        seed=args.seed
    )
    
    # Create callbacks
    callbacks = create_default_callbacks(
        early_stopping_patience=args.early_stopping_patience,
        save_best_model=True,
        metric_for_best_model="eval_accuracy" if args.task_type == "classification" else "eval_mse",
        greater_is_better=args.task_type == "classification",
        log_every_n_steps=args.logging_steps
    )
    
    # Create trainer
    trainer = TabGPTFineTuningTrainer(
        model=model,
        config=training_config,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=lambda x: compute_metrics(x, args.task_type),
        callbacks=callbacks
    )
    
    # Train model
    logger.info("Starting training...")
    trainer.train()
    
    # Save final model
    logger.info("Saving final model...")
    trainer.save_model()
    
    # Save LoRA weights separately if used
    if args.use_lora:
        from tabgpt.adapters import save_lora_weights
        save_lora_weights(model, os.path.join(args.output_dir, "lora"), lora_config)
    
    # Evaluate on test set if available
    if test_dataset is not None:
        logger.info("Evaluating on test set...")
        test_results = trainer.evaluate(test_dataset)
        
        # Save test results
        with open(os.path.join(args.output_dir, "test_results.json"), "w") as f:
            json.dump(test_results, f, indent=2)
        
        logger.info(f"Test results: {test_results}")
    
    # Save label encoder for classification
    if label_encoder is not None:
        import joblib
        joblib.dump(label_encoder, os.path.join(args.output_dir, "label_encoder.pkl"))
    
    logger.info(f"Fine-tuning completed! Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()