"""Transfer learning evaluation for TabGPT models."""

import logging
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
from dataclasses import dataclass, field
import warnings

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from .evaluator import EvaluationConfig, EvaluationResult, create_evaluator
from .metrics import compute_all_metrics
from .datasets import TabularDatasetLoader, create_synthetic_dataset

logger = logging.getLogger(__name__)


@dataclass
class TransferLearningResult:
    """Result of transfer learning evaluation."""
    
    source_dataset: str
    target_dataset: str
    model_name: str
    
    # Results without pre-training (baseline)
    baseline_metrics: Dict[str, float]
    baseline_training_time: float
    
    # Results with pre-training
    pretrained_metrics: Dict[str, float]
    pretrained_training_time: float
    
    # Transfer learning benefit
    improvement: Dict[str, float] = field(default_factory=dict)
    relative_improvement: Dict[str, float] = field(default_factory=dict)
    
    # Additional info
    source_dataset_size: int = 0
    target_dataset_size: int = 0
    fine_tuning_epochs: int = 0
    additional_info: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Compute improvements after initialization."""
        self._compute_improvements()
    
    def _compute_improvements(self):
        """Compute improvement metrics."""
        for metric in self.baseline_metrics:
            if metric in self.pretrained_metrics:
                baseline_val = self.baseline_metrics[metric]
                pretrained_val = self.pretrained_metrics[metric]
                
                # Absolute improvement
                self.improvement[metric] = pretrained_val - baseline_val
                
                # Relative improvement (percentage)
                if baseline_val != 0:
                    self.relative_improvement[metric] = (
                        (pretrained_val - baseline_val) / abs(baseline_val) * 100
                    )
                else:
                    self.relative_improvement[metric] = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'source_dataset': self.source_dataset,
            'target_dataset': self.target_dataset,
            'model_name': self.model_name,
            'baseline_metrics': self.baseline_metrics,
            'baseline_training_time': self.baseline_training_time,
            'pretrained_metrics': self.pretrained_metrics,
            'pretrained_training_time': self.pretrained_training_time,
            'improvement': self.improvement,
            'relative_improvement': self.relative_improvement,
            'source_dataset_size': self.source_dataset_size,
            'target_dataset_size': self.target_dataset_size,
            'fine_tuning_epochs': self.fine_tuning_epochs,
            'additional_info': self.additional_info
        }


class TransferLearningEvaluator:
    """Evaluator for transfer learning scenarios."""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.results: List[TransferLearningResult] = []
    
    def evaluate_transfer(
        self,
        pretrained_model: Any,
        baseline_model: Any,
        source_dataset: Tuple[pd.DataFrame, pd.Series],
        target_dataset: Tuple[pd.DataFrame, pd.Series],
        source_name: str = "source",
        target_name: str = "target",
        fine_tuning_epochs: int = 10,
        **kwargs
    ) -> TransferLearningResult:
        """Evaluate transfer learning from source to target dataset."""
        
        X_source, y_source = source_dataset
        X_target, y_target = target_dataset
        
        logger.info(f"Evaluating transfer learning: {source_name} -> {target_name}")
        
        # Split target dataset
        X_train, X_test, y_train, y_test = train_test_split(
            X_target, y_target,
            test_size=self.config.test_size,
            random_state=self.config.cv_random_state,
            stratify=y_target if len(np.unique(y_target)) < 20 else None
        )
        
        # Evaluate baseline model (no pre-training)
        logger.info("Evaluating baseline model (no pre-training)")
        baseline_result = self._evaluate_baseline(
            baseline_model, X_train, y_train, X_test, y_test, **kwargs
        )
        
        # Evaluate pre-trained model
        logger.info("Evaluating pre-trained model")
        pretrained_result = self._evaluate_pretrained(
            pretrained_model, X_source, y_source, X_train, y_train, X_test, y_test,
            fine_tuning_epochs=fine_tuning_epochs, **kwargs
        )
        
        # Create transfer learning result
        result = TransferLearningResult(
            source_dataset=source_name,
            target_dataset=target_name,
            model_name=pretrained_model.__class__.__name__,
            baseline_metrics=baseline_result['metrics'],
            baseline_training_time=baseline_result['training_time'],
            pretrained_metrics=pretrained_result['metrics'],
            pretrained_training_time=pretrained_result['training_time'],
            source_dataset_size=len(X_source),
            target_dataset_size=len(X_target),
            fine_tuning_epochs=fine_tuning_epochs
        )
        
        self.results.append(result)
        return result
    
    def _evaluate_baseline(
        self,
        model: Any,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        **kwargs
    ) -> Dict[str, Any]:
        """Evaluate baseline model without pre-training."""
        
        import time
        import copy
        
        # Create a fresh copy of the model
        baseline_model = copy.deepcopy(model)
        
        # Train from scratch
        start_time = time.time()
        if hasattr(baseline_model, 'fit'):
            baseline_model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        # Make predictions
        y_pred = baseline_model.predict(X_test)
        y_pred_proba = None
        if hasattr(baseline_model, 'predict_proba'):
            try:
                y_pred_proba = baseline_model.predict_proba(X_test)
            except Exception:
                pass
        
        # Compute metrics
        task_type = "classification" if len(np.unique(y_test)) < 20 else "regression"
        metrics = compute_all_metrics(y_test, y_pred, y_pred_proba, task_type)
        
        return {
            'metrics': metrics,
            'training_time': training_time
        }
    
    def _evaluate_pretrained(
        self,
        model: Any,
        X_source: pd.DataFrame,
        y_source: pd.Series,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        fine_tuning_epochs: int = 10,
        **kwargs
    ) -> Dict[str, Any]:
        """Evaluate pre-trained model with fine-tuning."""
        
        import time
        import copy
        
        # Create a copy of the pre-trained model
        pretrained_model = copy.deepcopy(model)
        
        start_time = time.time()
        
        # Step 1: Pre-train on source dataset (if not already pre-trained)
        if hasattr(pretrained_model, 'pretrain') and not getattr(pretrained_model, 'is_pretrained', False):
            logger.info("Pre-training on source dataset")
            pretrained_model.pretrain(X_source, y_source)
        
        # Step 2: Fine-tune on target dataset
        if hasattr(pretrained_model, 'fine_tune'):
            logger.info("Fine-tuning on target dataset")
            pretrained_model.fine_tune(X_train, y_train, epochs=fine_tuning_epochs)
        elif hasattr(pretrained_model, 'fit'):
            # Fallback to regular training
            pretrained_model.fit(X_train, y_train)
        
        training_time = time.time() - start_time
        
        # Make predictions
        y_pred = pretrained_model.predict(X_test)
        y_pred_proba = None
        if hasattr(pretrained_model, 'predict_proba'):
            try:
                y_pred_proba = pretrained_model.predict_proba(X_test)
            except Exception:
                pass
        
        # Compute metrics
        task_type = "classification" if len(np.unique(y_test)) < 20 else "regression"
        metrics = compute_all_metrics(y_test, y_pred, y_pred_proba, task_type)
        
        return {
            'metrics': metrics,
            'training_time': training_time
        }
    
    def get_summary(self) -> pd.DataFrame:
        """Get summary of transfer learning results."""
        if not self.results:
            return pd.DataFrame()
        
        summary_data = []
        for result in self.results:
            row = {
                'Source': result.source_dataset,
                'Target': result.target_dataset,
                'Model': result.model_name,
                'Source Size': result.source_dataset_size,
                'Target Size': result.target_dataset_size,
                'Fine-tuning Epochs': result.fine_tuning_epochs
            }
            
            # Add baseline metrics
            for metric, value in result.baseline_metrics.items():
                row[f'Baseline {metric}'] = value
            
            # Add pre-trained metrics
            for metric, value in result.pretrained_metrics.items():
                row[f'Pretrained {metric}'] = value
            
            # Add improvements
            for metric, value in result.improvement.items():
                row[f'Improvement {metric}'] = value
            
            # Add relative improvements
            for metric, value in result.relative_improvement.items():
                row[f'Relative Improvement {metric} (%)'] = value
            
            summary_data.append(row)
        
        return pd.DataFrame(summary_data)


class PretrainingBenefitAnalyzer:
    """Analyze the benefits of pre-training across different scenarios."""
    
    def __init__(self):
        self.results: List[TransferLearningResult] = []
    
    def analyze_dataset_size_effect(
        self,
        pretrained_model: Any,
        baseline_model: Any,
        source_dataset: Tuple[pd.DataFrame, pd.Series],
        target_dataset: Tuple[pd.DataFrame, pd.Series],
        target_sizes: List[int] = None,
        **kwargs
    ) -> List[TransferLearningResult]:
        """Analyze how pre-training benefit varies with target dataset size."""
        
        if target_sizes is None:
            target_sizes = [100, 500, 1000, 2000, 5000]
        
        X_target, y_target = target_dataset
        results = []
        
        for size in target_sizes:
            if size > len(X_target):
                continue
            
            logger.info(f"Analyzing with target size: {size}")
            
            # Sample target dataset
            if size < len(X_target):
                X_sample, _, y_sample, _ = train_test_split(
                    X_target, y_target, train_size=size, random_state=42,
                    stratify=y_target if len(np.unique(y_target)) < 20 else None
                )
            else:
                X_sample, y_sample = X_target, y_target
            
            # Evaluate transfer learning
            evaluator = TransferLearningEvaluator(EvaluationConfig())
            result = evaluator.evaluate_transfer(
                pretrained_model, baseline_model,
                source_dataset, (X_sample, y_sample),
                source_name="source", target_name=f"target_{size}",
                **kwargs
            )
            
            results.append(result)
            self.results.append(result)
        
        return results
    
    def analyze_domain_similarity_effect(
        self,
        pretrained_model: Any,
        baseline_model: Any,
        source_datasets: Dict[str, Tuple[pd.DataFrame, pd.Series]],
        target_dataset: Tuple[pd.DataFrame, pd.Series],
        target_name: str = "target",
        **kwargs
    ) -> List[TransferLearningResult]:
        """Analyze how domain similarity affects transfer learning benefit."""
        
        results = []
        
        for source_name, source_dataset in source_datasets.items():
            logger.info(f"Analyzing transfer from {source_name} to {target_name}")
            
            evaluator = TransferLearningEvaluator(EvaluationConfig())
            result = evaluator.evaluate_transfer(
                pretrained_model, baseline_model,
                source_dataset, target_dataset,
                source_name=source_name, target_name=target_name,
                **kwargs
            )
            
            results.append(result)
            self.results.append(result)
        
        return results
    
    def get_benefit_summary(self, metric: str = "accuracy") -> pd.DataFrame:
        """Get summary of pre-training benefits."""
        if not self.results:
            return pd.DataFrame()
        
        summary_data = []
        for result in self.results:
            if metric in result.relative_improvement:
                row = {
                    'Source': result.source_dataset,
                    'Target': result.target_dataset,
                    'Target Size': result.target_dataset_size,
                    f'Baseline {metric}': result.baseline_metrics.get(metric, 0),
                    f'Pretrained {metric}': result.pretrained_metrics.get(metric, 0),
                    f'Improvement': result.improvement.get(metric, 0),
                    f'Relative Improvement (%)': result.relative_improvement.get(metric, 0)
                }
                summary_data.append(row)
        
        return pd.DataFrame(summary_data)


class DomainAdaptationEvaluator:
    """Evaluate domain adaptation scenarios."""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.results: List[TransferLearningResult] = []
    
    def evaluate_cross_domain_transfer(
        self,
        model: Any,
        source_domains: Dict[str, Tuple[pd.DataFrame, pd.Series]],
        target_domains: Dict[str, Tuple[pd.DataFrame, pd.Series]],
        **kwargs
    ) -> List[TransferLearningResult]:
        """Evaluate cross-domain transfer learning."""
        
        results = []
        
        for source_name, source_data in source_domains.items():
            for target_name, target_data in target_domains.items():
                if source_name == target_name:
                    continue  # Skip same-domain transfer
                
                logger.info(f"Evaluating cross-domain transfer: {source_name} -> {target_name}")
                
                evaluator = TransferLearningEvaluator(self.config)
                result = evaluator.evaluate_transfer(
                    model, model,  # Use same model architecture
                    source_data, target_data,
                    source_name=source_name, target_name=target_name,
                    **kwargs
                )
                
                results.append(result)
                self.results.append(result)
        
        return results
    
    def get_domain_transfer_matrix(self, metric: str = "accuracy") -> pd.DataFrame:
        """Get domain transfer matrix showing transfer performance."""
        if not self.results:
            return pd.DataFrame()
        
        # Get unique domains
        sources = list(set(r.source_dataset for r in self.results))
        targets = list(set(r.target_dataset for r in self.results))
        
        # Create matrix
        matrix_data = []
        for source in sources:
            row = {'Source': source}
            for target in targets:
                # Find result for this source-target pair
                result = next(
                    (r for r in self.results 
                     if r.source_dataset == source and r.target_dataset == target),
                    None
                )
                
                if result and metric in result.pretrained_metrics:
                    row[target] = result.pretrained_metrics[metric]
                else:
                    row[target] = np.nan
            
            matrix_data.append(row)
        
        return pd.DataFrame(matrix_data).set_index('Source')


def evaluate_transfer_learning(
    pretrained_model: Any,
    baseline_model: Any,
    source_datasets: Dict[str, Tuple[pd.DataFrame, pd.Series]],
    target_datasets: Dict[str, Tuple[pd.DataFrame, pd.Series]],
    config: EvaluationConfig = None,
    **kwargs
) -> List[TransferLearningResult]:
    """Comprehensive transfer learning evaluation."""
    
    if config is None:
        config = EvaluationConfig()
    
    evaluator = TransferLearningEvaluator(config)
    all_results = []
    
    for source_name, source_data in source_datasets.items():
        for target_name, target_data in target_datasets.items():
            logger.info(f"Evaluating transfer: {source_name} -> {target_name}")
            
            result = evaluator.evaluate_transfer(
                pretrained_model, baseline_model,
                source_data, target_data,
                source_name=source_name, target_name=target_name,
                **kwargs
            )
            
            all_results.append(result)
    
    return all_results


def create_domain_datasets(
    domains: List[str] = None,
    n_samples_per_domain: int = 1000,
    n_features: int = 20,
    task_type: str = "classification"
) -> Dict[str, Tuple[pd.DataFrame, pd.Series]]:
    """Create synthetic datasets for different domains."""
    
    if domains is None:
        domains = ['healthcare', 'finance', 'retail', 'manufacturing']
    
    datasets = {}
    
    for i, domain in enumerate(domains):
        # Create domain-specific synthetic data with different characteristics
        np.random.seed(42 + i)  # Different seed for each domain
        
        if task_type == "classification":
            # Vary the class separation and feature correlations by domain
            class_sep = 0.5 + i * 0.3  # Different separability
            n_informative = max(5, n_features // (2 + i))  # Different informativeness
            
            X, y = create_synthetic_dataset(
                task_type="classification",
                n_samples=n_samples_per_domain,
                n_features=n_features,
                n_classes=2,
                n_informative=n_informative,
                class_sep=class_sep,
                random_state=42 + i
            )
        else:
            # Vary noise and feature relationships for regression
            noise = 0.1 + i * 0.1  # Different noise levels
            
            X, y = create_synthetic_dataset(
                task_type="regression",
                n_samples=n_samples_per_domain,
                n_features=n_features,
                noise=noise,
                random_state=42 + i
            )
        
        datasets[domain] = (X, y)
    
    return datasets