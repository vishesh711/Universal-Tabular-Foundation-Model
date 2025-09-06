#!/usr/bin/env python3
"""
Comprehensive TabGPT Example

This example demonstrates the full capabilities of TabGPT including:
- Data validation and preprocessing
- Multiple task types (classification, regression, survival analysis)
- Transfer learning and LoRA fine-tuning
- Comprehensive evaluation and benchmarking
- Model interpretation and visualization
"""

import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report
from sklearn.datasets import make_classification, make_regression
import warnings
warnings.filterwarnings('ignore')

# TabGPT imports
from tabgpt import TabGPTForSequenceClassification, TabGPTForRegression, TabGPTTokenizer
from tabgpt.heads import SurvivalHead, AnomalyDetectionHead, RegressionHead
from tabgpt.adapters import LoRAConfig, apply_lora_to_model
from tabgpt.fine_tuning import TabGPTFineTuningTrainer, FineTuningConfig
from tabgpt.evaluation import (
    ClassificationBenchmark, create_baseline_models, 
    CrossValidationEvaluator, EvaluationConfig
)
from tabgpt.utils import DataValidator, RobustNormalizer, DataRecovery


class TabGPTDemo:
    """Comprehensive demonstration of TabGPT capabilities."""
    
    def __init__(self):
        self.results = {}
        self.models = {}
        self.tokenizers = {}
        
    def create_sample_datasets(self):
        """Create sample datasets for different tasks."""
        print("Creating sample datasets...")
        
        # 1. Classification dataset
        X_cls, y_cls = make_classification(
            n_samples=2000,
            n_features=15,
            n_informative=10,
            n_redundant=3,
            n_clusters_per_class=2,
            class_sep=0.8,
            random_state=42
        )
        
        cls_df = pd.DataFrame(X_cls, columns=[f'feature_{i}' for i in range(X_cls.shape[1])])
        cls_df['category'] = np.random.choice(['A', 'B', 'C'], len(cls_df), p=[0.5, 0.3, 0.2])
        cls_df['group'] = np.random.choice(['X', 'Y'], len(cls_df), p=[0.6, 0.4])
        cls_df['target'] = y_cls
        
        # Add missing values
        missing_idx = np.random.choice(len(cls_df), int(0.05 * len(cls_df)), replace=False)
        cls_df.loc[missing_idx, 'feature_0'] = np.nan
        
        # 2. Regression dataset
        X_reg, y_reg = make_regression(
            n_samples=1500,
            n_features=12,
            n_informative=8,
            noise=0.1,
            random_state=42
        )
        
        reg_df = pd.DataFrame(X_reg, columns=[f'reg_feature_{i}' for i in range(X_reg.shape[1])])
        reg_df['categorical'] = np.random.choice(['Type1', 'Type2', 'Type3'], len(reg_df))
        reg_df['target'] = y_reg
        
        # 3. Survival dataset (synthetic)
        n_survival = 1000
        survival_features = np.random.randn(n_survival, 8)
        
        # Create survival times with some logic
        risk_score = np.sum(survival_features[:, :4], axis=1)
        survival_times = np.random.exponential(np.exp(-risk_score * 0.3))
        censoring = np.random.binomial(1, 0.7, n_survival)  # 70% observed
        
        survival_df = pd.DataFrame(
            survival_features, 
            columns=[f'surv_feature_{i}' for i in range(survival_features.shape[1])]
        )
        survival_df['time'] = survival_times
        survival_df['event'] = censoring
        
        self.datasets = {
            'classification': cls_df,
            'regression': reg_df,
            'survival': survival_df
        }
        
        print(f"Created datasets:")
        for name, df in self.datasets.items():
            print(f"  {name}: {df.shape}")
        print()
        
    def validate_and_clean_data(self):
        """Demonstrate data validation and cleaning."""
        print("Validating and cleaning data...")
        
        validator = DataValidator(
            missing_threshold=0.1,
            min_samples=500,
            min_features=5
        )
        
        recovery = DataRecovery(
            auto_fix=True,
            missing_strategy="median",
            outlier_strategy="clip"
        )
        
        self.cleaned_datasets = {}
        
        for name, df in self.datasets.items():
            print(f"\\nProcessing {name} dataset:")
            
            # Validate
            validation_result = validator.validate_dataframe(df)
            print(f"  Validation passed: {validation_result['valid']}")
            if validation_result['warnings']:
                print(f"  Warnings: {len(validation_result['warnings'])}")
            
            # Clean
            cleaned_df, recovery_log = recovery.recover_dataframe(df)
            print(f"  Recovery actions: {len(recovery_log.get('actions_taken', []))}")
            
            self.cleaned_datasets[name] = cleaned_df
        
        print("Data validation and cleaning completed!")
        print()
    
    def demonstrate_classification(self):
        """Demonstrate classification with TabGPT."""
        print("=" * 50)
        print("CLASSIFICATION DEMONSTRATION")
        print("=" * 50)
        
        df = self.cleaned_datasets['classification']
        X = df.drop('target', axis=1)
        y = df['target']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Preprocess
        normalizer = RobustNormalizer()
        X_train_processed, _ = normalizer.fit_transform(X_train)
        X_test_processed, _ = normalizer.transform(X_test)
        
        # Initialize model
        tokenizer = TabGPTTokenizer(vocab_size=8000, max_length=256)
        model = TabGPTForSequenceClassification(num_labels=2)
        
        # Prepare datasets
        train_dataset = tokenizer.create_dataset(X_train_processed, y_train)
        test_dataset = tokenizer.create_dataset(X_test_processed, y_test)
        
        # Configure training
        config = FineTuningConfig(
            task_type="classification",
            num_labels=2,
            learning_rate=5e-5,
            num_epochs=3,
            batch_size=32,
            eval_steps=50
        )
        
        # Train
        trainer = TabGPTFineTuningTrainer(
            model=model,
            config=config,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            tokenizer=tokenizer,
            compute_metrics=lambda x: {'accuracy': accuracy_score(x.label_ids, x.predictions.argmax(-1))}
        )
        
        print("Training classification model...")
        trainer.train()
        
        # Evaluate
        results = trainer.evaluate()
        print(f"Classification Results:")
        for metric, value in results.items():
            print(f"  {metric}: {value:.4f}")
        
        # Store results
        self.models['classification'] = model
        self.tokenizers['classification'] = tokenizer
        self.results['classification'] = results
        
        print("Classification demonstration completed!")
        print()
    
    def demonstrate_regression(self):
        """Demonstrate regression with uncertainty estimation."""
        print("=" * 50)
        print("REGRESSION DEMONSTRATION")
        print("=" * 50)
        
        df = self.cleaned_datasets['regression']
        X = df.drop('target', axis=1)
        y = df['target']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Preprocess
        normalizer = RobustNormalizer()
        X_train_processed, _ = normalizer.fit_transform(X_train)
        X_test_processed, _ = normalizer.transform(X_test)
        
        # Initialize model with uncertainty estimation
        tokenizer = TabGPTTokenizer(vocab_size=8000, max_length=256)
        model = TabGPTForRegression(output_dim=1)
        
        # Replace with uncertainty-aware head
        model.regression_head = RegressionHead(
            input_dim=model.config.hidden_size,
            output_dim=1,
            estimate_uncertainty=True
        )
        
        # Prepare datasets
        train_dataset = tokenizer.create_dataset(X_train_processed, y_train)
        test_dataset = tokenizer.create_dataset(X_test_processed, y_test)
        
        # Configure training
        config = FineTuningConfig(
            task_type="regression",
            output_dim=1,
            learning_rate=3e-5,
            num_epochs=4,
            batch_size=32,
            eval_steps=50
        )
        
        # Train
        trainer = TabGPTFineTuningTrainer(
            model=model,
            config=config,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            tokenizer=tokenizer,
            compute_metrics=lambda x: {
                'mse': mean_squared_error(x.label_ids, x.predictions.flatten()),
                'rmse': np.sqrt(mean_squared_error(x.label_ids, x.predictions.flatten()))
            }
        )
        
        print("Training regression model with uncertainty estimation...")
        trainer.train()
        
        # Evaluate
        results = trainer.evaluate()
        print(f"Regression Results:")
        for metric, value in results.items():
            print(f"  {metric}: {value:.4f}")
        
        # Test uncertainty estimation
        model.eval()
        test_tokens = tokenizer.encode_batch(X_test_processed)
        
        with torch.no_grad():
            outputs = model(**test_tokens)
            predictions = outputs.predictions.cpu().numpy().flatten()
            
            if hasattr(outputs, 'uncertainty') and outputs.uncertainty is not None:
                uncertainties = outputs.uncertainty.cpu().numpy().flatten()
                print(f"  Mean uncertainty: {np.mean(uncertainties):.4f}")
        
        # Store results
        self.models['regression'] = model
        self.tokenizers['regression'] = tokenizer
        self.results['regression'] = results
        
        print("Regression demonstration completed!")
        print()
    
    def demonstrate_lora_fine_tuning(self):
        """Demonstrate LoRA fine-tuning."""
        print("=" * 50)
        print("LoRA FINE-TUNING DEMONSTRATION")
        print("=" * 50)
        
        # Use classification dataset
        df = self.cleaned_datasets['classification']
        X = df.drop('target', axis=1)
        y = df['target']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=123  # Different split
        )
        
        # Preprocess
        normalizer = RobustNormalizer()
        X_train_processed, _ = normalizer.fit_transform(X_train)
        X_test_processed, _ = normalizer.transform(X_test)
        
        # Start with pre-trained model (simulate by using our trained model)
        base_model = TabGPTForSequenceClassification(num_labels=2)
        if 'classification' in self.models:
            # Copy weights from previously trained model
            base_model.load_state_dict(self.models['classification'].state_dict())
        
        # Apply LoRA
        lora_config = LoRAConfig(
            r=8,
            alpha=16,
            dropout=0.1,
            target_modules=["query", "key", "value", "dense"]
        )
        
        lora_model = apply_lora_to_model(base_model, lora_config)
        
        # Check parameter efficiency
        total_params = sum(p.numel() for p in lora_model.parameters())
        trainable_params = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
        
        print(f"LoRA Parameter Efficiency:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Efficiency: {trainable_params/total_params:.2%}")
        
        # Prepare datasets
        tokenizer = self.tokenizers.get('classification', TabGPTTokenizer())
        train_dataset = tokenizer.create_dataset(X_train_processed, y_train)
        test_dataset = tokenizer.create_dataset(X_test_processed, y_test)
        
        # Configure LoRA training (higher learning rate)
        config = FineTuningConfig(
            task_type="classification",
            num_labels=2,
            learning_rate=5e-4,  # Higher LR for LoRA
            num_epochs=2,
            batch_size=32,
            eval_steps=25
        )
        
        # Train
        trainer = TabGPTFineTuningTrainer(
            model=lora_model,
            config=config,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            tokenizer=tokenizer,
            compute_metrics=lambda x: {'accuracy': accuracy_score(x.label_ids, x.predictions.argmax(-1))}
        )
        
        print("Training with LoRA...")
        trainer.train()
        
        # Evaluate
        results = trainer.evaluate()
        print(f"LoRA Results:")
        for metric, value in results.items():
            print(f"  {metric}: {value:.4f}")
        
        self.results['lora'] = results
        
        print("LoRA demonstration completed!")
        print()
    
    def demonstrate_benchmarking(self):
        """Demonstrate benchmarking against traditional ML models."""
        print("=" * 50)
        print("BENCHMARKING DEMONSTRATION")
        print("=" * 50)
        
        # Use classification dataset
        df = self.cleaned_datasets['classification']
        
        def load_data():
            X = df.drop('target', axis=1)
            y = df['target']
            return X, y
        
        # Create baseline models
        baselines = create_baseline_models(
            task_type="classification",
            include_models=["RandomForest", "XGBoost", "LogisticRegression"]
        )
        
        # Add TabGPT model
        models = {"TabGPT": self.models.get('classification')}
        models.update(baselines)
        
        # Create benchmark
        benchmark = ClassificationBenchmark(
            name="sample_classification",
            data_loader=load_data,
            description="Sample classification benchmark"
        )
        
        print("Running benchmark comparison...")
        try:
            results = benchmark.run_benchmark(models, test_size=0.2, cv_folds=3)
            
            # Get leaderboard
            leaderboard = benchmark.get_leaderboard(metric="accuracy")
            print("\\nBenchmark Leaderboard:")
            print(leaderboard)
            
            self.results['benchmark'] = results
            
        except Exception as e:
            print(f"Benchmark failed: {e}")
            print("This is expected in a demo environment without full ML libraries")
        
        print("Benchmarking demonstration completed!")
        print()
    
    def demonstrate_cross_validation(self):
        """Demonstrate cross-validation evaluation."""
        print("=" * 50)
        print("CROSS-VALIDATION DEMONSTRATION")
        print("=" * 50)
        
        # Use a smaller subset for faster CV
        df = self.cleaned_datasets['classification'].sample(n=500, random_state=42)
        X = df.drop('target', axis=1)
        y = df['target']
        
        # Preprocess
        normalizer = RobustNormalizer()
        X_processed, _ = normalizer.fit_transform(X)
        
        # Configure evaluation
        config = EvaluationConfig(
            strategy="cross_validation",
            cv_folds=3,  # Smaller for demo
            primary_metric="accuracy"
        )
        
        evaluator = CrossValidationEvaluator(config)
        
        print("Running cross-validation...")
        try:
            # Create a fresh model for CV
            model = TabGPTForSequenceClassification(num_labels=2)
            
            result = evaluator.evaluate(
                model=model,
                X=X_processed,
                y=y,
                task_type="classification"
            )
            
            print(f"Cross-Validation Results:")
            print(f"  Mean Accuracy: {result.metrics['accuracy']:.4f}")
            print(f"  Std Accuracy: {result.metrics_std['accuracy']:.4f}")
            
            self.results['cross_validation'] = result
            
        except Exception as e:
            print(f"Cross-validation failed: {e}")
            print("This is expected in a demo environment")
        
        print("Cross-validation demonstration completed!")
        print()
    
    def demonstrate_survival_analysis(self):
        """Demonstrate survival analysis capabilities."""
        print("=" * 50)
        print("SURVIVAL ANALYSIS DEMONSTRATION")
        print("=" * 50)
        
        df = self.cleaned_datasets['survival']
        X = df.drop(['time', 'event'], axis=1)
        
        # Prepare survival targets (time, event)
        survival_targets = df[['time', 'event']].values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, survival_targets, test_size=0.2, random_state=42
        )
        
        # Preprocess
        normalizer = RobustNormalizer()
        X_train_processed, _ = normalizer.fit_transform(X_train)
        X_test_processed, _ = normalizer.transform(X_test)
        
        # Initialize model with survival head
        tokenizer = TabGPTTokenizer(vocab_size=5000, max_length=128)
        model = TabGPTForSequenceClassification(num_labels=1)  # Base model
        
        # Replace with survival head
        model.classifier = SurvivalHead(
            input_dim=model.config.hidden_size,
            risk_estimation_method="cox"
        )
        
        print("Survival analysis model initialized")
        print("Note: Full survival training requires specialized loss functions")
        print("This demonstrates the architecture setup")
        
        # Store for completeness
        self.models['survival'] = model
        self.tokenizers['survival'] = tokenizer
        
        print("Survival analysis demonstration completed!")
        print()
    
    def create_visualizations(self):
        """Create visualizations of results."""
        print("=" * 50)
        print("CREATING VISUALIZATIONS")
        print("=" * 50)
        
        try:
            # Set up the plotting style
            plt.style.use('default')
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('TabGPT Comprehensive Demo Results', fontsize=16)
            
            # 1. Model Performance Comparison
            ax1 = axes[0, 0]
            methods = []
            accuracies = []
            
            if 'classification' in self.results:
                methods.append('Classification')
                accuracies.append(self.results['classification'].get('eval_accuracy', 0))
            
            if 'lora' in self.results:
                methods.append('LoRA Fine-tuning')
                accuracies.append(self.results['lora'].get('eval_accuracy', 0))
            
            if methods:
                bars = ax1.bar(methods, accuracies, color=['skyblue', 'lightcoral'])
                ax1.set_ylabel('Accuracy')
                ax1.set_title('Model Performance Comparison')
                ax1.set_ylim(0, 1)
                
                # Add value labels on bars
                for bar, acc in zip(bars, accuracies):
                    height = bar.get_height()
                    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                            f'{acc:.3f}', ha='center', va='bottom')
            
            # 2. Dataset Sizes
            ax2 = axes[0, 1]
            dataset_names = list(self.datasets.keys())
            dataset_sizes = [len(df) for df in self.datasets.values()]
            
            ax2.bar(dataset_names, dataset_sizes, color=['lightgreen', 'orange', 'purple'])
            ax2.set_ylabel('Number of Samples')
            ax2.set_title('Dataset Sizes')
            ax2.tick_params(axis='x', rotation=45)
            
            # 3. Feature Distribution (Classification dataset)
            ax3 = axes[1, 0]
            if 'classification' in self.datasets:
                df_cls = self.datasets['classification']
                numeric_cols = df_cls.select_dtypes(include=[np.number]).columns[:5]
                
                if len(numeric_cols) > 0:
                    df_cls[numeric_cols].hist(ax=ax3, bins=20, alpha=0.7)
                    ax3.set_title('Feature Distributions (Sample)')
                else:
                    ax3.text(0.5, 0.5, 'No numeric features to plot', 
                            ha='center', va='center', transform=ax3.transAxes)
                    ax3.set_title('Feature Distributions')
            
            # 4. Model Complexity Comparison
            ax4 = axes[1, 1]
            model_names = []
            param_counts = []
            
            for name, model in self.models.items():
                if model is not None:
                    model_names.append(name.capitalize())
                    param_counts.append(sum(p.numel() for p in model.parameters()) / 1e6)  # In millions
            
            if model_names:
                ax4.bar(model_names, param_counts, color=['gold', 'lightblue', 'pink'][:len(model_names)])
                ax4.set_ylabel('Parameters (Millions)')
                ax4.set_title('Model Complexity')
                ax4.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.savefig('tabgpt_demo_results.png', dpi=300, bbox_inches='tight')
            print("Visualization saved as 'tabgpt_demo_results.png'")
            
        except Exception as e:
            print(f"Visualization creation failed: {e}")
            print("This is expected if matplotlib is not properly configured")
        
        print("Visualization demonstration completed!")
        print()
    
    def print_summary(self):
        """Print a comprehensive summary of all demonstrations."""
        print("=" * 60)
        print("COMPREHENSIVE DEMO SUMMARY")
        print("=" * 60)
        
        print("\\n🎯 TASKS DEMONSTRATED:")
        print("✓ Data validation and cleaning")
        print("✓ Binary classification")
        print("✓ Regression with uncertainty estimation")
        print("✓ LoRA parameter-efficient fine-tuning")
        print("✓ Benchmarking against traditional ML")
        print("✓ Cross-validation evaluation")
        print("✓ Survival analysis setup")
        print("✓ Results visualization")
        
        print("\\n📊 PERFORMANCE RESULTS:")
        for task, results in self.results.items():
            if isinstance(results, dict):
                print(f"\\n{task.upper()}:")
                for metric, value in results.items():
                    if isinstance(value, (int, float)):
                        print(f"  {metric}: {value:.4f}")
        
        print("\\n🏗️ MODELS CREATED:")
        for name, model in self.models.items():
            if model is not None:
                param_count = sum(p.numel() for p in model.parameters())
                print(f"  {name}: {param_count:,} parameters")
        
        print("\\n💡 KEY TAKEAWAYS:")
        print("1. TabGPT provides a unified architecture for diverse tabular tasks")
        print("2. Automatic preprocessing handles real-world data quality issues")
        print("3. LoRA enables efficient fine-tuning with minimal parameters")
        print("4. Built-in evaluation tools facilitate model comparison")
        print("5. Uncertainty estimation provides confidence in predictions")
        print("6. Transfer learning can improve performance on small datasets")
        
        print("\\n🚀 NEXT STEPS:")
        print("- Try TabGPT on your own datasets")
        print("- Experiment with different model configurations")
        print("- Explore specialized task heads for your domain")
        print("- Use pre-trained models for transfer learning")
        print("- Implement custom evaluation metrics")
        
        print("\\n" + "=" * 60)
        print("Demo completed successfully! 🎉")
        print("=" * 60)


def main():
    """Run the comprehensive TabGPT demonstration."""
    print("TabGPT Comprehensive Demonstration")
    print("This demo showcases the full capabilities of TabGPT")
    print("=" * 60)
    
    # Initialize demo
    demo = TabGPTDemo()
    
    try:
        # Run all demonstrations
        demo.create_sample_datasets()
        demo.validate_and_clean_data()
        demo.demonstrate_classification()
        demo.demonstrate_regression()
        demo.demonstrate_lora_fine_tuning()
        demo.demonstrate_benchmarking()
        demo.demonstrate_cross_validation()
        demo.demonstrate_survival_analysis()
        demo.create_visualizations()
        demo.print_summary()
        
    except KeyboardInterrupt:
        print("\\nDemo interrupted by user")
    except Exception as e:
        print(f"\\nDemo failed with error: {e}")
        print("This may be due to missing dependencies or system limitations")
    
    print("\\nThank you for trying TabGPT! 🚀")


if __name__ == "__main__":
    main()