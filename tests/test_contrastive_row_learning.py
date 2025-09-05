"""Tests for Contrastive Row Learning pre-training objective."""

import pytest
import torch
import numpy as np
import pandas as pd

from tabgpt.training.contrastive_row_learning import (
    ContrastiveRowLearningObjective,
    InfoNCELoss,
    NoiseInjectionAugmentation,
    FeatureDropoutAugmentation,
    ValuePerturbationAugmentation,
    FeatureShuffleAugmentation,
    CutMixAugmentation,
    MultiAugmentationStrategy,
    ContrastiveOutput,
    AugmentationType,
    create_default_augmentation_strategies
)
from tabgpt.tokenizers import TabularTokenizer
from tabgpt.models import TabGPTModel
from tabgpt.config import TabGPTConfig


class TestNoiseInjectionAugmentation:
    """Test noise injection augmentation strategy."""
    
    def test_noise_injection_creation(self):
        """Test noise injection augmentation creation."""
        aug = NoiseInjectionAugmentation(
            augmentation_probability=0.5,
            noise_std=0.1,
            numerical_only=True
        )
        assert aug.augmentation_probability == 0.5
        assert aug.noise_std == 0.1
        assert aug.numerical_only == True
    
    def test_noise_injection_forward(self):
        """Test noise injection forward pass."""
        batch_size, n_features, d_model = 4, 5, 64
        
        aug = NoiseInjectionAugmentation(
            augmentation_probability=1.0,  # Always augment
            noise_std=0.1
        )
        
        input_features = torch.randn(batch_size, n_features, d_model)
        attention_mask = torch.ones(batch_size, n_features, dtype=torch.bool)
        
        augmented = aug(input_features, attention_mask=attention_mask)
        
        assert augmented.shape == input_features.shape
        # Should be different due to noise (with high probability)
        assert not torch.allclose(augmented, input_features, atol=1e-6)
    
    def test_noise_injection_no_augmentation(self):
        """Test noise injection with zero probability."""
        batch_size, n_features, d_model = 2, 3, 32
        
        aug = NoiseInjectionAugmentation(augmentation_probability=0.0)  # Never augment
        
        input_features = torch.randn(batch_size, n_features, d_model)
        augmented = aug(input_features)
        
        # Should be identical when no augmentation
        assert torch.allclose(augmented, input_features)
    
    def test_noise_injection_with_metadata(self):
        """Test noise injection with column metadata."""
        batch_size, n_features, d_model = 2, 4, 32
        
        aug = NoiseInjectionAugmentation(
            augmentation_probability=1.0,
            numerical_only=True
        )
        
        input_features = torch.randn(batch_size, n_features, d_model)
        
        from tabgpt.tokenizers.tabular_tokenizer import ColumnMetadata
        column_metadata = [
            ColumnMetadata(name='num1', dtype='numerical', unique_values=None),
            ColumnMetadata(name='cat1', dtype='categorical', unique_values=5),
            ColumnMetadata(name='num2', dtype='numerical', unique_values=None),
            ColumnMetadata(name='bool1', dtype='boolean', unique_values=2)
        ]
        
        augmented = aug(input_features, column_metadata=column_metadata)
        
        assert augmented.shape == input_features.shape
        assert not torch.allclose(augmented, input_features)


class TestFeatureDropoutAugmentation:
    """Test feature dropout augmentation strategy."""
    
    def test_feature_dropout_creation(self):
        """Test feature dropout augmentation creation."""
        aug = FeatureDropoutAugmentation(
            augmentation_probability=0.5,
            dropout_probability=0.2,
            min_features=1
        )
        assert aug.augmentation_probability == 0.5
        assert aug.dropout_probability == 0.2
        assert aug.min_features == 1
    
    def test_feature_dropout_forward(self):
        """Test feature dropout forward pass."""
        batch_size, n_features, d_model = 3, 6, 32
        
        aug = FeatureDropoutAugmentation(
            augmentation_probability=1.0,  # Always augment
            dropout_probability=0.5,
            min_features=2
        )
        
        input_features = torch.randn(batch_size, n_features, d_model)
        attention_mask = torch.ones(batch_size, n_features, dtype=torch.bool)
        
        augmented = aug(input_features, attention_mask=attention_mask)
        
        assert augmented.shape == input_features.shape
        
        # Check that some features are zeroed out
        for i in range(batch_size):
            zero_features = (augmented[i] == 0).all(dim=-1).sum()
            non_zero_features = n_features - zero_features
            assert non_zero_features >= aug.min_features
    
    def test_feature_dropout_min_features(self):
        """Test that minimum features are preserved."""
        batch_size, n_features, d_model = 2, 3, 16
        
        aug = FeatureDropoutAugmentation(
            augmentation_probability=1.0,
            dropout_probability=1.0,  # Try to drop everything
            min_features=2
        )
        
        input_features = torch.randn(batch_size, n_features, d_model)
        augmented = aug(input_features)
        
        # Should preserve at least min_features
        for i in range(batch_size):
            non_zero_features = (~(augmented[i] == 0).all(dim=-1)).sum()
            assert non_zero_features >= aug.min_features


class TestValuePerturbationAugmentation:
    """Test value perturbation augmentation strategy."""
    
    def test_value_perturbation_creation(self):
        """Test value perturbation augmentation creation."""
        aug = ValuePerturbationAugmentation(
            augmentation_probability=0.6,
            numerical_perturbation_std=0.05,
            categorical_swap_probability=0.1
        )
        assert aug.augmentation_probability == 0.6
        assert aug.numerical_perturbation_std == 0.05
        assert aug.categorical_swap_probability == 0.1
    
    def test_value_perturbation_forward(self):
        """Test value perturbation forward pass."""
        batch_size, n_features, d_model = 2, 4, 32
        
        aug = ValuePerturbationAugmentation(
            augmentation_probability=1.0,
            numerical_perturbation_std=0.1
        )
        
        input_features = torch.randn(batch_size, n_features, d_model)
        
        from tabgpt.tokenizers.tabular_tokenizer import ColumnMetadata
        column_metadata = [
            ColumnMetadata(name='num1', dtype='numerical', unique_values=None),
            ColumnMetadata(name='cat1', dtype='categorical', unique_values=5),
            ColumnMetadata(name='num2', dtype='numerical', unique_values=None),
            ColumnMetadata(name='bool1', dtype='boolean', unique_values=2)
        ]
        
        augmented = aug(input_features, column_metadata=column_metadata)
        
        assert augmented.shape == input_features.shape
        # Should be different due to perturbation
        assert not torch.allclose(augmented, input_features, atol=1e-6)


class TestFeatureShuffleAugmentation:
    """Test feature shuffle augmentation strategy."""
    
    def test_feature_shuffle_creation(self):
        """Test feature shuffle augmentation creation."""
        aug = FeatureShuffleAugmentation(
            augmentation_probability=0.5,
            shuffle_ratio=0.3
        )
        assert aug.augmentation_probability == 0.5
        assert aug.shuffle_ratio == 0.3
    
    def test_feature_shuffle_forward(self):
        """Test feature shuffle forward pass."""
        batch_size, n_features, d_model = 2, 6, 32
        
        aug = FeatureShuffleAugmentation(
            augmentation_probability=1.0,  # Always augment
            shuffle_ratio=0.5
        )
        
        # Create distinctive features for easy verification
        input_features = torch.zeros(batch_size, n_features, d_model)
        for i in range(n_features):
            input_features[:, i, :] = i + 1  # Each feature has unique values
        
        augmented = aug(input_features)
        
        assert augmented.shape == input_features.shape
        # Should be different due to shuffling (with high probability)
        different_samples = ~torch.allclose(augmented, input_features, atol=1e-6)
        assert different_samples  # At least some samples should be different


class TestCutMixAugmentation:
    """Test CutMix augmentation strategy."""
    
    def test_cutmix_creation(self):
        """Test CutMix augmentation creation."""
        aug = CutMixAugmentation(
            augmentation_probability=0.5,
            mix_ratio=0.3,
            same_batch_only=True
        )
        assert aug.augmentation_probability == 0.5
        assert aug.mix_ratio == 0.3
        assert aug.same_batch_only == True
    
    def test_cutmix_forward(self):
        """Test CutMix forward pass."""
        batch_size, n_features, d_model = 4, 5, 32
        
        aug = CutMixAugmentation(
            augmentation_probability=1.0,  # Always augment
            mix_ratio=0.4
        )
        
        # Create distinctive features for each sample
        input_features = torch.zeros(batch_size, n_features, d_model)
        for i in range(batch_size):
            input_features[i, :, :] = (i + 1) * 10  # Each sample has unique values
        
        augmented = aug(input_features)
        
        assert augmented.shape == input_features.shape
        # Should be different due to mixing (with high probability)
        assert not torch.allclose(augmented, input_features, atol=1e-6)
    
    def test_cutmix_single_sample(self):
        """Test CutMix with single sample (should not change)."""
        batch_size, n_features, d_model = 1, 4, 32
        
        aug = CutMixAugmentation(augmentation_probability=1.0)
        
        input_features = torch.randn(batch_size, n_features, d_model)
        augmented = aug(input_features)
        
        # Should be identical with single sample
        assert torch.allclose(augmented, input_features)


class TestMultiAugmentationStrategy:
    """Test multi-augmentation strategy."""
    
    def test_multi_augmentation_creation(self):
        """Test multi-augmentation strategy creation."""
        strategies = [
            NoiseInjectionAugmentation(),
            FeatureDropoutAugmentation()
        ]
        
        multi_aug = MultiAugmentationStrategy(
            augmentation_strategies=strategies,
            max_augmentations=2
        )
        
        assert len(multi_aug.augmentation_strategies) == 2
        assert multi_aug.max_augmentations == 2
    
    def test_multi_augmentation_forward(self):
        """Test multi-augmentation forward pass."""
        batch_size, n_features, d_model = 2, 4, 32
        
        strategies = [
            NoiseInjectionAugmentation(augmentation_probability=1.0, noise_std=0.1),
            FeatureDropoutAugmentation(augmentation_probability=1.0, dropout_probability=0.2)
        ]
        
        multi_aug = MultiAugmentationStrategy(
            augmentation_strategies=strategies,
            max_augmentations=2
        )
        
        input_features = torch.randn(batch_size, n_features, d_model)
        augmented = multi_aug(input_features)
        
        assert augmented.shape == input_features.shape
        # Should be different due to multiple augmentations
        assert not torch.allclose(augmented, input_features, atol=1e-6)


class TestInfoNCELoss:
    """Test InfoNCE loss function."""
    
    def test_infonce_creation(self):
        """Test InfoNCE loss creation."""
        loss_fn = InfoNCELoss(temperature=0.07, normalize_embeddings=True)
        assert loss_fn.temperature == 0.07
        assert loss_fn.normalize_embeddings == True
    
    def test_infonce_forward_in_batch_negatives(self):
        """Test InfoNCE loss with in-batch negatives."""
        batch_size, d_model = 4, 64
        
        loss_fn = InfoNCELoss(temperature=0.1)
        
        anchor_embeddings = torch.randn(batch_size, d_model)
        positive_embeddings = torch.randn(batch_size, d_model)
        
        loss, logits, labels = loss_fn(anchor_embeddings, positive_embeddings)
        
        assert loss.numel() == 1  # Scalar loss
        assert logits.shape[0] == batch_size
        assert logits.shape[1] > 1  # Should have positive + negatives
        assert labels.shape == (batch_size,)
        assert (labels == 0).all()  # All labels should be 0 (positive at index 0)
        assert not torch.isnan(loss)
    
    def test_infonce_forward_with_negatives(self):
        """Test InfoNCE loss with provided negative embeddings."""
        batch_size, d_model = 3, 32
        n_negatives = 10
        
        loss_fn = InfoNCELoss(temperature=0.05)
        
        anchor_embeddings = torch.randn(batch_size, d_model)
        positive_embeddings = torch.randn(batch_size, d_model)
        negative_embeddings = torch.randn(n_negatives, d_model)
        
        loss, logits, labels = loss_fn(anchor_embeddings, positive_embeddings, negative_embeddings)
        
        assert loss.numel() == 1
        assert logits.shape == (batch_size, 1 + n_negatives)  # 1 positive + n_negatives
        assert labels.shape == (batch_size,)
        assert (labels == 0).all()
        assert not torch.isnan(loss)
    
    def test_infonce_normalization(self):
        """Test InfoNCE loss with and without normalization."""
        batch_size, d_model = 2, 16
        
        anchor_embeddings = torch.randn(batch_size, d_model) * 10  # Large magnitude
        positive_embeddings = torch.randn(batch_size, d_model) * 10
        
        # With normalization
        loss_fn_norm = InfoNCELoss(normalize_embeddings=True)
        loss_norm, _, _ = loss_fn_norm(anchor_embeddings, positive_embeddings)
        
        # Without normalization
        loss_fn_no_norm = InfoNCELoss(normalize_embeddings=False)
        loss_no_norm, _, _ = loss_fn_no_norm(anchor_embeddings, positive_embeddings)
        
        # Both should be valid losses
        assert not torch.isnan(loss_norm)
        assert not torch.isnan(loss_no_norm)
        # They should be different due to normalization
        assert not torch.allclose(loss_norm, loss_no_norm, atol=1e-6)


class TestContrastiveRowLearningObjective:
    """Test complete contrastive row learning objective."""
    
    def test_crl_objective_creation(self):
        """Test CRL objective creation."""
        objective = ContrastiveRowLearningObjective(
            d_model=128,
            temperature=0.07,
            use_projection_head=True
        )
        assert objective.d_model == 128
        assert objective.temperature == 0.07
        assert objective.use_projection_head == True
    
    def test_crl_objective_creation_custom_augmentations(self):
        """Test CRL objective with custom augmentations."""
        custom_augs = [
            NoiseInjectionAugmentation(noise_std=0.2),
            FeatureDropoutAugmentation(dropout_probability=0.1)
        ]
        
        objective = ContrastiveRowLearningObjective(
            d_model=64,
            augmentation_strategies=custom_augs,
            projection_dim=32
        )
        
        assert len(objective.augmentation_strategy.augmentation_strategies) == 2
    
    def test_create_augmented_pairs(self):
        """Test augmented pair creation."""
        batch_size, n_features, d_model = 3, 5, 64
        
        objective = ContrastiveRowLearningObjective(d_model=d_model)
        
        input_features = torch.randn(batch_size, n_features, d_model)
        attention_mask = torch.ones(batch_size, n_features, dtype=torch.bool)
        
        aug_v1, aug_v2 = objective.create_augmented_pairs(
            input_features,
            attention_mask=attention_mask
        )
        
        assert aug_v1.shape == input_features.shape
        assert aug_v2.shape == input_features.shape
        # The two augmented versions should be different
        assert not torch.allclose(aug_v1, aug_v2, atol=1e-6)
    
    def test_crl_objective_forward_basic(self):
        """Test CRL objective basic forward pass."""
        batch_size, n_features, d_model = 4, 6, 64
        
        objective = ContrastiveRowLearningObjective(
            d_model=d_model,
            temperature=0.1
        )
        
        row_embeddings = torch.randn(batch_size, d_model)
        input_features = torch.randn(batch_size, n_features, d_model)
        attention_mask = torch.ones(batch_size, n_features, dtype=torch.bool)
        
        output = objective(
            row_embeddings=row_embeddings,
            input_features=input_features,
            attention_mask=attention_mask
        )
        
        assert isinstance(output, ContrastiveOutput)
        assert output.loss is not None
        assert not torch.isnan(output.loss)
        assert output.logits.shape[0] == batch_size
        assert output.labels.shape == (batch_size,)
        assert output.temperature == 0.1
        assert 'contrastive_accuracy' in output.accuracy
    
    def test_crl_objective_forward_with_model_fn(self):
        """Test CRL objective with model forward function."""
        batch_size, n_features, d_model = 2, 4, 32
        
        objective = ContrastiveRowLearningObjective(d_model=d_model)
        
        # Mock model forward function
        def mock_model_fn(features, mask):
            return features.mean(dim=1)  # Simple aggregation
        
        row_embeddings = torch.randn(batch_size, d_model)
        input_features = torch.randn(batch_size, n_features, d_model)
        
        output = objective(
            row_embeddings=row_embeddings,
            input_features=input_features,
            model_forward_fn=mock_model_fn
        )
        
        assert isinstance(output, ContrastiveOutput)
        assert output.loss is not None
        assert not torch.isnan(output.loss)
    
    def test_compute_metrics(self):
        """Test metrics computation."""
        objective = ContrastiveRowLearningObjective(d_model=64)
        
        # Create mock output
        output = ContrastiveOutput(
            loss=torch.tensor(1.5),
            logits=torch.randn(4, 5),  # 4 samples, 5 classes (1 pos + 4 neg)
            labels=torch.zeros(4, dtype=torch.long),
            positive_pairs=torch.randn(4, 2, 64),
            negative_pairs=torch.randn(4, 64),
            temperature=0.07,
            accuracy={
                'contrastive_accuracy': 0.75,
                'positive_similarity': 0.8,
                'negative_similarity': 0.2,
                'temperature': 0.07,
                'batch_size': 4
            }
        )
        
        metrics = objective.compute_metrics(output)
        
        assert 'crl_loss' in metrics
        assert 'contrastive_accuracy' in metrics
        assert 'positive_similarity' in metrics
        assert 'negative_similarity' in metrics
        assert 'similarity_gap' in metrics
        assert 'temperature' in metrics
        assert 'batch_size' in metrics
        assert 'logit_mean' in metrics
        assert 'logit_std' in metrics
        
        assert metrics['crl_loss'] == 1.5
        assert metrics['contrastive_accuracy'] == 0.75
        assert metrics['similarity_gap'] == 0.6  # 0.8 - 0.2


class TestCRLIntegration:
    """Test CRL integration with TabGPT model."""
    
    def test_crl_with_tabgpt_model(self):
        """Test CRL objective with actual TabGPT model."""
        # Create test data
        df = pd.DataFrame({
            'feature1': [1.0, 2.5, 3.2, 1.8, 2.1],
            'feature2': ['A', 'B', 'C', 'A', 'B'],
            'feature3': [True, False, True, False, True],
            'feature4': [10.5, 20.3, 15.7, 12.1, 18.9]
        })
        
        # Tokenize data
        tokenizer = TabularTokenizer(embedding_dim=64)
        tokenized = tokenizer.fit_transform(df)
        
        # Create model
        config = TabGPTConfig(
            d_model=64,
            n_heads=4,
            n_layers=2,
            embedding_dim=64,
            max_features=10,
            column_embedding_dim=64
        )
        model = TabGPTModel(config)
        
        # Get model outputs
        with torch.no_grad():
            model_outputs = model(
                input_features=tokenized.tokens,
                attention_mask=tokenized.attention_mask
            )
        
        # Use pooled output as row embeddings
        row_embeddings = model_outputs['pooled_output']
        
        # Create CRL objective
        crl_objective = ContrastiveRowLearningObjective(
            d_model=64,
            temperature=0.1
        )
        
        # Forward pass
        output = crl_objective(
            row_embeddings=row_embeddings,
            input_features=tokenized.tokens,
            attention_mask=tokenized.attention_mask,
            column_metadata=tokenizer.column_metadata
        )
        
        assert isinstance(output, ContrastiveOutput)
        assert output.loss is not None
        assert not torch.isnan(output.loss)
        
        # Test metrics computation
        metrics = crl_objective.compute_metrics(output)
        assert isinstance(metrics, dict)
        assert 'crl_loss' in metrics
    
    def test_crl_gradient_flow(self):
        """Test gradient flow through CRL objective."""
        batch_size, n_features, d_model = 3, 4, 32
        
        # Create CRL objective
        crl_objective = ContrastiveRowLearningObjective(d_model=d_model)
        
        row_embeddings = torch.randn(batch_size, d_model, requires_grad=True)
        input_features = torch.randn(batch_size, n_features, d_model, requires_grad=True)
        
        # Forward pass
        output = crl_objective(
            row_embeddings=row_embeddings,
            input_features=input_features
        )
        
        # Backward pass
        output.loss.backward()
        
        # Check gradients
        assert row_embeddings.grad is not None
        assert input_features.grad is not None
        assert not torch.isnan(row_embeddings.grad).any()
        assert not torch.isnan(input_features.grad).any()
        
        # Check that CRL parameters have gradients
        crl_params_with_grad = [p for p in crl_objective.parameters() if p.grad is not None]
        assert len(crl_params_with_grad) > 0
        
        for param in crl_params_with_grad:
            assert not torch.isnan(param.grad).any()
    
    def test_crl_different_temperatures(self):
        """Test CRL with different temperature values."""
        batch_size, n_features, d_model = 2, 3, 32
        
        row_embeddings = torch.randn(batch_size, d_model)
        input_features = torch.randn(batch_size, n_features, d_model)
        
        temperatures = [0.01, 0.05, 0.1, 0.2, 0.5]
        losses = []
        
        for temp in temperatures:
            objective = ContrastiveRowLearningObjective(
                d_model=d_model,
                temperature=temp
            )
            
            output = objective(
                row_embeddings=row_embeddings,
                input_features=input_features
            )
            
            losses.append(output.loss.item())
            assert output.temperature == temp
        
        # Different temperatures should produce different losses
        assert len(set(losses)) > 1
    
    def test_crl_batch_size_robustness(self):
        """Test CRL with different batch sizes."""
        n_features, d_model = 4, 32
        
        batch_sizes = [1, 2, 4, 8]
        
        for batch_size in batch_sizes:
            objective = ContrastiveRowLearningObjective(d_model=d_model)
            
            row_embeddings = torch.randn(batch_size, d_model)
            input_features = torch.randn(batch_size, n_features, d_model)
            
            output = objective(
                row_embeddings=row_embeddings,
                input_features=input_features
            )
            
            assert isinstance(output, ContrastiveOutput)
            assert output.loss is not None
            assert not torch.isnan(output.loss)
            assert output.accuracy['batch_size'] == batch_size


class TestDefaultAugmentationStrategies:
    """Test default augmentation strategies creation."""
    
    def test_create_default_augmentation_strategies(self):
        """Test creation of default augmentation strategies."""
        strategies = create_default_augmentation_strategies()
        
        assert len(strategies) == 5  # Should have 5 default strategies
        assert isinstance(strategies[0], NoiseInjectionAugmentation)
        assert isinstance(strategies[1], FeatureDropoutAugmentation)
        assert isinstance(strategies[2], ValuePerturbationAugmentation)
        assert isinstance(strategies[3], FeatureShuffleAugmentation)
        assert isinstance(strategies[4], CutMixAugmentation)
    
    def test_create_default_augmentation_strategies_custom_params(self):
        """Test creation with custom parameters."""
        strategies = create_default_augmentation_strategies(
            noise_std=0.2,
            dropout_prob=0.3,
            perturbation_std=0.1,
            shuffle_ratio=0.4,
            mix_ratio=0.5
        )
        
        assert len(strategies) == 5
        assert strategies[0].noise_std == 0.2
        assert strategies[1].dropout_probability == 0.3
        assert strategies[2].numerical_perturbation_std == 0.1
        assert strategies[3].shuffle_ratio == 0.4
        assert strategies[4].mix_ratio == 0.5