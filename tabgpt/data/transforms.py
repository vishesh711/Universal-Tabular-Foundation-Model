"""Data transforms for tabular data augmentation and preprocessing."""

import random
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Tuple, Any, Callable
import pandas as pd
import numpy as np
import torch
from dataclasses import dataclass

from .preprocessing import DataType


class TabularTransform(ABC):
    """Base class for tabular data transforms."""
    
    @abstractmethod
    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Apply transform to a sample."""
        pass


class NormalizationTransform(TabularTransform):
    """Normalize numerical features in tokenized data."""
    
    def __init__(self, method: str = "standard", epsilon: float = 1e-8):
        self.method = method
        self.epsilon = epsilon
    
    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Apply normalization to input features."""
        if 'input_features' not in sample:
            return sample
        
        features = sample['input_features'].clone()
        
        if self.method == "standard":
            # Standardize: (x - mean) / std
            mean = features.mean(dim=-1, keepdim=True)
            std = features.std(dim=-1, keepdim=True) + self.epsilon
            features = (features - mean) / std
        
        elif self.method == "minmax":
            # Min-max scaling: (x - min) / (max - min)
            min_vals = features.min(dim=-1, keepdim=True)[0]
            max_vals = features.max(dim=-1, keepdim=True)[0]
            range_vals = max_vals - min_vals + self.epsilon
            features = (features - min_vals) / range_vals
        
        elif self.method == "robust":
            # Robust scaling: (x - median) / IQR
            median = features.median(dim=-1, keepdim=True)[0]
            q75 = features.quantile(0.75, dim=-1, keepdim=True)
            q25 = features.quantile(0.25, dim=-1, keepdim=True)
            iqr = q75 - q25 + self.epsilon
            features = (features - median) / iqr
        
        sample['input_features'] = features
        return sample


class NoiseInjectionTransform(TabularTransform):
    """Add Gaussian noise to numerical features."""
    
    def __init__(
        self,
        noise_std: float = 0.1,
        probability: float = 0.5,
        feature_probability: float = 0.3
    ):
        self.noise_std = noise_std
        self.probability = probability
        self.feature_probability = feature_probability
    
    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Add noise to input features."""
        if 'input_features' not in sample or random.random() > self.probability:
            return sample
        
        features = sample['input_features'].clone()
        
        # Create noise mask
        noise_mask = torch.rand_like(features) < self.feature_probability
        
        # Add Gaussian noise
        noise = torch.randn_like(features) * self.noise_std
        features = torch.where(noise_mask, features + noise, features)
        
        sample['input_features'] = features
        return sample


class FeatureDropoutTransform(TabularTransform):
    """Randomly dropout (zero out) features."""
    
    def __init__(
        self,
        dropout_probability: float = 0.1,
        probability: float = 0.5,
        min_features: int = 1
    ):
        self.dropout_probability = dropout_probability
        self.probability = probability
        self.min_features = min_features
    
    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Apply feature dropout."""
        if 'input_features' not in sample or random.random() > self.probability:
            return sample
        
        features = sample['input_features'].clone()
        attention_mask = sample.get('attention_mask', torch.ones(features.shape[:-1], dtype=torch.bool))
        
        # Determine available features
        available_features = attention_mask.sum().item()
        
        if available_features > self.min_features:
            # Calculate number of features to drop
            n_to_drop = int(available_features * self.dropout_probability)
            n_to_drop = min(n_to_drop, available_features - self.min_features)
            
            if n_to_drop > 0:
                # Get indices of available features
                available_indices = torch.where(attention_mask.flatten())[0]
                
                # Randomly select features to drop
                drop_indices = torch.randperm(len(available_indices))[:n_to_drop]
                features_to_drop = available_indices[drop_indices]
                
                # Convert back to 2D indices if needed
                if len(features.shape) == 2:  # [n_features, d_model]
                    features[features_to_drop] = 0
                    attention_mask[features_to_drop] = False
                elif len(features.shape) == 3:  # [seq_len, n_features, d_model]
                    for idx in features_to_drop:
                        seq_idx = idx // features.shape[1]
                        feat_idx = idx % features.shape[1]
                        features[seq_idx, feat_idx] = 0
                        attention_mask[seq_idx, feat_idx] = False
        
        sample['input_features'] = features
        sample['attention_mask'] = attention_mask
        return sample


class FeatureShuffleTransform(TabularTransform):
    """Randomly shuffle the order of features."""
    
    def __init__(self, probability: float = 0.3, shuffle_ratio: float = 0.5):
        self.probability = probability
        self.shuffle_ratio = shuffle_ratio
    
    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Shuffle feature order."""
        if 'input_features' not in sample or random.random() > self.probability:
            return sample
        
        features = sample['input_features'].clone()
        attention_mask = sample.get('attention_mask', torch.ones(features.shape[:-1], dtype=torch.bool))
        
        if len(features.shape) == 2:  # [n_features, d_model]
            n_features = features.shape[0]
            n_to_shuffle = max(2, int(n_features * self.shuffle_ratio))
            
            # Select features to shuffle
            shuffle_indices = torch.randperm(n_features)[:n_to_shuffle]
            shuffled_order = shuffle_indices[torch.randperm(len(shuffle_indices))]
            
            # Apply shuffle
            features[shuffle_indices] = features[shuffled_order]
            attention_mask[shuffle_indices] = attention_mask[shuffled_order]
        
        elif len(features.shape) == 3:  # [seq_len, n_features, d_model]
            # Shuffle features within each time step
            seq_len, n_features, d_model = features.shape
            n_to_shuffle = max(2, int(n_features * self.shuffle_ratio))
            
            for t in range(seq_len):
                shuffle_indices = torch.randperm(n_features)[:n_to_shuffle]
                shuffled_order = shuffle_indices[torch.randperm(len(shuffle_indices))]
                
                features[t, shuffle_indices] = features[t, shuffled_order]
                attention_mask[t, shuffle_indices] = attention_mask[t, shuffled_order]
        
        sample['input_features'] = features
        sample['attention_mask'] = attention_mask
        return sample


class CutMixTransform(TabularTransform):
    """Mix features from different samples (requires batch processing)."""
    
    def __init__(self, probability: float = 0.3, mix_ratio: float = 0.3):
        self.probability = probability
        self.mix_ratio = mix_ratio
        self._batch_cache = []
        self._cache_size = 10
    
    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Apply CutMix (requires maintaining a cache of samples)."""
        if 'input_features' not in sample or random.random() > self.probability:
            return sample
        
        # Add current sample to cache
        self._batch_cache.append(sample.copy())
        
        # Keep cache size manageable
        if len(self._batch_cache) > self._cache_size:
            self._batch_cache.pop(0)
        
        # Only apply CutMix if we have other samples
        if len(self._batch_cache) < 2:
            return sample
        
        # Select another sample from cache
        other_sample = random.choice(self._batch_cache[:-1])  # Exclude current sample
        
        features = sample['input_features'].clone()
        other_features = other_sample['input_features']
        
        # Ensure compatible shapes
        if features.shape != other_features.shape:
            return sample
        
        attention_mask = sample.get('attention_mask', torch.ones(features.shape[:-1], dtype=torch.bool))
        other_mask = other_sample.get('attention_mask', torch.ones(other_features.shape[:-1], dtype=torch.bool))
        
        if len(features.shape) == 2:  # [n_features, d_model]
            n_features = features.shape[0]
            n_to_mix = max(1, int(n_features * self.mix_ratio))
            
            # Select features to mix
            available_features = torch.where(attention_mask & other_mask)[0]
            if len(available_features) >= n_to_mix:
                mix_indices = available_features[torch.randperm(len(available_features))[:n_to_mix]]
                features[mix_indices] = other_features[mix_indices]
        
        elif len(features.shape) == 3:  # [seq_len, n_features, d_model]
            seq_len, n_features, d_model = features.shape
            n_to_mix = max(1, int(n_features * self.mix_ratio))
            
            for t in range(seq_len):
                available_features = torch.where(attention_mask[t] & other_mask[t])[0]
                if len(available_features) >= n_to_mix:
                    mix_indices = available_features[torch.randperm(len(available_features))[:n_to_mix]]
                    features[t, mix_indices] = other_features[t, mix_indices]
        
        sample['input_features'] = features
        return sample


class TemporalShiftTransform(TabularTransform):
    """Shift temporal sequences for data augmentation."""
    
    def __init__(self, max_shift: int = 2, probability: float = 0.5):
        self.max_shift = max_shift
        self.probability = probability
    
    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Apply temporal shift to sequence data."""
        if ('input_features' not in sample or 
            len(sample['input_features'].shape) != 3 or  # Must be sequence data
            random.random() > self.probability):
            return sample
        
        features = sample['input_features'].clone()
        attention_mask = sample.get('attention_mask', torch.ones(features.shape[:-1], dtype=torch.bool))
        
        seq_len = features.shape[0]
        
        if seq_len > self.max_shift * 2:
            # Random shift amount
            shift = random.randint(-self.max_shift, self.max_shift)
            
            if shift > 0:
                # Shift right (remove from beginning, pad at end)
                features = torch.cat([
                    features[shift:],
                    torch.zeros(shift, features.shape[1], features.shape[2])
                ], dim=0)
                attention_mask = torch.cat([
                    attention_mask[shift:],
                    torch.zeros(shift, attention_mask.shape[1], dtype=torch.bool)
                ], dim=0)
            
            elif shift < 0:
                # Shift left (remove from end, pad at beginning)
                shift = abs(shift)
                features = torch.cat([
                    torch.zeros(shift, features.shape[1], features.shape[2]),
                    features[:-shift]
                ], dim=0)
                attention_mask = torch.cat([
                    torch.zeros(shift, attention_mask.shape[1], dtype=torch.bool),
                    attention_mask[:-shift]
                ], dim=0)
        
        sample['input_features'] = features
        sample['attention_mask'] = attention_mask
        return sample


class ScalingTransform(TabularTransform):
    """Randomly scale features."""
    
    def __init__(
        self,
        scale_range: Tuple[float, float] = (0.8, 1.2),
        probability: float = 0.5,
        feature_probability: float = 0.3
    ):
        self.scale_range = scale_range
        self.probability = probability
        self.feature_probability = feature_probability
    
    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Apply random scaling to features."""
        if 'input_features' not in sample or random.random() > self.probability:
            return sample
        
        features = sample['input_features'].clone()
        
        # Create scaling mask
        scale_mask = torch.rand(features.shape[:-1]) < self.feature_probability
        
        # Generate random scales
        scales = torch.uniform(
            torch.full_like(features[..., 0], self.scale_range[0]),
            torch.full_like(features[..., 0], self.scale_range[1])
        )
        
        # Apply scaling
        scales = scales.unsqueeze(-1)  # Add dimension for broadcasting
        features = torch.where(scale_mask.unsqueeze(-1), features * scales, features)
        
        sample['input_features'] = features
        return sample


class ImputationTransform(TabularTransform):
    """Simulate missing values and imputation for robustness."""
    
    def __init__(
        self,
        missing_probability: float = 0.1,
        probability: float = 0.3,
        imputation_method: str = "mean"  # mean, median, zero, forward_fill
    ):
        self.missing_probability = missing_probability
        self.probability = probability
        self.imputation_method = imputation_method
    
    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate missing values and imputation."""
        if 'input_features' not in sample or random.random() > self.probability:
            return sample
        
        features = sample['input_features'].clone()
        attention_mask = sample.get('attention_mask', torch.ones(features.shape[:-1], dtype=torch.bool))
        
        # Create missing value mask
        missing_mask = torch.rand(features.shape[:-1]) < self.missing_probability
        missing_mask = missing_mask & attention_mask  # Only mask valid features
        
        if missing_mask.any():
            # Apply imputation based on method
            if self.imputation_method == "zero":
                features[missing_mask] = 0
            
            elif self.imputation_method == "mean":
                # Compute mean across non-missing values
                valid_features = features[~missing_mask]
                if len(valid_features) > 0:
                    mean_val = valid_features.mean()
                    features[missing_mask] = mean_val
            
            elif self.imputation_method == "median":
                # Compute median across non-missing values
                valid_features = features[~missing_mask]
                if len(valid_features) > 0:
                    median_val = valid_features.median()
                    features[missing_mask] = median_val
            
            elif self.imputation_method == "forward_fill" and len(features.shape) == 3:
                # Forward fill for temporal data
                for t in range(1, features.shape[0]):
                    mask_t = missing_mask[t]
                    if mask_t.any():
                        features[t][mask_t] = features[t-1][mask_t]
        
        sample['input_features'] = features
        return sample


class EncodingTransform(TabularTransform):
    """Apply different encoding schemes to categorical features."""
    
    def __init__(
        self,
        encoding_noise_std: float = 0.05,
        probability: float = 0.3
    ):
        self.encoding_noise_std = encoding_noise_std
        self.probability = probability
    
    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Add noise to categorical encodings."""
        if 'input_features' not in sample or random.random() > self.probability:
            return sample
        
        features = sample['input_features'].clone()
        
        # Add small amount of noise to simulate encoding variations
        noise = torch.randn_like(features) * self.encoding_noise_std
        features = features + noise
        
        sample['input_features'] = features
        return sample


class CompositeTransform(TabularTransform):
    """Compose multiple transforms together."""
    
    def __init__(
        self,
        transforms: List[TabularTransform],
        probabilities: Optional[List[float]] = None
    ):
        self.transforms = transforms
        self.probabilities = probabilities or [1.0] * len(transforms)
        
        assert len(self.transforms) == len(self.probabilities)
    
    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Apply transforms sequentially with individual probabilities."""
        for transform, prob in zip(self.transforms, self.probabilities):
            if random.random() < prob:
                sample = transform(sample)
        
        return sample


class AugmentationTransform(TabularTransform):
    """Comprehensive augmentation pipeline for tabular data."""
    
    def __init__(
        self,
        noise_std: float = 0.1,
        dropout_prob: float = 0.1,
        shuffle_prob: float = 0.3,
        cutmix_prob: float = 0.2,
        scale_range: Tuple[float, float] = (0.9, 1.1),
        missing_prob: float = 0.05,
        overall_probability: float = 0.8
    ):
        self.overall_probability = overall_probability
        
        # Create individual transforms
        transforms = [
            NoiseInjectionTransform(noise_std, probability=0.5),
            FeatureDropoutTransform(dropout_prob, probability=0.4),
            FeatureShuffleTransform(probability=shuffle_prob),
            CutMixTransform(probability=cutmix_prob),
            ScalingTransform(scale_range, probability=0.3),
            ImputationTransform(missing_prob, probability=0.3),
            EncodingTransform(probability=0.2)
        ]
        
        self.composite_transform = CompositeTransform(transforms)
    
    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Apply comprehensive augmentation."""
        if random.random() < self.overall_probability:
            return self.composite_transform(sample)
        return sample


def create_train_transforms(
    augmentation_strength: str = "medium"
) -> CompositeTransform:
    """Create training transforms with different augmentation strengths."""
    
    if augmentation_strength == "light":
        transforms = [
            NoiseInjectionTransform(noise_std=0.05, probability=0.3),
            FeatureDropoutTransform(dropout_probability=0.05, probability=0.2),
            NormalizationTransform(method="standard")
        ]
    
    elif augmentation_strength == "medium":
        transforms = [
            NoiseInjectionTransform(noise_std=0.1, probability=0.5),
            FeatureDropoutTransform(dropout_probability=0.1, probability=0.4),
            FeatureShuffleTransform(probability=0.3),
            ScalingTransform(scale_range=(0.9, 1.1), probability=0.3),
            NormalizationTransform(method="standard")
        ]
    
    elif augmentation_strength == "strong":
        transforms = [
            NoiseInjectionTransform(noise_std=0.15, probability=0.6),
            FeatureDropoutTransform(dropout_probability=0.15, probability=0.5),
            FeatureShuffleTransform(probability=0.4),
            CutMixTransform(probability=0.3),
            ScalingTransform(scale_range=(0.8, 1.2), probability=0.4),
            ImputationTransform(missing_probability=0.1, probability=0.3),
            NormalizationTransform(method="standard")
        ]
    
    else:
        raise ValueError(f"Unknown augmentation strength: {augmentation_strength}")
    
    return CompositeTransform(transforms)


def create_val_transforms() -> CompositeTransform:
    """Create validation/test transforms (minimal augmentation)."""
    transforms = [
        NormalizationTransform(method="standard")
    ]
    
    return CompositeTransform(transforms)