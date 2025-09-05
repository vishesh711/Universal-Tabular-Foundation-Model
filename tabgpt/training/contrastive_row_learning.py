"""Contrastive Row Learning (CRL) pre-training objective for TabGPT."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
from dataclasses import dataclass
from enum import Enum

from ..tokenizers.tabular_tokenizer import ColumnMetadata


class AugmentationType(Enum):
    """Types of data augmentation for contrastive learning."""
    NOISE_INJECTION = "noise_injection"
    FEATURE_DROPOUT = "feature_dropout"
    VALUE_PERTURBATION = "value_perturbation"
    FEATURE_SHUFFLE = "feature_shuffle"
    CUTMIX = "cutmix"


@dataclass
class ContrastiveOutput:
    """Output from contrastive row learning."""
    loss: torch.Tensor
    logits: torch.Tensor
    labels: torch.Tensor
    positive_pairs: torch.Tensor
    negative_pairs: torch.Tensor
    temperature: float
    accuracy: Dict[str, float]


class RowAugmentationStrategy(nn.Module):
    """Base class for row augmentation strategies."""
    
    def __init__(self, augmentation_probability: float = 0.5):
        super().__init__()
        self.augmentation_probability = augmentation_probability
    
    def forward(
        self,
        input_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        column_metadata: Optional[List[ColumnMetadata]] = None
    ) -> torch.Tensor:
        """Apply augmentation to input features."""
        raise NotImplementedError


class NoiseInjectionAugmentation(RowAugmentationStrategy):
    """Add Gaussian noise to numerical features."""
    
    def __init__(
        self,
        augmentation_probability: float = 0.5,
        noise_std: float = 0.1,
        numerical_only: bool = True
    ):
        super().__init__(augmentation_probability)
        self.noise_std = noise_std
        self.numerical_only = numerical_only
    
    def forward(
        self,
        input_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        column_metadata: Optional[List[ColumnMetadata]] = None
    ) -> torch.Tensor:
        """
        Add Gaussian noise to features.
        
        Args:
            input_features: Input features [batch_size, n_features, d_model]
            attention_mask: Attention mask [batch_size, n_features]
            column_metadata: Column metadata for type information
            
        Returns:
            Augmented features [batch_size, n_features, d_model]
        """
        augmented_features = input_features.clone()
        batch_size, n_features, d_model = input_features.shape
        
        # Create augmentation mask
        aug_mask = torch.rand(batch_size, n_features, device=input_features.device) < self.augmentation_probability
        
        if attention_mask is not None:
            aug_mask = aug_mask & attention_mask.bool()
        
        # Apply noise only to selected features
        if aug_mask.any():
            noise = torch.randn_like(input_features) * self.noise_std
            
            # If numerical_only, apply noise only to numerical columns
            if self.numerical_only and column_metadata is not None:
                numerical_mask = torch.zeros(n_features, dtype=torch.bool, device=input_features.device)
                for i, metadata in enumerate(column_metadata):
                    if i < n_features and metadata.dtype in ['numerical', 'datetime']:
                        numerical_mask[i] = True
                
                # Expand numerical mask to batch dimension
                numerical_mask = numerical_mask.unsqueeze(0).expand(batch_size, -1)
                aug_mask = aug_mask & numerical_mask
            
            # Apply noise where augmentation mask is True
            augmented_features[aug_mask] += noise[aug_mask]
        
        return augmented_features


class FeatureDropoutAugmentation(RowAugmentationStrategy):
    """Randomly dropout (zero out) entire features."""
    
    def __init__(
        self,
        augmentation_probability: float = 0.5,
        dropout_probability: float = 0.15,
        min_features: int = 1
    ):
        super().__init__(augmentation_probability)
        self.dropout_probability = dropout_probability
        self.min_features = min_features
    
    def forward(
        self,
        input_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        column_metadata: Optional[List[ColumnMetadata]] = None
    ) -> torch.Tensor:
        """
        Randomly dropout features.
        
        Args:
            input_features: Input features [batch_size, n_features, d_model]
            attention_mask: Attention mask [batch_size, n_features]
            column_metadata: Column metadata (unused in this augmentation)
            
        Returns:
            Augmented features [batch_size, n_features, d_model]
        """
        augmented_features = input_features.clone()
        batch_size, n_features, d_model = input_features.shape
        
        # Apply augmentation to each sample in the batch
        for i in range(batch_size):
            if torch.rand(1).item() < self.augmentation_probability:
                # Determine available features (considering attention mask)
                available_features = torch.arange(n_features, device=input_features.device)
                if attention_mask is not None:
                    available_features = available_features[attention_mask[i].bool()]
                
                # Ensure we keep at least min_features
                n_available = len(available_features)
                if n_available > self.min_features:
                    n_to_drop = int(n_available * self.dropout_probability)
                    n_to_drop = min(n_to_drop, n_available - self.min_features)
                    
                    if n_to_drop > 0:
                        # Randomly select features to drop
                        drop_indices = torch.randperm(n_available, device=input_features.device)[:n_to_drop]
                        features_to_drop = available_features[drop_indices]
                        
                        # Zero out selected features
                        augmented_features[i, features_to_drop] = 0.0
        
        return augmented_features


class ValuePerturbationAugmentation(RowAugmentationStrategy):
    """Perturb feature values based on their data type."""
    
    def __init__(
        self,
        augmentation_probability: float = 0.5,
        numerical_perturbation_std: float = 0.05,
        categorical_swap_probability: float = 0.1
    ):
        super().__init__(augmentation_probability)
        self.numerical_perturbation_std = numerical_perturbation_std
        self.categorical_swap_probability = categorical_swap_probability
    
    def forward(
        self,
        input_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        column_metadata: Optional[List[ColumnMetadata]] = None
    ) -> torch.Tensor:
        """
        Perturb feature values based on data type.
        
        Args:
            input_features: Input features [batch_size, n_features, d_model]
            attention_mask: Attention mask [batch_size, n_features]
            column_metadata: Column metadata for type-specific perturbation
            
        Returns:
            Augmented features [batch_size, n_features, d_model]
        """
        augmented_features = input_features.clone()
        batch_size, n_features, d_model = input_features.shape
        
        # Create augmentation mask
        aug_mask = torch.rand(batch_size, n_features, device=input_features.device) < self.augmentation_probability
        
        if attention_mask is not None:
            aug_mask = aug_mask & attention_mask.bool()
        
        if aug_mask.any() and column_metadata is not None:
            for i, metadata in enumerate(column_metadata):
                if i >= n_features:
                    break
                
                # Get features to augment for this column
                col_aug_mask = aug_mask[:, i]
                if not col_aug_mask.any():
                    continue
                
                if metadata.dtype in ['numerical', 'datetime']:
                    # Add small perturbation to numerical features
                    perturbation = torch.randn_like(augmented_features[col_aug_mask, i]) * self.numerical_perturbation_std
                    augmented_features[col_aug_mask, i] += perturbation
                
                elif metadata.dtype == 'categorical':
                    # For categorical features, add small random noise (simulating category swapping)
                    if torch.rand(1).item() < self.categorical_swap_probability:
                        noise = torch.randn_like(augmented_features[col_aug_mask, i]) * 0.1
                        augmented_features[col_aug_mask, i] += noise
        
        return augmented_features


class FeatureShuffleAugmentation(RowAugmentationStrategy):
    """Shuffle the order of features within each row."""
    
    def __init__(
        self,
        augmentation_probability: float = 0.5,
        shuffle_ratio: float = 0.3
    ):
        super().__init__(augmentation_probability)
        self.shuffle_ratio = shuffle_ratio
    
    def forward(
        self,
        input_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        column_metadata: Optional[List[ColumnMetadata]] = None
    ) -> torch.Tensor:
        """
        Shuffle feature order within rows.
        
        Args:
            input_features: Input features [batch_size, n_features, d_model]
            attention_mask: Attention mask [batch_size, n_features]
            column_metadata: Column metadata (unused in this augmentation)
            
        Returns:
            Augmented features [batch_size, n_features, d_model]
        """
        augmented_features = input_features.clone()
        batch_size, n_features, d_model = input_features.shape
        
        for i in range(batch_size):
            if torch.rand(1).item() < self.augmentation_probability:
                # Determine available features
                available_indices = torch.arange(n_features, device=input_features.device)
                if attention_mask is not None:
                    available_indices = available_indices[attention_mask[i].bool()]
                
                n_available = len(available_indices)
                if n_available > 1:
                    # Select features to shuffle
                    n_to_shuffle = max(2, int(n_available * self.shuffle_ratio))
                    shuffle_indices = available_indices[torch.randperm(n_available, device=input_features.device)[:n_to_shuffle]]
                    
                    # Shuffle selected features
                    shuffled_order = shuffle_indices[torch.randperm(len(shuffle_indices), device=input_features.device)]
                    augmented_features[i, shuffle_indices] = input_features[i, shuffled_order]
        
        return augmented_features


class CutMixAugmentation(RowAugmentationStrategy):
    """Mix features from different rows (CutMix for tabular data)."""
    
    def __init__(
        self,
        augmentation_probability: float = 0.5,
        mix_ratio: float = 0.3,
        same_batch_only: bool = True
    ):
        super().__init__(augmentation_probability)
        self.mix_ratio = mix_ratio
        self.same_batch_only = same_batch_only
    
    def forward(
        self,
        input_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        column_metadata: Optional[List[ColumnMetadata]] = None
    ) -> torch.Tensor:
        """
        Mix features from different rows.
        
        Args:
            input_features: Input features [batch_size, n_features, d_model]
            attention_mask: Attention mask [batch_size, n_features]
            column_metadata: Column metadata (unused in this augmentation)
            
        Returns:
            Augmented features [batch_size, n_features, d_model]
        """
        augmented_features = input_features.clone()
        batch_size, n_features, d_model = input_features.shape
        
        if batch_size < 2:
            return augmented_features
        
        for i in range(batch_size):
            if torch.rand(1).item() < self.augmentation_probability:
                # Select another row to mix with
                other_idx = torch.randint(0, batch_size, (1,), device=input_features.device).item()
                while other_idx == i and batch_size > 1:
                    other_idx = torch.randint(0, batch_size, (1,), device=input_features.device).item()
                
                # Determine available features
                available_indices = torch.arange(n_features, device=input_features.device)
                if attention_mask is not None:
                    mask_i = attention_mask[i].bool()
                    mask_other = attention_mask[other_idx].bool()
                    available_indices = available_indices[mask_i & mask_other]
                
                n_available = len(available_indices)
                if n_available > 0:
                    # Select features to mix
                    n_to_mix = max(1, int(n_available * self.mix_ratio))
                    mix_indices = available_indices[torch.randperm(n_available, device=input_features.device)[:n_to_mix]]
                    
                    # Mix features
                    augmented_features[i, mix_indices] = input_features[other_idx, mix_indices]
        
        return augmented_features


class MultiAugmentationStrategy(RowAugmentationStrategy):
    """Combine multiple augmentation strategies."""
    
    def __init__(
        self,
        augmentation_strategies: List[RowAugmentationStrategy],
        strategy_probabilities: Optional[List[float]] = None,
        max_augmentations: int = 2
    ):
        super().__init__(1.0)  # Always apply at least one augmentation
        self.augmentation_strategies = nn.ModuleList(augmentation_strategies)
        self.strategy_probabilities = strategy_probabilities or [1.0] * len(augmentation_strategies)
        self.max_augmentations = max_augmentations
        
        assert len(self.strategy_probabilities) == len(augmentation_strategies)
    
    def forward(
        self,
        input_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        column_metadata: Optional[List[ColumnMetadata]] = None
    ) -> torch.Tensor:
        """
        Apply multiple augmentation strategies.
        
        Args:
            input_features: Input features [batch_size, n_features, d_model]
            attention_mask: Attention mask [batch_size, n_features]
            column_metadata: Column metadata
            
        Returns:
            Augmented features [batch_size, n_features, d_model]
        """
        augmented_features = input_features.clone()
        
        # Randomly select augmentation strategies to apply
        n_strategies = len(self.augmentation_strategies)
        n_to_apply = torch.randint(1, min(self.max_augmentations + 1, n_strategies + 1), (1,)).item()
        
        # Sample strategies based on probabilities
        strategy_probs = torch.tensor(self.strategy_probabilities, device=input_features.device)
        selected_indices = torch.multinomial(strategy_probs, n_to_apply, replacement=False)
        
        # Apply selected augmentations sequentially
        for idx in selected_indices:
            strategy = self.augmentation_strategies[idx]
            augmented_features = strategy(
                augmented_features,
                attention_mask=attention_mask,
                column_metadata=column_metadata
            )
        
        return augmented_features


class InfoNCELoss(nn.Module):
    """InfoNCE loss for contrastive learning."""
    
    def __init__(
        self,
        temperature: float = 0.07,
        normalize_embeddings: bool = True
    ):
        super().__init__()
        self.temperature = temperature
        self.normalize_embeddings = normalize_embeddings
    
    def forward(
        self,
        anchor_embeddings: torch.Tensor,
        positive_embeddings: torch.Tensor,
        negative_embeddings: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute InfoNCE loss.
        
        Args:
            anchor_embeddings: Anchor embeddings [batch_size, d_model]
            positive_embeddings: Positive embeddings [batch_size, d_model]
            negative_embeddings: Negative embeddings [batch_size * (n_neg), d_model] or None
            
        Returns:
            Tuple of (loss, logits, labels)
        """
        batch_size = anchor_embeddings.shape[0]
        
        # Normalize embeddings if requested
        if self.normalize_embeddings:
            anchor_embeddings = F.normalize(anchor_embeddings, dim=-1)
            positive_embeddings = F.normalize(positive_embeddings, dim=-1)
            if negative_embeddings is not None:
                negative_embeddings = F.normalize(negative_embeddings, dim=-1)
        
        # Compute positive similarities
        pos_sim = torch.sum(anchor_embeddings * positive_embeddings, dim=-1) / self.temperature
        
        if negative_embeddings is not None:
            # Use provided negative embeddings
            # Compute similarities between anchors and all negatives
            neg_sim = torch.matmul(anchor_embeddings, negative_embeddings.T) / self.temperature
            
            # Combine positive and negative similarities
            logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
        else:
            # Use in-batch negatives (all other samples in the batch)
            # Compute similarities between all pairs
            all_embeddings = torch.cat([positive_embeddings, anchor_embeddings], dim=0)
            sim_matrix = torch.matmul(anchor_embeddings, all_embeddings.T) / self.temperature
            
            # Create mask to exclude self-similarities and extract positive similarities
            mask = torch.eye(batch_size, device=anchor_embeddings.device, dtype=torch.bool)
            
            # Positive similarities are with the corresponding positive samples
            pos_sim = sim_matrix[:, :batch_size][mask]
            
            # Negative similarities are with all other samples
            neg_mask = ~torch.eye(2 * batch_size, device=anchor_embeddings.device, dtype=torch.bool)[:batch_size]
            neg_sim = sim_matrix[neg_mask].view(batch_size, -1)
            
            # Combine positive and negative similarities
            logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
        
        # Labels: positive samples are always at index 0
        labels = torch.zeros(batch_size, dtype=torch.long, device=anchor_embeddings.device)
        
        # Compute cross-entropy loss
        loss = F.cross_entropy(logits, labels)
        
        return loss, logits, labels


class ContrastiveRowLearningObjective(nn.Module):
    """
    Complete Contrastive Row Learning objective with augmentation and InfoNCE loss.
    """
    
    def __init__(
        self,
        d_model: int,
        augmentation_strategies: Optional[List[RowAugmentationStrategy]] = None,
        temperature: float = 0.07,
        normalize_embeddings: bool = True,
        projection_dim: Optional[int] = None,
        use_projection_head: bool = True
    ):
        super().__init__()
        self.d_model = d_model
        self.temperature = temperature
        self.normalize_embeddings = normalize_embeddings
        self.use_projection_head = use_projection_head
        
        # Default augmentation strategies if none provided
        if augmentation_strategies is None:
            augmentation_strategies = [
                NoiseInjectionAugmentation(augmentation_probability=0.5, noise_std=0.1),
                FeatureDropoutAugmentation(augmentation_probability=0.5, dropout_probability=0.15),
                ValuePerturbationAugmentation(augmentation_probability=0.5),
                FeatureShuffleAugmentation(augmentation_probability=0.3, shuffle_ratio=0.2)
            ]
        
        self.augmentation_strategy = MultiAugmentationStrategy(
            augmentation_strategies=augmentation_strategies,
            max_augmentations=2
        )
        
        # Projection head for contrastive learning
        if use_projection_head:
            projection_dim = projection_dim or d_model // 2
            self.projection_head = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Linear(d_model, projection_dim)
            )
        else:
            self.projection_head = nn.Identity()
            projection_dim = d_model
        
        # InfoNCE loss
        self.infonce_loss = InfoNCELoss(
            temperature=temperature,
            normalize_embeddings=normalize_embeddings
        )
    
    def create_augmented_pairs(
        self,
        input_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        column_metadata: Optional[List[ColumnMetadata]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create augmented positive pairs from input features.
        
        Args:
            input_features: Input features [batch_size, n_features, d_model]
            attention_mask: Attention mask [batch_size, n_features]
            column_metadata: Column metadata
            
        Returns:
            Tuple of (original_features, augmented_features)
        """
        # Create two different augmented versions
        augmented_v1 = self.augmentation_strategy(
            input_features,
            attention_mask=attention_mask,
            column_metadata=column_metadata
        )
        
        augmented_v2 = self.augmentation_strategy(
            input_features,
            attention_mask=attention_mask,
            column_metadata=column_metadata
        )
        
        return augmented_v1, augmented_v2
    
    def forward(
        self,
        row_embeddings: torch.Tensor,
        input_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        column_metadata: Optional[List[ColumnMetadata]] = None,
        model_forward_fn: Optional[callable] = None
    ) -> ContrastiveOutput:
        """
        Forward pass for contrastive row learning.
        
        Args:
            row_embeddings: Row embeddings from model [batch_size, d_model]
            input_features: Original input features [batch_size, n_features, d_model]
            attention_mask: Attention mask [batch_size, n_features]
            column_metadata: Column metadata
            model_forward_fn: Function to compute embeddings from features
            
        Returns:
            ContrastiveOutput with loss and metrics
        """
        batch_size = row_embeddings.shape[0]
        
        # Create augmented pairs
        aug_features_v1, aug_features_v2 = self.create_augmented_pairs(
            input_features,
            attention_mask=attention_mask,
            column_metadata=column_metadata
        )
        
        # Get embeddings for augmented versions
        if model_forward_fn is not None:
            # Use provided function to compute embeddings
            aug_embeddings_v1 = model_forward_fn(aug_features_v1, attention_mask)
            aug_embeddings_v2 = model_forward_fn(aug_features_v2, attention_mask)
        else:
            # Use original embeddings as proxy (for testing)
            aug_embeddings_v1 = row_embeddings
            aug_embeddings_v2 = row_embeddings
        
        # Apply projection head
        anchor_proj = self.projection_head(aug_embeddings_v1)
        positive_proj = self.projection_head(aug_embeddings_v2)
        
        # Compute InfoNCE loss
        loss, logits, labels = self.infonce_loss(anchor_proj, positive_proj)
        
        # Compute accuracy
        predictions = logits.argmax(dim=-1)
        accuracy = (predictions == labels).float().mean().item()
        
        # Compute additional metrics
        positive_sim = logits[:, 0].mean().item()  # Average positive similarity
        negative_sim = logits[:, 1:].mean().item() if logits.shape[1] > 1 else 0.0  # Average negative similarity
        
        return ContrastiveOutput(
            loss=loss,
            logits=logits,
            labels=labels,
            positive_pairs=torch.stack([aug_embeddings_v1, aug_embeddings_v2], dim=1),
            negative_pairs=torch.zeros_like(aug_embeddings_v1),  # Placeholder
            temperature=self.temperature,
            accuracy={
                'contrastive_accuracy': accuracy,
                'positive_similarity': positive_sim,
                'negative_similarity': negative_sim,
                'temperature': self.temperature,
                'batch_size': batch_size
            }
        )
    
    def compute_metrics(self, outputs: ContrastiveOutput) -> Dict[str, float]:
        """
        Compute evaluation metrics for contrastive row learning.
        
        Args:
            outputs: ContrastiveOutput from forward pass
            
        Returns:
            Dictionary of metrics
        """
        metrics = {
            'crl_loss': outputs.loss.item(),
            'contrastive_accuracy': outputs.accuracy['contrastive_accuracy'],
            'positive_similarity': outputs.accuracy['positive_similarity'],
            'negative_similarity': outputs.accuracy['negative_similarity'],
            'temperature': outputs.temperature,
            'batch_size': outputs.accuracy['batch_size']
        }
        
        # Compute similarity gap (positive - negative)
        sim_gap = outputs.accuracy['positive_similarity'] - outputs.accuracy['negative_similarity']
        metrics['similarity_gap'] = sim_gap
        
        # Compute logit statistics
        if outputs.logits.numel() > 0:
            metrics['logit_mean'] = outputs.logits.mean().item()
            metrics['logit_std'] = outputs.logits.std().item()
            metrics['logit_max'] = outputs.logits.max().item()
            metrics['logit_min'] = outputs.logits.min().item()
        
        return metrics


def create_default_augmentation_strategies(
    noise_std: float = 0.1,
    dropout_prob: float = 0.15,
    perturbation_std: float = 0.05,
    shuffle_ratio: float = 0.2,
    mix_ratio: float = 0.3
) -> List[RowAugmentationStrategy]:
    """
    Create default set of augmentation strategies.
    
    Args:
        noise_std: Standard deviation for noise injection
        dropout_prob: Probability for feature dropout
        perturbation_std: Standard deviation for value perturbation
        shuffle_ratio: Ratio of features to shuffle
        mix_ratio: Ratio of features to mix in CutMix
        
    Returns:
        List of augmentation strategies
    """
    return [
        NoiseInjectionAugmentation(
            augmentation_probability=0.6,
            noise_std=noise_std,
            numerical_only=True
        ),
        FeatureDropoutAugmentation(
            augmentation_probability=0.5,
            dropout_probability=dropout_prob,
            min_features=1
        ),
        ValuePerturbationAugmentation(
            augmentation_probability=0.4,
            numerical_perturbation_std=perturbation_std,
            categorical_swap_probability=0.1
        ),
        FeatureShuffleAugmentation(
            augmentation_probability=0.3,
            shuffle_ratio=shuffle_ratio
        ),
        CutMixAugmentation(
            augmentation_probability=0.2,
            mix_ratio=mix_ratio,
            same_batch_only=True
        )
    ]