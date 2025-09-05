"""Anomaly detection heads for unsupervised outlier detection."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Union, Any, Tuple
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest

from .base import BaseTaskHead, TaskOutput, TaskType


class AnomalyDetectionHead(BaseTaskHead):
    """Base class for anomaly detection heads."""
    
    def __init__(
        self,
        input_dim: int,
        dropout: float = 0.1,
        activation: str = "relu",
        threshold_percentile: float = 95.0
    ):
        super().__init__(input_dim, TaskType.ANOMALY_DETECTION, dropout, activation)
        self.threshold_percentile = threshold_percentile
        self.anomaly_threshold = None
    
    def set_threshold(self, scores: torch.Tensor):
        """Set anomaly threshold based on training scores."""
        self.anomaly_threshold = torch.quantile(scores, self.threshold_percentile / 100.0).item()
    
    def get_anomaly_predictions(self, scores: torch.Tensor) -> torch.Tensor:
        """Convert anomaly scores to binary predictions."""
        if self.anomaly_threshold is None:
            # Use median as threshold if not set
            threshold = torch.median(scores).item()
        else:
            threshold = self.anomaly_threshold
        
        return (scores > threshold).float()
    
    def compute_metrics(
        self,
        predictions: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, float]:
        """Compute anomaly detection metrics."""
        metrics = {
            'mean_anomaly_score': predictions.mean().item(),
            'std_anomaly_score': predictions.std().item(),
            'max_anomaly_score': predictions.max().item(),
            'min_anomaly_score': predictions.min().item()
        }
        
        if targets is not None:
            # Supervised evaluation
            binary_preds = self.get_anomaly_predictions(predictions)
            
            # Basic metrics
            accuracy = (binary_preds == targets.float()).float().mean().item()
            
            # Confusion matrix
            tp = ((binary_preds == 1) & (targets == 1)).sum().item()
            fp = ((binary_preds == 1) & (targets == 0)).sum().item()
            tn = ((binary_preds == 0) & (targets == 0)).sum().item()
            fn = ((binary_preds == 0) & (targets == 1)).sum().item()
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            metrics.update({
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'true_positives': tp,
                'false_positives': fp,
                'true_negatives': tn,
                'false_negatives': fn
            })
        
        return metrics


class ReconstructionAnomalyHead(AnomalyDetectionHead):
    """Anomaly detection based on reconstruction error."""
    
    def __init__(
        self,
        input_dim: int,
        latent_dim: Optional[int] = None,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.1,
        activation: str = "relu",
        reconstruction_loss: str = "mse",
        threshold_percentile: float = 95.0
    ):
        super().__init__(input_dim, dropout, activation, threshold_percentile)
        
        self.latent_dim = latent_dim or input_dim // 4
        self.reconstruction_loss = reconstruction_loss
        
        # Default hidden dimensions for autoencoder
        if hidden_dims is None:
            hidden_dims = [input_dim // 2, self.latent_dim]
        
        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                self.activation_fn,
                self.dropout_layer
            ])
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder
        decoder_layers = []
        decoder_dims = hidden_dims[::-1][1:] + [input_dim]  # Reverse hidden dims + input dim
        
        for hidden_dim in decoder_dims:
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                self.activation_fn if hidden_dim != input_dim else nn.Identity(),
                self.dropout_layer if hidden_dim != input_dim else nn.Identity()
            ])
            prev_dim = hidden_dim
        
        self.decoder = nn.Sequential(*decoder_layers)
        
        # Loss function
        if reconstruction_loss == "mse":
            self.recon_loss_fn = nn.MSELoss(reduction='none')
        elif reconstruction_loss == "mae":
            self.recon_loss_fn = nn.L1Loss(reduction='none')
        else:
            raise ValueError(f"Unsupported reconstruction loss: {reconstruction_loss}")
    
    def forward(
        self,
        features: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        **kwargs
    ) -> TaskOutput:
        """Forward pass through autoencoder."""
        # Encode
        latent = self.encoder(features)
        
        # Decode
        reconstructed = self.decoder(latent)
        
        # Compute reconstruction error (anomaly scores)
        recon_error = self.recon_loss_fn(reconstructed, features)
        anomaly_scores = recon_error.mean(dim=-1)  # Average across features
        
        # Compute loss
        loss = anomaly_scores.mean()
        
        return TaskOutput(
            predictions=anomaly_scores,
            loss=loss,
            features=latent,
            metadata={
                'reconstructed': reconstructed,
                'latent_dim': self.latent_dim
            }
        )
    
    def compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """Compute reconstruction loss."""
        # For reconstruction-based anomaly detection, loss is the reconstruction error
        return predictions.mean()


class OneClassSVMHead(AnomalyDetectionHead):
    """Anomaly detection using One-Class SVM (requires sklearn)."""
    
    def __init__(
        self,
        input_dim: int,
        nu: float = 0.05,
        kernel: str = "rbf",
        gamma: str = "scale",
        dropout: float = 0.1,
        feature_projection_dim: Optional[int] = None
    ):
        super().__init__(input_dim, dropout)
        
        self.nu = nu
        self.kernel = kernel
        self.gamma = gamma
        self.feature_projection_dim = feature_projection_dim
        
        # Optional feature projection before SVM
        if feature_projection_dim:
            self.feature_projection = nn.Sequential(
                nn.Linear(input_dim, feature_projection_dim),
                nn.ReLU(),
                self.dropout_layer
            )
        else:
            self.feature_projection = nn.Identity()
            feature_projection_dim = input_dim
        
        # One-Class SVM (will be fitted during training)
        self.svm = OneClassSVM(nu=nu, kernel=kernel, gamma=gamma)
        self.is_fitted = False
    
    def fit_svm(self, features: torch.Tensor):
        """Fit the One-Class SVM on training features."""
        # Project features
        projected_features = self.feature_projection(features)
        
        # Convert to numpy for sklearn
        features_np = projected_features.detach().cpu().numpy()
        
        # Fit SVM
        self.svm.fit(features_np)
        self.is_fitted = True
    
    def forward(
        self,
        features: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        **kwargs
    ) -> TaskOutput:
        """Forward pass through One-Class SVM head."""
        # Project features
        projected_features = self.feature_projection(features)
        
        if not self.is_fitted:
            # If not fitted, fit on current batch (not ideal, but necessary)
            self.fit_svm(features)
        
        # Get anomaly scores
        features_np = projected_features.detach().cpu().numpy()
        
        # SVM decision function (distance to separating hyperplane)
        anomaly_scores = self.svm.decision_function(features_np)
        anomaly_scores = torch.tensor(anomaly_scores, device=features.device, dtype=features.dtype)
        
        # Convert to anomaly scores (negative distance = more anomalous)
        anomaly_scores = -anomaly_scores
        
        # No gradient-based loss for SVM
        loss = torch.tensor(0.0, device=features.device)
        
        return TaskOutput(
            predictions=anomaly_scores,
            loss=loss,
            features=projected_features,
            metadata={'svm_fitted': self.is_fitted}
        )
    
    def compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """One-Class SVM doesn't use gradient-based loss."""
        return torch.tensor(0.0, device=predictions.device)


class IsolationForestHead(AnomalyDetectionHead):
    """Anomaly detection using Isolation Forest (requires sklearn)."""
    
    def __init__(
        self,
        input_dim: int,
        n_estimators: int = 100,
        contamination: float = 0.1,
        random_state: int = 42,
        dropout: float = 0.1,
        feature_projection_dim: Optional[int] = None
    ):
        super().__init__(input_dim, dropout)
        
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.random_state = random_state
        self.feature_projection_dim = feature_projection_dim
        
        # Optional feature projection
        if feature_projection_dim:
            self.feature_projection = nn.Sequential(
                nn.Linear(input_dim, feature_projection_dim),
                nn.ReLU(),
                self.dropout_layer
            )
        else:
            self.feature_projection = nn.Identity()
        
        # Isolation Forest
        self.isolation_forest = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state
        )
        self.is_fitted = False
    
    def fit_isolation_forest(self, features: torch.Tensor):
        """Fit the Isolation Forest on training features."""
        # Project features
        projected_features = self.feature_projection(features)
        
        # Convert to numpy for sklearn
        features_np = projected_features.detach().cpu().numpy()
        
        # Fit Isolation Forest
        self.isolation_forest.fit(features_np)
        self.is_fitted = True
    
    def forward(
        self,
        features: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        **kwargs
    ) -> TaskOutput:
        """Forward pass through Isolation Forest head."""
        # Project features
        projected_features = self.feature_projection(features)
        
        if not self.is_fitted:
            # If not fitted, fit on current batch
            self.fit_isolation_forest(features)
        
        # Get anomaly scores
        features_np = projected_features.detach().cpu().numpy()
        
        # Isolation Forest anomaly scores
        anomaly_scores = self.isolation_forest.decision_function(features_np)
        anomaly_scores = torch.tensor(anomaly_scores, device=features.device, dtype=features.dtype)
        
        # Convert to anomaly scores (negative score = more anomalous)
        anomaly_scores = -anomaly_scores
        
        # No gradient-based loss for Isolation Forest
        loss = torch.tensor(0.0, device=features.device)
        
        return TaskOutput(
            predictions=anomaly_scores,
            loss=loss,
            features=projected_features,
            metadata={'isolation_forest_fitted': self.is_fitted}
        )
    
    def compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """Isolation Forest doesn't use gradient-based loss."""
        return torch.tensor(0.0, device=predictions.device)


class DeepSVDDHead(AnomalyDetectionHead):
    """Deep Support Vector Data Description for anomaly detection."""
    
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 32,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.1,
        activation: str = "relu",
        nu: float = 0.1,
        objective: str = "soft"  # "soft" or "one-class"
    ):
        super().__init__(input_dim, dropout, activation)
        
        self.latent_dim = latent_dim
        self.nu = nu
        self.objective = objective
        
        # Build encoder network
        if hidden_dims is None:
            hidden_dims = [input_dim // 2, input_dim // 4]
        
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                self.activation_fn,
                self.dropout_layer
            ])
            prev_dim = hidden_dim
        
        # Final layer to latent space
        layers.append(nn.Linear(prev_dim, latent_dim))
        
        self.encoder = nn.Sequential(*layers)
        
        # Center of the hypersphere (learnable parameter)
        self.center = nn.Parameter(torch.zeros(latent_dim))
        
        # Radius (for soft boundary)
        if objective == "soft":
            self.radius = nn.Parameter(torch.tensor(1.0))
        else:
            self.register_buffer('radius', torch.tensor(0.0))
    
    def forward(
        self,
        features: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        **kwargs
    ) -> TaskOutput:
        """Forward pass through Deep SVDD head."""
        # Encode to latent space
        latent = self.encoder(features)
        
        # Compute distances to center
        distances = torch.sum((latent - self.center) ** 2, dim=-1)
        
        # Anomaly scores are distances from center
        anomaly_scores = distances
        
        # Compute loss
        loss = self.compute_loss(anomaly_scores, targets, **kwargs)
        
        return TaskOutput(
            predictions=anomaly_scores,
            loss=loss,
            features=latent,
            metadata={
                'center': self.center.detach(),
                'radius': self.radius.detach() if hasattr(self, 'radius') else None,
                'latent_dim': self.latent_dim
            }
        )
    
    def compute_loss(
        self,
        predictions: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """Compute Deep SVDD loss."""
        if self.objective == "one-class":
            # One-class objective: minimize volume of hypersphere
            return predictions.mean()
        
        elif self.objective == "soft":
            # Soft boundary objective with slack variables
            if targets is not None:
                # Semi-supervised: use labels if available
                normal_mask = (targets == 0)
                anomaly_mask = (targets == 1)
                
                loss = 0.0
                
                # Normal samples should be close to center
                if normal_mask.any():
                    normal_distances = predictions[normal_mask]
                    loss += normal_distances.mean()
                
                # Anomalous samples should be far from center
                if anomaly_mask.any():
                    anomaly_distances = predictions[anomaly_mask]
                    # Encourage anomalies to be outside radius
                    loss += torch.clamp(self.radius - anomaly_distances, min=0).mean()
                
                return loss
            else:
                # Unsupervised: minimize distances (assume most data is normal)
                return predictions.mean()
        
        else:
            raise ValueError(f"Unknown objective: {self.objective}")


class VariationalAnomalyHead(AnomalyDetectionHead):
    """Variational autoencoder-based anomaly detection."""
    
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 32,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.1,
        activation: str = "relu",
        beta: float = 1.0,  # Weight for KL divergence
        threshold_percentile: float = 95.0
    ):
        super().__init__(input_dim, dropout, activation, threshold_percentile)
        
        self.latent_dim = latent_dim
        self.beta = beta
        
        if hidden_dims is None:
            hidden_dims = [input_dim // 2]
        
        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                self.activation_fn,
                self.dropout_layer
            ])
            prev_dim = hidden_dim
        
        self.encoder_shared = nn.Sequential(*encoder_layers)
        
        # Mean and log variance heads
        self.mu_head = nn.Linear(prev_dim, latent_dim)
        self.logvar_head = nn.Linear(prev_dim, latent_dim)
        
        # Decoder
        decoder_layers = []
        prev_dim = latent_dim
        decoder_dims = hidden_dims[::-1] + [input_dim]
        
        for hidden_dim in decoder_dims:
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                self.activation_fn if hidden_dim != input_dim else nn.Identity(),
                self.dropout_layer if hidden_dim != input_dim else nn.Identity()
            ])
            prev_dim = hidden_dim
        
        self.decoder = nn.Sequential(*decoder_layers)
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for VAE."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(
        self,
        features: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        **kwargs
    ) -> TaskOutput:
        """Forward pass through VAE anomaly detector."""
        # Encode
        encoded = self.encoder_shared(features)
        mu = self.mu_head(encoded)
        logvar = self.logvar_head(encoded)
        
        # Reparameterize
        z = self.reparameterize(mu, logvar)
        
        # Decode
        reconstructed = self.decoder(z)
        
        # Compute reconstruction error
        recon_error = F.mse_loss(reconstructed, features, reduction='none').mean(dim=-1)
        
        # Compute KL divergence
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
        
        # Total anomaly score (reconstruction error + KL divergence)
        anomaly_scores = recon_error + self.beta * kl_div
        
        # VAE loss
        loss = anomaly_scores.mean()
        
        return TaskOutput(
            predictions=anomaly_scores,
            loss=loss,
            features=z,
            metadata={
                'mu': mu,
                'logvar': logvar,
                'reconstructed': reconstructed,
                'recon_error': recon_error,
                'kl_divergence': kl_div,
                'beta': self.beta
            }
        )
    
    def compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """Compute VAE loss (already computed in forward)."""
        return predictions.mean()
    
    def compute_metrics(
        self,
        predictions: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, float]:
        """Compute VAE anomaly detection metrics."""
        metrics = super().compute_metrics(predictions, targets, **kwargs)
        
        # Add VAE-specific metrics if metadata is available
        if 'metadata' in kwargs:
            metadata = kwargs['metadata']
            if 'recon_error' in metadata:
                metrics['mean_recon_error'] = metadata['recon_error'].mean().item()
            if 'kl_divergence' in metadata:
                metrics['mean_kl_divergence'] = metadata['kl_divergence'].mean().item()
        
        return metrics