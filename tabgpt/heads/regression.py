"""Regression heads for continuous value prediction tasks."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Union, Any, Tuple
import numpy as np

from .base import BaseTaskHead, TaskOutput, TaskType, MLPHead


class RegressionHead(MLPHead):
    """General regression head for continuous value prediction."""
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.1,
        activation: str = "relu",
        output_activation: Optional[str] = None,
        loss_type: str = "mse",
        huber_delta: float = 1.0
    ):
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            task_type=TaskType.REGRESSION,
            hidden_dims=hidden_dims,
            dropout=dropout,
            activation=activation,
            final_activation=output_activation
        )
        
        self.loss_type = loss_type
        self.huber_delta = huber_delta
        
        # Loss function
        if loss_type == "mse":
            self.loss_fn = nn.MSELoss()
        elif loss_type == "mae":
            self.loss_fn = nn.L1Loss()
        elif loss_type == "huber":
            self.loss_fn = nn.HuberLoss(delta=huber_delta)
        elif loss_type == "smooth_l1":
            self.loss_fn = nn.SmoothL1Loss()
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")
    
    def compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """Compute regression loss."""
        return self.loss_fn(predictions, targets.float())
    
    def compute_metrics(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        **kwargs
    ) -> Dict[str, float]:
        """Compute regression-specific metrics."""
        predictions = predictions.squeeze()
        targets = targets.float().squeeze()
        
        # Basic metrics
        mse = torch.mean((predictions - targets) ** 2).item()
        mae = torch.mean(torch.abs(predictions - targets)).item()
        rmse = np.sqrt(mse)
        
        # R-squared
        ss_res = torch.sum((targets - predictions) ** 2).item()
        ss_tot = torch.sum((targets - torch.mean(targets)) ** 2).item()
        r2 = 1 - (ss_res / (ss_tot + 1e-8))
        
        # Mean Absolute Percentage Error (MAPE)
        mape = torch.mean(torch.abs((targets - predictions) / (targets + 1e-8))).item() * 100
        
        # Explained variance
        var_pred = torch.var(predictions).item()
        var_target = torch.var(targets).item()
        explained_variance = 1 - (torch.var(targets - predictions).item() / (var_target + 1e-8))
        
        return {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'r2_score': r2,
            'mape': mape,
            'explained_variance': explained_variance
        }


class MultiTargetRegressionHead(BaseTaskHead):
    """Head for multi-target regression tasks."""
    
    def __init__(
        self,
        input_dim: int,
        num_targets: int,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.1,
        activation: str = "relu",
        shared_layers: bool = True,
        target_weights: Optional[torch.Tensor] = None,
        loss_type: str = "mse"
    ):
        super().__init__(input_dim, TaskType.MULTI_TARGET_REGRESSION, dropout, activation)
        
        self.num_targets = num_targets
        self.shared_layers = shared_layers
        self.loss_type = loss_type
        
        # Target weights for weighted loss
        if target_weights is not None:
            self.register_buffer('target_weights', target_weights)
        else:
            self.target_weights = None
        
        if shared_layers:
            # Shared feature extraction
            if hidden_dims:
                layers = []
                prev_dim = input_dim
                for hidden_dim in hidden_dims:
                    layers.extend([
                        nn.Linear(prev_dim, hidden_dim),
                        self.activation_fn,
                        self.dropout_layer
                    ])
                    prev_dim = hidden_dim
                self.shared_network = nn.Sequential(*layers)
                feature_dim = prev_dim
            else:
                self.shared_network = nn.Identity()
                feature_dim = input_dim
            
            # Single output layer for all targets
            self.output_layer = nn.Linear(feature_dim, num_targets)
        
        else:
            # Separate networks for each target
            self.target_networks = nn.ModuleList()
            for _ in range(num_targets):
                if hidden_dims:
                    layers = []
                    prev_dim = input_dim
                    for hidden_dim in hidden_dims:
                        layers.extend([
                            nn.Linear(prev_dim, hidden_dim),
                            self.activation_fn,
                            self.dropout_layer
                        ])
                        prev_dim = hidden_dim
                    layers.append(nn.Linear(prev_dim, 1))
                    network = nn.Sequential(*layers)
                else:
                    network = nn.Linear(input_dim, 1)
                
                self.target_networks.append(network)
        
        # Loss function
        if loss_type == "mse":
            self.loss_fn = nn.MSELoss(reduction='none')
        elif loss_type == "mae":
            self.loss_fn = nn.L1Loss(reduction='none')
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")
    
    def forward(
        self,
        features: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        **kwargs
    ) -> TaskOutput:
        """Forward pass through multi-target regression head."""
        if self.shared_layers:
            # Shared network approach
            shared_features = self.shared_network(features)
            predictions = self.output_layer(shared_features)
        else:
            # Separate networks approach
            target_predictions = []
            for network in self.target_networks:
                pred = network(features)
                target_predictions.append(pred)
            predictions = torch.cat(target_predictions, dim=-1)
        
        # Compute loss if targets provided
        loss = None
        if targets is not None:
            loss = self.compute_loss(predictions, targets, **kwargs)
        
        return TaskOutput(
            predictions=predictions,
            loss=loss,
            features=shared_features if self.shared_layers else features,
            metadata={'num_targets': self.num_targets}
        )
    
    def compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """Compute multi-target regression loss."""
        # Compute loss for each target
        target_losses = self.loss_fn(predictions, targets.float())
        
        # Apply target weights if provided
        if self.target_weights is not None:
            target_losses = target_losses * self.target_weights.unsqueeze(0)
        
        # Average across targets and samples
        return target_losses.mean()
    
    def compute_metrics(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        **kwargs
    ) -> Dict[str, float]:
        """Compute multi-target regression metrics."""
        targets = targets.float()
        
        # Overall metrics
        overall_mse = torch.mean((predictions - targets) ** 2).item()
        overall_mae = torch.mean(torch.abs(predictions - targets)).item()
        
        # Per-target metrics
        target_mse = torch.mean((predictions - targets) ** 2, dim=0)
        target_mae = torch.mean(torch.abs(predictions - targets), dim=0)
        target_r2 = []
        
        for i in range(self.num_targets):
            pred_i = predictions[:, i]
            target_i = targets[:, i]
            
            ss_res = torch.sum((target_i - pred_i) ** 2).item()
            ss_tot = torch.sum((target_i - torch.mean(target_i)) ** 2).item()
            r2 = 1 - (ss_res / (ss_tot + 1e-8))
            target_r2.append(r2)
        
        metrics = {
            'overall_mse': overall_mse,
            'overall_mae': overall_mae,
            'overall_rmse': np.sqrt(overall_mse),
            'mean_target_r2': np.mean(target_r2)
        }
        
        # Add per-target metrics
        for i in range(self.num_targets):
            metrics[f'target_{i}_mse'] = target_mse[i].item()
            metrics[f'target_{i}_mae'] = target_mae[i].item()
            metrics[f'target_{i}_r2'] = target_r2[i]
        
        return metrics


class QuantileRegressionHead(BaseTaskHead):
    """Head for quantile regression to predict prediction intervals."""
    
    def __init__(
        self,
        input_dim: int,
        quantiles: List[float] = [0.1, 0.5, 0.9],
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.1,
        activation: str = "relu"
    ):
        super().__init__(input_dim, TaskType.QUANTILE_REGRESSION, dropout, activation)
        
        self.quantiles = sorted(quantiles)
        self.num_quantiles = len(quantiles)
        
        # Build network
        if hidden_dims:
            layers = []
            prev_dim = input_dim
            for hidden_dim in hidden_dims:
                layers.extend([
                    nn.Linear(prev_dim, hidden_dim),
                    self.activation_fn,
                    self.dropout_layer
                ])
                prev_dim = hidden_dim
            layers.append(nn.Linear(prev_dim, self.num_quantiles))
            self.network = nn.Sequential(*layers)
        else:
            self.network = nn.Linear(input_dim, self.num_quantiles)
    
    def forward(
        self,
        features: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        **kwargs
    ) -> TaskOutput:
        """Forward pass through quantile regression head."""
        predictions = self.network(features)
        
        # Compute loss if targets provided
        loss = None
        if targets is not None:
            loss = self.compute_loss(predictions, targets, **kwargs)
        
        return TaskOutput(
            predictions=predictions,
            loss=loss,
            features=features,
            metadata={
                'quantiles': self.quantiles,
                'num_quantiles': self.num_quantiles
            }
        )
    
    def compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """Compute quantile regression loss (pinball loss)."""
        targets = targets.float()
        
        # Expand targets to match quantile predictions
        if targets.dim() == 1:
            targets = targets.unsqueeze(-1).expand(-1, self.num_quantiles)
        
        # Compute quantile loss for each quantile
        quantile_losses = []
        for i, q in enumerate(self.quantiles):
            pred_q = predictions[:, i]
            target_q = targets[:, i] if targets.shape[1] == self.num_quantiles else targets[:, 0]
            
            # Pinball loss
            error = target_q - pred_q
            loss_q = torch.where(error >= 0, q * error, (q - 1) * error)
            quantile_losses.append(loss_q)
        
        # Average across quantiles and samples
        total_loss = torch.stack(quantile_losses, dim=-1).mean()
        return total_loss
    
    def compute_metrics(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        **kwargs
    ) -> Dict[str, float]:
        """Compute quantile regression metrics."""
        targets = targets.float()
        
        # Get median prediction (usually 0.5 quantile)
        median_idx = self.quantiles.index(0.5) if 0.5 in self.quantiles else len(self.quantiles) // 2
        median_pred = predictions[:, median_idx]
        target_values = targets[:, 0] if targets.shape[1] > 1 else targets.squeeze()
        
        # Basic regression metrics for median
        mse = torch.mean((median_pred - target_values) ** 2).item()
        mae = torch.mean(torch.abs(median_pred - target_values)).item()
        
        metrics = {
            'median_mse': mse,
            'median_mae': mae,
            'median_rmse': np.sqrt(mse)
        }
        
        # Prediction interval coverage (if we have confidence intervals)
        if len(self.quantiles) >= 3:
            # Assume symmetric intervals around median
            lower_idx = 0
            upper_idx = -1
            
            lower_pred = predictions[:, lower_idx]
            upper_pred = predictions[:, upper_idx]
            
            # Coverage probability
            coverage = ((target_values >= lower_pred) & (target_values <= upper_pred)).float().mean().item()
            
            # Interval width
            interval_width = torch.mean(upper_pred - lower_pred).item()
            
            metrics.update({
                'coverage_probability': coverage,
                'mean_interval_width': interval_width,
                f'expected_coverage': self.quantiles[upper_idx] - self.quantiles[lower_idx]
            })
        
        return metrics
    
    def get_prediction_intervals(
        self,
        predictions: torch.Tensor,
        confidence_level: float = 0.8
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extract prediction intervals from quantile predictions.
        
        Args:
            predictions: Quantile predictions [batch_size, num_quantiles]
            confidence_level: Desired confidence level (e.g., 0.8 for 80% interval)
            
        Returns:
            Tuple of (median, lower_bound, upper_bound)
        """
        # Find quantiles closest to the desired interval
        alpha = (1 - confidence_level) / 2
        lower_q = alpha
        upper_q = 1 - alpha
        
        # Find closest quantile indices
        lower_idx = min(range(len(self.quantiles)), key=lambda i: abs(self.quantiles[i] - lower_q))
        upper_idx = min(range(len(self.quantiles)), key=lambda i: abs(self.quantiles[i] - upper_q))
        median_idx = min(range(len(self.quantiles)), key=lambda i: abs(self.quantiles[i] - 0.5))
        
        median = predictions[:, median_idx]
        lower_bound = predictions[:, lower_idx]
        upper_bound = predictions[:, upper_idx]
        
        return median, lower_bound, upper_bound


class RobustRegressionHead(RegressionHead):
    """Robust regression head with outlier-resistant loss functions."""
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.1,
        activation: str = "relu",
        robust_loss_type: str = "huber",
        huber_delta: float = 1.0,
        quantile_alpha: float = 0.1
    ):
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
            activation=activation,
            loss_type=robust_loss_type,
            huber_delta=huber_delta
        )
        
        self.robust_loss_type = robust_loss_type
        self.quantile_alpha = quantile_alpha
        
        # Additional robust loss functions
        if robust_loss_type == "quantile":
            self.loss_fn = self._quantile_loss
        elif robust_loss_type == "log_cosh":
            self.loss_fn = self._log_cosh_loss
    
    def _quantile_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Quantile loss for robust regression."""
        error = targets - predictions
        loss = torch.where(error >= 0, self.quantile_alpha * error, (self.quantile_alpha - 1) * error)
        return loss.mean()
    
    def _log_cosh_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Log-cosh loss for robust regression."""
        error = predictions - targets
        return torch.mean(torch.log(torch.cosh(error)))
    
    def compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """Compute robust regression loss."""
        if self.robust_loss_type in ["quantile", "log_cosh"]:
            return self.loss_fn(predictions, targets.float())
        else:
            return super().compute_loss(predictions, targets, **kwargs)