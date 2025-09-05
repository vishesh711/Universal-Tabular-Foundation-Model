"""Survival analysis head for TabGPT."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple
import numpy as np
from .base import BaseTaskHead, TaskType, TaskOutput


class SurvivalHead(BaseTaskHead):
    """Survival analysis head for time-to-event prediction."""
    
    def __init__(
        self,
        input_dim: int,
        num_time_bins: int = 100,
        max_time: Optional[float] = None,
        risk_estimation_method: str = "cox",  # cox, discrete_time, parametric
        distribution: str = "weibull",  # weibull, exponential, log_normal
        dropout: float = 0.1,
        activation: str = "relu",
        hidden_dims: Optional[list] = None
    ):
        super().__init__(input_dim, TaskType.SURVIVAL_ANALYSIS, dropout, activation)
        
        self.num_time_bins = num_time_bins
        self.max_time = max_time
        self.risk_estimation_method = risk_estimation_method
        self.distribution = distribution
        self.hidden_dims = hidden_dims or [input_dim // 2]
        
        # Build the head architecture
        self._build_head()
        
    def _build_head(self):
        """Build the survival analysis head."""
        layers = []
        current_dim = self.input_dim
        
        # Hidden layers
        for hidden_dim in self.hidden_dims:
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                self.activation_fn,
                nn.Dropout(self.dropout)
            ])
            if self.use_batch_norm:
                layers.insert(-1, nn.BatchNorm1d(hidden_dim))
            current_dim = hidden_dim
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # Task-specific output layers
        if self.risk_estimation_method == "cox":
            # Cox proportional hazards - single risk score
            self.risk_head = nn.Linear(current_dim, 1)
        elif self.risk_estimation_method == "discrete_time":
            # Discrete-time survival - probability for each time bin
            self.hazard_head = nn.Linear(current_dim, self.num_time_bins)
        elif self.risk_estimation_method == "parametric":
            # Parametric survival - distribution parameters
            if self.distribution == "weibull":
                self.shape_head = nn.Linear(current_dim, 1)  # Shape parameter
                self.scale_head = nn.Linear(current_dim, 1)  # Scale parameter
            elif self.distribution == "exponential":
                self.rate_head = nn.Linear(current_dim, 1)   # Rate parameter
            elif self.distribution == "log_normal":
                self.mu_head = nn.Linear(current_dim, 1)     # Mean of log
                self.sigma_head = nn.Linear(current_dim, 1)  # Std of log
        
    def forward(
        self,
        features: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        **kwargs
    ) -> TaskOutput:
        """
        Forward pass of survival head.
        
        Args:
            features: Input features [batch_size, input_dim]
            targets: Optional target values
            
        Returns:
            TaskOutput containing survival predictions
        """
        # Extract features
        extracted_features = self.feature_extractor(features)
        
        outputs = {}
        predictions = None
        
        if self.risk_estimation_method == "cox":
            # Cox model - risk score (log hazard ratio)
            risk_score = self.risk_head(extracted_features)
            outputs["risk_score"] = risk_score
            outputs["hazard_ratio"] = torch.exp(risk_score)
            predictions = risk_score
            
        elif self.risk_estimation_method == "discrete_time":
            # Discrete-time model - hazard probabilities
            hazard_logits = self.hazard_head(extracted_features)
            hazard_probs = torch.sigmoid(hazard_logits)
            outputs["hazard_probs"] = hazard_probs
            
            # Compute survival probabilities
            survival_probs = torch.cumprod(1 - hazard_probs, dim=1)
            outputs["survival_probs"] = survival_probs
            predictions = hazard_probs
            
        elif self.risk_estimation_method == "parametric":
            if self.distribution == "weibull":
                shape = F.softplus(self.shape_head(extracted_features)) + 1e-6
                scale = F.softplus(self.scale_head(extracted_features)) + 1e-6
                outputs["shape"] = shape
                outputs["scale"] = scale
                predictions = torch.cat([shape, scale], dim=-1)
                
            elif self.distribution == "exponential":
                rate = F.softplus(self.rate_head(extracted_features)) + 1e-6
                outputs["rate"] = rate
                predictions = rate
                
            elif self.distribution == "log_normal":
                mu = self.mu_head(extracted_features)
                sigma = F.softplus(self.sigma_head(extracted_features)) + 1e-6
                outputs["mu"] = mu
                outputs["sigma"] = sigma
                predictions = torch.cat([mu, sigma], dim=-1)
        
        # Compute loss if targets provided
        loss = None
        if targets is not None:
            loss = self.compute_loss(predictions, targets)
        
        return TaskOutput(
            predictions=predictions,
            loss=loss,
            features=extracted_features,
            metadata=outputs
        )
    
    def compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute survival analysis loss.
        
        Args:
            predictions: Model predictions tensor
            targets: Target values
            
        Returns:
            Loss tensor
        """
        # For compatibility, we'll use a simple MSE loss as placeholder
        # In practice, you'd extract the survival-specific targets
        if predictions is None:
            predictions = torch.zeros_like(targets)
        return nn.MSELoss()(predictions, targets.float())
    
    def _cox_loss(
        self,
        risk_scores: torch.Tensor,
        time: torch.Tensor,
        event: torch.Tensor
    ) -> torch.Tensor:
        """Compute Cox proportional hazards loss (partial likelihood)."""
        # Sort by time (descending)
        sorted_indices = torch.argsort(time, descending=True)
        sorted_risk_scores = risk_scores[sorted_indices].squeeze()
        sorted_event = event[sorted_indices]
        
        # Compute risk sets (cumulative sum of exp(risk_scores))
        exp_risk_scores = torch.exp(sorted_risk_scores)
        risk_sets = torch.cumsum(exp_risk_scores, dim=0)
        
        # Partial likelihood for observed events
        log_likelihood = sorted_risk_scores - torch.log(risk_sets + 1e-8)
        
        # Only consider uncensored events
        loss = -torch.sum(log_likelihood * sorted_event) / (torch.sum(sorted_event) + 1e-8)
        
        return loss
    
    def _discrete_time_loss(
        self,
        hazard_probs: torch.Tensor,
        time: torch.Tensor,
        event: torch.Tensor
    ) -> torch.Tensor:
        """Compute discrete-time survival loss."""
        batch_size = hazard_probs.shape[0]
        
        # Convert continuous time to discrete bins
        if self.max_time is not None:
            time_bins = (time / self.max_time * self.num_time_bins).long()
        else:
            time_bins = (time * self.num_time_bins / time.max()).long()
        
        time_bins = torch.clamp(time_bins, 0, self.num_time_bins - 1)
        
        # Create masks for loss computation
        loss = 0.0
        for i in range(batch_size):
            t_bin = time_bins[i]
            is_event = event[i]
            
            if is_event:
                # Event occurred - maximize hazard at time t_bin
                loss += -torch.log(hazard_probs[i, t_bin] + 1e-8)
                # Minimize hazard before time t_bin
                if t_bin > 0:
                    loss += -torch.sum(torch.log(1 - hazard_probs[i, :t_bin] + 1e-8))
            else:
                # Censored - minimize hazard up to time t_bin
                loss += -torch.sum(torch.log(1 - hazard_probs[i, :t_bin+1] + 1e-8))
        
        return loss / batch_size
    
    def _parametric_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        time: torch.Tensor,
        event: torch.Tensor
    ) -> torch.Tensor:
        """Compute parametric survival loss."""
        if self.distribution == "weibull":
            return self._weibull_loss(predictions, time, event)
        elif self.distribution == "exponential":
            return self._exponential_loss(predictions, time, event)
        elif self.distribution == "log_normal":
            return self._log_normal_loss(predictions, time, event)
        else:
            raise ValueError(f"Unknown distribution: {self.distribution}")
    
    def _weibull_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        time: torch.Tensor,
        event: torch.Tensor
    ) -> torch.Tensor:
        """Compute Weibull distribution loss."""
        shape = predictions["shape"].squeeze()
        scale = predictions["scale"].squeeze()
        
        # Log-likelihood for Weibull distribution
        log_pdf = (torch.log(shape) - torch.log(scale) + 
                  (shape - 1) * (torch.log(time) - torch.log(scale)) -
                  torch.pow(time / scale, shape))
        
        log_survival = -torch.pow(time / scale, shape)
        
        # Combine based on censoring
        log_likelihood = event * log_pdf + (1 - event) * log_survival
        
        return -torch.mean(log_likelihood)
    
    def _exponential_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        time: torch.Tensor,
        event: torch.Tensor
    ) -> torch.Tensor:
        """Compute exponential distribution loss."""
        rate = predictions["rate"].squeeze()
        
        # Log-likelihood for exponential distribution
        log_pdf = torch.log(rate) - rate * time
        log_survival = -rate * time
        
        # Combine based on censoring
        log_likelihood = event * log_pdf + (1 - event) * log_survival
        
        return -torch.mean(log_likelihood)
    
    def _log_normal_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        time: torch.Tensor,
        event: torch.Tensor
    ) -> torch.Tensor:
        """Compute log-normal distribution loss."""
        mu = predictions["mu"].squeeze()
        sigma = predictions["sigma"].squeeze()
        
        log_time = torch.log(time + 1e-8)
        
        # Log-likelihood for log-normal distribution
        log_pdf = (-torch.log(sigma) - torch.log(time) - 0.5 * torch.log(2 * torch.pi) -
                  0.5 * torch.pow((log_time - mu) / sigma, 2))
        
        # Survival function using complementary error function
        z = (log_time - mu) / (sigma * np.sqrt(2))
        log_survival = torch.log(0.5 * torch.erfc(z) + 1e-8)
        
        # Combine based on censoring
        log_likelihood = event * log_pdf + (1 - event) * log_survival
        
        return -torch.mean(log_likelihood)
    
    def compute_metrics(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """
        Compute survival analysis metrics.
        
        Args:
            predictions: Model predictions
            targets: Target values
            
        Returns:
            Dictionary of metrics
        """
        time = targets["time"]
        event = targets["event"]
        
        metrics = {}
        
        if self.risk_estimation_method == "cox":
            # Concordance index (C-index)
            risk_scores = predictions["risk_score"].squeeze()
            c_index = self._compute_concordance_index(risk_scores, time, event)
            metrics["c_index"] = c_index.item()
            
        elif self.risk_estimation_method == "discrete_time":
            # Brier score and integrated Brier score
            survival_probs = predictions["survival_probs"]
            brier_score = self._compute_brier_score(survival_probs, time, event)
            metrics["brier_score"] = brier_score.item()
            
        # Add common metrics
        if "risk_score" in predictions:
            metrics["mean_risk_score"] = predictions["risk_score"].mean().item()
            metrics["std_risk_score"] = predictions["risk_score"].std().item()
        
        return metrics
    
    def _compute_concordance_index(
        self,
        risk_scores: torch.Tensor,
        time: torch.Tensor,
        event: torch.Tensor
    ) -> torch.Tensor:
        """Compute concordance index (C-index)."""
        n = len(time)
        concordant = 0
        comparable = 0
        
        for i in range(n):
            for j in range(i + 1, n):
                if event[i] == 1 and time[i] < time[j]:
                    # Patient i had event before patient j
                    comparable += 1
                    if risk_scores[i] > risk_scores[j]:
                        concordant += 1
                elif event[j] == 1 and time[j] < time[i]:
                    # Patient j had event before patient i
                    comparable += 1
                    if risk_scores[j] > risk_scores[i]:
                        concordant += 1
        
        if comparable == 0:
            return torch.tensor(0.5)
        
        return torch.tensor(concordant / comparable)
    
    def _compute_brier_score(
        self,
        survival_probs: torch.Tensor,
        time: torch.Tensor,
        event: torch.Tensor
    ) -> torch.Tensor:
        """Compute Brier score for survival prediction."""
        # Simplified Brier score computation
        # In practice, you'd want to use proper time-dependent evaluation
        
        # Convert to binary classification at median time
        median_time_idx = self.num_time_bins // 2
        predicted_survival = survival_probs[:, median_time_idx]
        
        # True survival at median time (simplified)
        median_time = self.max_time * 0.5 if self.max_time else time.median()
        true_survival = (time > median_time).float()
        
        brier_score = torch.mean((predicted_survival - true_survival) ** 2)
        return brier_score
    
    def predict_survival_function(
        self,
        predictions: Dict[str, torch.Tensor],
        time_points: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Predict survival function at given time points.
        
        Args:
            predictions: Model predictions
            time_points: Time points for evaluation
            
        Returns:
            Survival probabilities at time points
        """
        if time_points is None:
            if self.max_time is not None:
                time_points = torch.linspace(0, self.max_time, self.num_time_bins)
            else:
                time_points = torch.linspace(0, 1, self.num_time_bins)
        
        if self.risk_estimation_method == "discrete_time":
            return predictions["survival_probs"]
        
        elif self.risk_estimation_method == "parametric":
            batch_size = list(predictions.values())[0].shape[0]
            survival_probs = torch.zeros(batch_size, len(time_points))
            
            for i, t in enumerate(time_points):
                if self.distribution == "weibull":
                    shape = predictions["shape"].squeeze()
                    scale = predictions["scale"].squeeze()
                    survival_probs[:, i] = torch.exp(-torch.pow(t / scale, shape))
                    
                elif self.distribution == "exponential":
                    rate = predictions["rate"].squeeze()
                    survival_probs[:, i] = torch.exp(-rate * t)
                    
                elif self.distribution == "log_normal":
                    mu = predictions["mu"].squeeze()
                    sigma = predictions["sigma"].squeeze()
                    z = (torch.log(t + 1e-8) - mu) / (sigma * np.sqrt(2))
                    survival_probs[:, i] = 0.5 * torch.erfc(z)
            
            return survival_probs
        
        else:
            # For Cox model, need baseline hazard (not implemented here)
            raise NotImplementedError("Survival function prediction for Cox model requires baseline hazard estimation")