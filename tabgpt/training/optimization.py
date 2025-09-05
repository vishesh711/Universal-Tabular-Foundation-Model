"""Optimization utilities for TabGPT training."""
import math
from typing import Optional, Union, List
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class LinearWarmupCosineAnnealingLR(_LRScheduler):
    """
    Linear warmup followed by cosine annealing learning rate scheduler.
    
    Args:
        optimizer: Wrapped optimizer
        warmup_steps: Number of warmup steps
        total_steps: Total number of training steps
        min_lr_ratio: Minimum learning rate as a ratio of initial lr
        last_epoch: The index of last epoch
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr_ratio: float = 0.0,
        last_epoch: int = -1
    ):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            return [
                base_lr * (self.last_epoch + 1) / self.warmup_steps
                for base_lr in self.base_lrs
            ]
        else:
            # Cosine annealing
            progress = (self.last_epoch - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            progress = min(progress, 1.0)
            
            return [
                self.min_lr_ratio * base_lr + 
                (base_lr - self.min_lr_ratio * base_lr) * 
                (1 + math.cos(math.pi * progress)) / 2
                for base_lr in self.base_lrs
            ]


class PolynomialDecayLR(_LRScheduler):
    """
    Polynomial decay learning rate scheduler.
    
    Args:
        optimizer: Wrapped optimizer
        total_steps: Total number of training steps
        power: Power of polynomial decay
        min_lr_ratio: Minimum learning rate as a ratio of initial lr
        last_epoch: The index of last epoch
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        total_steps: int,
        power: float = 1.0,
        min_lr_ratio: float = 0.0,
        last_epoch: int = -1
    ):
        self.total_steps = total_steps
        self.power = power
        self.min_lr_ratio = min_lr_ratio
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        progress = min(self.last_epoch / self.total_steps, 1.0)
        decay_factor = (1 - progress) ** self.power
        
        return [
            self.min_lr_ratio * base_lr + 
            (base_lr - self.min_lr_ratio * base_lr) * decay_factor
            for base_lr in self.base_lrs
        ]


class WarmupConstantLR(_LRScheduler):
    """
    Linear warmup followed by constant learning rate.
    
    Args:
        optimizer: Wrapped optimizer
        warmup_steps: Number of warmup steps
        last_epoch: The index of last epoch
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        last_epoch: int = -1
    ):
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            return [
                base_lr * (self.last_epoch + 1) / self.warmup_steps
                for base_lr in self.base_lrs
            ]
        else:
            # Constant learning rate
            return self.base_lrs


def get_scheduler(
    optimizer: Optimizer,
    scheduler_type: str,
    num_training_steps: int,
    warmup_steps: int = 0,
    **kwargs
) -> _LRScheduler:
    """
    Get learning rate scheduler by type.
    
    Args:
        optimizer: Optimizer to wrap
        scheduler_type: Type of scheduler ('linear', 'cosine', 'polynomial', 'constant')
        num_training_steps: Total number of training steps
        warmup_steps: Number of warmup steps
        **kwargs: Additional scheduler-specific arguments
        
    Returns:
        Learning rate scheduler
    """
    if scheduler_type == "linear":
        from torch.optim.lr_scheduler import LinearLR
        return LinearLR(
            optimizer,
            start_factor=kwargs.get('start_factor', 1.0),
            end_factor=kwargs.get('end_factor', 0.0),
            total_iters=num_training_steps
        )
    
    elif scheduler_type == "cosine":
        if warmup_steps > 0:
            return LinearWarmupCosineAnnealingLR(
                optimizer,
                warmup_steps=warmup_steps,
                total_steps=num_training_steps,
                min_lr_ratio=kwargs.get('min_lr_ratio', 0.0)
            )
        else:
            from torch.optim.lr_scheduler import CosineAnnealingLR
            return CosineAnnealingLR(
                optimizer,
                T_max=num_training_steps,
                eta_min=kwargs.get('eta_min', 0)
            )
    
    elif scheduler_type == "polynomial":
        return PolynomialDecayLR(
            optimizer,
            total_steps=num_training_steps,
            power=kwargs.get('power', 1.0),
            min_lr_ratio=kwargs.get('min_lr_ratio', 0.0)
        )
    
    elif scheduler_type == "constant":
        if warmup_steps > 0:
            return WarmupConstantLR(optimizer, warmup_steps=warmup_steps)
        else:
            from torch.optim.lr_scheduler import ConstantLR
            return ConstantLR(optimizer, factor=1.0)
    
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")


class AdamWWithDecoupledWeightDecay(torch.optim.AdamW):
    """
    AdamW optimizer with improved weight decay handling.
    
    This implementation follows the original AdamW paper more closely
    and provides better separation of weight decay from gradient-based updates.
    """
    
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        amsgrad: bool = False,
        maximize: bool = False,
        foreach: Optional[bool] = None,
        capturable: bool = False,
        differentiable: bool = False,
        fused: Optional[bool] = None,
    ):
        super().__init__(
            params=params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            maximize=maximize,
            foreach=foreach,
            capturable=capturable,
            differentiable=differentiable,
            fused=fused,
        )


def create_optimizer(
    model: torch.nn.Module,
    optimizer_type: str = "adamw",
    learning_rate: float = 5e-4,
    weight_decay: float = 0.01,
    betas: tuple = (0.9, 0.999),
    eps: float = 1e-8,
    no_decay_params: Optional[List[str]] = None,
    **kwargs
) -> Optimizer:
    """
    Create optimizer with proper weight decay handling.
    
    Args:
        model: Model to optimize
        optimizer_type: Type of optimizer ('adamw', 'adam', 'sgd')
        learning_rate: Learning rate
        weight_decay: Weight decay coefficient
        betas: Adam beta parameters
        eps: Adam epsilon parameter
        no_decay_params: List of parameter name patterns to exclude from weight decay
        **kwargs: Additional optimizer arguments
        
    Returns:
        Configured optimizer
    """
    if no_decay_params is None:
        no_decay_params = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    
    # Separate parameters for weight decay
    decay_parameters = []
    no_decay_parameters = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            if any(nd in name for nd in no_decay_params):
                no_decay_parameters.append(param)
            else:
                decay_parameters.append(param)
    
    optimizer_grouped_parameters = [
        {
            "params": decay_parameters,
            "weight_decay": weight_decay,
        },
        {
            "params": no_decay_parameters,
            "weight_decay": 0.0,
        },
    ]
    
    if optimizer_type.lower() == "adamw":
        return AdamWWithDecoupledWeightDecay(
            optimizer_grouped_parameters,
            lr=learning_rate,
            betas=betas,
            eps=eps,
            **kwargs
        )
    elif optimizer_type.lower() == "adam":
        return torch.optim.Adam(
            optimizer_grouped_parameters,
            lr=learning_rate,
            betas=betas,
            eps=eps,
            **kwargs
        )
    elif optimizer_type.lower() == "sgd":
        return torch.optim.SGD(
            optimizer_grouped_parameters,
            lr=learning_rate,
            momentum=kwargs.get('momentum', 0.9),
            **kwargs
        )
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")


class GradientClipping:
    """Utility class for gradient clipping."""
    
    @staticmethod
    def clip_grad_norm(
        parameters,
        max_norm: float,
        norm_type: float = 2.0,
        error_if_nonfinite: bool = False
    ) -> torch.Tensor:
        """
        Clip gradient norm of parameters.
        
        Args:
            parameters: Iterable of parameters or single parameter
            max_norm: Maximum norm of gradients
            norm_type: Type of norm to use
            error_if_nonfinite: Whether to raise error on non-finite gradients
            
        Returns:
            Total norm of parameters
        """
        return torch.nn.utils.clip_grad_norm_(
            parameters,
            max_norm,
            norm_type=norm_type,
            error_if_nonfinite=error_if_nonfinite
        )
    
    @staticmethod
    def clip_grad_value(parameters, clip_value: float):
        """
        Clip gradient values of parameters.
        
        Args:
            parameters: Iterable of parameters or single parameter
            clip_value: Maximum absolute value of gradients
        """
        torch.nn.utils.clip_grad_value_(parameters, clip_value)


class EarlyStopping:
    """Early stopping utility."""
    
    def __init__(
        self,
        patience: int = 7,
        min_delta: float = 0.0,
        restore_best_weights: bool = True,
        mode: str = "min"
    ):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs with no improvement to wait
            min_delta: Minimum change to qualify as improvement
            restore_best_weights: Whether to restore best weights when stopping
            mode: 'min' for minimizing metric, 'max' for maximizing
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.mode = mode
        
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        
        if mode == "min":
            self.monitor_op = lambda current, best: current < best - min_delta
        elif mode == "max":
            self.monitor_op = lambda current, best: current > best + min_delta
        else:
            raise ValueError(f"Mode {mode} is unknown, use 'min' or 'max'")
    
    def __call__(self, score: float, model: torch.nn.Module) -> bool:
        """
        Check if training should stop.
        
        Args:
            score: Current metric score
            model: Model to potentially save weights from
            
        Returns:
            True if training should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
            if self.restore_best_weights:
                self.best_weights = {k: v.clone() for k, v in model.state_dict().items()}
        elif self.monitor_op(score, self.best_score):
            self.best_score = score
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
        
        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
            return True
        
        return False


def compute_num_parameters(model: torch.nn.Module) -> dict:
    """
    Compute number of parameters in model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "non_trainable_parameters": total_params - trainable_params,
        "total_parameters_millions": total_params / 1e6,
        "trainable_parameters_millions": trainable_params / 1e6,
    }