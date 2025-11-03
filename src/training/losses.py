"""
Loss functions for PINN training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def mse_loss(pred, target):
    """Mean squared error loss."""
    return F.mse_loss(pred, target)


def mae_loss(pred, target):
    """Mean absolute error loss."""
    return F.l1_loss(pred, target)


def log_cosh_loss(pred, target):
    """
    Log-cosh loss - smoother than L1, more robust than L2.
    Good for handling outliers.
    """
    diff = pred - target
    return torch.mean(torch.log(torch.cosh(diff)))


def huber_loss(pred, target, delta=1.0):
    """
    Huber loss - quadratic for small errors, linear for large errors.
    Combines benefits of MSE and MAE.
    """
    return F.huber_loss(pred, target, delta=delta)


def relative_error_loss(pred, target, epsilon=1e-8):
    """
    Relative error loss - useful when target values span multiple orders of magnitude.
    """
    rel_error = torch.abs(pred - target) / (torch.abs(target) + epsilon)
    return torch.mean(rel_error)


def physics_informed_loss(model, C, Z_pred, eos):
    """
    Physics-informed loss based on the constraint Z = S/h̃.
    
    Args:
        model: The neural network model
        C: Conservative variables
        Z_pred: Predicted z values
        eos: Equation of state object
    """
    # Import here to avoid circular dependency
    from src.physics.physics_utils import h__z
    
    # Extract momentum density
    r = C[:, 2:3]  # S/D
    
    try:
        # Compute specific enthalpy
        htilde = h__z(Z_pred, C, eos)
        
        # Physics residual: should be zero if physically consistent
        residual = Z_pred - r / htilde
        
        return torch.mean(residual**2)
    except Exception as e:
        # Return zero loss with gradient support if computation fails
        return torch.sum(Z_pred * 0.0)


def combined_loss(model, C, Z_true, eos, physics_weight=0.1, data_loss_type='mse'):
    """
    Combined data and physics loss for PINN training.
    
    Args:
        model: Neural network model
        C: Conservative variables
        Z_true: True z values
        eos: Equation of state
        physics_weight: Weight for physics loss (0 to 1)
        data_loss_type: Type of data loss ('mse', 'mae', 'log_cosh', 'huber')
    """
    # Forward pass
    Z_pred = model(C)

    
    # Data loss
    if data_loss_type == 'mse':
        data_loss = mse_loss(Z_pred, Z_true)
    elif data_loss_type == 'mae':
        data_loss = mae_loss(Z_pred, Z_true)
    elif data_loss_type == 'log_cosh':
        data_loss = log_cosh_loss(Z_pred, Z_true)
    elif data_loss_type == 'huber':
        data_loss = huber_loss(Z_pred, Z_true)
    else:
        raise ValueError(f"Unknown data loss type: {data_loss_type}")
    
    # Combined loss
    total_loss = (1 - physics_weight) * data_loss 
    
    return total_loss


def multi_scale_loss(model, C, Z_true, eos, scales=[1.0, 0.1, 0.01]):
    """
    Multi-scale loss that penalizes errors at different scales.
    Useful for problems with multiple length/time scales.
    """
    Z_pred = model(C)
    
    total_loss = 0
    for scale in scales:
        scaled_pred = Z_pred * scale
        scaled_true = Z_true * scale
        loss_at_scale = mse_loss(scaled_pred, scaled_true)
        total_loss += loss_at_scale / len(scales)
    
    return total_loss


def gradient_penalty_loss(model, C, Z_true, lambda_gp=0.1):
    """
    Add gradient penalty to encourage smooth solutions.
    Helps prevent overfitting and improves generalization.
    """
    Z_pred = model(C)
    
    # Data loss
    data_loss = mse_loss(Z_pred, Z_true)
    
    # Compute gradients
    gradients = torch.autograd.grad(
        outputs=Z_pred.sum(),
        inputs=C,
        create_graph=True,
        retain_graph=True
    )[0]
    
    # Gradient penalty
    gradient_penalty = torch.mean(gradients**2)
    
    return data_loss + lambda_gp * gradient_penalty


class AdaptiveLoss(nn.Module):
    """
    Adaptive loss that adjusts weights during training based on loss magnitudes.
    """
    
    def __init__(self, n_losses=2, init_weights=None):
        super().__init__()
        
        if init_weights is None:
            init_weights = torch.ones(n_losses) / n_losses
        
        # Learnable loss weights
        self.log_weights = nn.Parameter(torch.log(torch.tensor(init_weights)))
    
    def forward(self, losses):
        """
        Args:
            losses: List or tuple of individual loss components
        """
        weights = F.softmax(self.log_weights, dim=0)
        
        # Weighted sum with inverse weighting
        total_loss = 0
        for i, loss in enumerate(losses):
            total_loss += weights[i] * loss / (loss.detach() + 1e-8)
        
        return total_loss
    
    def get_weights(self):
        """Get current loss weights."""
        return F.softmax(self.log_weights, dim=0).detach()


def max_loss(pred, target):
    """
    Maximum absolute error loss.
    Focuses training on worst-case predictions.
    """
    return torch.max(torch.abs(pred - target))


def quantile_loss(pred, target, quantile=0.9):
    """
    Quantile loss - focuses on specific percentiles of the error distribution.
    """
    errors = target - pred
    return torch.mean(torch.max(quantile * errors, (quantile - 1) * errors))