#!/scratch/astro/miler/python-env/pytorch/bin/python3
# test_physics_guided_pinn.py
"""
Enhanced Physics-Guided PINN Comparison Script
Tests various physics-guided network architectures with different correction strategies
"""

import os
import sys
import time
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset, random_split

# Import your modules
from metric import metric
from hybrid_eos import hybrid_eos
from C2P_Solver import W__z, rho__z, eps__z, h__z
from data_utils import setup_initial_state_random, setup_initial_state_meshgrid_cold

# Define comprehensive_error_analysis and plot_zeta_scatter_analysis locally
def comprehensive_error_analysis(model_func, test_C, test_Z, eos):
    """
    Comprehensive error analysis for PINN model
    """
    with torch.no_grad():
        Z_pred = model_func(test_C)
        
        # Compute primitive variables from true Z
        Z_true = test_Z.view(-1, 1)
        rho_true = rho__z(Z_true, test_C)
        W_true = W__z(Z_true)
        eps_true = eps__z(Z_true, test_C)
        press_true = eos.press__eps_rho(eps_true, rho_true)
        
        # Compute primitive variables from predicted Z
        rho_pred = rho__z(Z_pred, test_C)
        W_pred = W__z(Z_pred)
        eps_pred = eps__z(Z_pred, test_C)
        press_pred = eos.press__eps_rho(eps_pred, rho_pred)
        
        # Calculate relative errors
        rho_error = torch.abs(rho_pred - rho_true) / torch.clamp(rho_true, min=1e-15)
        eps_error = torch.abs(eps_pred - eps_true) / torch.clamp(torch.abs(eps_true), min=1e-15)
        press_error = torch.abs(press_pred - press_true) / torch.clamp(torch.abs(press_true), min=1e-15)
        W_error = torch.abs(W_pred - W_true) / torch.clamp(W_true, min=1e-15)
        Z_error = torch.abs(Z_pred - Z_true) / torch.clamp(torch.abs(Z_true), min=1e-15)
        
        errors = {
            'rho_error_mean': torch.mean(rho_error).item(),
            'eps_error_mean': torch.mean(eps_error).item(),
            'press_error_mean': torch.mean(press_error).item(),
            'W_error_mean': torch.mean(W_error).item(),
            'Z_error_mean': torch.mean(Z_error).item()
        }
        
        return errors, Z_pred, Z_true

def plot_zeta_scatter_analysis(Z_true, Z_pred, save_fig=False, fig_path=None):
    """Simple scatter plot analysis"""
    Z_true_np = Z_true.detach().cpu().numpy().flatten()
    Z_pred_np = Z_pred.detach().cpu().numpy().flatten()
    
    plt.figure(figsize=(8, 6))
    plt.scatter(Z_true_np, Z_pred_np, alpha=0.6, s=20, c='blue')
    plt.plot([Z_true_np.min(), Z_true_np.max()], 
             [Z_true_np.min(), Z_true_np.max()], 'r--', lw=2, label='Perfect prediction')
    plt.xlabel('True Zeta (ζ)')
    plt.ylabel('Predicted Zeta (ζ)')
    plt.title('True vs Predicted Zeta Values')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    if save_fig:
        if fig_path is None:
            fig_path = 'zeta_scatter_analysis.png'
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()

def log_cosh_loss(y_pred, y_true):
    """
    Logarithm of the hyperbolic cosine of the prediction error.
    More robust to outliers than MSE.
    """
    diff = y_pred - y_true
    return torch.mean(torch.log(torch.cosh(diff)))

# =============================================================================
# PHYSICS-GUIDED PINN ARCHITECTURES
# =============================================================================

class PhysicsGuided_TinyPINN_v1(nn.Module):
    """Original: 3->6->1 with 0.1 correction scale"""
    def __init__(self, dtype=torch.float64):
        super().__init__()
        self.correction_net = nn.Sequential(
            nn.Linear(3, 6, dtype=dtype),
            nn.Tanh(),
            nn.Linear(6, 1, dtype=dtype),
        )
        self.correction_scale = 0.1
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=0.1)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    def analytical_guess(self, C):
        D, q, r = C[:, 0:1], C[:, 1:2], C[:, 2:3]
        z_kinematic = torch.sqrt(1.0 + r**2)
        thermal_factor = 1.0 + 0.5 * q
        relativistic_correction = 1.0 + 0.1 * r**2 / (1.0 + r**2)
        z_guess = z_kinematic * thermal_factor * relativistic_correction
        return torch.clamp(z_guess, min=1.0)
    
    def forward(self, x):
        z_baseline = self.analytical_guess(x)
        correction = self.correction_net(x)
        return z_baseline + self.correction_scale * correction
    
    def physics_loss(self, C, Z_pred, eos):
        try:
            htilde = h__z(Z_pred, C, eos)
            residual = Z_pred - (C[:, 2:3] / htilde)
            return torch.mean(residual**2)
        except Exception:
            return torch.sum(Z_pred * 0.0)

class PhysicsGuided_TinyPINN_v2(nn.Module):
    """Wider hidden layer: 3->12->1 with 0.1 correction scale"""
    def __init__(self, dtype=torch.float64):
        super().__init__()
        self.correction_net = nn.Sequential(
            nn.Linear(3, 12, dtype=dtype),
            nn.Tanh(),
            nn.Linear(12, 1, dtype=dtype),
        )
        self.correction_scale = 0.1
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=0.1)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    def analytical_guess(self, C):
        D, q, r = C[:, 0:1], C[:, 1:2], C[:, 2:3]
        z_kinematic = torch.sqrt(1.0 + r**2)
        thermal_factor = 1.0 + 0.5 * q
        relativistic_correction = 1.0 + 0.1 * r**2 / (1.0 + r**2)
        z_guess = z_kinematic * thermal_factor * relativistic_correction
        return torch.clamp(z_guess, min=1.0)
    
    def forward(self, x):
        z_baseline = self.analytical_guess(x)
        correction = self.correction_net(x)
        return z_baseline + self.correction_scale * correction
    
    def physics_loss(self, C, Z_pred, eos):
        try:
            htilde = h__z(Z_pred, C, eos)
            residual = Z_pred - (C[:, 2:3] / htilde)
            return torch.mean(residual**2)
        except Exception:
            return torch.sum(Z_pred * 0.0)

class PhysicsGuided_TinyPINN_v3(nn.Module):
    """Even wider: 3->24->1 with 0.1 correction scale"""
    def __init__(self, dtype=torch.float64):
        super().__init__()
        self.correction_net = nn.Sequential(
            nn.Linear(3, 24, dtype=dtype),
            nn.Tanh(),
            nn.Linear(24, 1, dtype=dtype),
        )
        self.correction_scale = 0.1
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=0.1)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    def analytical_guess(self, C):
        D, q, r = C[:, 0:1], C[:, 1:2], C[:, 2:3]
        z_kinematic = torch.sqrt(1.0 + r**2)
        thermal_factor = 1.0 + 0.5 * q
        relativistic_correction = 1.0 + 0.1 * r**2 / (1.0 + r**2)
        z_guess = z_kinematic * thermal_factor * relativistic_correction
        return torch.clamp(z_guess, min=1.0)
    
    def forward(self, x):
        z_baseline = self.analytical_guess(x)
        correction = self.correction_net(x)
        return z_baseline + self.correction_scale * correction
    
    def physics_loss(self, C, Z_pred, eos):
        try:
            htilde = h__z(Z_pred, C, eos)
            residual = Z_pred - (C[:, 2:3] / htilde)
            return torch.mean(residual**2)
        except Exception:
            return torch.sum(Z_pred * 0.0)

class PhysicsGuided_DeepTinyPINN_v1(nn.Module):
    """Deeper network: 3->8->4->1 with 0.1 correction scale"""
    def __init__(self, dtype=torch.float64):
        super().__init__()
        self.correction_net = nn.Sequential(
            nn.Linear(3, 8, dtype=dtype),
            nn.Tanh(),
            nn.Linear(8, 4, dtype=dtype),
            nn.Tanh(),
            nn.Linear(4, 1, dtype=dtype),
        )
        self.correction_scale = 0.1
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=0.1)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    def analytical_guess(self, C):
        D, q, r = C[:, 0:1], C[:, 1:2], C[:, 2:3]
        z_kinematic = torch.sqrt(1.0 + r**2)
        thermal_factor = 1.0 + 0.5 * q
        relativistic_correction = 1.0 + 0.1 * r**2 / (1.0 + r**2)
        z_guess = z_kinematic * thermal_factor * relativistic_correction
        return torch.clamp(z_guess, min=1.0)
    
    def forward(self, x):
        z_baseline = self.analytical_guess(x)
        correction = self.correction_net(x)
        return z_baseline + self.correction_scale * correction
    
    def physics_loss(self, C, Z_pred, eos):
        try:
            htilde = h__z(Z_pred, C, eos)
            residual = Z_pred - (C[:, 2:3] / htilde)
            return torch.mean(residual**2)
        except Exception:
            return torch.sum(Z_pred * 0.0)

class PhysicsGuided_DeepTinyPINN_v2(nn.Module):
    """Deeper network: 3->16->8->1 with 0.1 correction scale"""
    def __init__(self, dtype=torch.float64):
        super().__init__()
        self.correction_net = nn.Sequential(
            nn.Linear(3, 16, dtype=dtype),
            nn.SiLU(),
            nn.Linear(16, 8, dtype=dtype),
            nn.SiLU(),
            nn.Linear(8, 1, dtype=dtype),
        )
        self.correction_scale = 0.1
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=0.1)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    def analytical_guess(self, C):
        D, q, r = C[:, 0:1], C[:, 1:2], C[:, 2:3]
        z_kinematic = torch.sqrt(1.0 + r**2)
        thermal_factor = 1.0 + 0.5 * q
        relativistic_correction = 1.0 + 0.1 * r**2 / (1.0 + r**2)
        z_guess = z_kinematic * thermal_factor * relativistic_correction
        return torch.clamp(z_guess, min=1.0)
    
    def forward(self, x):
        z_baseline = self.analytical_guess(x)
        correction = self.correction_net(x)
        return z_baseline + self.correction_scale * correction
    
    def physics_loss(self, C, Z_pred, eos):
        try:
            htilde = h__z(Z_pred, C, eos)
            residual = Z_pred - (C[:, 2:3] / htilde)
            return torch.mean(residual**2)
        except Exception:
            return torch.sum(Z_pred * 0.0)

class PhysicsGuided_LargeCorrection_v1(nn.Module):
    """Larger correction scale: 3->12->1 with 0.2 correction scale"""
    def __init__(self, dtype=torch.float64):
        super().__init__()
        self.correction_net = nn.Sequential(
            nn.Linear(3, 12, dtype=dtype),
            nn.Tanh(),
            nn.Linear(12, 1, dtype=dtype),
        )
        self.correction_scale = 0.2
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=0.2)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    def analytical_guess(self, C):
        D, q, r = C[:, 0:1], C[:, 1:2], C[:, 2:3]
        z_kinematic = torch.sqrt(1.0 + r**2)
        thermal_factor = 1.0 + 0.5 * q
        relativistic_correction = 1.0 + 0.1 * r**2 / (1.0 + r**2)
        z_guess = z_kinematic * thermal_factor * relativistic_correction
        return torch.clamp(z_guess, min=1.0)
    
    def forward(self, x):
        z_baseline = self.analytical_guess(x)
        correction = self.correction_net(x)
        return z_baseline + self.correction_scale * correction
    
    def physics_loss(self, C, Z_pred, eos):
        try:
            htilde = h__z(Z_pred, C, eos)
            residual = Z_pred - (C[:, 2:3] / htilde)
            return torch.mean(residual**2)
        except Exception:
            return torch.sum(Z_pred * 0.0)

class PhysicsGuided_SmallCorrection_v1(nn.Module):
    """Smaller correction scale: 3->12->1 with 0.05 correction scale"""
    def __init__(self, dtype=torch.float64):
        super().__init__()
        self.correction_net = nn.Sequential(
            nn.Linear(3, 12, dtype=dtype),
            nn.Tanh(),
            nn.Linear(12, 1, dtype=dtype),
        )
        self.correction_scale = 0.05
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=0.05)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    def analytical_guess(self, C):
        D, q, r = C[:, 0:1], C[:, 1:2], C[:, 2:3]
        z_kinematic = torch.sqrt(1.0 + r**2)
        thermal_factor = 1.0 + 0.5 * q
        relativistic_correction = 1.0 + 0.1 * r**2 / (1.0 + r**2)
        z_guess = z_kinematic * thermal_factor * relativistic_correction
        return torch.clamp(z_guess, min=1.0)
    
    def forward(self, x):
        z_baseline = self.analytical_guess(x)
        correction = self.correction_net(x)
        return z_baseline + self.correction_scale * correction
    
    def physics_loss(self, C, Z_pred, eos):
        try:
            htilde = h__z(Z_pred, C, eos)
            residual = Z_pred - (C[:, 2:3] / htilde)
            return torch.mean(residual**2)
        except Exception:
            return torch.sum(Z_pred * 0.0)

class PhysicsGuided_ImprovedGuess_v1(nn.Module):
    """Improved analytical guess with better thermal modeling"""
    def __init__(self, dtype=torch.float64):
        super().__init__()
        self.correction_net = nn.Sequential(
            nn.Linear(3, 12, dtype=dtype),
            nn.Tanh(),
            nn.Linear(12, 1, dtype=dtype),
        )
        self.correction_scale = 0.1
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=0.1)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    def analytical_guess(self, C):
        D, q, r = C[:, 0:1], C[:, 1:2], C[:, 2:3]
        
        # More sophisticated analytical guess
        z_kinematic = torch.sqrt(1.0 + r**2)
        
        # Better thermal modeling
        thermal_factor = 1.0 + 0.8 * q / (1.0 + 0.1 * q)  # Saturating thermal effect
        
        # Improved relativistic correction
        relativistic_correction = 1.0 + 0.2 * r**2 / (1.0 + 0.5 * r**2)
        
        # Density-dependent correction
        density_factor = 1.0 + 0.05 * torch.log(1.0 + torch.abs(D))
        
        z_guess = z_kinematic * thermal_factor * relativistic_correction * density_factor
        return torch.clamp(z_guess, min=1.0)
    
    def forward(self, x):
        z_baseline = self.analytical_guess(x)
        correction = self.correction_net(x)
        return z_baseline + self.correction_scale * correction
    
    def physics_loss(self, C, Z_pred, eos):
        try:
            htilde = h__z(Z_pred, C, eos)
            residual = Z_pred - (C[:, 2:3] / htilde)
            return torch.mean(residual**2)
        except Exception:
            return torch.sum(Z_pred * 0.0)

class PhysicsGuided_ResidualCorrection_v1(nn.Module):
    """Residual connection approach"""
    def __init__(self, dtype=torch.float64):
        super().__init__()
        self.correction_net = nn.Sequential(
            nn.Linear(3, 16, dtype=dtype),
            nn.Tanh(),
            nn.Linear(16, 8, dtype=dtype),
            nn.Tanh(),
            nn.Linear(8, 1, dtype=dtype),
        )
        self.correction_scale = 0.1
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=0.1)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    def analytical_guess(self, C):
        D, q, r = C[:, 0:1], C[:, 1:2], C[:, 2:3]
        z_kinematic = torch.sqrt(1.0 + r**2)
        thermal_factor = 1.0 + 0.5 * q
        relativistic_correction = 1.0 + 0.1 * r**2 / (1.0 + r**2)
        z_guess = z_kinematic * thermal_factor * relativistic_correction
        return torch.clamp(z_guess, min=1.0)
    
    def forward(self, x):
        z_baseline = self.analytical_guess(x)
        
        # Residual connection: correction network gets both input and baseline
        correction_input = torch.cat([x, z_baseline], dim=-1)
        correction = self.correction_net(correction_input)
        
        return z_baseline + self.correction_scale * correction
    
    def physics_loss(self, C, Z_pred, eos):
        try:
            htilde = h__z(Z_pred, C, eos)
            residual = Z_pred - (C[:, 2:3] / htilde)
            return torch.mean(residual**2)
        except Exception:
            return torch.sum(Z_pred * 0.0)

class PhysicsGuided_AdaptiveCorrection_v1(nn.Module):
    """Adaptive correction scale based on input"""
    def __init__(self, dtype=torch.float64):
        super().__init__()
        self.correction_net = nn.Sequential(
            nn.Linear(3, 12, dtype=dtype),
            nn.Tanh(),
            nn.Linear(12, 1, dtype=dtype),
        )
        self.scale_net = nn.Sequential(
            nn.Linear(3, 8, dtype=dtype),
            nn.Tanh(),
            nn.Linear(8, 1, dtype=dtype),
            nn.Sigmoid(),
        )
        self.base_correction_scale = 0.1
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=0.1)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    def analytical_guess(self, C):
        D, q, r = C[:, 0:1], C[:, 1:2], C[:, 2:3]
        z_kinematic = torch.sqrt(1.0 + r**2)
        thermal_factor = 1.0 + 0.5 * q
        relativistic_correction = 1.0 + 0.1 * r**2 / (1.0 + r**2)
        z_guess = z_kinematic * thermal_factor * relativistic_correction
        return torch.clamp(z_guess, min=1.0)
    
    def forward(self, x):
        z_baseline = self.analytical_guess(x)
        correction = self.correction_net(x)
        adaptive_scale = self.scale_net(x) * self.base_correction_scale
        return z_baseline + adaptive_scale * correction
    
    def physics_loss(self, C, Z_pred, eos):
        try:
            htilde = h__z(Z_pred, C, eos)
            residual = Z_pred - (C[:, 2:3] / htilde)
            return torch.mean(residual**2)
        except Exception:
            return torch.sum(Z_pred * 0.0)

# Fix the ResidualCorrection network to handle the correct input size
class PhysicsGuided_ResidualCorrection_v1_Fixed(nn.Module):
    """Fixed residual connection approach"""
    def __init__(self, dtype=torch.float64):
        super().__init__()
        # Input size is 3 (original) + 1 (baseline) = 4
        self.correction_net = nn.Sequential(
            nn.Linear(4, 16, dtype=dtype),
            nn.Tanh(),
            nn.Linear(16, 8, dtype=dtype),
            nn.Tanh(),
            nn.Linear(8, 1, dtype=dtype),
        )
        self.correction_scale = 0.1
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=0.1)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    def analytical_guess(self, C):
        D, q, r = C[:, 0:1], C[:, 1:2], C[:, 2:3]
        z_kinematic = torch.sqrt(1.0 + r**2)
        thermal_factor = 1.0 + 0.5 * q
        relativistic_correction = 1.0 + 0.1 * r**2 / (1.0 + r**2)
        z_guess = z_kinematic * thermal_factor * relativistic_correction
        return torch.clamp(z_guess, min=1.0)
    
    def forward(self, x):
        z_baseline = self.analytical_guess(x)
        
        # Residual connection: correction network gets both input and baseline
        correction_input = torch.cat([x, z_baseline], dim=-1)
        correction = self.correction_net(correction_input)
        
        return z_baseline + self.correction_scale * correction
    
    def physics_loss(self, C, Z_pred, eos):
        try:
            htilde = h__z(Z_pred, C, eos)
            residual = Z_pred - (C[:, 2:3] / htilde)
            return torch.mean(residual**2)
        except Exception:
            return torch.sum(Z_pred * 0.0)

def count_parameters(model):
    """Count trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def estimate_flops(model, input_size=(1, 3)):
    """Rough FLOP estimation for forward pass"""
    total_flops = 0
    input_tensor = torch.randn(input_size)
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            total_flops += module.in_features * module.out_features + module.out_features
        elif isinstance(module, (nn.Tanh, nn.SiLU, nn.Sigmoid)):
            total_flops += input_tensor.numel()
    
    return total_flops

def setup_physics():
    """Setup metric and EOS"""
    g = torch.eye(3, dtype=torch.float64)
    beta = torch.zeros(3, dtype=torch.float64)
    alp = torch.tensor(1.0, dtype=torch.float64)
    
    metric_obj = metric(g, beta, alp)
    eos = hybrid_eos(100, 2, 1.8)
    
    return metric_obj, eos

def generate_test_data(metric_obj, eos, n_samples, device):
    """Generate test dataset"""
    print(f"Generating {n_samples} test samples...")
    
    C_test, Z_test = setup_initial_state_meshgrid_cold(
        metric_obj, eos, n_samples, device,
        lrhomin=-11, lrhomax=-2.8,
        Wmin=1.0, Wmax=2.0
    )
    
    C_test = C_test.to(dtype=torch.float64)
    Z_test = Z_test.to(dtype=torch.float64)
    
    return C_test, Z_test

def train_model(model, train_loader, val_loader, eos, device, epochs=200, lr=1e-3, physics_weight=1.0, use_log_cosh=False):
    """Train a single model"""
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-6)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    print(f"Training model with {count_parameters(model)} parameters...")
    print(f"Using {'Log-Cosh' if use_log_cosh else 'MSE'} loss...")
    
    for epoch in range(epochs):
        # Training
        model.train()
        epoch_train_loss = 0.0
        
        for batch_idx, (C_batch, Z_batch) in enumerate(train_loader):
            C_batch = C_batch.to(device, dtype=torch.float64)
            Z_batch = Z_batch.to(device, dtype=torch.float64)
            
            optimizer.zero_grad()
            
            # Forward pass
            Z_pred = model(C_batch)
            
            # Data loss - choose between MSE and Log-Cosh
            if use_log_cosh:
                data_loss = log_cosh_loss(Z_pred, Z_batch)
            else:
                data_loss = torch.mean((Z_pred - Z_batch)**2)
            
            # Physics loss
            physics_loss = model.physics_loss(C_batch, Z_pred, eos)
            
            # Total loss
            total_loss = data_loss + physics_weight * physics_loss
            
            total_loss.backward()
            optimizer.step()
            
            epoch_train_loss += total_loss.item()
        
        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        epoch_val_loss = 0.0
        
        with torch.no_grad():
            for C_batch, Z_batch in val_loader:
                C_batch = C_batch.to(device, dtype=torch.float64)
                Z_batch = Z_batch.to(device, dtype=torch.float64)
                Z_pred = model(C_batch)
                
                if use_log_cosh:
                    data_loss = log_cosh_loss(Z_pred, Z_batch)
                else:
                    data_loss = torch.mean((Z_pred - Z_batch)**2)
                physics_loss = model.physics_loss(C_batch, Z_pred, eos)
                total_loss = data_loss + physics_weight * physics_loss
                
                epoch_val_loss += total_loss.item()
        
        avg_val_loss = epoch_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        scheduler.step(avg_val_loss)
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
        
        if epoch % 50 == 0:
            print(f"Epoch {epoch:3d}: Train Loss = {avg_train_loss:.2e}, Val Loss = {avg_val_loss:.2e}")
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    return model, train_losses, val_losses

def evaluate_model(model, test_C, test_Z, eos, device):
    """Evaluate model performance"""
    model.eval()
    test_C = test_C.to(device, dtype=torch.float64)
    test_Z = test_Z.to(device, dtype=torch.float64)
    
    start_time = time.time()
    
    with torch.no_grad():
        Z_pred = model(test_C)
    
    inference_time = time.time() - start_time
    
    # Calculate errors
    abs_error = torch.abs(Z_pred - test_Z)
    rel_error = abs_error / (torch.abs(test_Z) + 1e-12)
    
    # Physics consistency
    physics_error = model.physics_loss(test_C, Z_pred, eos)
    
    # Compute primitive variables for comprehensive error analysis
    errors, Z_pred_out, Z_true_out = comprehensive_error_analysis(
        lambda C: model(C), test_C, test_Z, eos
    )
    
    # Safe correlation calculation
    Z_pred_flat = Z_pred.flatten()
    Z_test_flat = test_Z.flatten()
    
    if len(Z_pred_flat) > 1 and torch.std(Z_pred_flat) > 1e-8 and torch.std(Z_test_flat) > 1e-8:
        try:
            correlation_matrix = torch.corrcoef(torch.stack([Z_pred_flat, Z_test_flat]))
            correlation = correlation_matrix[0, 1].item()
        except:
            mean_pred = torch.mean(Z_pred_flat)
            mean_true = torch.mean(Z_test_flat)
            numerator = torch.sum((Z_pred_flat - mean_pred) * (Z_test_flat - mean_true))
            denominator = torch.sqrt(torch.sum((Z_pred_flat - mean_pred)**2) * torch.sum((Z_test_flat - mean_true)**2))
            correlation = (numerator / (denominator + 1e-12)).item()
    else:
        correlation = 0.0
    
    return {
        'abs_error_mean': torch.mean(abs_error).item(),
        'abs_error_max': torch.max(abs_error).item(),
        'abs_error_std': torch.std(abs_error).item(),
        'rel_error_mean': torch.mean(rel_error).item(),
        'rel_error_max': torch.max(rel_error).item(),
        'rel_error_std': torch.std(rel_error).item(),
        'physics_error': physics_error.item(),
        'inference_time': inference_time,
        'correlation': correlation,
        'comprehensive_errors': errors,
        'Z_pred': Z_pred_out,
        'Z_true': Z_true_out
    }

def run_comparison(args):
    """Main comparison function"""
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Setup physics
    metric_obj, eos = setup_physics()
    
    # Generate datasets
    print("\nGenerating datasets...")
    C_train, Z_train = generate_test_data(metric_obj, eos, args.n_train, device)
    C_val, Z_val = generate_test_data(metric_obj, eos, args.n_val, device)
    C_test, Z_test = generate_test_data(metric_obj, eos, args.n_test, device)
    
    # Create data loaders
    train_dataset = TensorDataset(C_train, Z_train)
    val_dataset = TensorDataset(C_val, Z_val)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Define ONLY physics-guided models to test
    models_to_test = {
        'PhysicsGuided_TinyPINN_v1': PhysicsGuided_TinyPINN_v1(),
        'PhysicsGuided_TinyPINN_v2': PhysicsGuided_TinyPINN_v2(),
        'PhysicsGuided_TinyPINN_v3': PhysicsGuided_TinyPINN_v3(),
        'PhysicsGuided_DeepTinyPINN_v1': PhysicsGuided_DeepTinyPINN_v1(),
        'PhysicsGuided_DeepTinyPINN_v2': PhysicsGuided_DeepTinyPINN_v2(),
        'PhysicsGuided_LargeCorrection_v1': PhysicsGuided_LargeCorrection_v1(),
        'PhysicsGuided_SmallCorrection_v1': PhysicsGuided_SmallCorrection_v1(),
        'PhysicsGuided_ImprovedGuess_v1': PhysicsGuided_ImprovedGuess_v1(),
        'PhysicsGuided_ResidualCorrection_v1': PhysicsGuided_ResidualCorrection_v1_Fixed(),
        'PhysicsGuided_AdaptiveCorrection_v1': PhysicsGuided_AdaptiveCorrection_v1(),
    }
    
    results = {}
    
    # Train and evaluate each model
    for model_name, model in models_to_test.items():
        print(f"\n{'='*60}")
        print(f"Testing {model_name}")
        print(f"{'='*60}")
        
        # Model info
        n_params = count_parameters(model)
        flops = estimate_flops(model)
        print(f"Parameters: {n_params:,}")
        print(f"Estimated FLOPs: {flops:,}")
        
        # Train model
        trained_model, train_losses, val_losses = train_model(
            model, train_loader, val_loader, eos, device,
            epochs=args.epochs, lr=args.learning_rate, 
            physics_weight=args.physics_weight, use_log_cosh=args.use_log_cosh
        )
        
        # Evaluate model
        print("Evaluating on test set...")
        test_results = evaluate_model(trained_model, C_test, Z_test, eos, device)
        
        # Store results
        results[model_name] = {
            'n_parameters': n_params,
            'estimated_flops': flops,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'test_results': test_results
        }
        
        # Print summary
        print(f"\nResults for {model_name}:")
        print(f"  Parameters: {n_params:,}")
        print(f"  Mean Abs Error: {test_results['abs_error_mean']:.2e}")
        print(f"  Mean Rel Error: {test_results['rel_error_mean']:.2e}")
        print(f"  Physics Error: {test_results['physics_error']:.2e}")
        print(f"  Correlation: {test_results['correlation']:.6f}")
        print(f"  Inference Time: {test_results['inference_time']:.4f}s")
        
        # Save model
        model_path = f"{args.output_dir}/model_{model_name}.pth"
        torch.save({
            'model_state_dict': trained_model.state_dict(),
            'model_class': model.__class__.__name__,
            'n_parameters': n_params,
            'results': test_results
        }, model_path)
        print(f"Model saved to {model_path}")
    
    return results

def create_plots(results, output_dir):
    """Create comparison plots"""
    
    model_names = list(results.keys())
    n_params = [results[name]['n_parameters'] for name in model_names]
    abs_errors = [results[name]['test_results']['abs_error_mean'] for name in model_names]
    rel_errors = [results[name]['test_results']['rel_error_mean'] for name in model_names]
    physics_errors = [results[name]['test_results']['physics_error'] for name in model_names]
    correlations = [results[name]['test_results']['correlation'] for name in model_names]
    inference_times = [results[name]['test_results']['inference_time'] for name in model_names]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Plot 1: Parameters vs Absolute Error
    axes[0, 0].loglog(n_params, abs_errors, 'bo-', linewidth=2, markersize=8)
    axes[0, 0].set_xlabel('Number of Parameters')
    axes[0, 0].set_ylabel('Mean Absolute Error')
    axes[0, 0].set_title('Model Size vs Absolute Error')
    axes[0, 0].grid(True, alpha=0.3)
    for i, name in enumerate(model_names):
        short_name = name.split('_')[-1]  # Get version number
        axes[0, 0].annotate(short_name, (n_params[i], abs_errors[i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # Plot 2: Parameters vs Relative Error
    axes[0, 1].loglog(n_params, rel_errors, 'ro-', linewidth=2, markersize=8)
    axes[0, 1].set_xlabel('Number of Parameters')
    axes[0, 1].set_ylabel('Mean Relative Error')
    axes[0, 1].set_title('Model Size vs Relative Error')
    axes[0, 1].grid(True, alpha=0.3)
    for i, name in enumerate(model_names):
        short_name = name.split('_')[-1]
        axes[0, 1].annotate(short_name, (n_params[i], rel_errors[i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # Plot 3: Parameters vs Physics Error
    axes[0, 2].loglog(n_params, physics_errors, 'go-', linewidth=2, markersize=8)
    axes[0, 2].set_xlabel('Number of Parameters')
    axes[0, 2].set_ylabel('Physics Error')
    axes[0, 2].set_title('Model Size vs Physics Consistency')
    axes[0, 2].grid(True, alpha=0.3)
    for i, name in enumerate(model_names):
        short_name = name.split('_')[-1]
        axes[0, 2].annotate(short_name, (n_params[i], physics_errors[i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # Plot 4: Parameters vs Correlation
    axes[1, 0].semilogx(n_params, correlations, 'mo-', linewidth=2, markersize=8)
    axes[1, 0].set_xlabel('Number of Parameters')
    axes[1, 0].set_ylabel('Correlation')
    axes[1, 0].set_title('Model Size vs Prediction Correlation')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim([0.9, 1.0])
    for i, name in enumerate(model_names):
        short_name = name.split('_')[-1]
        axes[1, 0].annotate(short_name, (n_params[i], correlations[i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # Plot 5: Parameters vs Inference Time
    axes[1, 1].loglog(n_params, inference_times, 'co-', linewidth=2, markersize=8)
    axes[1, 1].set_xlabel('Number of Parameters')
    axes[1, 1].set_ylabel('Inference Time (s)')
    axes[1, 1].set_title('Model Size vs Inference Speed')
    axes[1, 1].grid(True, alpha=0.3)
    for i, name in enumerate(model_names):
        short_name = name.split('_')[-1]
        axes[1, 1].annotate(short_name, (n_params[i], inference_times[i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # Plot 6: Efficiency scatter (Error vs Speed)
    axes[1, 2].loglog(inference_times, abs_errors, 'ko', markersize=10)
    axes[1, 2].set_xlabel('Inference Time (s)')
    axes[1, 2].set_ylabel('Mean Absolute Error')
    axes[1, 2].set_title('Speed vs Accuracy Trade-off')
    axes[1, 2].grid(True, alpha=0.3)
    for i, name in enumerate(model_names):
        short_name = name.split('_')[-1]
        axes[1, 2].annotate(short_name, (inference_times[i], abs_errors[i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/physics_guided_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create individual training curves
    n_models = len(results)
    cols = 4
    rows = (n_models + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(20, 5*rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    for i, (name, result) in enumerate(results.items()):
        ax = axes[i]
        epochs = range(len(result['train_losses']))
        ax.semilogy(epochs, result['train_losses'], label='Train', linewidth=2)
        ax.semilogy(epochs, result['val_losses'], label='Validation', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title(f'{name}\n({result["n_parameters"]} params)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(len(results), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/physics_guided_training_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create performance comparison bar chart
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Absolute Error
    axes[0, 0].bar(range(len(model_names)), abs_errors, color='skyblue')
    axes[0, 0].set_ylabel('Mean Absolute Error')
    axes[0, 0].set_title('Absolute Error Comparison')
    axes[0, 0].set_xticks(range(len(model_names)))
    axes[0, 0].set_xticklabels([name.split('_')[-1] for name in model_names], rotation=45)
    axes[0, 0].set_yscale('log')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Relative Error
    axes[0, 1].bar(range(len(model_names)), rel_errors, color='lightcoral')
    axes[0, 1].set_ylabel('Mean Relative Error')
    axes[0, 1].set_title('Relative Error Comparison')
    axes[0, 1].set_xticks(range(len(model_names)))
    axes[0, 1].set_xticklabels([name.split('_')[-1] for name in model_names], rotation=45)
    axes[0, 1].set_yscale('log')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Physics Error
    axes[1, 0].bar(range(len(model_names)), physics_errors, color='lightgreen')
    axes[1, 0].set_ylabel('Physics Error')
    axes[1, 0].set_title('Physics Consistency Comparison')
    axes[1, 0].set_xticks(range(len(model_names)))
    axes[1, 0].set_xticklabels([name.split('_')[-1] for name in model_names], rotation=45)
    axes[1, 0].set_yscale('log')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Correlation
    axes[1, 1].bar(range(len(model_names)), correlations, color='gold')
    axes[1, 1].set_ylabel('Correlation')
    axes[1, 1].set_title('Prediction Correlation Comparison')
    axes[1, 1].set_xticks(range(len(model_names)))
    axes[1, 1].set_xticklabels([name.split('_')[-1] for name in model_names], rotation=45)
    axes[1, 1].set_ylim([0.9, 1.0])
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/physics_guided_performance_bars.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Physics-Guided PINN Comparison Script')
    parser.add_argument('--n_train', type=int, default=400, help='Number of training samples')
    parser.add_argument('--n_val', type=int, default=200, help='Number of validation samples')
    parser.add_argument('--n_test', type=int, default=200, help='Number of test samples')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--epochs', type=int, default=300, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--physics_weight', type=float, default=1.0, help='Physics loss weight')
    parser.add_argument('--use_mse', action='store_true', help='Use MSE loss instead of Log-Cosh')
    parser.add_argument('--output_dir', type=str, default='./physics_guided_pinn_comparison', help='Output directory')
    
    args = parser.parse_args()
    
    # Set use_log_cosh based on the flag
    args.use_log_cosh = not args.use_mse
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("Physics-Guided PINN Comparison Study")
    print("=" * 60)
    print(f"Training samples: {args.n_train}")
    print(f"Validation samples: {args.n_val}")
    print(f"Test samples: {args.n_test}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Physics weight: {args.physics_weight}")
    print(f"Loss function: {'Log-Cosh' if args.use_log_cosh else 'MSE'}")
    print(f"Output directory: {args.output_dir}")
    
    # Run comparison
    results = run_comparison(args)
    
    # Save results
    results_file = f"{args.output_dir}/physics_guided_results.json"
    
    # Convert results to JSON-serializable format
    json_results = {}
    for model_name, result in results.items():
        json_results[model_name] = {
            'n_parameters': result['n_parameters'],
            'estimated_flops': result['estimated_flops'],
            'final_train_loss': result['train_losses'][-1],
            'final_val_loss': result['val_losses'][-1],
            'test_results': {k: v for k, v in result['test_results'].items() 
                           if k not in ['comprehensive_errors', 'Z_pred', 'Z_true']}
        }
    
    with open(results_file, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"\nResults saved to {results_file}")
    
    # Create plots
    print("Creating comparison plots...")
    create_plots(results, args.output_dir)
    
    print("\nPhysics-Guided PINN Comparison complete!")
    print(f"Results saved in: {args.output_dir}")
    
    # Print detailed summary table
    print("\n" + "="*100)
    print("DETAILED PHYSICS-GUIDED PINN COMPARISON")
    print("="*100)
    print(f"{'Model':<35} {'Params':<8} {'Abs Err':<12} {'Rel Err':<12} {'Physics Err':<12} {'Correlation':<12} {'Time (s)':<8}")
    print("-"*100)
    
    # Sort by absolute error for better comparison
    sorted_results = sorted(results.items(), key=lambda x: x[1]['test_results']['abs_error_mean'])
    
    for model_name, result in sorted_results:
        test_res = result['test_results']
        print(f"{model_name:<35} {result['n_parameters']:<8} {test_res['abs_error_mean']:<12.2e} "
              f"{test_res['rel_error_mean']:<12.2e} {test_res['physics_error']:<12.2e} "
              f"{test_res['correlation']:<12.6f} {test_res['inference_time']:<8.4f}")
    
    print("="*100)
    
    # Find best performing models
    best_abs_error = min(results.items(), key=lambda x: x[1]['test_results']['abs_error_mean'])
    best_physics_error = min(results.items(), key=lambda x: x[1]['test_results']['physics_error'])
    best_correlation = max(results.items(), key=lambda x: x[1]['test_results']['correlation'])
    fastest = min(results.items(), key=lambda x: x[1]['test_results']['inference_time'])
    
    print("\nBEST PERFORMERS:")
    print(f"Best Absolute Error: {best_abs_error[0]} ({best_abs_error[1]['test_results']['abs_error_mean']:.2e})")
    print(f"Best Physics Error: {best_physics_error[0]} ({best_physics_error[1]['test_results']['physics_error']:.2e})")
    print(f"Best Correlation: {best_correlation[0]} ({best_correlation[1]['test_results']['correlation']:.6f})")
    print(f"Fastest Inference: {fastest[0]} ({fastest[1]['test_results']['inference_time']:.4f}s)")

if __name__ == "__main__":
    main()