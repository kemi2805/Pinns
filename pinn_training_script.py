#!/usr/bin/env python3
"""
Complete PINN Training Script for Conservative-to-Primitive Variable Conversion
Combines all functionality from the original Jupyter notebook into a single script.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
import math

# Physics modules (assuming these are available)
from metric import metric
from hybrid_eos import hybrid_eos
from C2P_Solver import *

# =============================================================================
# CONFIGURATION AND SETUP
# =============================================================================

# Set up device and precision
use_fp64 = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dtype = torch.float64 if use_fp64 else torch.float32
torch.set_default_dtype(dtype)

# Set number of CPU threads - you can adjust this based on your node's CPU count
# For AMD MI300A nodes, they typically have many CPU cores
import os
num_cpu_threads = int(os.environ.get('OMP_NUM_THREADS', torch.get_num_threads()))
torch.set_num_threads(min(num_cpu_threads, 32))  # Cap at 32 to avoid overhead

print(f"Using device: {device}")
print(f"Using precision: {dtype}")
print(f"Number of GPUs available: {torch.cuda.device_count()}")
print(f"Current GPU: {torch.cuda.current_device()}")
print(f"GPU name: {torch.cuda.get_device_name()}")
print(f"Number of CPU threads: {torch.get_num_threads()}")

# =============================================================================
# PHYSICS SETUP
# =============================================================================

# Minkowski metric
eta = metric(
    torch.eye(3, device=device),
    torch.zeros(3, device=device),
    torch.ones(1, device=device)
)

# Gamma = 2 EOS with ideal gas thermal contribution
eos = hybrid_eos(100, 2, 1.8)

# =============================================================================
# NEURAL NETWORK ARCHITECTURES
# =============================================================================

def init_weights(m):
    """Xavier initialization for better training stability"""
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

class ResidualBlock(nn.Module):
    """Residual block with skip connection"""
    def __init__(self, in_dim, out_dim, activation, dtype=torch.float64):
        super().__init__()
        self.linear1 = nn.Linear(in_dim, out_dim, dtype=dtype)
        self.linear2 = nn.Linear(out_dim, out_dim, dtype=dtype)
        self.activation = activation
        
        # Skip connection (projection if dimensions don't match)
        if in_dim != out_dim:
            self.skip = nn.Linear(in_dim, out_dim, dtype=dtype)
        else:
            self.skip = nn.Identity()
    
    def forward(self, x):
        identity = self.skip(x)
        out = self.activation(self.linear1(x))
        out = self.linear2(out)
        return self.activation(out + identity)

class C2P_PINN(nn.Module):
    """
    Physics-Informed Neural Network for conservative-to-primitive transformation
    """
    def __init__(self, dtype=torch.float64, physics_weight=1.0):
        super().__init__()
        self.input_dim = 3
        self.hidden_dims = [384, 384, 192]
        self.activation = nn.SiLU()
        self.physics_weight = physics_weight
        
        # Define residual blocks
        layers = []
        in_dim = self.input_dim
        for out_dim in self.hidden_dims:
            layers.append(ResidualBlock(in_dim, out_dim, self.activation, dtype))
            in_dim = out_dim
        self.blocks = nn.Sequential(*layers)
        
        # Final layer with skip connection from original input
        self.final_fc = nn.Linear(self.hidden_dims[-1] + self.input_dim, 1, dtype=dtype)
        
        # Ensure positive output (physical constraint)
        self.output_activation = nn.Sigmoid()
        
        # Apply weight initialization
        self.apply(init_weights)
    
    def forward(self, x):
        """Forward pass with residual connections"""
        identity = x  # Keep original input for skip connection
        x = self.blocks(x)
        x = torch.cat([x, identity], dim=1)  # Concatenate with original input
        x = self.final_fc(x)
        return self.output_activation(x)
    
    def physics_loss(self, C, Z_pred, eos):
        """
        Physics-informed loss based on relativistic hydro constraints
        """
        D = C[:, 0:1]  # Conserved density
        q = C[:, 1:2]  # τ/D
        r = C[:, 2:3]  # S/D
        
        try:
            htilde = h__z(Z_pred, C, eos)
            physics_residual = Z_pred - r / htilde
            return torch.mean(physics_residual**2)
        except:
            return torch.tensor(0.0, device=C.device, dtype=C.dtype)

class PhysicsGuided_TinyPINNvKen(nn.Module):
    """
    Physics-guided approach: Start with analytical approximation,
    then learn small corrections
    """
    def __init__(self, dtype=torch.float64, correction_scale=0.2):
        super().__init__()
        
        # Tiny correction network
        self.correction_net = nn.Sequential(
            nn.Linear(3, 64, dtype=dtype),
            nn.Tanh(),
            nn.Linear(64, 1, dtype=dtype),
        )
        
        # Small correction scale
        self.correction_scale = correction_scale
        
        # Initialize
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=0.1)  # Small corrections
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    def analytical_guess(self, C):
        """
        Improved physics-based initial guess for Z (Lorentz factor)
        Based on GRMHD conservative-to-primitive relationships
        """
        D = C[:, 0:1]  # Conserved density
        q = C[:, 1:2]  # τ/D (internal energy density)
        r = C[:, 2:3]  # S/D (momentum density)
        
        # Method 1: Newton-Raphson inspired guess
        # For cold matter: Z ≈ sqrt(1 + r²)
        # For hot matter: include thermal pressure effects
        
        # Basic kinematic guess
        z_kinematic = torch.sqrt(1.0 + r**2)
        
        # Thermal correction based on internal energy
        # Higher q suggests more thermal energy → higher Z
        thermal_factor = 1.0 + 0.5 * q  # Empirical correction
        
        # Relativistic correction for high velocities
        # When r is large, additional relativistic effects
        relativistic_correction = 1.0 + 0.1 * r**2 / (1.0 + r**2)

        # At high density, the rest mass energy becomes significant
        # The enthalpy includes rest mass: h = (ρ + u + p)/ρ
        density_correction = 1.0 + 0.1 * torch.log(1.0 + D)
        
        # Combined guess
        z_guess = z_kinematic * thermal_factor * relativistic_correction * density_correction
        
        # Ensure physical bounds: Z ≥ 1
        z_guess = torch.clamp(z_guess, min=1.0)
        
        return z_guess
    
    def forward(self, x):
        # Start with physics-based guess
        z_baseline = self.analytical_guess(x)
        
        # Small learned correction
        correction = self.correction_net(x)
        
        return z_baseline + self.correction_scale * correction
    
    def physics_loss(self, C, Z_pred, eos):
        """
        Physics-informed loss: Z = S / h̃, where h̃ is derived from Z_pred
        """
        D = C[:, 0:1]  # Conserved density
        q = C[:, 1:2]  # τ/D
        r = C[:, 2:3]  # S/D
        
        try:
            # Compute specific enthalpy from prediction
            htilde = h__z(Z_pred, C, eos)
            # Physics residual: should be zero if physically correct
            residual = Z_pred - (r / htilde)
            return torch.mean(residual**2)
        except Exception as e:
            print(f"[physics_loss] Failed to compute htilde: {e}")
            # Return dummy loss with gradient support
            return torch.sum(Z_pred * 0.0)

# =============================================================================
# DATA GENERATION
# =============================================================================

def setup_initial_state_meshgrid_cold(metric, eos, N, device,
                                      lrhomin=-12, lrhomax=-2.8,
                                      Wmin=1.0, Wmax=2.0, bias_strength=2):
    """
    Generate conservative variables on a cold matter meshgrid
    """
    # Generate biased linspace: more points near edges using beta-like transformation
    def biased_linspace(a, b, n, bias_strength, endpoint=True, side='both'):
        t = torch.linspace(0, 1, n, device=device)
        if side == 'lower':
            t = t**bias_strength
        elif side == 'upper':
            t = 1 - (1 - t)**bias_strength
        elif side == 'both':
            t = 0.5 * ((2*t)**bias_strength * (t < 0.5) +
                      (2*(1 - t))**bias_strength * (t >= 0.5))
            t = torch.where(t < 0.5, t, 1 - t)
            t = 2 * t  # Rescale to [0,1]
        else:  # none
            t = t
        return a + (b - a) * t
    
    # Create meshgrid
    W = biased_linspace(Wmin, Wmax, N, bias_strength, side='both')
    rho = 10**(torch.linspace(lrhomin, lrhomax, N, device=device))
    rho_mesh, W_mesh = torch.meshgrid(rho, W, indexing='ij')
    
    # Flatten
    rho = rho_mesh.flatten()
    W = W_mesh.flatten()
    
    # Cold matter (T = 0)
    T = torch.zeros_like(rho, device=device)
    
    # EOS call
    press, eps = eos.press_eps__temp_rho(T, rho)
    
    # Compute primitive z
    Z = torch.sqrt(1 - 1/W**2) * W
    
    # Compute conservative variables
    sqrtg = metric.sqrtg
    u0 = W / sqrtg
    dens = sqrtg * W * rho
    rho0_h = rho * (1 + eps) + press
    g4uptt = -1 / metric.alp**2
    Tuptt = rho0_h * u0**2 + press * g4uptt
    tau = metric.alp**2 * sqrtg * Tuptt - dens
    S = torch.sqrt((W**2 - 1)) * rho0_h * W
    
    # Assemble conservative variables: [D, tau/D, S/D]
    C = torch.cat((
        dens.view(-1, 1) / metric.sqrtg,
        tau.view(-1, 1) / dens.view(-1, 1),
        S.view(-1, 1) / dens.view(-1, 1)
    ), dim=1)
    
    return C, Z.view(-1, 1)

# =============================================================================
# DATASET CLASS
# =============================================================================

class C2P_Dataset(TensorDataset):
    """Dataset class with normalization capabilities"""
    def __init__(self, C, Z, normalize_data=True, margin=0.05):
        self.C = C.to(dtype)
        self.Z = Z.to(dtype)
        self.margin = margin
        
        if normalize_data:
            self.normalize()
    
    def normalize(self):
        """Min-max normalization with safety checks"""
        self.C_max = torch.max(self.C, dim=0, keepdim=True)[0]
        self.C_min = torch.min(self.C, dim=0, keepdim=True)[0]
        
        # Safety check for C normalization
        C_range = self.C_max - self.C_min
        C_range = torch.where(C_range == 0, torch.ones_like(C_range), C_range)
        self.C = (self.C - self.C_min) / C_range
        
        self.Z_max = torch.max(self.Z, dim=0, keepdim=True)[0]
        self.Z_min = torch.min(self.Z, dim=0, keepdim=True)[0]
        Z_range = self.Z_max - self.Z_min
        
        # Safety check for Z normalization - if range is too small, add artificial range
        if Z_range.item() < 1e-10:
            print(f"Warning: Z range is very small ({Z_range.item():.2e}), adding artificial range")
            Z_center = (self.Z_max + self.Z_min) / 2
            artificial_range = torch.maximum(torch.abs(Z_center) * 0.1, torch.tensor(0.1).to(self.Z.device))
            self.Z_min = Z_center - artificial_range / 2
            self.Z_max = Z_center + artificial_range / 2
            Z_range = self.Z_max - self.Z_min
        
        # Apply margin
        self.Z_min -= self.margin * Z_range
        self.Z_max += self.margin * Z_range
        
        # Final normalization
        final_range = self.Z_max - self.Z_min
        self.Z = (self.Z - self.Z_min) / final_range
    
    def denormalize(self):
        """Reverse normalization"""
        self.C = self.C * (self.C_max - self.C_min) + self.C_min
        self.Z = self.Z * (self.Z_max - self.Z_min) + self.Z_min
    
    def __len__(self):
        return self.C.shape[0]
    
    def __getitem__(self, idx):
        return self.C[idx, :], self.Z[idx, :]

# =============================================================================
# LOSS FUNCTIONS
# =============================================================================

def log_cosh_loss(y_true, y_pred):
    """Robust loss function that's less sensitive to outliers"""
    return torch.mean(torch.log(torch.cosh(y_pred - y_true)))

def max_loss(y_true, y_pred):
    """Focus on harder examples"""
    max_val = torch.max(torch.abs(y_true - y_pred))
    return max_val

def huber_loss(y_true, y_pred, delta=1.0):
    """Huber loss for robust regression"""
    residual = torch.abs(y_true - y_pred)
    condition = residual <= delta
    squared_loss = 0.5 * (y_true - y_pred)**2
    linear_loss = delta * residual - 0.5 * delta**2
    return torch.mean(torch.where(condition, squared_loss, linear_loss))

def compute_loss(model, C, C_min, C_max, Z, Z_min, Z_max, eos=None,
                 physics_weight=1.0, loss_type='log_cosh'):
    """
    Compute combined data loss and physics loss
    """
    # Forward pass
    Z_pred = model(C)
    
    # Data fitting loss
    if loss_type == 'log_cosh':
        data_loss = log_cosh_loss(Z, Z_pred)
    elif loss_type == 'huber':
        data_loss = huber_loss(Z, Z_pred)
    elif loss_type == 'mse':
        data_loss = nn.MSELoss()(Z_pred, Z)
    elif loss_type == 'max':
        data_loss = max_loss(Z, Z_pred)
    else:
        data_loss = log_cosh_loss(Z, Z_pred)
    
    # Physics loss (if model supports it and parameters provided)
    physics_loss = 0.0
    if hasattr(model, 'physics_loss') and eos is not None:
        # Convert normalized inputs back to physical units for physics loss
        C_real = C * (C_max - C_min) + C_min
        Z_real = Z_pred * (Z_max - Z_min) + Z_min
        physics_loss = model.physics_loss(C_real, Z_real, eos)
    
    # Combined loss
    total_loss = data_loss + physics_weight * physics_loss
    
    return total_loss, data_loss, physics_loss

def compute_validation_loss(model, C, C_min, C_max, Z, Z_min, Z_max, eos):
    """
    Compute validation metrics including physical consistency checks
    """
    with torch.no_grad():
        Z_pred = model(C)
        # Convert back to physical units
        Z_pred_real = Z_pred * (Z_max - Z_min) + Z_min
        Z_real = Z * (Z_max - Z_min) + Z_min
        C_real = C * (C_max - C_min) + C_min
        
        # Physics consistency check
        err = sanity_check(Z_pred_real, C_real, eta, eos)
        
    return err

# =============================================================================
# TRAINING FUNCTION
# =============================================================================

def train_pinn_model(model, optimizer, scheduler, training_loader, validation_loader,
                     C_min, C_max, Z_min, Z_max, num_epochs, eos, 
                     physics_weight=0.1, loss_type='log_cosh',
                     device='cpu', patience=50):
    """
    Train PINN model with comprehensive logging and early stopping
    """
    
    # Move model to device
    model = model.to(device)
    
    # Training history
    training_losses = []
    validation_losses = []
    physics_losses = []
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    print(f"Training PINN for {num_epochs} epochs...")
    print(f"Physics weight: {physics_weight}, Loss type: {loss_type}")
    print("-" * 60)
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        epoch_loss = 0
        epoch_physics_loss = 0
        
        for batch_idx, (C_data, Z_data) in enumerate(training_loader):
            C_data, Z_data = C_data.to(device), Z_data.to(device)
            
            optimizer.zero_grad()
            
            # Compute loss
            total_loss, data_loss, phys_loss = compute_loss(
                model, C_data, C_min, C_max, Z_data, Z_min, Z_max,
                eos, physics_weight, loss_type
            )
            
            # Backward pass
            total_loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Accumulate losses
            epoch_loss += total_loss.item()
            if isinstance(phys_loss, torch.Tensor):
                epoch_physics_loss += phys_loss.item()
        
        # Average losses
        epoch_loss /= len(training_loader)
        epoch_physics_loss /= len(training_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for C_test, Z_test in validation_loader:
                C_test, Z_test = C_test.to(device), Z_test.to(device)
                
                val_err = compute_validation_loss(
                    model, C_test, C_min, C_max, Z_test, Z_min, Z_max, eos
                )
                
                val_loss += val_err.item()
        
        val_loss /= len(validation_loader)
        
        # Store history
        training_losses.append(epoch_loss)
        validation_losses.append(val_loss)
        physics_losses.append(epoch_physics_loss)
        
        # Learning rate scheduling
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)
        else:
            scheduler.step()
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        # Print progress
        if epoch % 10 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch:4d} | "
                  f"Train Loss: {epoch_loss:.3e} | "
                  f"Val Loss: {val_loss:.3e} | "
                  f"Physics: {epoch_physics_loss:.3e} | "
                  f"LR: {current_lr:.2e}")
        
        # Early stopping check
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered at epoch {epoch}")
            break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Loaded best model with validation loss: {best_val_loss:.3e}")
    
    return {
        'training_losses': training_losses,
        'validation_losses': validation_losses,
        'physics_losses': physics_losses,
        'best_val_loss': best_val_loss
    }

# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def plot_training_history(history):
    """Plot training curves for analysis"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Training and validation loss
    axes[0, 0].semilogy(history['training_losses'], label='Training Loss')
    axes[0, 0].semilogy(history['validation_losses'], label='Validation Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Physics loss
    axes[0, 1].semilogy(history['physics_losses'])
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Physics Loss')
    axes[0, 1].set_title('Physics-Informed Loss')
    axes[0, 1].grid(True)
    
    # Loss components comparison
    axes[1, 0].semilogy(history['training_losses'], label='Total Loss')
    axes[1, 0].semilogy(history['physics_losses'], label='Physics Loss')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].set_title('Loss Components')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Convergence analysis
    min_loss = min(history['validation_losses'])
    axes[1, 1].semilogy([min_loss] * len(history['validation_losses']),
                       '--', label=f'Best Val Loss: {min_loss:.2e}')
    axes[1, 1].semilogy(history['validation_losses'], label='Validation Loss')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].set_title('Convergence Analysis')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.show()

def plot_error_analysis_with_zeta_absolute(solver, test_C, test_Z, N=50, save_fig=False, fig_path=None):
    """
    Plot comprehensive absolute error analysis for C2P solver including zeta components
    """
    # Predict on test data
    rho_pred, eps_pred, press_pred, W_pred = solver.invert(test_C)
    
    # Compute true values
    Z_true = test_Z.view(-1, 1)
    print(f"Z_true shape: {Z_true.shape}")
    print(f"Z_true sample: {Z_true[:5].flatten()}")

    print(f"Test C range: [{test_C.min():.6f}, {test_C.max():.6f}]")
    print(f"Test Z range: [{test_Z.min():.6f}, {test_Z.max():.6f}]")
    print(f"Training C_min/max: {solver.C_min.flatten()}, {solver.C_max.flatten()}")
    print(f"Training Z_min/max: {solver.Z_min.item():.6f}, {solver.Z_max.item():.6f}")
    
    rho_true = rho__z(Z_true, test_C)
    W_true = W__z(Z_true)
    eps_true = eps__z(Z_true, test_C)
    press_true = solver.eos.press__eps_rho(eps_true, rho_true)
    
    # Get predicted zeta from the model
    with torch.no_grad():
        # Normalize input conservative variables
        C_norm = (test_C - solver.C_min) / (solver.C_max - solver.C_min)
        Z_pred_norm = solver.model(C_norm)
        # Denormalize predicted zeta
        Z_pred = Z_pred_norm * (solver.Z_max - solver.Z_min) + solver.Z_min
    
    print(f"Z_pred shape: {Z_pred.shape}")
    print(f"Z_pred sample: {Z_pred[:5].flatten()}")
    
    # Calculate ABSOLUTE errors for physical variables
    rho_error = torch.abs(rho_pred - rho_true)
    eps_error = torch.abs(eps_pred - eps_true)
    press_error = torch.abs(press_pred - press_true)
    W_error = torch.abs(W_pred - W_true)
    
    # Calculate absolute zeta error
    Z_error = torch.abs(Z_pred - Z_true)
    
    # Convert to numpy for plotting
    errors = {
        'Density (ρ)': rho_error.detach().cpu().numpy().flatten(),
        'Specific Energy (ε)': eps_error.detach().cpu().numpy().flatten(),
        'Pressure (P)': press_error.detach().cpu().numpy().flatten(),
        'Lorentz Factor (W)': W_error.detach().cpu().numpy().flatten(),
        'Zeta (ζ)': Z_error.detach().cpu().numpy().flatten()
    }
    
    # Create error distribution plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    colors = ['skyblue', 'lightcoral', 'lightgreen', 'plum', 'gold']
    
    for i, (name, error) in enumerate(errors.items()):
        # Remove any infinite or NaN values
        error_clean = error[np.isfinite(error)]
        
        # For absolute errors, use log scale when errors span multiple orders of magnitude
        min_error = np.min(error_clean[error_clean > 0]) if np.any(error_clean > 0) else 1e-15
        max_error = np.max(error_clean)
        
        # Use log scale if errors span more than 2 orders of magnitude
        use_log = (max_error / min_error) > 100 if min_error > 0 else False
        
        if use_log:
            log_error = np.log10(np.maximum(error_clean, 1e-15))
            hist_data = log_error
            xlabel = 'log₁₀(Absolute Error)'
        else:
            hist_data = error_clean
            xlabel = 'Absolute Error'
        
        # Create histogram
        n, bins, patches = axes[i].hist(hist_data, bins=N, alpha=0.7, 
                                       color=colors[i % len(colors)], 
                                       edgecolor='black', linewidth=0.5)
        
        axes[i].set_xlabel(xlabel, fontsize=12)
        axes[i].set_ylabel('Frequency', fontsize=12)
        axes[i].set_title(f'{name} Absolute Error Distribution', fontsize=14, fontweight='bold')
        axes[i].grid(True, alpha=0.3)
        
        # Add statistics
        mean_error = np.mean(error_clean)
        max_error_val = np.max(error_clean)
        median_error = np.median(error_clean)
        
        # Add vertical lines for statistics
        if use_log:
            axes[i].axvline(np.log10(mean_error), color='red', linestyle='--', linewidth=2,
                           label=f'Mean: {mean_error:.2e}')
            axes[i].axvline(np.log10(median_error), color='orange', linestyle='--', linewidth=2,
                           label=f'Median: {median_error:.2e}')
        else:
            axes[i].axvline(mean_error, color='red', linestyle='--', linewidth=2,
                           label=f'Mean: {mean_error:.2e}')
            axes[i].axvline(median_error, color='orange', linestyle='--', linewidth=2,
                           label=f'Median: {median_error:.2e}')
        
        axes[i].legend(fontsize=9)
    
    # Hide unused subplot
    if len(errors) < len(axes):
        axes[-1].set_visible(False)
    
    plt.tight_layout()
    
    if save_fig:
        if fig_path is None:
            fig_path = 'absolute_error_analysis_with_zeta.png'
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {fig_path}")
    
    plt.show()
    
    # Print comprehensive summary statistics
    print("\n" + "="*70)
    print("COMPREHENSIVE ABSOLUTE ERROR ANALYSIS SUMMARY (INCLUDING ZETA)")
    print("="*70)
    
    for name, error in errors.items():
        error_clean = error[np.isfinite(error)]
        print(f"\n{name:20s}:")
        print(f"  Mean abs error:     {np.mean(error_clean):.2e}")
        print(f"  Median abs error:   {np.median(error_clean):.2e}")
        print(f"  Max abs error:      {np.max(error_clean):.2e}")
        print(f"  Samples:            {len(error_clean)}")
    
    print("\n" + "="*70)
    
    return errors, Z_pred, Z_true

def plot_zeta_scatter_analysis(Z_true, Z_pred, save_fig=False, fig_path=None):
    """
    Create scatter plots comparing true vs predicted zeta values
    """
    Z_true_np = Z_true.detach().cpu().numpy().flatten()
    Z_pred_np = Z_pred.detach().cpu().numpy().flatten()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Scatter plot: True vs Predicted
    axes[0].scatter(Z_true_np, Z_pred_np, alpha=0.6, s=20, c='blue')
    axes[0].plot([Z_true_np.min(), Z_true_np.max()], 
                 [Z_true_np.min(), Z_true_np.max()], 'r--', lw=2, label='Perfect prediction')
    axes[0].set_xlabel('True Zeta (ζ)', fontsize=12)
    axes[0].set_ylabel('Predicted Zeta (ζ)', fontsize=12)
    axes[0].set_title('True vs Predicted Zeta Values', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # Add correlation coefficient
    correlation = np.corrcoef(Z_true_np, Z_pred_np)[0,1]
    axes[0].text(0.05, 0.95, f'Correlation: {correlation:.6f}', 
                transform=axes[0].transAxes, fontsize=12,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Residual plot
    residuals = Z_pred_np - Z_true_np
    axes[1].scatter(Z_true_np, residuals, alpha=0.6, s=20, c='green')
    axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[1].set_xlabel('True Zeta (ζ)', fontsize=12)
    axes[1].set_ylabel('Residuals (Pred - True)', fontsize=12)
    axes[1].set_title('Zeta Prediction Residuals', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    # Add residual statistics
    residual_std = np.std(residuals)
    residual_mean = np.mean(residuals)
    axes[1].text(0.05, 0.95, f'Mean: {residual_mean:.2e}\nStd: {residual_std:.2e}', 
                transform=axes[1].transAxes, fontsize=12,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if save_fig:
        if fig_path is None:
            fig_path = 'zeta_scatter_analysis.png'
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"Zeta scatter analysis saved to {fig_path}")
    
    plt.show()

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function"""
    print("Starting PINN Training for Conservative-to-Primitive Variable Conversion")
    print("=" * 80)
    
    # Generate training data
    print("Generating training data...")
    C, Z = setup_initial_state_meshgrid_cold(
        eta, eos, 500, device,
        lrhomin=-8,
        lrhomax=-2.7,
        Wmin=1.0,
        Wmax=1.3,  # Increased from 1.3 to 2.0 for more Z variation
        bias_strength=1
    )
    
    # Initial sanity check
    err = sanity_check(Z, C, eta, eos)
    print(f"Initial sanity check error: {err:.2e}")
    print(f"Z statistics: min={Z.min():.6f}, max={Z.max():.6f}, mean={Z.mean():.6f}, std={Z.std():.6f}")
    
    # Create dataset
    dataset = C2P_Dataset(C, Z, margin=0.00)
    
    # Split into train/validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create data loaders - larger batch size for multi-GPU
    batch_size = 512 if torch.cuda.device_count() > 1 else 25
    batch_size = 1028 if torch.cuda.device_count() > 3 else batch_size
    print("MY batch size is = ", batch_size)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Create model
    model = PhysicsGuided_TinyPINNvKen(dtype=dtype, correction_scale=0.2)
    
    # Enable multi-GPU if available
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for training!")
        model = torch.nn.DataParallel(model)
    
    model = model.to(device)
    
    # Setup optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=5e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=6, verbose=True
    )
    
    # Print dataset information
    print(f"C range after normalization: [{dataset.C.min():.6f}, {dataset.C.max():.6f}]")
    print(f"Z range after normalization: [{dataset.Z.min():.6f}, {dataset.Z.max():.6f}]")
    print(f"Original C_min: {dataset.C_min}")
    print(f"Original C_max: {dataset.C_max}")
    print(f"Original Z_min: {dataset.Z_min}")
    print(f"Original Z_max: {dataset.Z_max}")
    
    # Train the model
    print("\nStarting training...")
    history = train_pinn_model(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        training_loader=train_loader,
        validation_loader=val_loader,
        C_min=dataset.C_min,
        C_max=dataset.C_max,
        Z_min=dataset.Z_min,
        Z_max=dataset.Z_max,
        num_epochs=100,
        eos=eos,
        physics_weight=0.1,
        loss_type='max',
        device=device,
        patience=40
    )
    
    # Plot training results
    print("\nPlotting training history...")
    plot_training_history(history)
    
    # Generate test data for error analysis
    print("\nGenerating test data for error analysis...")
    C_test, Z_test = setup_initial_state_meshgrid_cold(
        eta, eos, 200, device, 
        Wmin=1.0, Wmax=2.0,  # Increased from 1.3 to 2.0 for consistency
        lrhomin=-8, lrhomax=-3.0
    )
    
    err_test = sanity_check(Z_test, C_test, eta, eos)
    print(f"Test data sanity check error: {err_test:.2e}")
    
    dataset_val = C2P_Dataset(C_test, Z_test, normalize_data=False)
    
    # Create solver for error analysis
    solver = C2P_Solver(model, eos, dataset.C_min, dataset.C_max,
                       dataset.Z_min, dataset.Z_max, device=device)
    print(f"Solver device: {solver.device}")
    
    # Run enhanced error analysis with zeta
    print("\nRunning error analysis...")
    errors, Z_pred, Z_true = plot_error_analysis_with_zeta_absolute(
        solver=solver,
        test_C=dataset_val.C,
        test_Z=dataset_val.Z,
        save_fig=True
    )
    
    # Additional zeta-specific scatter plot analysis
    print("\nCreating scatter plot analysis...")
    plot_zeta_scatter_analysis(Z_true, Z_pred, save_fig=True)
    
    print("\nTraining and analysis complete!")
    print("=" * 80)
    
    return model, solver, history, errors

if __name__ == "__main__":
    # Execute main function
    model, solver, history, errors = main()