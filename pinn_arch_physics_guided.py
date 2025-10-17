#pinn_arch_physics_guided.py


import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split

# Physics modules (assuming these are available)
from metric import metric
from hybrid_eos import hybrid_eos
from C2P_Solver import *


# =============================================================================
# Many different Physics-Guided PINN architectures
# =============================================================================

class PhysicsGuided_TinyPINNv1(nn.Module):
    """
    Physics-guided approach: Start with analytical approximation,
    then learn small corrections
    """
    def __init__(self, dtype=torch.float64):
        super().__init__()
        
        # Tiny correction network
        self.correction_net = nn.Sequential(
            nn.Linear(3, 6, dtype=dtype),    # 3*6 + 6 = 24 FLOPs
            nn.Tanh(),                       # 6 FLOPs
            nn.Linear(6, 1, dtype=dtype),    # 6*1 + 1 = 7 FLOPs
        )
        
        # Small correction scale
        self.correction_scale = 0.05
        
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
        
        # Combined guess
        z_guess = z_kinematic * thermal_factor * relativistic_correction
        
        # Ensure physical bounds: Z ≥ 1
        z_guess = torch.clamp(z_guess, min=1.0)
        
        return z_guess
    
    def forward(self, x):
        # Start with physics-based guess
        z_baseline = self.analytical_guess(x)
        
        # Small learned correction
        correction = self.correction_net(x)
        
        return z_baseline + self.correction_scale * correction
        # Total: ~37 FLOPs + analytical_guess, ~43 parameters

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

class PhysicsGuided_TinyPINNv2(nn.Module):
    """
    Physics-guided approach: Start with analytical approximation,
    then learn small corrections
    """
    def __init__(self, dtype=torch.float64):
        super().__init__()
        
        # Tiny correction network
        self.correction_net = nn.Sequential(
        nn.Linear(3, 12, dtype=dtype),    # 3*12 = 36 FLOPs (not 24)
        nn.Tanh(),                       # 12 FLOPs (not 6)
        nn.Linear(12, 1, dtype=dtype),    # 12*1 = 12 FLOPs (not 7)
        )
        # Total: ~60 FLOPs + analytical_guess, ~49 parameters (3*12 + 12 + 12*1 + 1)

        
        # Small correction scale
        self.correction_scale = 0.1
        
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
        
        # Combined guess
        z_guess = z_kinematic * thermal_factor * relativistic_correction
        
        # Ensure physical bounds: Z ≥ 1
        z_guess = torch.clamp(z_guess, min=1.0)
        
        return z_guess
    
    def forward(self, x):
        # Start with physics-based guess
        z_baseline = self.analytical_guess(x)
        
        # Small learned correction
        correction = self.correction_net(x)
        
        return z_baseline + self.correction_scale * correction
        # Total: ~37 FLOPs + analytical_guess, ~43 parameters

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


class PhysicsGuided_TinyPINNv3(nn.Module):
    """
    Physics-guided approach: Start with analytical approximation,
    then learn small corrections
    """
    def __init__(self, dtype=torch.float64, correction_scale=0.2):
        super().__init__()
        
        # Tiny correction network
        self.correction_net = nn.Sequential(
            nn.Linear(3, 24, dtype=dtype),    # 3*24 = 72 FLOPs (not 48)
            nn.Tanh(),                       # 24 FLOPs (not 24, this was correct)
            nn.Linear(24, 1, dtype=dtype),    # 24*1 = 24 FLOPs (not 7)
        )
        # Total: ~72 FLOPs + analytical_guess, ~97 parameters (3*24 + 24 + 24*1 + 1)

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
        # Total: ~37 FLOPs + analytical_guess, ~43 parameters

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
        
class PhysicsGuided_TinyPINNvKen(nn.Module):
    """
    Physics-guided approach: Start with analytical approximation,
    then learn small corrections
    """
    def __init__(self, dtype=torch.float64, correction_scale=0.2):
        super().__init__()
        
        # Tiny correction network
        self.correction_net = nn.Sequential(
            nn.Linear(3, 64, dtype=dtype),    # 3*64 = 192 FLOPs (not 48)
            nn.Tanh(),                       # 64 FLOPs (not 24, this was correct)
            nn.Linear(64, 1, dtype=dtype),    # 64*1 = 64 FLOPs (not 7)
        )
        # Total: ~192 FLOPs + analytical_guess, ~97 parameters (3*64 + 64 + 64*1 + 1)

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
        Based on power law relationship: Z = 1.693 * q^0.577
        """
        D = C[:, 0:1]  # Conserved density
        q = C[:, 1:2]  # τ/D (internal energy density)
        r = C[:, 2:3]  # S/D (momentum density)
        
        # Primary relationship: Z from q using power law fit
        # Handle potential negative q values by taking absolute value
        q_safe = torch.abs(q) + 1e-12  # Small epsilon to avoid q=0
        z_primary = 1.693 * torch.pow(q_safe, 0.577)
        
        # Optional: Add small corrections from other variables if needed
        # z_correction = 0.05 * r  # Small momentum correction
        # z_guess = z_primary + z_correction
        
        z_guess = z_primary
        
        # Physical bounds: 0 ≤ z ≤ max_expected_z
        return torch.clamp(z_guess, min=0.0, max=0.9)
    
    def forward(self, x):
        # Start with physics-based guess
        z_baseline = self.analytical_guess(x)
        
        # Small learned correction
        correction = self.correction_net(x)
        
        return z_baseline + self.correction_scale * correction
        # Total: ~37 FLOPs + analytical_guess, ~43 parameters

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



class NeutronStarPhysicsGuidedPINN(nn.Module):
    """
    Multi-regime neural network for neutron star GRMHD
    Handles extreme density variations from vacuum to nuclear density
    """
    
    def __init__(self, dtype=torch.float64):
        super().__init__()
        
        # Density regime thresholds (in code units)
        self.vacuum_threshold = 1e-8
        self.crust_threshold = 1e-5
        self.core_threshold = 1e-3
        
        # Regime-specific neural networks
        self.vacuum_net = self._create_regime_net(3, 4, 1, dtype)  # Simplest
        self.crust_net = self._create_regime_net(3, 6, 1, dtype)   # Medium
        self.core_net = self._create_regime_net(3, 8, 1, dtype)    # Most complex
        
        # Density-dependent correction scales
        self.vacuum_scale = 0.05   # Small corrections for vacuum
        self.crust_scale = 0.1     # Medium corrections for crust
        self.core_scale = 0.2      # Large corrections for core
        
        self.apply(self._init_weights)
    
    def _create_regime_net(self, input_dim, hidden_dim, output_dim, dtype):
        """Create a regime-specific neural network"""
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim, dtype=dtype),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim//2, dtype=dtype),
            nn.Tanh(),
            nn.Linear(hidden_dim//2, output_dim, dtype=dtype)
        )
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=0.1)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    def analytical_guess_multi_regime(self, C):
        """
        Regime-specific analytical guesses for neutron star
        """
        D = C[:, 0:1]  # Conserved density
        q = C[:, 1:2]  # τ/D (internal energy density)
        r = C[:, 2:3]  # S/D (momentum density)
        
        # Basic kinematic component
        z_kinematic = torch.sqrt(1.0 + r**2)
        
        # Regime-specific corrections
        vacuum_mask = D < self.vacuum_threshold
        crust_mask = (D >= self.vacuum_threshold) & (D < self.crust_threshold)
        core_mask = (D >= self.crust_threshold) & (D < self.core_threshold)
        nuclear_mask = D >= self.core_threshold
        
        # Initialize with kinematic guess
        z_guess = z_kinematic.clone()
        
        # VACUUM REGIME: Ideal gas approximation
        if vacuum_mask.any():
            thermal_factor = 1.0 + 1.5 * q[vacuum_mask]  # γ = 5/3
            z_guess[vacuum_mask] = z_kinematic[vacuum_mask] * thermal_factor
        
        # CRUST REGIME: Includes degeneracy pressure
        if crust_mask.any():
            # Degeneracy pressure becomes important
            degeneracy_factor = 1.0 + 2.0 * q[crust_mask]
            density_correction = 1.0 + 0.1 * torch.log(1.0 + D[crust_mask] * 1e5)
            z_guess[crust_mask] = (z_kinematic[crust_mask] * 
                                   degeneracy_factor * density_correction)
        
        # CORE REGIME: Relativistic equation of state
        if core_mask.any():
            # Highly relativistic, stiff EOS
            relativistic_factor = 1.0 + 3.0 * q[core_mask]
            stiffness_correction = 1.0 + 0.2 * D[core_mask] / (1.0 + D[core_mask])
            z_guess[core_mask] = (z_kinematic[core_mask] * 
                                  relativistic_factor * stiffness_correction)
        
        # NUCLEAR REGIME: Exotic matter
        if nuclear_mask.any():
            # Extremely stiff EOS, possible phase transitions
            exotic_factor = 1.0 + 5.0 * q[nuclear_mask] / (1.0 + q[nuclear_mask])
            nuclear_correction = 1.0 + 0.5 * torch.log(1.0 + D[nuclear_mask] * 1e3)
            z_guess[nuclear_mask] = (z_kinematic[nuclear_mask] * 
                                     exotic_factor * nuclear_correction)
        
        return torch.clamp(z_guess, min=1.0, max=100.0)
    
    def forward(self, x):
        """
        Forward pass with regime-specific neural corrections
        """
        D = x[:, 0:1]
        
        # Get regime-specific analytical baseline
        z_baseline = self.analytical_guess_multi_regime(x)
        
        # Apply regime-specific neural corrections
        corrections = torch.zeros_like(D)
        
        # Vacuum regime
        vacuum_mask = D < self.vacuum_threshold
        if vacuum_mask.any():
            vacuum_correction = self.vacuum_net(x[vacuum_mask.squeeze()])
            corrections[vacuum_mask] = self.vacuum_scale * vacuum_correction
        
        # Crust regime
        crust_mask = (D >= self.vacuum_threshold) & (D < self.crust_threshold)
        if crust_mask.any():
            crust_correction = self.crust_net(x[crust_mask.squeeze()])
            corrections[crust_mask] = self.crust_scale * crust_correction
        
        # Core regime
        core_mask = (D >= self.crust_threshold) & (D < self.core_threshold)
        if core_mask.any():
            core_correction = self.core_net(x[core_mask.squeeze()])
            corrections[core_mask] = self.core_scale * core_correction
        
        # Nuclear regime (use core network with larger correction)
        nuclear_mask = D >= self.core_threshold
        if nuclear_mask.any():
            nuclear_correction = self.core_net(x[nuclear_mask.squeeze()])
            corrections[nuclear_mask] = 2.0 * self.core_scale * nuclear_correction
        
        return z_baseline + corrections
    
    def regime_aware_physics_loss(self, C, Z_pred, eos):
        """
        Physics loss with regime-specific weighting
        """
        D = C[:, 0:1]
        q = C[:, 1:2]
        r = C[:, 2:3]
        
        try:
            # Compute physics residual
            htilde = self.compute_htilde(Z_pred, C, eos)
            residual = Z_pred - (r / htilde)
            
            # Regime-specific loss weights
            weights = torch.ones_like(D)
            
            # Higher weight for high-density regions (where errors matter more)
            high_density_mask = D > self.crust_threshold
            weights[high_density_mask] *= 10.0  # 10x higher weight for core
            
            # Even higher weight for nuclear density
            nuclear_mask = D > self.core_threshold
            weights[nuclear_mask] *= 50.0  # 50x higher weight for nuclear
            
            # Weighted physics loss
            physics_loss = torch.mean(weights * residual**2)
            
            # Add conservation constraints
            conservation_loss = self.compute_conservation_loss(C, Z_pred)
            
            return physics_loss + 0.1 * conservation_loss
            
        except Exception as e:
            print(f"Physics loss failed: {e}")
            return torch.sum(Z_pred * 0.0)
    
    def compute_htilde(self, Z, C, eos):
        """Compute specific enthalpy"""
        # This would interface with your EOS
        # Placeholder implementation
        D = C[:, 0:1]
        q = C[:, 1:2]
        
        rho = D / Z
        u = q - 0.5 * C[:, 2:3]**2 / Z
        
        # EOS call (simplified)
        gamma = 5.0/3.0
        h = 1.0 + gamma * u / rho
        
        return h
    
    def compute_conservation_loss(self, C, Z):
        """Additional conservation constraints"""
        # Energy conservation constraint
        D = C[:, 0:1]
        q = C[:, 1:2]
        r = C[:, 2:3]
        
        # Ensure internal energy is positive
        u = q - 0.5 * r**2 / Z
        energy_constraint = torch.mean(torch.relu(-u))  # Penalize negative energy
        
        # Ensure density is positive
        rho = D / Z
        density_constraint = torch.mean(torch.relu(-rho))
        
        return energy_constraint + density_constraint
