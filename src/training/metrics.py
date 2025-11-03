"""
Evaluation metrics for PINN model assessment.
"""

import torch
import numpy as np
from typing import Dict, Tuple
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from physics.physics_utils import W__z, rho__z, eps__z, h__z, sanity_check


def compute_primitives(z, C, eos):
    """
    Compute all primitive variables from z and conservative variables.
    
    Returns:
        rho: Rest-mass density
        eps: Specific internal energy  
        press: Pressure
        W: Lorentz factor
        v: Three-velocity magnitude
    """
    # Basic primitives
    W = W__z(z)
    rho = rho__z(z, C)
    eps = eps__z(z, C)
    press = eos.press__eps_rho(eps, rho)
    
    # Velocity
    v = torch.sqrt(1 - 1/W**2)
    
    return rho, eps, press, W, v


def compute_metrics(solver, C_test, Z_test, eos) -> Dict:
    """
    Compute comprehensive evaluation metrics.
    
    Args:
        solver: C2P_Solver instance
        C_test: Test conservative variables
        Z_test: Test z values
        eos: Equation of state
    
    Returns:
        Dictionary of computed metrics
    """
    with torch.no_grad():
        # Get predictions using solver
        rho_pred, eps_pred, press_pred, W_pred = solver.invert(C_test)
        Z_pred = solver.model(solver.normalize_C(C_test))
        Z_pred = solver.denormalize_Z(Z_pred)
        
        # Compute true primitives
        Z_true = Z_test.view(-1, 1)
        rho_true, eps_true, press_true, W_true, v_true = compute_primitives(
            Z_true, C_test, eos
        )
        
        # Compute errors
        metrics = {}
        
        # Absolute errors
        metrics['Z_mae'] = torch.mean(torch.abs(Z_pred - Z_true)).item()
        metrics['Z_rmse'] = torch.sqrt(torch.mean((Z_pred - Z_true)**2)).item()
        metrics['Z_max_error'] = torch.max(torch.abs(Z_pred - Z_true)).item()
        
        # Relative errors
        metrics['rho_rel_error'] = torch.mean(
            torch.abs(rho_pred - rho_true) / (rho_true + 1e-15)
        ).item()
        
        metrics['eps_rel_error'] = torch.mean(
            torch.abs(eps_pred - eps_true) / (torch.abs(eps_true) + 1e-15)
        ).item()
        
        metrics['press_rel_error'] = torch.mean(
            torch.abs(press_pred - press_true) / (press_true + 1e-15)
        ).item()
        
        metrics['W_rel_error'] = torch.mean(
            torch.abs(W_pred - W_true) / (W_true + 1e-15)
        ).item()
        
        # Correlation coefficient
        if len(Z_pred) > 1:
            Z_pred_flat = Z_pred.flatten()
            Z_true_flat = Z_true.flatten()
            
            if torch.std(Z_pred_flat) > 1e-8 and torch.std(Z_true_flat) > 1e-8:
                correlation = torch.corrcoef(
                    torch.stack([Z_pred_flat, Z_true_flat])
                )[0, 1].item()
            else:
                correlation = 0.0
        else:
            correlation = 0.0
        
        metrics['correlation'] = correlation
        
        # Percentile errors
        errors = torch.abs(Z_pred - Z_true)
        metrics['error_p50'] = torch.median(errors).item()
        metrics['error_p90'] = torch.quantile(errors, 0.9).item()
        metrics['error_p95'] = torch.quantile(errors, 0.95).item()
        metrics['error_p99'] = torch.quantile(errors, 0.99).item()
        
        # Physics consistency check
        physics_error = sanity_check(Z_pred, C_test, None, eos)
        metrics['physics_consistency'] = physics_error.item()
        
    return metrics


def compute_regime_metrics(solver, C_test_dict, Z_test_dict, eos, regime_names):
    """
    Compute metrics for different physical regimes.
    
    Args:
        C_test_dict: Dictionary of regime -> conservative variables
        Z_test_dict: Dictionary of regime -> z values
        regime_names: List of regime names
    
    Returns:
        Dictionary of regime-specific metrics
    """
    regime_metrics = {}
    
    for regime in regime_names:
        C_test = C_test_dict[regime]
        Z_test = Z_test_dict[regime]
        
        metrics = compute_metrics(solver, C_test, Z_test, eos)
        regime_metrics[regime] = metrics
    
    return regime_metrics


def compute_speed_metrics(model, C_test, device, n_warmup=10, n_runs=100):
    """
    Compute inference speed metrics.
    
    Returns:
        Dictionary with timing statistics
    """
    model.eval()
    C_test = C_test.to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(C_test)
    
    # Timing runs
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    import time
    times = []
    
    with torch.no_grad():
        for _ in range(n_runs):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            start = time.perf_counter()
            _ = model(C_test)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            end = time.perf_counter()
            times.append(end - start)
    
    times = np.array(times)
    
    return {
        'mean_time': np.mean(times),
        'std_time': np.std(times),
        'min_time': np.min(times),
        'max_time': np.max(times),
        'samples_per_second': len(C_test) / np.mean(times)
    }