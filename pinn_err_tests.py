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

def plot_error_analysis(solver, test_C, test_Z, N=50):
    """Plot comprehensive error analysis"""
    
    # Predict on test data
    rho_pred, eps_pred, press_pred, W_pred = solver.invert(test_C)
    
    # Compute true values
    Z_true = test_Z.view(-1, 1)
    rho_true = rho__z(Z_true, test_C)
    W_true = W__z(Z_true)
    eps_true = eps__z(Z_true, test_C)
    press_true = solver.eos.press__eps_rho(eps_true, rho_true)
    
    # Calculate relative errors
    rho_error = torch.abs(rho_pred - rho_true) / rho_true
    eps_error = torch.abs(eps_pred - eps_true) / (eps_true + 1e-10)
    press_error = torch.abs(press_pred - press_true) / (press_true + 1e-10)
    W_error = torch.abs(W_pred - W_true) / W_true
    
    # Convert to numpy for plotting
    errors = {
        'Density': rho_error.cpu().numpy().flatten(),
        'Specific Energy': eps_error.cpu().numpy().flatten(), 
        'Pressure': press_error.cpu().numpy().flatten(),
        'Lorentz Factor': W_error.cpu().numpy().flatten()
    }
    
    # Create error distribution plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, (name, error) in enumerate(errors.items()):
        axes[i].hist(torch.log10(error + 1e-15), bins=50, alpha=0.7)
        axes[i].set_xlabel('log10(Relative Error)')
        axes[i].set_ylabel('Frequency')
        axes[i].set_title(f'{name} Error Distribution')
        axes[i].grid(True)
        
        # Add statistics
        mean_error = torch.mean(error)
        max_error = torch.max(error)
        axes[i].axvline(torch.log10(mean_error), color='r', linestyle='--', 
                       label=f'Mean: {mean_error:.2e}')
        axes[i].axvline(torch.log10(max_error), color='orange', linestyle='--', 
                       label=f'Max: {max_error:.2e}')
        axes[i].legend()
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print("Error Analysis Summary:")
    print("-" * 40)
    for name, error in errors.items():
        print(f"{name:15s}: Mean={torch.mean(error):.2e}, Max={torch.max(error):.2e}")


def plot_error_analysis_with_zeta(solver, test_C, test_Z, N=50, save_fig=False, fig_path=None):
    """
    Plot comprehensive error analysis for C2P solver including zeta components
    
    Args:
        solver: C2P solver instance
        test_C: Test conservative variables
        test_Z: Test primitive variables (Z/zeta)
        N: Number of bins for histograms
        save_fig: Whether to save the figure
        fig_path: Path to save figure (if save_fig=True)
    """
    # Predict on test data
    rho_pred, eps_pred, press_pred, W_pred = solver.invert(test_C)
    
    # Compute true values
    Z_true = test_Z.view(-1, 1)
    print(f"Z_true shape: {Z_true.shape}")
    print(f"Z_true sample: {Z_true[:5].flatten()}")
    
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
    
    # Calculate relative errors for physical variables
    rho_error = torch.abs(rho_pred - rho_true) / torch.clamp(rho_true, min=1e-15)
    eps_error = torch.abs(eps_pred - eps_true) / torch.clamp(torch.abs(eps_true), min=1e-15)
    press_error = torch.abs(press_pred - press_true) / torch.clamp(torch.abs(press_true), min=1e-15)
    W_error = torch.abs(W_pred - W_true) / torch.clamp(W_true, min=1e-15)
    
    # Calculate zeta error
    Z_error = torch.abs(Z_pred - Z_true) / torch.clamp(torch.abs(Z_true), min=1e-15)
    
    # Convert to numpy for plotting
    errors = {
        'Density (ρ)': rho_error.detach().cpu().numpy().flatten(),
        'Specific Energy (ε)': eps_error.detach().cpu().numpy().flatten(),
        'Pressure (P)': press_error.detach().cpu().numpy().flatten(),
        'Lorentz Factor (W)': W_error.detach().cpu().numpy().flatten(),
        'Zeta (ζ)': Z_error.detach().cpu().numpy().flatten()
    }
    
    # Determine subplot layout based on number of variables
    n_vars = len(errors)
    if n_vars <= 4:
        nrows, ncols = 2, 2
    elif n_vars <= 6:
        nrows, ncols = 2, 3
    else:
        nrows, ncols = 3, 3
    
    # Create error distribution plots
    fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 5*nrows))
    if n_vars == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    colors = ['skyblue', 'lightcoral', 'lightgreen', 'plum', 'gold', 'lightpink']
    
    for i, (name, error) in enumerate(errors.items()):
        # Remove any infinite or NaN values
        error_clean = error[np.isfinite(error)]
        log_error = np.log10(np.maximum(error_clean, 1e-15))
        
        # Create histogram
        n, bins, patches = axes[i].hist(log_error, bins=N, alpha=0.7, 
                                       color=colors[i % len(colors)], 
                                       edgecolor='black', linewidth=0.5)
        
        axes[i].set_xlabel('log₁₀(Relative Error)', fontsize=12)
        axes[i].set_ylabel('Frequency', fontsize=12)
        axes[i].set_title(f'{name} Error Distribution', fontsize=14, fontweight='bold')
        axes[i].grid(True, alpha=0.3)
        
        # Add statistics
        mean_error = np.mean(error_clean)
        max_error = np.max(error_clean)
        median_error = np.median(error_clean)
        p95_error = np.percentile(error_clean, 95)
        
        # Add vertical lines for statistics
        axes[i].axvline(np.log10(mean_error), color='red', linestyle='--', linewidth=2,
                       label=f'Mean: {mean_error:.2e}')
        axes[i].axvline(np.log10(median_error), color='orange', linestyle='--', linewidth=2,
                       label=f'Median: {median_error:.2e}')
        axes[i].axvline(np.log10(p95_error), color='purple', linestyle='--', linewidth=2,
                       label=f'95th percentile: {p95_error:.2e}')
        
        axes[i].legend(fontsize=9)
        
        # Add text box with additional stats
        stats_text = f'Max: {max_error:.2e}\nStd: {np.std(error_clean):.2e}\nSamples: {len(error_clean)}'
        axes[i].text(0.02, 0.98, stats_text, transform=axes[i].transAxes, 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Hide unused subplots
    for i in range(len(errors), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    if save_fig:
        if fig_path is None:
            fig_path = 'error_analysis_with_zeta.png'
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {fig_path}")
    
    plt.show()
    
    # Print comprehensive summary statistics
    print("\n" + "="*70)
    print("COMPREHENSIVE ERROR ANALYSIS SUMMARY (INCLUDING ZETA)")
    print("="*70)
    
    for name, error in errors.items():
        error_clean = error[np.isfinite(error)]
        print(f"\n{name:20s}:")
        print(f"  Mean:       {np.mean(error_clean):.2e}")
        print(f"  Median:     {np.median(error_clean):.2e}")
        print(f"  Std:        {np.std(error_clean):.2e}")
        print(f"  Min:        {np.min(error_clean):.2e}")
        print(f"  Max:        {np.max(error_clean):.2e}")
        print(f"  95th perc:  {np.percentile(error_clean, 95):.2e}")
        print(f"  99th perc:  {np.percentile(error_clean, 99):.2e}")
        print(f"  Samples:    {len(error_clean)}")
        
        # Error quality assessment
        excellent = np.sum(error_clean < 1e-12)
        good = np.sum((error_clean >= 1e-12) & (error_clean < 1e-8))
        fair = np.sum((error_clean >= 1e-8) & (error_clean < 1e-6))
        poor = np.sum(error_clean >= 1e-6)
        
        print(f"  Quality breakdown:")
        print(f"    Excellent (<1e-12): {excellent:6d} ({100*excellent/len(error_clean):5.1f}%)")
        print(f"    Good (1e-12-1e-8):  {good:6d} ({100*good/len(error_clean):5.1f}%)")
        print(f"    Fair (1e-8-1e-6):   {fair:6d} ({100*fair/len(error_clean):5.1f}%)")
        print(f"    Poor (>1e-6):       {poor:6d} ({100*poor/len(error_clean):5.1f}%)")
    
    print("\n" + "="*70)
    
    # Additional zeta-specific analysis
    print("\nZETA-SPECIFIC ANALYSIS:")
    print("-" * 30)
    Z_true_np = Z_true.detach().cpu().numpy().flatten()
    Z_pred_np = Z_pred.detach().cpu().numpy().flatten()
    Z_error_np = Z_error.detach().cpu().numpy().flatten()
    
    print(f"Zeta range (true):      [{np.min(Z_true_np):.4f}, {np.max(Z_true_np):.4f}]")
    print(f"Zeta range (predicted): [{np.min(Z_pred_np):.4f}, {np.max(Z_pred_np):.4f}]")
    print(f"Zeta correlation:       {np.corrcoef(Z_true_np, Z_pred_np)[0,1]:.6f}")
    
    # Find worst predictions
    worst_indices = np.argsort(Z_error_np)[-5:]
    print(f"\nWorst 5 zeta predictions:")
    for idx in worst_indices:
        print(f"  Sample {idx}: True={Z_true_np[idx]:.6f}, Pred={Z_pred_np[idx]:.6f}, Error={Z_error_np[idx]:.2e}")
    
    return errors, Z_pred, Z_true

def plot_zeta_scatter_analysis(Z_true, Z_pred, save_fig=False, fig_path=None):
    """
    Create scatter plots comparing true vs predicted zeta values
    
    Args:
        Z_true: True zeta values
        Z_pred: Predicted zeta values
        save_fig: Whether to save the figure
        fig_path: Path to save figure
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
