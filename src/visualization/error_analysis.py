"""
Error analysis and visualization functions.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from physics.physics_utils import W__z, rho__z, eps__z


def plot_error_analysis_with_zeta(solver, test_C, test_Z, save_fig=False, fig_path=None):
    """
    Plot comprehensive error analysis for C2P solver including zeta.
    """
    # Predict on test data
    rho_pred, eps_pred, press_pred, W_pred = solver.invert(test_C)
    
    # Get Z predictions
    Z_pred = solver.model(solver.normalize_C(test_C))
    Z_pred = solver.denormalize_Z(Z_pred)
    
    # Compute true values
    Z_true = test_Z.view(-1, 1)
    rho_true = rho__z(Z_true, test_C)
    W_true = W__z(Z_true)
    eps_true = eps__z(Z_true, test_C)
    press_true = solver.eos.press__eps_rho(eps_true, rho_true)
    
    # Calculate absolute errors
    rho_error = torch.abs(rho_pred - rho_true)
    eps_error = torch.abs(eps_pred - eps_true)
    press_error = torch.abs(press_pred - press_true)
    W_error = torch.abs(W_pred - W_true)
    Z_error = torch.abs(Z_pred - Z_true)
    
    # Convert to numpy for plotting
    errors = {
        'Density': rho_error.cpu().numpy().flatten(),
        'Specific Energy': eps_error.cpu().numpy().flatten(),
        'Pressure': press_error.cpu().numpy().flatten(),
        'Lorentz Factor': W_error.cpu().numpy().flatten(),
        'Zeta (z)': Z_error.cpu().detach().numpy().flatten()
    }
    
    # Create comprehensive plot
    fig = plt.figure(figsize=(20, 12))
    
    # Plot 1: Error distributions
    for i, (name, error) in enumerate(errors.items(), 1):
        ax = plt.subplot(3, 5, i)
        ax.hist(np.log10(error + 1e-15), bins=50, alpha=0.7, color=f'C{i-1}')
        ax.set_xlabel('log10(Absolute Error)')
        ax.set_ylabel('Frequency')
        ax.set_title(f'{name} Error Distribution')
        ax.grid(True, alpha=0.3)
        
        # Add statistics
        mean_error = np.mean(error)
        max_error = np.max(error)
        ax.axvline(np.log10(mean_error), color='r', linestyle='--',
                   label=f'Mean: {mean_error:.2e}')
        ax.axvline(np.log10(max_error), color='orange', linestyle='--',
                   label=f'Max: {max_error:.2e}')
        ax.legend()
    
    # Plot 2: Scatter plots - True vs Predicted
    primitives = {
        'Zeta (z)': (Z_true.cpu().detach().numpy(), Z_pred.cpu().detach().numpy()),
        'Density': (rho_true.cpu().detach().numpy(), rho_pred.cpu().detach().numpy()),
        'Lorentz Factor': (W_true.cpu().detach().numpy(), W_pred.cpu().detach().numpy())
    }
    
    for i, (name, (true_vals, pred_vals)) in enumerate(primitives.items(), 1):
        ax = plt.subplot(3, 5, 5 + i)
        ax.scatter(true_vals.flatten(), pred_vals.flatten(), alpha=0.5, s=1)
        
        # Add perfect prediction line
        min_val = min(true_vals.min(), pred_vals.min())
        max_val = max(true_vals.max(), pred_vals.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect')
        
        ax.set_xlabel(f'True {name}')
        ax.set_ylabel(f'Predicted {name}')
        ax.set_title(f'{name}: True vs Predicted')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    # Plot 3: Error vs Input magnitude
    ax = plt.subplot(3, 5, 9)
    C_magnitude = torch.norm(test_C, dim=1).cpu().detach().numpy()
    ax.scatter(C_magnitude, Z_error.cpu().detach().numpy().flatten(), alpha=0.5, s=1)
    ax.set_xlabel('||C|| (Input Magnitude)')
    ax.set_ylabel('Z Absolute Error')
    ax.set_title('Error vs Input Magnitude')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Error correlation matrix
    ax = plt.subplot(3, 5, 10)
    error_matrix = np.array([errors[key] for key in ['Density', 'Specific Energy', 
                                                       'Pressure', 'Lorentz Factor', 'Zeta (z)']])
    corr_matrix = np.corrcoef(error_matrix)
    im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    ax.set_xticklabels(['ρ', 'ε', 'P', 'W', 'z'], rotation=45)
    ax.set_yticklabels(['ρ', 'ε', 'P', 'W', 'z'])
    ax.set_title('Error Correlations')
    plt.colorbar(im, ax=ax)
    
    # Plot 5: Summary statistics
    ax = plt.subplot(3, 5, 11)
    ax.axis('off')
    summary_text = "Summary Statistics\n" + "="*30 + "\n"
    for name, error in errors.items():
        summary_text += f"{name}:\n"
        summary_text += f"  Mean: {np.mean(error):.2e}\n"
        summary_text += f"  Max:  {np.max(error):.2e}\n"
        summary_text += f"  Std:  {np.std(error):.2e}\n\n"
    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace')
    
    plt.suptitle('Comprehensive Error Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_fig:
        if fig_path is None:
            fig_path = Path('experiments/figures/error_analysis.png')
        fig_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"Error analysis saved to {fig_path}")
    
    plt.show()
    
    # Return errors for further analysis
    return errors, Z_pred, Z_true


def plot_zeta_scatter_analysis(Z_true, Z_pred, save_fig=False, fig_path=None):
    """
    Detailed scatter plot analysis for zeta predictions.
    """
    Z_true_np = Z_true.cpu().numpy().flatten()
    Z_pred_np = Z_pred.cpu().numpy().flatten()
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Scatter plot with density
    axes[0].hexbin(Z_true_np, Z_pred_np, gridsize=50, cmap='Blues', mincnt=1)
    axes[0].plot([Z_true_np.min(), Z_true_np.max()],
                 [Z_true_np.min(), Z_true_np.max()],
                 'r--', label='Perfect Prediction', linewidth=2)
    
    axes[0].set_xlabel('True Zeta (z)', fontsize=12)
    axes[0].set_ylabel('Predicted Zeta (z)', fontsize=12)
    axes[0].set_title('Zeta Predictions', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # Residual plot
    residuals = Z_pred_np - Z_true_np
    axes[1].scatter(Z_true_np, residuals, alpha=0.5, s=1)
    axes[1].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[1].set_xlabel('True Zeta (z)', fontsize=12)
    axes[1].set_ylabel('Residual (Pred - True)', fontsize=12)
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
            fig_path = Path('experiments/figures/zeta_scatter.png')
        fig_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"Zeta scatter analysis saved to {fig_path}")
    
    plt.show()