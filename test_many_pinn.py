#!/scratch/astro/miler/python-env/pytorch/bin/python3
"""
Comprehensive PINN Size Comparison Script for PhysicsGuided_TinyPINN
Tests different network architectures with the same input data to evaluate size vs performance tradeoffs
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

class TinyPINN_Variant1(nn.Module):
    """Ultra-tiny: 3->4->1 (25 params)"""
    def __init__(self, dtype=torch.float64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(3, 4, dtype=dtype),
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
    
    def forward(self, x):
        return self.network(x) * self.correction_scale
    
    def physics_loss(self, C, Z_pred, eos):
        D, q, r = C[:, 0:1], C[:, 1:2], C[:, 2:3]
        try:
            htilde = h__z(Z_pred, C, eos)
            residual = Z_pred - (r / htilde)
            return torch.mean(residual**2)
        except Exception:
            return torch.sum(Z_pred * 0.0)


class TinyPINN_Variant2(nn.Module):
    """Small: 3->6->1 (31 params) - Your original PhysicsGuided_TinyPINN"""
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
    
    def forward(self, x):
        return self.correction_net(x) * self.correction_scale
    
    def physics_loss(self, C, Z_pred, eos):
        D, q, r = C[:, 0:1], C[:, 1:2], C[:, 2:3]
        try:
            htilde = h__z(Z_pred, C, eos)
            residual = Z_pred - (r / htilde)
            return torch.mean(residual**2)
        except Exception:
            return torch.sum(Z_pred * 0.0)


class TinyPINN_Variant3(nn.Module):
    """Medium: 3->8->4->1 (57 params)"""
    def __init__(self, dtype=torch.float64):
        super().__init__()
        self.network = nn.Sequential(
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
    
    def forward(self, x):
        return self.network(x) * self.correction_scale
    
    def physics_loss(self, C, Z_pred, eos):
        D, q, r = C[:, 0:1], C[:, 1:2], C[:, 2:3]
        try:
            htilde = h__z(Z_pred, C, eos)
            residual = Z_pred - (r / htilde)
            return torch.mean(residual**2)
        except Exception:
            return torch.sum(Z_pred * 0.0)


class TinyPINN_Variant4(nn.Module):
    """Larger: 3->16->8->1 (153 params)"""
    def __init__(self, dtype=torch.float64):
        super().__init__()
        self.network = nn.Sequential(
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
    
    def forward(self, x):
        return self.network(x) * self.correction_scale
    
    def physics_loss(self, C, Z_pred, eos):
        D, q, r = C[:, 0:1], C[:, 1:2], C[:, 2:3]
        try:
            htilde = h__z(Z_pred, C, eos)
            residual = Z_pred - (r / htilde)
            return torch.mean(residual**2)
        except Exception:
            return torch.sum(Z_pred * 0.0)


class TinyPINN_Variant5(nn.Module):
    """Large: 3->32->16->8->1 (585 params)"""
    def __init__(self, dtype=torch.float64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(3, 32, dtype=dtype),
            nn.SiLU(),
            nn.Linear(32, 16, dtype=dtype),
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
    
    def forward(self, x):
        return self.network(x) * self.correction_scale
    
    def physics_loss(self, C, Z_pred, eos):
        D, q, r = C[:, 0:1], C[:, 1:2], C[:, 2:3]
        try:
            htilde = h__z(Z_pred, C, eos)
            residual = Z_pred - (r / htilde)
            return torch.mean(residual**2)
        except Exception:
            return torch.sum(Z_pred * 0.0)


class FourierPINN_Small(nn.Module):
    """Small Fourier PINN: 4 features -> 8 hidden -> 1 (~145 params)"""
    def __init__(self, num_fourier_features=4, fourier_sigma=3.0, dtype=torch.float64):
        super().__init__()
        self.register_buffer('two_pi', torch.tensor(2.0 * torch.pi, dtype=dtype))
        fourier_matrix = torch.randn(3, num_fourier_features, dtype=dtype) * fourier_sigma
        self.register_buffer('fourier_B', fourier_matrix)
        
        fourier_dim = 2 * num_fourier_features
        self.network = nn.Sequential(
            nn.Linear(fourier_dim, 8, dtype=dtype),
            nn.SiLU(),
            nn.Linear(8, 1, dtype=dtype),
        )
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=1.0)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        x_proj = self.two_pi * (x @ self.fourier_B)
        fourier_features = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        return self.network(fourier_features)
    
    def physics_loss(self, C, Z_pred, eos):
        D, q, r = C[:, 0:1], C[:, 1:2], C[:, 2:3]
        try:
            htilde = h__z(Z_pred, C, eos)
            residual = Z_pred - (r / htilde)
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
            # FLOPs = input_features * output_features + output_features (bias)
            total_flops += module.in_features * module.out_features + module.out_features
        elif isinstance(module, (nn.Tanh, nn.SiLU)):
            # Activation functions: roughly 1 FLOP per element
            # Estimate based on previous layer output size
            total_flops += input_tensor.numel()
    
    return total_flops


def setup_physics():
    """Setup metric and EOS"""
    # Minkowski metric in 3+1 decomposition
    g = torch.eye(3, dtype=torch.float64)
    beta = torch.zeros(3, dtype=torch.float64)
    alp = torch.tensor(1.0, dtype=torch.float64)
    
    # Create metric object
    metric_obj = metric(g, beta, alp)
    
    # Setup EOS (hybrid)
    eos = hybrid_eos(100, 2, 1.8)
    
    return metric_obj, eos


def generate_test_data(metric_obj, eos, n_samples, device):
    """Generate test dataset"""
    print(f"Generating {n_samples} test samples...")
    
    # Generate random state
    C_test, Z_test = setup_initial_state_random(
        metric_obj, eos, n_samples, device,
        lrhomin=-12, lrhomax=-2.8,
        ltempmin=-1, ltempmax=2.3,
        Wmin=1.0, Wmax=2.0
    )
    
    # Ensure double precision
    C_test = C_test.to(dtype=torch.float64)
    Z_test = Z_test.to(dtype=torch.float64)
    
    return C_test, Z_test


def train_model(model, train_loader, val_loader, eos, device, epochs=200, lr=1e-3, physics_weight=1.0):
    """Train a single model"""
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=20, factor=0.5)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    print(f"Training model with {count_parameters(model)} parameters...")
    
    for epoch in range(epochs):
        # Training
        model.train()
        epoch_train_loss = 0.0
        
        for batch_idx, (C_batch, Z_batch) in enumerate(train_loader):
            # Ensure double precision
            C_batch = C_batch.to(device, dtype=torch.float64)
            Z_batch = Z_batch.to(device, dtype=torch.float64)
            
            optimizer.zero_grad()
            
            # Forward pass
            Z_pred = model(C_batch)
            
            # Data loss (MSE)
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
                # Ensure double precision
                C_batch = C_batch.to(device, dtype=torch.float64)
                Z_batch = Z_batch.to(device, dtype=torch.float64)
                Z_pred = model(C_batch)
                
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
    # Ensure double precision
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
    
    return {
        'abs_error_mean': torch.mean(abs_error).item(),
        'abs_error_max': torch.max(abs_error).item(),
        'abs_error_std': torch.std(abs_error).item(),
        'rel_error_mean': torch.mean(rel_error).item(),
        'rel_error_max': torch.max(rel_error).item(),
        'rel_error_std': torch.std(rel_error).item(),
        'physics_error': physics_error.item(),
        'inference_time': inference_time,
        'correlation': torch.corrcoef(torch.cat([Z_pred.flatten(), test_Z.flatten()]))[0, 1].item(),
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
    
    # Define models to test
    models_to_test = {
        'TinyPINN_V1_4nodes': TinyPINN_Variant1(),
        'TinyPINN_V2_6nodes': TinyPINN_Variant2(),  # Your original
        'TinyPINN_V3_8x4': TinyPINN_Variant3(),
        'TinyPINN_V4_16x8': TinyPINN_Variant4(),
        'TinyPINN_V5_32x16x8': TinyPINN_Variant5(),
        'FourierPINN_Small': FourierPINN_Small(),
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
            physics_weight=args.physics_weight
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
        axes[0, 0].annotate(name.split('_')[1], (n_params[i], abs_errors[i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # Plot 2: Parameters vs Relative Error
    axes[0, 1].loglog(n_params, rel_errors, 'ro-', linewidth=2, markersize=8)
    axes[0, 1].set_xlabel('Number of Parameters')
    axes[0, 1].set_ylabel('Mean Relative Error')
    axes[0, 1].set_title('Model Size vs Relative Error')
    axes[0, 1].grid(True, alpha=0.3)
    for i, name in enumerate(model_names):
        axes[0, 1].annotate(name.split('_')[1], (n_params[i], rel_errors[i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # Plot 3: Parameters vs Physics Error
    axes[0, 2].loglog(n_params, physics_errors, 'go-', linewidth=2, markersize=8)
    axes[0, 2].set_xlabel('Number of Parameters')
    axes[0, 2].set_ylabel('Physics Error')
    axes[0, 2].set_title('Model Size vs Physics Consistency')
    axes[0, 2].grid(True, alpha=0.3)
    for i, name in enumerate(model_names):
        axes[0, 2].annotate(name.split('_')[1], (n_params[i], physics_errors[i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # Plot 4: Parameters vs Correlation
    axes[1, 0].semilogx(n_params, correlations, 'mo-', linewidth=2, markersize=8)
    axes[1, 0].set_xlabel('Number of Parameters')
    axes[1, 0].set_ylabel('Correlation')
    axes[1, 0].set_title('Model Size vs Prediction Correlation')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim([0.9, 1.0])
    for i, name in enumerate(model_names):
        axes[1, 0].annotate(name.split('_')[1], (n_params[i], correlations[i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # Plot 5: Parameters vs Inference Time
    axes[1, 1].loglog(n_params, inference_times, 'co-', linewidth=2, markersize=8)
    axes[1, 1].set_xlabel('Number of Parameters')
    axes[1, 1].set_ylabel('Inference Time (s)')
    axes[1, 1].set_title('Model Size vs Inference Speed')
    axes[1, 1].grid(True, alpha=0.3)
    for i, name in enumerate(model_names):
        axes[1, 1].annotate(name.split('_')[1], (n_params[i], inference_times[i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # Plot 6: Efficiency scatter (Error vs Speed)
    axes[1, 2].loglog(inference_times, abs_errors, 'ko', markersize=10)
    axes[1, 2].set_xlabel('Inference Time (s)')
    axes[1, 2].set_ylabel('Mean Absolute Error')
    axes[1, 2].set_title('Speed vs Accuracy Trade-off')
    axes[1, 2].grid(True, alpha=0.3)
    for i, name in enumerate(model_names):
        axes[1, 2].annotate(name.split('_')[1], (inference_times[i], abs_errors[i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create individual training curves
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, (name, result) in enumerate(results.items()):
        if i >= 6:
            break
        ax = axes[i]
        epochs = range(len(result['train_losses']))
        ax.semilogy(epochs, result['train_losses'], label='Train', linewidth=2)
        ax.semilogy(epochs, result['val_losses'], label='Validation', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title(f'{name} ({result["n_parameters"]} params)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/training_curves.png', dpi=300, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='PINN Size Comparison Script')
    parser.add_argument('--n_train', type=int, default=200, help='Number of training samples')
    parser.add_argument('--n_val', type=int, default=100, help='Number of validation samples')
    parser.add_argument('--n_test', type=int, default=100, help='Number of test samples')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--physics_weight', type=float, default=1.0, help='Physics loss weight')
    parser.add_argument('--output_dir', type=str, default='./pinn_size_comparison', help='Output directory')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("PINN Size Comparison Study")
    print("=" * 50)
    print(f"Training samples: {args.n_train}")
    print(f"Validation samples: {args.n_val}")
    print(f"Test samples: {args.n_test}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Physics weight: {args.physics_weight}")
    print(f"Output directory: {args.output_dir}")
    
    # Run comparison
    results = run_comparison(args)
    
    # Save results
    results_file = f"{args.output_dir}/comparison_results.json"
    
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
    
    print("\nComparison complete!")
    print(f"Results saved in: {args.output_dir}")
    
    # Print summary table
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    print(f"{'Model':<20} {'Params':<8} {'Abs Err':<12} {'Rel Err':<12} {'Physics Err':<12} {'Correlation':<12} {'Time (s)':<8}")
    print("-"*80)
    
    for model_name, result in results.items():
        test_res = result['test_results']
        print(f"{model_name:<20} {result['n_parameters']:<8} {test_res['abs_error_mean']:<12.2e} "
              f"{test_res['rel_error_mean']:<12.2e} {test_res['physics_error']:<12.2e} "
              f"{test_res['correlation']:<12.6f} {test_res['inference_time']:<8.4f}")
    
    print("="*80)


if __name__ == "__main__":
    main()