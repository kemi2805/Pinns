#!/usr/bin/env python3
"""
Fast and efficient PINN training script for Conservative-to-Primitive conversion.
Optimized for multi-GPU training and high performance.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DataParallel, DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

# Import custom modules
from data.data_utils import setup_initial_state_meshgrid_cold
from data.dataset import C2P_Dataset
from models.physics_guided import PhysicsGuided_TinyPINNvKen
from physics.hybrid_eos import hybrid_eos
from physics.metric import metric
from physics.physics_utils import sanity_check
from solvers.c2p_solver import C2P_Solver
from training.losses import combined_loss
from training.metrics import compute_metrics
from visualization.error_analysis import plot_error_analysis_with_zeta
from visualization.plots import plot_training_history
from analysis.analization import analyze_analytical_relationship
from solvers.model_io import export_physics_guided_model_to_hdf5


def setup_device():
    """Setup compute device and optimization settings."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Set precision
    torch.set_default_dtype(torch.float64)
    
    # Optimize CUDA settings if available
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        print(f"Using GPU: {torch.cuda.get_device_name()}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
    else:
        # Optimize CPU settings
        num_threads = min(os.cpu_count() or 1, 32)
        torch.set_num_threads(num_threads)
        print(f"Using CPU with {num_threads} threads")
    
    return device


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_physics_objects(device, config):
    """Create metric and EOS objects."""
    # Minkowski metric
    eta = metric(
        torch.eye(3, device=device),
        torch.zeros(3, device=device),
        torch.ones(1, device=device)
    )
    
    # Equation of state
    eos_params = config['physics']['eos']
    eos = hybrid_eos(
        eos_params['K'],
        eos_params['gamma'],
        eos_params['gamma_thermal']
    )
    
    return eta, eos


def generate_data(eta, eos, device, config):
    """Generate training and validation data."""
    data_config = config['data']
    
    print("Generating training data...")
    C_train, Z_train = setup_initial_state_meshgrid_cold(
        eta, eos,
        N=data_config['n_train'],
        device=device,
        lrhomin=data_config['lrho_min'],
        lrhomax=data_config['lrho_max'],
        Wmin=data_config['W_min'],
        Wmax=data_config['W_max']
    )
    
    print("Generating validation data...")
    C_val, Z_val = setup_initial_state_meshgrid_cold(
        eta, eos,
        N=data_config['n_val'],
        device=device,
        lrhomin=data_config['lrho_min'],
        lrhomax=data_config['lrho_max'],
        Wmin=data_config['W_min'],
        Wmax=data_config['W_max']
    )
    
    # Sanity check
    err_train = sanity_check(Z_train, C_train, eta, eos)
    err_val = sanity_check(Z_val, C_val, eta, eos)
    print(f"Training data sanity check: {err_train:.2e}")
    print(f"Validation data sanity check: {err_val:.2e}")
    
    return C_train, Z_train, C_val, Z_val


def create_dataloaders(C_train, Z_train, C_val, Z_val, config):
    """Create DataLoaders for training and validation."""
    batch_size = config['training']['batch_size']
    
    # Create datasets
    train_dataset = C2P_Dataset(C_train, Z_train, normalize_data=True)
    val_dataset = C2P_Dataset(C_val, Z_val, 
                             normalize_data=True,
                             )
    
    # Create dataloaders with optimization
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        #pin_memory=True,
        #persistent_workers=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        #pin_memory=True,
        #persistent_workers=True
    )
    
    return train_loader, val_loader, train_dataset


def create_model(device, config):
    """Create and initialize the model."""
    model_config = config['model']
    
    # Select model based on config
    model_type = model_config['type']
    
    if model_type == 'PhysicsGuided_TinyPINNvKen':
        model = PhysicsGuided_TinyPINNvKen(
            dtype=torch.float64,
            correction_scale=model_config.get('correction_scale', 0.2)
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Multi-GPU support
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = DataParallel(model)
    
    model = model.to(device)
    
    return model


def setup_training(model, config):
    """Setup optimizer, scheduler, and scaler for training."""
    train_config = config['training']
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=train_config['learning_rate'],
        weight_decay=train_config.get('weight_decay', 1e-5)
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=train_config.get('scheduler_patience', 10),
    )
    
    # Mixed precision scaler (optional)
    use_amp = train_config.get('use_amp', False)
    scaler = GradScaler() if use_amp else None
    
    return optimizer, scheduler, scaler


def train_epoch(model, train_loader, optimizer, eos, config, scaler=None):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    physics_weight = config['training']['physics_weight']
    
    for batch_idx, (C_batch, Z_batch) in enumerate(train_loader):
        optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
        
        if scaler is not None:
            with autocast():
                loss = combined_loss(
                    model, C_batch, Z_batch, eos,
                    physics_weight=physics_weight
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss = combined_loss(
                model, C_batch, Z_batch, eos,
                physics_weight=physics_weight
            )
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)


def validate(model, val_loader, eos, config):
    """Validate the model."""
    model.eval()
    total_loss = 0
    physics_weight = config['training']['physics_weight']
    
    with torch.no_grad():
        for C_batch, Z_batch in val_loader:
            loss = combined_loss(
                model, C_batch, Z_batch, eos,
                physics_weight=physics_weight
            )
            total_loss += loss.item()
    
    return total_loss / len(val_loader)


def train(model, train_loader, val_loader, eos, config, device):
    """Main training loop."""
    # Setup training
    optimizer, scheduler, scaler = setup_training(model, config)
    
    # Setup logging
    log_dir = Path(config['output']['log_dir']) / datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir)
    
    # Training history
    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    patience_counter = 0
    
    # Training loop
    num_epochs = config['training']['num_epochs']
    early_stopping_patience = config['training'].get('early_stopping_patience', 30)
    
    print("\nStarting training...")
    for epoch in range(num_epochs):
        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, eos, config, scaler
        )
        
        # Validate
        val_loss = validate(model, val_loader, eos, config)
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Logging
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/validation', val_loss, epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Train Loss: {train_loss:.6e}, Val Loss: {val_loss:.6e}, LR: {current_lr:.3e}")

        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            checkpoint_path = Path(config['output']['checkpoint_dir']) / 'best_model.pth'
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': config
            }, checkpoint_path)
            
            print(f"  → Saved best model (val_loss: {val_loss:.6e})")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= early_stopping_patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    writer.close()
    return history


def evaluate_model(model, eos, config, device, train_dataset):
    """Evaluate the trained model on test data."""
    print("\nGenerating test data...")
    eta, _ = create_physics_objects(device, config)
    
    # Generate test data with different parameters
    C_test, Z_test = setup_initial_state_meshgrid_cold(
        eta, eos,
        N=config['data'].get('n_test', 200),
        device=device,
        lrhomin=-7,
        lrhomax=-3,
        Wmin=1.0,
        Wmax=1.4
    )
    
    # Create solver
    solver = C2P_Solver(
        model, eos,
        train_dataset.C_min, train_dataset.C_max,
        train_dataset.Z_min, train_dataset.Z_max,
        device=device
    )
    
    # Compute metrics
    metrics = compute_metrics(solver, C_test, Z_test, eos)
    
    print("\nTest Results:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.6e}")
    
    # Generate error plots
    if config['output'].get('generate_plots', True):
        plot_error_analysis_with_zeta(
            solver, C_test, Z_test,
            save_fig=True,
            fig_path=Path(config['output']['figure_dir']) / 'error_analysis.png'
        )
    
    return metrics


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Train PINN for Conservative-to-Primitive conversion'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/default.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to checkpoint to resume training'
    )
    parser.add_argument(
        '--analyze-only', 
        action='store_true',
        help='Only run analytical analysis without training')

    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup device
    device = setup_device()
    
    # Create physics objects
    eta, eos = create_physics_objects(device, config)
    
    # Generate data
    C_train, Z_train, C_val, Z_val = generate_data(eta, eos, device, config)
    
    # Create dataloaders
    train_loader, val_loader, train_dataset = create_dataloaders(
        C_train, Z_train, C_val, Z_val, config
    )


    
    if args.analyze_only:
        # NEW: Analytical relationship analysis
        print("\n" + "="*60)
        print("RUNNING ANALYTICAL RELATIONSHIP ANALYSIS")
        print("="*60)

        analysis_results = analyze_analytical_relationship(
            C_train, Z_train, train_dataset,
            save_dir=config['output'].get('figure_dir', 'experiments/figures')
        )

        print("\nAnalysis complete. Exiting without training.")
        return

    # Create model
    model = create_model(device, config)
    
    # Load checkpoint if provided
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    
    # Train model
    history = train(model, train_loader, val_loader, eos, config, device)
    
    # Plot training history
    if config['output'].get('generate_plots', True):
        plot_training_history(
            history,
            save_path=Path(config['output']['figure_dir']) / 'training_history.png'
        )
    
    # Evaluate model
    metrics = evaluate_model(model, eos, config, device, train_dataset)
    
    # Save metrics
    metrics_path = Path(config['output']['checkpoint_dir']) / 'metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print("\nTraining complete!")
    print(f"Results saved to: {config['output']['checkpoint_dir']}")

    # ============================================================
    # Export model for GRACE (HDF5 format)
    # ============================================================
    # Create directory
    model_dir = config['output'].get('model_dir', 'saved_models')
    os.makedirs(model_dir, exist_ok=True)

    # Generate filename
    best_val_loss = min(history['val_loss'])
    h5_filename = os.path.join(
        model_dir,
        f"pinn_c2p_vloss{best_val_loss:.2e}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.h5"
    )

    # Export
    export_physics_guided_model_to_hdf5(
        model=model,
        dataset=train_dataset,
        filename=h5_filename,
        activation_name="tanh"
    )
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"  Checkpoints:  {config['output']['checkpoint_dir']}")
    print(f"  GRACE file:   {h5_filename}")
    print(f"  Metrics:      {metrics_path}")
    print("="*60)


if __name__ == "__main__":
    main()