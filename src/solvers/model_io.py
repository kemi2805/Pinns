"""
Model saving and loading utilities for PINN C2P solver.
HDF5 export for GRACE C++ integration.
"""

import torch
import h5py
import os
from datetime import datetime


def export_physics_guided_model_to_hdf5(model, dataset, filename, activation_name="tanh"):
    """
    Export PhysicsGuided_TinyPINNvKen model to HDF5 format for GRACE.
    
    Args:
        model: The trained PINN model
        dataset: The C2P_Dataset used for training (contains normalization params)
        filename: Output HDF5 filename
        activation_name: Name of activation function (default: "tanh")
    """
    model.eval()
    
    # Unwrap DataParallel if necessary
    model_to_export = model.module if hasattr(model, 'module') else model
    
    with h5py.File(filename, 'w') as f:
        # Extract weights from correction_net
        state_dict = model_to_export.state_dict()
        
        # Get weights from the correction network layers
        # Layer 0: Linear(3, 64)
        # Layer 1: Tanh()
        # Layer 2: Linear(64, 1)
        weights_ih = state_dict['correction_net.0.weight'].detach().cpu().numpy()
        bias_h = state_dict['correction_net.0.bias'].detach().cpu().numpy()
        weights_ho = state_dict['correction_net.2.weight'].detach().cpu().numpy()
        bias_o = state_dict['correction_net.2.bias'].detach().cpu().numpy()
        
        # Store network parameters
        f.create_dataset('weights_input_to_hidden', data=weights_ih)
        f.create_dataset('bias_hidden', data=bias_h)
        f.create_dataset('weights_hidden_to_output', data=weights_ho)
        f.create_dataset('bias_output', data=bias_o)
        
        # Store metadata
        f.attrs['activation_function'] = activation_name
        f.attrs['input_size'] = weights_ih.shape[1]
        f.attrs['hidden_size'] = weights_ih.shape[0]
        f.attrs['output_size'] = weights_ho.shape[0]
        f.attrs['correction_scale'] = float(model_to_export.correction_scale)
        f.attrs['model_type'] = 'physics_guided'
        
        # Store normalization parameters from dataset
        f.create_dataset('input_min', data=dataset.C_min.cpu().numpy().flatten())
        f.create_dataset('input_max', data=dataset.C_max.cpu().numpy().flatten())
        f.create_dataset('output_min', data=dataset.Z_min.cpu().numpy().flatten())
        f.create_dataset('output_max', data=dataset.Z_max.cpu().numpy().flatten())
        
        # Store additional metadata for debugging/verification
        f.attrs['export_timestamp'] = str(datetime.now().isoformat())
        f.attrs['pytorch_version'] = str(torch.__version__)
        f.attrs['model_class'] = str(model_to_export.__class__.__name__)
    
    print(f"\n{'='*60}")
    print("MODEL EXPORTED TO HDF5 FOR GRACE")
    print(f"{'='*60}")
    print(f"  File: {filename}")
    print(f"  Architecture: {weights_ih.shape[1]} -> {weights_ih.shape[0]} -> {weights_ho.shape[0]}")
    print(f"  Correction scale: {model_to_export.correction_scale}")
    print(f"  Activation: {activation_name}")
    print(f"  Normalization ranges:")
    print(f"    C: [{dataset.C_min.cpu().numpy().flatten()}] to [{dataset.C_max.cpu().numpy().flatten()}]")
    print(f"    Z: [{dataset.Z_min.cpu().numpy().flatten()}] to [{dataset.Z_max.cpu().numpy().flatten()}]")
    print(f"{'='*60}\n")


def save_pinn_model_for_grace(model, dataset, save_dir="saved_models", model_name=None):
    """
    Save model in both PyTorch (.pt) and HDF5 (.h5) formats.
    
    Args:
        model: The trained PINN model
        dataset: The C2P_Dataset used for training
        save_dir: Directory to save the model
        model_name: Optional custom name for the model
    
    Returns:
        tuple: (pt_path, h5_path)
    """
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Generate model name if not provided
    if model_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"pinn_c2p_{timestamp}"
    
    # Save as PyTorch checkpoint (for Python inference/retraining)
    pt_path = os.path.join(save_dir, f"{model_name}.pt")
    model_to_save = model.module if hasattr(model, 'module') else model
    
    torch.save({
        'model_state_dict': model_to_save.state_dict(),
        'model_class': model_to_save.__class__.__name__,
        'correction_scale': model_to_save.correction_scale,
        'normalization': {
            'C_min': dataset.C_min.cpu().numpy(),
            'C_max': dataset.C_max.cpu().numpy(),
            'Z_min': dataset.Z_min.cpu().numpy(),
            'Z_max': dataset.Z_max.cpu().numpy(),
        }
    }, pt_path)
    
    # Save as HDF5 (for GRACE C++ inference)
    h5_path = os.path.join(save_dir, f"{model_name}.h5")
    export_physics_guided_model_to_hdf5(
        model=model,
        dataset=dataset,
        filename=h5_path,
        activation_name="tanh"
    )
    
    print(f"✓ PyTorch checkpoint: {pt_path}")
    print(f"✓ HDF5 for GRACE:     {h5_path}")
    
    return pt_path, h5_path