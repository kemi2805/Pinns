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
        self.hidden_dims = [64, 64, 64, 32]
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
        # Your conservative variables are [D, τ/D, S/D]
        # The main physics constraint is the sanity check: Z = S/h̃
        
        # Convert back to physical conservative variables
        D = C[:, 0:1]  # Conserved density
        q = C[:, 1:2]  # τ/D 
        r = C[:, 2:3]  # S/D
        
        # Compute specific enthalpy from Z prediction
        try:
            htilde = h__z(Z_pred, C, eos)
            # Physics constraint: Z should equal r/htilde
            physics_residual = Z_pred - r / htilde
            return torch.mean(physics_residual**2)
        except:
            # If physics calculation fails, return zero loss
            return torch.tensor(0.0, device=C.device, dtype=C.dtype)


# Advanced PINN Implementation
# All operations use PyTorch tensors to maintain gradient flow

# =============================================================================
# FOURIER FEATURES (RECOMMENDED FOR YOUR PROBLEM)
# =============================================================================

class FourierPINN(nn.Module):
    """
    Fourier feature PINN with all tensor operations
    Best for complex rootfinding problems like C2P
    """
    def __init__(self, num_fourier_features=128, fourier_sigma=10.0, physics_weight=1.0, dtype=torch.float64):
        super().__init__()
        
        # Store constants as tensors
        self.register_buffer('two_pi', torch.tensor(2.0 * torch.pi, dtype=dtype))
        
        # Random Fourier feature matrix (fixed during training)
        fourier_matrix = torch.randn(3, num_fourier_features, dtype=dtype) * fourier_sigma
        self.register_buffer('fourier_B', fourier_matrix)
        self.physics_weight = physics_weight
        
        # Main network processes Fourier features
        fourier_dim = 2 * num_fourier_features  # sin + cos
        
        self.network = nn.Sequential(
            nn.Linear(fourier_dim, 512, dtype=dtype),
            nn.SiLU(),
            nn.Linear(512, 512, dtype=dtype), 
            nn.SiLU(),
            nn.Linear(512, 256, dtype=dtype),
            nn.SiLU(),
            nn.Linear(256, 128, dtype=dtype),
            nn.SiLU(),
            nn.Linear(128, 1, dtype=dtype),
            nn.Sigmoid()
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # Xavier initialization scaled for SiLU
            nn.init.xavier_uniform_(m.weight, gain=1.0)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        # Fourier feature mapping: [sin(2πBx), cos(2πBx)]
        x_proj = self.two_pi * (x @ self.fourier_B)  # All tensor operations
        fourier_features = torch.cat([
            torch.sin(x_proj), 
            torch.cos(x_proj)
        ], dim=-1)
        
        return self.network(fourier_features)

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
# MULTI-SCALE NETWORK
# =============================================================================

class MultiScalePINN(nn.Module):
    """
    Multi-scale network with tensor-safe operations
    """
    def __init__(self, dtype=torch.float64):
        super().__init__()
        # Input embedding
        self.input_embed = nn.Linear(3, 128, dtype=dtype)
        
        # Multiple scale pathways
        self.scale1_layers = nn.ModuleList([
            nn.Linear(128, 128, dtype=dtype) for _ in range(3)
        ])
        
        self.scale2_layers = nn.ModuleList([
            nn.Linear(128, 64, dtype=dtype),
            nn.Linear(64, 128, dtype=dtype)
        ])
        
        self.scale3_layers = nn.ModuleList([
            nn.Linear(128, 256, dtype=dtype),
            nn.Linear(256, 128, dtype=dtype)
        ])
        
        # Attention for scale combination - FIXED
        self.scale_attention = nn.Sequential(
            nn.Linear(128, 64, dtype=dtype),
            nn.SiLU(),
            nn.Linear(64, 3, dtype=dtype),
            nn.Softmax(dim=-1)
        )
        
        # Output with skip connection - FIXED
        self.output_layer = nn.Sequential(
            nn.Linear(128 + 3, 64, dtype=dtype),  # +3 for input skip
            nn.SiLU(),
            nn.Linear(64, 1, dtype=dtype),
            nn.Sigmoid()
        )
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        identity = x
        
        # Input embedding - FIXED
        h = F.silu(self.input_embed(x))
        
        # Scale 1: Standard pathway - FIXED
        h1 = h
        for layer in self.scale1_layers:
            h1 = F.silu(layer(h1))
        h1 = h1 + h  # Residual
        
        # Scale 2: Compressed pathway - FIXED
        h2 = h
        for layer in self.scale2_layers:
            h2 = F.silu(layer(h2))
        h2 = h2 + h  # Residual
        
        # Scale 3: Expanded pathway - FIXED
        h3 = h
        for layer in self.scale3_layers:
            h3 = F.silu(layer(h3))
        h3 = h3 + h  # Residual
        
        # Attention-based combination
        attention_weights = self.scale_attention(h)  # [batch, 3]
        combined = (attention_weights[:, 0:1] * h1 +
                   attention_weights[:, 1:2] * h2 +
                   attention_weights[:, 2:3] * h3)
        
        # Output with global skip connection
        output_input = torch.cat([combined, identity], dim=-1)
        return self.output_layer(output_input)
    
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
# RESIDUAL NETWORK (ENHANCED VERSION OF YOUR CURRENT)
# =============================================================================

class DeepResidualPINN(nn.Module):
    """
    Deep residual network - enhanced version of your current architecture
    """
    def __init__(self, depth=8, width=128, dtype=torch.float64):
        super().__init__()
        
        # Input projection
        self.input_layer = nn.Linear(3, width, dtype=dtype)
        
        # Deep residual blocks
        self.residual_blocks = nn.ModuleList()
        for _ in range(depth):
            block = nn.ModuleList([
                nn.Linear(width, width, dtype=dtype),
                nn.Linear(width, width, dtype=dtype)
            ])
            self.residual_blocks.append(block)
        
        # Output with multiple skip connections
        self.output_layer = nn.Sequential(
            nn.Linear(width + 3, width // 2, dtype=dtype),  # Global skip
            nn.SiLU(),
            nn.Linear(width // 2, 1, dtype=dtype),
            nn.Sigmoid()
        )
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # He initialization for SiLU activation
            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        identity = x
        
        # Input embedding - FIXED
        h = F.silu(self.input_layer(x))
        
        # Deep residual processing
        for block in self.residual_blocks:
            # Pre-activation residual block
            residual = h
            h = F.silu(block[0](h))  # FIXED
            h = block[1](h)
            h = h + residual  # Residual connection
            h = F.silu(h)  # Post-activation - FIXED
        
        # Global skip connection to output
        output_input = torch.cat([h, identity], dim=-1)
        return self.output_layer(output_input)

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