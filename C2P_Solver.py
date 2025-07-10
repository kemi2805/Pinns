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

def W__z(z):
    """Lorentz factor from z"""
    return torch.sqrt(1 + z**2)

def rho__z(z, C):
    """Density from z and conservative variables"""
    return C[:, 0:1] / W__z(z)

def eps__z(z, C):
    """Specific internal energy from z and conservative variables"""
    q = C[:, 1:2]
    r = C[:, 2:3]
    W = W__z(z)
    return W * q - z * r + z**2 / (1 + W)

def a__z(z, C, eos):
    """Specific enthalpy ratio from z and conservative variables"""
    eps = eps__z(z, C)
    rho = rho__z(z, C)
    press = eos.press__eps_rho(eps, rho)
    return press / (rho * (1 + eps))

def h__z(z, C, eos):
    """Specific enthalpy from z and conservative variables"""
    eps = eps__z(z, C)
    a = a__z(z, C, eos)
    return (1 + eps) * (1 + a)

def sanity_check(Z, C, metric, eos):
    """Physics consistency check"""
    t, q, r = torch.split(C, [1, 1, 1], dim=1)
    htilde = h__z(Z, C, eos)
    return torch.mean((Z - r / htilde)**2)


class C2P_Solver:
    """Conservative-to-primitive variable solver using trained PINN"""
    
    def __init__(self, model, eos, C_min, C_max, Z_min, Z_max, device='cpu'):
        self.model = model

        self.C_min = torch.tensor(C_min, device=device, dtype=torch.float64)
        self.C_max = torch.tensor(C_max, device=device, dtype=torch.float64)
        self.Z_min = torch.tensor(Z_min, device=device, dtype=torch.float64)
        self.Z_max = torch.tensor(Z_max, device=device, dtype=torch.float64)
        self.eos = eos 

        self.model.eval()

        self.device = device

    
    def invert(self, C):
        """
        Invert conservative variables to primitive variables
        """
        with torch.no_grad():
            # Ensure C is on the same device
            if not isinstance(C, torch.Tensor):
                print("not instance of torch.Tensor, converting...")
                C = torch.tensor(C, device=self.C_min.device, dtype=torch.float64)

            # Normalize input
            C_norm = (C - self.C_min) / (self.C_max - self.C_min)
            
            # Predict normalized Z
            Z_norm = self.model(C_norm)
            
            # Denormalize Z
            Z = Z_norm * (self.Z_max - self.Z_min) + self.Z_min
        
            # Compute primitive variables
            rho = rho__z(Z, C)
            W = W__z(Z)
            eps = eps__z(Z, C)
            press = self.eos.press__eps_rho(eps, rho)
            
            # Cold matter (T = 0)
            #T = torch.zeros_like(rho, device=device)
            #press, eps = eos.press_eps__temp_rho(T, rho)

            return rho, eps, press, W
