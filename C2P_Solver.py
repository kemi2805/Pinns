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
    
    def invert_analytic(self, C, eos):
        """
        Invert conservative variables to primitive variables using analytic method
        """
        with torch.no_grad():
            # Ensure C is on the same device
            if not isinstance(C, torch.Tensor):
                print("not instance of torch.Tensor, converting...")
                C = torch.tensor(C, device=self.C_min.device, dtype=torch.float64)

            batch_size = C.shape[0]
            
            # Define function for each sample separately to avoid dimension issues
            def func_for_sample(zeta, C_sample):
                zeta = zeta.view(-1, 1)
                C_sample = C_sample.unsqueeze(0)  # Add batch dimension
                return (zeta - C_sample[:, 2:3] / h__z(zeta, C_sample, self.eos)).flatten()

            k = C[:, 2:3]/(1+C[:, 1:2])
            zm = 0.5*k/torch.sqrt(1 - (0.5*k)**2)
            zp = 1e-6 + k/torch.sqrt(1 - (k)**2)

            # Solve for each sample individually
            Z_pred = torch.zeros(batch_size, 1, device=C.device, dtype=C.dtype)
            converged = torch.zeros(batch_size, dtype=torch.bool, device=C.device)
            
            for i in range(batch_size):
                # Create function for this specific sample
                func_i = lambda zeta: func_for_sample(zeta, C[i])
                
                # Solve using Brent's method for this sample
                z_i, conv_i = brent_torch_single(func_i, zm[i, 0], zp[i, 0], tol=1e-12, maxiter=100)
                Z_pred[i, 0] = z_i
                converged[i] = conv_i

            # Check convergence
            non_converged = ~converged
            if non_converged.any():
                print(f"Warning: {non_converged.sum().item()} out of {batch_size} samples did not converge")
                # For non-converged samples, use the midpoint as fallback
                Z_pred[non_converged] = 0.5 * (zm[non_converged] + zp[non_converged])

            # Compute primitive variables from Z
            rho = rho__z(Z_pred, C)
            W = W__z(Z_pred)
            eps = eps__z(Z_pred, C)
            press = eos.press__eps_rho(eps, rho)
        
            return rho, eps, press, W, Z_pred


def brent_torch_single(func, a, b, tol=1e-12, maxiter=100):
    """
    Brent's method root finding for a single scalar function
    
    Args:
        func: Function that takes scalar tensor input and returns scalar tensor output
        a, b: Scalar bounds
        tol: Tolerance for convergence
        maxiter: Maximum iterations
    
    Returns:
        root: Scalar root
        converged: Boolean indicating convergence
    """
    device = a.device
    dtype = a.dtype
    
    # Ensure a and b are scalars
    a = a.clone().detach()
    b = b.clone().detach()
    
    # Evaluate function at bounds
    fa = func(a)
    fb = func(b)
    
    # Check if root is at boundary
    if torch.abs(fa) < tol:
        return a, True
    if torch.abs(fb) < tol:
        return b, True
    
    # Initialize variables
    c = b.clone()
    fc = fb.clone()
    d = b - a
    e = d.clone()
    
    for iteration in range(maxiter):
        # Check convergence
        if torch.abs(fb) < tol:
            return b, True
            
        # Ensure fa and fb have opposite signs
        if torch.sign(fa) == torch.sign(fb):
            c = a.clone()
            fc = fa.clone()
            d = b - a
            e = d.clone()
        
        # Ensure |fa| >= |fb|
        if torch.abs(fa) < torch.abs(fb):
            # Swap a and b
            a, b = b, a
            fa, fb = fb, fa
        
        # Convergence check
        if torch.abs(b - a) < tol:
            return b, True
        
        # Determine interpolation method
        # Default to bisection
        m = 0.5 * (a + c)
        
        # Try inverse quadratic interpolation
        if (torch.abs(fa - fc) > tol) and (torch.abs(fb - fc) > tol):
            # Inverse quadratic interpolation
            s_quad = (a * fb * fc / ((fa - fb) * (fa - fc)) +
                     b * fa * fc / ((fb - fa) * (fb - fc)) +
                     c * fa * fb / ((fc - fa) * (fc - fb)))
            
            # Check if quadratic interpolation is acceptable
            if (torch.abs(s_quad - b) < 0.75 * torch.abs(b - a)) and \
               (torch.abs(s_quad - b) < 0.5 * torch.abs(e)):
                m = s_quad
        
        # Try linear interpolation (secant method)
        elif torch.abs(fa - fb) > tol:
            s_linear = b - fb * (b - a) / (fb - fa)
            
            # Check if linear interpolation is acceptable
            if (torch.abs(s_linear - b) < 0.75 * torch.abs(b - a)) and \
               (torch.abs(s_linear - b) < 0.5 * torch.abs(e)):
                m = s_linear
        
        # Ensure minimum step size
        min_step = tol * torch.abs(b) + tol
        if torch.abs(m - b) < min_step:
            m = b + torch.sign(m - b) * min_step
        
        # Update for next iteration
        c = b.clone()
        fc = fb.clone()
        e = d.clone()
        d = m - b
        
        b = m
        fb = func(b)
        
        # Update a if necessary
        if torch.sign(fa) == torch.sign(fb):
            a = c.clone()
            fa = fc.clone()
    
    # Final convergence check
    final_converged = torch.abs(fb) < tol
    
    return b, final_converged


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_fp64 = True
    dtype = torch.float64 if use_fp64 else torch.float32

    # Minkowski metric 
    eta = metric(
        torch.eye(3, device=device), 
        torch.zeros(3, device=device), 
        torch.ones(1, device=device)
    )
    
    # Gamma = 2 EOS with ideal gas thermal contribution 
    eos = hybrid_eos(100, 2, 1.8)

    C_test = torch.tensor([[1.9e-3, 8.4186e-1, 1.4623]], dtype=torch.float64)
    class FakeModel(nn.Module):
        def __init__(self, input_dim=4, hidden_dim=64, output_dim=1):
            super(FakeModel, self).__init__()
            self.network = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim)
            )

        def forward(self, x):
            return self.network(x)
    fake_model = FakeModel().to(device)
    solver = C2P_Solver(model=fake_model,  # Replace with actual model
                        eos=eos,
                        C_min=torch.tensor([0.0, 0.0, 0.0],device=device),
                        C_max=torch.tensor([1.0, 1.0, 1.0],device=device),
                        Z_min=torch.tensor([0.0],device=device),
                        Z_max=torch.tensor([1.0],device=device),
                        device=device)

    rho, eps, press, W = solver.invert_analytic(C_test, eos)
    print("Density:", rho)
    print("Specific internal energy:", eps)
    print("Pressure:", press)
    print("Lorentz factor:", W)
