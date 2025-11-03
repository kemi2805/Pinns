"""
Physics utility functions for conservative-to-primitive variable conversion.
"""

import torch


def W__z(z):
    """Lorentz factor from z parameter."""
    return torch.sqrt(1 + z**2)


def rho__z(z, C):
    """Rest-mass density from z and conservative variables."""
    D = C[:, 0:1]  # Conserved density
    W = W__z(z)
    return D / W


def eps__z(z, C):
    """Specific internal energy from z and conservative variables."""
    q = C[:, 1:2]  # τ/D
    r = C[:, 2:3]  # S/D
    W = W__z(z)
    return W * q - z * r + z**2 / (1 + W)


def a__z(z, C, eos):
    """Specific enthalpy ratio from z and conservative variables."""
    eps = eps__z(z, C)
    rho = rho__z(z, C)
    press = eos.press__eps_rho(eps, rho)
    return press / (rho * (1 + eps))


def h__z(z, C, eos):
    """Specific enthalpy from z and conservative variables."""
    eps = eps__z(z, C)
    a = a__z(z, C, eos)
    return (1 + eps) * (1 + a)


def sanity_check(Z, C, metric, eos):
    """
    Physics consistency check.
    Verifies that Z satisfies the relation Z = S/h̃.
    """
    t, q, r = torch.split(C, [1, 1, 1], dim=1)
    htilde = h__z(Z, C, eos)
    residual = Z - r / htilde
    return torch.mean(residual**2)


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


def conservative_to_primitive_exact(C, eos, tol=1e-10, max_iter=50):
    """
    Exact iterative solver for conservative-to-primitive conversion.
    Uses Newton-Raphson method.
    """
    D = C[:, 0:1]  # Conserved density
    q = C[:, 1:2]  # τ/D
    r = C[:, 2:3]  # S/D
    
    # Initial guess for z
    z = torch.sqrt(r**2)
    
    for iteration in range(max_iter):
        # Compute h̃ and its derivative
        htilde = h__z(z, C, eos)
        
        # Residual: f(z) = z - r/h̃(z)
        f = z - r / htilde
        
        # Check convergence
        if torch.max(torch.abs(f)) < tol:
            break
        
        # Compute derivative (numerical for stability)
        dz = 1e-8
        htilde_plus = h__z(z + dz, C, eos)
        dhtilde_dz = (htilde_plus - htilde) / dz
        
        # Newton update
        df_dz = 1 + r * dhtilde_dz / htilde**2
        z = z - f / df_dz
        
        # Ensure z stays positive
        z = torch.clamp(z, min=0.0)
    
    return z


def validate_conservative_variables(C, eos):
    """
    Check if conservative variables are physically valid.
    
    Returns:
        valid: Boolean tensor indicating valid samples
        messages: List of validation messages
    """
    D = C[:, 0:1]  # Conserved density
    q = C[:, 1:2]  # τ/D
    r = C[:, 2:3]  # S/D
    
    messages = []
    
    # Check D > 0
    valid_D = D > 0
    if not torch.all(valid_D):
        messages.append(f"Invalid D (negative density): {(~valid_D).sum()} samples")
    
    # Check τ > 0 (for most cases)
    tau = q * D
    valid_tau = tau > -D  # τ > -D is the physical bound
    if not torch.all(valid_tau):
        messages.append(f"Invalid τ: {(~valid_tau).sum()} samples")
    
    # Combined validity
    valid = valid_D & valid_tau
    
    return valid, messages