
# ============================================================
# ANALYTICAL GUESS IMPLEMENTATIONS (NORMALIZED DATA)
# Generated from normalized data analysis
# ============================================================

# Normalization parameters (from training data):
C_min = torch.tensor([[9.999999999999999e-09, 9.999999999878992e-07, 0.0]], dtype=torch.float64)
C_max = torch.tensor([[0.0029928934724533186, 0.9655612068260717, 1.564188205671296]], dtype=torch.float64)
Z_min = torch.tensor([[0.0]], dtype=torch.float64)
Z_max = torch.tensor([[1.118033988749895]], dtype=torch.float64)

# Option 1: Simple Linear Regression (Best for normalized space)
def analytical_guess_linear(self, C):
    """Linear regression on normalized inputs."""
    # Normalize C
    C_norm = (C - C_min) / (C_max - C_min + 1e-12)
    
    # Linear prediction in normalized space
    Z_norm = 0.019822 + \
             -0.313205 * C_norm[:, 0:1] + \
             0.026544 * C_norm[:, 1:2] + \
             1.341895 * C_norm[:, 2:3]
    
    # Clamp to [0, 1] in normalized space
    Z_norm = torch.clamp(Z_norm, min=0.0, max=1.0)
    
    # Denormalize (network expects normalized output, so skip this!)
    # Z = Z_norm * (Z_max - Z_min) + Z_min
    
    return Z_norm  # Return normalized!

# Option 2: Power Law on normalized q
def analytical_guess_power_norm(self, C):
    """Power law on normalized q variable."""
    # Normalize C
    C_norm = (C - C_min) / (C_max - C_min + 1e-12)
    q_norm = C_norm[:, 1:2]
    
    # Power law in normalized space
    Z_norm = 1.272109 * torch.pow(q_norm, 0.505202)
    Z_norm = torch.clamp(Z_norm, min=0.0, max=1.0)
    
    return Z_norm  # Return normalized!

# Option 3: Simple proportional (simplest!)
def analytical_guess_simple(self, C):
    """Simple proportionality: Z_norm ~ q_norm."""
    # Normalize C
    C_norm = (C - C_min) / (C_max - C_min + 1e-12)
    q_norm = C_norm[:, 1:2]
    
    # Simple scaling
    Z_norm = 1.981300 * q_norm
    Z_norm = torch.clamp(Z_norm, min=0.0, max=1.0)
    
    return Z_norm  # Return normalized!

# ============================================================
# CRITICAL: Your forward() must match the normalization!
# ============================================================
def forward(self, x):
    # x is ALREADY normalized by the DataLoader!
    z_baseline = self.analytical_guess(x)  # Returns normalized
    correction = self.correction_net(x)
    return z_baseline + self.correction_scale * correction  # All normalized

# ============================================================
# USAGE: Replace analytical_guess() in your PhysicsGuided_TinyPINNvKen
# ============================================================
