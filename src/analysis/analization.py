import numpy as np
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
from pathlib import Path
import torch

def analyze_analytical_relationship(C_train, Z_train, train_dataset, save_dir='experiments/analysis'):
    """
    Analyze and find analytical relationships between NORMALIZED C and Z.
    
    Args:
        C_train: Conservative variables [N, 3] - (D, q, r) - RAW
        Z_train: Target Z values [N, 1] - RAW
        train_dataset: C2P_Dataset object with normalization parameters
        save_dir: Directory to save analysis results
    """
    # Get NORMALIZED data (this is what the network sees!)
    C_min = train_dataset.C_min.cpu().numpy()
    C_max = train_dataset.C_max.cpu().numpy()
    Z_min = train_dataset.Z_min.cpu().numpy()
    Z_max = train_dataset.Z_max.cpu().numpy()
    C_cpu = C_train.cpu().numpy()
    Z_cpu = Z_train.cpu().numpy()
    C_norm = (C_cpu - C_min) / (C_max - C_min)
    Z_norm = (Z_cpu - Z_min) / (Z_max - Z_min)

    print("TYPE(Z) = ", type(Z_norm))
    print("TYPE(C) = ", type(C_norm))
    
    # Also keep raw for comparison
    C_raw = C_train.cpu().numpy()
    Z_raw = Z_train.cpu().numpy().flatten()
    
    D_norm = C_norm[:, 0]
    q_norm = C_norm[:, 1]
    r_norm = C_norm[:, 2]
    Z_norm = Z_norm.flatten()

    # DEBUG: Print shapes
    print("SHAPE(D_norm) = ", D_norm.shape)  # Should be (1000000,)
    print("SHAPE(Z_norm) = ", Z_norm.shape)  # Should be (1000000,)
    print("D_norm.ndim = ", D_norm.ndim)     # Should be 1
    print("Z_norm.ndim = ", Z_norm.ndim)     # Should be 1
    
    # Create save directory
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("ANALYTICAL RELATIONSHIP ANALYSIS (NORMALIZED DATA)")
    print("="*60)
    print(f"\nNormalization ranges:")
    print(f"  C_min: {train_dataset.C_min.cpu().numpy()}")
    print(f"  C_max: {train_dataset.C_max.cpu().numpy()}")
    print(f"  Z_min: {train_dataset.Z_min.cpu().numpy()}")
    print(f"  Z_max: {train_dataset.Z_max.cpu().numpy()}")
    print(f"\nNormalized data ranges:")
    print(f"  D_norm: [{D_norm.min():.3f}, {D_norm.max():.3f}]")
    print(f"  q_norm: [{q_norm.min():.3f}, {q_norm.max():.3f}]")
    print(f"  r_norm: [{r_norm.min():.3f}, {r_norm.max():.3f}]")
    print(f"  Z_norm: [{Z_norm.min():.3f}, {Z_norm.max():.3f}]")
    
    # 1. Basic correlations on NORMALIZED data
    print("\n1. CORRELATION ANALYSIS (Normalized):")
    print("-"*60)
    correlations = {
        'D_norm': np.corrcoef(D_norm, Z_norm)[0, 1],
        'q_norm': np.corrcoef(q_norm, Z_norm)[0, 1],
        'r_norm': np.corrcoef(r_norm, Z_norm)[0, 1],
        'sqrt(1+r_norm²)': np.corrcoef(np.sqrt(1 + r_norm**2), Z_norm)[0, 1],
        'q_norm/r_norm': np.corrcoef(q_norm/(r_norm + 1e-12), Z_norm)[0, 1],
        'q_norm*r_norm': np.corrcoef(q_norm * r_norm, Z_norm)[0, 1],
        'q_norm²': np.corrcoef(q_norm**2, Z_norm)[0, 1],
        'r_norm²': np.corrcoef(r_norm**2, Z_norm)[0, 1],
    }
    
    for key, val in sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True):
        print(f"  corr(Z_norm, {key:20s}) = {val:+.6f}")
    
    # 2. Linear fit: Z_norm = a + b*D + c*q + d*r
    print("\n2. LINEAR REGRESSION: Z_norm = a + b*D + c*q + d*r")
    print("-"*60)
    
    reg_linear = LinearRegression()
    reg_linear.fit(C_norm, Z_norm)
    Z_pred_linear = reg_linear.predict(C_norm)
    
    rmse_linear = np.sqrt(np.mean((Z_norm - Z_pred_linear)**2))
    mae_linear = np.mean(np.abs(Z_norm - Z_pred_linear))
    r2_linear = reg_linear.score(C_norm, Z_norm)
    
    print(f"  Z_norm = {reg_linear.intercept_:.6f} + "
          f"{reg_linear.coef_[0]:.6f}*D + "
          f"{reg_linear.coef_[1]:.6f}*q + "
          f"{reg_linear.coef_[2]:.6f}*r")
    print(f"  RMSE: {rmse_linear:.6e}")
    print(f"  MAE:  {mae_linear:.6e}")
    print(f"  R²:   {r2_linear:.6f}")
    
    # 3. Polynomial fitting on normalized data
    print("\n3. POLYNOMIAL FITTING: Z_norm = f(q_norm, r_norm)")
    print("-"*60)
    
    best_poly_degree = None
    best_poly_mae = float('inf')
    best_poly_model = None
    best_poly = None
    
    for degree in [1, 2, 3]:
        poly = PolynomialFeatures(degree=degree, include_bias=True)
        C_poly = poly.fit_transform(C_norm)
        
        reg = LinearRegression()
        reg.fit(C_poly, Z_norm)
        Z_pred_poly = reg.predict(C_poly)
        
        rmse_poly = np.sqrt(np.mean((Z_norm - Z_pred_poly)**2))
        mae_poly = np.mean(np.abs(Z_norm - Z_pred_poly))
        r2_poly = reg.score(C_poly, Z_norm)
        
        print(f"\n  Degree {degree}:")
        print(f"    RMSE: {rmse_poly:.6e}")
        print(f"    MAE:  {mae_poly:.6e}")
        print(f"    R²:   {r2_poly:.6f}")
        
        if mae_poly < best_poly_mae:
            best_poly_mae = mae_poly
            best_poly_degree = degree
            best_poly_model = reg
            best_poly = poly
        
        if degree == 2:  # Show coefficients for quadratic
            feature_names = poly.get_feature_names_out(['D', 'q', 'r'])
            print(f"    Key coefficients:")
            for name, coef in zip(feature_names, reg.coef_):
                if abs(coef) > 0.01:  # Only show significant ones
                    print(f"      {name:10s}: {coef:+.6e}")
            print(f"      intercept: {reg.intercept_:+.6e}")
    
    # 4. Power law on NORMALIZED data: Z_norm = a * q_norm^b
    print("\n4. POWER LAW FITTING: Z_norm = a * q_norm^b (Normalized)")
    print("-"*60)
    
    # Filter out zeros/negatives for power law
    mask = (q_norm > 0.01) & (Z_norm > 0.01)  # Normalized data should be positive
    q_safe = q_norm[mask]
    Z_safe = Z_norm[mask]
    
    def power_law(q, a, b):
        return a * np.power(q, b)
    
    try:
        result = curve_fit(power_law, q_safe, Z_safe, p0=[1.0, 0.5], maxfev=10000)
        popt_power = result[0]  # Just take first element (optimal parameters)
        a_power, b_power = popt_power
        Z_pred_power = power_law(q_norm, *popt_power)
        rmse_power = np.sqrt(np.mean((Z_norm - Z_pred_power)**2))
        mae_power = np.mean(np.abs(Z_norm - Z_pred_power))
        
        print(f"  Z_norm = {a_power:.6f} * q_norm^{b_power:.6f}")
        print(f"  RMSE: {rmse_power:.6e}")
        print(f"  MAE:  {mae_power:.6e}")
    except Exception as e:
        print(f"  Power law fitting failed: {e}")
        a_power, b_power = 1.0, 1.0
        mae_power = float('inf')
    
    # 5. Simple scaled relationships (since normalized data is in [0,1])
    print("\n5. SIMPLE NORMALIZED RELATIONSHIPS:")
    print("-"*60)
    
    # a) Z_norm ≈ q_norm (simple proportionality)
    Z_simple_q = q_norm
    mae_simple_q = np.mean(np.abs(Z_norm - Z_simple_q))
    print(f"  Z_norm = q_norm")
    print(f"    MAE: {mae_simple_q:.6e}")
    
    # b) Z_norm ≈ α*q_norm + β*r_norm
    Z_simple_qr = reg_linear.coef_[1] * q_norm + reg_linear.coef_[2] * r_norm + reg_linear.intercept_
    mae_simple_qr = np.mean(np.abs(Z_norm - Z_simple_qr))
    print(f"\n  Z_norm = {reg_linear.coef_[1]:.6f}*q_norm + {reg_linear.coef_[2]:.6f}*r_norm + {reg_linear.intercept_:.6f}")
    print(f"    MAE: {mae_simple_qr:.6e}")
    
    # c) Weighted average
    alpha_q = np.dot(q_norm, Z_norm) / np.dot(q_norm, q_norm)
    Z_weighted_q = alpha_q * q_norm
    mae_weighted_q = np.mean(np.abs(Z_norm - Z_weighted_q))
    print(f"\n  Z_norm = {alpha_q:.6f}*q_norm (least squares optimal)")
    print(f"    MAE: {mae_weighted_q:.6e}")
    
    # 6. Summary and recommendation
    print("\n" + "="*60)
    print("SUMMARY - Best Analytical Approximations (Normalized):")
    print("="*60)
    
    results = {
        f'Linear: {reg_linear.coef_[1]:.3f}*q + {reg_linear.coef_[2]:.3f}*r + {reg_linear.intercept_:.3f}': mae_linear,
        f'Power Law: {a_power:.3f}*q^{b_power:.3f}': mae_power,
        f'Polynomial (degree {best_poly_degree})': best_poly_mae,
        f'Simple q-proportional: {alpha_q:.3f}*q': mae_weighted_q,
    }
    
    # Convert Z_range to scalar ONCE before the loop
    Z_range = float((train_dataset.Z_max - train_dataset.Z_min).cpu().numpy())
    
    for name, mae in sorted(results.items(), key=lambda x: x[1]):
        print(f"  {name}")
        print(f"    → MAE (normalized): {mae:.6e}")
        # Convert to physical units for comparison
        mae_physical = mae * Z_range
        print(f"    → MAE (physical):   {mae_physical:.6e}")
    
    print("\n" + "="*60)
    print("RECOMMENDED for analytical_guess():")
    print("="*60)
    best_formula = min(results.items(), key=lambda x: x[1])
    print(f"\n  Use: {best_formula[0]}")
    print(f"  Expected MAE (normalized): {best_formula[1]:.6e}")
    print(f"  Expected MAE (physical):   {best_formula[1] * Z_range:.6e}")    
    # 7. Generate visualization
    _plot_normalized_fits(C_norm, Z_norm, save_path,
                         reg_linear, a_power, b_power, alpha_q,
                         best_poly, best_poly_model, Z_raw)
    
    # 8. Generate code snippet with NORMALIZATION
    _generate_normalized_code_snippet(
        reg_linear, a_power, b_power, alpha_q,
        train_dataset, save_path
    )
    
    return {
        'linear': (reg_linear.coef_, reg_linear.intercept_, mae_linear),
        'power_law': (a_power, b_power, mae_power),
        'simple_proportional': (alpha_q, mae_weighted_q),
        'polynomial': (best_poly_degree, best_poly_mae)
    }


def _plot_normalized_fits(C_norm, Z_norm, save_path, reg_linear, 
                         a_power, b_power, alpha_q, best_poly, best_poly_model, Z_raw):
    """Plot comparison of different analytical fits on normalized data."""
    
    q_norm = C_norm[:, 1]
    r_norm = C_norm[:, 2]
    
    # Compute predictions
    Z_pred_linear = reg_linear.predict(C_norm)
    Z_pred_power = a_power * np.power(q_norm, b_power)
    Z_pred_simple = alpha_q * q_norm
    
    if best_poly is not None:
        C_poly = best_poly.transform(C_norm)
        Z_pred_poly = best_poly_model.predict(C_poly)
    else:
        Z_pred_poly = Z_pred_linear
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    models = [
        ('Linear Regression', Z_norm, Z_pred_linear),
        (f'Power Law: {a_power:.3f}*q^{b_power:.3f}', Z_norm, Z_pred_power),
        ('Polynomial (best)', Z_norm, Z_pred_poly),
        (f'Simple: {alpha_q:.3f}*q', Z_norm, Z_pred_simple),
    ]
    
    for idx, (name, true, pred) in enumerate(models, 1):
        ax = axes.flatten()[idx]
        
        # Scatter plot
        ax.scatter(true, pred, alpha=0.3, s=1)
        
        # Perfect prediction line
        min_val = 0
        max_val = 1
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect')
        
        # Compute metrics
        mae = np.mean(np.abs(true - pred))
        rmse = np.sqrt(np.mean((true - pred)**2))
        r2 = 1 - np.sum((true - pred)**2) / np.sum((true - true.mean())**2)
        
        ax.set_xlabel('True Z (normalized)', fontsize=12)
        ax.set_ylabel('Predicted Z (normalized)', fontsize=12)
        ax.set_title(f'{name}\nMAE={mae:.3e}, R²={r2:.4f}', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
    
    # Error distributions
    ax = axes.flatten()[0]
    errors = [
        ('Linear', Z_norm - Z_pred_linear),
        ('Power', Z_norm - Z_pred_power),
        ('Poly', Z_norm - Z_pred_poly),
        ('Simple', Z_norm - Z_pred_simple),
    ]
    
    for name, err in errors:
        ax.hist(err, bins=50, alpha=0.5, label=f'{name} (σ={np.std(err):.3e})')
    
    ax.set_xlabel('Error (True - Predicted) [Normalized]', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Error Distributions (Normalized Space)', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Show distribution of Z values (both normalized and raw)
    ax = axes.flatten()[5]
    ax.hist(Z_norm, bins=50, alpha=0.7, label='Z normalized', color='blue')
    ax2 = ax.twinx()
    ax2.hist(Z_raw, bins=50, alpha=0.7, label='Z raw', color='orange')
    ax.set_xlabel('Z value', fontsize=12)
    ax.set_ylabel('Frequency (normalized)', fontsize=12, color='blue')
    ax2.set_ylabel('Frequency (raw)', fontsize=12, color='orange')
    ax.set_title('Z Distribution', fontsize=12)
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path / 'normalized_analytical_fits.png', dpi=300, bbox_inches='tight')
    print(f"\n  → Saved visualization to {save_path / 'normalized_analytical_fits.png'}")
    plt.close()


def _generate_normalized_code_snippet(reg_linear, a_power, b_power, alpha_q,
                                      train_dataset, save_path):
    """Generate code snippets that work with NORMALIZED data."""
    
    C_min = train_dataset.C_min.cpu().numpy()
    C_max = train_dataset.C_max.cpu().numpy()
    Z_min = train_dataset.Z_min.cpu().numpy()
    Z_max = train_dataset.Z_max.cpu().numpy()
    
    coef = reg_linear.coef_
    intercept = reg_linear.intercept_
    
    code = f"""
# ============================================================
# ANALYTICAL GUESS IMPLEMENTATIONS (NORMALIZED DATA)
# Generated from normalized data analysis
# ============================================================

# Normalization parameters (from training data):
C_min = torch.tensor({C_min.tolist()}, dtype=torch.float64)
C_max = torch.tensor({C_max.tolist()}, dtype=torch.float64)
Z_min = torch.tensor({Z_min.tolist()}, dtype=torch.float64)
Z_max = torch.tensor({Z_max.tolist()}, dtype=torch.float64)

# Option 1: Simple Linear Regression (Best for normalized space)
def analytical_guess_linear(self, C):
    \"\"\"Linear regression on normalized inputs.\"\"\"
    # Normalize C
    C_norm = (C - C_min) / (C_max - C_min + 1e-12)
    
    # Linear prediction in normalized space
    Z_norm = {intercept:.6f} + \\
             {coef[0]:.6f} * C_norm[:, 0:1] + \\
             {coef[1]:.6f} * C_norm[:, 1:2] + \\
             {coef[2]:.6f} * C_norm[:, 2:3]
    
    # Clamp to [0, 1] in normalized space
    Z_norm = torch.clamp(Z_norm, min=0.0, max=1.0)
    
    # Denormalize (network expects normalized output, so skip this!)
    # Z = Z_norm * (Z_max - Z_min) + Z_min
    
    return Z_norm  # Return normalized!

# Option 2: Power Law on normalized q
def analytical_guess_power_norm(self, C):
    \"\"\"Power law on normalized q variable.\"\"\"
    # Normalize C
    C_norm = (C - C_min) / (C_max - C_min + 1e-12)
    q_norm = C_norm[:, 1:2]
    
    # Power law in normalized space
    Z_norm = {a_power:.6f} * torch.pow(q_norm, {b_power:.6f})
    Z_norm = torch.clamp(Z_norm, min=0.0, max=1.0)
    
    return Z_norm  # Return normalized!

# Option 3: Simple proportional (simplest!)
def analytical_guess_simple(self, C):
    \"\"\"Simple proportionality: Z_norm ~ q_norm.\"\"\"
    # Normalize C
    C_norm = (C - C_min) / (C_max - C_min + 1e-12)
    q_norm = C_norm[:, 1:2]
    
    # Simple scaling
    Z_norm = {alpha_q:.6f} * q_norm
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
"""
    
    with open(save_path / 'normalized_analytical_guess_code.py', 'w') as f:
        f.write(code)
    
    print(f"  → Saved code snippets to {save_path / 'normalized_analytical_guess_code.py'}")

