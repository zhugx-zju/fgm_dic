"""
Data Generation for FNO Training
Generate random E(x) fields and corresponding displacement u(x) via FEM

Three types of logE(x) generators:
1. KL/Fourier: Low-frequency random field (smooth, fast to compute)
2. Piecewise: Segmented constant field (discontinuities, layered materials)
3. GRF: Gaussian Random Field with true correlation structure (statistically rigorous)

All generators use local RNG for reproducibility without global state pollution.
"""
import numpy as np
import torch
from scipy.linalg import cholesky
from fem1d import FEM1DBar


def generate_random_logE_KL(n_node, L, rng, n_modes=10, lengthscale=0.2, sigma=0.5, mean_logE=0.0):
    """
    Generate random logE(x) using truncated KL expansion (low-frequency random field)

    This is a "low-frequency random field approximation" using Fourier basis.
    Good for smooth material distributions. The 'lengthscale' parameter controls
    modal energy decay rate (NOT a strict correlation length like in GRF).

    Parameters:
    -----------
    n_node : int
        Number of nodes
    L : float
        Domain length
    rng : np.random.Generator
        Random number generator (do NOT use np.random.seed inside)
    n_modes : int
        Number of KL modes (higher = more details)
    lengthscale : float
        Modal decay rate (typical: 0.1-0.5)
    sigma : float
        Standard deviation of logE variations
    mean_logE : float
        Mean of logE (e.g., log(1.0) = 0.0)

    Returns:
    --------
    logE : array (n_node,)
        Random logE field
    """
    x = np.linspace(0, L, n_node)
    logE = np.ones(n_node) * mean_logE

    # KL expansion with exponential modal decay
    for k in range(1, n_modes + 1):
        # Eigenvalue decay
        lambda_k = sigma**2 * np.exp(-k * lengthscale)

        # Random coefficient
        xi_k = rng.standard_normal()

        # Basis function (alternating sine/cosine)
        if k % 2 == 1:
            phi_k = np.sqrt(2) * np.sin(k * np.pi * x / L)
        else:
            phi_k = np.sqrt(2) * np.cos(k * np.pi * x / L)

        logE += np.sqrt(lambda_k) * xi_k * phi_k

    return logE


def generate_random_logE_piecewise(n_node, L, rng, n_segments=4, logE_range=(-0.5, 0.5)):
    """
    Generate piecewise constant logE(x)

    Simulates layered/segmented materials with discontinuities.
    Useful for testing robustness to sharp material interfaces.

    Parameters:
    -----------
    n_node : int
        Number of nodes
    L : float
        Domain length
    rng : np.random.Generator
        Random number generator
    n_segments : int
        Number of constant segments
    logE_range : tuple
        (min_logE, max_logE)

    Returns:
    --------
    logE : array (n_node,)
        Piecewise constant logE field
    """
    x = np.linspace(0, L, n_node)
    logE = np.zeros(n_node)

    # Segment boundaries
    segment_boundaries = np.linspace(0, L, n_segments + 1)

    # Random segment values
    segment_values = rng.uniform(logE_range[0], logE_range[1], n_segments)

    # Assign values
    for i in range(n_segments):
        if i < n_segments - 1:
            mask = (x >= segment_boundaries[i]) & (x < segment_boundaries[i + 1])
        else:
            # Last segment includes endpoint
            mask = (x >= segment_boundaries[i]) & (x <= segment_boundaries[i + 1])
        logE[mask] = segment_values[i]

    return logE


def generate_random_logE_GRF(n_node, L, rng, kernel='exp', corr_len=0.2,
                              sigma=0.5, mean_logE=0.0, jitter=1e-8):
    """
    Generate random logE(x) using Gaussian Random Field (GRF)

    This is a statistically rigorous GRF with well-defined correlation structure.
    The correlation length 'corr_len' directly controls spatial correlation scale.

    Recommended as the baseline distribution for continuous material parameters
    that exhibit spatial correlation (e.g., manufacturing variations, geological media).

    Parameters:
    -----------
    n_node : int
        Number of nodes
    L : float
        Domain length
    rng : np.random.Generator
        Random number generator
    kernel : str
        Covariance kernel type:
        - 'exp': Exponential kernel (Ornstein-Uhlenbeck process)
                 K(r) = σ² exp(-r/ℓ), smooth but not differentiable
        - 'rbf' or 'sqexp': Squared exponential / RBF kernel
                 K(r) = σ² exp(-r²/(2ℓ²)), infinitely differentiable (very smooth)
    corr_len : float
        Correlation length scale (relative to L, typically 0.05-0.5)
        Larger values → smoother, more correlated field
        ℓ = corr_len * L (absolute correlation length)
    sigma : float
        Standard deviation of logE (marginal std)
    mean_logE : float
        Mean of logE
    jitter : float
        Numerical stability term added to diagonal (default: 1e-8)

    Returns:
    --------
    logE : array (n_node,)
        GRF sample of logE

    Notes:
    ------
    - For corr_len=0.2, L=1: ~20% of domain length correlation
    - exp kernel: E[∂logE/∂x exists a.e.], rough texture
    - rbf kernel: E[∂ⁿlogE/∂xⁿ exists ∀n], ultra-smooth
    - Cost: O(n_node³) due to Cholesky (acceptable for 1D with n_node~100-500)
    """
    x = np.linspace(0, L, n_node)

    # Absolute correlation length
    ell = corr_len * L

    # Compute pairwise distances
    dx = x[:, None] - x[None, :]  # (n_node, n_node)
    r = np.abs(dx)

    # Build covariance matrix
    if kernel == 'exp':
        # Exponential kernel (Matern-1/2)
        K = sigma**2 * np.exp(-r / ell)
    elif kernel in ['rbf', 'sqexp']:
        # Squared exponential / RBF kernel (Gaussian)
        K = sigma**2 * np.exp(-r**2 / (2 * ell**2))
    else:
        raise ValueError(f"Unknown kernel: {kernel}. Use 'exp' or 'rbf'.")

    # Add jitter for numerical stability
    K += jitter * np.eye(n_node)

    # Cholesky decomposition: K = L L^T
    try:
        L_chol = cholesky(K, lower=True)
    except np.linalg.LinAlgError:
        # Fallback: increase jitter if Cholesky fails
        print(f"Warning: Cholesky failed with jitter={jitter}, increasing to {jitter*100}")
        K += jitter * 99 * np.eye(n_node)
        L_chol = cholesky(K, lower=True)

    # Sample: logE = mean + L @ z, where z ~ N(0, I)
    z = rng.standard_normal(n_node)
    logE = mean_logE + L_chol @ z

    return logE


def generate_dataset(n_samples, n_node, L, A, F_range=(0.01, 0.1),
                     logE_type='GRF', logE_params=None, seed_base=0):
    """
    Generate dataset: (u, x/L, F) -> logE

    Each sample is fully determined by seed_base+i for reproducibility.

    Parameters:
    -----------
    n_samples : int
        Number of samples
    n_node : int
        Number of nodes
    L : float
        Bar length
    A : float
        Cross-sectional area
    F_range : tuple
        (F_min, F_max) for random force sampling
    logE_type : str
        'KL', 'piecewise', or 'GRF'
    logE_params : dict or None
        Parameters for logE generation:
        - KL: {'n_modes': 10, 'lengthscale': 0.2, 'sigma': 0.5, 'mean_logE': 0.0}
        - piecewise: {'n_segments': 4, 'logE_range': (-0.5, 0.5)}
        - GRF: {'kernel': 'exp', 'corr_len': 0.2, 'sigma': 0.5, 'mean_logE': 0.0}
    seed_base : int
        Base seed for reproducibility (sample i uses seed_base + i)

    Returns:
    --------
    data : dict with keys
        'u' : (n_samples, n_node) displacement
        'x' : (n_samples, n_node) normalized coordinates x/L
        'F' : (n_samples,) applied forces
        'logE_true' : (n_samples, n_node) true logE
        'E_true' : (n_samples, n_node) true E
    """
    fem = FEM1DBar(L, A, n_node)
    x_normalized = fem.coord / L

    u_all = np.zeros((n_samples, n_node))
    F_all = np.zeros(n_samples)
    logE_all = np.zeros((n_samples, n_node))
    E_all = np.zeros((n_samples, n_node))

    if logE_params is None:
        # Default parameters for each type
        if logE_type == 'KL':
            logE_params = {'n_modes': 10, 'lengthscale': 0.2, 'sigma': 0.5, 'mean_logE': 0.0}
        elif logE_type == 'piecewise':
            logE_params = {'n_segments': 4, 'logE_range': (-0.5, 0.5)}
        elif logE_type == 'GRF':
            logE_params = {'kernel': 'exp', 'corr_len': 0.2, 'sigma': 0.5, 'mean_logE': 0.0}
        else:
            raise ValueError(f"Unknown logE_type: {logE_type}")

    print(f"Generating {n_samples} samples with logE_type='{logE_type}'...")
    print(f"logE_params: {logE_params}")

    for i in range(n_samples):
        # Create local RNG for this sample (fully reproducible)
        rng_i = np.random.default_rng(seed_base + i)

        # Generate random logE using the same rng
        if logE_type == 'KL':
            logE = generate_random_logE_KL(n_node, L, rng_i, **logE_params)
        elif logE_type == 'piecewise':
            logE = generate_random_logE_piecewise(n_node, L, rng_i, **logE_params)
        elif logE_type == 'GRF':
            logE = generate_random_logE_GRF(n_node, L, rng_i, **logE_params)
        else:
            raise ValueError(f"Unknown logE_type: {logE_type}")

        E = np.exp(logE)

        # Generate random force using the same rng (ensures reproducibility)
        F = rng_i.uniform(F_range[0], F_range[1])

        # FEM forward solve
        u = fem.forward_solve(E, F)

        # Store
        u_all[i] = u
        F_all[i] = F
        logE_all[i] = logE
        E_all[i] = E

        if (i + 1) % 100 == 0:
            print(f"  Generated {i+1}/{n_samples} samples")

    # Replicate x/L for all samples
    x_all = np.tile(x_normalized, (n_samples, 1))

    data = {
        'u': u_all.astype(np.float32),
        'x': x_all.astype(np.float32),
        'F': F_all.astype(np.float32),
        'logE_true': logE_all.astype(np.float32),
        'E_true': E_all.astype(np.float32)
    }

    return data


def save_dataset(data, filename):
    """Save dataset to file"""
    torch.save(data, filename)
    print(f"Dataset saved to {filename}")


def load_dataset(filename):
    """Load dataset from file"""
    data = torch.load(filename)
    print(f"Dataset loaded from {filename}")
    return data


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # Configuration
    config = {
        'n_samples_train': 1000,
        'n_samples_val': 200,
        'n_samples_test': 200,
        'n_node': 101,
        'L': 1.0,
        'A': 1.0,
        'F_range': (0.01, 0.1),
        'logE_type': 'GRF',  # Options: 'KL', 'piecewise', 'GRF'
        'logE_params': {
            # For GRF:
            'kernel': 'exp',      # 'exp' or 'rbf'
            'corr_len': 0.2,      # Correlation length (relative to L)
            'sigma': 0.5,         # Standard deviation
            'mean_logE': 0.0      # Mean of logE

            # For KL (alternative):
            # 'n_modes': 10,
            # 'lengthscale': 0.2,
            # 'sigma': 0.5,
            # 'mean_logE': 0.0

            # For piecewise (alternative):
            # 'n_segments': 4,
            # 'logE_range': (-0.5, 0.5)
        },
        'seed_train': 0,
        'seed_val': 10000,
        'seed_test': 20000
    }

    print("="*80)
    print("Data Generation Configuration")
    print("="*80)
    for key, val in config.items():
        print(f"{key}: {val}")
    print("="*80)

    # Generate training dataset
    print("\n[1/3] Generating training dataset...")
    train_data = generate_dataset(
        n_samples=config['n_samples_train'],
        n_node=config['n_node'],
        L=config['L'],
        A=config['A'],
        F_range=config['F_range'],
        logE_type=config['logE_type'],
        logE_params=config['logE_params'],
        seed_base=config['seed_train']
    )
    save_dataset(train_data, 'train_data.pt')

    # Generate validation dataset
    print("\n[2/3] Generating validation dataset...")
    val_data = generate_dataset(
        n_samples=config['n_samples_val'],
        n_node=config['n_node'],
        L=config['L'],
        A=config['A'],
        F_range=config['F_range'],
        logE_type=config['logE_type'],
        logE_params=config['logE_params'],
        seed_base=config['seed_val']
    )
    save_dataset(val_data, 'val_data.pt')

    # Generate test dataset
    print("\n[3/3] Generating test dataset...")
    test_data = generate_dataset(
        n_samples=config['n_samples_test'],
        n_node=config['n_node'],
        L=config['L'],
        A=config['A'],
        F_range=config['F_range'],
        logE_type=config['logE_type'],
        logE_params=config['logE_params'],
        seed_base=config['seed_test']
    )
    save_dataset(test_data, 'test_data.pt')

    print("\n" + "="*80)
    print("Dataset generation complete!")
    print("="*80)

    # Visualize a few samples
    print("\nGenerating visualization of sample E(x) fields...")
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    x = np.linspace(0, config['L'], config['n_node'])

    for i in range(6):
        axes[i].plot(x, train_data['E_true'][i], 'b-', linewidth=2)
        axes[i].set_xlabel('x')
        axes[i].set_ylabel('E(x)')
        axes[i].set_title(f'Sample {i}: F={train_data["F"][i]:.4f}')
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('sample_E_fields.png', dpi=300, bbox_inches='tight')
    print("Sample visualization saved to sample_E_fields.png")

    # Statistics
    print("\n" + "="*80)
    print("Dataset Statistics")
    print("="*80)
    print(f"E range: [{train_data['E_true'].min():.4f}, {train_data['E_true'].max():.4f}]")
    print(f"logE range: [{train_data['logE_true'].min():.4f}, {train_data['logE_true'].max():.4f}]")
    print(f"u range: [{train_data['u'].min():.6f}, {train_data['u'].max():.6f}]")
    print(f"F range: [{train_data['F'].min():.4f}, {train_data['F'].max():.4f}]")
    print("="*80)
