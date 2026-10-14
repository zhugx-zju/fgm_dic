"""
Utility functions for FGM solvers.
"""

import numpy as np
from .results_io import format_noise_tag, get_noise_output_dir, save_inverse_results as save_results


def add_noise_to_displacement(U, noise_level=0.00, seed=42):
    """
    Add Gaussian noise to displacement data.

    Args:
        U: Clean displacement vector [n_dof]
        noise_level: Noise level as fraction of signal magnitude
        seed: Random seed for reproducibility

    Returns:
        U_noisy: Noisy displacement vector
    """
    if noise_level > 0:
        np.random.seed(seed)
        noise_amp = noise_level * np.abs(U)
        U_noisy = U + noise_amp * np.random.randn(len(U))
    else:
        U_noisy = U.copy()

    return U_noisy


def compute_errors(E_true, E_reconstructed):
    """
    Compute error metrics between true and reconstructed modulus.

    Args:
        E_true: True modulus field
        E_reconstructed: Reconstructed modulus field

    Returns:
        errors: Dictionary with error metrics
    """
    rel_error = 100 * np.abs(E_true - E_reconstructed) / E_true
    mae = np.mean(rel_error)
    max_err = np.max(rel_error)
    rmse = np.sqrt(np.mean((E_true - E_reconstructed)**2))

    errors = {
        'mae': mae,
        'max_error': max_err,
        'rmse': rmse,
        'rel_error_field': rel_error
    }

    return errors


def summarize_x_only_reconstruction(mesh_info, E_true, E_reconstructed):
    """Compare an unconstrained 2D reconstruction with an x-only synthetic truth.

    The mesh stores nodes row-by-row in y, with x varying within each row.
    The transverse spread is measured around the reconstructed y-mean, so it
    remains distinct from the error of that mean relative to the true E(x).
    """
    expected_shape = (mesh_info.nods_y, mesh_info.nods_x)
    true_field = np.asarray(E_true, dtype=float)
    reconstructed = np.asarray(E_reconstructed, dtype=float)
    if true_field.shape != expected_shape:
        raise ValueError(f"E_true must have shape {expected_shape}")
    if reconstructed.shape == (mesh_info.n_nod,):
        reconstructed = reconstructed.reshape(expected_shape)
    elif reconstructed.shape != expected_shape:
        raise ValueError(f"E_reconstructed must have shape {expected_shape} or ({mesh_info.n_nod},)")
    if not np.all(np.isfinite(true_field)) or not np.all(np.isfinite(reconstructed)):
        raise ValueError("modulus fields must be finite")
    if np.any(true_field <= 0):
        raise ValueError("E_true must be positive")
    if not np.allclose(true_field, true_field[0:1], rtol=1e-12, atol=1e-12):
        raise ValueError("synthetic truth must depend on x only")

    true_profile = true_field[0].copy()
    mean_profile = reconstructed.mean(axis=0)
    minimum_profile = reconstructed.min(axis=0)
    maximum_profile = reconstructed.max(axis=0)
    transverse_deviation = reconstructed - mean_profile[None, :]

    return {
        "x_mm": np.asarray(mesh_info.plot_x[0], dtype=float).copy(),
        "true_profile": true_profile,
        "reconstructed_mean": mean_profile,
        "reconstructed_min": minimum_profile,
        "reconstructed_max": maximum_profile,
        "profile_mae_pct": float(100 * np.mean(np.abs(mean_profile - true_profile) / true_profile)),
        "transverse_rms_pct": float(
            100 * np.sqrt(np.mean(transverse_deviation**2)) / np.mean(true_profile)
        ),
        "transverse_max_range_pct": float(
            100 * np.max((maximum_profile - minimum_profile) / true_profile)
        ),
    }
