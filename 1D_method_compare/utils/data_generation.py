import numpy as np

def generate_E_true(distribution_type, coord, L, E0=1.0):
    if distribution_type == 'sinusoidal':
        k = 5
        a = 2 * np.pi * k / L
        E_true = E0 * (2 + np.sin(a * coord))
    elif distribution_type == 'linear':
        E_true = E0 * (1 + coord / L)
    elif distribution_type == 'quadratic':
        E_true = E0 * (1 + (coord / L)**2)
    elif distribution_type == 'heaviside':
        r_coord = coord / L
        E_true = np.empty_like(coord, dtype=float)
        E_true[r_coord < 0.25] = 2.00 * E0
        E_true[(r_coord >= 0.25) & (r_coord < 0.50)] = 1.75 * E0
        E_true[(r_coord >= 0.50) & (r_coord < 0.75)] = 1.50 * E0
        E_true[r_coord >= 0.75] = 1.25 * E0
    else:
        raise ValueError("distribution_type must be 'sinusoidal', 'linear', 'quadratic', or 'heaviside'.")
    return E_true

def generate_strain_data(distribution_type, coord, nel, nstrain, P, A, E0, L):
    x_strain = np.linspace(0, L, (nstrain - 1) * nel + 1)
    if distribution_type == 'sinusoidal':
        k = 5
        strain_exact = P / (2 + np.sin(2 * np.pi * k * x_strain / L)) / A / E0
    elif distribution_type == 'linear':
        strain_exact = P / (1 + x_strain / L) / A / E0
    elif distribution_type == 'quadratic':
        strain_exact = P / (1 + (x_strain / L) ** 2) / A / E0
    elif distribution_type == 'heaviside':
        r_strain = x_strain / L
        denom = np.empty_like(x_strain, dtype=float)
        denom[r_strain < 0.25] = 2.00
        denom[(r_strain >= 0.25) & (r_strain < 0.50)] = 1.75
        denom[(r_strain >= 0.50) & (r_strain < 0.75)] = 1.50
        denom[r_strain >= 0.75] = 1.25
        strain_exact = P / denom / A / E0
    else:
        raise ValueError("distribution_type must be 'sinusoidal', 'linear', 'quadratic', or 'heaviside'.")
    
    return strain_exact

def generate_displacement_data(asm, E_true):
    u_exact = asm.forward_solve(E_true)
    return u_exact

def generate_noise_data(org_data, noise_level=1, seed=4):
    # noise level = ||noise_data - org_data||_2 / ||org_data||_2 * 100%
    denom = np.linalg.norm(org_data)
    # r = noise_level / 100
    r = noise_level / 100.0
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(org_data.shape)
    z_norm = np.linalg.norm(z)
    # alpha = r * ||org_data||_2 / ||z||_2
    alpha = r * denom / z_norm
    # e = alpha * z
    e = alpha * z
    noise_data = org_data + e
    return noise_data
