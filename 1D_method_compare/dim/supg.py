import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
import os

# =========================
# ParaSet
# =========================
nel = 20
nNod = nel + 1
distribution_type = 'heaviside'
nstrain = 20
P = 1.0
A = 1.0
L = 1.0
E0 = 1.0

# SUPG parameter (0 -> Bubnov)
# tau has unit of length; a practical starting scale is Le/2 ~ L/(2*nel)
tau0 = 0.5 * (L / nel)

coord = np.linspace(0, L, nNod)
x_strain = np.linspace(0, L, (nstrain - 1) * nel + 1)

# =========================
# Strain data
# =========================
if distribution_type == 'sinusoidal':
    k = 0.5
    E_true = E0 * (2 + np.sin(2 * np.pi * k * coord / L))
    strain = P / (2 + np.sin(2 * np.pi * k * x_strain / L)) / A / E0
elif distribution_type == 'linear':
    E_true = E0 * (1 + coord / L)
    strain = P / (1 + x_strain / L) / A / E0
elif distribution_type == 'quadratic':
    E_true = E0 * (1 + (coord / L) ** 2)
    strain = P / (1 + (x_strain / L) ** 2) / A / E0
elif distribution_type == 'heaviside':
    # ASME JAM 2024, Eq. (28): 4-step (Heaviside) modulus distribution
    r_coord = coord / L
    E_true = np.empty_like(coord, dtype=float)
    E_true[r_coord < 0.25] = 2.00 * E0
    E_true[(r_coord >= 0.25) & (r_coord < 0.50)] = 1.75 * E0
    E_true[(r_coord >= 0.50) & (r_coord < 0.75)] = 1.50 * E0
    E_true[r_coord >= 0.75] = 1.25 * E0

    r_strain = x_strain / L
    denom = np.empty_like(x_strain, dtype=float)
    denom[r_strain < 0.25] = 2.00
    denom[(r_strain >= 0.25) & (r_strain < 0.50)] = 1.75
    denom[(r_strain >= 0.50) & (r_strain < 0.75)] = 1.50
    denom[r_strain >= 0.75] = 1.25

    strain = P / denom / A / E0
else:
    raise ValueError("distribution_type must be 'sinusoidal', 'linear', 'quadratic', or 'heaviside'.")


# =========================
# FEM info
# =========================
F = np.zeros(nNod)
F[0] = -P

meshnodes = np.arange(1, nel + 1).reshape(-1, 1)
eleNodesID = meshnodes + np.array([0, 1]) - 1  # 0-based
strainlist = np.arange(0, 1 + (nstrain - 1) * (nel - 1), nstrain - 1)
IDlist = np.arange(0, nstrain)
StrainID = strainlist[:, np.newaxis] + IDlist
eleStrainID = StrainID.T.flatten()

ele_strain = strain[eleStrainID].reshape(nstrain, nel).T
ele_point = x_strain[eleStrainID].reshape(nstrain, nel).T

xEle = coord[eleNodesID]
zeta = (ele_point - xEle[:, [0]]) * nel / L  # local coord in [0,1] (approx)

# assemble index pairs for global C
si = np.array([0, 0, 1, 1])
sii = np.array([0, 1, 0, 1])
ik = eleNodesID[:, si].T
jk = eleNodesID[:, sii].T
Ivar = np.column_stack([ik.flatten(), jk.flatten()])

# =========================
# Bubnov weights
# =========================
W11 = np.zeros((nel, nstrain))
W11[:, 0] = (zeta[:, 1] - zeta[:, 0]) * (zeta[:, 0] - 1) / 2
W11[:, -1] = (zeta[:, -1] - zeta[:, -2]) * (zeta[:, -1] - 1) / 2
W11[:, 1:-1] = (zeta[:, 2:] - zeta[:, :-2]) * (zeta[:, 1:-1] - 1) / 2
W11 = A * W11

W12 = np.zeros((nel, nstrain))
W12[:, 0] = (zeta[:, 0] - zeta[:, 1]) * zeta[:, 0] / 2
W12[:, -1] = (zeta[:, -2] - zeta[:, -1]) * zeta[:, -1] / 2
W12[:, 1:-1] = (zeta[:, :-2] - zeta[:, 2:]) * zeta[:, 1:-1] / 2
W12 = A * W12

W21 = np.zeros((nel, nstrain))
W21[:, 0] = (zeta[:, 1] - zeta[:, 0]) * (1 - zeta[:, 0]) / 2
W21[:, -1] = (zeta[:, -1] - zeta[:, -2]) * (1 - zeta[:, -1]) / 2
W21[:, 1:-1] = (zeta[:, 2:] - zeta[:, :-2]) * (1 - zeta[:, 1:-1]) / 2
W21 = A * W21

W22 = np.zeros((nel, nstrain))
W22[:, 0] = (zeta[:, 1] - zeta[:, 0]) * zeta[:, 0] / 2
W22[:, -1] = (zeta[:, -1] - zeta[:, -2]) * zeta[:, -1] / 2
W22[:, 1:-1] = (zeta[:, 2:] - zeta[:, :-2]) * zeta[:, 1:-1] / 2
W22 = A * W22

B11 = (ele_strain * W11).sum(axis=1)
B12 = (ele_strain * W12).sum(axis=1)
B21 = (ele_strain * W21).sum(axis=1)
B22 = (ele_strain * W22).sum(axis=1)

# =========================
# SUPG stabilization term
# =========================
# In 1D with linear elements:
# dN1/dx = -1/Le, dN2/dx = +1/Le (constant in element)
# SUPG test: w = N_i - tau * dN_i/dx
# This adds a term that is proportional to tau and tends to damp oscillations.
Le = L / nel
# For linear 1D element on [x_i, x_{i+1}]
# phi1' = -1/Le , phi2' = +1/Le
phi_prime = np.array([-1.0 / Le, 1.0 / Le])  # [phi1', phi2']
# --- elementwise strain gradient epsilon' ---
# paper: epsilon' needed; compute average gradient per element by linear regression
# (same idea as paper discussion: fit straight line to data within each element)
eps_prime = np.zeros(nel)
for e in range(nel):
    x_e = ele_point[e, :]
    eps_e = ele_strain[e, :]
    x_mean = x_e.mean()
    denom = np.sum((x_e - x_mean) ** 2)
    if denom < 1e-30:
        eps_prime[e] = 0.0
    else:
        eps_prime[e] = np.sum((x_e - x_mean) * (eps_e - eps_e.mean())) / denom
# --- stabilization parameter tau_e ---
# paper uses tau_e = h_e / (2|eps|) with "velocity" = -eps.
# Here eps varies inside element; use element-average |eps|.
eps_abs_mean = np.mean(np.abs(ele_strain), axis=1)
eps_abs_mean = np.maximum(eps_abs_mean, 1e-12)
tau_e = (Le / 2.0) / eps_abs_mean  # shape (nel,)
# NOTE: if you want to force same tau as your tau0, uncomment:
# tau_e = tau0 * np.ones(nel)
# --- trapezoidal integration weights for generic integrals over element ---
# We need weights for ∫ g(x) dx. For points x_k in element, trapezoid weights are:
# w0 = (x1-x0)/2, wN = (xN-x_{N-1})/2, wk = (x_{k+1}-x_{k-1})/2
dx_e = np.zeros((nel, nstrain))
dx_e[:, 0] = (ele_point[:, 1] - ele_point[:, 0]) / 2.0
dx_e[:, -1] = (ele_point[:, -1] - ele_point[:, -2]) / 2.0
dx_e[:, 1:-1] = (ele_point[:, 2:] - ele_point[:, :-2]) / 2.0
# --- shape functions evaluated at measurement points (in natural coord zeta ~ xi in [0,1]) ---
# phi1 = 1 - zeta, phi2 = zeta
phi1_val = 1.0 - zeta
phi2_val = zeta
# --- term 2: ∫ tau * eps^2 * phi_i' * phi_j' * a dx ---
# since phi_i' and phi_j' constant, pull them out
int_tau_eps2_a = (tau_e[:, None] * (ele_strain ** 2) * A * dx_e).sum(axis=1)  # (nel,)
B11_supg2 = (phi_prime[0] * phi_prime[0]) * int_tau_eps2_a
B12_supg2 = (phi_prime[0] * phi_prime[1]) * int_tau_eps2_a
B21_supg2 = (phi_prime[1] * phi_prime[0]) * int_tau_eps2_a
B22_supg2 = (phi_prime[1] * phi_prime[1]) * int_tau_eps2_a
# --- term 3: ∫ tau * eps * eps' * phi_i' * phi_j * a dx ---
# eps' is taken as element-average eps_prime[e], constant in element
int_tau_eps_epsp_a_phi1 = (tau_e[:, None] * ele_strain * eps_prime[:, None] * A * phi1_val * dx_e).sum(axis=1)
int_tau_eps_epsp_a_phi2 = (tau_e[:, None] * ele_strain * eps_prime[:, None] * A * phi2_val * dx_e).sum(axis=1)
B11_supg3 = phi_prime[0] * int_tau_eps_epsp_a_phi1
B12_supg3 = phi_prime[0] * int_tau_eps_epsp_a_phi2
B21_supg3 = phi_prime[1] * int_tau_eps_epsp_a_phi1
B22_supg3 = phi_prime[1] * int_tau_eps_epsp_a_phi2
# total element contributions
# B11_t = B11 - B11_supg2 - B11_supg3
# B12_t = B12 - B12_supg2 - B12_supg3
# B21_t = B21 - B21_supg2 - B21_supg3
# B22_t = B22 - B22_supg2 - B22_supg3

B11_t = B11 - B11_supg2
B12_t = B12 - B12_supg2
B21_t = B21 - B21_supg2
B22_t = B22 - B22_supg2

B_tot = np.concatenate([B11_t, B12_t, B21_t, B22_t])
C = csr_matrix((B_tot, (Ivar[:, 0], Ivar[:, 1])), shape=(nNod, nNod)).toarray()

# =========================
# Boundary conditions
# =========================
E_end = P / (A * strain[-1])
# Eliminate the last DOF from all other equations
F[:-1] -= C[:-1, -1] * E_end
# Strongly enforce E[-1] = E_end
C[-1, :] = 0.0
C[:, -1] = 0.0
C[-1, -1] = 1.0
F[-1] = E_end
# =========================
# Solve
# =========================
E_pred = np.linalg.solve(C, F)

# =========================
# Plot
# =========================
plt.figure(figsize=(8, 6))
plt.plot(coord, E_true, 'o', label='E_true', linewidth=2)
plt.plot(coord, E_pred, 'r', label=f'E_pred', linewidth=2)
plt.legend()
plt.xlabel('x')
plt.ylabel('E')

script_dir = os.path.dirname(os.path.abspath(__file__))
filename = f'SUPG_strain_{distribution_type}_{nel}_{nstrain}'
filepath_png = os.path.join(script_dir, f'{filename}.png')
filepath_tif = os.path.join(script_dir, f'{filename}.tif')
# plt.savefig(filepath_png, dpi=600, bbox_inches='tight')
# plt.savefig(filepath_tif, dpi=600, bbox_inches='tight')
plt.show()
