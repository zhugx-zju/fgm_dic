import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
import os

# ParaSet
nel = 20  # Number of Elements
nNod = nel + 1  # Number of Nodes
distribution_type = 'heaviside'
nstrain = 10  # Number of strain data points per element
P = 1.0  # Load
A = 1.0  # Section Area
L = 1.0  # Length of the bar
E0 = 1.0  # Modulus at the left end
coord = np.linspace(0, L, nNod)  # Coordinates of nodes
x_strain = np.linspace(0, L, (nstrain - 1) * nel + 1)  # Coordinates of strain data

# Strain Data
if distribution_type == 'sinusoidal':
    k = 0.5
    E_true = E0 * (2 + np.sin(2 * np.pi * k * coord / L))
    strain = P / (2 + np.sin(2 * np.pi * k * x_strain / L)) / A / E0
elif distribution_type == 'linear':
    E_true = E0 * (1 + coord / L)
    strain = P / (1 + x_strain / L) / A / E0
elif distribution_type == 'quadratic':
    E_true = E0 * (1 + (coord / L)**2)
    strain = P / (1 + (x_strain / L)**2) / A / E0
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

# Force Vector
F = np.zeros(nNod)
F[0] = -P

# FEM Info
meshnodes = np.arange(1, nel + 1).reshape(-1, 1)
eleNodesID = meshnodes + np.array([0, 1]) - 1  # Convert to 0-based indexing
strainlist = np.arange(0, 1 + (nstrain - 1) * (nel - 1), nstrain - 1)
IDlist = np.arange(0, nstrain)
StrainID = strainlist[:, np.newaxis] + IDlist
eleStrainID = StrainID.T.flatten()
ele_strain = strain[eleStrainID].reshape(nstrain, nel).T
ele_point = x_strain[eleStrainID].reshape(nstrain, nel).T
xEle = coord[eleNodesID]
zeta = (ele_point - xEle[:, [0]]) * nel / L

# Assemble ParaSet
si = np.array([0, 0, 1, 1])
sii = np.array([0, 1, 0, 1])
ik = eleNodesID[:, si].T
jk = eleNodesID[:, sii].T
Ivar = np.column_stack([ik.flatten(), jk.flatten()])

# Coefficients Matrix Calculation
W11 = np.zeros((nel, nstrain))  # Weights Matrix for B11
W11[:, 0] = (zeta[:, 1] - zeta[:, 0]) * (zeta[:, 0] - 1) / 2
W11[:, -1] = (zeta[:, -1] - zeta[:, -2]) * (zeta[:, -1] - 1) / 2
W11[:, 1:-1] = (zeta[:, 2:] - zeta[:, :-2]) * (zeta[:, 1:-1] - 1) / 2
W11 = A * W11

W12 = np.zeros((nel, nstrain))  # Weights Matrix for B12
W12[:, 0] = (zeta[:, 0] - zeta[:, 1]) * zeta[:, 0] / 2
W12[:, -1] = (zeta[:, -2] - zeta[:, -1]) * zeta[:, -1] / 2
W12[:, 1:-1] = (zeta[:, :-2] - zeta[:, 2:]) * zeta[:, 1:-1] / 2
W12 = A * W12

W21 = np.zeros((nel, nstrain))  # Weights Matrix for B21
W21[:, 0] = (zeta[:, 1] - zeta[:, 0]) * (1 - zeta[:, 0]) / 2
W21[:, -1] = (zeta[:, -1] - zeta[:, -2]) * (1 - zeta[:, -1]) / 2
W21[:, 1:-1] = (zeta[:, 2:] - zeta[:, :-2]) * (1 - zeta[:, 1:-1]) / 2
W21 = A * W21

W22 = np.zeros((nel, nstrain))  # Weights Matrix for B22
W22[:, 0] = (zeta[:, 1] - zeta[:, 0]) * zeta[:, 0] / 2
W22[:, -1] = (zeta[:, -1] - zeta[:, -2]) * zeta[:, -1] / 2
W22[:, 1:-1] = (zeta[:, 2:] - zeta[:, :-2]) * zeta[:, 1:-1] / 2
W22 = A * W22

B11 = (ele_strain * W11).sum(axis=1)
B12 = (ele_strain * W12).sum(axis=1)
B21 = (ele_strain * W21).sum(axis=1)
B22 = (ele_strain * W22).sum(axis=1)
B_tot = np.concatenate([B11, B12, B21, B22])
C = csr_matrix((B_tot, (Ivar[:, 0], Ivar[:, 1])), shape=(nNod, nNod)).toarray()

# # Boundary Condition
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
# Plot
plt.figure(figsize=(8, 6))
plt.plot(coord, E_true, 'o', label='E_true', linewidth=2)
plt.plot(coord, E_pred, 'r', label='E_pred', linewidth=2)
plt.legend()
plt.xlabel('x')
plt.ylabel('E')
# plt.grid(True)

# Save figure
script_dir = os.path.dirname(os.path.abspath(__file__))
filename = f'strain_{distribution_type}_{nel}_{nstrain}'
filepath_png = os.path.join(script_dir, f'{filename}.png')
filepath_tif = os.path.join(script_dir, f'{filename}.tif')
# plt.savefig(filepath_png, dpi=600, bbox_inches='tight')
# plt.savefig(filepath_tif, dpi=600, bbox_inches='tight')
plt.show()
