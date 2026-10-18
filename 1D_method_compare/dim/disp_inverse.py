import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
import os
import math

# ParaSet
nel = 40  # Number of Elements
nNod = nel + 1  # Number of Nodes
distribution_type = 'linear'
P = 1.0  # Load
A = 1.0  # Section Area
L = 1.0  # Length of the bar
E0 = 1.0  # Modulus at the left end
coord = np.linspace(0, L, nNod)  # Coordinates of nodes
amp = P / A / E0
# disp Data
if distribution_type == 'sinusoidal':
    k = 5
    a = 2 * math.pi * k / L
    E_true = E0 * (2 + np.sin(a * coord / L))
    disp = 2 * (amp / a / math.sqrt(3)) * (np.arctan((2 * np.tan(a * coord / 2) + 1) / math.sqrt(3)) - math.pi / 6 )
elif distribution_type == 'linear':
    E_true = E0 * (1 + coord / L)
    disp = amp * L * np.log(1 + coord / L)
elif distribution_type == 'quadratic':
    E_true = E0 * (1 + (coord / L)**2)
    disp = amp * L * np.arctan(coord /L)

# Force Vector
F = np.zeros(nNod)

# FEM Info
meshnodes = np.arange(1, nel + 1).reshape(-1, 1)
eleNodesID = meshnodes + np.array([0, 1]) - 1  # Convert to 0-based indexing

delta_disp = A * nel * (disp[1:]-disp[0:-1]) / L
                        
B11 = - delta_disp / 2
B12 = - delta_disp / 2
B21 = delta_disp / 2
B22 = delta_disp / 2
B_tot = np.concatenate([B11, B12, B21, B22])

# Assemble ParaSet
si = np.array([0, 0, 1, 1])
sii = np.array([0, 1, 0, 1])
ik = eleNodesID[:, si].T
jk = eleNodesID[:, sii].T
Ivar = np.column_stack([ik.flatten(), jk.flatten()])
C = csr_matrix((B_tot, (Ivar[:, 0], Ivar[:, 1])), shape=(nNod, nNod)).toarray()

# Boundary Condition
C[0, 0] = C[0, 0] + (disp[1] - disp[0]) * A * nel / L
F[-2] = -C[-2, -1] * P / (A * (disp[-1] - disp[-2]) * nel / L)

E_pred = np.zeros(nNod)
E_pred[:-1] = np.linalg.solve(C[:-1, :-1], F[:-1])
E_pred[-1] = L * P / A / (disp[-1] - disp[-2]) / nel

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
filename = f'disp_{distribution_type}_{nel}'
filepath_png = os.path.join(script_dir, f'{filename}.png')
filepath_tif = os.path.join(script_dir, f'{filename}.tif')
# plt.savefig(filepath_png, dpi=600, bbox_inches='tight')
# plt.savefig(filepath_tif, dpi=600, bbox_inches='tight')
plt.show()
