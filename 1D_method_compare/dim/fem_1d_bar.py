import numpy as np
import matplotlib.pyplot as plt

def fem_1d_bar(nel, L, A, E, P):
    """
    1D bar finite element analysis

    Parameters:
    -----------
    nel : int
        Number of elements
    L : float
        Total length of bar
    A : float
        Cross-sectional area
    E : float or array
        Young's modulus (constant or array for each element)
    P : float
        Applied load at free end

    Returns:
    --------
    u : array
        Nodal displacements
    coord : array
        Node coordinates
    """

    # Number of nodes
    nNod = nel + 1

    # Node coordinates
    coord = np.linspace(0, L, nNod)

    # Element length
    Le = L / nel

    # Initialize global stiffness matrix and force vector
    K_global = np.zeros((nNod, nNod))
    F_global = np.zeros(nNod)

    # If E is scalar, convert to array
    if np.isscalar(E):
        E_array = np.full(nel, E)
    else:
        E_array = E

    # Assemble global stiffness matrix
    for e in range(nel):
        # Element stiffness matrix
        k_e = (A * E_array[e] / Le) * np.array([[1, -1], [-1, 1]])

        # Global node indices for this element
        nodes = [e, e + 1]

        # Assemble into global matrix
        for i in range(2):
            for j in range(2):
                K_global[nodes[i], nodes[j]] += k_e[i, j]

    # Apply boundary conditions: fixed at x=0
    # Apply load at free end
    F_global[-1] = P

    # Solve: remove first DOF (fixed boundary)
    K_reduced = K_global[1:, 1:]
    F_reduced = F_global[1:]

    # Solve for displacements
    u_reduced = np.linalg.solve(K_reduced, F_reduced)

    # Full displacement vector
    u = np.zeros(nNod)
    u[1:] = u_reduced

    return u, coord


def calculate_stress_strain(u, coord, A, E, nel):
    """
    Calculate stress and strain in each element

    Parameters:
    -----------
    u : array
        Nodal displacements
    coord : array
        Node coordinates
    A : float
        Cross-sectional area
    E : float or array
        Young's modulus
    nel : int
        Number of elements

    Returns:
    --------
    strain : array
        Strain in each element
    stress : array
        Stress in each element
    x_center : array
        Element center coordinates
    """

    L = coord[-1] - coord[0]
    Le = L / nel

    if np.isscalar(E):
        E_array = np.full(nel, E)
    else:
        E_array = E

    strain = np.zeros(nel)
    stress = np.zeros(nel)
    x_center = np.zeros(nel)

    for e in range(nel):
        # Strain = du/dx
        strain[e] = (u[e + 1] - u[e]) / Le
        # Stress = E * strain
        stress[e] = E_array[e] * strain[e]
        # Element center coordinate
        x_center[e] = (coord[e] + coord[e + 1]) / 2

    return strain, stress, x_center


# Example: Uniform bar
if __name__ == "__main__":
    # Parameters
    nel = 20  # Number of elements
    L = 1.0  # Length (m)
    A = 0.01  # Cross-sectional area (m^2)
    E = 200e9  # Young's modulus (Pa)
    P = 1000  # Applied load (N)

    # Solve
    u, coord = fem_1d_bar(nel, L, A, E, P)
    strain, stress, x_center = calculate_stress_strain(u, coord, A, E, nel)

    # Analytical solution for uniform bar
    u_analytical = P * coord / (A * E)
    strain_analytical = P / (A * E)
    stress_analytical = P / A

    # Plot results
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Displacement
    axes[0].plot(coord, u * 1000, 'ro-', label='FEM', markersize=4)
    axes[0].plot(coord, u_analytical * 1000, 'b--', label='Analytical', linewidth=2)
    axes[0].set_xlabel('Position x (m)')
    axes[0].set_ylabel('Displacement u (mm)')
    axes[0].set_title('Displacement')
    axes[0].legend()
    axes[0].grid(True)

    # Strain
    axes[1].plot(x_center, strain, 'ro-', label='FEM', markersize=4)
    axes[1].axhline(y=strain_analytical, color='b', linestyle='--', label='Analytical', linewidth=2)
    axes[1].set_xlabel('Position x (m)')
    axes[1].set_ylabel('Strain ε')
    axes[1].set_title('Strain')
    axes[1].legend()
    axes[1].grid(True)

    # Stress
    axes[2].plot(x_center, stress / 1e6, 'ro-', label='FEM', markersize=4)
    axes[2].axhline(y=stress_analytical / 1e6, color='b', linestyle='--', label='Analytical', linewidth=2)
    axes[2].set_xlabel('Position x (m)')
    axes[2].set_ylabel('Stress σ (MPa)')
    axes[2].set_title('Stress')
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    plt.show()

    # Print results
    print(f"Maximum displacement: {u[-1] * 1000:.6f} mm")
    print(f"Analytical displacement: {u_analytical[-1] * 1000:.6f} mm")
    print(f"Average strain: {np.mean(strain):.6e}")
    print(f"Analytical strain: {strain_analytical:.6e}")
    print(f"Average stress: {np.mean(stress) / 1e6:.6f} MPa")
    print(f"Analytical stress: {stress_analytical / 1e6:.6f} MPa")
