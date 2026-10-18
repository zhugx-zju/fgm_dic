"""
1D Bar FEM Module
Handles mesh generation, stiffness assembly, boundary conditions, and forward solve
"""
import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve


class FEM1DBar:
    """
    1D Bar Finite Element solver
    - Left end fixed: u(0) = 0
    - Right end loaded: F applied at x = L
    """

    def __init__(self, L, A, n_node):
        """
        Parameters:
        -----------
        L : float
            Length of bar
        A : float
            Cross-sectional area
        n_node : int
            Number of nodes
        """
        self.L = L
        self.A = A
        self.n_node = n_node
        self.n_elem = n_node - 1
        self.coord = np.linspace(0, L, n_node)
        self.h = L / self.n_elem  # Element length

    def compute_element_stiffness(self, E1, E2):
        """
        Compute element stiffness matrix with linear interpolation of E

        Parameters:
        -----------
        E1, E2 : float
            Young's modulus at element nodes

        Returns:
        --------
        k_e : array (2x2)
            Element stiffness matrix
        """
        # Gauss quadrature (2-point)
        xi = np.array([0.5 - np.sqrt(3)/6, 0.5 + np.sqrt(3)/6])
        w = np.array([0.5, 0.5])

        # Shape function derivatives in natural coordinates
        dN_dxi = np.array([-1.0, 1.0])
        B = dN_dxi / self.h  # B = dN/dx

        k_e = np.zeros((2, 2))
        for i in range(len(xi)):
            # Shape functions at Gauss point
            N1 = 1.0 - xi[i]
            N2 = xi[i]

            # E at Gauss point (linear interpolation)
            E_xi = E1 * N1 + E2 * N2

            # k_e += w * E * A * B^T * B * h
            k_e += w[i] * E_xi * self.A * np.outer(B, B) * self.h

        return k_e

    def assemble_stiffness(self, E_nodal):
        """
        Assemble global stiffness matrix

        Parameters:
        -----------
        E_nodal : array (n_node,)
            Young's modulus at each node

        Returns:
        --------
        K : sparse matrix (n_node, n_node)
            Global stiffness matrix
        """
        K = lil_matrix((self.n_node, self.n_node))

        for e in range(self.n_elem):
            E1 = E_nodal[e]
            E2 = E_nodal[e + 1]
            k_e = self.compute_element_stiffness(E1, E2)

            # Assembly
            nodes = [e, e + 1]
            for i in range(2):
                for j in range(2):
                    K[nodes[i], nodes[j]] += k_e[i, j]

        return K.tocsr()

    def apply_bc_and_solve(self, K, F):
        """
        Apply Dirichlet BC (u(0)=0) and solve K*u = f

        Parameters:
        -----------
        K : sparse matrix (n_node, n_node)
            Global stiffness matrix
        F : float
            Applied force at right end

        Returns:
        --------
        u : array (n_node,)
            Nodal displacements
        """
        # Force vector
        f = np.zeros(self.n_node)
        f[-1] = F  # Load at right end

        # Apply Dirichlet BC: u(0) = 0 by elimination
        K_reduced = K[1:, 1:]
        f_reduced = f[1:]

        # Solve reduced system
        u_reduced = spsolve(K_reduced, f_reduced)

        # Full solution
        u = np.zeros(self.n_node)
        u[1:] = u_reduced

        return u

    def forward_solve(self, E_nodal, F):
        """
        Complete forward solve: E -> u

        Parameters:
        -----------
        E_nodal : array (n_node,)
            Young's modulus at nodes
        F : float
            Applied force

        Returns:
        --------
        u : array (n_node,)
            Displacement field
        """
        K = self.assemble_stiffness(E_nodal)
        u = self.apply_bc_and_solve(K, F)
        return u

    def compute_residual(self, E_nodal, u_obs, F):
        """
        Compute FEM residual: r = K(E)*u - f
        Used for physics-based loss

        Parameters:
        -----------
        E_nodal : array (n_node,)
            Predicted Young's modulus
        u_obs : array (n_node,)
            Observed displacement
        F : float
            Applied force

        Returns:
        --------
        residual_norm : float
            Normalized residual ||r||/||f|| on free DOFs
        """
        K = self.assemble_stiffness(E_nodal)

        # Force vector
        f = np.zeros(self.n_node)
        f[-1] = F

        # Compute residual on free DOFs (exclude fixed node 0)
        K_free = K[1:, 1:]
        u_free = u_obs[1:]
        f_free = f[1:]

        # r = K*u - f
        r = K_free @ u_free - f_free

        # Normalized residual
        residual_norm = np.linalg.norm(r) / (np.linalg.norm(f_free) + 1e-10)

        return residual_norm
