import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
from scipy.optimize import minimize

class AdjointStateMethod1DBar:
    """
    Adjoint State Method for identifying Young's modulus distribution in 1D bar
    """

    def __init__(self, nel, L, A, P, alpha=0.0, beta=0.0):
        """
        Initialize the problem

        Parameters:
        -----------
        nel : int
            Number of elements
        L : float
            Length of bar
        A : float
            Cross-sectional area
        P : float
            Applied load at free end
        alpha : float
            Tikhonov regularization parameter (default: 0.0, no regularization)
        beta : float
            Boundary force penalty parameter (default: 0.0, no penalty)
        """
        self.nel = nel
        self.L = L
        self.A = A
        self.P = P
        self.alpha = alpha
        self.beta = beta
        self.nNod = nel + 1
        self.coord = np.linspace(0, L, self.nNod)
        self.Le = L / nel

    def compute_element_stiffness(self, E1, E2):
        """
        Compute element stiffness matrix with E interpolation using shape functions

        K_ij = ∫ E(ξ) × A × dN_i/dx × dN_j/dx dx

        For linear elements:
        N1(ξ) = 1 - ξ, N2(ξ) = ξ
        E(ξ) = E1 × (1-ξ) + E2 × ξ
        dN1/dx = -1/Le, dN2/dx = 1/Le

        Parameters:
        -----------
        E1, E2 : float
            Young's modulus at element nodes

        Returns:
        --------
        k_e : array (2x2)
            Element stiffness matrix
        """
        # Gaussian quadrature (2-point for exact integration of quadratic polynomial)
        xi = np.array([0.5 - np.sqrt(3)/6, 0.5 + np.sqrt(3)/6])  # Gauss points in [0,1]
        w = np.array([0.5, 0.5])  # Gauss weights

        k_e = np.zeros((2, 2))

        # Shape function derivatives: dN/dxi
        dN_dxi = np.array([-1.0, 1.0])

        # B matrix: dN/dx = dN/dxi * dxi/dx = dN/dxi / Le
        B = dN_dxi / self.Le

        for i in range(len(xi)):
            # Interpolate E at Gauss point
            N1 = 1.0 - xi[i]
            N2 = xi[i]
            E_xi = E1 * N1 + E2 * N2

            # Element stiffness contribution
            # K = B^T × E × A × B × Le (Le is Jacobian)
            k_e += w[i] * E_xi * self.A * np.outer(B, B) * self.Le

        return k_e

    def forward_solve(self, E_nodal):
        """
        Forward problem: solve for displacement given E at nodes

        Parameters:
        -----------
        E_nodal : array (nNod,)
            Young's modulus at each node

        Returns:
        --------
        u : array
            Nodal displacements
        """
        # Global stiffness matrix
        K = lil_matrix((self.nNod, self.nNod))

        for e in range(self.nel):
            # Get nodal E values for this element
            E1 = E_nodal[e]
            E2 = E_nodal[e + 1]

            # Element stiffness with E interpolation
            k_e = self.compute_element_stiffness(E1, E2)

            # Assemble
            nodes = [e, e + 1]
            for i in range(2):
                for j in range(2):
                    K[nodes[i], nodes[j]] += k_e[i, j]

        # Force vector
        F = np.zeros(self.nNod)
        F[-1] = self.P

        # Apply boundary condition: u(0) = 0
        K = K.tocsr()
        K_reduced = K[1:, 1:]
        F_reduced = F[1:]

        # Solve
        u_reduced = spsolve(K_reduced, F_reduced)

        u = np.zeros(self.nNod)
        u[1:] = u_reduced

        return u

    def compute_strain_stress(self, u, E_nodal):
        """
        Compute strain and stress from displacement

        Parameters:
        -----------
        u : array
            Nodal displacements
        E_nodal : array
            Young's modulus at each node

        Returns:
        --------
        strain : array
            Strain in each element (at element center)
        stress : array
            Stress in each element (at element center)
        """
        strain = np.zeros(self.nel)
        stress = np.zeros(self.nel)

        for e in range(self.nel):
            strain[e] = (u[e + 1] - u[e]) / self.Le
            # E at element center (average of nodal values)
            E_center = (E_nodal[e] + E_nodal[e + 1]) / 2
            stress[e] = E_center * strain[e]

        return strain, stress

    def objective_function(self, E, u_noise):
        """
        Objective function with Tikhonov regularization and boundary force penalty

        J = 0.5 * ∫ (u_computed - u_noise)^2 dx
            + 0.5 * alpha * ∫ (dE/dx)^2 dx
            + 0.5 * beta * (E*A*du/dx|_{x=L} - P)^2

        Parameters:
        -----------
        E : array
            Young's modulus at each node
        u_noise : array
            Measured displacement data

        Returns:
        --------
        J : float
            Objective function value
        """
        u_computed = self.forward_solve(E)

        # Data misfit term: ∫ (u - u_m)^2 dx
        J_data = 0.0
        xi = np.array([0.5 - np.sqrt(3)/6, 0.5 + np.sqrt(3)/6])
        w = np.array([0.5, 0.5])

        for e in range(self.nel):
            u_elem = np.array([u_computed[e], u_computed[e + 1]])
            u_m_elem = np.array([u_noise[e], u_noise[e + 1]])

            for i in range(len(xi)):
                N = np.array([1.0 - xi[i], xi[i]])
                u_xi = np.dot(N, u_elem)
                u_m_xi = np.dot(N, u_m_elem)
                J_data += 0.5 * w[i] * (u_xi - u_m_xi)**2 * self.Le

        # Regularization term: alpha * ∫ exp(2m) * (dm/dx)^2 dx
        # Need to compute m from E: m = log(E - E_min)
        J_reg = 0.0
        if self.alpha > 0:
            dN_dxi = np.array([-1.0, 1.0])
            B = dN_dxi / self.Le
            E_min_assumed = 1e-4

            for e in range(self.nel):
                E_elem = np.array([E[e], E[e + 1]])
                m_elem = np.log(E_elem - E_min_assumed + 1e-15)

                for i in range(len(xi)):
                    N = np.array([1.0 - xi[i], xi[i]])
                    dm_dx = np.dot(B, m_elem)
                    m_gauss = np.dot(N, m_elem)
                    exp_2m = np.exp(2 * m_gauss)
                    J_reg += 0.5 * self.alpha * w[i] * exp_2m * dm_dx**2 * self.Le

        # Boundary force penalty: beta * (E*A*du/dx|_{x=L} - P)^2
        J_force = 0.0
        if self.beta > 0:
            # Compute strain at right boundary: du/dx|_{x=L}
            # Using last element
            du_dx_L = (u_computed[-1] - u_computed[-2]) / self.Le

            # Interpolate E at x=L (right boundary)
            E_L = E[-1]

            # Reaction force at right boundary
            F_computed = E_L * self.A * du_dx_L

            # Penalty term
            J_force = 0.5 * self.beta * (F_computed - self.P)**2

        return J_data + J_reg + J_force

    def adjoint_solve(self, E_nodal, u_computed, u_noise):
        """
        Adjoint problem: solve for adjoint variable lambda

        K^T * lambda = -(u_computed - u_noise)

        Parameters:
        -----------
        E_nodal : array
            Young's modulus at each node
        u_computed : array
            Computed displacement
        u_noise : array
            Measured displacement

        Returns:
        --------
        lambda_adj : array
            Adjoint variable
        """
        # Same stiffness matrix as forward problem
        K = lil_matrix((self.nNod, self.nNod))

        for e in range(self.nel):
            # Get nodal E values for this element
            E1 = E_nodal[e]
            E2 = E_nodal[e + 1]

            # Element stiffness with E interpolation
            k_e = self.compute_element_stiffness(E1, E2)

            nodes = [e, e + 1]
            for i in range(2):
                for j in range(2):
                    K[nodes[i], nodes[j]] += k_e[i, j]

        # Right-hand side: -∫ (u - u_m) × N dx
        rhs = np.zeros(self.nNod)

        xi = np.array([0.5 - np.sqrt(3)/6, 0.5 + np.sqrt(3)/6])
        w = np.array([0.5, 0.5])

        for e in range(self.nel):
            u_elem = np.array([u_computed[e], u_computed[e + 1]])
            u_m_elem = np.array([u_noise[e], u_noise[e + 1]])

            rhs_elem = np.zeros(2)

            for i in range(len(xi)):
                # Shape functions
                N = np.array([1.0 - xi[i], xi[i]])

                # Interpolate u and u_m
                u_xi = np.dot(N, u_elem)
                u_m_xi = np.dot(N, u_m_elem)

                # RHS contribution
                rhs_elem += w[i] * (u_xi - u_m_xi) * N * self.Le

            # Assemble
            rhs[e] -= rhs_elem[0]
            rhs[e + 1] -= rhs_elem[1]

        # Apply boundary condition: λ(0) = 0
        K = K.tocsr()
        K_reduced = K[1:, 1:].T  # Transpose for adjoint
        rhs_reduced = rhs[1:]

        # Solve
        lambda_reduced = spsolve(K_reduced, rhs_reduced)

        lambda_adj = np.zeros(self.nNod)
        lambda_adj[1:] = lambda_reduced

        return lambda_adj

    def compute_gradient(self, u, lambda_adj, m, E_min=1e-4):
        """
        Compute gradient of objective function w.r.t. m

        For E = E_min + exp(m):
        - Data term: dJ_data/dm_i = exp(m_i) * [-lambda^T × (dK/dE_i) × u]
        - Regularization: dJ_reg/dm_i = alpha * ∫ exp(2m) * [N_i * (dm/dx)^2 + (dm/dx) * (dN_i/dx)] dx
        - Force penalty: dJ_force/dm_i = exp(m_i) * beta * (F - P) * A * du/dx|_{x=L}

        where dK/dE_i = ∫ N_i(ξ) × A × B^T × B dx

        Parameters:
        -----------
        u : array
            Displacement field
        lambda_adj : array
            Adjoint variable
        m : array
            Current m distribution
        E_min : float
            Minimum E value

        Returns:
        --------
        grad : array (nNod,)
            Gradient w.r.t. m at nodes
        """

        E = E_min + np.exp(m)
        grad = np.zeros(self.nNod)

        # Gaussian quadrature
        xi = np.array([0.5 - np.sqrt(3)/6, 0.5 + np.sqrt(3)/6])
        w = np.array([0.5, 0.5])

        # Shape function derivatives: dN/dxi
        dN_dxi = np.array([-1.0, 1.0])
        B = dN_dxi / self.Le  # B is constant for linear elements

        # B^T × B (outer product, 2x2 matrix)
        BTB = np.outer(B, B)

        for e in range(self.nel):
            # Nodal values for this element
            u_elem = np.array([u[e], u[e + 1]])
            lambda_elem = np.array([lambda_adj[e], lambda_adj[e + 1]])
            m_elem = np.array([m[e], m[e + 1]])

            # Gradient contributions for this element
            grad_elem = np.zeros(2)

            for i in range(len(xi)):
                # Shape functions at Gauss point
                N = np.array([1.0 - xi[i], xi[i]])

                # ===== Data term =====
                # dJ_data/dE_i, then convert to dJ_data/dm_i by multiplying exp(m_i)
                for node_local in range(2):
                    dK_dE = w[i] * N[node_local] * self.A * BTB * self.Le
                    # This is dJ_data/dE_i
                    grad_data_wrt_E = np.dot(lambda_elem, np.dot(dK_dE, u_elem))
                    # Convert to dJ_data/dm_i = exp(m_i) * dJ_data/dE_i
                    grad_elem[node_local] += grad_data_wrt_E * np.exp(m_elem[node_local])

                # ===== Regularization term =====
                # J_reg = (1/2) * alpha * ∫ exp(2m) * (dm/dx)^2 dx
                # dJ_reg/dm_i = alpha * ∫ exp(2m) * [N_i * (dm/dx)^2 + (dm/dx) * (dN_i/dx)] dx
                if self.alpha > 0:
                    # dm/dx at Gauss point
                    dm_dx = np.dot(B, m_elem)

                    # exp(2m) at Gauss point
                    m_gauss = np.dot(N, m_elem)
                    exp_2m = np.exp(2 * m_gauss)

                    # Two terms in regularization gradient
                    for node_local in range(2):
                        # Term 1: exp(2m) * N_i * (dm/dx)^2
                        term1 = exp_2m * N[node_local] * dm_dx**2
                        # Term 2: exp(2m) * (dm/dx) * (dN_i/dx)
                        term2 = exp_2m * dm_dx * B[node_local]

                        grad_elem[node_local] += self.alpha * w[i] * (term1 + term2) * self.Le

            # Assemble to global gradient
            grad[e] += grad_elem[0]
            grad[e + 1] += grad_elem[1]

        # ===== Boundary force penalty term =====
        if self.beta > 0:
            # Compute boundary force
            du_dx_L = (u[-1] - u[-2]) / self.Le
            E_L = E[-1]
            F_computed = E_L * self.A * du_dx_L

            # Force residual
            force_residual = F_computed - self.P

            # dJ_force/dE[-1] = beta * (F - P) * A * du/dx|_{x=L}
            # dJ_force/dm[-1] = exp(m[-1]) * dJ_force/dE[-1]
            grad[-1] += self.beta * force_residual * self.A * du_dx_L * np.exp(m[-1])

        return grad

    def optimize(self, u_noise, m_init, E_min=1e-4, method='L-BFGS-B', max_iter=100, tol=1e-8, **kwargs):

        self.u_noise = u_noise
        self.history = {
            'J': [],
            'grad_norm': [],
            'm': [],
            'iteration': [],
            'm_rel_change': []
        }
        
        self.iteration = 0
        self.m_prev = m_init.copy()
        self.grad_cache = None
        self.m_cache = None
        self.converged = False

        def objective_wrapper(m):
            E = E_min + np.exp(m)
            return self.objective_function(E, self.u_noise)

        def gradient_wrapper(m):
            E = E_min + np.exp(m)
            u = self.forward_solve(E)
            lambda_adj = self.adjoint_solve(E, u, self.u_noise)
            grad = self.compute_gradient(u, lambda_adj, m, E_min)

            self.grad_cache = grad.copy()
            self.m_cache = m.copy()

            return grad

        def record(m):
            E = E_min + np.exp(m)
            J = self.objective_function(E, self.u_noise)

            if self.m_cache is not None and np.allclose(m, self.m_cache):
                grad_norm = np.linalg.norm(self.grad_cache)
            else:
                grad = gradient_wrapper(m)
                grad_norm = np.linalg.norm(grad)

            if self.iteration == 0:
                m_rel_change = np.inf
            else:
                m_norm_prev = np.linalg.norm(self.m_prev)
                m_rel_change = np.linalg.norm(m - self.m_prev) / m_norm_prev if m_norm_prev > 1e-12 else 0.0

            self.history['J'].append(J)
            self.history['grad_norm'].append(grad_norm)
            self.history['m'].append(m.copy())
            self.history['iteration'].append(self.iteration)
            self.history['m_rel_change'].append(m_rel_change)

            print(f"Iter {self.iteration}: J = {J:.6e}, ||grad|| = {grad_norm:.6e}, ||Δm||/||m|| = {m_rel_change:.6e}")

            self.m_prev = m.copy()
            self.iteration += 1

            # Check convergence based on relative change in E
            if m_rel_change < tol and self.iteration > 1:
                self.converged = True

        def callback(xk, state=None):
            record(np.array(xk, copy=True))
            # Stop optimization if converged
            if self.converged:
                return True  # Returning True stops the optimization

        # Method-specific options
        if method == 'L-BFGS-B':
            options = {
                'maxiter': max_iter,
                'ftol': 0,  # Disable ftol, use custom convergence
                'gtol': 0,  # Disable gtol, use custom convergence
                'maxls': kwargs.get('maxls', 100),
                'disp': kwargs.get('disp', False),
                'maxcor': 20
            }
            result = minimize(
                objective_wrapper, m_init, method='L-BFGS-B',
                jac=gradient_wrapper, options=options,
                callback=callback
            )

        elif method == 'trust-constr':
            options = {
                'maxiter': max_iter,
                'gtol': 0,  # Disable gtol, use custom convergence
                'xtol': 0,  # Disable xtol, use custom convergence
                'verbose': kwargs.get('verbose', 0)
            }
            result = minimize(
                objective_wrapper, m_init, method='trust-constr',
                jac=gradient_wrapper, options=options,
                callback=callback
            )

        elif method == 'CG':
            options = {
                'maxiter': max_iter,
                'gtol': 0,  # Disable gtol, use custom convergence
                'disp': kwargs.get('disp', False),
                'norm': np.inf
            }
            result = minimize(
                objective_wrapper, m_init, method='CG',
                jac=gradient_wrapper, options=options,
                callback=callback
            )

        else:
            raise ValueError(f"Unknown method: {method}. Choose from 'L-BFGS-B', 'trust-constr', 'CG', 'gradient-descent'")

        if self.converged:
            print(f"\nConverged: ||Δm||/||m|| = {self.history['m_rel_change'][-1]:.6e} < {tol:.6e}")
        else:
            print(f"\nTerminated: Maximum iterations reached ({max_iter})")

        return result.x, self.history

    def compute_gradient_finite_diff(self, m, E_min=1e-4, eps=1e-7):
        """
        Compute gradient using finite difference for verification

        Parameters:
        -----------
        m : array
            Current m values
        E_min : float
            Minimum E value
        eps : float
            Finite difference step size

        Returns:
        --------
        grad_fd : array
            Finite difference gradient
        """
        grad_fd = np.zeros_like(m)
        E = E_min + np.exp(m)

        # Compute base objective
        J0 = self.objective_function(E, self.u_noise)

        # Central difference for each component
        for i in range(len(m)):
            m_plus = m.copy()
            m_plus[i] += eps
            E_plus = E_min + np.exp(m_plus)
            J_plus = self.objective_function(E_plus, self.u_noise)

            m_minus = m.copy()
            m_minus[i] -= eps
            E_minus = E_min + np.exp(m_minus)
            J_minus = self.objective_function(E_minus, self.u_noise)

            grad_fd[i] = (J_plus - J_minus) / (2 * eps)

        return grad_fd

    def verify_gradient(self, m, E_min=1e-4, eps=1e-7, num_samples=10):
        """
        Verify analytical gradient against finite difference

        Parameters:
        -----------
        m : array
            Current m values
        E_min : float
            Minimum E value
        eps : float
            Finite difference step size
        num_samples : int
            Number of random components to check

        Returns:
        --------
        results : dict
            Verification results including errors
        """
        E = E_min + np.exp(m)

        # Compute analytical gradient
        u = self.forward_solve(E)
        lambda_adj = self.adjoint_solve(E, u, self.u_noise)
        grad_analytical = self.compute_gradient(u, lambda_adj, m, E_min)

        # Compute finite difference gradient
        print("Computing finite difference gradient...")
        grad_fd = self.compute_gradient_finite_diff(m, E_min, eps)

        # Compute errors
        abs_error = np.abs(grad_analytical - grad_fd)
        rel_error = abs_error / (np.abs(grad_fd) + 1e-12)

        # Sample random indices for detailed comparison
        if num_samples > 0:
            indices = np.random.choice(len(m), min(num_samples, len(m)), replace=False)
            print("\n" + "="*80)
            print("Gradient Verification (sample components)")
            print("="*80)
            print(f"{'Index':<8} {'Analytical':<15} {'Finite Diff':<15} {'Abs Error':<15} {'Rel Error':<15}")
            print("-"*80)
            for idx in sorted(indices):
                print(f"{idx:<8} {grad_analytical[idx]:<15.6e} {grad_fd[idx]:<15.6e} "
                      f"{abs_error[idx]:<15.6e} {rel_error[idx]:<15.6e}")

        print("\n" + "="*80)
        print("Overall Statistics")
        print("="*80)
        print(f"Max absolute error:  {np.max(abs_error):.6e}")
        print(f"Mean absolute error: {np.mean(abs_error):.6e}")
        print(f"Max relative error:  {np.max(rel_error):.6e}")
        print(f"Mean relative error: {np.mean(rel_error):.6e}")
        print(f"||grad_analytical||:  {np.linalg.norm(grad_analytical):.6e}")
        print(f"||grad_fd||:          {np.linalg.norm(grad_fd):.6e}")
        print(f"||difference||:       {np.linalg.norm(grad_analytical - grad_fd):.6e}")
        print("="*80)

        # Check if gradient is correct (relative error < 1e-5 is typically good for central difference)
        if np.max(rel_error) < 1e-5:
            print("✓ Gradient verification PASSED!")
        else:
            print("X Gradient verification FAILED - check implementation")

        return {
            'grad_analytical': grad_analytical,
            'grad_fd': grad_fd,
            'abs_error': abs_error,
            'rel_error': rel_error,
            'max_abs_error': np.max(abs_error),
            'max_rel_error': np.max(rel_error)
        }


def generate_noise_disp(nel, L, A, P, E_true, noise_level=0.0):
    """
    Generate synthetic displacement data

    Parameters:
    -----------
    nel : int
        Number of elements
    L : float
        Length
    A : float
        Area
    P : float
        Load
    E_true : array
        True Young's modulus distribution
    noise_level : float
        Noise level (fraction of signal)

    Returns:
    --------
    u_noise : array
        Measured displacement with noise
    """
    asm = AdjointStateMethod1DBar(nel, L, A, P)
    u_exact = asm.forward_solve(E_true)

    # Add noise
    noise = noise_level * np.max(np.abs(u_exact)) * np.random.randn(len(u_exact))
    u_noise = u_exact + noise

    return u_exact, u_noise
