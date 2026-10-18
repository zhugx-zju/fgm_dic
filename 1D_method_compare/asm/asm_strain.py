import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
from scipy.optimize import minimize

class AdjointStateMethod1DBar:
    """
    Adjoint State Method for identifying Young's modulus distribution in 1D bar
    Direct optimization of E with positivity constraint E >= E_min
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
            Tikhonov regularization parameter (default: 0.0)
        beta : float
            Boundary force penalty parameter (default: 0.0)
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
        Compute element stiffness matrix with E interpolation

        Parameters:
        -----------
        E1, E2 : float
            Young's modulus at element nodes

        Returns:
        --------
        k_e : array (2x2)
            Element stiffness matrix
        """
        xi = np.array([0.5 - np.sqrt(3)/6, 0.5 + np.sqrt(3)/6])
        w = np.array([0.5, 0.5])
        k_e = np.zeros((2, 2))
        dN_dxi = np.array([-1.0, 1.0])
        B = dN_dxi / self.Le

        for i in range(len(xi)):
            N1 = 1.0 - xi[i]
            N2 = xi[i]
            E_xi = E1 * N1 + E2 * N2
            k_e += w[i] * E_xi * self.A * np.outer(B, B) * self.Le

        return k_e

    def compute_strain_fitting(self, E, strain_measured):
        u_computed = self.forward_solve(E)

        # Data misfit term using trapezoidal rule
        n_meas = len(strain_measured)
        x_meas = np.linspace(0, self.L, n_meas)

        # Shape function derivatives (constant in each element)
        dN_dxi = np.array([-1.0, 1.0])
        B = dN_dxi / self.Le  # B = dN/dx

        J_data = 0.0
        for i in range(n_meas):
            x = x_meas[i]
            # Find element containing this measurement point
            e = int(x / self.Le)
            if e >= self.nel:
                e = self.nel - 1

            # Get element nodal displacements
            u_elem = np.array([u_computed[e], u_computed[e + 1]])

            # Compute strain using B matrix: ε = B * u_elem
            strain_computed = np.dot(B, u_elem)
            strain_m = strain_measured[i]

            # Trapezoidal rule weights
            # if i == 0 or i == n_meas - 1:
            #     weight = 0.5 * self.L / (n_meas - 1)
            # else:
            #     weight = self.L / (n_meas - 1)

            # Same weights for boundary node
            weight = self.L / (n_meas - 1)

            J_data += (strain_computed - strain_m)**2 * weight
        
        return J_data

    def compute_tikhonov_term(self, E):
        # Regularization term: ∫(dE/dx)^2 dx
        xi = np.array([0.5 - np.sqrt(3)/6, 0.5 + np.sqrt(3)/6])
        w = np.array([0.5, 0.5])
        J_reg = 0.0

        dN_dxi = np.array([-1.0, 1.0])
        B_reg = dN_dxi / self.Le

        for e in range(self.nel):
            E_elem = np.array([E[e], E[e + 1]])
            for i in range(len(xi)):
                dE_dx = np.dot(B_reg, E_elem)
                J_reg += w[i] * dE_dx**2 * self.Le

        return J_reg

    def compute_boundary_force(self, E, u_computed):
        du_dx_L = (u_computed[-1] - u_computed[-2]) / self.Le
        E_L = E[-1]
        F_computed = E_L * self.A * du_dx_L
        J_force = (F_computed - self.P)**2
        return J_force

    def forward_solve(self, E_nodal):
        """
        Forward problem: solve for displacement

        Parameters:
        -----------
        E_nodal : array (nNod,)
            Young's modulus at each node

        Returns:
        --------
        u : array
            Nodal displacements
        """
        K = lil_matrix((self.nNod, self.nNod))

        for e in range(self.nel):
            E1 = E_nodal[e]
            E2 = E_nodal[e + 1]
            k_e = self.compute_element_stiffness(E1, E2)

            nodes = [e, e + 1]
            for i in range(2):
                for j in range(2):
                    K[nodes[i], nodes[j]] += k_e[i, j]

        # Boundary condition: u(0) = 0
        K = K.tocsr()
        K_reduced = K[1:, 1:]

        # Load: P at free end
        F = np.zeros(self.nNod)
        F[-1] = self.P
        F_reduced = F[1:]

        u_reduced = spsolve(K_reduced, F_reduced)

        u = np.zeros(self.nNod)
        u[1:] = u_reduced

        return u

    def objective_function(self, E, strain_measured):
        """
        Objective function based on strain data: J = J_data + J_reg + J_force

        Parameters:
        -----------
        E : array
            Young's modulus at each node
        strain_measured : array
            Measured strain data at specified measurement points

        Returns:
        --------
        J : float
            Objective function value
        """
        u_computed = self.forward_solve(E)
        
        # Strain mismatch term
        J_data = 0.5 * self.compute_strain_fitting(E, strain_measured)

        # Regularization term: \alpha /2 * ∫(dE/dx)^2 dx
        J_reg = 0.0
        if self.alpha > 0:
            J_reg = 0.5 * self.alpha * self.compute_tikhonov_term(E)

        # Boundary force penalty
        J_force = 0.0
        if self.beta > 0:
            J_force = 0.5 * self.beta * self.compute_boundary_force(E, u_computed)

        return J_data + J_reg + J_force

    def adjoint_solve(self, E_nodal, u_computed, strain_measured):
        """
        Adjoint problem for strain-based objective

        For J = Σ 0.5 * w_i * (ε - ε_m)² where ε = B * u
        dJ/du_j = Σ w_i * (ε - ε_m) * B_j
        where B_j is the j-th component of B matrix

        Adjoint equation: K^T * lambda = -dJ/du

        Parameters:
        -----------
        E_nodal : array
            Young's modulus at each node
        u_computed : array
            Computed displacement
        strain_measured : array
            Measured strain data at specified measurement points

        Returns:
        --------
        lambda_adj : array
            Adjoint variable
        """
        K = lil_matrix((self.nNod, self.nNod))

        for e in range(self.nel):
            E1 = E_nodal[e]
            E2 = E_nodal[e + 1]
            k_e = self.compute_element_stiffness(E1, E2)

            nodes = [e, e + 1]
            for i in range(2):
                for j in range(2):
                    K[nodes[i], nodes[j]] += k_e[i, j]

        # RHS: -dJ/du
        # For each measurement point, compute contribution using B matrix
        n_meas = len(strain_measured)
        x_meas = np.linspace(0, self.L, n_meas)

        # B matrix (constant for linear elements)
        dN_dxi = np.array([-1.0, 1.0])
        B = dN_dxi / self.Le  # B = [-1/Le, 1/Le]

        rhs = np.zeros(self.nNod)

        for i in range(n_meas):
            x = x_meas[i]
            # Find element containing this measurement point
            e = int(x / self.Le)
            if e >= self.nel:
                e = self.nel - 1

            # Get element nodal displacements
            u_elem = np.array([u_computed[e], u_computed[e + 1]])

            # Compute strain: ε = B * u_elem
            strain_computed = np.dot(B, u_elem)
            strain_m = strain_measured[i]
            strain_error = strain_computed - strain_m

            # Trapezoidal rule weight
            # if i == 0 or i == n_meas - 1:
            #     weight = 0.5 * self.L / (n_meas - 1)
            # else:
            #     weight = self.L / (n_meas - 1)
            
            # Same weights for boundary node
            weight = self.L / (n_meas - 1)
            # dJ/du = strain_error * weight * B
            # Contribution to RHS: -dJ/du
            rhs[e] -= strain_error * weight * B[0]      # B[0] = -1/Le
            rhs[e + 1] -= strain_error * weight * B[1]  # B[1] = 1/Le

        # Apply boundary condition: λ(0) = 0
        K = K.tocsr()
        K_reduced = K[1:, 1:].T  # Transpose for adjoint
        rhs_reduced = rhs[1:]

        lambda_reduced = spsolve(K_reduced, rhs_reduced)

        lambda_adj = np.zeros(self.nNod)
        lambda_adj[1:] = lambda_reduced

        return lambda_adj

    def compute_gradient(self, u, lambda_adj, E):
        """
        Compute gradient dJ/dE

        Parameters:
        -----------
        u : array
            Displacement field
        lambda_adj : array
            Adjoint variable
        E : array
            Current E distribution

        Returns:
        --------
        grad : array (nNod,)
            Gradient w.r.t. E
        """
        grad = np.zeros(self.nNod)

        xi = np.array([0.5 - np.sqrt(3)/6, 0.5 + np.sqrt(3)/6])
        w = np.array([0.5, 0.5])
        dN_dxi = np.array([-1.0, 1.0])
        B = dN_dxi / self.Le
        BTB = np.outer(B, B)

        for e in range(self.nel):
            u_elem = np.array([u[e], u[e + 1]])
            lambda_elem = np.array([lambda_adj[e], lambda_adj[e + 1]])
            E_elem = np.array([E[e], E[e + 1]])
            grad_elem = np.zeros(2)

            for i in range(len(xi)):
                N = np.array([1.0 - xi[i], xi[i]])

                # Data term
                for node_local in range(2):
                    dK_dE = w[i] * N[node_local] * self.A * BTB * self.Le
                    grad_elem[node_local] += np.dot(lambda_elem, np.dot(dK_dE, u_elem))

                # Regularization term
                if self.alpha > 0:
                    dE_dx = np.dot(B, E_elem)
                    for node_local in range(2):
                        grad_elem[node_local] += self.alpha * w[i] * dE_dx * B[node_local] * self.Le

            grad[e] += grad_elem[0]
            grad[e + 1] += grad_elem[1]

        # Boundary force term
        if self.beta > 0:
            du_dx_L = (u[-1] - u[-2]) / self.Le
            E_L = E[-1]
            F_computed = E_L * self.A * du_dx_L
            force_residual = F_computed - self.P
            grad[-1] += self.beta * force_residual * self.A * du_dx_L

        return grad

    def optimize(self, strain_measured, E_init, E_min=1e-4, method='L-BFGS-B', max_iter=100, tol=1e-8, **kwargs):
        """
        Optimize E directly with bound constraint E >= E_min
        Using strain data instead of displacement data

        Parameters:
        -----------
        strain_measured : array
            Measured strain data at specified measurement points
        E_init : array
            Initial E guess
        E_min : float
            Minimum E (positivity constraint)
        method : str
            'L-BFGS-B', 'trust-constr', or 'CG'
        max_iter : int
            Max iterations
        tol : float
            Convergence tolerance ||ΔE||/||E||

        Returns:
        --------
        E_opt : array
            Optimized E
        history : dict
            Optimization history
        """
        self.strain_measured = strain_measured
        self.history = {
            'J': [],
            'grad_norm': [],
            'E': [],
            'iteration': [],
            'E_rel_change': []
        }

        self.iteration = 0
        self.E_prev = E_init.copy()
        self.grad_cache = None
        self.E_cache = None
        self.converged = False

        def objective_wrapper(E):
            return self.objective_function(E, self.strain_measured)

        def gradient_wrapper(E):
            u = self.forward_solve(E)
            lambda_adj = self.adjoint_solve(E, u, self.strain_measured)
            grad = self.compute_gradient(u, lambda_adj, E)

            self.grad_cache = grad.copy()
            self.E_cache = E.copy()

            return grad

        def record(E):
            J = self.objective_function(E, self.strain_measured)

            if self.E_cache is not None and np.allclose(E, self.E_cache):
                grad_norm = np.linalg.norm(self.grad_cache)
            else:
                grad = gradient_wrapper(E)
                grad_norm = np.linalg.norm(grad)

            if self.iteration == 0:
                E_rel_change = np.inf
            else:
                E_norm_prev = np.linalg.norm(self.E_prev)
                E_rel_change = np.linalg.norm(E - self.E_prev) / E_norm_prev if E_norm_prev > 1e-12 else 0.0

            self.history['J'].append(J)
            self.history['grad_norm'].append(grad_norm)
            self.history['E'].append(E.copy())
            self.history['iteration'].append(self.iteration)
            self.history['E_rel_change'].append(E_rel_change)

            print(f"Iter {self.iteration}: J = {J:.6e}, ||grad|| = {grad_norm:.6e}, ||ΔE||/||E|| = {E_rel_change:.6e}")

            self.E_prev = E.copy()
            self.iteration += 1

            if E_rel_change < tol and self.iteration > 1:
                self.converged = True

        def callback(xk, state=None):
            record(np.array(xk, copy=True))
            if self.converged:
                return True

        # Bounds: E >= E_min
        bounds = [(E_min, None) for _ in range(len(E_init))]

        if method == 'L-BFGS-B':
            options = {
                'maxiter': max_iter,
                'ftol': 0,
                'gtol': 0,
                'maxls': kwargs.get('maxls', 100),
                'disp': kwargs.get('disp', False),
                'maxcor': 20
            }
            result = minimize(
                objective_wrapper, E_init, method='L-BFGS-B',
                jac=gradient_wrapper, bounds=bounds, options=options,
                callback=callback
            )

        elif method == 'trust-constr':
            options = {
                'maxiter': max_iter,
                'gtol': 0,
                'xtol': 0,
                'verbose': kwargs.get('verbose', 0)
            }
            result = minimize(
                objective_wrapper, E_init, method='trust-constr',
                jac=gradient_wrapper, options=options,
                callback=callback
            )

        elif method == 'CG':
            # CG doesn't support bounds, manually project
            def project(E):
                return np.maximum(E, E_min)

            def obj_proj(E):
                return objective_wrapper(project(E))

            def grad_proj(E):
                return gradient_wrapper(project(E))

            options = {
                'maxiter': max_iter,
                'gtol': 0,
                'disp': kwargs.get('disp', False),
                'norm': np.inf
            }
            result = minimize(
                obj_proj, E_init, method='CG',
                jac=grad_proj, options=options,
                callback=callback
            )
            result.x = project(result.x)

        else:
            raise ValueError(f"Unknown method: {method}")

        if self.converged:
            print(f"\nConverged: ||ΔE||/||E|| = {self.history['E_rel_change'][-1]:.6e} < {tol:.6e}")
        else:
            print(f"\nTerminated: Max iterations ({max_iter})")

        return result.x, self.history


def generate_noise_disp(nel, L, A, P, E_true, noise_level=0.0):
    """Generate noisy displacement data"""
    asm = AdjointStateMethod1DBar(nel, L, A, P)
    u_exact = asm.forward_solve(E_true)

    if noise_level > 0:
        noise = noise_level * np.max(np.abs(u_exact)) * np.random.randn(len(u_exact))
        u_noise = u_exact + noise
    else:
        u_noise = u_exact.copy()

    return u_noise, u_exact
