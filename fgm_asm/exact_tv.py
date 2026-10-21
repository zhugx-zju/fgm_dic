"""Unsmoothed, quadrature isotropic TV for independent Q4 nodal moduli.

The objective is f(E) + gamma * sum_q a_q |(G E)_q|, E = exp(m).
The FE displacement trial/test functions and discrete adjoint are unchanged.
ADMM splits z = G E. Its nonlinear material subproblem has no global
convergence guarantee; success requires the saved approximate KKT checks.
No reference modulus or noise normalization enters the objective.
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.sparse import csr_matrix

from .inverse_solver import _evaluate_inverse_state
from .material import MaterialInfo


class QuadratureGradient:
    """Physical derivatives (1/mm), ordered by Gauss point then element."""

    def __init__(self, mesh):
        nodes = mesh.ele_nods_id - 1
        count = 4 * mesh.n_el
        rows = np.repeat(np.arange(count), 4)
        columns = np.broadcast_to(nodes, (4, *nodes.shape)).ravel()
        self.gx, self.gy = [csr_matrix(
            (values.ravel(), (rows, columns)), shape=(count, mesh.n_nod))
            for values in (mesh.gauss_dNdx, mesh.gauss_dNdy)]
        self.area = (mesh.gauss_abs_det_j * mesh.gauss_w[:, None]).ravel()
        self.total_area = float(self.area.sum())

    def apply(self, values):
        # Subtracting a constant avoids cancellation at a uniform material.
        centered = values - np.mean(values)
        return np.column_stack((self.gx @ centered, self.gy @ centered))

    def adjoint(self, vectors):
        result = np.asarray(self.gx.T @ (self.area * vectors[:, 0]) +
                            self.gy.T @ (self.area * vectors[:, 1])).ravel()
        # Exact transpose of the centering operation in apply().
        return result - result.mean()

    def value(self, values):
        return float(self.area @ np.linalg.norm(self.apply(values), axis=1))

    def rms(self, vectors):
        return float(np.sqrt(self.area @ np.sum(vectors**2, axis=1) / self.total_area))


def vector_shrink(vectors, threshold):
    """Exact proximal map of threshold*Euclidean norm, including zero."""
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError('Shrinkage threshold must be finite and nonnegative')
    lengths = np.linalg.norm(vectors, axis=1)
    scale = np.zeros_like(lengths)
    active = lengths > threshold
    scale[active] = 1.0 - threshold / lengths[active]
    return vectors * scale[:, None]


class FEDataOracle:
    """Reuse the established forward/rigid projection/adjoint without a prior."""

    def __init__(self, mesh, cases, nu):
        self.mesh, self.cases = mesh, tuple(cases)
        if not self.cases:
            raise ValueError('At least one identification case is required')
        self.material = MaterialInfo(nu=nu)
        self.evaluations = 0
        self.last_state = None

    def __call__(self, modulus):
        self.evaluations += 1
        state = _evaluate_inverse_state(
            self.mesh, self.cases[0].bc_info, None, self.material, 0.0,
            modulus, self.evaluations, observation_cases=self.cases)
        self.last_state = state
        return state['cost_tar'], state['grad_tar_E']


@dataclass(frozen=True)
class ExactTVOptions:
    max_outer: int = 300
    max_inner: int = 150
    max_evaluations: int = 12000
    stationarity_atol: float = 1e-9  # derivative w.r.t. log(E), raw objective
    stationarity_rtol: float = 1e-3
    primal_atol: float = 1e-6       # MPa/mm
    primal_rtol: float = 1e-4
    gap_atol: float = 1e-9          # same units as the raw objective
    gap_rtol: float = 1e-4
    adapt_interval: int = 10


def solve_exact_tv(mesh, cases, initial_modulus, gamma, nu=0.26,
                   options=None, callback=None, rho=None):
    """Return a candidate and certificates; never equate inner ftol with success.

    Area-weighted augmented term: rho/2 * sum a_q |G E-z+b|^2.
    Therefore the exact vector shrink threshold is gamma/rho (not gamma*a/rho).
    p=rho*b is the dual field; |p_q|<=gamma and the Fenchel gap plus
    data/dual stationarity certify approximate first-order optimality.
    Budget and failure states retain the last candidate for diagnosis only.
    """
    opts = options or ExactTVOptions()
    initial = np.asarray(initial_modulus, dtype=float)
    if (initial.shape != (mesh.n_nod,) or
            not np.all(np.isfinite(initial) & (initial > 0)) or
            not np.isfinite(gamma) or gamma <= 0):
        raise ValueError('Expected positive finite nodal moduli and gamma')
    if (min(opts.max_outer, opts.max_inner, opts.adapt_interval) < 1 or
            opts.max_evaluations < 2):
        raise ValueError('Iteration budgets must be positive; evaluations must be at least two')
    tolerances = (opts.stationarity_atol, opts.stationarity_rtol, opts.primal_atol,
                  opts.primal_rtol, opts.gap_atol, opts.gap_rtol)
    if not all(np.isfinite(t) and t > 0 for t in tolerances):
        raise ValueError('All certificate tolerances must be positive and finite')
    op = QuadratureGradient(mesh)
    oracle = FEDataOracle(mesh, cases, nu)
    # A numerical penalty initialization, not a prior or objective rescaling.
    if rho is None:
        rho = gamma * min(mesh.el, mesh.eh) / float(np.mean(initial))
    if not np.isfinite(rho) or rho <= 0:
        raise ValueError('ADMM penalty rho must be positive and finite')
    m = np.log(initial)
    z = op.apply(initial)
    b = np.zeros_like(z)
    history = []
    status = 'outer_budget'
    for outer in range(1, opts.max_outer + 1):
        remaining = opts.max_evaluations - oracle.evaluations
        if remaining < 2:
            status = 'evaluation_budget'
            break

        def augmented(log_modulus):
            modulus = np.exp(log_modulus)
            if not np.all(np.isfinite(modulus) & (modulus > 0)):
                return np.inf, np.zeros_like(log_modulus)
            value, gradient = oracle(modulus)
            mismatch = op.apply(modulus) - z + b
            penalty = .5 * rho * float(op.area @ np.sum(mismatch**2, axis=1))
            return value + penalty, (gradient + rho * op.adjoint(mismatch)) * modulus

        inner = minimize(augmented, m, jac=True, method='L-BFGS-B', options={
            'maxiter': opts.max_inner, 'maxfun': max(1, remaining-1),
            'gtol': .1 * opts.stationarity_atol, 'ftol': 0.0,
            'maxls': 40, 'maxcor': 20})
        m = inner.x
        modulus = np.exp(m)
        state = oracle.last_state
        if not np.array_equal(state['E_vec'], modulus):
            oracle(modulus)
            state = oracle.last_state
        gradient = op.apply(modulus)
        old_z = z.copy()
        z = vector_shrink(gradient + b, gamma/rho)
        b += gradient - z
        p = rho * b
        tv = op.value(modulus)
        data_grad_m = state['grad_tar_E'] * modulus
        dual_grad_m = op.adjoint(p) * modulus
        stationarity = float(np.linalg.norm(data_grad_m + dual_grad_m, np.inf))
        stationarity_tol = opts.stationarity_atol + opts.stationarity_rtol * max(
            np.linalg.norm(data_grad_m, np.inf), np.linalg.norm(dual_grad_m, np.inf))
        primal = op.rms(gradient-z)
        primal_tol = opts.primal_atol + opts.primal_rtol * max(op.rms(gradient), op.rms(z))
        gap = float(op.area @ (gamma*np.linalg.norm(gradient, axis=1) -
                              np.sum(p*gradient, axis=1)))
        gap_tol = opts.gap_atol + opts.gap_rtol * gamma * tv
        dual_violation = float(max(0.0, np.max(np.linalg.norm(p, axis=1))-gamma))
        dual_step = float(np.linalg.norm(rho * op.adjoint(z-old_z) * modulus, np.inf))
        row = dict(iteration=outer, evaluations=oracle.evaluations, rho=float(rho),
                   cost_data=float(state['cost_tar']), tv=tv,
                   objective=float(state['cost_tar']+gamma*tv),
                   stationarity=stationarity, stationarity_tol=float(stationarity_tol),
                   primal_rms=primal, primal_tol=primal_tol,
                   fenchel_gap=gap, gap_tol=gap_tol, dual_violation=dual_violation,
                   dual_step=dual_step, inner_message=str(inner.message),
                   inner_iterations=int(inner.nit))
        history.append(row)
        if callback:
            callback(row)
        if (stationarity <= stationarity_tol and primal <= primal_tol and
                abs(gap) <= gap_tol and dual_violation <= 1e-12*gamma):
            status = 'kkt_converged'
            break
        if oracle.evaluations >= opts.max_evaluations:
            status = 'evaluation_budget'
            break
        if outer % opts.adapt_interval == 0:
            primal_ratio = max(primal/primal_tol, abs(gap)/gap_tol)
            dual_ratio = dual_step / stationarity_tol
            factor = 2.0 if primal_ratio > 10*dual_ratio else (
                .5 if dual_ratio > 10*primal_ratio else 1.0)
            rho *= factor
            b /= factor  # preserve the unscaled dual p when rho changes
    return dict(E=modulus, dual=p, split_gradient=z, history=history,
                status=status, converged=(status == 'kkt_converged'),
                evaluations=oracle.evaluations, state=state,
                gamma=float(gamma), objective_coefficient='gamma*TV')
