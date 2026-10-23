"""Mean/fluctuation inverse metric and controlled projected L-BFGS driver.

The metric changes search directions only. The objective, log-coefficient box,
and original-coordinate projected-gradient criterion remain unchanged.
"""
from __future__ import annotations

import time
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.sparse import bmat, csr_matrix, diags
from scipy.sparse.linalg import splu

from .unified_inverse import UnifiedObjective, EvaluationBudgetReached


class MeanFluctuationMetric:
    def __init__(self, material_map, mean_hessian):
        self.map = material_map
        n, k = material_map.n_material, material_map.n_regions
        self.membership = np.zeros((n, k))
        mean = np.zeros((k, n))
        for i, segment in enumerate(material_map.slices):
            self.membership[segment, i] = 1.
            weights = material_map.support_mass[segment]
            mean[i, segment] = weights/weights.sum()
        self.mean = csr_matrix(mean)
        matrix = material_map.coefficient_laplacian
        matrix = (matrix+matrix.T)*.5
        # Repair only floating-point row-sum error in the auxiliary metric.
        # The objective's regional quadratic forms are never changed.
        correction = np.asarray(matrix.sum(axis=1)).ravel()
        self.row_sum_correction_max = float(abs(correction).max())
        matrix = matrix-diags(correction)
        self.factor = splu(bmat([[matrix, self.mean.T],
                                [self.mean, csr_matrix((k, k))]], format='csc'))
        hessian = np.asarray(mean_hessian, dtype=float)
        if hessian.shape != (k, k) or not np.isfinite(hessian).all():
            raise ValueError('Mean Hessian must be a finite regional square matrix')
        self.eigenvalues = np.linalg.eigvalsh(hessian)
        if self.eigenvalues[0] <= 1e-12*self.eigenvalues[-1]:
            raise ValueError('Regional mean responses are insufficient for this metric')
        self.coarse_factor = cho_factor(hessian)

    def apply(self, gradient, theta, gamma):
        if gamma <= 0:
            raise ValueError('Mean/fluctuation metric needs positive gamma')
        coefficients = np.exp(theta)
        rhs = gradient/coefficients
        fluctuation = self.factor.solve(np.r_[rhs, np.zeros(self.map.n_regions)])[:len(rhs)]
        mean_step = cho_solve(self.coarse_factor, self.membership.T @ rhs)
        result = (fluctuation/gamma+self.membership @ mean_step)/coefficients
        if float(gradient @ result) <= 0 and np.linalg.norm(gradient) > 0:
            raise FloatingPointError('Inverse metric lost positive definiteness')
        return result


def solve_projected_lbfgs(material_map, case, theta0, *, gamma, nu=.26,
                          modulus_bounds=(100., 6000.), max_iterations=400,
                          max_evaluations=600, gtol=1e-9, metric=None, progress=None):
    if not 0 < modulus_bounds[0] < modulus_bounds[1]:
        raise ValueError('Modulus bounds must be positive and strictly ordered')
    if max_iterations < 1 or max_evaluations < 1 or gtol <= 0:
        raise ValueError('Budgets and projected-gradient tolerance must be positive')
    lower, upper = np.log(modulus_bounds)
    theta = np.asarray(theta0, dtype=float).copy()
    material_map.unpack(theta)
    if np.any(theta < lower) or np.any(theta > upper):
        raise ValueError('Initial parameters violate coefficient bounds')
    objective = UnifiedObjective(material_map, case, nu=nu, gamma=gamma,
                                 max_evaluations=max_evaluations, progress=progress)
    state = objective.evaluate(theta)
    initial_value, start = state['value'], time.perf_counter()
    memory, iterations, fallback_count = [], 0, 0
    message = 'Reached iteration budget'
    while iterations < max_iterations:
        gradient = state['gradient']
        pg = theta-np.clip(theta-gradient, lower, upper)
        if np.max(abs(pg)) <= gtol:
            message = 'Original log-coordinate projected gradient passed'; break
        blocked = ((theta <= lower+1e-12) & (gradient > 0)) | ((theta >= upper-1e-12) & (gradient < 0))
        q = gradient.copy(); q[blocked] = 0
        alphas = []
        for s, y, inverse_sy in reversed(memory):
            alpha = inverse_sy*float(s @ q)
            alphas.append(alpha); q -= alpha*y
        if metric is None:
            scale = float(memory[-1][0] @ memory[-1][1]/(memory[-1][1] @ memory[-1][1])) if memory else 1.
            direction = scale*q
        else:
            direction = metric.apply(q, theta, gamma)
        for (s, y, inverse_sy), alpha in zip(memory, reversed(alphas)):
            direction += s*(alpha-inverse_sy*float(y @ direction))
        direction = -direction
        direction[blocked] = 0
        direction[((theta <= lower+1e-12) & (direction < 0)) |
                  ((theta >= upper-1e-12) & (direction > 0))] = 0
        if float(gradient @ direction) >= 0 or not np.isfinite(direction).all():
            direction = -pg; memory.clear(); fallback_count += 1
        accepted, step = False, 1.
        try:
            for _ in range(30):
                candidate = np.clip(theta+step*direction, lower, upper)
                displacement = candidate-theta
                slope = float(gradient @ displacement)
                if slope < 0 and np.any(displacement):
                    trial = objective.evaluate(candidate)
                    if trial['value'] <= state['value']+1e-4*slope:
                        accepted = True; break
                step *= .5
        except EvaluationBudgetReached:
            message = 'Reached forward-evaluation budget'; break
        if not accepted:
            message = 'Projected Armijo line search failed'; break
        s, y = displacement, trial['gradient']-gradient
        sy = float(s @ y)
        if sy > 1e-12*np.linalg.norm(s)*np.linalg.norm(y):
            memory.append((s, y, 1/sy)); memory = memory[-15:]
        theta, state = candidate, trial
        iterations += 1
    gradient = state['gradient']
    pg = theta-np.clip(theta-gradient, lower, upper)
    passed = bool(np.max(abs(pg)) <= gtol)
    evidence = dict(solver_success=passed, first_order_passed=passed, message=message,
        gradient_inf=float(abs(gradient).max()), projected_gradient_inf=float(abs(pg).max()),
        iterations=iterations, forward_evaluations=objective.evaluations,
        adjoint_evaluations=objective.evaluations, initial_value=initial_value,
        final_value=state['value'], elapsed_seconds=time.perf_counter()-start,
        final_is_accepted_iterate=True, descent_fallback_count=fallback_count,
        active_bounds=int(np.count_nonzero(np.isclose(theta, lower, rtol=0, atol=1e-8) |
                                          np.isclose(theta, upper, rtol=0, atol=1e-8))))
    return state, evidence, objective.history
