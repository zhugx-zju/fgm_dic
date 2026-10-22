"""Small-scale region/Q4 inverse prototype using the production FEM/adjoint."""
from __future__ import annotations

import time
import numpy as np
from scipy.optimize import minimize

from .fem_forward import fem_assemble, forward_solver, gauss_stiffness_sensitivity
from .inverse_solver import lambda_distance
from .load_case import evaluate_observation
from .unified_material import MappedMaterial


class EvaluationBudgetReached(RuntimeError):
    pass


class UnifiedObjective:
    """One platform only; raw area-weighted cost plus gamma*R_intra/2."""

    def __init__(self, parameter_map, case, *, nu=0.26, gamma=0., max_evaluations=None,
                 progress=None):
        if not np.isfinite(gamma) or gamma < 0 or not 0.24 <= nu <= 0.33:
            raise ValueError('gamma must be nonnegative; nu must lie in [0.24, 0.33]')
        if case.observation.H.shape[1] != parameter_map.mesh.n_dof:
            raise ValueError('Observation and material meshes do not match')
        self.parameter_map, self.case = parameter_map, case
        self.nu, self.gamma = nu, gamma
        self.max_evaluations = max_evaluations
        self.evaluations = 0
        self.last_state = None
        self.history = []
        self.progress = progress

    def evaluate(self, theta):
        if self.max_evaluations is not None and self.evaluations >= self.max_evaluations:
            raise EvaluationBudgetReached('Reached the predeclared forward-evaluation budget')
        material = MappedMaterial(self.nu, self.parameter_map, theta)
        fem = fem_assemble(self.parameter_map.mesh, material, self.case.bc_info)
        u = forward_solver(fem)
        self.evaluations += 1
        obs = evaluate_observation(self.case, u)
        adjoint = lambda_distance(fem, rhs=obs['adjoint_rhs'])
        sensitivity = gauss_stiffness_sensitivity(fem, adjoint, u)
        grad_data = self.parameter_map.pullback(theta, sensitivity)
        reg, grad_reg = self.parameter_map.regularization(theta)
        gradient = grad_data + self.gamma*grad_reg/2
        state = dict(theta=np.asarray(theta).copy(), value=obs['cost_tar']+self.gamma*reg/2,
                     data_cost=obs['cost_tar'], regularization=reg, gradient=gradient,
                     data_gradient=grad_data, regularization_gradient=grad_reg,
                     U=u, prediction=obs['prediction'], residual=obs['residual'],
                     rigid_coefficients=obs['rigid_coefficients'])
        self.last_state = state
        self.history.append(dict(evaluation=self.evaluations, value=state['value'],
                                 data_cost=state['data_cost'], regularization=reg,
                                 gradient_inf=float(np.max(np.abs(gradient)))))
        if self.progress is not None and self.evaluations % 100 == 0:
            self.progress(self.history[-1])
        return state

    def __call__(self, theta):
        state = self.evaluate(theta)
        return state['value'], state['gradient']


def fit_uniform_initial_modulus(parameter_map, case, nu=0.26):
    """Fit a single scale to this case's projected displacement; no truth input.

    Force-controlled linear elasticity gives u(E)=u(1)/E. Project both data and
    the unit-modulus response onto the same weighted nonrigid subspace before
    fitting inverse modulus. This is initialization, not objective scaling.
    """
    material = MappedMaterial(nu, parameter_map, parameter_map.initial_parameters(1.))
    fem = fem_assemble(parameter_map.mesh, material, case.bc_info)
    unit = forward_solver(fem)
    d = evaluate_observation(case, np.zeros_like(unit))['residual']
    projected = d - evaluate_observation(case, unit)['residual']
    w = case.observation.weight_vector
    numerator, denominator = float(projected @ (w*projected)), float(projected @ (w*d))
    if denominator <= 0 or numerator <= 0:
        raise ValueError('Identification displacements cannot define a positive uniform starting modulus')
    return numerator/denominator


def solve_unified(parameter_map, case, initial_modulus, *, nu=0.26, gamma=0.,
                  modulus_bounds=(100., 6000.), geometry_bounds=None,
                  max_iterations=500, max_evaluations=800, ftol=1e-16, gtol=1e-10,
                  initial_parameters=None, progress=None):
    """Bounded L-BFGS-B with exact FE budget and independently reported evidence.

    Ordered disjoint control intervals are required for multiple moving
    interfaces. Bounds encode geometry feasibility, never a manufacturing
    truth in the unknown-interface protocol. Last accepted iterate is retained
    if a line search exhausts the budget; a trial field is never called final.
    """
    lower, upper = modulus_bounds
    if not 0 < lower < upper or not lower <= initial_modulus <= upper:
        raise ValueError('Initial modulus must lie inside positive declared bounds')
    if max_iterations < 1 or max_evaluations < 1 or ftol < 0 or gtol <= 0:
        raise ValueError('Budgets and gtol must be positive; ftol must be nonnegative')
    theta0 = (parameter_map.initial_parameters(initial_modulus) if initial_parameters is None
              else np.asarray(initial_parameters, dtype=float).copy())
    parameter_map.unpack(theta0)
    bounds = [(np.log(lower), np.log(upper))]*parameter_map.n_material
    if parameter_map.optimize_geometry:
        shape = parameter_map.interfaces.shape
        if geometry_bounds is None:
            if shape[0] != 1:
                raise ValueError('Provide disjoint geometry_bounds for multiple moving interfaces')
            geometry_bounds = [(0.1*parameter_map.mesh.geo_l, 0.9*parameter_map.mesh.geo_l)]*shape[1]
        gb = np.asarray(geometry_bounds, dtype=float).reshape(*shape, 2)
        if (not np.isfinite(gb).all() or np.any(gb[..., 0] >= gb[..., 1]) or
                np.any(gb[..., 0] <= 0) or np.any(gb[..., 1] >= parameter_map.mesh.geo_l) or
                np.any(gb[:-1, :, 1] >= gb[1:, :, 0])):
            raise ValueError('Geometry bounds must be ordered and disjoint inside the domain')
        bounds += [tuple(b) for b in gb.reshape(-1, 2)]
    b = np.asarray(bounds)
    if np.any(theta0 < b[:, 0]) or np.any(theta0 > b[:, 1]):
        raise ValueError('Initial parameters lie outside the declared bounds')
    objective = UnifiedObjective(parameter_map, case, nu=nu, gamma=gamma,
                                 max_evaluations=max_evaluations, progress=progress)
    accepted = objective.evaluate(theta0)
    initial_value = accepted['value']
    iterations = 0

    def callback(theta):
        nonlocal accepted, iterations
        iterations += 1
        if not np.array_equal(theta, objective.last_state['theta']):
            raise RuntimeError('Optimizer callback did not match the evaluated state')
        accepted = objective.last_state

    start = time.perf_counter()
    try:
        result = minimize(objective, theta0, jac=True, method='L-BFGS-B', bounds=bounds,
                          callback=callback, options=dict(maxiter=max_iterations,
                          maxfun=max_evaluations, ftol=ftol, gtol=gtol, maxls=30, maxcor=15))
        if np.array_equal(result.x, objective.last_state['theta']):
            accepted = objective.last_state
        elif not np.array_equal(result.x, accepted['theta']):
            raise RuntimeError('Final optimizer iterate has no cached accepted state')
        solver_success, message = bool(result.success), str(result.message)
    except EvaluationBudgetReached as exc:
        solver_success, message = False, str(exc)
    gradient = accepted['gradient']
    # L-BFGS-B's projected gradient uses a unit projected descent step.
    projected_gradient = accepted['theta'] - np.clip(accepted['theta']-gradient, b[:, 0], b[:, 1])
    evidence = dict(solver_success=solver_success, message=message,
                    first_order_passed=bool(np.max(np.abs(projected_gradient)) <= gtol),
                    gradient_inf=float(np.max(np.abs(gradient))),
                    projected_gradient_inf=float(np.max(np.abs(projected_gradient))),
                    iterations=iterations, forward_evaluations=objective.evaluations,
                    adjoint_evaluations=objective.evaluations,
                    initial_value=initial_value, final_value=accepted['value'],
                    elapsed_seconds=time.perf_counter()-start,
                    final_is_accepted_iterate=True,
                    active_bounds=int(np.sum(np.isclose(accepted['theta'], b[:, 0], rtol=0, atol=1e-8)
                                            | np.isclose(accepted['theta'], b[:, 1], rtol=0, atol=1e-8))))
    return accepted, evidence, objective.history
