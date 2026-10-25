"""Axially varying nodal modulus with the existing two-dimensional FE kernel.

This optional constrained model repeats each axial modulus across every y row.
Only optimizer coordinates change: the forward problem, observations, rigid
registration and area-integrated H1 objective use inverse_solver unchanged.
"""

import numpy as np
from scipy.optimize import minimize

from .inverse_solver import _evaluate_inverse_state
from .material import MaterialInfo


def expand_axial(mesh, axial_values):
    """Map nodal values [nx+1] to the mesh's row-major [n_nodes] order."""
    values = np.asarray(axial_values, dtype=float)
    if values.shape != (mesh.nods_x,) or not np.isfinite(values).all():
        raise ValueError('Expected finite axial nodal values of shape (nel_x+1,)')
    if not np.allclose(mesh.plot_x, mesh.plot_x[0], rtol=0, atol=1e-7):
        raise ValueError('Axial parameterization requires aligned rectangular node columns')
    return np.tile(values, mesh.nods_y)


def pullback_axial(mesh, nodal_gradient):
    """Apply the transpose repetition map; sum, never average, y-row gradients."""
    gradient = np.asarray(nodal_gradient, dtype=float)
    if gradient.shape != (mesh.n_nod,):
        raise ValueError('Expected one full-field gradient per material node')
    return gradient.reshape(mesh.nods_y, mesh.nods_x).sum(axis=0)


def axial_h1_inverse_solver(mesh, *, observation_cases, E_init, gamma, nu,
                            max_iter, max_fun, ftol, gtol,
                            regularization_mode='gradient', regularization_options=None):
    """Minimize the original H1 objective in 122 (on 121x19) log-moduli.

For E=P exp(a), dJ/da=P.T dJ/dlog(E). The original two-dimensional integral
equals H*integral((dE/dx)^2 dx); its physical width factor is retained.
Full nodal fields are returned for the ordinary saved-field/plotting workflow.
The stopping gradient is explicitly the reduced-coordinate gradient.
"""
    if regularization_mode != 'gradient' or regularization_options:
        raise ValueError('The axial diagnostic currently supports original H1 only')
    if not np.isfinite(gamma) or gamma < 0:
        raise ValueError('gamma must be finite and nonnegative')
    cases = tuple(observation_cases)
    if not cases or any(c.observation.H.shape[1] != mesh.n_dof for c in cases):
        raise ValueError('Provide observations matching the two-dimensional FE mesh')
    initial = np.asarray(E_init, dtype=float)
    if initial.shape != (mesh.n_nod,) or not np.all(np.isfinite(initial) & (initial > 0)):
        raise ValueError('E_init must be a positive finite full nodal vector')
    axial_initial = initial.reshape(mesh.nods_y, mesh.nods_x)[0]
    if not np.allclose(initial, expand_axial(mesh, axial_initial), rtol=1e-12, atol=1e-10):
        raise ValueError('Axial initialization must already be constant across y')
    if mesh.M is None:
        mesh.assemble_mass_matrix()
    mesh.get_regularization_matrix()
    material = MaterialInfo(nu=nu)
    cache = {'a': None, 'state': None, 'evaluations': 0}

    def evaluate(a):
        if cache['a'] is None or not np.array_equal(a, cache['a']):
            cache['evaluations'] += 1
            state = _evaluate_inverse_state(
                mesh, cases[0].bc_info, None, material, gamma,
                expand_axial(mesh, np.exp(a)), cache['evaluations'],
                regularization_mode='gradient', observation_cases=cases)
            cache.update(a=np.array(a, copy=True), state=state)
        return cache['state']

    def objective(a):
        state = evaluate(a)
        return state['objective'], pullback_axial(mesh, state['grad_m'])

    a0 = np.log(axial_initial)
    initial_state = evaluate(a0)
    costs = [initial_state['objective']]
    data_costs = [initial_state['cost_tar']]
    reg_costs = [initial_state['cost_reg']]
    norms = [float(np.linalg.norm(pullback_axial(mesh, initial_state['grad_m'])))]

    def callback(a):
        state = evaluate(a)
        costs.append(state['objective'])
        data_costs.append(state['cost_tar'])
        reg_costs.append(state['cost_reg'])
        norms.append(float(np.linalg.norm(pullback_axial(mesh, state['grad_m']))))
        if (len(costs)-1) % 100 == 0:
            print(f"  axial iteration={len(costs)-1}, objective={costs[-1]:.7g}, "
                  f"FE evaluations={cache['evaluations']}", flush=True)

    result = minimize(objective, a0, jac=True, method='L-BFGS-B', callback=callback,
                      options=dict(maxiter=max_iter, maxfun=max_fun, ftol=ftol,
                                   gtol=gtol, maxls=50, maxcor=10))
    state = evaluate(result.x)
    gradient = pullback_axial(mesh, state['grad_m'])
    return dict(
        E_final=state['E_vec'], U_final=state['forw_U'],
        cost_history=np.asarray(costs), cost_tar_history=np.asarray(data_costs),
        cost_reg_history=np.asarray(reg_costs), gradient_norms=np.asarray(norms),
        grad_norm_history=np.asarray(norms), cond_H_history=np.ones(len(norms)),
        n_iterations=int(result.nit), n_evaluations=cache['evaluations'],
        converged=bool(result.success), message=str(result.message),
        final_cost=state['objective'], residual_norm=state['residual_norm'],
        regularization_norm=state['regularization_norm'], regularization_mode='gradient',
        final_gradient=gradient,
        final_grad_tar=pullback_axial(mesh, state['grad_tar_m']),
        final_grad_reg=pullback_axial(mesh, state['grad_reg_m']),
        case_states=state['case_states'], data_mode='observation_cases',
        material_parameterization='axial', inverse_parameter_count=mesh.nods_x,
        gradient_coordinates='independent axial log-moduli; y-row gradients summed',
        full_nodal_gradient_inf=float(np.max(np.abs(state['grad_m']))))
