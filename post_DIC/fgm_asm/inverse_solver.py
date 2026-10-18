"""
Inverse problem solver using SciPy L-BFGS-B with Tikhonov regularization.
"""

import numpy as np
from scipy.optimize import minimize

from .fem_forward import fem_assemble, forward_solver, solve_system, compute_tensile_end_force
from .observation import ObservationPackage
from .regularization import get_tikhonov_regularization, get_tikhonov_gradient


def lambda_distance(fem_info, forw_U=None, observation_package: ObservationPackage | None = None, rhs=None):
    """
    Compute the adjoint variable for the inverse problem.

    Args:
        fem_info: FEMInfo object
        forw_U: Forward solution displacement [n_dof]
        rhs: Optional precomputed adjoint right-hand side [n_dof]

    Returns:
        lambda_vec: Lagrange multiplier [n_dof]
    """
    if rhs is None:
        if forw_U is None:
            raise ValueError("Either rhs or forw_U must be provided.")
        rhs = get_data_misfit_adjoint_rhs(
            mesh_info=fem_info.mesh_info,
            forw_U=forw_U,
            observation_package=observation_package,
        )
    # The adjoint problem uses homogeneous Dirichlet conditions on the
    # prescribed displacement boundary, not the forward loading values.
    return solve_system(fem_info, rhs, prescribed_values=np.zeros(fem_info.mesh_info.n_dof, dtype=float))


def get_stiffness_gradient(fem_info, lambda_vec, forw_U):
    """
    Calculate the gradient of the objective function with respect to nodal modulus.

    Args:
        fem_info: FEMInfo object
        lambda_vec: Lagrange multiplier [n_dof]
        forw_U: Forward solution displacement [n_dof]

    Returns:
        grad: Gradient vector [n_nod]
    """
    mesh_info = fem_info.mesh_info
    ele_dof_id = mesh_info.ele_dof_id
    ele_nods_id = mesh_info.ele_nods_id - 1
    id_list = ele_nods_id.T.ravel()

    lambda_ele = lambda_vec[ele_dof_id]
    forw_U_ele = forw_U[ele_dof_id]

    grad_E = np.zeros((mesh_info.n_el, 4))

    for ig in range(mesh_info.gauss_N.shape[0]):
        B_i = fem_info.B_gauss[ig]
        BD0B = np.einsum('eia,ij,ejb,e->eab', B_i, fem_info.D0, B_i, fem_info.det_j_gauss[ig])
        directional_term = np.einsum('ea,eab,eb->e', lambda_ele, BD0B, forw_U_ele)
        grad_E += (mesh_info.gauss_w[ig] * directional_term)[:, None] * mesh_info.gauss_N[ig][None, :]

    return -np.bincount(id_list, weights=grad_E.T.ravel(), minlength=mesh_info.n_nod)


def _validate_ehat_bounds(ehat_min, ehat_max):
    lower = float(ehat_min)
    upper = float(ehat_max)
    if not (0.0 < lower < 1.0 < upper):
        raise ValueError("Require 0 < ehat_min < 1 < ehat_max to keep mean(Ehat) = 1.")
    return lower, upper


def _sigmoid_shifted_to_unit_mean(raw_vec, ehat_min, ehat_max):
    lower, upper = _validate_ehat_bounds(ehat_min, ehat_max)
    raw_vec = np.asarray(raw_vec, dtype=float)
    span = upper - lower
    target_sigmoid_mean = (1.0 - lower) / span

    def sigmoid_with_shift(shift):
        z = np.clip(raw_vec + shift, -60.0, 60.0)
        return 1.0 / (1.0 + np.exp(-z))

    lo = -80.0 - float(np.max(raw_vec))
    hi = 80.0 - float(np.min(raw_vec))
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if float(np.mean(sigmoid_with_shift(mid))) < target_sigmoid_mean:
            lo = mid
        else:
            hi = mid

    sigmoid = sigmoid_with_shift(0.5 * (lo + hi))
    return sigmoid, lower + span * sigmoid


def _raw_to_ehat(raw_vec, ehat_min=None, ehat_max=None):
    """
    Map the unconstrained optimization vector to a positive normalized field.

    Args:
        raw_vec: Unconstrained field parameters [n_nod]

    Returns:
        Ehat_vec: Positive field with unit mean [n_nod]
    """
    raw_vec = np.asarray(raw_vec, dtype=float)
    if ehat_min is not None or ehat_max is not None:
        if ehat_min is None or ehat_max is None:
            raise ValueError("ehat_min and ehat_max must be provided together.")
        _, ehat_vec = _sigmoid_shifted_to_unit_mean(raw_vec, ehat_min, ehat_max)
        return ehat_vec

    raw_shifted = raw_vec - np.max(raw_vec)
    exp_vec = np.exp(raw_shifted)
    return exp_vec / np.mean(exp_vec)


def _raw_value_for_ehat(target_ehat, ehat_min, ehat_max):
    lower = float(ehat_min)
    upper = float(ehat_max)
    target = float(target_ehat)
    if not (lower < target < upper):
        raise ValueError("target_ehat must lie strictly inside the Ehat bounds.")
    fraction = (target - lower) / (upper - lower)
    return float(np.log(fraction / (1.0 - fraction)))


def _transform_gradient_to_raw(raw_vec, grad_ehat, ehat_min=None, ehat_max=None):
    """
    Transform the gradient from normalized field space to raw parameter space.

    Args:
        raw_vec: Unconstrained field parameters [n_nod]
        grad_ehat: Gradient with respect to the normalized positive field [n_nod]

    Returns:
        grad_raw: Gradient with respect to the unconstrained parameters [n_nod]
    """
    if ehat_min is not None or ehat_max is not None:
        if ehat_min is None or ehat_max is None:
            raise ValueError("ehat_min and ehat_max must be provided together.")
        lower, upper = _validate_ehat_bounds(ehat_min, ehat_max)
        sigmoid, _ = _sigmoid_shifted_to_unit_mean(raw_vec, ehat_min, ehat_max)
        derivative = sigmoid * (1.0 - sigmoid)
        derivative_sum = float(np.sum(derivative))
        if derivative_sum <= 1e-15:
            return np.zeros_like(grad_ehat, dtype=float)
        weighted_mean = float(np.sum(grad_ehat * derivative) / derivative_sum)
        return (upper - lower) * derivative * (grad_ehat - weighted_mean)

    ehat_vec = _raw_to_ehat(raw_vec)
    weighted_mean = float(np.mean(grad_ehat * ehat_vec))
    return ehat_vec * (grad_ehat - weighted_mean)


def _recover_alpha_from_force(mesh_info, bc_info, material_info, ehat_vec, iteration, tensile_end_force):
    """
    Recover the global modulus scale from the measured tensile-end force.

    Args:
        mesh_info: MeshInfo object
        bc_info: Boundary condition dictionary
        material_info: MaterialInfo object
        ehat_vec: Positive normalized modulus field [n_nod]
        iteration: Iteration or evaluation identifier
        tensile_end_force: Measured resultant force on the loading edge

    Returns:
        alpha: Positive global modulus scale
        force_unit: Predicted tensile-end force for alpha = 1
    """
    material_info.update(ehat_vec, iteration=iteration)
    fem_info = fem_assemble(mesh_info, material_info, bc_info)
    forw_U = forward_solver(fem_info)
    force_unit = compute_tensile_end_force(fem_info, forw_U)

    if force_unit <= 0.0:
        raise ValueError("Recovered unit-force response must be positive for displacement-controlled tension.")
    if tensile_end_force <= 0.0:
        raise ValueError("Measured tensile-end force must be positive.")

    alpha = float(tensile_end_force / force_unit)
    return alpha, force_unit


def _compute_data_misfit_state(mesh_info, forw_U, observation_package: ObservationPackage):
    """
    Compute the data-misfit quantities for either nodal or pointwise observations.

    Returns a dictionary containing the residual vector, weighted residual,
    adjoint right-hand side, and scalar mismatch measures.
    """
    forw_U = np.asarray(forw_U, dtype=float).ravel()

    measurement_prediction = np.asarray(observation_package.H @ forw_U).ravel()
    residual = observation_package.u_measured - measurement_prediction
    weighted_residual = np.asarray(observation_package.W @ residual).ravel()
    adjoint_rhs = -np.asarray(observation_package.H.T @ weighted_residual).ravel()
    data_mode = f"observation_operator:{observation_package.sampling_mode}"
    n_measurements = observation_package.x_meas.size

    weight_vector = np.asarray(observation_package.weight_vector, dtype=float).ravel()
    weight_sum = float(np.sum(weight_vector))
    if weight_sum <= 0.0:
        raise ValueError("Observation weights must have a positive sum.")

    residual_energy = max(float(residual @ weighted_residual), 0.0)
    measured_weighted = np.asarray(observation_package.W @ observation_package.u_measured).ravel()
    measured_energy = max(float(observation_package.u_measured @ measured_weighted), 0.0)

    cost_tar = 0.5 * residual_energy
    weighted_rms_residual = float(np.sqrt(max(residual_energy / weight_sum, 0.0)))
    measurement_weighted_rms = float(np.sqrt(max(measured_energy / weight_sum, 0.0)))
    relative_weighted_rms_residual = weighted_rms_residual / max(measurement_weighted_rms, 1e-15)
    return {
        "measurement_prediction": measurement_prediction,
        "residual": residual,
        "weighted_residual": weighted_residual,
        "adjoint_rhs": adjoint_rhs,
        "cost_tar": cost_tar,
        "residual_norm": float(np.sqrt(max(2.0 * cost_tar, 0.0))),
        "weighted_rms_residual": weighted_rms_residual,
        "measurement_weighted_rms": measurement_weighted_rms,
        "relative_weighted_rms_residual": float(relative_weighted_rms_residual),
        "weight_sum": weight_sum,
        "data_mode": data_mode,
        "n_measurements": int(n_measurements),
    }


def get_data_misfit_adjoint_rhs(mesh_info, forw_U, observation_package: ObservationPackage):
    """Return the adjoint right-hand side associated with the current data term."""
    return _compute_data_misfit_state(
        mesh_info=mesh_info,
        forw_U=forw_U,
        observation_package=observation_package,
    )["adjoint_rhs"]


def _evaluate_inverse_state(mesh_info, bc_info, observation_package, material_info, gamma,
                            raw_vec, iteration, ehat_min=None, ehat_max=None):
    """
    Evaluate the inverse objective, gradient, and reusable solver state.

    Args:
        mesh_info: MeshInfo object
        bc_info: Boundary condition dictionary
        observation_package: Pointwise observation package
        material_info: MaterialInfo object
        gamma: Regularization coefficient
        raw_vec: Unconstrained positive-field parameters [n_nod]
        iteration: Iteration or evaluation identifier

    Returns:
        state: Dictionary with objective, gradient, and final solver state
    """
    ehat_vec = _raw_to_ehat(raw_vec, ehat_min=ehat_min, ehat_max=ehat_max)
    material_info.update(ehat_vec, iteration=iteration)
    fem_info = fem_assemble(mesh_info, material_info, bc_info)
    forw_U = forward_solver(fem_info)

    data_state = _compute_data_misfit_state(
        mesh_info=mesh_info,
        forw_U=forw_U,
        observation_package=observation_package,
    )
    residual = data_state["residual"]
    cost_tar = data_state["cost_tar"]

    reg_value = get_tikhonov_regularization(mesh_info, material_info)
    cost_reg = 0.5 * gamma * reg_value

    lambda_vec = lambda_distance(fem_info, rhs=data_state["adjoint_rhs"])
    grad_tar_ehat = get_stiffness_gradient(fem_info, lambda_vec, forw_U)
    grad_reg_ehat = 0.5 * gamma * get_tikhonov_gradient(mesh_info, material_info)
    grad_ehat = grad_tar_ehat + grad_reg_ehat
    grad_tar_raw = _transform_gradient_to_raw(raw_vec, grad_tar_ehat, ehat_min=ehat_min, ehat_max=ehat_max)
    grad_reg_raw = _transform_gradient_to_raw(raw_vec, grad_reg_ehat, ehat_min=ehat_min, ehat_max=ehat_max)
    grad_raw = grad_tar_raw + grad_reg_raw

    return {
        'raw_vec': np.array(raw_vec, copy=True),
        'Ehat_vec': np.array(ehat_vec, copy=True),
        'E_vec': np.array(ehat_vec, copy=True),
        'fem_info': fem_info,
        'forw_U': forw_U,
        'measurement_prediction': data_state["measurement_prediction"],
        'lambda_vec': lambda_vec,
        'residual': residual,
        'weighted_residual': data_state["weighted_residual"],
        'cost_tar': cost_tar,
        'cost_reg': cost_reg,
        'objective': cost_tar + cost_reg,
        'reg_value': reg_value,
        'residual_norm': data_state["residual_norm"],
        'weighted_rms_residual': data_state["weighted_rms_residual"],
        'measurement_weighted_rms': data_state["measurement_weighted_rms"],
        'relative_weighted_rms_residual': data_state["relative_weighted_rms_residual"],
        'weight_sum': data_state["weight_sum"],
        'regularization_norm': float(np.sqrt(max(reg_value, 0.0))),
        'data_mode': data_state["data_mode"],
        'n_measurements': data_state["n_measurements"],
        'grad_tar_Ehat': grad_tar_ehat,
        'grad_reg_Ehat': grad_reg_ehat,
        'grad_Ehat': grad_ehat,
        'grad_tar_raw': grad_tar_raw,
        'grad_reg_raw': grad_reg_raw,
        'grad_raw': grad_raw,
    }


def lbfgs_inverse_solver_scipy(mesh_info, bc_info, observation_package: ObservationPackage,
                               tensile_end_force,
                               raw_init=None, gamma=1e-6, E_max=1000.0,
                               max_iter=2000, ftol=1e-12, gtol=1e-8, nu=0.3,
                               ehat_min=None, ehat_max=None):
    """
    Solve the inverse problem using a positive normalized modulus field Ehat.

    Args:
        mesh_info: MeshInfo object
        bc_info: Boundary condition dictionary
        observation_package: Pointwise observation package
        tensile_end_force: Measured resultant force on the loading edge
        raw_init: Initial unconstrained field parameters [n_nod]
        gamma: Regularization coefficient
        E_max: Maximum modulus (kept for compatibility)
        max_iter: Maximum iterations
        ftol: Function tolerance
        gtol: Gradient tolerance in raw-parameter space
        nu: Poisson's ratio

    Returns:
        results: Dictionary containing optimization results and cached final state
    """
    del E_max
    from .material import MaterialInfo

    mesh_info.get_regularization_matrix()

    if raw_init is None:
        if ehat_min is not None or ehat_max is not None:
            raw_vec0 = np.full(
                mesh_info.n_nod,
                _raw_value_for_ehat(1.0, ehat_min, ehat_max),
                dtype=float,
            )
        else:
            raw_vec0 = np.zeros(mesh_info.n_nod)
    else:
        raw_vec0 = np.asarray(raw_init, dtype=float).copy()

    ehat_vec0 = _raw_to_ehat(raw_vec0, ehat_min=ehat_min, ehat_max=ehat_max)
    material_info = MaterialInfo(nu=nu)

    print("Starting L-BFGS-B optimization for the normalized positive modulus field...")
    print(f"  Initial Ehat: min={ehat_vec0.min():.4f}, max={ehat_vec0.max():.4f}, mean={ehat_vec0.mean():.4f}")
    if ehat_min is not None or ehat_max is not None:
        print(f"  Ehat bounds: [{float(ehat_min):.4f}, {float(ehat_max):.4f}]")
    print(f"  Max iterations: {max_iter}, ftol: {ftol:.2e}, gtol: {gtol:.2e}")
    print(f"  Data term: pointwise observations ({observation_package.sampling_mode}), n = {observation_package.x_meas.size}")

    evaluation_counter = [0]
    state_cache = {'raw': np.array(raw_vec0, copy=True), 'state': None}

    initial_state = _evaluate_inverse_state(
        mesh_info, bc_info, observation_package, material_info, gamma,
        raw_vec0, iteration=1, ehat_min=ehat_min, ehat_max=ehat_max
    )
    state_cache['state'] = initial_state

    cost_history = [initial_state['objective']]
    cost_tar_history = [initial_state['cost_tar']]
    cost_reg_history = [initial_state['cost_reg']]
    gradient_norms = [np.linalg.norm(initial_state['grad_raw'])]
    print(
        "  Initial misfit: "
        f"cost = {initial_state['objective']:.6e}, "
        f"weighted RMS = {initial_state['weighted_rms_residual']:.6e}, "
        f"relative = {100.0 * initial_state['relative_weighted_rms_residual']:.2f}%"
    )

    def objective_and_gradient(raw_vec_flat):
        raw_vec_use = np.array(raw_vec_flat, copy=False)
        evaluation_counter[0] += 1
        state = _evaluate_inverse_state(
            mesh_info,
            bc_info,
            observation_package,
            material_info,
            gamma,
            raw_vec_use,
            iteration=evaluation_counter[0] + 1,
            ehat_min=ehat_min,
            ehat_max=ehat_max,
        )
        state_cache['raw'] = np.array(raw_vec_use, copy=True)
        state_cache['state'] = state
        return state['objective'], state['grad_raw']

    def callback(xk):
        raw_current = np.array(xk, copy=False)
        state = state_cache['state']

        if state is None or state_cache['raw'] is None or not np.array_equal(raw_current, state_cache['raw']):
            state = _evaluate_inverse_state(
                mesh_info,
                bc_info,
                observation_package,
                material_info,
                gamma,
                raw_current,
                iteration=evaluation_counter[0] + 1,
                ehat_min=ehat_min,
                ehat_max=ehat_max,
            )
            state_cache['raw'] = np.array(raw_current, copy=True)
            state_cache['state'] = state

        cost_history.append(state['objective'])
        cost_tar_history.append(state['cost_tar'])
        cost_reg_history.append(state['cost_reg'])
        gradient_norms.append(np.linalg.norm(state['grad_raw']))

        accepted_iterations = len(cost_history) - 1
        if accepted_iterations == 1 or (accepted_iterations > 0 and accepted_iterations % 10 == 0):
            print(f"  Iteration {accepted_iterations:4d}: "
                  f"Cost = {state['objective']:.6e}, "
                  f"weighted RMS = {state['weighted_rms_residual']:.6e}, "
                  f"relative = {100.0 * state['relative_weighted_rms_residual']:.2f}%, "
                  f"||grad_raw|| = {gradient_norms[-1]:.6e}")

    result = minimize(
        fun=objective_and_gradient,
        x0=raw_vec0,
        method='L-BFGS-B',
        jac=True,
        options={
            'maxiter': max_iter,
            'ftol': ftol,
            'gtol': gtol,
            'maxls': 50,
            'maxcor': 10,
        },
        callback=callback,
    )

    if state_cache['raw'] is not None and np.array_equal(result.x, state_cache['raw']):
        final_state = state_cache['state']
    else:
        final_state = _evaluate_inverse_state(
            mesh_info,
            bc_info,
            observation_package,
            material_info,
            gamma,
            result.x,
            iteration=evaluation_counter[0] + 1,
            ehat_min=ehat_min,
            ehat_max=ehat_max,
        )

    Ehat_final = final_state['Ehat_vec']
    # The global scale alpha is not part of the optimization variables.
    # It is recovered once from the measured tensile-end force after Ehat converges.
    alpha_final, force_unit_final = _recover_alpha_from_force(
        mesh_info,
        bc_info,
        material_info,
        Ehat_final,
        evaluation_counter[0] + 2,
        tensile_end_force,
    )
    E_final = alpha_final * Ehat_final

    print(f"  Optimization completed: {result.message}")
    print(f"  Solver stats: nit = {result.nit}, nfev = {result.nfev}, njev = {result.njev}")
    print(
        "  Final misfit: "
        f"cost = {final_state['objective']:.6e}, "
        f"weighted RMS = {final_state['weighted_rms_residual']:.6e}, "
        f"relative = {100.0 * final_state['relative_weighted_rms_residual']:.2f}%"
    )
    print(f"  Final alpha: {alpha_final:.6e}")
    print(f"  Final Ehat: min={Ehat_final.min():.4f}, max={Ehat_final.max():.4f}, mean={Ehat_final.mean():.4f}")
    print(f"  Final modulus: min={E_final.min():.4f}, max={E_final.max():.4f}, mean={E_final.mean():.4f}")

    return {
        'E_final': E_final,
        'Ehat_final': Ehat_final,
        'alpha_final': alpha_final,
        'raw_final': np.array(final_state['raw_vec'], copy=True),
        'U_final': final_state['forw_U'],
        'cost_history': np.array(cost_history),
        'cost_tar_history': np.array(cost_tar_history),
        'cost_reg_history': np.array(cost_reg_history),
        'grad_norm_history': np.array(gradient_norms),
        'gradient_norms': np.array(gradient_norms),
        'cond_H_history': np.ones(len(gradient_norms)),
        'n_iterations': int(result.nit),
        'converged': bool(result.success),
        'message': str(result.message),
        'final_cost': final_state['objective'],
        'force_unit_final': force_unit_final,
        'force_target': float(tensile_end_force),
        'residual_norm': final_state['residual_norm'],
        'weighted_rms_residual': final_state['weighted_rms_residual'],
        'measurement_weighted_rms': final_state['measurement_weighted_rms'],
        'relative_weighted_rms_residual': final_state['relative_weighted_rms_residual'],
        'weight_sum': final_state['weight_sum'],
        'regularization_norm': final_state['regularization_norm'],
        'data_mode': final_state['data_mode'],
        'n_measurements': final_state['n_measurements'],
        'measurement_prediction_final': final_state['measurement_prediction'],
        'final_gradient': final_state['grad_raw'],
        'final_grad_tar': final_state['grad_tar_raw'],
        'final_grad_reg': final_state['grad_reg_raw'],
    }
