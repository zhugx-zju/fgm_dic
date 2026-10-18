"""
Strong-stabilized CASE inverse solver.

This is an experimental implementation of the BC-free Coupled Adjoint-State
Equation idea from Seidl et al. The state displacement is not solved from
prescribed displacement or traction boundary conditions. Instead, it is the
closest measured-displacement fit that satisfies interior equilibrium.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.sparse import bmat, csc_matrix
from scipy.sparse.linalg import spsolve

from .fem_forward import fem_assemble
from .inverse_solver import (
    _raw_to_ehat,
    _recover_alpha_from_force,
    _transform_gradient_to_raw,
)
from .material import MaterialInfo
from .regularization import get_tikhonov_gradient, get_tikhonov_regularization


def _boundary_dofs(mesh_info):
    x = mesh_info.X
    y = mesh_info.Y
    boundary_nodes = np.where(
        np.isclose(x, 0.0)
        | np.isclose(x, mesh_info.geo_l)
        | np.isclose(y, 0.0)
        | np.isclose(y, mesh_info.geo_h)
    )[0]
    dofs = np.empty(2 * boundary_nodes.size, dtype=np.int32)
    dofs[0::2] = 2 * boundary_nodes
    dofs[1::2] = 2 * boundary_nodes + 1
    return np.sort(dofs)


def _interior_adjoint_dofs(mesh_info):
    boundary = _boundary_dofs(mesh_info)
    mask = np.ones(mesh_info.n_dof, dtype=bool)
    mask[boundary] = False
    return np.flatnonzero(mask)


def _case_stiffness_gradient(mesh_info, material_info, u_vec, w_full):
    D0 = material_info.get_elasticity_matrix()
    ele_dof_id = mesh_info.ele_dof_id
    ele_nods_id = mesh_info.ele_nods_id - 1
    id_list = ele_nods_id.T.ravel()

    u_ele = u_vec[ele_dof_id]
    w_ele = w_full[ele_dof_id]
    grad_E = np.zeros((mesh_info.n_el, 4))

    for ig in range(mesh_info.gauss_N.shape[0]):
        B_i = mesh_info.B_gauss[ig]
        BD0B = np.einsum("eia,ij,ejb,e->eab", B_i, D0, B_i, mesh_info.gauss_det_j[ig])
        directional_term = np.einsum("ea,eab,eb->e", w_ele, BD0B, u_ele)
        grad_E += mesh_info.gauss_w[ig] * directional_term[:, None] * mesh_info.gauss_N[ig][None, :]

    return np.bincount(id_list, weights=grad_E.T.ravel(), minlength=mesh_info.n_nod)


def evaluate_case_inverse_state(
    mesh_info,
    bc_info,
    observation_package,
    tensile_end_force,
    gamma,
    raw_vec,
    nu,
    ehat_min=None,
    ehat_max=None,
    iteration=1,
):
    ehat_vec = _raw_to_ehat(raw_vec, ehat_min=ehat_min, ehat_max=ehat_max)
    material_info = MaterialInfo(nu=nu)
    material_info.update(ehat_vec, iteration=iteration)
    fem_info = fem_assemble(mesh_info, material_info, {"fixdof": np.array([], dtype=np.int32), "disp": np.zeros(mesh_info.n_dof)})

    H = observation_package.H
    W = observation_package.W
    u_measured = observation_package.u_measured
    data_matrix = H.T @ W @ H
    data_rhs = np.asarray(H.T @ (W @ u_measured)).ravel()

    interior = _interior_adjoint_dofs(mesh_info)
    K_int = fem_info.K[interior, :]
    zero = csc_matrix((interior.size, interior.size))
    system = bmat([[data_matrix.tocsc(), K_int.T.tocsc()], [K_int.tocsc(), zero]], format="csc")
    rhs = np.concatenate([data_rhs, np.zeros(interior.size, dtype=float)])
    solution = spsolve(system, rhs)

    u_vec = np.asarray(solution[: mesh_info.n_dof]).ravel()
    w_int = np.asarray(solution[mesh_info.n_dof :]).ravel()
    w_full = np.zeros(mesh_info.n_dof, dtype=float)
    w_full[interior] = w_int

    prediction = np.asarray(H @ u_vec).ravel()
    residual = u_measured - prediction
    weighted_residual = np.asarray(W @ residual).ravel()
    residual_energy = max(float(residual @ weighted_residual), 0.0)
    measured_weighted = np.asarray(W @ u_measured).ravel()
    measured_energy = max(float(u_measured @ measured_weighted), 0.0)
    weight_sum = float(np.sum(observation_package.weight_vector))

    reg_value = get_tikhonov_regularization(mesh_info, material_info)
    cost_tar = 0.5 * residual_energy
    cost_reg = 0.5 * gamma * reg_value

    grad_tar_ehat = _case_stiffness_gradient(mesh_info, material_info, u_vec, w_full)
    grad_reg_ehat = 0.5 * gamma * get_tikhonov_gradient(mesh_info, material_info)
    grad_ehat = grad_tar_ehat + grad_reg_ehat
    grad_raw = _transform_gradient_to_raw(raw_vec, grad_ehat, ehat_min=ehat_min, ehat_max=ehat_max)

    alpha_final, _force_unit = _recover_alpha_from_force(
        mesh_info,
        bc_info,
        material_info,
        ehat_vec,
        iteration + 1,
        tensile_end_force,
    )

    return {
        "raw_vec": np.array(raw_vec, copy=True),
        "Ehat_vec": ehat_vec,
        "E_vec": alpha_final * ehat_vec,
        "alpha_final": alpha_final,
        "U_case": u_vec,
        "W_case": w_full,
        "measurement_prediction": prediction,
        "residual": residual,
        "weighted_residual": weighted_residual,
        "cost_tar": cost_tar,
        "cost_reg": cost_reg,
        "objective": cost_tar + cost_reg,
        "reg_value": reg_value,
        "grad_raw": grad_raw,
        "grad_Ehat": grad_ehat,
        "weighted_rms_residual": float(np.sqrt(max(residual_energy / weight_sum, 0.0))),
        "relative_weighted_rms_residual": float(
            np.sqrt(max(residual_energy, 0.0)) / max(np.sqrt(measured_energy), 1e-15)
        ),
        "regularization_norm": float(np.sqrt(max(reg_value, 0.0))),
    }


def case_inverse_solver_scipy(
    mesh_info,
    bc_info,
    observation_package,
    tensile_end_force,
    gamma,
    max_iter=200,
    ftol=1e-12,
    gtol=1e-7,
    nu=0.3,
    ehat_min=None,
    ehat_max=None,
):
    if ehat_min is not None:
        from .inverse_solver import _raw_value_for_ehat

        raw0 = np.full(mesh_info.n_nod, _raw_value_for_ehat(1.0, ehat_min, ehat_max), dtype=float)
    else:
        raw0 = np.zeros(mesh_info.n_nod, dtype=float)

    cache = {"raw": None, "state": None}
    history = {"cost": [], "cost_tar": [], "cost_reg": [], "grad": []}
    counter = [0]

    def objective_and_gradient(raw):
        counter[0] += 1
        state = evaluate_case_inverse_state(
            mesh_info,
            bc_info,
            observation_package,
            tensile_end_force,
            gamma,
            raw,
            nu,
            ehat_min=ehat_min,
            ehat_max=ehat_max,
            iteration=counter[0],
        )
        cache["raw"] = np.array(raw, copy=True)
        cache["state"] = state
        return state["objective"], state["grad_raw"]

    def callback(xk):
        state = cache["state"]
        if state is None or not np.array_equal(xk, cache["raw"]):
            state = evaluate_case_inverse_state(
                mesh_info,
                bc_info,
                observation_package,
                tensile_end_force,
                gamma,
                xk,
                nu,
                ehat_min=ehat_min,
                ehat_max=ehat_max,
                iteration=counter[0] + 1,
            )
        history["cost"].append(state["objective"])
        history["cost_tar"].append(state["cost_tar"])
        history["cost_reg"].append(state["cost_reg"])
        history["grad"].append(float(np.linalg.norm(state["grad_raw"])))
        accepted = len(history["cost"])
        if accepted == 1 or accepted % 10 == 0:
            print(
                f"  CASE iter {accepted:4d}: cost={state['objective']:.6e}, "
                f"rel={100.0 * state['relative_weighted_rms_residual']:.2f}%, "
                f"||grad||={history['grad'][-1]:.6e}"
            )

    print("Starting strong-stabilized CASE inverse solve...")
    result = minimize(
        fun=objective_and_gradient,
        x0=raw0,
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": max_iter, "ftol": ftol, "gtol": gtol, "maxls": 50, "maxcor": 10},
        callback=callback,
    )

    final_state = evaluate_case_inverse_state(
        mesh_info,
        bc_info,
        observation_package,
        tensile_end_force,
        gamma,
        result.x,
        nu,
        ehat_min=ehat_min,
        ehat_max=ehat_max,
        iteration=counter[0] + 1,
    )
    print(f"  CASE completed: {result.message}")
    print(
        f"  Final Ehat: min={final_state['Ehat_vec'].min():.4f}, "
        f"max={final_state['Ehat_vec'].max():.4f}, mean={final_state['Ehat_vec'].mean():.4f}"
    )

    return {
        "E_final": final_state["E_vec"],
        "Ehat_final": final_state["Ehat_vec"],
        "alpha_final": final_state["alpha_final"],
        "U_final": final_state["U_case"],
        "W_case": final_state["W_case"],
        "measurement_prediction_final": final_state["measurement_prediction"],
        "residual": final_state["residual"],
        "cost_history": np.array(history["cost"]),
        "cost_tar_history": np.array(history["cost_tar"]),
        "cost_reg_history": np.array(history["cost_reg"]),
        "grad_norm_history": np.array(history["grad"]),
        "n_iterations": int(result.nit),
        "converged": bool(result.success),
        "message": str(result.message),
        "final_cost": final_state["objective"],
        "weighted_rms_residual": final_state["weighted_rms_residual"],
        "relative_weighted_rms_residual": final_state["relative_weighted_rms_residual"],
        "regularization_norm": final_state["regularization_norm"],
        "final_gradient": final_state["grad_raw"],
    }
