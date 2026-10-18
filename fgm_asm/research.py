"""Shared forward, observation and prediction steps for the research workflows."""

import numpy as np

from .fem_forward import fem_assemble, forward_solver
from .load_case import ObservationCase, evaluate_observation
from .material import MaterialInfo
from .mesh import setup_boundary_conditions
from .observation import build_observation_package


def partition_modulus(moduli, boundaries):
    """Piecewise E(x) with explicitly configured physical partition boundaries.

    Sharp group values are evaluated at Gauss points in independent forward
    calculations. Nodal field inversion retains its continuous Q4 representation.
    """
    moduli = np.asarray(moduli, dtype=float)
    boundaries = np.asarray(boundaries, dtype=float)
    if moduli.shape != (len(boundaries) + 1,) or not np.all(
            np.isfinite(moduli) & (moduli > 0)):
        raise ValueError('Expected one positive modulus per partition')
    if not np.all(np.isfinite(boundaries)) or np.any(np.diff(boundaries) <= 0):
        raise ValueError('Partition boundaries must be finite and strictly increasing')

    def field(x, y):
        return moduli[np.searchsorted(boundaries, x, side='right')]

    return field


def solve_forward(mesh, force_N, thickness, nu, modulus):
    """Use the same two-end traction kernel for nodal or analytical modulus."""
    function = modulus if callable(modulus) else None
    values = function(mesh.X, mesh.Y) if function else np.asarray(modulus, dtype=float)
    if np.shape(values) != (mesh.n_nod,) or not np.all(np.isfinite(values) & (values > 0)):
        raise ValueError('Forward modulus must be a positive finite nodal field or callable')
    material = MaterialInfo(nu=nu, modulus_function=function)
    material.update(values)
    bc = setup_boundary_conditions(mesh, mesh.geo_l, mesh.geo_h, force_N, thickness)
    fem = fem_assemble(mesh, material, bc)
    return forward_solver(fem), bc


def make_observation_case(mesh, name, force_N, thickness, xy, uv,
                          point_weights=None, noise_std=None, metadata=None,
                          remove_rigid_body=True):
    """Construct identical pointwise inputs for simulations and DIC.

    Point weights represent area; noise_std is an optional component standard
    deviation in mm. The objective uses area/sigma^2, not area/sigma.
    """
    xy, uv = np.asarray(xy, dtype=float), np.asarray(uv, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2 or uv.shape != xy.shape:
        raise ValueError('Expected coordinates and displacement with shape (n_points,2)')
    component_weights = np.ones_like(uv)
    if noise_std is not None:
        sigma = np.broadcast_to(np.asarray(noise_std, dtype=float), uv.shape)
        if not np.all(np.isfinite(sigma) & (sigma > 0)):
            raise ValueError('Noise standard deviations must be positive and finite')
        component_weights = 1.0 / sigma**2
    package = build_observation_package(
        mesh, xy[:, 0], xy[:, 1], uv[:, 0], uv[:, 1], point_weights,
        sampling_mode='physical_points', ux_component_weight=component_weights[:, 0],
        uy_component_weight=component_weights[:, 1],
    )
    bc = setup_boundary_conditions(mesh, mesh.geo_l, mesh.geo_h, force_N, thickness)
    return ObservationCase(name, bc, package, remove_rigid_body, metadata or {})


def compare_displacement(mesh, modulus, cases, nu=0.3):
    """Compare a fixed material with observations, fitting only rigid motions."""
    states = []
    for case in cases:
        u, _ = solve_forward(mesh, case.bc_info['tensile_end_force'],
                             case.bc_info['thickness'], nu, modulus)
        state = evaluate_observation(case, u)
        state['U'] = u
        state['rmse_components_mm'] = np.sqrt(np.mean(
            state['residual'].reshape(-1, 2)**2, axis=0,
        ))
        states.append(state)
    return states
