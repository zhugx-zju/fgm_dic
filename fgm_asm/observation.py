"""
Observation-operator helpers for comparing FE predictions with measured DIC data.

The key idea is to keep the measured displacement values at real DIC points and
let a sparse observation operator ``H`` evaluate the FE displacement field at
those same physical locations.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
from scipy.sparse import csr_matrix, diags, load_npz, save_npz

from .mesh import MeshInfo


@dataclass(frozen=True)
class ObservationPackage:
    """Prepared measurement data and the operators used in the data-misfit term."""

    x_meas: np.ndarray
    y_meas: np.ndarray
    ux_meas: np.ndarray
    uy_meas: np.ndarray
    u_measured: np.ndarray
    point_weights: np.ndarray
    weight_vector: np.ndarray
    H: csr_matrix
    W: csr_matrix
    sampling_mode: str


def load_preprocessed_dic_grids(grid_paths: Mapping[str, Path]) -> dict[str, np.ndarray]:
    """
    Load the structured DIC grids saved by the preprocessing workflow.

    The returned arrays are oriented so that x increases left-to-right and
    y increases bottom-to-top, which matches the FE mesh convention.
    """
    required_keys = ("x_grid", "y_grid", "ux_grid", "uy_grid")
    missing = [key for key in required_keys if key not in grid_paths]
    if missing:
        raise KeyError(f"Missing required grid paths: {missing}")

    x_grid = np.load(grid_paths["x_grid"])
    y_grid = np.load(grid_paths["y_grid"])
    ux_grid = np.load(grid_paths["ux_grid"])
    uy_grid = np.load(grid_paths["uy_grid"])

    _validate_grid_shapes(x_grid, y_grid, ux_grid, uy_grid)
    x_grid, y_grid, ux_grid, uy_grid = orient_structured_grid_to_fe_basis(
        x_grid=x_grid,
        y_grid=y_grid,
        ux_grid=ux_grid,
        uy_grid=uy_grid,
    )

    return {
        "x_grid": x_grid,
        "y_grid": y_grid,
        "ux_grid": ux_grid,
        "uy_grid": uy_grid,
    }


def orient_structured_grid_to_fe_basis(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    ux_grid: np.ndarray,
    uy_grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Flip the structured grids so that x and y both increase."""
    _validate_grid_shapes(x_grid, y_grid, ux_grid, uy_grid)

    x_use = np.array(x_grid, copy=True)
    y_use = np.array(y_grid, copy=True)
    ux_use = np.array(ux_grid, copy=True)
    uy_use = np.array(uy_grid, copy=True)

    if np.mean(np.diff(x_use[0, :])) < 0.0:
        x_use = np.fliplr(x_use)
        y_use = np.fliplr(y_use)
        ux_use = np.fliplr(ux_use)
        uy_use = np.fliplr(uy_use)

    if np.mean(np.diff(y_use[:, 0])) < 0.0:
        x_use = np.flipud(x_use)
        y_use = np.flipud(y_use)
        ux_use = np.flipud(ux_use)
        uy_use = np.flipud(uy_use)

    return x_use, y_use, ux_use, uy_use


def sample_dic_on_uniform_subgrid(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    ux_grid: np.ndarray,
    uy_grid: np.ndarray,
    n_x: int,
    n_y: int,
) -> dict[str, np.ndarray]:
    """
    Select a uniform subgrid of real DIC points without changing their values.

    This is the simplest way to keep only true measurements while reducing the
    number of comparison points.
    """
    _validate_grid_shapes(x_grid, y_grid, ux_grid, uy_grid)
    if n_x < 2 or n_y < 2:
        raise ValueError("n_x and n_y must both be at least 2.")

    n_cols = x_grid.shape[1]
    n_rows = x_grid.shape[0]
    ix = np.rint(np.linspace(0, n_cols - 1, n_x)).astype(int)
    iy = np.rint(np.linspace(0, n_rows - 1, n_y)).astype(int)

    x_meas_grid = x_grid[np.ix_(iy, ix)]
    y_meas_grid = y_grid[np.ix_(iy, ix)]
    ux_meas_grid = ux_grid[np.ix_(iy, ix)]
    uy_meas_grid = uy_grid[np.ix_(iy, ix)]

    x_axis = x_meas_grid[0, :]
    y_axis = y_meas_grid[:, 0]
    wx = compute_axis_cell_widths(x_axis)
    wy = compute_axis_cell_widths(y_axis)
    point_weights = np.outer(wy, wx).ravel()

    return {
        "x_meas": x_meas_grid.ravel(),
        "y_meas": y_meas_grid.ravel(),
        "ux_meas": ux_meas_grid.ravel(),
        "uy_meas": uy_meas_grid.ravel(),
        "point_weights": point_weights,
        "grid_row_indices": iy,
        "grid_col_indices": ix,
    }


def get_fe_gauss_targets(mesh_info: MeshInfo) -> dict[str, np.ndarray]:
    """
    Build physical target positions at the FE Gauss points.

    These targets are useful when the user wants a smaller set of measurement
    points that still respects the FE integration layout.
    """
    x_target = np.column_stack(
        [np.sum(mesh_info.x_ele * shape_values[None, :], axis=1) for shape_values in mesh_info.gauss_N]
    ).ravel()
    y_target = np.column_stack(
        [np.sum(mesh_info.y_ele * shape_values[None, :], axis=1) for shape_values in mesh_info.gauss_N]
    ).ravel()
    point_weights = (mesh_info.gauss_abs_det_j.T * mesh_info.gauss_w[None, :]).ravel()
    return {
        "x_target": x_target,
        "y_target": y_target,
        "point_weights": point_weights,
    }


def sample_dic_nearest_to_targets(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    ux_grid: np.ndarray,
    uy_grid: np.ndarray,
    x_target: np.ndarray,
    y_target: np.ndarray,
    point_weights: np.ndarray | None = None,
    collapse_duplicates: bool = True,
) -> dict[str, np.ndarray]:
    """
    Snap arbitrary target positions to the nearest real DIC points.

    This keeps the measurement values untouched. If several targets snap to the
    same DIC point, their weights can be merged into one observation.
    """
    _validate_grid_shapes(x_grid, y_grid, ux_grid, uy_grid)

    x_target = np.asarray(x_target, dtype=float).ravel()
    y_target = np.asarray(y_target, dtype=float).ravel()
    if x_target.shape != y_target.shape:
        raise ValueError("x_target and y_target must have the same shape.")

    if point_weights is None:
        weights = np.ones_like(x_target, dtype=float)
    else:
        weights = np.asarray(point_weights, dtype=float).ravel()
        if weights.shape != x_target.shape:
            raise ValueError("point_weights must have the same shape as x_target and y_target.")

    x_axis = np.asarray(x_grid[0, :], dtype=float)
    y_axis = np.asarray(y_grid[:, 0], dtype=float)
    ix = nearest_axis_indices(x_axis, x_target)
    iy = nearest_axis_indices(y_axis, y_target)

    x_meas = x_grid[iy, ix]
    y_meas = y_grid[iy, ix]
    ux_meas = ux_grid[iy, ix]
    uy_meas = uy_grid[iy, ix]

    if not collapse_duplicates:
        return {
            "x_meas": x_meas,
            "y_meas": y_meas,
            "ux_meas": ux_meas,
            "uy_meas": uy_meas,
            "point_weights": weights,
            "grid_row_indices": iy,
            "grid_col_indices": ix,
            "n_duplicate_points_merged": np.array(0, dtype=int),
        }

    cell_id = iy * x_grid.shape[1] + ix
    unique_ids, first_index, inverse_index = np.unique(
        cell_id,
        return_index=True,
        return_inverse=True,
    )
    merged_weights = np.zeros(unique_ids.shape[0], dtype=float)
    np.add.at(merged_weights, inverse_index, weights)

    return {
        "x_meas": x_meas[first_index],
        "y_meas": y_meas[first_index],
        "ux_meas": ux_meas[first_index],
        "uy_meas": uy_meas[first_index],
        "point_weights": merged_weights,
        "grid_row_indices": iy[first_index],
        "grid_col_indices": ix[first_index],
        "n_duplicate_points_merged": np.array(x_target.size - unique_ids.size, dtype=int),
    }


def build_observation_operator(
    mesh_info: MeshInfo,
    x_meas: np.ndarray,
    y_meas: np.ndarray,
) -> csr_matrix:
    """
    Assemble the sparse observation matrix ``H`` for bilinear quad elements.

    For each measurement point, ``H`` evaluates the FE nodal displacement field
    at the corresponding physical position.
    """
    x_meas = np.asarray(x_meas, dtype=float).ravel()
    y_meas = np.asarray(y_meas, dtype=float).ravel()
    if x_meas.shape != y_meas.shape:
        raise ValueError("x_meas and y_meas must have the same shape.")
    if not np.all(np.isfinite(x_meas)) or not np.all(np.isfinite(y_meas)):
        raise ValueError('Measurement coordinates must be finite')

    tol = 1e-9 * max(mesh_info.geo_l, mesh_info.geo_h, 1.0) + 1e-12
    if np.any(x_meas < -tol) or np.any(x_meas > mesh_info.geo_l + tol):
        raise ValueError("At least one measurement x-coordinate lies outside the FE domain.")
    if np.any(y_meas < -tol) or np.any(y_meas > mesh_info.geo_h + tol):
        raise ValueError("At least one measurement y-coordinate lies outside the FE domain.")

    if getattr(mesh_info, 'measured_nodal_grid', False):
        from scipy.spatial import cKDTree
        distance, nodes = cKDTree(mesh_info.coord).query(np.column_stack([x_meas, y_meas]))
        if np.any(distance > 1e-7):
            raise ValueError('DIC-grid observations must match measured FE nodes within 1e-7 mm')
        columns = np.column_stack([2*nodes, 2*nodes+1]).ravel()
        return csr_matrix((np.ones(len(columns)), (np.arange(len(columns)), columns)),
                          shape=(len(columns), mesh_info.n_dof))

    x_use = np.clip(x_meas, 0.0, mesh_info.geo_l)
    y_use = np.clip(y_meas, 0.0, mesh_info.geo_h)

    ex = np.floor(x_use / mesh_info.el).astype(int)
    ey = np.floor(y_use / mesh_info.eh).astype(int)
    ex = np.clip(ex, 0, mesh_info.nel_x - 1)
    ey = np.clip(ey, 0, mesh_info.nel_y - 1)

    xi = (x_use - ex * mesh_info.el) / mesh_info.el
    eta = (y_use - ey * mesh_info.eh) / mesh_info.eh
    xi[np.abs(x_use - mesh_info.geo_l) <= tol] = 1.0
    eta[np.abs(y_use - mesh_info.geo_h) <= tol] = 1.0
    xi = np.clip(xi, 0.0, 1.0)
    eta = np.clip(eta, 0.0, 1.0)

    n_points = x_use.size
    nods_x = mesh_info.nods_x
    node_bl = ey * nods_x + ex
    node_br = node_bl + 1
    node_tr = node_bl + nods_x + 1
    node_tl = node_bl + nods_x

    n1 = (1.0 - xi) * (1.0 - eta)
    n2 = xi * (1.0 - eta)
    n3 = xi * eta
    n4 = (1.0 - xi) * eta
    shape_values = np.column_stack([n1, n2, n3, n4])

    base_rows = 2 * np.arange(n_points, dtype=int)
    x_rows = np.repeat(base_rows, 4)
    y_rows = np.repeat(base_rows + 1, 4)

    x_cols = np.column_stack([2 * node_bl, 2 * node_br, 2 * node_tr, 2 * node_tl]).ravel()
    y_cols = np.column_stack([2 * node_bl + 1, 2 * node_br + 1, 2 * node_tr + 1, 2 * node_tl + 1]).ravel()
    data = shape_values.ravel()

    H = csr_matrix(
        (
            np.concatenate([data, data]),
            (
                np.concatenate([x_rows, y_rows]),
                np.concatenate([x_cols, y_cols]),
            ),
        ),
        shape=(2 * n_points, mesh_info.n_dof),
    )
    H.sum_duplicates()
    return H.tocsr()


def build_observation_package(
    mesh_info: MeshInfo,
    x_meas: np.ndarray,
    y_meas: np.ndarray,
    ux_meas: np.ndarray,
    uy_meas: np.ndarray,
    point_weights: np.ndarray | None = None,
    sampling_mode: str = "manual",
    ux_component_weight: float | np.ndarray = 1.0,
    uy_component_weight: float | np.ndarray = 1.0,
) -> ObservationPackage:
    """
    Create the full observation package used by a pointwise data-misfit term.

    The returned vectors are interleaved as ``[ux_1, uy_1, ux_2, uy_2, ...]``.
    """
    x_meas = np.asarray(x_meas, dtype=float).ravel()
    y_meas = np.asarray(y_meas, dtype=float).ravel()
    ux_meas = np.asarray(ux_meas, dtype=float).ravel()
    uy_meas = np.asarray(uy_meas, dtype=float).ravel()
    if not (x_meas.shape == y_meas.shape == ux_meas.shape == uy_meas.shape):
        raise ValueError("Measurement coordinate and displacement arrays must all share the same shape.")

    if point_weights is None:
        point_weights_use = np.ones_like(x_meas, dtype=float)
    else:
        point_weights_use = np.asarray(point_weights, dtype=float).ravel()
        if point_weights_use.shape != x_meas.shape:
            raise ValueError("point_weights must have the same shape as the measurement arrays.")

    H = build_observation_operator(mesh_info, x_meas, y_meas)
    u_measured = interleave_components(ux_meas, uy_meas)

    ux_weights = _coerce_component_weights(ux_component_weight, x_meas.shape, "ux_component_weight")
    uy_weights = _coerce_component_weights(uy_component_weight, x_meas.shape, "uy_component_weight")

    weight_vector = np.empty(2 * point_weights_use.size, dtype=float)
    weight_vector[0::2] = point_weights_use * ux_weights
    weight_vector[1::2] = point_weights_use * uy_weights
    W = diags(weight_vector, offsets=0, shape=(weight_vector.size, weight_vector.size), format="csr")

    return ObservationPackage(
        x_meas=x_meas,
        y_meas=y_meas,
        ux_meas=ux_meas,
        uy_meas=uy_meas,
        u_measured=u_measured,
        point_weights=point_weights_use,
        weight_vector=weight_vector,
        H=H,
        W=W,
        sampling_mode=sampling_mode,
    )


def prepare_observation_package_from_grids(
    mesh_info: MeshInfo,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    ux_grid: np.ndarray,
    uy_grid: np.ndarray,
    sampling_mode: str,
    uniform_subgrid_nx: int | None = None,
    uniform_subgrid_ny: int | None = None,
    ux_component_weight: float = 1.0,
    uy_component_weight: float = 1.0,
    component_weight_mode: str = "constant",
    local_max_epsilon: float = 0.05,
    collapse_duplicates: bool = True,
) -> tuple[ObservationPackage, dict[str, np.ndarray]]:
    """
    Prepare an observation package directly from structured DIC grids.

    This helper performs the orientation fix, chooses real DIC points according
    to ``sampling_mode``, and assembles the final ``H`` / ``W`` operators.
    """
    x_use, y_use, ux_use, uy_use = orient_structured_grid_to_fe_basis(
        x_grid=x_grid,
        y_grid=y_grid,
        ux_grid=ux_grid,
        uy_grid=uy_grid,
    )

    if sampling_mode == "uniform_subgrid":
        if uniform_subgrid_nx is None or uniform_subgrid_ny is None:
            raise ValueError("uniform_subgrid_nx and uniform_subgrid_ny are required for uniform_subgrid mode.")
        sampled = sample_dic_on_uniform_subgrid(
            x_grid=x_use,
            y_grid=y_use,
            ux_grid=ux_use,
            uy_grid=uy_use,
            n_x=uniform_subgrid_nx,
            n_y=uniform_subgrid_ny,
        )
    elif sampling_mode == "fe_gauss_nearest":
        targets = get_fe_gauss_targets(mesh_info)
        sampled = sample_dic_nearest_to_targets(
            x_grid=x_use,
            y_grid=y_use,
            ux_grid=ux_use,
            uy_grid=uy_use,
            x_target=targets["x_target"],
            y_target=targets["y_target"],
            point_weights=targets["point_weights"],
            collapse_duplicates=collapse_duplicates,
        )
    else:
        raise ValueError(f"Unsupported sampling mode: {sampling_mode}")

    ux_weight_values, uy_weight_values = _build_component_weight_values(
        ux_meas=sampled["ux_meas"],
        uy_meas=sampled["uy_meas"],
        ux_component_weight=ux_component_weight,
        uy_component_weight=uy_component_weight,
        component_weight_mode=component_weight_mode,
        local_max_epsilon=local_max_epsilon,
    )

    package = build_observation_package(
        mesh_info=mesh_info,
        x_meas=sampled["x_meas"],
        y_meas=sampled["y_meas"],
        ux_meas=sampled["ux_meas"],
        uy_meas=sampled["uy_meas"],
        point_weights=sampled["point_weights"],
        sampling_mode=sampling_mode,
        ux_component_weight=ux_weight_values,
        uy_component_weight=uy_weight_values,
    )
    return package, sampled


def save_observation_package(
    output_dir: Path,
    case_name: str,
    package: ObservationPackage,
) -> dict[str, Path]:
    """Save the observation package arrays and sparse matrices to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = output_dir / case_name
    saved_paths = {
        "x_meas": prefix.with_name(f"{prefix.name}_x_meas.npy"),
        "y_meas": prefix.with_name(f"{prefix.name}_y_meas.npy"),
        "ux_meas": prefix.with_name(f"{prefix.name}_ux_meas.npy"),
        "uy_meas": prefix.with_name(f"{prefix.name}_uy_meas.npy"),
        "u_measured": prefix.with_name(f"{prefix.name}_u_measured.npy"),
        "point_weights": prefix.with_name(f"{prefix.name}_point_weights.npy"),
        "weight_vector": prefix.with_name(f"{prefix.name}_weight_vector.npy"),
        "H": prefix.with_name(f"{prefix.name}_H.npz"),
        "W": prefix.with_name(f"{prefix.name}_W.npz"),
    }

    np.save(saved_paths["x_meas"], package.x_meas)
    np.save(saved_paths["y_meas"], package.y_meas)
    np.save(saved_paths["ux_meas"], package.ux_meas)
    np.save(saved_paths["uy_meas"], package.uy_meas)
    np.save(saved_paths["u_measured"], package.u_measured)
    np.save(saved_paths["point_weights"], package.point_weights)
    np.save(saved_paths["weight_vector"], package.weight_vector)
    save_npz(saved_paths["H"], package.H)
    save_npz(saved_paths["W"], package.W)
    return saved_paths


def load_observation_package(package_paths: Mapping[str, Path], sampling_mode: str | None = None) -> ObservationPackage:
    """Load a previously saved observation package from disk."""
    required_keys = ("x_meas", "y_meas", "ux_meas", "uy_meas", "u_measured", "point_weights", "weight_vector", "H", "W")
    missing = [key for key in required_keys if key not in package_paths]
    if missing:
        raise KeyError(f"Missing required observation package paths: {missing}")

    sampling_mode_use = _infer_sampling_mode_from_filename(package_paths["H"]) if sampling_mode is None else sampling_mode

    return ObservationPackage(
        x_meas=np.load(package_paths["x_meas"]),
        y_meas=np.load(package_paths["y_meas"]),
        ux_meas=np.load(package_paths["ux_meas"]),
        uy_meas=np.load(package_paths["uy_meas"]),
        u_measured=np.load(package_paths["u_measured"]),
        point_weights=np.load(package_paths["point_weights"]),
        weight_vector=np.load(package_paths["weight_vector"]),
        H=load_npz(package_paths["H"]).tocsr(),
        W=load_npz(package_paths["W"]).tocsr(),
        sampling_mode=sampling_mode_use,
    )


def compute_axis_cell_widths(axis: np.ndarray) -> np.ndarray:
    """Compute 1D Voronoi-like cell widths used to form area weights."""
    axis = np.asarray(axis, dtype=float).ravel()
    if axis.size < 2:
        raise ValueError("Axis must contain at least 2 points.")
    if np.any(np.diff(axis) <= 0.0):
        raise ValueError("Axis must be strictly increasing.")

    midpoints = 0.5 * (axis[:-1] + axis[1:])
    left = np.empty_like(axis)
    right = np.empty_like(axis)
    left[0] = axis[0]
    left[1:] = midpoints
    right[:-1] = midpoints
    right[-1] = axis[-1]
    return right - left


def interleave_components(ux: np.ndarray, uy: np.ndarray) -> np.ndarray:
    """Interleave x and y displacement components point-by-point."""
    ux = np.asarray(ux, dtype=float).ravel()
    uy = np.asarray(uy, dtype=float).ravel()
    if ux.shape != uy.shape:
        raise ValueError("ux and uy must have the same shape.")

    u_measured = np.empty(2 * ux.size, dtype=float)
    u_measured[0::2] = ux
    u_measured[1::2] = uy
    return u_measured


def _coerce_component_weights(
    component_weight: float | np.ndarray,
    expected_shape: tuple[int, ...],
    name: str,
) -> np.ndarray:
    weights = np.asarray(component_weight, dtype=float)
    if weights.ndim == 0:
        weights = np.full(expected_shape, float(weights), dtype=float)
    else:
        weights = weights.ravel()
        if weights.shape != expected_shape:
            raise ValueError(f"{name} must be scalar or have the same shape as the measurement arrays.")
    if np.any(weights < 0.0):
        raise ValueError(f"{name} must be non-negative.")
    return weights


def _build_component_weight_values(
    ux_meas: np.ndarray,
    uy_meas: np.ndarray,
    ux_component_weight: float,
    uy_component_weight: float,
    component_weight_mode: str,
    local_max_epsilon: float,
) -> tuple[np.ndarray, np.ndarray]:
    ux = np.asarray(ux_meas, dtype=float).ravel()
    uy = np.asarray(uy_meas, dtype=float).ravel()
    mode = component_weight_mode.lower()
    if mode in {"constant", "global"}:
        return (
            np.full_like(ux, float(ux_component_weight), dtype=float),
            np.full_like(uy, float(uy_component_weight), dtype=float),
        )
    if mode != "local_max":
        raise ValueError("component_weight_mode must be 'constant' or 'local_max'.")

    epsilon = float(local_max_epsilon)
    if epsilon <= 0.0:
        raise ValueError("local_max_epsilon must be positive.")

    ux_floor = max(epsilon * float(np.max(np.abs(ux))), 1e-15)
    uy_floor = max(epsilon * float(np.max(np.abs(uy))), 1e-15)

    ux_scale = np.maximum(np.abs(ux), ux_floor)
    uy_scale = np.maximum(np.abs(uy), uy_floor)
    ux_values = float(ux_component_weight) / (ux_scale ** 2)
    uy_values = float(uy_component_weight) / (uy_scale ** 2)
    return ux_values, uy_values


def nearest_axis_indices(axis: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Find nearest indices on a strictly increasing structured-grid axis."""
    axis = np.asarray(axis, dtype=float).ravel()
    targets = np.asarray(targets, dtype=float).ravel()
    if axis.size < 2:
        raise ValueError("Axis must contain at least 2 points.")
    if np.any(np.diff(axis) <= 0.0):
        raise ValueError("Axis must be strictly increasing.")

    idx = np.searchsorted(axis, targets, side="left")
    idx = np.clip(idx, 0, axis.size - 1)

    left_idx = np.clip(idx - 1, 0, axis.size - 1)
    right_idx = idx
    choose_left = np.abs(targets - axis[left_idx]) <= np.abs(axis[right_idx] - targets)
    return np.where(choose_left, left_idx, right_idx)


def _validate_grid_shapes(*arrays: np.ndarray) -> None:
    """Validate that all structured-grid arrays share one common 2D shape."""
    reference_shape = arrays[0].shape
    if len(reference_shape) != 2:
        raise ValueError("Structured-grid arrays must be 2D.")

    for array in arrays[1:]:
        if array.shape != reference_shape:
            raise ValueError("All structured-grid arrays must have the same shape.")


def _infer_sampling_mode_from_filename(path: Path) -> str:
    """Recover the configured sampling mode from a saved observation filename."""
    stem = Path(path).stem
    if "_uniform_subgrid_observation_" in stem:
        return "uniform_subgrid"
    if "_fe_gauss_nearest_observation_" in stem:
        return "fe_gauss_nearest"
    return "loaded"
