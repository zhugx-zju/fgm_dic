from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from lib.dic_grid_orientation import orient_grid_field
from lib.dic_grid_utils import reconstruct_structured_grid
from lib.fixed_point_reference import apply_fixed_point_reference_to_grid
from lib.processing_utils import load_raw_processed
from lib.rigid_body_reference import remove_rigid_body_motion_from_grid


def _centers_to_edges(centers: np.ndarray) -> np.ndarray:
    centers_1d = np.asarray(centers, dtype=float)
    if centers_1d.ndim != 1:
        raise ValueError("Grid centers must be 1D when constructing pcolormesh edges.")
    if centers_1d.size == 1:
        delta = 0.5
        return np.array([centers_1d[0] - delta, centers_1d[0] + delta], dtype=float)

    deltas = np.diff(centers_1d)
    edges = np.empty(centers_1d.size + 1, dtype=float)
    edges[1:-1] = centers_1d[:-1] + 0.5 * deltas
    edges[0] = centers_1d[0] - 0.5 * deltas[0]
    edges[-1] = centers_1d[-1] + 0.5 * deltas[-1]
    return edges


def apply_displacement_sign_control(
    ux_grid: np.ndarray,
    uy_grid: np.ndarray,
    ux_sign: float = 1.0,
    uy_sign: float = 1.0,
) -> dict[str, np.ndarray | dict[str, float]]:
    ux_factor = float(ux_sign)
    uy_factor = float(uy_sign)
    if ux_factor not in {-1.0, 1.0} or uy_factor not in {-1.0, 1.0}:
        raise ValueError("Displacement sign control must use +1 or -1 for both ux_sign and uy_sign.")

    return {
        "UX_grid": ux_factor * np.asarray(ux_grid, dtype=float),
        "UY_grid": uy_factor * np.asarray(uy_grid, dtype=float),
        "metadata": {
            "ux_sign": ux_factor,
            "uy_sign": uy_factor,
        },
    }


def compute_preprocessed_grid_summary(arrays: dict[str, object]) -> dict[str, float]:
    x_grid = np.asarray(arrays["X_grid"], dtype=float)
    y_grid = np.asarray(arrays["Y_grid"], dtype=float)
    ux_grid = np.asarray(arrays["UX_grid"], dtype=float)

    geo_l = float(np.nanmax(x_grid) - np.nanmin(x_grid))
    geo_h = float(np.nanmax(y_grid) - np.nanmin(y_grid))
    right_edge_x = float(np.nanmax(x_grid))
    right_edge_mask = np.isclose(x_grid, right_edge_x)
    prescribed_right_edge_displacement = float(np.nanmean(ux_grid[right_edge_mask]))

    return {
        "geo_l": geo_l,
        "geo_h": geo_h,
        "prescribed_right_edge_displacement": prescribed_right_edge_displacement,
    }


def _field_stats(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    return {
        "mean": float(np.nanmean(values)),
        "min": float(np.nanmin(values)),
        "max": float(np.nanmax(values)),
    }


def compute_preprocessed_boundary_diagnostics(arrays: dict[str, object]) -> dict[str, object]:
    """
    Summarize FE-relevant boundary values after preprocessing.

    The inverse workflow fixes the FE left-edge x displacement and the
    FE bottom-left y displacement, so these diagnostics are the fastest
    way to spot a reference mismatch before starting the inverse solve.
    """
    x_grid = np.asarray(arrays["X_grid"], dtype=float)
    y_grid = np.asarray(arrays["Y_grid"], dtype=float)
    ux_grid = np.asarray(arrays["UX_grid"], dtype=float)
    uy_grid = np.asarray(arrays["UY_grid"], dtype=float)

    x_min = float(np.nanmin(x_grid))
    x_max = float(np.nanmax(x_grid))
    y_min = float(np.nanmin(y_grid))

    left_mask = np.isclose(x_grid, x_min)
    right_mask = np.isclose(x_grid, x_max)

    bottom_left_distance = (x_grid - x_min) ** 2 + (y_grid - y_min) ** 2
    bottom_left_row, bottom_left_col = np.unravel_index(int(np.nanargmin(bottom_left_distance)), x_grid.shape)

    return {
        "ux_left": _field_stats(ux_grid[left_mask]),
        "ux_right": _field_stats(ux_grid[right_mask]),
        "bottom_left": {
            "x": float(x_grid[bottom_left_row, bottom_left_col]),
            "y": float(y_grid[bottom_left_row, bottom_left_col]),
            "ux": float(ux_grid[bottom_left_row, bottom_left_col]),
            "uy": float(uy_grid[bottom_left_row, bottom_left_col]),
        },
    }


def _skip_fixed_point_reference(
    ux_grid: np.ndarray,
    uy_grid: np.ndarray,
) -> dict[str, np.ndarray | dict[str, float | str | None]]:
    return {
        "UX_grid": np.asarray(ux_grid, dtype=float),
        "UY_grid": np.asarray(uy_grid, dtype=float),
        "metadata": {
            "fixed_point": None,
            "reference_point_x": float("nan"),
            "reference_point_y": float("nan"),
            "reference_dx": 0.0,
            "reference_dy": 0.0,
        },
    }


def _normalize_fixed_point_option(fixed_point: str | None) -> str | None:
    if fixed_point is None:
        return None
    if isinstance(fixed_point, str) and fixed_point.strip().lower() in {"", "none"}:
        return None
    return fixed_point


def preprocess_structured_dic(
    csv_path: Path,
    fixed_point: str | None,
    fixed_end_side: str,
    strip_width_ratio: float,
    strip_y_margin_ratio: float,
    displacement_signs: tuple[float, float] = (1.0, 1.0),
    orientation_operations: list[str] | tuple[str, ...] | None = None,
) -> dict[str, object]:
    fixed_point = _normalize_fixed_point_option(fixed_point)
    df = load_raw_processed(csv_path, required_fields=["X", "Y", "DX", "DY", "grid_row", "grid_col"])
    grid = reconstruct_structured_grid(df)

    rigid_removed = remove_rigid_body_motion_from_grid(
        x_grid=grid["X_ref"],
        y_grid=grid["Y_ref"],
        ux_grid=grid["DX"],
        uy_grid=grid["DY"],
        strip_side=fixed_end_side,
        strip_width_ratio=strip_width_ratio,
        y_margin_ratio=strip_y_margin_ratio,
    )
    if fixed_point is None:
        fixed_point_corrected = _skip_fixed_point_reference(
            ux_grid=np.asarray(rigid_removed["UX_grid"], dtype=float),
            uy_grid=np.asarray(rigid_removed["UY_grid"], dtype=float),
        )
    else:
        fixed_point_corrected = apply_fixed_point_reference_to_grid(
            x_grid=grid["X_ref"],
            y_grid=grid["Y_ref"],
            ux_grid=np.asarray(rigid_removed["UX_grid"], dtype=float),
            uy_grid=np.asarray(rigid_removed["UY_grid"], dtype=float),
            point=fixed_point,
        )
    oriented = orient_grid_field(
        x_grid=grid["X_ref"],
        y_grid=grid["Y_ref"],
        ux_grid=np.asarray(fixed_point_corrected["UX_grid"], dtype=float),
        uy_grid=np.asarray(fixed_point_corrected["UY_grid"], dtype=float),
        operations=orientation_operations,
    )
    signed = apply_displacement_sign_control(
        ux_grid=np.asarray(oriented["UX_grid"], dtype=float),
        uy_grid=np.asarray(oriented["UY_grid"], dtype=float),
        ux_sign=displacement_signs[0],
        uy_sign=displacement_signs[1],
    )

    return {
        "X_grid": np.asarray(oriented["X_grid"], dtype=float),
        "Y_grid": np.asarray(oriented["Y_grid"], dtype=float),
        "UX_grid": np.asarray(signed["UX_grid"], dtype=float),
        "UY_grid": np.asarray(signed["UY_grid"], dtype=float),
        "metadata": {
            "source_csv": str(csv_path),
            "fixed_point": fixed_point,
            "strip_side": fixed_end_side,
            "rigid": rigid_removed["metadata"],
            "fixed_point_reference": fixed_point_corrected["metadata"],
            "orientation": oriented["metadata"],
            "sign_control": signed["metadata"],
        },
    }


def preprocess_structured_dic_for_case(
    csv_path: Path,
    displacement_signs: tuple[float, float] = (1.0, 1.0),
    orientation_operations: list[str] | tuple[str, ...] | None = None,
    remove_translation: bool = True,
) -> dict[str, object]:
    """
    Prepare DIC grids for BC-free CASE inversion.

    CASE does not require the measured field to satisfy a prescribed FE
    boundary condition, so this path intentionally skips fixed-end strip
    rigid-body removal and fixed-point referencing. A global translation
    removal is kept optional because it is a harmless reference change for
    the measured displacement field.
    """
    df = load_raw_processed(csv_path, required_fields=["X", "Y", "DX", "DY", "grid_row", "grid_col"])
    grid = reconstruct_structured_grid(df)
    ux_grid = np.asarray(grid["DX"], dtype=float)
    uy_grid = np.asarray(grid["DY"], dtype=float)

    translation = {"ux_mean": 0.0, "uy_mean": 0.0}
    if remove_translation:
        translation = {
            "ux_mean": float(np.nanmean(ux_grid)),
            "uy_mean": float(np.nanmean(uy_grid)),
        }
        ux_grid = ux_grid - translation["ux_mean"]
        uy_grid = uy_grid - translation["uy_mean"]

    oriented = orient_grid_field(
        x_grid=grid["X_ref"],
        y_grid=grid["Y_ref"],
        ux_grid=ux_grid,
        uy_grid=uy_grid,
        operations=orientation_operations,
    )
    signed = apply_displacement_sign_control(
        ux_grid=np.asarray(oriented["UX_grid"], dtype=float),
        uy_grid=np.asarray(oriented["UY_grid"], dtype=float),
        ux_sign=displacement_signs[0],
        uy_sign=displacement_signs[1],
    )

    return {
        "X_grid": np.asarray(oriented["X_grid"], dtype=float),
        "Y_grid": np.asarray(oriented["Y_grid"], dtype=float),
        "UX_grid": np.asarray(signed["UX_grid"], dtype=float),
        "UY_grid": np.asarray(signed["UY_grid"], dtype=float),
        "metadata": {
            "source_csv": str(csv_path),
            "case_preprocessing": True,
            "remove_translation": bool(remove_translation),
            "translation": translation,
            "orientation": oriented["metadata"],
            "sign_control": signed["metadata"],
        },
    }


def save_preprocessed_arrays(output_dir: Path, case_name: str, arrays: dict[str, object]) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = {
        "x_grid": output_dir / f"{case_name}_x_grid.npy",
        "y_grid": output_dir / f"{case_name}_y_grid.npy",
        "ux_grid": output_dir / f"{case_name}_ux_grid.npy",
        "uy_grid": output_dir / f"{case_name}_uy_grid.npy",
    }
    np.save(saved_paths["x_grid"], arrays["X_grid"])
    np.save(saved_paths["y_grid"], arrays["Y_grid"])
    np.save(saved_paths["ux_grid"], arrays["UX_grid"])
    np.save(saved_paths["uy_grid"], arrays["UY_grid"])
    return saved_paths


def plot_grid_fields(
    output_dir: Path,
    case_name: str,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    fields: list[tuple[str, np.ndarray, str]],
    cmap: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    x_grid = np.asarray(x_grid, dtype=float)
    y_grid = np.asarray(y_grid, dtype=float)
    x_centers = np.mean(x_grid, axis=0)
    y_centers = np.mean(y_grid, axis=1)
    x_edges = _centers_to_edges(x_centers)
    y_edges = _centers_to_edges(y_centers)

    for suffix, field, title in fields:
        fig, ax = plt.subplots(figsize=(7, 6))
        mesh = ax.pcolormesh(x_edges, y_edges, field, shading="auto", cmap=cmap)
        ax.set_aspect("equal")
        ax.set_title(f"Preprocessed {title}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        plt.colorbar(mesh, ax=ax)
        fig.tight_layout()
        fig.savefig(output_dir / f"{case_name}_{suffix}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


def plot_preprocessed_displacement_fields(output_dir: Path, case_name: str, arrays: dict[str, object], cmap: str) -> None:
    fields = [
        ("ux", np.asarray(arrays["UX_grid"], dtype=float), r"$u_x$"),
        ("uy", np.asarray(arrays["UY_grid"], dtype=float), r"$u_y$"),
    ]
    plot_grid_fields(
        output_dir=output_dir,
        case_name=case_name,
        x_grid=np.asarray(arrays["X_grid"], dtype=float),
        y_grid=np.asarray(arrays["Y_grid"], dtype=float),
        fields=fields,
        cmap=cmap,
    )
