from __future__ import annotations

import numpy as np


def _build_strip_mask(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    strip_side: str,
    strip_width_ratio: float,
    y_margin_ratio: float = 0.0,
) -> tuple[np.ndarray, dict[str, float | str]]:
    side = strip_side.lower()
    x_min = float(np.nanmin(x_grid))
    x_max = float(np.nanmax(x_grid))
    width = x_max - x_min
    strip_width = max(width * float(strip_width_ratio), 1e-12)
    y_min = float(np.nanmin(y_grid))
    y_max = float(np.nanmax(y_grid))
    height = y_max - y_min
    y_margin = max(height * float(y_margin_ratio), 0.0)

    if side == "left":
        bound = x_min + strip_width
        x_mask = x_grid <= bound
    elif side == "right":
        bound = x_max - strip_width
        x_mask = x_grid >= bound
    else:
        raise ValueError("strip_side must be 'left' or 'right'.")

    if y_margin > 0.0 and (y_max - y_min) > 2.0 * y_margin:
        y_mask = (y_grid >= (y_min + y_margin)) & (y_grid <= (y_max - y_margin))
    else:
        y_mask = np.ones_like(y_grid, dtype=bool)

    mask = x_mask & y_mask & np.isfinite(x_grid) & np.isfinite(y_grid)
    if not np.any(mask):
        raise ValueError("Fixed-end strip selection produced an empty mask.")

    metadata = {
        "strip_side": side,
        "strip_width_ratio": float(strip_width_ratio),
        "y_margin_ratio": float(y_margin_ratio),
        "strip_x_min": float(np.nanmin(x_grid[mask])),
        "strip_x_max": float(np.nanmax(x_grid[mask])),
        "strip_y_min": float(np.nanmin(y_grid[mask])),
        "strip_y_max": float(np.nanmax(y_grid[mask])),
        "strip_point_count": int(np.count_nonzero(mask)),
    }
    return mask, metadata


def _solve_rigid_parameters(
    x_values: np.ndarray,
    y_values: np.ndarray,
    ux_values: np.ndarray,
    uy_values: np.ndarray,
) -> tuple[float, float, float]:
    a_matrix = np.zeros((2 * len(x_values), 3), dtype=float)
    b_vector = np.zeros(2 * len(x_values), dtype=float)
    a_matrix[0::2, 0] = 1.0
    a_matrix[0::2, 2] = -y_values
    a_matrix[1::2, 1] = 1.0
    a_matrix[1::2, 2] = x_values
    b_vector[0::2] = ux_values
    b_vector[1::2] = uy_values
    tx, ty, theta = np.linalg.lstsq(a_matrix, b_vector, rcond=None)[0]
    tx = float(tx)
    ty = float(ty)
    theta = float(theta)

    return tx, ty, theta


def _rigid_residual_magnitude(
    x_values: np.ndarray,
    y_values: np.ndarray,
    ux_values: np.ndarray,
    uy_values: np.ndarray,
    tx: float,
    ty: float,
    theta: float,
) -> np.ndarray:
    ux_fit = tx - theta * y_values
    uy_fit = ty + theta * x_values
    return np.sqrt((ux_values - ux_fit) ** 2 + (uy_values - uy_fit) ** 2)


def remove_rigid_body_motion_from_grid(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    ux_grid: np.ndarray,
    uy_grid: np.ndarray,
    strip_side: str,
    strip_width_ratio: float = 0.1,
    y_margin_ratio: float = 0.0,
) -> dict[str, np.ndarray | dict[str, object]]:
    mask, strip_metadata = _build_strip_mask(
        x_grid,
        y_grid,
        strip_side=strip_side,
        strip_width_ratio=strip_width_ratio,
        y_margin_ratio=y_margin_ratio,
    )
    x_strip = x_grid[mask]
    y_strip = y_grid[mask]
    ux_strip = ux_grid[mask]
    uy_strip = uy_grid[mask]

    tx, ty, theta = _solve_rigid_parameters(x_strip, y_strip, ux_strip, uy_strip)

    ux_rigid = tx - theta * y_grid
    uy_rigid = ty + theta * x_grid
    residual = _rigid_residual_magnitude(x_strip, y_strip, ux_strip, uy_strip, tx, ty, theta)

    return {
        "UX_grid": ux_grid - ux_rigid,
        "UY_grid": uy_grid - uy_rigid,
        "metadata": {
            **strip_metadata,
            "rigid_parameters": {
                "tx": tx,
                "ty": ty,
                "theta": theta,
            },
            "strip_residual_rms": float(np.sqrt(np.mean(residual ** 2))),
            "strip_residual_max": float(np.max(residual)),
        },
    }
