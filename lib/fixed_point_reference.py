import numpy as np
import pandas as pd


def locate_reference_point(df: pd.DataFrame, point: str) -> tuple[float, float, float, float]:
    point_key = point.lower()
    x = df["X_ref"].to_numpy()
    y = df["Y_ref"].to_numpy()
    dx = df["DX"].to_numpy()
    dy = df["DY"].to_numpy()

    if point_key == "top_right":
        target_x, target_y = x.max(), y.max()
    elif point_key == "bottom_right":
        target_x, target_y = x.max(), y.min()
    elif point_key == "top_left":
        target_x, target_y = x.min(), y.max()
    elif point_key == "bottom_left":
        target_x, target_y = x.min(), y.min()
    else:
        raise ValueError("Unsupported fixed point. Use one of: top_right, bottom_right, top_left, bottom_left.")

    distance_sq = (x - target_x) ** 2 + (y - target_y) ** 2
    index = int(np.argmin(distance_sq))
    return float(x[index]), float(y[index]), float(dx[index]), float(dy[index])


def apply_fixed_point_reference_correction(
    df: pd.DataFrame,
    point: str,
) -> tuple[pd.DataFrame, tuple[float, float, float, float]]:
    # Legacy dataframe-based fixed-point correction for the older tabular workflow.
    ref_x, ref_y, ref_dx, ref_dy = locate_reference_point(df, point=point)
    corrected = df.copy()
    corrected["DX_raw"] = corrected["DX"]
    corrected["DY_raw"] = corrected["DY"]
    corrected["DX_corr"] = corrected["DX_raw"] - ref_dx
    corrected["DY_corr"] = corrected["DY_raw"] - ref_dy
    corrected["DX"] = corrected["DX_corr"]
    corrected["DY"] = corrected["DY_corr"]
    corrected["fixed_point"] = point.lower()
    corrected["reference_point_x"] = ref_x
    corrected["reference_point_y"] = ref_y
    corrected["reference_dx"] = ref_dx
    corrected["reference_dy"] = ref_dy
    return corrected, (ref_x, ref_y, ref_dx, ref_dy)


def _grid_reference_indices(x_grid: np.ndarray, y_grid: np.ndarray, point: str) -> tuple[int, int]:
    point_key = point.lower()
    if point_key == "top_right":
        target_x, target_y = np.nanmax(x_grid), np.nanmax(y_grid)
    elif point_key == "bottom_right":
        target_x, target_y = np.nanmax(x_grid), np.nanmin(y_grid)
    elif point_key == "top_left":
        target_x, target_y = np.nanmin(x_grid), np.nanmax(y_grid)
    elif point_key == "bottom_left":
        target_x, target_y = np.nanmin(x_grid), np.nanmin(y_grid)
    else:
        raise ValueError("Unsupported fixed point. Use one of: top_right, bottom_right, top_left, bottom_left.")

    distance_sq = (x_grid - target_x) ** 2 + (y_grid - target_y) ** 2
    flat_index = int(np.nanargmin(distance_sq))
    return tuple(np.unravel_index(flat_index, x_grid.shape))


def apply_fixed_point_reference_to_grid(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    ux_grid: np.ndarray,
    uy_grid: np.ndarray,
    point: str,
) -> dict[str, np.ndarray | dict[str, float | str]]:
    row_idx, col_idx = _grid_reference_indices(x_grid=x_grid, y_grid=y_grid, point=point)
    ref_x = float(x_grid[row_idx, col_idx])
    ref_y = float(y_grid[row_idx, col_idx])
    ref_dx = float(ux_grid[row_idx, col_idx])
    ref_dy = float(uy_grid[row_idx, col_idx])

    return {
        "UX_grid": ux_grid - ref_dx,
        "UY_grid": uy_grid - ref_dy,
        "metadata": {
            "fixed_point": point.lower(),
            "reference_point_x": ref_x,
            "reference_point_y": ref_y,
            "reference_dx": ref_dx,
            "reference_dy": ref_dy,
        },
    }
