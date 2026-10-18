from __future__ import annotations

import numpy as np


def _transform_coords(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    coords = np.stack([x_grid, y_grid], axis=-1) @ matrix.T
    x_new = coords[..., 0]
    y_new = coords[..., 1]
    x_new = x_new - float(np.nanmin(x_new))
    y_new = y_new - float(np.nanmin(y_new))
    return x_new, y_new


def _transform_displacements(
    ux_grid: np.ndarray,
    uy_grid: np.ndarray,
    matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    displacement = np.stack([ux_grid, uy_grid], axis=-1) @ matrix.T
    return displacement[..., 0], displacement[..., 1]


def orient_grid_field(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    ux_grid: np.ndarray,
    uy_grid: np.ndarray,
    operations: list[str] | tuple[str, ...] | None = None,
) -> dict[str, np.ndarray | dict[str, object]]:
    x_work = np.array(x_grid, copy=True)
    y_work = np.array(y_grid, copy=True)
    ux_work = np.array(ux_grid, copy=True)
    uy_work = np.array(uy_grid, copy=True)
    transform = np.eye(2, dtype=float)
    applied_operations: list[str] = []
    for operation in operations or []:
        op = operation.lower()
        if op == "rot90":
            x_work = np.rot90(x_work, 1)
            y_work = np.rot90(y_work, 1)
            ux_work = np.rot90(ux_work, 1)
            uy_work = np.rot90(uy_work, 1)
            transform = np.array([[0.0, -1.0], [1.0, 0.0]]) @ transform
        elif op == "rot180":
            x_work = np.rot90(x_work, 2)
            y_work = np.rot90(y_work, 2)
            ux_work = np.rot90(ux_work, 2)
            uy_work = np.rot90(uy_work, 2)
            transform = np.array([[-1.0, 0.0], [0.0, -1.0]]) @ transform
        elif op == "rot270":
            x_work = np.rot90(x_work, 3)
            y_work = np.rot90(y_work, 3)
            ux_work = np.rot90(ux_work, 3)
            uy_work = np.rot90(uy_work, 3)
            transform = np.array([[0.0, 1.0], [-1.0, 0.0]]) @ transform
        elif op == "fliplr":
            x_work = np.fliplr(x_work)
            y_work = np.fliplr(y_work)
            ux_work = np.fliplr(ux_work)
            uy_work = np.fliplr(uy_work)
            transform = np.array([[-1.0, 0.0], [0.0, 1.0]]) @ transform
        elif op == "flipud":
            x_work = np.flipud(x_work)
            y_work = np.flipud(y_work)
            ux_work = np.flipud(ux_work)
            uy_work = np.flipud(uy_work)
            transform = np.array([[1.0, 0.0], [0.0, -1.0]]) @ transform
        elif op == "transpose":
            x_work = x_work.T
            y_work = y_work.T
            ux_work = ux_work.T
            uy_work = uy_work.T
            transform = np.array([[0.0, 1.0], [1.0, 0.0]]) @ transform
        elif op in {"rot0", "identity", "none"}:
            pass
        else:
            raise ValueError(
                "Unsupported orientation operation. Use one of: "
                "rot90, rot180, rot270, fliplr, flipud, transpose."
            )
        applied_operations.append(op)

    x_aligned, y_aligned = _transform_coords(
        x_work,
        y_work,
        transform,
    )
    ux_aligned, uy_aligned = _transform_displacements(
        ux_work,
        uy_work,
        transform,
    )
    return {
        "X_grid": x_aligned,
        "Y_grid": y_aligned,
        "UX_grid": ux_aligned,
        "UY_grid": uy_aligned,
        "metadata": {
            "operations": applied_operations,
            "transform_matrix": transform,
        },
    }
