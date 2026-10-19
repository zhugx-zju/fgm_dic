"""DIC lattice reconstruction and original post_DIC cloud plots."""

from __future__ import annotations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Preserve the original post_DIC typography without importing legacy plotting.
plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Times New Roman'],
    'mathtext.fontset': 'custom', 'mathtext.rm': 'Times New Roman',
    'mathtext.it': 'Times New Roman:italic', 'mathtext.bf': 'Times New Roman:bold',
})


def plot_grid_fields(
    output_dir: Path,
    case_name: str,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    fields: list[tuple[str, np.ndarray, str]],
    cmap: str,
    *,
    limits: dict[str, tuple[float, float]] | None = None,
    title_prefix: str = "Preprocessed ",
    unit: str | None = None,
) -> None:
    """Original post_DIC cloud style, shared by preprocessing and comparisons.

    Values live at DIC cell centers. Missing samples stay masked; nearest
    shading extends the boundary by half a cell, as the original edge plot did.
    Full 2D coordinates also preserve a rotated reference lattice.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    x_grid = np.asarray(x_grid, dtype=float)
    y_grid = np.asarray(y_grid, dtype=float)
    for suffix, field, title in fields:
        fig, ax = plt.subplots(figsize=(7, 6))
        low, high = (limits or {}).get(suffix, (None, None))
        mesh = ax.pcolormesh(x_grid, y_grid, np.ma.masked_invalid(field),
                             shading="nearest", cmap=cmap, vmin=low, vmax=high,
                             rasterized=True)
        ax.set_aspect("equal")
        ax.set_title(f"{title_prefix}{title}")
        ax.set_xlabel("x (mm)" if unit else "X")
        ax.set_ylabel("y (mm)" if unit else "Y")
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        colorbar_axis = make_axes_locatable(ax).append_axes("right", size="5%", pad=.12)
        fig.colorbar(mesh, cax=colorbar_axis, label=unit or "")
        fig.tight_layout()
        fig.savefig(output_dir / f"{case_name}_{suffix}.png", dpi=300, bbox_inches="tight")
        fig.savefig(output_dir / f"{case_name}_{suffix}.svg", bbox_inches="tight")
        plt.close(fig)

def grid_fields_from_points(coordinates, fields, grid_ids=None):
    """Restore lattice ordering without resampling any displacement values.

    New preparations retain integer grid IDs. For older frozen, axis-aligned
    clouds, recover IDs from equal coordinates within 1e-6 mm. Reject ambiguous
    layouts instead of filling or interpolating observations. Coordinates of
    missing cells alone use the reference affine lattice; their values are NaN.
    """
    xy = np.asarray(coordinates, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2 or not np.isfinite(xy).all():
        raise ValueError("Expected finite DIC coordinates of shape (points, 2)")
    if grid_ids is None:
        recovered = []
        for values in xy.T:
            order = np.argsort(values)
            starts = np.r_[True, np.diff(values[order]) > 1e-6]
            labels = np.empty(len(values), dtype=int)
            labels[order] = np.cumsum(starts)-1
            recovered.append(labels)
        grid_ids = np.column_stack([recovered[1], recovered[0]])
    ids = np.asarray(grid_ids)
    if (ids.shape != xy.shape or not np.isfinite(ids).all() or
            not np.array_equal(ids, np.round(ids)) or len(np.unique(ids, axis=0)) != len(ids)):
        raise ValueError("DIC grid IDs must be unique integer pairs, aligned with the saved points")
    ids = ids.astype(int)
    size = np.ptp(ids, axis=0)+1
    if min(size) < 2 or np.prod(size) > max(16*len(ids), 100):
        raise ValueError("Cannot reconstruct a supported DIC lattice; prepare again with grid IDs")
    design = np.column_stack([np.ones(len(ids)), ids])
    transform = np.linalg.lstsq(design, xy, rcond=None)[0]
    if np.linalg.matrix_rank(design) < 3 or np.max(np.abs(design @ transform-xy)) > 1e-5:
        raise ValueError("Reference points do not follow an affine DIC lattice; no interpolation is applied")
    row, col = np.indices(tuple(size))
    cells = np.column_stack([np.ones(row.size), row.ravel()+ids[:, 0].min(),
                             col.ravel()+ids[:, 1].min()]) @ transform
    cells = cells.reshape(*size, 2)
    index = tuple((ids-ids.min(axis=0)).T)
    cells[index] = xy
    grids = {}
    for name, values in fields.items():
        values = np.asarray(values, dtype=float)
        if values.shape != (len(xy),) or not np.isfinite(values).all():
            raise ValueError(f"{name}: expected one finite value per saved DIC point")
        grids[name] = np.full(tuple(size), np.nan)
        grids[name][index] = values
    if abs(transform[1, 0]) > abs(transform[2, 0]):
        cells = cells.transpose(1, 0, 2)
        grids = {name: values.T for name, values in grids.items()}
    return cells[..., 0], cells[..., 1], grids

def save_dic_comparison(output, cases, forward_states, inverse_states, fit_names, grid_ids=None):
    """DIC, calibrated forward, inverse and residual maps in post_DIC style.

    Predictions already include the solver's rigid registration. A common DIC
    component mean is subtracted only for displaying the three displacement
    maps; errors and exported point values remain in the original reference.
    """
    import pandas as pd

    output = Path(output)
    rows = []
    for case, forward, inverse in zip(cases, forward_states, inverse_states):
        xy = np.column_stack([case.observation.x_meas, case.observation.y_meas])
        observed = case.observation.u_measured.reshape(-1, 2)
        predictions = [state['prediction'].reshape(-1, 2) for state in (forward, inverse)]
        errors = [value-observed for value in predictions]
        maps, limits, titles = {}, {}, {}
        point_table = {'x_mm': xy[:, 0], 'y_mm': xy[:, 1]}
        for component, label in enumerate(('ux', 'uy')):
            origin = observed[:, component].mean()
            displayed = [values[:, component]-origin for values in (observed, *predictions)]
            low, high = min(v.min() for v in displayed), max(v.max() for v in displayed)
            if high == low:
                low, high = low-1e-12, high+1e-12
            bound = max(1e-12, *(np.abs(error[:, component]).max() for error in errors))
            names = ('dic', 'forward', 'inverse', 'forward_error', 'inverse_error')
            headings = ('DIC', 'Independent calibration', 'Inverse back-substitution',
                        'Independent - DIC', 'Inverse - DIC')
            values = [*displayed, *(error[:, component] for error in errors)]
            for i, (name, heading, value) in enumerate(zip(names, headings, values)):
                key = f'{name}_{label}'
                maps[key], titles[key] = value, f'{case.name}: {heading}, {label}'
                limits[key] = (low, high) if i < 3 else (-bound, bound)
                point_table[f'{key}_mm'] = value+origin if i < 3 else value
        x_grid, y_grid, fields = grid_fields_from_points(xy, maps, (grid_ids or {}).get(case.name))
        plot_grid_fields(output/case.name, case.name, x_grid, y_grid,
                         [(key, value, titles[key]) for key, value in fields.items()],
                         cmap='jet', limits=limits, title_prefix='', unit='mm')
        pd.DataFrame(point_table).to_csv(output/case.name/'point_fields.csv', index=False)
        for model, state, error in zip(('forward', 'inverse'), (forward, inverse), errors):
            rows.append({'case': case.name, 'role': 'fit' if case.name in fit_names else 'prediction',
                         'model': model, 'weighted_rms_mm': state['weighted_rms'],
                         'rmse_ux_mm': np.sqrt(np.mean(error[:, 0]**2)),
                         'rmse_uy_mm': np.sqrt(np.mean(error[:, 1]**2)),
                         'absolute_p95_ux_mm': np.quantile(np.abs(error[:, 0]), .95),
                         'absolute_p95_uy_mm': np.quantile(np.abs(error[:, 1]), .95)})
    table = pd.DataFrame(rows)
    table.to_csv(output/'forward_inverse_comparison.csv', index=False)
    return table
