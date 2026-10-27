"""DIC lattice reconstruction and original post_DIC cloud plots."""

from __future__ import annotations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RELATIVE_ERROR_DESCRIPTION = (
    '有符号相对误差 e_c=100×(u_model,c−u_DIC,c)/s_c（%）；'
    's_c=||u_DIC,c的变形分量||_W/√Σw，变形分量已去除两平移和一微小转动。'
    '整体位移相对误差 ε_c=100×||u_model,c−u_DIC,c||_W/||u_DIC,c的变形分量||_W。'
    '分母只由 DIC 决定，同一平台的全部模型与 ν 共用，不逐点除以过零位移。'
    '无可分辨变形的分量标为未定义，不用任意小常数替代分母。'
)

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
    bounds: tuple[float, float, float, float] | None = None,
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
        if np.isfinite(field).any():
            mesh = ax.pcolormesh(x_grid, y_grid, np.ma.masked_invalid(field),
                                 shading="nearest", cmap=cmap, vmin=low, vmax=high,
                                 rasterized=True)
            from mpl_toolkits.axes_grid1 import make_axes_locatable

            colorbar_axis = make_axes_locatable(ax).append_axes("right", size="5%", pad=.12)
            fig.colorbar(mesh, cax=colorbar_axis, label=unit or "")
        else:
            ax.set_xlim(float(x_grid.min()), float(x_grid.max()))
            ax.set_ylim(float(y_grid.min()), float(y_grid.max()))
            ax.text(.5, .5, 'No resolved DIC deformation\nRelative error undefined',
                    ha='center', va='center', transform=ax.transAxes)
        ax.set_aspect("equal")
        ax.set_title(f"{title_prefix}{title}")
        ax.set_xlabel("x (mm)" if unit else "X")
        ax.set_ylabel("y (mm)" if unit else "Y")
        if bounds is not None:
            ax.set_xlim(bounds[:2])
            ax.set_ylim(bounds[2:])
        fig.tight_layout()
        fig.savefig(output_dir / f"{case_name}_{suffix}.png", dpi=300, bbox_inches="tight")
        fig.savefig(output_dir / f"{case_name}_{suffix}.svg", bbox_inches="tight")
        plt.close(fig)


def plot_displacement_clouds(output, name, xy, displacement, grid_ids=None,
                             display_origin=(0., 0.), title=''):
    """Plot saved point values; reference changes affect display only."""
    values = np.asarray(displacement, dtype=float).reshape(-1, 2)
    origin = np.asarray(display_origin, dtype=float)
    if values.shape != np.asarray(xy).shape or origin.shape != (2,):
        raise ValueError('Displacements and coordinates must be (points, 2); display origin must be (2,)')
    x, y, fields = grid_fields_from_points(
        xy, {label: values[:, component]-origin[component]
             for component, label in enumerate(('ux', 'uy'))}, grid_ids)
    plot_grid_fields(Path(output), name, x, y,
                     [(label, field, f'{title}: {label}') for label, field in fields.items()],
                     cmap='jet', unit='mm', title_prefix='')

def grid_fields_from_points(coordinates, fields, grid_ids=None, *, allow_nan=False):
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
        valid = not np.isinf(values).any() if allow_nan else np.isfinite(values).all()
        if values.shape != (len(xy),) or not valid:
            raise ValueError(f"{name}: expected one finite value per saved DIC point")
        grids[name] = np.full(tuple(size), np.nan)
        grids[name][index] = values
    if abs(transform[1, 0]) > abs(transform[2, 0]):
        cells = cells.transpose(1, 0, 2)
        grids = {name: values.T for name, values in grids.items()}
    return cells[..., 0], cells[..., 1], grids

def dic_relative_error_scale(case):
    """DIC-only component RMS in mm, using the existing rigid projection.

    This calls the observation operator at zero displacement, not a FE solve.
    Its residual is DIC minus its weighted best rigid fit. The normalization
    is invariant to addition of the same three rigid modes to both fields.
    A roundoff-level zero has no relative error definition; return NaN.
    """
    from fgm_asm.load_case import evaluate_observation

    obs = case.observation
    if not case.remove_rigid_body:
        raise ValueError('Relative DIC deformation errors require three-mode rigid registration')
    deformation = evaluate_observation(case, np.zeros(obs.H.shape[1]))['residual'].reshape(-1, 2)
    weights = obs.weight_vector.reshape(-1, 2)
    total = weights.sum(axis=0)
    if np.any(total <= 0):
        raise ValueError('Each DIC component needs positive observation weight')
    scale = np.sqrt(np.sum(weights*deformation**2, axis=0)/total)
    raw_rms = np.sqrt(np.sum(weights*obs.u_measured.reshape(-1, 2)**2, axis=0)/total)
    scale[scale <= 64*np.finfo(float).eps*np.maximum(raw_rms, np.finfo(float).tiny)] = np.nan
    return scale


def relative_displacement_errors(case, prediction):
    """Area-weighted L2 relative errors, normalized only by DIC deformation.

    The prediction includes the workflow's fitted rigid modes. No pointwise
    displacement division or material/noise normalization enters this metric.
    """
    from fgm_asm.load_case import evaluate_observation

    obs = case.observation
    predicted = np.asarray(prediction, dtype=float).reshape(-1)
    if predicted.shape != obs.u_measured.shape or not np.isfinite(predicted).all():
        raise ValueError('Prediction must be finite and aligned with the flattened observation vector.')
    deformation = evaluate_observation(case, np.zeros(obs.H.shape[1]))['residual'].reshape(-1, 2)
    residual = (obs.u_measured-predicted).reshape(-1, 2)
    weight = obs.weight_vector.reshape(-1, 2)
    numerator = np.sum(weight*residual**2, axis=0)
    denominator = np.sum(weight*deformation**2, axis=0)
    defined = np.isfinite(dic_relative_error_scale(case))
    result = np.full(2, np.nan)
    result[defined] = 100*np.sqrt(numerator[defined]/denominator[defined])
    total = (100*np.sqrt(numerator.sum()/denominator.sum()) if defined.any() else np.nan)
    return {'axial_relative_error_percent': float(result[0]),
            'transverse_relative_error_percent': float(result[1]),
            'displacement_relative_error_percent': float(total)}


def plot_dic_fe_components(output, name, x_grid, y_grid, fields, limits, model, title,
                           observation_label='Processed DIC'):
    """Show both displacement components in one figure, in the DIC reference.

    ``fields`` contains d-mean(d) and HU+Rc-mean(d), already evaluated at
    identical DIC positions. This is a display of the existing residual, not
    a separate rigid fit, FE interpolation, or displacement normalization.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 5.4), sharex=True, sharey=True,
                             layout='constrained')
    fig.suptitle(title, fontsize=14)
    for column, component in enumerate(('ux', 'uy')):
        low, high = limits[f'dic_{component}']
        for row, (key, label) in enumerate((('dic', observation_label), (model, 'Registered FE'))):
            ax = axes[row, column]
            image = ax.pcolormesh(x_grid, y_grid, np.ma.masked_invalid(fields[f'{key}_{component}']),
                                  shading='nearest', cmap='jet', vmin=low, vmax=high,
                                  rasterized=True)
            ax.set(xlim=(x_grid.min(), x_grid.max()), ylim=(y_grid.min(), y_grid.max()),
                   aspect='equal', title=f'{label}: $u_{component[-1]}$')
            ax.set_xlabel('x (mm) - loading direction')
            ax.set_ylabel('y (mm)')
            ax.tick_params(labelbottom=True, labelleft=True)
        fig.colorbar(image, ax=axes[:, column], label=f'$u_{component[-1]}$ (mm)', shrink=.85)
    fig.supxlabel('Same observation points and FEM axes; one color scale per component.\n'
                  'Observation: d - mean(d); FE: HU + fitted rigid motion - mean(d).', fontsize=10)
    stem = Path(output)/f'{name}_{model}_dic_components'
    fig.savefig(stem.with_suffix('.png'), dpi=300)
    fig.savefig(stem.with_suffix('.svg'))
    plt.close(fig)


def save_dic_comparison(output, cases, model_states, fit_names, grid_ids=None, *, nu=None):
    """DIC, one or both FE models, and signed relative errors in post_DIC style.

    Predictions already include the solver's rigid registration. A common DIC
    component mean is subtracted only for displacement display. Point CSVs
    retain original mm displacements/residuals and add normalized percentages.
    """
    import pandas as pd

    output = Path(output)
    headings = {'forward': 'Independent calibration', 'inverse': 'Inverse back-substitution'}
    if not model_states or set(model_states)-headings.keys():
        raise ValueError('Expected forward and/or inverse saved model states')
    if any(len(states) != len(cases) for states in model_states.values()):
        raise ValueError('Each model must contain one state per DIC case')
    rows, relative_rows = [], []
    for index, case in enumerate(cases):
        observation_label = 'Synthetic FE' if case.metadata.get('synthetic_observations') else 'DIC'
        condition = case.name if nu is None else f'{case.name}, nu={nu:.2f}'
        xy = np.column_stack([case.observation.x_meas, case.observation.y_meas])
        observed = case.observation.u_measured.reshape(-1, 2)
        states = {model: values[index] for model, values in model_states.items()}
        predictions = {model: state['prediction'].reshape(-1, 2) for model, state in states.items()}
        if any(value.shape != observed.shape or not np.isfinite(value).all() for value in predictions.values()):
            raise ValueError('Each prediction must be finite and aligned with DIC points')
        errors = {model: value-observed for model, value in predictions.items()}
        scale = dic_relative_error_scale(case)
        relative = {model: 100*error/scale for model, error in errors.items()}
        maps, error_maps, limits, titles = {}, {}, {}, {}
        point_table = {'x_mm': xy[:, 0], 'y_mm': xy[:, 1]}
        for component, label in enumerate(('ux', 'uy')):
            origin = observed[:, component].mean()
            displayed = {model: values[:, component]-origin for model, values in
                         {'dic': observed, **predictions}.items()}
            low, high = min(v.min() for v in displayed.values()), max(v.max() for v in displayed.values())
            if high == low:
                low, high = low-1e-12, high+1e-12
            bound = (max(1e-12, *(np.abs(error[:, component]).max() for error in relative.values()))
                     if np.isfinite(scale[component]) else 1.)
            for model, value in displayed.items():
                key = f'{model}_{label}'
                maps[key], titles[key] = value, f'{condition}\n{headings.get(model, observation_label)}: {label}'
                limits[key] = (low, high)
                point_table[f'{key}_mm'] = {'dic': observed, **predictions}[model][:, component]
            for model, error in errors.items():
                key = f'{model}_error_{label}'
                error_maps[key] = relative[model][:, component]
                titles[key] = f'{condition}: {model} relative error, {label}\n{observation_label} deformation amplitude = {scale[component]:.5g} mm'
                limits[key] = (-bound, bound)
                point_table[f'{key}_mm'] = error[:, component]
                point_table[f'{model}_relative_error_{label}_percent'] = relative[model][:, component]
            point_table[f'dic_deformation_rms_{label}_mm'] = np.full(len(xy), scale[component])
        ids = (grid_ids or {}).get(case.name)
        x_grid, y_grid, fields = grid_fields_from_points(xy, maps, ids)
        plot_grid_fields(output/case.name, case.name, x_grid, y_grid,
                         [(key, value, titles[key]) for key, value in fields.items()],
                         cmap='jet', limits=limits, title_prefix='', unit='mm')
        for model in states:
            label = headings[model]
            if model == 'inverse' and case.name not in fit_names:
                label = 'Inverse modulus: held-out prediction'
            plot_dic_fe_components(output/case.name, case.name, x_grid, y_grid,
                                   fields, limits, model, f'{condition} | {label}',
                                   observation_label=observation_label)
        x_grid, y_grid, fields = grid_fields_from_points(xy, error_maps, ids, allow_nan=True)
        plot_grid_fields(output/case.name, case.name, x_grid, y_grid,
                         [(key, value, titles[key]) for key, value in fields.items()],
                         cmap='jet', limits=limits, title_prefix='', unit=f'% of {observation_label} deformation amplitude')
        pd.DataFrame(point_table).to_csv(output/case.name/'point_fields.csv', index=False)
        for model, state in states.items():
            error = errors[model]
            rows.append({'case': case.name, 'role': 'fit' if case.name in fit_names else 'prediction',
                         'model': model, 'weighted_rms_mm': state['weighted_rms'],
                         'rmse_ux_mm': np.sqrt(np.mean(error[:, 0]**2)),
                         'rmse_uy_mm': np.sqrt(np.mean(error[:, 1]**2)),
                         'absolute_p95_ux_mm': np.quantile(np.abs(error[:, 0]), .95),
                         'absolute_p95_uy_mm': np.quantile(np.abs(error[:, 1]), .95)})
            weights = case.observation.weight_vector.reshape(-1, 2)
            for component, label in enumerate(('ux', 'uy')):
                values = relative[model][:, component]
                relative_rows.append({
                    'case': case.name, 'role': rows[-1]['role'], 'model': model, 'component': label,
                    'dic_deformation_rms_mm': scale[component], 'relative_defined': bool(np.isfinite(scale[component])),
                    'relative_weighted_rms_percent': np.sqrt(np.average(values**2, weights=weights[:, component])),
                    'absolute_relative_p95_percent': np.quantile(np.abs(values), .95),
                })
    table = pd.DataFrame(rows)
    if len(model_states) == 2:
        table.to_csv(output/'forward_inverse_comparison.csv', index=False)
    pd.DataFrame(relative_rows).to_csv(output/'relative_error_summary.csv', index=False)
    return table
