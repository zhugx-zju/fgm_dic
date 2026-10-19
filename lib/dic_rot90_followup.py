"""Canonical rot90 presentation and frame-level mechanical consistency checks."""

import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyxlsb import open_workbook
from scipy.spatial import cKDTree

from fgm_asm.research import compare_displacement, make_observation_case, partition_modulus
from .dic_grid_orientation import orient_grid_field
from .dic_grid_utils import reconstruct_structured_grid
from .dic_revision_reporting import save_conditional_comparison
from .dic_revision_study import grid_cell_area, make_cases, state_metrics
from .experiment_data import (DICPlatform, check_platform_reference, file_sha256,
                              load_calibration, load_dic_frame_series, load_platform_force)
from .research_reporting import plot_modulus_comparison, save_run, save_x_profile
from .research_workflows import _inverse, _mesh, _regional_comparison, _solver_summary


def rigid_projection(xy: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Remove only translations and infinitesimal rotation from (...,points,2).

    Equal cell areas and a common sigma in this study make this projection
    identical to the observation residual's weighted rigid subspace.
    """
    xy, values = np.asarray(xy, dtype=float), np.asarray(values, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2 or values.shape[-2:] != xy.shape:
        raise ValueError('Expected coordinates (points,2) and fields (...,points,2)')
    centered_xy = xy-xy.mean(axis=0)
    rotational_mode = np.column_stack([-centered_xy[:, 1], centered_xy[:, 0]])
    denominator = np.sum(rotational_mode**2)
    if denominator <= 0 or not (np.isfinite(xy).all() and np.isfinite(values).all()):
        raise ValueError('Finite fields at distinct points are required')
    centered = values-values.mean(axis=-2, keepdims=True)
    angle = np.sum(centered*rotational_mode, axis=(-2, -1))/denominator
    return centered-angle[..., None, None]*rotational_mode


def residual_energy(residuals: np.ndarray) -> dict:
    """Separate persistent mean and within-window variation exactly, without IID assumptions."""
    values = np.asarray(residuals, dtype=float)
    mean = values.mean(axis=0)
    total = float(np.mean(values**2))
    persistent = float(np.mean(mean**2))
    changing = float(np.mean((values-mean)**2))
    return {'total_rms_mm': np.sqrt(total), 'persistent_mean_rms_mm': np.sqrt(persistent),
            'within_window_rms_mm': np.sqrt(changing),
            'persistent_energy_fraction': persistent/total if total > 0 else 0.,
            'decomposition_roundoff_mm2': abs(total-persistent-changing)}


def map_frame_forces(machine: np.ndarray, stages: np.ndarray, frequency: float,
                     offset_s: float, baseline_N: float) -> np.ndarray:
    """Interpolate actual force at mapped DIC times; reject extrapolation."""
    machine = np.asarray(machine, dtype=float)
    times = np.asarray(stages, dtype=float)/frequency-offset_s
    if (machine.ndim != 2 or machine.shape[1] != 2 or frequency <= 0 or
            not np.isfinite(machine).all() or np.any(np.diff(machine[:, 0]) <= 0)):
        raise ValueError('Need increasing machine times and finite [time_s,force_N] rows')
    if not np.isfinite(times).all() or times.min() < machine[0, 0] or times.max() > machine[-1, 0]:
        raise ValueError('DIC frames exceed available measured-force coverage')
    result = np.interp(times, machine[:, 0], machine[:, 1])-baseline_N
    if not np.isfinite(result).all() or np.any(result <= 0):
        raise ValueError('Corrected actual frame force must be positive')
    return result


def existing_orientation_registration(platform, length, width, operations):
    """Obtain the vector transform from the maintained orientation implementation."""
    table = pd.DataFrame(np.column_stack([platform.grid_ids, platform.coordinates,
                                         platform.displacement]),
                         columns=['grid_row', 'grid_col', 'X_ref', 'Y_ref', 'DX', 'DY'])
    grid = reconstruct_structured_grid(table, fields=('X_ref', 'Y_ref', 'DX', 'DY'))
    oriented = orient_grid_field(grid['X_ref'], grid['Y_ref'], grid['DX'], grid['DY'], operations)
    matrix = oriented['metadata']['transform_matrix']
    raw_center = (platform.coordinates.min(axis=0)+platform.coordinates.max(axis=0))/2
    offset = np.array([length/2, width/2])-matrix @ raw_center
    xy = platform.coordinates @ matrix.T+offset
    uv = platform.displacement @ matrix.T
    # Verify joint coordinate/vector mapping, including the array reordering.
    gx, gy = oriented['X_grid'], oriented['Y_grid']
    grid_xy = np.column_stack([gx.ravel()+(length-np.ptp(gx))/2,
                              gy.ravel()+(width-np.ptp(gy))/2])
    grid_uv = np.column_stack([oriented['UX_grid'].ravel(), oriented['UY_grid'].ravel()])
    # Nearly tied coordinate components can change lexicographic ordering under
    # roundoff after a constant translation. Match spatial points instead.
    distances, matched = cKDTree(xy).query(grid_xy)
    if len(np.unique(matched)) != len(xy) or np.max(distances) > 1e-9:
        raise ValueError('Maintained grid orientation and pointwise transform disagree')
    np.testing.assert_allclose(uv[matched], grid_uv, atol=1e-12, rtol=0)
    return matrix, offset


def _frame_diagnostics(cfg, series, forces, matrix, offset, fine_cases, calibrated,
                       inverse_cases, inverse_states):
    with open_workbook(str(cfg.MACHINE_FORCE_XLSB)) as book:
        with book.get_sheet(1) as sheet:
            rows = [[cell.v for cell in row] for row in sheet.rows()]
    machine = np.asarray(rows[1:], dtype=float)[:, :2]
    timing = json.loads(cfg.TIMING_AUDIT_JSON.read_text(encoding='utf-8'))['records']['3']
    baseline = float(np.median(machine[machine[:, 0] <= .08, 1]))
    np.testing.assert_allclose(baseline, timing['baseline_force_N'], atol=1e-10)
    frame_rows, affine_rows, energies, arrays, normalized = [], [], {}, {}, {}
    for name, source, fine_case, fixed_state, inv_case, inv_state in zip(
            series, series.values(), fine_cases, calibrated, inverse_cases, inverse_states):
        stages = np.array([record['stage'] for record in source.metadata['sources']])
        frame_forces = map_frame_forces(machine, stages, cfg.FRAME_RATE_HZ,
                                        timing['offset_median_s'], baseline)
        mean_force = forces[name]['mean_corrected_force_N']
        np.testing.assert_allclose(frame_forces.mean(), mean_force, rtol=0, atol=1e-9)
        xy = source.coordinates @ matrix.T+offset
        frames = source.frame_displacements @ matrix.T
        if len(xy) != len(inv_case.observation.x_meas):
            raise ValueError('Frame diagnostic currently requires full cloud inside the assumed ROI')
        normalized[name] = rigid_projection(xy, frames/frame_forces[:, None, None])
        design = np.column_stack([np.ones(len(xy)), xy-xy.mean(axis=0)])
        for stage, force, observed in zip(stages, frame_forces, frames):
            gradient = np.linalg.lstsq(design, observed, rcond=None)[0][1:].T
            affine_rows.append({'case': name, 'stage': stage, 'force_N': force,
                                'axial_strain': gradient[0, 0], 'transverse_strain': gradient[1, 1],
                                'apparent_contraction_ratio': -gradient[1, 1]/gradient[0, 0]})
        energies[name] = {}
        for model, case, state in [('independent', fine_case, fixed_state),
                                    ('inverse', inv_case, inv_state)]:
            base_prediction = np.asarray(case.observation.H @ state['U']).reshape(-1, 2)
            residuals = rigid_projection(xy, frames-base_prediction[None, :, :]*(frame_forces/mean_force)[:, None, None])
            energies[name][model] = residual_energy(residuals)
            arrays[f'{name}_{model}_frame_residuals'] = residuals
            for stage, force, residual in zip(stages, frame_forces, residuals):
                frame_rows.append({'case': name, 'stage': stage, 'model': model,
                                   'force_N': force, 'weighted_rms_mm': np.sqrt(np.mean(residual**2)),
                                   'rmse_ux_mm': np.sqrt(np.mean(residual[:, 0]**2)),
                                   'rmse_uy_mm': np.sqrt(np.mean(residual[:, 1]**2))})
        arrays[f'{name}_frame_forces_N'] = frame_forces
        arrays[f'{name}_stages'] = stages
        arrays[f'{name}_coordinates'] = xy
    first, second = normalized[cfg.FIT_CASE], normalized[cfg.PREDICT_CASE]
    # Every frame pair is diagnostic only; no favorable pair is selected as a replacement dataset.
    difference = (second[:, None, :, :]-first[None, :, :, :])*forces[cfg.PREDICT_CASE]['mean_corrected_force_N']
    pair_rms = np.sqrt(np.mean(difference**2, axis=(-2, -1)))
    arrays['all_frame_pair_proportionality_rms_mm'] = pair_rms
    affine = pd.DataFrame(affine_rows)
    means = affine.groupby('case')[['axial_strain', 'transverse_strain']].mean()
    ratios = means.loc[cfg.PREDICT_CASE]/means.loc[cfg.FIT_CASE]
    force1, force2 = [forces[key]['mean_corrected_force_N'] for key in (cfg.FIT_CASE, cfg.PREDICT_CASE)]
    implied = (force2-ratios*force1)/(ratios-1)
    summary = {'persistent_residual_energy': energies,
               'all_frame_pairs': {'count': int(pair_rms.size), 'min_rms_mm': pair_rms.min(),
                                   'median_rms_mm': np.median(pair_rms), 'max_rms_mm': pair_rms.max(),
                                   'scope': 'All pairs, force-normalized at measured frame forces; no frame selection.'},
               'global_strain_ratios_280_over_180': ratios.to_dict(),
               'implied_common_force_offset_N': implied.to_dict(),
               'force_offset_scope': 'Diagnostic offsets needed separately by axial/transverse normal strains; never applied to forces.',
               'force_source': {'path': cfg.MACHINE_FORCE_XLSB, 'sha256': file_sha256(cfg.MACHINE_FORCE_XLSB)},
               'timing_source': {'path': cfg.TIMING_AUDIT_JSON, 'sha256': file_sha256(cfg.TIMING_AUDIT_JSON)},
               'force_baseline_N': baseline, 'time_offset_s': timing['offset_median_s']}
    return summary, pd.DataFrame(frame_rows), affine, arrays


def run_rot90_followup(cfg, source, output):
    source, output = Path(source), Path(output)
    if output.exists():
        raise FileExistsError('Choose a new rot90 output directory')
    source_meta_path = source / 'inverse_primary' / 'run_metadata.json'
    source_meta = json.loads(source_meta_path.read_text(encoding='utf-8'))
    for key, digest in source_meta['source_sha256'].items():
        if key.replace('\\', '/').startswith('fgm_asm/'):
            if file_sha256(cfg.PROJECT_ROOT / key) != digest:
                raise ValueError('Mechanics kernel differs from the saved inverse')
    for key in ('GEO_L', 'GEO_H', 'THICKNESS', 'NU', 'NEL_X', 'NEL_Y', 'E_REF'):
        if source_meta['configuration'][key] != getattr(cfg, key):
            raise ValueError(f'{key} differs from the saved fit')
    series, platforms, forces = {}, {}, {}
    for name, stages in cfg.STAGES.items():
        series[name] = load_dic_frame_series(cfg.DATA_ROOT / (name+cfg.PRIMARY_REVISION), *stages)
        s = series[name]
        if s.metadata['sources'] != source_meta['platforms'][cfg.PRIMARY_REVISION][name]['sources']:
            raise ValueError('Saved and current DIC sources do not match')
        platforms[name] = DICPlatform(s.coordinates, s.frame_displacements.mean(axis=0),
                                      s.frame_displacements.std(axis=0, ddof=1), s.grid_ids, s.metadata)
        _, forces[name] = load_platform_force(cfg.FORCE_PAIRING_CSV, int(name[:-1]))
        if tuple(stages) != (forces[name]['stage_first'], forces[name]['stage_last']):
            raise ValueError('DIC window and paired force window differ')
    check_platform_reference(platforms)
    matrix, offset = existing_orientation_registration(platforms[cfg.FIT_CASE], cfg.GEO_L, cfg.GEO_H,
                                                        cfg.DIC_ORIENTATION_OPERATIONS)
    relative = matrix @ np.asarray(source_meta['matrix']).T
    np.testing.assert_allclose(relative, np.diag([-1., 1.]), atol=1e-12)
    mesh, fine = _mesh(cfg), _mesh(cfg, forward=True)
    with np.load(source / 'inverse_primary' / 'fields.npz') as data:
        field = data['E_inverse'].reshape(mesh.nods_y, mesh.nods_x)[:, ::-1].ravel()
        history = data['cost_history'].copy()
        old_predictions = {name: data[f'{name}_inverse_prediction'].reshape(-1, 2).copy() for name in series}
    calibration = load_calibration(cfg.CALIBRATION_CSV, cfg.CALIBRATION_METHOD, cfg.REGION_INFILL_PERCENT)
    reference = partition_modulus(calibration.mean_E_MPa, cfg.REGION_BOUNDARIES_X)
    fine_cases = make_cases(cfg, fine, platforms, forces, matrix, offset)
    inverse_cases = make_cases(cfg, mesh, platforms, forces, matrix, offset)
    calibrated = compare_displacement(fine, reference, fine_cases, cfg.NU)
    inverse_states = compare_displacement(mesh, field, inverse_cases, cfg.NU)
    for name, state in zip(series, inverse_states):
        np.testing.assert_allclose(state['prediction'].reshape(-1, 2), old_predictions[name] @ relative.T,
                                   atol=1e-10, rtol=1e-10)
    frame_summary, frame_table, affine, arrays = _frame_diagnostics(
        cfg, series, forces, matrix, offset, fine_cases, calibrated, inverse_cases, inverse_states)
    summary = {'workflow': 'rot90_reused_inverse_and_frame_diagnosis', 'coordinates_verified': False,
               'orientation_operations': cfg.DIC_ORIENTATION_OPERATIONS, 'matrix': matrix, 'offset_mm': offset,
               'registration_assumption': cfg.REGISTRATION_ASSUMPTION,
               'source_metadata_sha256': file_sha256(source_meta_path),
               'source_solver': source_meta['solver'], 'new_optimizer_iterations': 0,
               'selected_gamma_frozen': source_meta['selected_gamma'],
               'reuse_verification': 'Exact x reflection of saved nodal E; recomputed predictions match transformed saved predictions.',
               'force_pairing': forces, 'calibration': calibration.to_dict(orient='records'),
               'calibration_source_sha256': file_sha256(cfg.CALIBRATION_CSV),
               'platforms': {name: s.metadata for name, s in series.items()},
               'comparison': [{'case': state['name'], 'model': model, **state_metrics(state)}
                              for model, states in [('independent', calibrated), ('inverse', inverse_states)]
                              for state in states], 'frame_diagnostics': frame_summary}
    arrays.update(coordinates=mesh.coord, E_inverse=field, E_calibrated=reference(mesh.X, mesh.Y),
                  calibrated_coordinates=fine.coord, cost_history=history)
    for name, case, base, inv in zip(series, inverse_cases, calibrated, inverse_states):
        arrays[f'{name}_observed'] = case.observation.u_measured
        arrays[f'{name}_calibrated_prediction'] = base['prediction']
        arrays[f'{name}_inverse_prediction'] = inv['prediction']
        arrays[f'{name}_calibrated_U'] = base['U']
        arrays[f'{name}_inverse_U'] = inv['U']
    save_run(output, cfg, summary, arrays)
    frame_table.to_csv(output / 'frame_residuals.csv', index=False)
    affine.to_csv(output / 'frame_affine_strains.csv', index=False)
    save_conditional_comparison(output, inverse_cases, calibrated, inverse_states, [cfg.FIT_CASE])
    _regional_comparison(mesh, field, calibration, cfg.REGION_BOUNDARIES_X).to_csv(
        output / 'regional_modulus_comparison.csv', index=False)
    plot_modulus_comparison(output, mesh, arrays['E_calibrated'], field)
    save_x_profile(output, mesh, field, reference)
    _plot_frame_diagnostics(output, frame_table, affine, arrays, cfg)
    _incremental_diagnostic(cfg, output, source_meta['selected_gamma'], mesh, fine,
                             platforms, forces, matrix, offset, reference, calibration)
    return summary


def _incremental_diagnostic(cfg, output, gamma, mesh, fine, platforms, forces,
                             matrix, offset, reference, calibration):
    """Both loads are consumed here; this is a separate diagnostic, not held-out validation.

    A common additive reference displacement error cancels algebraically.
    True load evolution, state-dependent optical errors and anisotropy do not.
    Gamma is frozen; no result from this diagnostic changes the main estimator.
    """
    first, second = platforms[cfg.FIT_CASE], platforms[cfg.PREDICT_CASE]
    np.testing.assert_array_equal(first.grid_ids, second.grid_ids)
    xy = first.coordinates @ matrix.T+offset
    delta = (second.displacement-first.displacement) @ matrix.T
    force = forces[cfg.PREDICT_CASE]['mean_corrected_force_N']-forces[cfg.FIT_CASE]['mean_corrected_force_N']
    area = np.full(len(xy), grid_cell_area(first))
    cases = [make_observation_case(grid, 'increment_180_to_280', force, cfg.THICKNESS,
                                   xy, delta, area, cfg.WORKING_SIGMA_MM)
             for grid in (mesh, fine)]
    run_cfg = SimpleNamespace(**{k: v for k, v in vars(cfg).items() if k.isupper()},
                              __file__=cfg.__file__)
    run_cfg.GAMMA = gamma
    result = _inverse(run_cfg, mesh, cases[:1])
    inv = compare_displacement(mesh, result['E_final'], cases[:1], cfg.NU)
    base = compare_displacement(fine, reference, cases[1:], cfg.NU)
    design = np.column_stack([np.ones(len(xy)), xy-xy.mean(axis=0)])
    gradient = np.linalg.lstsq(design, delta, rcond=None)[0][1:].T
    summary = {'workflow': 'shared_reference_cancellation_diagnostic', 'coordinates_verified': False,
               'scope': 'Uses BOTH 180N and 280N. No held-out prediction or new model selection claim.',
               'reference_rule': 'u_280-u_180 at matched initial coordinates; F_280-F_180',
               'delta_force_N': force, 'gamma_frozen': gamma, 'solver': _solver_summary(result),
               'increment_global_axial_strain': gradient[0, 0],
               'increment_global_transverse_strain': gradient[1, 1],
               'increment_apparent_contraction_ratio': -gradient[1, 1]/gradient[0, 0],
               'independent': state_metrics(base[0]), 'inverse': state_metrics(inv[0])}
    destination = output / 'incremental_diagnostic'
    save_run(destination, run_cfg, summary,
             {'coordinates': mesh.coord, 'E_inverse': result['E_final'], 'observation_coordinates': xy,
              'increment_observed': delta, 'inverse_prediction': inv[0]['prediction'],
              'independent_prediction': base[0]['prediction'], 'cost_history': result['cost_history'],
              'gradient_norms': result['gradient_norms']})
    save_conditional_comparison(destination, cases[:1], base, inv, ['increment_180_to_280'])
    _regional_comparison(mesh, result['E_final'], calibration, cfg.REGION_BOUNDARIES_X).to_csv(
        destination / 'regional_modulus_comparison.csv', index=False)
    save_x_profile(destination, mesh, result['E_final'], reference)


def _plot_frame_diagnostics(output, table, affine, arrays, cfg):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for col, name in enumerate(cfg.STAGES):
        part = table[table.case == name]
        for model, color in [('independent', '#666666'), ('inverse', '#0072B2')]:
            values = part[part.model == model]
            axes[0, col].plot(values.stage, values.weighted_rms_mm*1000, 'o-', ms=3,
                              color=color, label=model)
        axes[0, col].set(title=f'{name}: each frame at measured force', xlabel='DIC stage', ylabel='RMS (micrometers)')
        axes[0, col].legend(frameon=False)
        values = affine[affine.case == name]
        axes[1, col].plot(values.stage, values.apparent_contraction_ratio, 'o-', ms=3, color='#D55E00')
        axes[1, col].axhline(cfg.NU, ls='--', color='black', lw=1, label='Assumed isotropic nu')
        axes[1, col].set(title='Global affine contraction ratio (not a nu measurement)',
                         xlabel='DIC stage', ylabel='-transverse / axial strain', ylim=(.25, .7))
        axes[1, col].legend(frameon=False)
    fig.suptitle('rot90 working coordinates; frame diagnostics do not verify physical landmarks', fontsize=11)
    for ext in ('png', 'svg'):
        fig.savefig(output / f'frame_diagnostics.{ext}', dpi=250)
    plt.close(fig)
