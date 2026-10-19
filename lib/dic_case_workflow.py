"""Independent preparation, forward, inverse and comparison workflow stages.

Physical geometry is independent of observation coverage. Both forward and
inverse use the same Q4 mesh; raw point observations retain their coordinates.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from fgm_asm import MeshInfo, lbfgs_inverse_solver_scipy
from fgm_asm.load_case import evaluate_observation
from fgm_asm.research import make_observation_case, partition_modulus, solve_forward
from .dic_revision_reporting import save_conditional_comparison
from .dic_revision_study import block_split
from .experiment_data import (check_platform_reference, dic_roi_mask, file_sha256,
                              load_calibration, load_dic_platform)
from .research_reporting import plot_modulus_comparison, save_run, save_x_profile
from .research_workflows import _regional_comparison, _solver_summary


def registration_transform(reference_xy, specification):
    """Return one unit-scale rigid/reflection transform and auditable status.

    Landmark fitting never estimates scale. A centered cloud is an explicitly
    conditional placement, not evidence for material-zone registration.
    """
    mode = specification['mode']
    note = specification.get('note', '').strip()
    if mode == 'pending':
        return None, None, False
    if not note:
        raise ValueError('Registration requires a provenance/assumption note')
    if mode == 'landmarks':
        raw = np.asarray(specification['dic_points_mm'], dtype=float)
        target = np.asarray(specification['fem_points_mm'], dtype=float)
        if (raw.shape != target.shape or raw.ndim != 2 or raw.shape[1] != 2 or len(raw) < 3 or
                not np.isfinite(raw).all() or not np.isfinite(target).all() or
                np.linalg.matrix_rank(raw-raw.mean(axis=0)) < 2 or
                np.linalg.matrix_rank(target-target.mean(axis=0)) < 2):
            raise ValueError('At least three finite noncollinear paired landmarks are required')
        u, _, vt = np.linalg.svd((raw-raw.mean(axis=0)).T @ (target-target.mean(axis=0)))
        matrix = (u @ vt).T
        offset = target.mean(axis=0)-matrix @ raw.mean(axis=0)
        error = np.max(np.linalg.norm(raw @ matrix.T+offset-target, axis=1))
        tolerance = float(specification['tolerance_mm'])
        if not np.isfinite(tolerance) or tolerance <= 0 or error > tolerance:
            raise ValueError(f'Landmarks disagree under unit-scale registration: max error {error:.6g} mm')
        verified = True
    else:
        matrix = np.asarray(specification['matrix'], dtype=float)
        if mode == 'explicit_verified':
            offset = np.asarray(specification['offset_mm'], dtype=float)
            verified = True
        elif mode == 'assumed_center':
            xy = np.asarray(reference_xy, dtype=float)
            offset = np.asarray(specification['cloud_center_fem_mm'])-matrix @ (
                xy.min(axis=0)+xy.max(axis=0))/2
            verified = False
        else:
            raise ValueError('Registration mode must be pending, landmarks, explicit_verified or assumed_center')
    if (matrix.shape != (2, 2) or offset.shape != (2,) or not np.isfinite(matrix).all() or
            not np.isfinite(offset).all() or
            not np.allclose(matrix @ matrix.T, np.eye(2), atol=1e-10, rtol=1e-9)):
        raise ValueError('Coordinates are mm: matrix must preserve lengths; no span normalization')
    return matrix, offset, verified


def measured_grid_spacing(platform, matrix):
    """Spacing measured from adjacent grid IDs, independent of ROI dimensions."""
    xy = platform.coordinates @ matrix.T
    result = []
    for axis in (0, 1):
        order = np.lexsort((platform.grid_ids[:, 1-axis], platform.grid_ids[:, axis]))
        ids, points = platform.grid_ids[order], xy[order]
        delta = np.diff(ids, axis=0)
        adjacent = (delta[:, axis] == 0) & (np.abs(delta[:, 1-axis]) == 1)
        distances = np.linalg.norm(np.diff(points, axis=0)[adjacent], axis=1)
        if not len(distances) or np.min(distances) <= 0:
            raise ValueError('Need adjacent grid IDs to determine DIC sampling')
        spacing = float(np.median(distances))
        if np.max(np.abs(distances/spacing-1)) > 1e-3:
            raise ValueError('Nonuniform DIC sampling needs explicit point-area weights')
        result.append(spacing)
    return result


def _force_record(cfg, name, item):
    if 'force_N' in item:
        if not item.get('force_note', '').strip():
            raise ValueError(f'{name}: explicit actual force requires a force_note')
        record = {'mean_corrected_force_N': float(item['force_N']),
                  'provenance': item['force_note'], 'reference': cfg.FORCE_REFERENCE}
    else:
        table = pd.read_csv(cfg.FORCE_PAIRING_CSV)
        selected = table[(table['run'] == item['run']) & (table['setting_N'] == item['setting_N'])]
        if len(selected) != 1:
            raise ValueError(f'{name}: force pairing must match exactly one run/setting row')
        record = selected.iloc[0].to_dict()
        if record['full_force_coverage'] != True:
            raise ValueError(f'{name}: force record does not cover the full DIC window')
        if tuple(item['stages']) != (record['stage_first'], record['stage_last']):
            raise ValueError(f'{name}: DIC and force windows differ')
    force = record['mean_corrected_force_N']
    if not np.isfinite(force) or force <= 0:
        raise ValueError(f'{name}: actual force must be finite and positive')
    return record


def _validate(cfg):
    if cfg.DIC_UNIT != 'mm':
        raise ValueError('Convert positions AND displacements to mm before this pipeline')
    if (cfg.DISPLACEMENT_REFERENCE != 'initial_unloaded_image' or
            cfg.FORCE_REFERENCE != 'corrected_force_from_unloaded_state'):
        raise ValueError('These steps require unloaded-reference displacement and matched actual force')
    for key in ('GEO_L', 'GEO_H', 'THICKNESS', 'E_REF', 'NOISE_SCALE_MM'):
        if not np.isfinite(getattr(cfg, key)) or getattr(cfg, key) <= 0:
            raise ValueError(f'{key} must be positive and finite')
    for key in ('NEL_X', 'NEL_Y', 'MAX_ITER', 'MAX_FUN'):
        if int(getattr(cfg, key)) != getattr(cfg, key) or getattr(cfg, key) <= 0:
            raise ValueError(f'{key} must be a positive integer')
    if not cfg.NU_VALUES or any(not 0 < nu < .5 for nu in cfg.NU_VALUES):
        raise ValueError('Supply a nonempty list of physical Poisson ratios')
    if cfg.PRIMARY_NU not in cfg.NU_VALUES:
        raise ValueError('PRIMARY_NU must be fixed in NU_VALUES before comparison')
    if set(cfg.FIT_CASES) & set(cfg.PREDICT_CASES) or not cfg.FIT_CASES or not cfg.PREDICT_CASES:
        raise ValueError('Need disjoint, nonempty fit and held-out case lists')
    if set(cfg.FIT_CASES) | set(cfg.PREDICT_CASES) != set(cfg.PLATFORMS):
        raise ValueError('Every platform must be assigned to fit or prediction')
    if cfg.OUTSIDE_POINT_POLICY not in ('reject', 'crop_to_physical_roi'):
        raise ValueError('Choose reject or crop_to_physical_roi for points outside the model')
    if not cfg.GAMMA_CANDIDATES or any(g < 0 or not np.isfinite(g) for g in cfg.GAMMA_CANDIDATES):
        raise ValueError('Supply finite nonnegative gamma candidates')
    edges = np.r_[0., cfg.REGION_BOUNDARIES_X, cfg.GEO_L]
    if len(edges) != len(cfg.REGION_INFILL_PERCENT)+1 or np.any(np.diff(edges) <= 0):
        raise ValueError('Material regions must partition the physical model length')


def _subset_case(case, mask):
    """Preserve the model, weights and rigid policy while changing only observed points."""
    from dataclasses import replace

    obs = case.observation
    dofs = np.column_stack([2*np.flatnonzero(mask), 2*np.flatnonzero(mask)+1]).ravel()
    subset = replace(obs, x_meas=obs.x_meas[mask], y_meas=obs.y_meas[mask],
                     ux_meas=obs.ux_meas[mask], uy_meas=obs.uy_meas[mask],
                     u_measured=obs.u_measured[dofs], point_weights=obs.point_weights[mask],
                     weight_vector=obs.weight_vector[dofs], H=obs.H[dofs], W=obs.W[dofs][:, dofs])
    return replace(case, observation=subset)


def _optimize(cfg, mesh, cases, gamma, nu):
    return lbfgs_inverse_solver_scipy(
        mesh, observation_cases=cases, E_init=np.full(mesh.n_nod, cfg.E_REF),
        gamma=gamma, modulus_scale=cfg.E_REF, max_iter=cfg.MAX_ITER, max_fun=cfg.MAX_FUN,
        ftol=cfg.FTOL, gtol=cfg.GTOL, nu=nu,
        regularization_mode=cfg.REGULARIZATION_MODE, tgv_options=cfg.TGV_OPTIONS)


def _metrics(state):
    residual = state['residual'].reshape(-1, 2)
    return {'rms_mm': state['weighted_rms'],
            'rmse_ux_mm': np.sqrt(np.mean(residual[:, 0]**2)),
            'rmse_uy_mm': np.sqrt(np.mean(residual[:, 1]**2)),
            'p95_ux_mm': np.quantile(np.abs(residual[:, 0]), .95),
            'p95_uy_mm': np.quantile(np.abs(residual[:, 1]), .95)}


def prepare_dic(cfg, output):
    """Stage 1: audit and freeze one case's inputs; do not run FE or inversion."""
    _validate(cfg)
    output = Path(output)
    if output.exists():
        raise FileExistsError(f'Choose a new preparation output: {output}')
    platforms, records = {}, {}
    for name, item in cfg.PLATFORMS.items():
        platforms[name] = load_dic_platform(Path(item['folder']), *item['stages'],
                                            coordinate_state=cfg.DIC_COORDINATE_STATE)
        records[name] = _force_record(cfg, name, item)
    shared = check_platform_reference(platforms)
    calibration = load_calibration(Path(cfg.CALIBRATION_CSV), cfg.CALIBRATION_METHOD, cfg.REGION_INFILL_PERCENT)
    reference = platforms[cfg.FIT_CASES[0]]
    matrix, offset, verified = registration_transform(reference.coordinates, cfg.REGISTRATION)
    summary = {'workflow': 'dic_case_preparation', 'status': 'prepared', 'case_id': cfg.CASE_ID,
               'coordinates_verified': verified, 'registration': cfg.REGISTRATION,
               'matrix': matrix, 'offset_mm': offset,
               'physical_model_mm': [cfg.GEO_L, cfg.GEO_H, cfg.THICKNESS],
               'shared_mesh': {'nel_x': cfg.NEL_X, 'nel_y': cfg.NEL_Y,
                               'hx_mm': cfg.GEO_L/cfg.NEL_X, 'hy_mm': cfg.GEO_H/cfg.NEL_Y,
                               'same_for_forward_and_inverse': True},
               'force_pairing': records, 'shared_reference': shared,
               'platforms': {name: p.metadata for name, p in platforms.items()},
               'calibration': calibration.to_dict(orient='records'),
               'calibration_source_sha256': file_sha256(Path(cfg.CALIBRATION_CSV)),
               'rigid_policy': 'FE has three gauge constraints; observed residual fits the same two translations and infinitesimal rotation',
               'primary_nu_prespecified': cfg.PRIMARY_NU,
               'physical_validation_status': 'not_established; geometry registration and boundary-equivalence evidence are distinct',
               'scope': 'registered comparison' if verified else 'conditional comparison under explicit registration assumptions'}
    if hasattr(cfg, 'FORCE_PAIRING_CSV') and any('force_N' not in i for i in cfg.PLATFORMS.values()):
        summary['force_source_sha256'] = file_sha256(Path(cfg.FORCE_PAIRING_CSV))
    arrays, rows = {}, []
    for name, platform in platforms.items():
        spacing = measured_grid_spacing(platform, matrix if matrix is not None else np.eye(2))
        row = {'case': name, 'dic_row_step_mm': spacing[0], 'dic_col_step_mm': spacing[1],
               'points': len(platform.coordinates), 'frame_count': platform.metadata['frame_count']}
        arrays[f'{name}_raw_reference_coordinates'] = platform.coordinates
        arrays[f'{name}_raw_mean_displacement'] = platform.displacement
        if matrix is not None:
            xy, uv = platform.coordinates @ matrix.T+offset, platform.displacement @ matrix.T
            inside = dic_roi_mask(xy, cfg.GEO_L, cfg.GEO_H)
            if not inside.all() and cfg.OUTSIDE_POINT_POLICY == 'reject':
                raise ValueError(f'{name}: {(~inside).sum()} points outside physical ROI; check registration or explicitly configure cropping')
            if inside.sum() < 4 or np.any(np.ptp(xy[inside], axis=0) == 0):
                raise ValueError(f'{name}: insufficient 2D observation coverage')
            row.update(points_inside=int(inside.sum()), points_outside=int((~inside).sum()),
                       x_min_mm=float(xy[inside, 0].min()), x_max_mm=float(xy[inside, 0].max()),
                       y_min_mm=float(xy[inside, 1].min()), y_max_mm=float(xy[inside, 1].max()))
            arrays[f'{name}_coordinates'] = xy[inside]
            arrays[f'{name}_observed'] = uv[inside]
            arrays[f'{name}_point_area_mm2'] = np.full(inside.sum(), np.prod(spacing))
        rows.append(row)
    summary['coverage'] = rows
    if matrix is None:
        summary['status'] = 'needs_registration'
    save_run(output, cfg, summary, arrays)
    pd.DataFrame(rows).to_csv(output / 'input_audit.csv', index=False)
    return summary


def load_prepared(source):
    """Downstream stages consume the frozen preparation, not changed raw inputs."""
    source = Path(source)
    metadata = json.loads((source / 'run_metadata.json').read_text(encoding='utf-8'))
    if metadata['workflow'] != 'dic_case_preparation':
        raise ValueError('Expected stage 1 preparation directory')
    if metadata['status'] == 'needs_registration':
        raise ValueError('Preparation records pending registration; supply physical landmarks or an explicit conditional placement')
    root = Path(__file__).resolve().parents[1]
    for path, digest in metadata['source_sha256'].items():
        if path.replace('\\', '/').startswith('fgm_asm/'):
            if file_sha256(root / path) != digest:
                raise ValueError('FE kernel changed since preparation; prepare a new case')
    cfg = SimpleNamespace(**metadata['configuration'], __file__=str(source/'configuration_snapshot.py')
                          if (source/'configuration_snapshot.py').exists() else None)
    mesh = MeshInfo(cfg.GEO_L, cfg.GEO_H, cfg.NEL_X, cfg.NEL_Y)
    arrays, cases = {}, []
    with np.load(source/'fields.npz') as fields:
        arrays = {key: fields[key].copy() for key in fields.files}
    for name in cfg.PLATFORMS:
        cases.append(make_observation_case(
            mesh, name, metadata['force_pairing'][name]['mean_corrected_force_N'], cfg.THICKNESS,
            arrays[f'{name}_coordinates'], arrays[f'{name}_observed'],
            arrays[f'{name}_point_area_mm2'], cfg.NOISE_SCALE_MM,
            metadata={'coordinates_verified': metadata['coordinates_verified'],
                      'registration': cfg.REGISTRATION, 'displacement_reference': cfg.DISPLACEMENT_REFERENCE}))
    provenance = {'prepared_metadata_sha256': file_sha256(source/'run_metadata.json'),
                  'prepared_fields_sha256': file_sha256(source/'fields.npz')}
    return cfg, mesh, cases, metadata, provenance


def _stage_summary(prepared, provenance, stage):
    summary = {key: prepared[key] for key in ('case_id', 'coordinates_verified', 'registration',
                                              'matrix', 'offset_mm', 'physical_model_mm', 'shared_mesh',
                                              'coverage', 'force_pairing', 'scope', 'primary_nu_prespecified')}
    summary.update(provenance, workflow=stage, status='running')
    return summary


def forward_calibrated(prepared, output):
    """Stage 2: independent calibrated forward predictions on the shared mesh."""
    if Path(output).exists():
        raise FileExistsError('Choose a new forward output')
    cfg, mesh, cases, metadata, provenance = load_prepared(prepared)
    calibration = pd.DataFrame(metadata['calibration'])
    field = partition_modulus(calibration.mean_E_MPa, cfg.REGION_BOUNDARIES_X)
    summary = _stage_summary(metadata, provenance, 'dic_calibrated_forward')
    save_run(output, cfg, summary)
    output = Path(output)
    metrics = []
    for nu in cfg.NU_VALUES:
        arrays = {'coordinates': mesh.coord, 'E_calibrated': field(mesh.X, mesh.Y)}
        for case in cases:
            u, _ = solve_forward(mesh, case.bc_info['tensile_end_force'], cfg.THICKNESS, nu, field)
            state = evaluate_observation(case, u)
            arrays[f'{case.name}_prediction'] = state['prediction']
            arrays[f'{case.name}_U'] = u
            metrics.append({'nu': nu, 'case': case.name, 'model': 'forward',
                            'role': 'fit' if case.name in cfg.FIT_CASES else 'prediction', **_metrics(state)})
        save_run(output/f'nu_{nu:.2f}', cfg, dict(summary, nu=nu, status='computed'), arrays)
    pd.DataFrame(metrics).to_csv(output/'comparison.csv', index=False)
    summary['status'] = 'computed'
    _update_summary(output, summary)
    return summary


def inverse_dic(prepared, output):
    """Stage 3: fit DIC, select gamma within training, predict held-out loads."""
    if Path(output).exists():
        raise FileExistsError('Choose a new inverse output')
    cfg, mesh, cases, metadata, provenance = load_prepared(prepared)
    calibration = pd.DataFrame(metadata['calibration'])
    summary = _stage_summary(metadata, provenance, 'dic_case_inverse')
    save_run(output, cfg, summary)
    output = Path(output)
    fit = [c for c in cases if c.name in cfg.FIT_CASES]
    training, validation = [], []
    split_arrays = {}
    for case in fit:
        xy = np.column_stack([case.observation.x_meas, case.observation.y_meas])
        train, valid = block_split(xy, cfg.VALIDATION_BLOCK_MM, cfg.VALIDATION_BUFFER_MM)
        training.append(_subset_case(case, train))
        validation.append(_subset_case(case, valid))
        split_arrays[f'{case.name}_train'] = train
        split_arrays[f'{case.name}_validation'] = valid
    selection = []
    for gamma in cfg.GAMMA_CANDIDATES:
        result = _optimize(cfg, mesh, training, gamma, cfg.PRIMARY_NU)
        scores = []
        for case in validation:
            u, _ = solve_forward(mesh, case.bc_info['tensile_end_force'], cfg.THICKNESS,
                                  cfg.PRIMARY_NU, result['E_final'])
            state = evaluate_observation(case, u)
            scores.append((2*state['cost_tar'], case.observation.weight_vector.sum()))
        score = np.sqrt(sum(s[0] for s in scores)/sum(s[1] for s in scores))
        record = {'gamma': gamma, 'nu': cfg.PRIMARY_NU, 'converged': result['converged'],
                  'validation_rms_mm': score, 'iterations': result['n_iterations']}
        selection.append(record)
        save_run(output / f'selection_gamma_{gamma:g}', cfg,
                 {'solver': _solver_summary(result), 'selection': record, 'uses_prediction_cases': False},
                 dict(split_arrays, E_inverse=result['E_final'], cost_history=result['cost_history']))
        pd.DataFrame(selection).to_csv(output / 'selection.csv', index=False)
    eligible = [r for r in selection if r['converged']]
    if not eligible:
        summary['status'] = 'selection_not_converged'
        _update_summary(output, summary)
        return summary
    gamma = min(eligible, key=lambda r: r['validation_rms_mm'])['gamma']
    summary['selected_gamma'] = gamma
    summary['gamma_at_scan_boundary'] = gamma in (min(cfg.GAMMA_CANDIDATES), max(cfg.GAMMA_CANDIDATES))
    all_metrics, all_regions, solvers = [], [], []
    for nu in cfg.NU_VALUES:
        destination = output / f'nu_{nu:.2f}'
        result = _optimize(cfg, mesh, fit, gamma, nu)
        fields = {'coordinates': mesh.coord, 'E_inverse': result['E_final'],
                  'cost_history': result['cost_history'], 'gradient_norms': result['gradient_norms']}
        for case in cases:
            u, _ = solve_forward(mesh, case.bc_info['tensile_end_force'], cfg.THICKNESS, nu, result['E_final'])
            state = evaluate_observation(case, u)
            fields[f'{case.name}_prediction'] = state['prediction']
            fields[f'{case.name}_residual'] = state['residual']
            fields[f'{case.name}_U'] = u
            all_metrics.append({'nu': nu, 'case': case.name, 'model': 'inverse',
                                'role': 'fit' if case.name in cfg.FIT_CASES else 'prediction',
                                'converged': result['converged'], **_metrics(state)})
            fields[f'{case.name}_coordinates'] = np.column_stack([case.observation.x_meas, case.observation.y_meas])
            fields[f'{case.name}_observed'] = case.observation.u_measured
        regions = _regional_comparison(mesh, result['E_final'], calibration, cfg.REGION_BOUNDARIES_X)
        regions.insert(0, 'nu', nu)
        all_regions.extend(regions.to_dict(orient='records'))
        solvers.append({'nu': nu, **_solver_summary(result)})
        run_config = SimpleNamespace(**{k: v for k, v in vars(cfg).items() if k.isupper()},
                                     __file__=getattr(cfg, '__file__', None))
        run_config.NU, run_config.GAMMA = nu, gamma
        save_run(destination, run_config, dict(summary, solver=_solver_summary(result), nu=nu,
                                               inverse_parameter_count=mesh.n_nod), fields)
        regions.to_csv(destination / 'regional_modulus_comparison.csv', index=False)
        pd.DataFrame(all_metrics).to_csv(output / 'comparison.csv', index=False)
        pd.DataFrame(all_regions).to_csv(output / 'regional_comparison.csv', index=False)
    summary['solvers'] = solvers
    summary['status'] = ('computed' if all(r['converged'] for r in solvers) else 'inverse_not_converged')
    _update_summary(output, summary)
    return summary


def compare_dic_results(prepared, forward, inverse, output, plots=True):
    """Stage 4: plot and compare saved results; no FE solve or optimization."""
    if Path(output).exists():
        raise FileExistsError('Choose a new comparison output')
    cfg, mesh, cases, metadata, provenance = load_prepared(prepared)
    forward, inverse, output = Path(forward), Path(inverse), Path(output)
    sources = []
    for directory, workflow in ((forward, 'dic_calibrated_forward'), (inverse, 'dic_case_inverse')):
        record = json.loads((directory/'run_metadata.json').read_text(encoding='utf-8'))
        if record['workflow'] != workflow or any(record[key] != value for key, value in provenance.items()):
            raise ValueError('Forward/inverse results must refer to the SAME frozen preparation')
        if record['status'] not in ('computed', 'inverse_not_converged'):
            raise ValueError(f'Incomplete upstream stage: {directory}')
        sources.append(record)
    summary = _stage_summary(metadata, provenance, 'dic_case_comparison')
    summary.update(status=sources[1]['status'], selected_gamma=sources[1]['selected_gamma'],
                   gamma_at_scan_boundary=sources[1]['gamma_at_scan_boundary'], solvers=sources[1]['solvers'],
                   upstream_sources={str(p): file_sha256(p/'run_metadata.json') for p in (forward, inverse)})
    metrics = pd.concat([pd.read_csv(forward/'comparison.csv'), pd.read_csv(inverse/'comparison.csv')], ignore_index=True)
    regions = pd.read_csv(inverse/'regional_comparison.csv')
    selection = pd.read_csv(inverse/'selection.csv')
    save_run(output, cfg, summary)
    reference = partition_modulus(pd.DataFrame(metadata['calibration']).mean_E_MPa, cfg.REGION_BOUNDARIES_X)
    for nu in cfg.NU_VALUES:
        destination = output/f'nu_{nu:.2f}'
        with np.load(forward/f'nu_{nu:.2f}'/'fields.npz') as base, np.load(inverse/f'nu_{nu:.2f}'/'fields.npz') as reconstructed:
            np.testing.assert_allclose(base['coordinates'], reconstructed['coordinates'], atol=0, rtol=0)
            states = [[], []]
            for case in cases:
                for values, collection, label in ((base, states[0], 'forward'), (reconstructed, states[1], 'inverse')):
                    prediction = values[f'{case.name}_prediction'].copy()
                    residual = case.observation.u_measured-prediction
                    rms = np.sqrt(np.sum(case.observation.weight_vector*residual**2)/case.observation.weight_vector.sum())
                    recorded = metrics[(metrics.nu == nu) & (metrics['case'] == case.name) & (metrics.model == label)]
                    if len(recorded) != 1:
                        raise ValueError('Missing/duplicate upstream displacement metric')
                    np.testing.assert_allclose(rms, recorded.rms_mm.iloc[0], atol=1e-12)
                    collection.append({'name': case.name, 'prediction': prediction, 'weighted_rms': rms})
            field = reconstructed['E_inverse'].copy()
        save_run(destination, cfg, dict(summary, nu=nu), {'coordinates': mesh.coord, 'E_inverse': field})
        if plots:
            save_conditional_comparison(destination, cases, states[0], states[1], cfg.FIT_CASES)
            plot_modulus_comparison(destination, mesh, reference(mesh.X, mesh.Y), field)
            save_x_profile(destination, mesh, field, reference)
    metrics.to_csv(output/'comparison.csv', index=False)
    regions.to_csv(output/'regional_comparison.csv', index=False)
    selection.to_csv(output/'selection.csv', index=False)
    pd.DataFrame(summary['coverage']).to_csv(output/'input_audit.csv', index=False)
    from .dic_case_reporting import write_case_report
    write_case_report(output, summary, metrics.to_dict(orient='records'), regions.to_dict(orient='records'), selection.to_dict(orient='records'))
    return summary


def _update_summary(output, summary):
    from .research_reporting import _json_value

    path = output / 'run_metadata.json'
    metadata = json.loads(path.read_text(encoding='utf-8'))
    metadata.update(_json_value(summary))
    path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
