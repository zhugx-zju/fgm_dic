"""Independent preparation, forward, inverse and comparison workflow stages.

The computation ROI may follow DIC coverage; the actual load-section width
is separate. Both forward and inverse retain the same raw point observations.
"""

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy.sparse import diags

from fgm_asm import MeshInfo, lbfgs_inverse_solver_scipy
from fgm_asm.load_case import evaluate_observation
from fgm_asm.l_curve import select_l_curve_corner
from fgm_asm.exact_tv import ExactTVOptions, exact_tv_inverse_solver
from fgm_asm.research import make_observation_case, partition_modulus, solve_forward
from .dic_preprocessing import (grid_fields_from_points, plot_displacement_clouds,
                                plot_grid_fields, save_dic_comparison)
from .experiment_data import (check_platform_reference, dic_roi_mask, file_sha256,
                              load_calibration, load_dic_platform)
from .research_reporting import plot_l_curve, plot_modulus_comparison, save_run, save_x_profile


# Source files affecting equilibrium, observations and inversion. Package
# exports and old plotting/file wrappers do not affect numerical compatibility.
MECHANICS_SOURCE_FILES = (
    'mesh.py', 'material.py', 'grf.py', 'fem_forward.py', 'observation.py',
    'load_case.py', 'regularization.py', 'inverse_solver.py', 'tgv.py', 'research.py', 'l_curve.py',
)


def _solver_summary(result):
    summary = {key: result[key] for key in ('converged', 'message', 'n_iterations',
                                          'n_evaluations', 'final_cost', 'regularization_mode',
                                          'residual_norm', 'regularization_norm')}
    summary['final_gradient_inf'] = float(np.max(np.abs(result['final_gradient'])))
    summary['line_search_restarts'] = result.get('line_search_restarts', [])
    for key in ('tv_certificate', 'solution_method', 'regularization_resolved'):
        if key in result:
            summary[key] = result[key]
    return summary


def _regularization_cost(cfg, gamma, eta):
    return gamma*eta if cfg.REGULARIZATION_MODE == 'exact_tv' else .5*gamma*eta**2


def _tv_arrays(result):
    return {key: result[key] for key in ('tv_dual', 'tv_split_gradient') if key in result}


def _save_tv_evidence(destination, result):
    if 'tv_history' in result:
        pd.DataFrame(result['tv_history']).to_csv(Path(destination)/'tv_history.csv', index=False)

def _regional_comparison(mesh, field, calibration, boundaries):
    values = (field[mesh.ele_nods_id - 1] @ mesh.gauss_N.T).T
    x = np.einsum('en,gn->ge', mesh.x_ele, mesh.gauss_N)
    weights = mesh.gauss_abs_det_j * mesh.gauss_w[:, None]
    groups = np.searchsorted(boundaries, x, side='right')
    rows = []
    for index, record in enumerate(calibration.to_dict(orient='records')):
        mask = groups == index
        if not mask.any():
            raise ValueError('Inversion mesh does not resolve every selected calibration region')
        average = float(np.average(values[mask], weights=weights[mask]))
        spread = float(np.sqrt(np.average((values[mask] - average)**2, weights=weights[mask])))
        rows.append(dict(record, inverse_mean_E_MPa=average, inverse_spatial_sd_MPa=spread,
                         difference_percent=100 * (average / record['mean_E_MPa'] - 1)))
    return pd.DataFrame(rows)


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
        if mode in ('explicit_verified', 'specimen_marks', 'material_coordinates'):
            offset = np.asarray(specification['offset_mm'], dtype=float)
            # Approximate end distances need not reproduce the independent
            # DIC millimetre span exactly; midpoint alignment is conditional.
            verified = mode == 'explicit_verified'
        elif mode == 'assumed_center':
            xy = np.asarray(reference_xy, dtype=float)
            offset = np.asarray(specification['cloud_center_fem_mm'])-matrix @ (
                xy.min(axis=0)+xy.max(axis=0))/2
            verified = False
        else:
            raise ValueError('Unknown registration mode; supply explicit physical coordinate information')
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


def displacement_component_weights(cfg):
    """Quadratic coefficients for FEM (axial ux, transverse uy), not amplitudes."""
    weights = np.asarray(getattr(cfg, 'DISPLACEMENT_COMPONENT_WEIGHTS', (1., 1.)), dtype=float)
    if weights.shape != (2,) or not np.all(np.isfinite(weights) & (weights > 0)):
        raise ValueError('DISPLACEMENT_COMPONENT_WEIGHTS must contain two positive finite coefficients (ux, uy)')
    return weights


def _validate(cfg):
    displacement_component_weights(cfg)
    if cfg.DIC_UNIT != 'mm':
        raise ValueError('Convert positions AND displacements to mm before this pipeline')
    if (cfg.DISPLACEMENT_REFERENCE != 'initial_unloaded_image' or
            cfg.FORCE_REFERENCE != 'corrected_force_from_unloaded_state'):
        raise ValueError('These steps require unloaded-reference displacement and matched actual force')
    for key in ('GEO_L', 'GEO_H', 'THICKNESS'):
        if not np.isfinite(getattr(cfg, key)) or getattr(cfg, key) <= 0:
            raise ValueError(f'{key} must be positive and finite')
    width = getattr(cfg, 'LOAD_SECTION_WIDTH', cfg.GEO_H)
    if not np.isfinite(width) or width < cfg.GEO_H-1e-7:
        raise ValueError('LOAD_SECTION_WIDTH must be finite and cover the model width')
    if width > cfg.GEO_H+1e-7 and getattr(cfg, 'INTERNAL_SIDE_BOUNDARY', None) != 'traction_free_approximation':
        raise ValueError('An internal-width ROI requires explicit INTERNAL_SIDE_BOUNDARY=traction_free_approximation; exact cut tractions are not implemented')
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
    if (len(cfg.GAMMA_CANDIDATES) < 5 or len(set(cfg.GAMMA_CANDIDATES)) != len(cfg.GAMMA_CANDIDATES) or
            any(g <= 0 or not np.isfinite(g) for g in cfg.GAMMA_CANDIDATES)):
        raise ValueError('L-curve requires at least five distinct positive finite gamma candidates')
    if cfg.REGULARIZATION_MODE not in ('gradient', 'huber_tv', 'exact_tv'):
        raise ValueError('Choose gradient, huber_tv or exact_tv on the free nodal E field')
    if cfg.REGULARIZATION_MODE == 'exact_tv':
        ExactTVOptions(**cfg.EXACT_TV_OPTIONS, max_evaluations=cfg.MAX_FUN)
        if getattr(cfg, 'EXACT_TV_SOLVER', 'admm') not in ('admm', 'primal_dual'):
            raise ValueError('EXACT_TV_SOLVER must be admm or primal_dual')
    if cfg.REGULARIZATION_MODE == 'huber_tv' and (
            not np.isfinite(cfg.TV_DELTA) or cfg.TV_DELTA <= 0):
        raise ValueError('TV_DELTA must be positive and finite (MPa/mm)')
    if any(not np.isfinite(getattr(cfg, key)) or getattr(cfg, key) < 0 for key in ('FTOL', 'GTOL')):
        raise ValueError('FTOL and GTOL must be finite and nonnegative')
    edges = np.r_[0., cfg.REGION_BOUNDARIES_X, cfg.GEO_L]
    if (len(edges) != len(cfg.REGION_INFILL_PERCENT)+1 or
            not np.isfinite(edges).all() or np.any(np.diff(edges) <= 0)):
        raise ValueError('Material regions must partition the physical model length')


def _optimize(cfg, mesh, cases, gamma, nu, initial=None, initial_dual=None):
    if cfg.REGULARIZATION_MODE == 'exact_tv':
        def progress(row):
            interval = 200 if getattr(cfg, 'EXACT_TV_SOLVER', 'admm') == 'primal_dual' else 20
            if row['iteration'] % interval == 0:
                print(f"  TV outer={row['iteration']}, evaluations={row['evaluations']}, "
                      f"stationarity/tol={row['stationarity']/row['stationarity_tol']:.3g}, "
                      f"primal/tol={row['primal_rms']/row['primal_tol']:.3g}, "
                      f"gap/tol={abs(row['fenchel_gap'])/row['gap_tol']:.3g}", flush=True)
        return exact_tv_inverse_solver(mesh, cases, initial, gamma, nu,
            options=ExactTVOptions(**cfg.EXACT_TV_OPTIONS, max_evaluations=cfg.MAX_FUN),
            initial_dual=initial_dual, callback=progress,
            solver=getattr(cfg, 'EXACT_TV_SOLVER', 'admm'),
            primal_step=getattr(cfg, 'EXACT_TV_PRIMAL_STEP', 2e8),
            coupling=getattr(cfg, 'EXACT_TV_COUPLING', .9))
    options = {'delta': cfg.TV_DELTA} if cfg.REGULARIZATION_MODE == 'huber_tv' else None
    result = lbfgs_inverse_solver_scipy(
        mesh, observation_cases=cases, E_init=initial,
        gamma=gamma, max_iter=cfg.MAX_ITER, max_fun=cfg.MAX_FUN,
        ftol=cfg.FTOL, gtol=cfg.GTOL, nu=nu,
        regularization_mode=cfg.REGULARIZATION_MODE, regularization_options=options)
    restarts = []
    # A failed L-BFGS line search can leave an almost stationary iterate.
    # Reset its history, without changing tolerances or the per-point budget.
    while (not result['converged'] and 'ABNORMAL' in result['message'].upper() and
           np.max(np.abs(result['final_gradient'])) <= 10*cfg.GTOL and
           len(restarts) < 2):
        remaining_iterations = cfg.MAX_ITER-result['n_iterations']
        remaining_evaluations = cfg.MAX_FUN-result['n_evaluations']
        if min(remaining_iterations, remaining_evaluations) <= 0:
            break
        previous = result
        retried = lbfgs_inverse_solver_scipy(
            mesh, observation_cases=cases, E_init=previous['E_final'], gamma=gamma,
            max_iter=remaining_iterations, max_fun=remaining_evaluations,
            ftol=cfg.FTOL, gtol=cfg.GTOL, nu=nu, regularization_mode=cfg.REGULARIZATION_MODE,
            regularization_options=options)
        restarts.append({'previous_message': previous['message'],
                         'previous_gradient_inf': float(np.max(np.abs(previous['final_gradient']))),
                         'restart_message': retried['message'],
                         'restart_iterations': retried['n_iterations'],
                         'restart_evaluations': retried['n_evaluations']})
        for key in ('cost_history', 'cost_tar_history', 'cost_reg_history',
                    'gradient_norms', 'grad_norm_history'):
            retried[key] = np.r_[previous[key], retried[key][1:]]
        retried['n_iterations'] += previous['n_iterations']
        retried['n_evaluations'] += previous['n_evaluations']
        result = retried
    result['line_search_restarts'] = restarts
    return result


def homogeneous_dic_initial(mesh, cases, thickness, nu):
    """Fit one uniform starting E from fit observations, with separate rigid modes.

    U(E)=U(1 MPa)/E for uniform material under fixed loads. This determines
    only an initial value; the inverse retains every independent nodal modulus.
    No calibration values, reference-modulus normalization or held-out data enter.
    """
    numerator, denominator = 0., 0.
    if not cases:
        raise ValueError('Homogeneous initialization needs fit observations')
    for case in cases:
        unit_u, _ = solve_forward(mesh, case.bc_info['tensile_end_force'], thickness,
                                  nu, np.ones(mesh.n_nod))
        measured = evaluate_observation(case, np.zeros(mesh.n_dof))['residual']
        predicted = measured-evaluate_observation(case, unit_u)['residual']
        weights = case.observation.weight_vector
        numerator += float(predicted @ (weights*measured))
        denominator += float(predicted @ (weights*predicted))
    if not np.isfinite([numerator, denominator]).all() or min(numerator, denominator) <= 0:
        raise ValueError('Fit DIC does not support a positive finite homogeneous initial modulus')
    initial_E = denominator/numerator
    return np.full(mesh.n_nod, initial_E), {
        'source': 'homogeneous_dic_fit', 'uniform_E_MPa': initial_E,
        'fit_cases': [case.name for case in cases], 'nu': nu,
        'uses_calibration': False, 'uses_prediction_cases': False,
    }


def _metrics(state):
    residual = state['residual'].reshape(-1, 2)
    return {'rms_mm': state['weighted_rms'],
            'rmse_ux_mm': np.sqrt(np.mean(residual[:, 0]**2)),
            'rmse_uy_mm': np.sqrt(np.mean(residual[:, 1]**2)),
            'p95_ux_mm': np.quantile(np.abs(residual[:, 0]), .95),
            'p95_uy_mm': np.quantile(np.abs(residual[:, 1]), .95)}


def dic_mesh_coordinates(reference, matrix, offset):
    """Complete measured lattice in FE order, retaining exported millimetres."""
    xy = reference.coordinates @ np.asarray(matrix).T + np.asarray(offset)
    x, y, _ = grid_fields_from_points(xy, {}, reference.grid_ids)
    if x.size != len(xy):
        raise ValueError('DIC mesh requires a complete rectangular point grid; no missing nodes are filled')
    iy, ix = np.argsort(y.mean(axis=1)), np.argsort(x.mean(axis=0))
    x, y = x[np.ix_(iy, ix)], y[np.ix_(iy, ix)]
    if np.any(np.diff(x, axis=1) <= 0) or np.any(np.diff(y, axis=0) <= 0):
        raise ValueError('DIC mesh axes must be strictly ordered')
    coordinates = np.column_stack([x.ravel(), y.ravel()])
    trial = MeshInfo(np.ptp(x), np.ptp(y), x.shape[1]-1, x.shape[0]-1,
                     nodal_coordinates=coordinates)
    return trial.coord, (trial.nel_x, trial.nel_y)


def extract_material_window(platforms, cfg):
    """Clip one continuous coordinate window without resampling DIC points.

    The material map is inherited from the confirmed shared reference image.
    Available whole axial rows are retained; missing end coverage is recorded,
    never filled or reassigned to a different material partition.
    """
    if not hasattr(cfg, 'MATERIAL_WINDOW_MM'):
        return platforms
    spec = cfg.REGISTRATION
    if spec['mode'] != 'material_coordinates' or not spec.get('shared_reference_confirmed'):
        raise ValueError('Material-window extraction requires a confirmed shared reference and material map')
    lower, upper = np.asarray(cfg.MATERIAL_WINDOW_MM, dtype=float)
    if not np.isfinite([lower, upper]).all() or lower >= upper:
        raise ValueError('MATERIAL_WINDOW_MM must contain two increasing finite coordinates')
    matrix = np.asarray(spec['matrix'], dtype=float)
    sign = spec['material_x_sign']
    if sign not in (-1, 1):
        raise ValueError('material_x_sign must be +1 or -1')
    selected = {}
    for name, platform in platforms.items():
        axial = (platform.coordinates @ matrix.T)[:, 0]
        material = spec['manufacturing_anchor_mm'] + sign*(axial-spec['raw_axial_anchor_mm'])
        inside = (material >= lower-1e-7) & (material <= upper+1e-7)
        if inside.sum() < 4:
            raise ValueError(f'{name}: requested material window has insufficient DIC coverage')
        ids = platform.grid_ids[inside]
        if len(ids) != np.prod(np.ptp(ids, axis=0)+1):
            raise ValueError(f'{name}: coordinate crop must retain a complete rectangular DIC grid')
        bounds = [float(material[inside].min()), float(material[inside].max())]
        extraction = {
            'method': 'continuous_material_coordinate_window_no_resampling',
            'requested_manufacturing_bounds_mm': [float(lower), float(upper)],
            'available_manufacturing_bounds_mm': [float(material.min()), float(material.max())],
            'selected_manufacturing_bounds_mm': bounds,
            'unavailable_requested_end_mm': [max(0., float(material.min()-lower)),
                                             max(0., float(upper-material.max()))],
            'boundary_sampling_remainder_mm': [bounds[0]-max(lower, float(material.min())),
                                               min(upper, float(material.max()))-bounds[1]],
            'input_points': len(axial), 'selected_points': int(inside.sum()),
            'selected_grid_shape': (np.ptp(ids, axis=0)+1).tolist(),
        }
        selected[name] = replace(platform, coordinates=platform.coordinates[inside],
                                 displacement=platform.displacement[inside],
                                 temporal_std=platform.temporal_std[inside], grid_ids=ids,
                                 metadata=dict(platform.metadata, extraction=extraction))
    return selected


def resolve_dic_geometry(cfg, reference):
    """Resolve ROI from fit data only, without scaling coordinates or material zones.

    Material coordinates come from measured marks or a legacy center assumption.
    Extents are computed after the fixed direction transform, before translation.
    The caller's module is not mutated; resolved values enter the frozen config.
    """
    resolved = SimpleNamespace(**vars(cfg))
    source = getattr(cfg, 'DOMAIN_SOURCE', 'configured_geometry')
    if source not in ('configured_geometry', 'dic_reference_bounds'):
        raise ValueError('DOMAIN_SOURCE must be configured_geometry or dic_reference_bounds')
    if source == 'dic_reference_bounds':
        mode = cfg.REGISTRATION['mode']
        if mode not in ('assumed_center', 'specimen_marks', 'material_coordinates'):
            raise ValueError('DIC-bound ROI requires explicit material positioning or an assumed center')
        matrix = np.asarray(cfg.REGISTRATION['matrix'], dtype=float)
        if (matrix.shape != (2, 2) or not np.isfinite(matrix).all() or
                not np.allclose(matrix @ matrix.T, np.eye(2), atol=1e-10, rtol=1e-9)):
            raise ValueError('ROI direction transform must preserve lengths')
        xy = reference.coordinates @ matrix.T
        span = np.ptp(xy, axis=0)
        if not np.isfinite(span).all() or np.any(span <= 0):
            raise ValueError('DIC reference must define a finite 2D ROI')
        resolved.GEO_L, resolved.GEO_H = map(float, span)
        if mode == 'assumed_center':
            resolved.REGISTRATION = dict(cfg.REGISTRATION, cloud_center_fem_mm=(span/2).tolist())
            resolved.REGION_BOUNDARIES_X = (np.asarray(cfg.REGION_BOUNDARIES_CENTERED_X, dtype=float)+span[0]/2).tolist()
        elif mode == 'material_coordinates':
            spec = cfg.REGISTRATION
            sign = spec['material_x_sign']
            if sign not in (-1, 1) or not spec.get('shared_reference_confirmed'):
                raise ValueError('Explicit material coordinates require sign and shared-reference confirmation')
            material_offset = float(spec['manufacturing_anchor_mm'] +
                                    sign*(xy[:, 0].min()-spec['raw_axial_anchor_mm']))
            boundaries = np.asarray(cfg.REGION_BOUNDARIES_MANUFACTURING_X, dtype=float)
            if (not np.isfinite(boundaries).all() or np.any(np.diff(boundaries) <= 0) or
                    np.any(sign*np.diff(cfg.REGION_INFILL_PERCENT) <= 0)):
                raise ValueError('Material interfaces and group order must follow the configured direction')
            resolved.MANUFACTURING_X_SIGN = sign
            resolved.MANUFACTURING_X_OFFSET_MM = material_offset
            resolved.REGION_BOUNDARIES_X = np.sort((boundaries-material_offset)/sign).tolist()
            resolved.REGISTRATION = dict(spec, offset_mm=(-xy.min(axis=0)).tolist())
            resolved.MATERIAL_REGISTRATION = {
                'method': 'shared_reference_material_coordinates_unit_scale',
                'manufacturing_x_offset_mm': material_offset,
                'manufacturing_x_fem_coefficient': sign,
                'manufacturing_x_relation': f'manufacturing_x = {sign} * FEM_x + {material_offset:.12g} mm',
                'infill_order_along_fem_x': list(cfg.REGION_INFILL_PERCENT),
                'manufacturing_roi_x_mm': [material_offset, float(material_offset+sign*span[0])],
                'manufacturing_mark_x_mm': [-spec['specimen_length_mm']/2+spec['low_infill_mark_end_distance_mm'],
                                           spec['specimen_length_mm']/2-spec['high_infill_mark_end_distance_mm']],
                'raw_axial_anchor_mm': spec['raw_axial_anchor_mm'],
                'manufacturing_anchor_mm': spec['manufacturing_anchor_mm'],
                'anchor_prepared_metadata_sha256': spec['anchor_prepared_metadata_sha256'],
                'length_scale_factor': 1., 'direction_note': spec['note'],
                'transverse_location': 'internal DIC ROI bounds; no measured specimen edge registration',
            }
        else:
            spec = cfg.REGISTRATION
            if spec.get('mark_reference') != 'roi_axial_extrema':
                raise ValueError('specimen_marks requires the confirmed region ends at the two marks')
            if spec.get('length_scale_source') != 'independent_dic_calibration':
                raise ValueError('Resolve DIC length calibration first; mark alignment never rescales data')
            lengths = np.asarray([spec['specimen_length_mm'], spec['low_infill_mark_end_distance_mm'],
                                  spec['high_infill_mark_end_distance_mm']], dtype=float)
            if not np.isfinite(lengths).all() or np.any(lengths <= 0) or lengths[1:].sum() >= lengths[0]:
                raise ValueError('Mark end distances must be positive and leave a positive gap')
            low_mark = -lengths[0]/2+lengths[1]
            high_mark = lengths[0]/2-lengths[2]
            mark_center = (low_mark+high_mark)/2
            material_sign = spec.get('material_x_sign', 1)
            if material_sign not in (-1, 1):
                raise ValueError('material_x_sign must be +1 or -1')
            material_offset = float(mark_center-material_sign*span[0]/2)
            boundaries = np.asarray(cfg.REGION_BOUNDARIES_MANUFACTURING_X, dtype=float)
            if (not np.isfinite(boundaries).all() or np.any(np.diff(boundaries) <= 0) or
                    np.any(material_sign*np.diff(cfg.REGION_INFILL_PERCENT) <= 0)):
                raise ValueError('Manufacturing interfaces must increase; infill order must match material_x_sign')
            resolved.MANUFACTURING_X_SIGN = material_sign
            resolved.MANUFACTURING_X_OFFSET_MM = material_offset
            resolved.REGION_BOUNDARIES_X = np.sort((boundaries-material_offset)/material_sign).tolist()
            resolved.REGISTRATION = dict(spec, offset_mm=(-xy.min(axis=0)).tolist())
            mapped_bounds = [material_offset, float(material_offset+material_sign*span[0])]
            mark_order = [float(low_mark), float(high_mark)] if material_sign == 1 else [float(high_mark), float(low_mark)]
            resolved.MATERIAL_REGISTRATION = {
                'method': 'confirmed_marks_midpoint_unit_scale',
                'manufacturing_x_offset_mm': material_offset,
                'manufacturing_x_fem_coefficient': material_sign,
                'manufacturing_x_relation': f'manufacturing_x = {material_sign} * FEM_x + {material_offset:.12g} mm',
                'infill_order_along_fem_x': list(cfg.REGION_INFILL_PERCENT),
                'manufacturing_mark_x_mm': mark_order,
                'manufacturing_roi_x_mm': mapped_bounds,
                'nominal_mark_gap_mm': float(high_mark-low_mark),
                'dic_span_mm': float(span[0]),
                'dic_minus_nominal_mark_gap_mm': float(span[0]-(high_mark-low_mark)),
                'endpoint_discrepancy_mm': [value-mark for value, mark in zip(mapped_bounds, mark_order)],
                'length_scale_factor': 1.,
                'direction_note': spec['note'],
                'transverse_location': 'internal DIC ROI bounds; no measured specimen edge registration',
            }
    mesh_source = getattr(cfg, 'MESH_SOURCE', 'configured_grid')
    if mesh_source == 'dic_grid':
        matrix, offset, _ = registration_transform(reference.coordinates, resolved.REGISTRATION)
        if matrix is None:
            raise ValueError('DIC mesh requires an explicit coordinate transform')
        _, counts = dic_mesh_coordinates(reference, matrix, offset)
        resolved.NEL_X, resolved.NEL_Y = counts
    elif mesh_source != 'configured_grid':
        raise ValueError('MESH_SOURCE must be configured_grid or dic_grid')
    return resolved


def prepare_dic(cfg, output):
    """Stage 1: freeze experimental inputs, or explicitly generated FE data."""
    if hasattr(cfg, 'SYNTHETIC_SOURCE_PREPARED'):
        from .synthetic_recovery import prepare_synthetic
        return prepare_synthetic(cfg, output)
    output = Path(output)
    if output.exists():
        raise FileExistsError(f'Choose a new preparation output: {output}')
    from .config_utils import configuration_fingerprint
    requested_fingerprint = configuration_fingerprint(cfg)
    data_mode = getattr(cfg, 'DATA_MODE', 'dic')
    if data_mode not in ('dic', 'synthetic_layered', 'synthetic_nodal'):
        raise ValueError('DATA_MODE must be dic, synthetic_layered or synthetic_nodal')
    platforms, records = {}, {}
    for name, item in cfg.PLATFORMS.items():
        platforms[name] = load_dic_platform(Path(item['folder']), *item['stages'],
                                            coordinate_state=cfg.DIC_COORDINATE_STATE)
        records[name] = _force_record(cfg, name, item)
    platforms = extract_material_window(platforms, cfg)
    shared = check_platform_reference(platforms)
    reference = platforms[cfg.FIT_CASES[0]]
    cfg = resolve_dic_geometry(cfg, reference)
    _validate(cfg)
    calibration = load_calibration(Path(cfg.CALIBRATION_CSV), cfg.CALIBRATION_METHOD, cfg.REGION_INFILL_PERCENT)
    section_width = getattr(cfg, 'LOAD_SECTION_WIDTH', cfg.GEO_H)
    for record in records.values():
        # The existing FE kernel expects a force over its own loaded edge.
        # F_roi = F_measured * w_roi / w_specimen preserves F/(w_specimen*t).
        record['model_end_force_N'] = record['mean_corrected_force_N']*cfg.GEO_H/section_width
        record['uniform_traction_MPa'] = record['mean_corrected_force_N']/(section_width*cfg.THICKNESS)
        record['load_section_width_mm'] = section_width
        record['force_reference_note'] = getattr(cfg, 'FORCE_REFERENCE_NOTE', '')
    matrix, offset, verified = registration_transform(reference.coordinates, cfg.REGISTRATION)
    summary = {'workflow': 'dic_case_preparation', 'status': 'prepared', 'case_id': cfg.CASE_ID,
               'requested_configuration_sha256': requested_fingerprint,
               'coordinates_verified': verified, 'registration': cfg.REGISTRATION,
               'matrix': matrix, 'offset_mm': offset,
               'physical_model_mm': [cfg.GEO_L, cfg.GEO_H, cfg.THICKNESS],
               'domain_geometry': {'source': getattr(cfg, 'DOMAIN_SOURCE', 'configured_geometry'),
                                   'source_platform': cfg.FIT_CASES[0],
                                   'load_section_width_mm': section_width,
                                   'lateral_boundary': getattr(cfg, 'INTERNAL_SIDE_BOUNDARY', 'physical_free_edges'),
                                   'material_position': (cfg.MATERIAL_REGISTRATION['method']
                                                         if hasattr(cfg, 'MATERIAL_REGISTRATION') else
                                                         'centered assumption' if not verified else 'registered')},
               'shared_mesh': {'nel_x': cfg.NEL_X, 'nel_y': cfg.NEL_Y,
                               'hx_mm': cfg.GEO_L/cfg.NEL_X, 'hy_mm': cfg.GEO_H/cfg.NEL_Y,
                                'source': getattr(cfg, 'MESH_SOURCE', 'configured_grid'),
                                'nodes': (cfg.NEL_X+1)*(cfg.NEL_Y+1),
                               'same_for_forward_and_inverse': True},
               'force_pairing': records, 'shared_reference': shared,
               'platforms': {name: p.metadata for name, p in platforms.items()},
               'calibration': calibration.to_dict(orient='records'),
               'calibration_source_sha256': file_sha256(Path(cfg.CALIBRATION_CSV)),
               'rigid_policy': 'FE has three gauge constraints; observed residual fits the same two translations and infinitesimal rotation',
               'primary_nu_prespecified': cfg.PRIMARY_NU,
               'objective_definition': '0.5*r.T@W_area@r + 0.5*gamma*integral(|grad(E)|^2); no modulus or noise normalization',
               'physical_validation_status': 'not_established; geometry registration and boundary-equivalence evidence are distinct',
               'scope': 'registered comparison' if verified else 'conditional comparison under explicit registration assumptions'}
    component_weights = displacement_component_weights(cfg)
    summary['displacement_component_weights'] = dict(zip(('ux', 'uy'), component_weights.tolist()))
    if not np.array_equal(component_weights, [1., 1.]):
        summary['objective_definition'] = (
            f'0.5*sum(area_i*({component_weights[0]:g}*r_ux_i^2 + '
            f'{component_weights[1]:g}*r_uy_i^2)) + 0.5*gamma*integral(|grad(E)|^2); '
            'quadratic component coefficients; no modulus or noise normalization')
    if cfg.REGULARIZATION_MODE == 'huber_tv':
        summary['objective_definition'] = (
            f'0.5*sum(area_i*({component_weights[0]:g}*r_ux_i^2 + '
            f'{component_weights[1]:g}*r_uy_i^2)) + 0.5*gamma*integral(rho_delta(|grad(E)|)); '
            f'Huber delta={cfg.TV_DELTA:g} MPa/mm; no modulus or noise normalization')
    if cfg.REGULARIZATION_MODE == 'exact_tv':
        summary['objective_definition'] = (
            f'0.5*sum(area_i*({component_weights[0]:g}*r_ux_i^2 + '
            f'{component_weights[1]:g}*r_uy_i^2)) + gamma*integral(sqrt(E_x^2+E_y^2)); '
            'unsmoothed isotropic TV, 2x2 Gauss quadrature, free continuous Q4 nodal E; '
            'no modulus or noise normalization; displacement trial/test functions unchanged')
    if hasattr(cfg, 'MATERIAL_REGISTRATION'):
        summary['material_registration'] = cfg.MATERIAL_REGISTRATION
    if hasattr(cfg, 'FORCE_PAIRING_CSV') and any('force_N' not in i for i in cfg.PLATFORMS.values()):
        summary['force_source_sha256'] = file_sha256(Path(cfg.FORCE_PAIRING_CSV))
    arrays, rows = {}, []
    if getattr(cfg, 'MESH_SOURCE', 'configured_grid') == 'dic_grid':
        arrays['dic_mesh_coordinates'], _ = dic_mesh_coordinates(reference, matrix, offset)
    for name, platform in platforms.items():
        spacing = measured_grid_spacing(platform, matrix if matrix is not None else np.eye(2))
        row = {'case': name, 'dic_row_step_mm': spacing[0], 'dic_col_step_mm': spacing[1],
               'points': len(platform.coordinates), 'frame_count': platform.metadata['frame_count']}
        arrays[f'{name}_raw_reference_coordinates'] = platform.coordinates
        arrays[f'{name}_raw_mean_displacement'] = platform.displacement
        arrays[f'{name}_raw_grid_ids'] = platform.grid_ids
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
            arrays[f'{name}_grid_ids'] = platform.grid_ids[inside]
        rows.append(row)
    summary['coverage'] = rows
    if matrix is None:
        summary['status'] = 'needs_registration'
    truth = None
    if data_mode != 'dic':
        if matrix is None:
            raise ValueError('Synthetic observations require resolved physical coordinates')
        from .synthetic_recovery import generate_synthetic_fields
        mesh = MeshInfo(cfg.GEO_L, cfg.GEO_H, cfg.NEL_X, cfg.NEL_Y,
                        nodal_coordinates=arrays.get('dic_mesh_coordinates'))
        # Only H and the force are used to generate observations. In particular,
        # experimental displacement values never enter the synthetic field.
        cases = [make_observation_case(
            mesh, name, records[name]['model_end_force_N'], cfg.THICKNESS,
            arrays[f'{name}_coordinates'], np.zeros_like(arrays[f'{name}_coordinates']),
            point_weights=arrays[f'{name}_point_area_mm2']) for name in cfg.PLATFORMS]
        arrays, truth = generate_synthetic_fields(cfg, mesh, cases, summary, arrays,
            {'geometry_source': 'DIC reference coordinates and force from the unified configuration'})
    save_run(output, cfg, summary, arrays)
    if truth is not None:
        save_run(output/'truth', cfg, {'workflow': 'synthetic_truth',
            'synthetic_observations': summary['synthetic_observations']}, truth)
    pd.DataFrame(rows).to_csv(output / 'input_audit.csv', index=False)
    plot_saved_stage(output)
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
    recorded = {path.replace('\\', '/'): digest for path, digest in metadata['source_sha256'].items()}
    mechanics = MECHANICS_SOURCE_FILES + (('exact_tv.py',) if
        metadata['configuration']['REGULARIZATION_MODE'] == 'exact_tv' else ())
    for name in mechanics:
        path = f'fgm_asm/{name}'
        if path not in recorded or file_sha256(root / path) != recorded[path]:
            raise ValueError('FE kernel changed since preparation; prepare a new case')
    cfg = SimpleNamespace(**metadata['configuration'], __file__=str(source/'configuration_snapshot.py')
                          if (source/'configuration_snapshot.py').exists() else None)
    arrays, cases = {}, []
    with np.load(source/'fields.npz') as fields:
        arrays = {key: fields[key].copy() for key in fields.files}
    mesh = MeshInfo(cfg.GEO_L, cfg.GEO_H, cfg.NEL_X, cfg.NEL_Y,
                    nodal_coordinates=arrays.get('dic_mesh_coordinates'))
    component_weights = displacement_component_weights(cfg)
    for name in cfg.PLATFORMS:
        force_record = metadata['force_pairing'][name]
        measured_force = force_record['mean_corrected_force_N']
        model_force = force_record.get('model_end_force_N', measured_force)
        case = make_observation_case(
            mesh, name, model_force, cfg.THICKNESS,
            arrays[f'{name}_coordinates'], arrays[f'{name}_observed'],
            point_weights=arrays[f'{name}_point_area_mm2'],
            metadata={'coordinates_verified': metadata['coordinates_verified'],
                      'registration': cfg.REGISTRATION, 'displacement_reference': cfg.DISPLACEMENT_REFERENCE,
                      'measured_force_N': measured_force, 'model_end_force_N': model_force,
                      'synthetic_observations': metadata.get('synthetic_observations'),
                      'displacement_component_weights': component_weights.tolist()})
        if not np.array_equal(component_weights, [1., 1.]):
            # One W drives the rigid fit, objective, adjoint, initialization and
            # L-curve. Observed values, H and physical point areas stay unchanged.
            weights = case.observation.weight_vector * np.tile(component_weights, len(case.observation.x_meas))
            observation = replace(case.observation, weight_vector=weights, W=diags(weights, format='csr'))
            case = replace(case, observation=observation)
        cases.append(case)
    provenance = {'prepared_metadata_sha256': file_sha256(source/'run_metadata.json'),
                  'prepared_fields_sha256': file_sha256(source/'fields.npz')}
    return cfg, mesh, cases, metadata, provenance


def _stage_summary(prepared, provenance, stage):
    summary = {key: prepared[key] for key in ('case_id', 'coordinates_verified', 'registration',
                                              'matrix', 'offset_mm', 'physical_model_mm', 'shared_mesh',
                                              'coverage', 'force_pairing', 'scope', 'primary_nu_prespecified',
                                              'objective_definition')}
    summary.update(provenance, workflow=stage, status='running')
    if 'domain_geometry' in prepared:
        summary['domain_geometry'] = prepared['domain_geometry']
    if 'material_registration' in prepared:
        summary['material_registration'] = prepared['material_registration']
    if 'displacement_component_weights' in prepared:
        summary['displacement_component_weights'] = prepared['displacement_component_weights']
    if 'synthetic_observations' in prepared:
        summary['synthetic_observations'] = prepared['synthetic_observations']
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
    plot_saved_stage(output, prepared)
    return summary


def resume_l_curve(prepared, output):
    """Retry abnormal line searches using only each point's remaining budget.

    Original trial arrays, metadata and the pre-recovery table are retained.
    Iteration-limit failures have no remaining budget and are not retried.
    """
    output = Path(output)
    cfg, mesh, cases, metadata, provenance = load_prepared(prepared)
    if cfg.REGULARIZATION_MODE == 'exact_tv':
        raise ValueError('Exact-TV scan trials retain KKT evidence; L-BFGS line-search recovery does not apply')
    summary = json.loads((output/'run_metadata.json').read_text(encoding='utf-8'))
    if (summary['workflow'] != 'dic_case_inverse' or
            not summary['status'].startswith('l_curve_') or
            any(summary[key] != value for key, value in provenance.items()) or
            summary['configuration'] != metadata['configuration']):
        raise ValueError('Resume requires an unresolved scan with the same frozen preparation and settings')
    scan_path = output/'selection.csv'
    archive = output/'selection_before_recovery.csv'
    if archive.exists():
        if archive.read_bytes() != scan_path.read_bytes():
            raise FileExistsError('A recovery has already changed the scan; inspect it before another attempt')
    else:
        archive.write_bytes(scan_path.read_bytes())
    selection = pd.read_csv(scan_path).to_dict(orient='records')
    fit = [c for c in cases if c.name in cfg.FIT_CASES]
    for index, previous in enumerate(selection):
        previous = {key: value for key, value in previous.items()
                    if key not in ('curvature', 'eligible', 'selected', 'solution_directory')}
        if previous['converged'] or 'ABNORMAL' not in previous['stop_message'].upper():
            continue
        remaining_iter = cfg.MAX_ITER-int(previous['iterations'])
        remaining_fun = cfg.MAX_FUN-int(previous['evaluations'])
        if min(remaining_iter, remaining_fun) <= 0:
            continue
        gamma = previous['gamma']
        trial = output/'l_curve'/f'gamma_{gamma:g}'
        attempt = 1
        while (trial/f'recovery_{attempt}').exists():
            attempt += 1
        recovered = trial/f'recovery_{attempt}'
        with np.load(trial/'fields.npz') as saved:
            old_fields = {key: saved[key].copy() for key in saved.files}
        limited = SimpleNamespace(**vars(cfg))
        limited.MAX_ITER, limited.MAX_FUN = remaining_iter, remaining_fun
        print(f'Recover gamma={gamma:.12g}: remaining iterations={remaining_iter}, evaluations={remaining_fun}', flush=True)
        result = _optimize(limited, mesh, fit, gamma, cfg.PRIMARY_NU, old_fields['E_inverse'])
        result['n_iterations'] += int(previous['iterations'])
        result['n_evaluations'] += int(previous['evaluations'])
        for key in ('cost_history', 'cost_tar_history', 'cost_reg_history', 'gradient_norms'):
            result[key] = np.r_[old_fields[key], result[key][1:]]
        result['line_search_restarts'].insert(0, {
            'previous_message': previous['stop_message'], 'previous_iterations': previous['iterations'],
            'previous_evaluations': previous['evaluations'], 'remaining_iterations': remaining_iter,
            'remaining_evaluations': remaining_fun, 'restart_message': result['message']})
        solver = _solver_summary(result)
        record = dict(previous, converged=result['converged'], residual_norm=result['residual_norm'],
                      regularization_norm=result['regularization_norm'],
                      data_cost=.5*result['residual_norm']**2,
                      regularization_cost=.5*gamma*result['regularization_norm']**2,
                      iterations=result['n_iterations'], evaluations=result['n_evaluations'],
                      final_gradient_inf=solver['final_gradient_inf'],
                      gradient_tolerance_satisfied=solver['final_gradient_inf'] <= cfg.GTOL,
                      stop_message=result['message'], solution_directory=str(recovered.relative_to(output)))
        save_run(recovered, cfg, {'solver': solver, 'selection': record,
                 'previous_selection': previous, 'fit_cases': cfg.FIT_CASES, 'uses_prediction_cases': False},
                 {'E_inverse': result['E_final'], 'cost_history': result['cost_history'],
                  'cost_tar_history': result['cost_tar_history'], 'cost_reg_history': result['cost_reg_history'],
                  'gradient_norms': result['gradient_norms'], 'final_gradient': result['final_gradient']})
        selection[index] = record
    summary['scan_recovery'] = {
        'method': 'reset L-BFGS history only for abnormal line-search failures, within original total budget',
        'original_selection': str(archive.resolve()),
        'source_sha256': {str(path.relative_to(Path(__file__).resolve().parents[1])): file_sha256(path)
                          for path in (Path(__file__), Path(__file__).resolve().parents[1]/'script/step3_inverse_dic.py')}}
    summary['status'] = 'running'
    _update_summary(output, summary)
    return _finish_dic_inverse(cfg, mesh, cases, metadata, summary, selection, prepared, output)


def inverse_dic(prepared, output):
    """Stage 3: configured objective, fit-only L-curve, held-out prediction."""
    if Path(output).exists():
        raise FileExistsError('Choose a new inverse output')
    cfg, mesh, cases, metadata, provenance = load_prepared(prepared)
    calibration = pd.DataFrame(metadata['calibration'])
    fit = [c for c in cases if c.name in cfg.FIT_CASES]
    initial, initialization = homogeneous_dic_initial(mesh, fit, cfg.THICKNESS, cfg.PRIMARY_NU)
    summary = _stage_summary(metadata, provenance, 'dic_case_inverse')
    summary.update(gamma_selection_method='l_curve', l_curve_enabled=True, selected_gamma=None,
                   gamma_at_scan_boundary=None,
                   scan_initialization='fit-DIC homogeneous initialization, then decreasing-gamma converged continuation',
                   initialization=initialization,
                   final_initialization='selected L-curve field, identical for all nu')
    save_run(output, cfg, summary, {'scan_initial_E': initial})
    output = Path(output)
    selection = []
    initial_dual = None
    initial_source = 'homogeneous_dic_fit'
    for scan_index, gamma in enumerate(sorted(cfg.GAMMA_CANDIDATES, reverse=True), start=1):
        print(f'L-curve {scan_index}/{len(cfg.GAMMA_CANDIDATES)}: gamma={gamma:.12g}', flush=True)
        result = _optimize(cfg, mesh, fit, gamma, cfg.PRIMARY_NU, initial, initial_dual)
        solver = _solver_summary(result)
        record = {'gamma': gamma, 'nu': cfg.PRIMARY_NU, 'converged': result['converged'],
                  'residual_norm': result['residual_norm'],
                  'regularization_norm': result['regularization_norm'],
                  'data_cost': .5*result['residual_norm']**2,
                  'regularization_cost': _regularization_cost(cfg, gamma, result['regularization_norm']),
                  'regularization_resolved': result.get('regularization_resolved', True),
                  'iterations': result['n_iterations'], 'evaluations': result['n_evaluations'],
                  'final_gradient_inf': solver['final_gradient_inf'],
                  'gradient_tolerance_satisfied': solver['final_gradient_inf'] <= cfg.GTOL,
                  'stop_message': result['message'], 'initial_source': initial_source}
        selection.append(record)
        save_run(output/'l_curve'/f'gamma_{gamma:g}', cfg,
                 {'solver': solver, 'selection': record, 'fit_cases': cfg.FIT_CASES,
                  'uses_prediction_cases': False, 'objective_definition': summary['objective_definition']},
                 {'E_inverse': result['E_final'], 'cost_history': result['cost_history'],
                  'cost_tar_history': result['cost_tar_history'], 'cost_reg_history': result['cost_reg_history'],
                  'gradient_norms': result['gradient_norms'], 'final_gradient': result['final_gradient'],
                  **_tv_arrays(result)})
        _save_tv_evidence(output/'l_curve'/f'gamma_{gamma:g}', result)
        pd.DataFrame(selection).to_csv(output / 'selection.csv', index=False)
        if result['converged']:
            initial, initial_source = result['E_final'].copy(), f'gamma_{gamma:g}'
            initial_dual = result.get('tv_dual') if np.ptp(initial) > 0 else None
        print(f"  stop={result['message']}; evaluations={result['n_evaluations']}; "
              f"rho={result['residual_norm']:.7g}; eta={result['regularization_norm']:.7g}", flush=True)
    return _finish_dic_inverse(cfg, mesh, cases, metadata, summary, selection, prepared, output)


def _finish_dic_inverse(cfg, mesh, cases, metadata, summary, selection, prepared, output):
    calibration = pd.DataFrame(metadata['calibration'])
    fit = [c for c in cases if c.name in cfg.FIT_CASES]
    scan = pd.DataFrame(selection)
    qualified = scan.converged & scan.get('regularization_resolved', True)
    corner = select_l_curve_corner(scan.gamma, scan.residual_norm, scan.regularization_norm, qualified)
    scan['curvature'] = corner['curvature']
    scan['eligible'] = corner['eligible']
    scan['selected'] = False
    if corner['optimal_index'] is not None:
        scan.loc[corner['optimal_index'], 'selected'] = True
    scan.to_csv(output/'selection.csv', index=False)
    summary.update(l_curve_status=corner['status'], l_curve_message=corner['message'])
    if corner['selected_gamma'] is None:
        summary['status'] = f'l_curve_{corner["status"]}'
        _update_summary(output, summary)
        plot_saved_stage(output, prepared)
        return summary
    gamma = corner['selected_gamma']
    summary['selected_gamma'] = gamma
    summary['gamma_at_scan_boundary'] = False
    directory = scan.loc[corner['optimal_index']].get('solution_directory')
    selected_directory = output/directory if isinstance(directory, str) else output/'l_curve'/f'gamma_{gamma:g}'
    with np.load(selected_directory/'fields.npz') as selected:
        initial = selected['E_inverse'].copy()
        initial_dual = selected['tv_dual'].copy() if 'tv_dual' in selected else None
    all_metrics, all_regions, solvers = [], [], []
    for nu in cfg.NU_VALUES:
        destination = output / f'nu_{nu:.2f}'
        result = _optimize(cfg, mesh, fit, gamma, nu, initial.copy(), initial_dual)
        fields = {'coordinates': mesh.coord, 'E_inverse': result['E_final'],
                  'cost_history': result['cost_history'], 'gradient_norms': result['gradient_norms'],
                  'cost_tar_history': result['cost_tar_history'], 'cost_reg_history': result['cost_reg_history'],
                  'final_gradient': result['final_gradient'], **_tv_arrays(result)}
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
        solvers.append({'nu': nu, **_solver_summary(result),
                        'gradient_tolerance_satisfied': np.max(np.abs(result['final_gradient'])) <= cfg.GTOL})
        run_config = SimpleNamespace(**{k: v for k, v in vars(cfg).items() if k.isupper()},
                                     __file__=getattr(cfg, '__file__', None))
        run_config.NU, run_config.GAMMA = nu, gamma
        save_run(destination, run_config, dict(summary, solver=_solver_summary(result), nu=nu,
                                               inverse_parameter_count=mesh.n_nod), fields)
        _save_tv_evidence(destination, result)
        regions.to_csv(destination / 'regional_modulus_comparison.csv', index=False)
        pd.DataFrame(all_metrics).to_csv(output / 'comparison.csv', index=False)
        pd.DataFrame(all_regions).to_csv(output / 'regional_comparison.csv', index=False)
    summary['solvers'] = solvers
    summary['status'] = ('computed' if all(r['converged'] for r in solvers) else 'inverse_not_converged')
    _update_summary(output, summary)
    plot_saved_stage(output, prepared)
    return summary


def plot_saved_stage(output, prepared=None):
    """Draw stage 1/2/3 from saved arrays without FE solves or optimization.

    Only derived images, profiles and the gallery are refreshed. Frozen input,
    solver fields, metadata and identification/validation metrics are read-only.
    """
    from .dic_case_reporting import write_stage_gallery

    output = Path(output)
    summary = json.loads((output/'run_metadata.json').read_text(encoding='utf-8'))
    workflow = summary['workflow']
    if workflow == 'dic_case_preparation':
        with np.load(output/'fields.npz') as fields:
            for name in summary['configuration']['PLATFORMS']:
                if f'{name}_coordinates' not in fields:
                    continue
                measured = fields[f'{name}_observed']
                plot_displacement_clouds(
                    output/name, f'{name}_dic', fields[f'{name}_coordinates'], measured,
                    fields[f'{name}_grid_ids'] if f'{name}_grid_ids' in fields else None,
                    measured.mean(axis=0), f'{name}: Synthetic FE' if summary.get('synthetic_observations') else f'{name}: DIC')
        write_stage_gallery(output, summary, '① DIC 位移云图')
        return summary
    if workflow not in ('dic_calibrated_forward', 'dic_case_inverse'):
        raise ValueError('Expected saved stage 1, 2 or 3 results')
    prepared = Path(prepared) if prepared is not None else output.parent/'01_prepared'
    cfg, mesh, cases, metadata, provenance = load_prepared(prepared)
    if any(summary.get(key) != value for key, value in provenance.items()):
        raise ValueError('Saved stage must refer to the SAME frozen preparation')
    if workflow == 'dic_case_inverse':
        plot_l_curve(output, pd.read_csv(output/'selection.csv'), cfg.REGULARIZATION_MODE)
        if summary['status'].startswith(('l_curve_', 'interrupted_')):
            write_stage_gallery(output, dict(summary, prepared_directory=str(prepared.resolve())),
                                '③ L-curve 选参诊断（尚未生成模量结论）')
            return summary
    if summary['status'] not in ('computed', 'inverse_not_converged'):
        raise ValueError('Complete the numerical stage before requesting its plots')
    with np.load(prepared/'fields.npz') as fields:
        grid_ids = {case.name: fields[f'{case.name}_grid_ids'].copy() for case in cases
                    if f'{case.name}_grid_ids' in fields}
    reference = partition_modulus(pd.DataFrame(metadata['calibration']).mean_E_MPa, cfg.REGION_BOUNDARIES_X)
    bounds = (0., cfg.GEO_L, 0., cfg.GEO_H)
    model = 'forward' if workflow == 'dic_calibrated_forward' else 'inverse'
    if model == 'forward':
        x, y = np.meshgrid((mesh.plot_x[0, :-1]+mesh.plot_x[0, 1:])/2,
                            (mesh.plot_y[:-1, 0]+mesh.plot_y[1:, 0])/2)
        plot_grid_fields(output, 'calibrated_modulus', x, y,
                         [('E', reference(x, y), 'Independent calibrated E')], 'jet',
                         title_prefix='', unit='MPa', bounds=bounds)
    metrics = pd.read_csv(output/'comparison.csv')
    for nu in cfg.NU_VALUES:
        destination = output/f'nu_{nu:.2f}'
        with np.load(destination/'fields.npz') as fields:
            np.testing.assert_allclose(fields['coordinates'], mesh.coord, rtol=0, atol=0)
            if model == 'inverse':
                field = fields['E_inverse']
                if field.shape != (mesh.n_nod,) or not np.isfinite(field).all() or np.any(field <= 0):
                    raise ValueError('Saved inverse must contain one finite positive modulus per node')
                plot_grid_fields(destination, 'inverse_modulus', mesh.plot_x, mesh.plot_y,
                                 [('E', field.reshape(mesh.nods_y, mesh.nods_x), f'Free 2D E, nu={nu:.2f}')],
                                 'jet', title_prefix='', unit='MPa', bounds=bounds)
                save_x_profile(destination, mesh, field, reference)
            else:
                np.testing.assert_allclose(fields['E_calibrated'], reference(mesh.X, mesh.Y), rtol=0, atol=0)
            states = []
            for case in cases:
                prediction = fields[f'{case.name}_prediction']
                residual = case.observation.u_measured-prediction
                rms = np.sqrt(np.sum(case.observation.weight_vector*residual**2)/case.observation.weight_vector.sum())
                row = metrics[(metrics.nu == nu) & (metrics['case'] == case.name) & (metrics.model == model)]
                if len(row) != 1:
                    raise ValueError('Missing/duplicate saved stage metric')
                np.testing.assert_allclose(rms, row.rms_mm.iloc[0], atol=1e-12, rtol=1e-9)
                states.append({'prediction': prediction.copy(), 'weighted_rms': rms})
        save_dic_comparison(destination, cases, {model: states}, cfg.FIT_CASES, grid_ids, nu=nu)
    write_stage_gallery(output, dict(summary, prepared_directory=str(prepared.resolve())),
                        '② 独立标定正演云图' if model == 'forward' else '③ DIC 反演模量及位移云图')
    return summary


def compare_dic_results(prepared, forward, inverse, output, plots=True, plot_only=False):
    """Stage 4: plot and compare saved results; no FE solve or optimization."""
    if Path(output).exists() and not plot_only:
        raise FileExistsError('Choose a new comparison output')
    if plot_only and not (Path(output)/'run_metadata.json').exists():
        raise ValueError('--plot-only requires existing stage 4 results')
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
                   prepared_directory=str(Path(prepared).resolve()),
                   gamma_at_scan_boundary=sources[1]['gamma_at_scan_boundary'], solvers=sources[1]['solvers'],
                   upstream_sources={str(p): file_sha256(p/'run_metadata.json') for p in (forward, inverse)},
                   inverse_settings={key: sources[1]['configuration'][key]
                                     for key in ('FTOL', 'GTOL', 'MAX_ITER', 'MAX_FUN', 'GAMMA_CANDIDATES',
                                                 'REGULARIZATION_MODE', 'EXACT_TV_OPTIONS', 'EXACT_TV_SOLVER',
                                                 'EXACT_TV_PRIMAL_STEP', 'EXACT_TV_COUPLING')
                                     if key in sources[1]['configuration']},
                   gamma_selection_method=sources[1]['gamma_selection_method'],
                   l_curve_enabled=sources[1]['l_curve_enabled'],
                   scan_initialization=sources[1].get('scan_initialization', 'See saved inverse metadata'),
                   initialization=sources[1].get('initialization'),
                   inverse_directory=str(inverse.resolve()),
                   l_curve_status=sources[1]['l_curve_status'], l_curve_message=sources[1]['l_curve_message'])
    metrics = pd.concat([pd.read_csv(forward/'comparison.csv'), pd.read_csv(inverse/'comparison.csv')], ignore_index=True)
    regions = pd.read_csv(inverse/'regional_comparison.csv')
    selection = pd.read_csv(inverse/'selection.csv')
    if plot_only:
        previous = json.loads((output/'run_metadata.json').read_text(encoding='utf-8'))
        # Absolute locations may change when a complete result tree is moved.
        # Each upstream workflow and frozen preparation were checked above;
        # retain the exact metadata hashes as the identity of both sources.
        if (previous['workflow'] != summary['workflow'] or
                any(previous.get(key) != value for key, value in provenance.items()) or
                sorted(previous['upstream_sources'].values()) != sorted(summary['upstream_sources'].values())):
            raise ValueError('Saved comparison must refer to the SAME frozen upstream results')
        for filename, frame in (('comparison.csv', metrics), ('regional_comparison.csv', regions),
                                ('selection.csv', selection)):
            pd.testing.assert_frame_equal(pd.read_csv(output/filename), frame, check_dtype=False)
    else:
        save_run(output, cfg, summary)
    reference = partition_modulus(pd.DataFrame(metadata['calibration']).mean_E_MPa, cfg.REGION_BOUNDARIES_X)
    with np.load(Path(prepared)/'fields.npz') as fields:
        grid_ids = {case.name: fields[f'{case.name}_grid_ids'].copy() for case in cases
                    if f'{case.name}_grid_ids' in fields}
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
        if not plot_only:
            save_run(destination, cfg, dict(summary, nu=nu), {'coordinates': mesh.coord, 'E_inverse': field})
        if plots:
            save_dic_comparison(destination, cases, {'forward': states[0], 'inverse': states[1]},
                                cfg.FIT_CASES, grid_ids, nu=nu)
            plot_modulus_comparison(destination, mesh, reference(mesh.X, mesh.Y), field)
            save_x_profile(destination, mesh, field, reference)
    if not plot_only:
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
