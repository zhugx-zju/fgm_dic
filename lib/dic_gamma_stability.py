"""DIC-only temporal/initialization checks at three saved full-domain H1 gammas.

Run one frozen case per invocation. This is a limited sensitivity experiment,
not an orchestrator for the four project steps and not a new gamma selector.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pandas as pd

from fgm_asm.research import solve_forward
from fgm_asm.load_case import evaluate_observation
from .dic_case_workflow import load_prepared, homogeneous_dic_initial, _optimize
from .dic_preprocessing import relative_displacement_errors
from .experiment_data import load_dic_frame_series, file_sha256

# Existing scan values, common to all three runs and fixed before these checks.
GAMMAS = (1.2067926406393291e-8, 3.906939937054621e-9, 1.2648552168552956e-9)


def changed_observation(case, values):
    values = np.asarray(values)
    obs = replace(case.observation, ux_meas=values[:, 0], uy_meas=values[:, 1], u_measured=values.ravel())
    return replace(case, observation=obs)


def field_distance(mesh, first, second):
    """Area-integrated Q4 L2 relative modulus difference; no pointwise ratios."""
    weights = mesh.gauss_abs_det_j*mesh.gauss_w[:, None]
    def integral(field):
        values = (field[mesh.ele_nods_id-1] @ mesh.gauss_N.T).T
        return np.sum(weights*values**2)
    return float(100*np.sqrt(integral(first-second)/integral(second)))


def target_distance(mesh, case, other):
    base = evaluate_observation(case, np.zeros(mesh.n_dof))['residual']
    target = evaluate_observation(other, np.zeros(mesh.n_dof))['residual']
    w = case.observation.weight_vector
    return float(100*np.sqrt(np.sum(w*(target-base)**2)/np.sum(w*base**2)))


def run_case(root: Path, output: Path, *, resume=False, smoke=False):
    if output.exists() and not resume:
        raise FileExistsError('Use a new study directory or explicit --resume for completed job caches.')
    output.mkdir(parents=True, exist_ok=resume)
    cfg, mesh, cases, metadata, _ = load_prepared(root/'01_prepared')
    if len(cases) != 1 or cfg.REGULARIZATION_MODE != 'gradient' or cfg.CONFIGURED_ELEMENTS != [121, 19]:
        raise ValueError('This protocol requires one frozen full-domain 121x19 H1 observation case.')
    case = cases[0]
    item = cfg.PLATFORMS[case.name]
    run = item['run']
    series = load_dic_frame_series(Path(item['folder']), *item['stages'], coordinate_state=cfg.DIC_COORDINATE_STATE)
    with np.load(root/'01_prepared/fields.npz') as frozen:
        np.testing.assert_array_equal(series.grid_ids, frozen[f'{case.name}_raw_grid_ids'])
    matrix = np.asarray(metadata['matrix'])
    reference = series.coordinates @ matrix.T+np.asarray(metadata['offset_mm'])
    np.testing.assert_allclose(reference[:, 0], case.observation.x_meas, atol=1e-10, rtol=0)
    np.testing.assert_allclose(reference[:, 1], case.observation.y_meas, atol=1e-10, rtol=0)
    frames = series.frame_displacements @ matrix.T
    targets = {'full': frames.mean(axis=0), 'early': frames[:5].mean(axis=0), 'late': frames[-5:].mean(axis=0)}
    np.testing.assert_allclose(targets['full'].ravel(), case.observation.u_measured, atol=1e-12, rtol=0)
    variants = {name: changed_observation(case, value) for name, value in targets.items()}
    initial, init_info = homogeneous_dic_initial(mesh, [case], cfg.THICKNESS, cfg.PRIMARY_NU)
    perturbation = .2*np.cos(2*np.pi*mesh.coord[:, 0]/cfg.GEO_L)*np.cos(np.pi*mesh.coord[:, 1]/cfg.GEO_H)
    perturbed_initial = initial*np.exp(perturbation)
    scan = pd.read_csv(root/'03_inverse/selection.csv')
    inverse_meta = json.loads((root/'03_inverse/run_metadata.json').read_text(encoding='utf-8'))
    protocol = dict(run=run, source_case=str(root.resolve()), gamma_candidates=GAMMAS,
        prepared_metadata_sha256=file_sha256(root/'01_prepared/run_metadata.json'),
        prepared_fields_sha256=file_sha256(root/'01_prepared/fields.npz'),
        original_selected_gamma=inverse_meta['selected_gamma'],
        window_stages=item['stages'], frame_sources=series.metadata['sources'],
        temporal_splits={'full': list(range(15)), 'early': list(range(5)), 'late': list(range(10, 15))},
        initial_uniform=init_info, initial_perturbation='E0*exp(0.2*cos(2*pi*x/L)*cos(pi*y/H)); deterministic',
        initial_source_for_all_temporal_fits='Same full-window DIC homogeneous fit; no independent retuning per split.',
        nu=cfg.PRIMARY_NU, max_iter=cfg.MAX_ITER, max_fun=cfg.MAX_FUN, ftol=cfg.FTOL, gtol=cfg.GTOL,
        calibration_used_in_fit=False, source_implementation_sha256=file_sha256(Path(__file__)),
        hypothesis='Lower gamma should retain DIC-consistent structure across frames/initializations without using calibration to rank it.')
    # JSON conversion makes tuple/list comparisons insensitive to serialization.
    protocol = json.loads(json.dumps(protocol, ensure_ascii=False))
    protocol_path = output/'protocol.json'
    if protocol_path.exists():
        if json.loads(protocol_path.read_text(encoding='utf-8')) != protocol:
            raise ValueError('Study protocol changed; choose a new output directory.')
    else:
        protocol_path.write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding='utf-8')
        np.savez_compressed(output/'observations.npz', coordinates=reference, **targets,
                            initial_uniform=initial, initial_perturbed=perturbed_initial)
    baseline = []
    for gamma in GAMMAS:
        index = int(np.argmin(abs(np.log(scan.gamma.to_numpy()/gamma))))
        row = scan.iloc[index]
        np.testing.assert_allclose(row.gamma, gamma, rtol=1e-12, atol=0)
        source = root/'03_inverse/l_curve'/f'gamma_{row.gamma:g}'/'fields.npz'
        with np.load(source) as saved:
            field = saved['E_inverse'].copy()
        baseline.append(dict(gamma=gamma, row=row, path=source, field=field))
    jobs = [(i, gamma, target, init) for i, gamma in enumerate(GAMMAS)
            for target, init in [('full', 'uniform'), ('early', 'uniform'), ('late', 'uniform')]]
    jobs.append((1, GAMMAS[1], 'full', 'perturbed'))
    if smoke:
        jobs = [(1, GAMMAS[1], 'full', 'uniform')]
    for gamma_index, gamma, target_name, init_name in jobs:
        name = f'g{gamma_index}_{target_name}_{init_name}'
        destination = output/name
        complete = destination/'metrics.json'
        if complete.exists():
            if not resume:
                raise FileExistsError(f'Existing result: {destination}')
            print(f'Run {run} {name}: reusing complete job.', flush=True)
            continue
        if destination.exists():
            raise FileExistsError(f'Incomplete job retained at {destination}; choose a new study for a retry.')
        destination.mkdir()
        (output/'progress.json').write_text(json.dumps({'current_job': name, 'state': 'running'}), encoding='utf-8')
        target = variants[target_name]
        start = initial if init_name == 'uniform' else perturbed_initial
        print(f'Run {run} {name}: gamma={gamma:.8g}, {len(frames) if target_name=="full" else 5} frames.', flush=True)
        with (destination/'solver.log').open('x', encoding='utf-8') as stream, redirect_stdout(stream):
            result = _optimize(cfg, mesh, [target], gamma, cfg.PRIMARY_NU, start.copy())
        field = result['E_final']
        displacement, _ = solve_forward(mesh, target.bc_info['tensile_end_force'], cfg.THICKNESS, cfg.PRIMARY_NU, field)
        state = evaluate_observation(target, displacement)
        rho = float(np.sqrt(np.sum(target.observation.weight_vector*state['residual']**2)))
        np.testing.assert_allclose(rho, result['residual_norm'], rtol=1e-8, atol=1e-10)
        comparison = {}
        for split_name, comparison_case in variants.items():
            prediction = evaluate_observation(comparison_case, displacement)['prediction']
            comparison[split_name] = relative_displacement_errors(comparison_case, prediction)
        base = baseline[gamma_index]
        metric = dict(run=run, job=name, gamma=gamma, split=target_name, initialization=init_name,
            converged=bool(result['converged']), stop_message=result['message'],
            iterations=result['n_iterations'], evaluations=result['n_evaluations'],
            final_gradient_inf=float(np.max(abs(result['final_gradient']))),
            gradient_tolerance_satisfied=bool(np.max(abs(result['final_gradient'])) <= cfg.GTOL),
            final_cost=float(result['final_cost']), residual_norm=result['residual_norm'], regularization_norm=result['regularization_norm'],
            target_change_from_full_percent=target_distance(mesh, case, target),
            field_change_from_saved_continuation_percent=field_distance(mesh, field, base['field']),
            saved_continuation_converged=bool(base['row'].converged),
            saved_continuation_cost=float(base['row'].data_cost+base['row'].regularization_cost),
            saved_continuation_fields_sha256=file_sha256(base['path']), errors_on_splits=comparison,
            minimum_E_MPa=float(field.min()), maximum_E_MPa=float(field.max()),
            source_calibration_used=False, line_search_restarts=result.get('line_search_restarts', []))
        np.savez_compressed(destination/'fields.npz', E_inverse=field, E_initial=start,
                            cost_history=result['cost_history'], gradient_norms=result['gradient_norms'],
                            final_gradient=result['final_gradient'])
        complete.write_text(json.dumps(metric, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f"  stop={result['message']}; iterations={result['n_iterations']}; field Δ={metric['field_change_from_saved_continuation_percent']:.4g}%", flush=True)
    (output/'progress.json').write_text(json.dumps({'state': 'smoke_complete' if smoke else 'complete'}), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    run_case(args.case, args.output, resume=args.resume, smoke=args.smoke)
