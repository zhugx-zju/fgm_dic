"""Explicit additional-budget recovery of a TV L-curve's missing support.

The complete original scan remains immutable in its trial directories. Only
failed solves needed to bracket the existing positive-curvature peak are
restarted, at the same gamma and with the same data and KKT tolerances.
Neither calibration discrepancies nor held-out predictions guide recovery.
"""

import argparse
import json
import re
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from fgm_asm.l_curve import select_l_curve_corner, MIN_POSITIVE_CURVATURE
from .dic_case_reporting import publish_stage
from .dic_case_workflow import (load_prepared, _optimize, _solver_summary,
    _tv_arrays, _save_tv_evidence, _update_summary, _finish_dic_inverse)
from .experiment_data import file_sha256
from .research_reporting import save_run


def complete_interrupted_scan(root, retry_evaluations=2000, retry_step=None):
    """Save a bounded, smaller-step retry of the missing final scan point.

    The interrupted attempt has no recoverable last iterate. Its FE cost is
    therefore bounded from the saved progress log, never invented or omitted.
    Earlier trials are untouched. A failed retry remains excluded from curvature.
    """
    root = Path(root)
    prepared, output = root/'01_prepared', root/'03_inverse'
    cfg, mesh, cases, metadata, provenance = load_prepared(prepared)
    summary = json.loads((output/'run_metadata.json').read_text(encoding='utf-8'))
    scan_path = output/'selection.csv'
    scan = pd.read_csv(scan_path, float_precision='round_trip')
    gammas = sorted(cfg.GAMMA_CANDIDATES, reverse=True)
    if (summary['status'] != 'running' or cfg.REGULARIZATION_MODE != 'exact_tv' or
            len(scan) != len(gammas)-1 or summary['configuration'] != metadata['configuration'] or
            any(summary.get(k) != v for k,v in provenance.items())):
        raise ValueError('Expected an interrupted exact-TV scan missing only its final point')
    np.testing.assert_allclose(scan.gamma, gammas[:-1], rtol=1e-14, atol=0)
    log_path = root/'checks/step3.log'
    log = log_path.read_text(encoding='utf-8')
    if 'FloatingPointError: No positive finite primal step' not in log:
        raise ValueError('The recorded interruption does not match a positivity-step breakdown')
    matches = re.findall(r'evaluations=(\d+)',log.split(f'L-curve {len(gammas)}/{len(gammas)}:')[-1])
    if not matches or not 2 <= retry_evaluations <= cfg.MAX_FUN:
        raise ValueError('Need a logged FE lower bound and a bounded retry budget')
    step = cfg.EXACT_TV_PRIMAL_STEP/4 if retry_step is None else retry_step
    if not np.isfinite(step) or not 0 < step < cfg.EXACT_TV_PRIMAL_STEP:
        raise ValueError('Retry must use a smaller positive numerical step')
    protocol_path = output/'interrupted_scan_protocol.json'
    if protocol_path.exists():
        raise FileExistsError('An interrupted-point completion has already been attempted')
    previous = scan[scan.converged].iloc[-1]
    trial = output/'l_curve'/f'gamma_{previous.gamma:g}'
    gamma = gammas[-1]
    protocol = dict(interrupted_gamma=gamma, exception='No positive finite primal step',
        interrupted_FE_lower_bound=max(map(int,matches)), interrupted_FE_upper_bound=cfg.MAX_FUN,
        exact_interrupted_FE_count_available=False, log_sha256=file_sha256(log_path),
        retry_evaluation_budget=retry_evaluations, retry_primal_step=step,
        retry_initial_source=str(trial.resolve()), retry_initial_fields_sha256=file_sha256(trial/'fields.npz'),
        uses_calibration_or_prediction=False, tolerances_and_objective_unchanged=True, **provenance)
    protocol_path.write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding='utf-8')
    (output/'selection_before_completion.csv').write_bytes(scan_path.read_bytes())
    (output/'run_metadata_before_completion.json').write_bytes((output/'run_metadata.json').read_bytes())
    with np.load(trial/'fields.npz') as saved:
        initial, dual = saved['E_inverse'].copy(), saved['tv_dual'].copy()
    limited = SimpleNamespace(**{k:v for k,v in vars(cfg).items() if k.isupper()},__file__=None)
    limited.MAX_FUN, limited.EXACT_TV_PRIMAL_STEP = retry_evaluations, step
    limited.EXACT_TV_OPTIONS = dict(cfg.EXACT_TV_OPTIONS,max_outer=retry_evaluations)
    fit = [c for c in cases if c.name in cfg.FIT_CASES]
    print(f'Complete final gamma={gamma:g}: smaller step={step:g}, retry FE budget={retry_evaluations}',flush=True)
    result = _optimize(limited,mesh,fit,gamma,cfg.PRIMARY_NU,initial,dual)
    solver = _solver_summary(result)
    record = dict(gamma=gamma,nu=cfg.PRIMARY_NU,converged=result['converged'],
        residual_norm=result['residual_norm'],regularization_norm=result['regularization_norm'],
        data_cost=.5*result['residual_norm']**2,regularization_cost=gamma*result['regularization_norm'],
        regularization_resolved=result['regularization_resolved'],iterations=result['n_iterations'],
        evaluations=result['n_evaluations'],final_gradient_inf=solver['final_gradient_inf'],
        gradient_tolerance_satisfied=solver['final_gradient_inf'] <= cfg.GTOL,
        stop_message=result['message'],initial_source=f'gamma_{previous.gamma:g}; smaller-step retry after numerical breakdown')
    destination = output/'l_curve'/f'gamma_{gamma:g}'
    save_run(destination,limited,dict(solver=solver,selection=record,interrupted_attempt=protocol,
        fit_cases=cfg.FIT_CASES,uses_prediction_cases=False),dict(E_inverse=result['E_final'],
        final_gradient=result['final_gradient'],**{k:result[k] for k in (
            'cost_history','cost_tar_history','cost_reg_history','gradient_norms')},**_tv_arrays(result)))
    _save_tv_evidence(destination,result)
    scan = pd.concat([scan,pd.DataFrame([record])],ignore_index=True)
    scan.to_csv(scan_path,index=False)
    summary['scan_completion'] = dict(protocol,retry_actual_evaluations=result['n_evaluations'],retry_converged=result['converged'])
    _update_summary(output,summary)
    result = _finish_dic_inverse(cfg,mesh,cases,metadata,summary,scan.to_dict(orient='records'),prepared,output)
    publish_stage(root,'03_inverse',output,result)
    return result


def missing_peak_support(scan, attempted=()):
    """Select failed neighbors of the largest positive curvature, without truth."""
    qualified = scan.converged & scan.regularization_resolved
    corner = select_l_curve_corner(scan.gamma, scan.residual_norm,
                                  scan.regularization_norm, qualified)
    if corner['selected_gamma'] is not None:
        return corner, []
    curvature = corner['curvature']
    candidates = np.flatnonzero(np.isfinite(curvature) & (curvature > MIN_POSITIVE_CURVATURE))
    if not len(candidates):
        return corner, []
    peak = int(candidates[np.argmax(curvature[candidates])])
    order = np.argsort(scan.gamma.to_numpy())
    position = int(np.flatnonzero(order == peak)[0])
    if position < 2 or position > len(order)-3:
        return corner, []
    targets = [int(i) for i in order[position-2:position+3]
               if not bool(scan.iloc[i].converged) and int(i) not in attempted
               and 'budget' in str(scan.iloc[i].stop_message)]
    # Recompute curvature after each repair. A newly resolved stronger-gamma
    # peak can make a farther weak-gamma retry unnecessary.
    return corner, sorted(targets, key=lambda i: scan.iloc[i].gamma, reverse=True)[:1]


def recover(root, extra_evaluations=12000, max_rounds=3):
    """Freeze an explicit budget extension, then finish the original stage 3."""
    root = Path(root)
    if extra_evaluations < 2 or max_rounds < 1:
        raise ValueError('Recovery needs at least two evaluations and one round')
    prepared, output = root/'01_prepared', root/'03_inverse'
    cfg, mesh, cases, metadata, provenance = load_prepared(prepared)
    summary = json.loads((output/'run_metadata.json').read_text(encoding='utf-8'))
    if (cfg.REGULARIZATION_MODE != 'exact_tv' or
            not summary['status'].startswith('l_curve_') or
            summary['workflow'] != 'dic_case_inverse' or
            summary['configuration'] != metadata['configuration'] or
            any(summary.get(k) != v for k, v in provenance.items())):
        raise ValueError('Recovery requires a completed unresolved exact-TV scan with unchanged frozen inputs')
    if any(output.glob('nu_*/fields.npz')):
        raise FileExistsError('Final fields already exist; recovery must not overwrite them')
    scan_path = output/'selection.csv'
    scan = pd.read_csv(scan_path, float_precision='round_trip')
    if len(scan) != len(cfg.GAMMA_CANDIDATES) or scan.gamma.duplicated().any():
        raise ValueError('Complete the declared scan before extending its budget')
    np.testing.assert_allclose(np.sort(scan.gamma), np.sort(cfg.GAMMA_CANDIDATES), rtol=1e-14, atol=0)
    archive = output/'selection_before_tv_recovery.csv'
    protocol_path = output/'tv_recovery_protocol.json'
    if archive.exists() or protocol_path.exists():
        raise FileExistsError('A recovery already exists; inspect it before another attempt')
    protocol = dict(method='same-gamma primal/dual restart only for missing support of the existing positive-curvature peak',
        order='one failed higher-gamma neighbor first, then recompute the entire curve before any further retry',
        base_evaluation_budget_per_gamma=cfg.MAX_FUN,
        additional_evaluation_budget_per_attempt=extra_evaluations,
        maximum_attempts_per_gamma=1, maximum_support_rounds=max_rounds,
        restart='saved failed E and dual; reset extrapolation to E',
        selection_data=list(cfg.FIT_CASES), uses_calibration_or_held_out_for_decisions=False,
        tolerances_unchanged=True,
        original_selection_sha256=file_sha256(scan_path),
        original_summary_sha256=file_sha256(output/'run_metadata.json'),
        source_sha256={str(Path(__file__).resolve()):file_sha256(Path(__file__))}, **provenance)
    archive.write_bytes(scan_path.read_bytes())
    (output/'run_metadata_before_tv_recovery.json').write_bytes((output/'run_metadata.json').read_bytes())
    protocol_path.write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding='utf-8')
    fit = [case for case in cases if case.name in cfg.FIT_CASES]
    attempted, attempts = set(), []
    for current_round in range(max_rounds):
        corner, targets = missing_peak_support(scan, attempted)
        if not targets:
            break
        for index in targets:
            previous = {k:v for k,v in scan.iloc[index].to_dict().items()
                        if k not in ('curvature','eligible','selected','solution_directory')}
            gamma = float(previous['gamma'])
            trial = output/'l_curve'/f'gamma_{gamma:g}'
            destination = trial/'tv_recovery_1'
            with np.load(trial/'fields.npz') as saved:
                initial = saved['E_inverse'].copy()
                dual = saved['tv_dual'].copy()
                old = {key:saved[key].copy() for key in (
                    'cost_history','cost_tar_history','cost_reg_history','gradient_norms')}
            limited = SimpleNamespace(**{k:v for k,v in vars(cfg).items() if k.isupper()}, __file__=None)
            limited.MAX_FUN = extra_evaluations
            limited.EXACT_TV_OPTIONS = dict(cfg.EXACT_TV_OPTIONS, max_outer=extra_evaluations)
            print(f'TV support recovery gamma={gamma:.12g}, extra FE budget={extra_evaluations}', flush=True)
            start = time.perf_counter()
            result = _optimize(limited, mesh, fit, gamma, cfg.PRIMARY_NU, initial, dual)
            extra_used = int(result['n_evaluations'])
            elapsed = time.perf_counter()-start
            base_iterations, base_evaluations = int(previous['iterations']), int(previous['evaluations'])
            for row in result['tv_history']:
                row['iteration'] += base_iterations
                row['evaluations'] += base_evaluations
            result['tv_certificate'] = result['tv_history'][-1]
            old_history = pd.read_csv(trial/'tv_history.csv').to_dict(orient='records')
            result['tv_history'] = old_history+result['tv_history']
            result['n_iterations'] += base_iterations
            result['n_evaluations'] += base_evaluations
            for key in old:
                result[key] = np.r_[old[key], result[key]]
            solver = _solver_summary(result)
            record = dict(previous, converged=result['converged'],
                residual_norm=result['residual_norm'], regularization_norm=result['regularization_norm'],
                data_cost=.5*result['residual_norm']**2,
                regularization_cost=gamma*result['regularization_norm'],
                regularization_resolved=result['regularization_resolved'],
                iterations=result['n_iterations'], evaluations=result['n_evaluations'],
                final_gradient_inf=solver['final_gradient_inf'],
                gradient_tolerance_satisfied=solver['final_gradient_inf'] <= cfg.GTOL,
                stop_message=result['message'], solution_directory=str(destination.relative_to(output)))
            record['initial_source'] = str(previous['initial_source'])+'; same-gamma failed-field restart'
            budget_cfg = SimpleNamespace(**{k:v for k,v in vars(cfg).items() if k.isupper()}, __file__=None)
            budget_cfg.MAX_FUN = base_evaluations+extra_evaluations
            budget_cfg.EXACT_TV_OPTIONS = dict(cfg.EXACT_TV_OPTIONS,
                max_outer=base_iterations+extra_evaluations)
            attempt = dict(round=current_round+1, gamma=gamma, extra_evaluations=extra_used,
                total_evaluations=result['n_evaluations'], converged=result['converged'], elapsed_seconds=elapsed)
            save_run(destination, budget_cfg, dict(solver=solver, selection=record,
                previous_selection=previous, budget_extension=protocol, attempt=attempt,
                fit_cases=cfg.FIT_CASES, uses_prediction_cases=False,
                initial_fields_sha256=file_sha256(trial/'fields.npz')),
                dict(E_inverse=result['E_final'], final_gradient=result['final_gradient'],
                     **{k:result[k] for k in old}, **_tv_arrays(result)))
            _save_tv_evidence(destination, result)
            if 'solution_directory' not in scan:
                scan['solution_directory'] = None
            for key, value in record.items():
                scan.at[index, key] = value
            scan.to_csv(scan_path, index=False)
            attempted.add(index)
            attempts.append(attempt)
    summary['scan_recovery'] = dict(protocol, attempts=attempts,
        original_selection=str(archive.resolve()), actual_extra_FE_evaluations=sum(a['extra_evaluations'] for a in attempts))
    summary['status'] = 'running'
    _update_summary(output, summary)
    result = _finish_dic_inverse(cfg, mesh, cases, metadata, summary,
                                scan.to_dict(orient='records'), prepared, output)
    publish_stage(root, '03_inverse', output, result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--extra-evaluations', type=int, default=12000)
    parser.add_argument('--max-rounds', type=int, default=3)
    parser.add_argument('--complete-interrupted', action='store_true')
    parser.add_argument('--retry-evaluations', type=int, default=2000)
    args = parser.parse_args()
    result = (complete_interrupted_scan(args.root,args.retry_evaluations) if args.complete_interrupted else
              recover(args.root, args.extra_evaluations, args.max_rounds))
    print('TV recovery:',result['status'],result.get('selected_gamma'))
    if result['status'] != 'computed':
        raise SystemExit(2)
