"""Finish only pending local-TV candidates after a recorded numeric exception.

No failed candidate is retried. Frozen steps, budgets and certificates remain
unchanged. A candidate without a returned finite field is explicitly excluded;
it is never replaced by the preceding converged field under a new gamma.
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from fgm_asm.exact_tv import ExactTVOptions, exact_tv_inverse_solver
from .dic_case_workflow import (load_prepared, _solver_summary, _tv_arrays,
    _save_tv_evidence, _finish_dic_inverse)
from .dic_case_reporting import publish_stage
from .experiment_data import file_sha256
from .research_reporting import save_run


def complete(root):
    root = Path(root).resolve()
    prepared, output = root/'01_prepared', root/'03_inverse'
    cfg, mesh, cases, metadata, _ = load_prepared(prepared)
    if cfg.REGULARIZATION_MODE != 'exact_tv':
        raise ValueError('Only the interrupted exact-TV local experiment is supported')
    summary = json.loads((output/'run_metadata.json').read_text(encoding='utf-8'))
    if summary['status'] != 'running':
        raise ValueError('Expected a recorded interrupted scan, not a completed experiment')
    scan = pd.read_csv(output/'selection.csv', float_precision='round_trip')
    archive = output/'selection_before_numeric_completion.csv'
    if archive.exists():
        raise FileExistsError('Completion was already attempted; inspect its saved evidence')
    archive.write_bytes((output/'selection.csv').read_bytes())
    raw = (root/'step3.log').read_bytes()
    encoding = 'utf-16' if raw[:2] in (b'\xff\xfe', b'\xfe\xff') else 'utf-8-sig'
    log = raw.decode(encoding, errors='replace')
    if 'FloatingPointError: No positive finite primal step' not in log:
        raise ValueError('Expected the explicitly diagnosed positive-step underflow')
    segment = log[log.rfind('L-curve '):]
    index = int(re.search(r'L-curve (\d+)/50:', segment)[1])-1
    gammas = sorted(cfg.GAMMA_CANDIDATES, reverse=True)
    interrupted = gammas[index]
    if len(scan) != index:
        raise ValueError('Interrupted gamma must immediately follow the saved prefix')
    lower = int(re.findall(r'evaluations=(\d+)', segment)[-1])
    upper = min(lower+200, cfg.MAX_FUN)
    rows = scan.to_dict(orient='records')
    diagnostic = output/'numerical_failures'/f'gamma_{interrupted:g}'
    diagnostic.mkdir(parents=True, exist_ok=False)
    first = dict(gamma=interrupted, nu=cfg.PRIMARY_NU, converged=False,
        residual_norm=np.nan, regularization_norm=np.nan, data_cost=np.nan,
        regularization_cost=np.nan, regularization_resolved=False, iterations=np.nan,
        evaluations=upper, evaluations_lower_bound=lower, evaluations_upper_bound=upper,
        evaluations_are_exact=False, final_gradient_inf=np.nan, gradient_tolerance_satisfied=False,
        stop_message='numerical_error_no_returned_field', numerical_exception='FloatingPointError',
        initial_source=rows[-1]['initial_source'], solution_directory=str(diagnostic.relative_to(output)))
    rows.append(first)
    (diagnostic/'exception.json').write_text(json.dumps(dict(
        exception='FloatingPointError: No positive finite primal step', gamma=interrupted,
        returned_field_available=False, excluded_from_selection=True,
        FE_lower_bound=lower, FE_upper_bound=upper, FE_count_exact=False,
        count_source='last original progress row, then fewer than 200 FE calls before exception',
        original_log_sha256=file_sha256(root/'step3.log')), indent=2), encoding='utf-8')
    valid = scan[scan.converged]
    if valid.empty:
        raise ValueError('No existing certified continuation field')
    parent = valid.iloc[-1]
    initial_source = f'gamma_{parent.gamma:g}'
    with np.load(output/'l_curve'/initial_source/'fields.npz') as z:
        initial, dual = z['E_inverse'].copy(), z['tv_dual'].copy()
    fit = [case for case in cases if case.name in cfg.FIT_CASES]
    pd.DataFrame(rows).to_csv(output/'selection.csv', index=False)
    for gamma in gammas[index+1:]:
        print(f'Pending exact-TV gamma={gamma:.12g}; original frozen step and budget', flush=True)
        history = []
        def progress(row):
            history.append(dict(row))
            if row['iteration'] % 200 == 0:
                print(f"  evaluations={row['evaluations']}, stationarity/tol="
                      f"{row['stationarity']/row['stationarity_tol']:.4g}", flush=True)
        try:
            result = exact_tv_inverse_solver(mesh, fit, initial.copy(), gamma, cfg.PRIMARY_NU,
                options=ExactTVOptions(**cfg.EXACT_TV_OPTIONS, max_evaluations=cfg.MAX_FUN),
                callback=progress, initial_dual=dual.copy(), solver=cfg.EXACT_TV_SOLVER,
                primal_step=cfg.EXACT_TV_PRIMAL_STEP, coupling=cfg.EXACT_TV_COUPLING)
        except FloatingPointError as error:
            if str(error) != 'No positive finite primal step' or not history:
                raise
            last = history[-1]
            count = int(last['evaluations'])
            diagnostic = output/'numerical_failures'/f'gamma_{gamma:g}'
            diagnostic.mkdir(parents=True, exist_ok=False)
            pd.DataFrame(history).to_csv(diagnostic/'history_until_exception.csv', index=False)
            (diagnostic/'exception.json').write_text(json.dumps(dict(
                exception=f'FloatingPointError: {error}', gamma=gamma,
                returned_field_available=False, excluded_from_selection=True,
                FE_lower_bound=count, FE_upper_bound=count, FE_count_exact=True,
                count_source='callback executes immediately before positive-step loop at same FE state'),
                indent=2), encoding='utf-8')
            record = dict(first, gamma=gamma, iterations=last['iteration'], evaluations=count,
                evaluations_lower_bound=count, evaluations_upper_bound=count, evaluations_are_exact=True,
                initial_source=initial_source, solution_directory=str(diagnostic.relative_to(output)))
        else:
            solver = _solver_summary(result)
            count = int(result['n_evaluations'])
            record = dict(gamma=gamma, nu=cfg.PRIMARY_NU, converged=result['converged'],
                residual_norm=result['residual_norm'], regularization_norm=result['regularization_norm'],
                data_cost=.5*result['residual_norm']**2, regularization_cost=gamma*result['regularization_norm'],
                regularization_resolved=result['regularization_resolved'], iterations=result['n_iterations'],
                evaluations=count, evaluations_lower_bound=count, evaluations_upper_bound=count,
                evaluations_are_exact=True, final_gradient_inf=solver['final_gradient_inf'],
                gradient_tolerance_satisfied=solver['final_gradient_inf'] <= cfg.GTOL,
                stop_message=result['message'], initial_source=initial_source,
                solution_directory=f'l_curve/gamma_{gamma:g}')
            destination = output/record['solution_directory']
            save_run(destination, cfg, dict(solver=solver, selection=record, fit_cases=cfg.FIT_CASES,
                uses_prediction_cases=False, objective_definition=summary['objective_definition']),
                dict(E_inverse=result['E_final'], cost_history=result['cost_history'],
                     cost_tar_history=result['cost_tar_history'], cost_reg_history=result['cost_reg_history'],
                     gradient_norms=result['gradient_norms'], final_gradient=result['final_gradient'], **_tv_arrays(result)))
            _save_tv_evidence(destination, result)
            if result['converged']:
                initial, dual = result['E_final'].copy(), result['tv_dual'].copy()
                initial_source = f'gamma_{gamma:g}'
        rows.append(record)
        pd.DataFrame(rows).to_csv(output/'selection.csv', index=False)
        print(f"  {record['stop_message']}, FE={record['evaluations']}", flush=True)
    summary['scan_completion'] = dict(
        method='record numeric failures without returned fields; evaluate only pending candidates, no retries',
        initial_exception_gamma=interrupted, initial_exception_FE_lower_bound=lower,
        initial_exception_FE_upper_bound=upper, original_selection_sha256=file_sha256(archive),
        source_sha256=file_sha256(Path(__file__)), changed_solver_or_budget=False)
    result = _finish_dic_inverse(cfg, mesh, cases, metadata, summary, rows, prepared, output)
    publish_stage(root, '03_inverse', output, result)
    print('Completion:', result['status'], output, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    complete(parser.parse_args().root)
