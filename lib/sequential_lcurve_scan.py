"""L-curve scan with a full sequential 1->2->3 inversion inside every gamma.

The primary endpoint curve uses ONE terminal field to replay ALL observations.
The terminal-stage curve is also retained because the stages optimize current-
case objectives, rather than a summed three-case objective.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from fgm_asm.load_case import evaluate_observation
from fgm_asm.material import MaterialInfo
from fgm_asm.regularization import get_tikhonov_regularization
from fgm_asm.research import solve_forward
from .dic_case_workflow import _optimize, homogeneous_dic_initial
from .dic_preprocessing import relative_displacement_errors
from .experiment_data import file_sha256
from .joint_reference_diagnostic import build_common_cases


def optimize_stage(cfg, mesh, case, gamma, initial):
    """Recover line-search failures using only the ORIGINAL remaining budget."""
    current = initial.copy()
    histories = {name: [] for name in ('cost_history', 'cost_tar_history', 'cost_reg_history', 'gradient_norms')}
    attempts, total_iterations, total_evaluations = [], 0, 0
    for attempt in range(3):
        limited = SimpleNamespace(**vars(cfg))
        limited.MAX_ITER = cfg.MAX_ITER-total_iterations
        limited.MAX_FUN = cfg.MAX_FUN-total_evaluations
        if min(limited.MAX_ITER, limited.MAX_FUN) <= 0:
            break
        result = _optimize(limited, mesh, [case], gamma, cfg.PRIMARY_NU, current.copy())
        total_iterations += result['n_iterations']
        total_evaluations += result['n_evaluations']
        attempts.append(dict(attempt=attempt+1, message=result['message'], converged=bool(result['converged']),
            iterations=result['n_iterations'], evaluations=result['n_evaluations'],
            final_gradient_inf=float(np.max(abs(result['final_gradient']))),
            internal_restarts=result.get('line_search_restarts', [])))
        for name in histories:
            histories[name].extend(result[name] if attempt == 0 else result[name][1:])
        current = result['E_final'].copy()
        if result['converged'] or 'ABNORMAL' not in result['message'].upper():
            break
    for name, history in histories.items():
        result[name] = np.asarray(history)
    result.update(n_iterations=total_iterations, n_evaluations=total_evaluations,
                  attempts=attempts, within_original_budget=bool(total_iterations <= cfg.MAX_ITER and total_evaluations <= cfg.MAX_FUN))
    return result


def verify_sources(audit):
    for record in audit:
        root = Path(record['source'])/'01_prepared'
        assert file_sha256(root/'fields.npz') == record['prepared_fields_sha256']
        assert file_sha256(root/'run_metadata.json') == record['prepared_metadata_sha256']


def run(roots: list[Path], output: Path, *, resume=False, smoke=False, gamma_log10_range=None):
    if output.exists() and not resume:
        raise FileExistsError('Use a new output, or explicit --resume for identical completed gamma caches.')
    cfg, mesh, cases, audit, arrays, _ = build_common_cases(roots)
    initial, initialization = homogeneous_dic_initial(mesh, [cases[0]], cfg.THICKNESS, cfg.PRIMARY_NU)
    gammas = np.sort(np.asarray(cfg.GAMMA_CANDIDATES))[::-1]
    if gamma_log10_range is not None:
        bounds=np.asarray(gamma_log10_range,dtype=float)
        if bounds.shape!=(2,) or not np.isfinite(bounds).all() or bounds[0]>=bounds[1]:
            raise ValueError('Gamma log10 bounds must be two finite, strictly increasing numbers.')
        gammas=np.logspace(bounds[0],bounds[1],len(gammas))[::-1]
        if not (np.isfinite(gammas).all() and (gammas>0).all()):
            raise ValueError('Gamma bounds overflow or underflow the numerical range.')
    output.mkdir(parents=True, exist_ok=resume)
    protocol = dict(source_cases=audit, common_geometry_mm=[mesh.geo_l, mesh.geo_h, cfg.THICKNESS],
        elements=[121,19], nu=cfg.PRIMARY_NU, gamma_candidates=gammas.tolist(), order=[c.name for c in cases],
        gamma_grid_source='explicit_experiment_log10_bounds' if gamma_log10_range is not None else 'frozen_source_candidates',
        gamma_log10_range=[float(np.log10(gammas.min())),float(np.log10(gammas.max()))],
        stage_objective='0.5*r_current.T W_current r_current + 0.5*gamma*integral(|grad E|^2)',
        per_gamma_initialization=initialization, cross_gamma_continuation=False,
        stage_transfer='Previous E_final becomes the next E_initial exactly; reset L-BFGS history.',
        primary_rho='sqrt(sum_c r_c(E_terminal).T W_c r_c(E_terminal)); all residuals replayed with ONE terminal field',
        primary_eta='sqrt(integral(|grad E_terminal|^2)); no gamma factor',
        secondary_curve='Terminal current-case residual norm and the same terminal regularization norm.',
        mathematical_scope='Endpoint all-case curve is a sequential diagnostic. Terminal field does not minimize the all-case objective. Original geometric corner checks are retained without assuming monotonicity.',
        case_weights='Original area weights; no force, modulus, noise or relative-error normalization.',
        max_iter_per_stage=cfg.MAX_ITER, max_fun_per_stage=cfg.MAX_FUN, ftol=cfg.FTOL, gtol=cfg.GTOL,
        failure_policy='At most two history resets after ABNORMAL within original remaining per-stage budget. Stop chain if still unsuccessful; never pass a failed field onward.',
        calibration_used_in_fit_or_gamma_selection=False,
        coordinate_scope='Earlier conditional center registration, independent reference images and force increments; no material-landmark certification.',
        implementation_sha256=file_sha256(Path(__file__)),
        common_preparation_sha256=file_sha256(Path(__file__).with_name('joint_reference_diagnostic.py')))
    path = output/'protocol.json'
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8')) != protocol:
            raise ValueError('Protocol/source changed. Start a new output rather than overwrite cached scans.')
    else:
        path.write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding='utf-8')
        (output/'implementation_snapshot.py').write_bytes(Path(__file__).read_bytes())
        np.savez_compressed(output/'observations.npz', **arrays, mesh_coordinates=mesh.coord, E_initial=initial)
    selected_gammas = [gammas[np.argmin(abs(np.log(gammas/1e-8)))]] if smoke else gammas
    for gamma in selected_gammas:
        destination = output/'scan'/f'gamma_{gamma:.12g}'
        complete = destination/'endpoint.json'
        if complete.exists():
            print(f'gamma={gamma:.12g}: reusing completed cache.',flush=True)
            continue
        if destination.exists():
            raise FileExistsError(f'Incomplete gamma cache retained: {destination}; choose a new output for a retry.')
        destination.mkdir(parents=True)
        (output/'progress.json').write_text(json.dumps(dict(state='running',gamma=float(gamma))),encoding='utf-8')
        print(f'gamma={gamma:.12g}: begin full 1->2->3 chain.',flush=True)
        current, stages, success = initial.copy(), [], True
        for index, case in enumerate(cases, start=1):
            stage = destination/f'stage{index}_{case.name}'
            stage.mkdir()
            if index == 1:
                np.testing.assert_array_equal(current,initial)
            else:
                with np.load(destination/f'stage{index-1}_{cases[index-2].name}/fields.npz') as previous:
                    np.testing.assert_array_equal(current,previous['E_inverse'])
            starting = current.copy()
            with (stage/'solver.log').open('x',encoding='utf-8') as stream, redirect_stdout(stream):
                result=optimize_stage(cfg,mesh,case,float(gamma),starting)
            current=result['E_final'].copy()
            material=MaterialInfo(cfg.PRIMARY_NU)
            material.update(current)
            eta=float(np.sqrt(get_tikhonov_regularization(mesh,material)))
            u,_=solve_forward(mesh,case.bc_info['tensile_end_force'],cfg.THICKNESS,cfg.PRIMARY_NU,current)
            state=evaluate_observation(case,u)
            rho=float(np.sqrt(2*state['cost_tar']))
            np.testing.assert_allclose(rho,result['residual_norm'],rtol=1e-8,atol=1e-10)
            np.testing.assert_allclose(eta,result['regularization_norm'],rtol=1e-9,atol=1e-9)
            qualified=bool(result['converged'] and result['within_original_budget'])
            record=dict(gamma=float(gamma),stage=index,case=case.name,converged=qualified,
                solver_success=bool(result['converged']),message=result['message'],
                iterations=result['n_iterations'],evaluations=result['n_evaluations'],
                gradient_inf=float(np.max(abs(result['final_gradient']))),
                gradient_tolerance_satisfied=bool(np.max(abs(result['final_gradient']))<=cfg.GTOL),
                within_original_budget=result['within_original_budget'],attempts=result['attempts'],
                residual_norm=rho,regularization_norm=eta,
                objective=float(.5*rho**2+.5*gamma*eta**2),
                source_initial='same_first_case_homogeneous' if index==1 else f'stage{index-1}',
                **relative_displacement_errors(case,state['prediction']))
            np.savez_compressed(stage/'fields.npz',E_inverse=current,E_initial=starting,
                cost_history=result['cost_history'],cost_tar_history=result['cost_tar_history'],
                cost_reg_history=result['cost_reg_history'],gradient_norms=result['gradient_norms'],final_gradient=result['final_gradient'])
            (stage/'metrics.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
            stages.append(record)
            print(f'  stage{index} {case.name}: success={qualified}, iterations={record["iterations"]}, attempts={len(record["attempts"])}',flush=True)
            if not qualified:
                success=False
                break
        endpoint=dict(gamma=float(gamma),converged=success and len(stages)==3,completed_stages=len(stages),
            iterations=sum(r['iterations'] for r in stages),evaluations=sum(r['evaluations'] for r in stages),
            all_stage_gradient_tolerances_satisfied=bool(len(stages)==3 and all(r['gradient_tolerance_satisfied'] for r in stages)),
            stage_records=stages,residual_norm=None,regularization_norm=None,terminal_case_residual_norm=None)
        if endpoint['converged']:
            errors, residual_square, fields = [], 0., dict(E_inverse=current,coordinates=mesh.coord)
            for case in cases:
                u,_=solve_forward(mesh,case.bc_info['tensile_end_force'],cfg.THICKNESS,cfg.PRIMARY_NU,current)
                state=evaluate_observation(case,u)
                residual_square+=2*state['cost_tar']
                errors.append(dict(case=case.name,data_cost=state['cost_tar'],**relative_displacement_errors(case,state['prediction'])))
                fields.update({f'{case.name}_prediction':state['prediction'],f'{case.name}_residual':state['residual']})
            # Check the aggregate residual by direct concatenation, not only cost sums.
            direct=np.concatenate([np.sqrt(case.observation.weight_vector)*fields[f'{case.name}_residual'] for case in cases])
            rho=float(np.sqrt(residual_square))
            np.testing.assert_allclose(rho,np.linalg.norm(direct),rtol=1e-12,atol=1e-12)
            endpoint.update(residual_norm=rho,regularization_norm=stages[-1]['regularization_norm'],
                terminal_case_residual_norm=stages[-1]['residual_norm'],case_errors=errors,
                minimum_E_MPa=float(current.min()),maximum_E_MPa=float(current.max()))
            np.savez_compressed(destination/'fields.npz',**fields)
        complete.write_text(json.dumps(endpoint,ensure_ascii=False,indent=2),encoding='utf-8')
        endpoints=[json.loads(p.read_text(encoding='utf-8')) for p in (output/'scan').glob('*/endpoint.json')]
        pd.DataFrame([{k:v for k,v in row.items() if k not in ('stage_records','case_errors')} for row in endpoints]).sort_values('gamma').to_csv(output/'scan_progress.csv',index=False,lineterminator='\n')
    verify_sources(audit)
    (output/'progress.json').write_text(json.dumps(dict(state='smoke_complete' if smoke else 'scan_complete',
        cached_gammas=len(list((output/'scan').glob('*/endpoint.json'))))),encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases',type=Path,nargs=3,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--gamma-log10-range',type=float,nargs=2,metavar=('LOW','HIGH'),
                        help='Experimental gamma bounds in log10 units; retain the frozen candidate count.')
    args=parser.parse_args()
    run(args.cases,args.output,resume=args.resume,smoke=args.smoke,gamma_log10_range=args.gamma_log10_range)
