"""Complete independent weak-gamma 380N TV trials with unchanged per-point budget."""
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import json
import re
import numpy as np
import pandas as pd
from lib.dic_case_workflow import (load_prepared,_optimize,_solver_summary,_tv_arrays,
    _save_tv_evidence,_update_summary,_finish_dic_inverse)
from lib.research_reporting import save_run
from lib.experiment_data import file_sha256

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/'results/dic_full/load380_20261022/exact_tv'

def complete():
    prepared=OUTPUT/'01_prepared';inverse=OUTPUT/'03_inverse'
    cfg,mesh,cases,meta,_=load_prepared(prepared);fit=[c for c in cases if c.name=='380N']
    summary=json.loads((inverse/'run_metadata.json').read_text(encoding='utf-8'))
    if summary['status']!='running':raise ValueError('Expected interrupted unfinished coarse TV scan')
    scan=pd.read_csv(inverse/'selection.csv',float_precision='round_trip')
    archive=inverse/'selection_before_parallel.csv'
    if archive.exists():raise FileExistsError(archive)
    archive.write_bytes((inverse/'selection.csv').read_bytes())
    valid=scan[scan.converged & scan.regularization_resolved]
    parent=valid.sort_values('gamma').iloc[0]
    source=inverse/'l_curve'/f'gamma_{parent.gamma:g}'
    with np.load(source/'fields.npz') as z:initial=z['E_inverse'].copy();dual=z['tv_dual'].copy()
    destination=inverse/'parallel_scan';destination.mkdir(exist_ok=False)
    rows=scan.to_dict(orient='records');done=set(scan.gamma)
    pending=[g for g in sorted(cfg.GAMMA_CANDIDATES,reverse=True) if g not in done]
    print('Parallel remaining coarse gamma trials:',len(pending),'shared converged start gamma:',parent.gamma,flush=True)
    def solve(gamma):
        p=dual.copy();mag=np.linalg.norm(p,axis=1);p*=np.minimum(1.,gamma/np.maximum(mag,np.finfo(float).tiny))[:,None]
        result=_optimize(cfg,mesh,fit,gamma,.26,initial.copy(),p);solver=_solver_summary(result)
        record=dict(gamma=gamma,nu=.26,converged=result['converged'],residual_norm=result['residual_norm'],
            regularization_norm=result['regularization_norm'],data_cost=.5*result['residual_norm']**2,
            regularization_cost=gamma*result['regularization_norm'],regularization_resolved=result.get('regularization_resolved',True),
            iterations=result['n_iterations'],evaluations=result['n_evaluations'],final_gradient_inf=solver['final_gradient_inf'],
            gradient_tolerance_satisfied=solver['final_gradient_inf']<=cfg.GTOL,stop_message=result['message'],
            initial_source=f'fixed converged 380N coarse gamma_{parent.gamma:g}',
            solution_directory=str((destination/f'gamma_{gamma:g}').relative_to(inverse)))
        place=inverse/record['solution_directory']
        save_run(place,cfg,dict(solver=solver,selection=record,fit_cases=['380N'],uses_prediction_cases=False,
            numerical_initialization_source=str(source),initial_field_sha256=file_sha256(source/'fields.npz')),
            dict(E_inverse=result['E_final'],cost_history=result['cost_history'],
                cost_tar_history=result['cost_tar_history'],cost_reg_history=result['cost_reg_history'],
                gradient_norms=result['gradient_norms'],final_gradient=result['final_gradient'],**_tv_arrays(result)))
        _save_tv_evidence(place,result)
        return record
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures=[pool.submit(solve,gamma) for gamma in pending]
        for future in as_completed(futures):
            rows.append(future.result());rows.sort(key=lambda r:r['gamma'],reverse=True)
            pd.DataFrame(rows).to_csv(inverse/'selection.csv',index=False)
    log=(OUTPUT/'step3.log').read_text(encoding='utf-8-sig',errors='replace')
    last=re.findall(r'L-curve (\d+)/50: gamma=([^\s]+)',log)[-1]
    segment=log[log.rfind('L-curve '):]
    printed=re.findall(r'evaluations=(\d+)',segment)
    protocol=dict(workers=6,completed_serial_trials=len(scan),remaining_parallel_trials=len(pending),
        last_converged_serial_gamma=float(parent.gamma),starting_field_sha256=file_sha256(source/'fields.npz'),
        same_per_point_budget=12000,same_objective_and_tolerances=True,
        uses_only_380N_for_initialization=True,uses_calibration_or_prediction=False,
        original_scan_archive=str(archive),original_trials_preserved=True,
        aborted_trial_gamma=float(last[1]),aborted_trial_exact_FE_count_known=False,
        aborted_trial_last_printed_FE_count=int(printed[-1]) if printed else 0,
        source_sha256=file_sha256(Path(__file__)))
    summary['parallel_continuation']=protocol
    summary['scan_initialization']='Serial converged decreasing-gamma continuation first; remaining weak-gamma trials share the last converged 380N field for independent parallel solves. Every trial retains original gamma, FE budget and KKT tolerances.'
    (destination/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    _update_summary(inverse,summary)
    result=_finish_dic_inverse(cfg,mesh,cases,meta,summary,rows,prepared,inverse)
    print('Coarse TV final status:',result['status'],flush=True)

if __name__=='__main__':complete()
