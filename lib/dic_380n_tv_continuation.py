"""Coarse-to-fine numerical starts for the unchanged fine-grid 380N TV objective."""
from pathlib import Path
import argparse
import json
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from lib.dic_case_workflow import load_prepared,_optimize
from lib.experiment_data import file_sha256

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/'results/dic_full/load380_20261022'

def warm_start(fine_mesh,gamma):
    coarse=OUTPUT/'exact_tv'
    cfg,mesh,_,_,_=load_prepared(coarse/'01_prepared')
    source=coarse/'03_inverse/l_curve'/f'gamma_{gamma:g}'
    metadata=json.loads((source/'run_metadata.json').read_text(encoding='utf-8'))
    if not metadata['solver']['converged']:raise ValueError('A failed coarse field cannot initialize the fine scan')
    with np.load(source/'fields.npz') as z:E=z['E_inverse'].copy();dual=z['tv_dual'].copy()
    field=RegularGridInterpolator((mesh.plot_y[:,0],mesh.plot_x[0]),E.reshape(mesh.nods_y,mesh.nods_x))(
        np.column_stack([fine_mesh.Y,fine_mesh.X]))
    x=(mesh.x_ele@mesh.gauss_N.T).T.ravel();y=(mesh.y_ele@mesh.gauss_N.T).T.ravel()
    xs,ix=np.unique(np.round(x,10),return_inverse=True);ys,iy=np.unique(np.round(y,10),return_inverse=True)
    values=np.empty((len(ys),len(xs),2));values[iy,ix]=dual
    assert len(xs)*len(ys)==len(dual)
    xf=(fine_mesh.x_ele@fine_mesh.gauss_N.T).T.ravel();yf=(fine_mesh.y_ele@fine_mesh.gauss_N.T).T.ravel()
    points=np.column_stack([np.clip(yf,ys[0],ys[-1]),np.clip(xf,xs[0],xs[-1])])
    p=RegularGridInterpolator((ys,xs),values)(points)
    norms=np.linalg.norm(p,axis=1)
    p*=np.minimum(1.,gamma/np.maximum(norms,np.finfo(float).tiny))[:,None]
    return field,p,dict(source=str(source),fields_sha256=file_sha256(source/'fields.npz'),
        selection_uses_calibration_or_prediction=False,coarse_gamma=gamma)

def probe():
    destination=OUTPUT/'fine_multigrid_probe';destination.mkdir(exist_ok=False)
    cfg,mesh,cases,_,_=load_prepared(OUTPUT/'exact_tv_fine/01_prepared')
    fit=[c for c in cases if c.name=='380N']
    cfg.MAX_FUN=1800;cfg.EXACT_TV_OPTIONS=dict(cfg.EXACT_TV_OPTIONS,max_outer=1800)
    cfg.EXACT_TV_PRIMAL_STEP=2e6
    rows=[]
    for gamma in (1.1513953993264481e-5,1.757510624854793e-6):
        initial,dual,provenance=warm_start(mesh,gamma)
        result=_optimize(cfg,mesh,fit,gamma,.26,initial,dual)
        h=result['tv_history'][-1]
        row=dict(gamma=gamma,step=2e6,evaluations=result['n_evaluations'],status=result['message'],
            stationarity_ratio=h['stationarity']/h['stationarity_tol'],gap_ratio=abs(h['fenchel_gap'])/h['gap_tol'],
            objective=h['objective'],provenance=provenance)
        rows.append(row);print(json.dumps(row),flush=True)
        np.savez_compressed(destination/f'gamma_{gamma:g}.npz',E=result['E_final'],dual=result['tv_dual'])
        (destination/'summary.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')




def resume():
    """Screen all remaining gammas, then spend budget only on corner support.

    Only verified KKT points enter final selection. Approximate norms schedule
    numerical work; they never provide an accepted gamma or material result.
    """
    from types import SimpleNamespace
    import pandas as pd
    from fgm_asm.l_curve import select_l_curve_corner
    from lib.dic_case_workflow import (_solver_summary,_tv_arrays,_save_tv_evidence,
        _regularization_cost,_finish_dic_inverse,_update_summary)
    from lib.research_reporting import save_run
    root=OUTPUT/'exact_tv_fine';prepared=root/'01_prepared';inverse=root/'03_inverse'
    cfg,mesh,cases,meta,_=load_prepared(prepared)
    fit=[c for c in cases if c.name=='380N']
    summary=json.loads((inverse/'run_metadata.json').read_text(encoding='utf-8'))
    if summary['status']!='running':raise ValueError('Expected an interrupted unfinished fine scan')
    original=pd.read_csv(inverse/'selection.csv',float_precision='round_trip').to_dict(orient='records')
    archive=inverse/'selection_before_multigrid.csv'
    if not archive.exists():
        archive.write_bytes((inverse/'selection.csv').read_bytes())
        (inverse/'metadata_before_multigrid.json').write_bytes((inverse/'run_metadata.json').read_bytes())
    destination=inverse/'multigrid';destination.mkdir(exist_ok=True)
    cfg.EXACT_TV_PRIMAL_STEP=2e7;cfg.__file__=None
    by_gamma={r['gamma']:r for r in original}
    coarse_path=OUTPUT/'exact_tv/03_inverse/selection.csv'
    work=[]
    for saved in sorted(destination.glob('gamma_*/screen/run_metadata.json')):
        record=json.loads(saved.read_text(encoding='utf-8'))
        work.append(dict(gamma=record['selection']['gamma'],phase='screen',additional_evaluations=record['solver']['n_evaluations'],status=record['solver']['message'],directory=str(saved.parent.relative_to(inverse)),source=record['start_provenance']))
    coarse_snapshot=pd.read_csv(coarse_path,float_precision='round_trip')
    coarse_snapshot.to_csv(destination/'coarse_start_scan_snapshot.csv',index=False)
    def obtain_start(gamma):
        table=coarse_snapshot
        qualified=table[table.converged & table.regularization_resolved]
        if qualified.empty:raise ValueError('Need a converged nonconstant coarse 380N field')
        candidate=qualified.iloc[np.argmin(abs(np.log(qualified.gamma.to_numpy()/gamma)))]
        field,dual,provenance=warm_start(mesh,float(candidate.gamma))
        magnitude=np.linalg.norm(dual,axis=1)
        dual*=np.minimum(1.,gamma/np.maximum(magnitude,np.finfo(float).tiny))[:,None]
        return field,dual,provenance
    def compute(gamma,budget,prior=None):
        if prior is None:
            field,dual,provenance=obtain_start(gamma);phase='screen'
            consumed=0
        else:
            with np.load(inverse/prior['solution_directory']/'fields.npz') as z:
                field=z['E_inverse'].copy();dual=z['tv_dual'].copy()
            provenance=dict(source=prior['solution_directory'],phase='resume same gamma from screened state')
            consumed=int(prior['evaluations']);phase='support'
        limited=SimpleNamespace(**vars(cfg))
        limited.MAX_FUN=int(budget);limited.EXACT_TV_OPTIONS=dict(cfg.EXACT_TV_OPTIONS,max_outer=int(budget))
        print(f'Fine {phase}: gamma={gamma:.12g}, extra_budget={budget}, start={provenance}',flush=True)
        result=_optimize(limited,mesh,fit,gamma,.26,field,dual)
        solver=_solver_summary(result)
        record=dict(gamma=gamma,nu=.26,converged=result['converged'],residual_norm=result['residual_norm'],
            regularization_norm=result['regularization_norm'],data_cost=.5*result['residual_norm']**2,
            regularization_cost=_regularization_cost(cfg,gamma,result['regularization_norm']),
            regularization_resolved=result.get('regularization_resolved',True),
            iterations=result['n_iterations']+(int(prior['iterations']) if prior else 0),
            evaluations=result['n_evaluations']+consumed,final_gradient_inf=solver['final_gradient_inf'],
            gradient_tolerance_satisfied=solver['final_gradient_inf']<=cfg.GTOL,stop_message=result['message'],
            initial_source='coarse 380N fit only; independent fine objective and gamma',
            solution_directory=str((destination/f'gamma_{gamma:g}'/phase).relative_to(inverse)))
        place=inverse/record['solution_directory']
        save_run(place,limited,dict(solver=solver,selection=record,start_provenance=provenance,
            fit_cases=['380N'],uses_prediction_cases=False,phase_budget=int(budget)),
            dict(E_inverse=result['E_final'],cost_history=result['cost_history'],
                cost_tar_history=result['cost_tar_history'],cost_reg_history=result['cost_reg_history'],
                gradient_norms=result['gradient_norms'],final_gradient=result['final_gradient'],**_tv_arrays(result)))
        _save_tv_evidence(place,result)
        work.append(dict(gamma=gamma,phase=phase,additional_evaluations=result['n_evaluations'],
            status=result['message'],directory=record['solution_directory'],source=provenance))
        return record
    from concurrent.futures import ThreadPoolExecutor,as_completed
    order=list(sorted(cfg.GAMMA_CANDIDATES,reverse=True))
    records=[None]*len(order);pending=[]
    for index,gamma in enumerate(order):
        old=by_gamma.get(gamma)
        stored=destination/f'gamma_{gamma:g}/screen/run_metadata.json'
        if stored.exists():
            records[index]=json.loads(stored.read_text(encoding='utf-8'))['selection']
        elif old is not None and old['converged']:
            records[index]=old
        else:pending.append((index,gamma))
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures={pool.submit(compute,gamma,600):index for index,gamma in pending}
        for future in as_completed(futures):
            index=futures[future];records[index]=future.result()
            pd.DataFrame([r for r in records if r is not None]).to_csv(inverse/'selection.csv',index=False)
    pd.DataFrame(records).to_csv(inverse/'selection.csv',index=False)
    # Use an approximate curve only to allocate compute, never to accept gamma.
    table=pd.DataFrame(records)
    optimistic=select_l_curve_corner(table.gamma,table.residual_norm,table.regularization_norm,
        table.regularization_resolved & (table.regularization_norm>0))
    support=[]
    if optimistic['optimal_index'] is not None:
        order=np.argsort(table.gamma.to_numpy());position=int(np.flatnonzero(order==optimistic['optimal_index'])[0])
        support=list(map(int,order[max(0,position-2):position+3]))
    else:
        finite=np.flatnonzero(np.isfinite(optimistic['curvature']))
        if len(finite):
            peak=int(finite[np.argmax(optimistic['curvature'][finite])])
            order=np.argsort(table.gamma.to_numpy());position=int(np.flatnonzero(order==peak)[0])
            support=list(map(int,order[max(0,position-2):position+3]))
    with ThreadPoolExecutor(max_workers=5) as pool:
        futures={}
        for index in support:
            previous=records[index]
            if previous['converged']:continue
            remaining=12000-int(previous['evaluations'])
            if remaining>0:futures[pool.submit(compute,previous['gamma'],remaining,previous)]=index
        for future in as_completed(futures):
            records[futures[future]]=future.result()
            pd.DataFrame(records).to_csv(inverse/'selection.csv',index=False)
    protocol=dict(method='coarse-to-fine numerical initialization plus bounded support refinement',
        objective_and_gamma_candidates_unchanged=True,initial_screen_evaluations=600,parallel_screen_workers=6,parallel_support_workers=5,
        support_rule='Approximate finite-norm curve allocates remaining budget to at most five adjacent points; final L-curve still excludes all non-KKT points.',
        support_indices=support,per_point_resumed_phase_max_evaluations=12000,
        primal_step=2e7,coarse_source_uses_only_380N=True,uses_calibration_or_prediction=False,
        original_scan_archive=str(archive),original_trials_preserved=True,
        aborted_trial=json.loads((root/'fine_restart_protocol.json').read_text(encoding='utf-8')),
        work=work,additional_FE_evaluations=sum(r['additional_evaluations'] for r in work),
        same_total_budget_as_coarse=False,source_sha256=file_sha256(Path(__file__)))
    (destination/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    summary['numerical_continuation']=protocol
    summary['scan_initialization']='Original uniform-start trials preserved; remaining fine trials initialized from converged coarse 380N fields, each gamma independently evaluated on the fine mesh.'
    summary['configuration']={k:v for k,v in vars(cfg).items() if k.isupper()}
    _update_summary(inverse,summary)
    result=_finish_dic_inverse(cfg,mesh,cases,meta,summary,records,prepared,inverse)
    print('Fine continuation status:',result['status'],flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['probe','resume'],default='probe',nargs='?')
    args=parser.parse_args();probe() if args.mode=='probe' else resume()
