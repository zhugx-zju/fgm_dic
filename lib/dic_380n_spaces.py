"""Repeat the five existing material/test-space probes on 380N only.

The old verified implementation is imported unchanged; production mechanics
remain untouched. Each method gets its own 50-point fit-only L-curve.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import time
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from lib.dic_case_workflow import load_prepared
from lib.experiment_data import file_sha256

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/'results/dic_full/load380_20261022/spaces'
LEGACY=ROOT/'lib/material_test_spaces.py'
spec=importlib.util.spec_from_file_location('material_spaces_legacy',LEGACY)
p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)
p.OUT=OUTPUT
PREPARED=OUTPUT.parent/'h1/01_prepared'

def inputs(mesh):
    cfg,_,_,metadata,_=load_prepared(PREPARED)
    with np.load(PREPARED/'fields.npz') as saved:
        values={k:saved[k].copy() for k in saved.files}
    assert abs(cfg.GEO_L-p.L)<1e-7 and abs(cfg.GEO_H-p.H)<1e-7
    data={name:p.Data(mesh,values[name+'_coordinates'],values[name+'_observed'],
            values[name+'_point_area_mm2'],metadata['force_pairing'][name]['mean_corrected_force_N'])
          for name in ('380N','180N','280N')}
    truth=np.array([r['mean_E_MPa'] for r in metadata['calibration']])
    return data,truth,metadata,values

def profile(mesh,E,method):
    if method.startswith('P0'):
        return E.reshape(mesh.nel_y,mesh.nel_x).mean(0)
    return np.trapezoid(E.reshape(mesh.nods_y,mesh.nods_x),dx=mesh.eh,axis=0)/p.H

def regional_means(mesh,E,method):
    f=profile(mesh,E,method);edges=np.linspace(0,p.L,mesh.nel_x+1)
    limits=np.r_[0,p.BOUNDARIES,p.L];result=[]
    for a,b in zip(limits[:-1],limits[1:]):
        if method.startswith('P0'):
            overlap=np.maximum(0,np.minimum(edges[1:],b)-np.maximum(edges[:-1],a))
            result.append(float(overlap@f/(b-a)))
        else:
            x=np.r_[a,edges[(edges>a)&(edges<b)],b]
            result.append(float(np.trapezoid(np.interp(x,edges,f),x)/(b-a)))
    return np.array(result)

def check():
    OUTPUT.mkdir(exist_ok=False)
    p.gradient_checks()
    mesh=p.MeshInfo(p.L,p.H,8,3)
    datasets,truth,meta,values=inputs(mesh)
    rng=np.random.default_rng(20261022);rows=[]
    for method in p.METHODS:
        model=p.Model(mesh,method,datasets['380N'].bc)
        m=np.log(1800+100*rng.normal(size=model.n))
        d=rng.normal(size=model.n);d/=np.linalg.norm(d)
        gamma=1e-8 if '_H1' in method else 1e-6
        state=model.value_gradient(m,datasets['380N'],gamma)
        analytical=float(state['gradient']@d);errors=[]
        for h in (1e-3,1e-4,1e-5):
            fd=(model.value_gradient(m+h*d,datasets['380N'],gamma)['value']-
                model.value_gradient(m-h*d,datasets['380N'],gamma)['value'])/(2*h)
            errors.append(abs(fd-analytical)/max(abs(fd),abs(analytical),1e-20))
        assert min(errors)<2e-5,(method,errors)
        assert np.max(abs(regional_means(mesh,np.full(model.n,1800.),method)-1800))<1e-10
        rows.append(dict(method=method,directional_relative_errors=errors))
    p.save_json(OUTPUT/'380N_gradient_checks.json',dict(passed=True,checks=rows))
    p.save_json(OUTPUT/'protocol.json',dict(fit_only='380N',prediction_only=['180N','280N'],
        force_N=319.9,force_is_assumed=True,nu=.26,nx=42,ny=8,gamma_count=50,
        maxiter=1500,maxfun=12000,ftol=1e-30,gtol=1e-12,
        line_search_recovery='At most two restarts of ABNORMAL points, within original remaining iterations and function budget. Original trials preserved.',
        initialization='Each method fits a uniform E from 380N; ascending gamma continuation from converged trials, matching the old space probe.',
        calibration_used_for_initialization_or_selection=False,known_interfaces_in_inverse=False,
        legacy_source=str(LEGACY),legacy_source_sha256=file_sha256(LEGACY),
        new_source_sha256=file_sha256(Path(__file__)),prepared_sha256=file_sha256(PREPARED/'fields.npz'),
        source_inputs_sha256={r['path']:r['sha256'] for v in meta['platforms'].values() for r in v['sources']}))
    print('All five 380N gradients and uniform-region integration checks passed.',flush=True)

def run(method):
    assert (OUTPUT/'380N_gradient_checks.json').exists(),'Run check first'
    mesh=p.MeshInfo(p.L,p.H,42,8);datasets,truth,meta,values=inputs(mesh)
    data=datasets['380N'];model=p.Model(mesh,method,data.bc)
    folder=OUTPUT/method
    selection=p.scan(model,data,folder,50)
    scan=pd.read_csv(folder/'scan.csv',float_precision='round_trip').to_dict(orient='records')
    repaired={};restarts=[]
    for j,row in enumerate(scan):
        if row['success'] or 'ABNORMAL' not in row['message'].upper():continue
        E=np.load(folder/f'gamma_{j:02d}.npz')['E']
        for attempt in (1,2):
            rem_iter=1500-int(row['iterations']);rem_fun=12000-int(row['evaluations'])
            if min(rem_iter,rem_fun)<=0:break
            before=dict(row);cache={}
            def objective(m):
                st=model.value_gradient(m,data,row['gamma']);cache['x']=m.copy();cache['state']=st
                return st['value'],st['gradient']
            opt=minimize(objective,np.log(E),jac=True,method='L-BFGS-B',options=dict(
                maxiter=rem_iter,maxfun=rem_fun,ftol=1e-30,gtol=1e-12,maxls=50,maxcor=10))
            st=cache['state'] if np.array_equal(cache['x'],opt.x) else model.value_gradient(opt.x,data,row['gamma'])
            dest=folder/f'gamma_{j:02d}_recovery_{attempt}.npz'
            np.savez_compressed(dest,E=st['E'],u=st['u'],gradient=st['gradient'])
            row.update(rho=float(np.sqrt(2*st['data_cost'])),eta=float(np.sqrt(max(st['reg'],0))),
                success=bool(opt.success),gtol=bool(abs(st['gradient']).max()<=1e-12),
                gradient_inf=float(abs(st['gradient']).max()),iterations=before['iterations']+int(opt.nit),
                evaluations=before['evaluations']+int(opt.nfev),message=str(opt.message))
            repaired[j]=dest;E=st['E'];restarts.append(dict(index=j,attempt=attempt,before=before,after=dict(row)))
            if opt.success or 'ABNORMAL' not in row['message'].upper():break
    p.write_csv(folder/'scan_after_recovery.csv',scan)
    p.save_json(folder/'line_search_restarts.json',restarts)
    selection=p.select_l_curve_corner([r['gamma'] for r in scan],[r['rho'] for r in scan],
        [r['eta'] for r in scan],[r['success'] for r in scan])
    p.save_json(folder/'selection_after_recovery.json',selection)
    summary=dict(method=method,status=selection['status'],gamma=selection['selected_gamma'],
        mesh=[42,8],n_parameters=model.n,successful_trials=sum(r['success'] for r in scan),
        gtol_trials=sum(r['gtol'] for r in scan),count=50,fit_only='380N',force_assumed=True)
    if selection['optimal_index'] is not None:
        j=selection['optimal_index'];chosen=repaired.get(j,folder/f'gamma_{j:02d}.npz')
        with np.load(chosen) as z:E=z['E'].copy();gradient=z['gradient'].copy()
        means=regional_means(mesh,E,method)
        summary.update(regional_means_MPa=means.tolist(),
            regional_mean_absolute_difference_percent=float(np.mean(abs(means/truth-1))*100),
            selected_gradient_inf=float(abs(gradient).max()),selected_gtol=bool(abs(gradient).max()<=1e-12),
            selected_stop_message=scan[j]['message'],selected_source=str(chosen),metrics={})
        arrays=dict(E=E,gradient=gradient,coordinates=mesh.coord)
        for name,other_data in datasets.items():
            native=p.Model(mesh,method,other_data.bc)
            u,_,_=native.forward(E);_,_,r,c=other_data.evaluate(u)
            common=p.Model(mesh,'P0_TV' if model.p0 else 'Q1_H1',other_data.bc)
            uc,_,_=common.forward(E);_,_,rc,cc=other_data.evaluate(uc)
            calibrated=truth[np.searchsorted(p.BOUNDARIES,native.xg,side='right')]
            uf,_,_=native.forward(gauss=calibrated);_,_,rf,cf=other_data.evaluate(uf)
            def rms(res):return (np.sqrt(np.average(res.reshape(-1,2)**2,weights=other_data.w[::2],axis=0))*1000).tolist()
            summary['metrics'][name]=dict(native_rmse_um=rms(r),common_galerkin_rmse_um=rms(rc),
                independent_calibration_rmse_um=rms(rf),role='fit' if name=='380N' else 'cross_load_diagnostic')
            arrays.update({name+'_observed':other_data.d,name+'_coordinates':values[name+'_coordinates'],
                name+'_prediction':other_data.H@u+other_data.R@c,name+'_residual':r,
                name+'_common_prediction':other_data.H@uc+other_data.R@cc,
                name+'_forward_prediction':other_data.H@uf+other_data.R@cf})
        np.savez_compressed(folder/'selected.npz',**arrays)
    p.save_json(folder/'summary.json',summary)
    print(json.dumps(summary,default=p.plain),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('method',choices=['check',*p.METHODS]);args=parser.parse_args()
    check() if args.method=='check' else run(args.method)
