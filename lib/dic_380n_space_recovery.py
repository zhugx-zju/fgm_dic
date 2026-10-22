"""Recover abnormal line searches of the matched 180N space controls only."""
import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from lib.dic_380n_spaces import inputs,regional_means,p
from lib.experiment_data import file_sha256

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/'results/dic_full/load380_20261022'

def recover(method):
    original=OUTPUT/'spaces_180N_control'/method
    destination=OUTPUT/'spaces_180N_recovered'/method
    destination.mkdir(parents=True,exist_ok=False)
    mesh=p.MeshInfo(p.L,p.H,42,8);data,truth,meta,arrays=inputs(mesh)
    model=p.Model(mesh,method,data['180N'].bc)
    rows=pd.read_csv(original/'scan.csv',float_precision='round_trip').to_dict(orient='records')
    repaired={};restarts=[]
    for j,row in enumerate(rows):
        if row['success'] or 'ABNORMAL' not in row['message'].upper():continue
        E=np.load(original/f'gamma_{j:02d}.npz')['E']
        for attempt in (1,2):
            rem_iter=1500-int(row['iterations']);rem_fun=12000-int(row['evaluations'])
            if min(rem_iter,rem_fun)<=0:break
            before=dict(row);cache={}
            def objective(m):
                st=model.value_gradient(m,data['180N'],row['gamma']);cache['x']=m.copy();cache['state']=st
                return st['value'],st['gradient']
            opt=minimize(objective,np.log(E),jac=True,method='L-BFGS-B',options=dict(
                maxiter=rem_iter,maxfun=rem_fun,ftol=1e-30,gtol=1e-12,maxls=50,maxcor=10))
            st=cache['state'] if np.array_equal(cache['x'],opt.x) else model.value_gradient(opt.x,data['180N'],row['gamma'])
            dest=destination/f'gamma_{j:02d}_recovery_{attempt}.npz'
            np.savez_compressed(dest,E=st['E'],u=st['u'],gradient=st['gradient'])
            row.update(rho=float(np.sqrt(2*st['data_cost'])),eta=float(np.sqrt(max(st['reg'],0))),
                success=bool(opt.success),gtol=bool(abs(st['gradient']).max()<=1e-12),
                gradient_inf=float(abs(st['gradient']).max()),iterations=before['iterations']+int(opt.nit),
                evaluations=before['evaluations']+int(opt.nfev),message=str(opt.message))
            repaired[j]=dest;E=st['E'];restarts.append(dict(index=j,attempt=attempt,before=before,after=dict(row)))
            if opt.success or 'ABNORMAL' not in row['message'].upper():break
    p.write_csv(destination/'scan_after_recovery.csv',rows)
    p.save_json(destination/'line_search_restarts.json',restarts)
    selection=p.select_l_curve_corner([r['gamma'] for r in rows],[r['rho'] for r in rows],
        [r['eta'] for r in rows],[r['success'] for r in rows])
    p.save_json(destination/'selection.json',selection)
    row=dict(method=method,fit_only='180N',gamma_count=50,nu=.26,
        gamma=selection['selected_gamma'],status=selection['status'],source_sha256=file_sha256(Path(__file__)),
        original_scan_sha256=file_sha256(original/'scan.csv'),
        recovery='At most two ABNORMAL restarts inside original remaining iteration/function budgets; same policy as 380N.',
        selection_uses_380N_or_280N=False)
    if selection['optimal_index'] is not None:
        j=selection['optimal_index'];source=repaired.get(j,original/f'gamma_{j:02d}.npz')
        with np.load(source) as z:E=z['E'].copy();gradient=z['gradient'].copy()
        means=regional_means(mesh,E,method)
        row.update(regional_means_MPa=means.tolist(),
            regional_mean_absolute_difference_percent=float(np.mean(abs(means/truth-1))*100),
            selected_gradient_inf=float(abs(gradient).max()),selected_gtol=bool(abs(gradient).max()<=1e-12),metrics={})
        fields=dict(E=E,gradient=gradient,coordinates=mesh.coord)
        for name,case in data.items():
            native=p.Model(mesh,method,case.bc);u,_,_=native.forward(E)
            _,_,res,c=case.evaluate(u)
            row['metrics'][name]=dict(native_rmse_um=(np.sqrt(np.mean(res.reshape(-1,2)**2,axis=0))*1000).tolist())
            fields[name+'_prediction']=case.H@u+case.R@c
        np.savez_compressed(destination/'selected.npz',**fields)
    p.save_json(destination/'summary.json',row)
    print(json.dumps(row,default=p.plain),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('method',choices=p.METHODS);recover(parser.parse_args().method)
