"""Matched 50-point 180N control for the existing small-mesh space probes."""
import argparse
import json
from pathlib import Path
import numpy as np
from lib.dic_380n_spaces import inputs,regional_means,p
from lib.experiment_data import file_sha256

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/'results/dic_full/load380_20261022/spaces_180N_control'

def run(method):
    OUTPUT.mkdir(exist_ok=True)
    mesh=p.MeshInfo(p.L,p.H,42,8);data,truth,meta,arrays=inputs(mesh)
    model=p.Model(mesh,method,data['180N'].bc)
    folder=OUTPUT/method;selection=p.scan(model,data['180N'],folder,50)
    row=dict(method=method,fit_only='180N',gamma_count=50,nu=.26,
        gamma=selection['selected_gamma'],status=selection['status'],
        source_sha256=file_sha256(Path(__file__)),legacy_source_sha256=p.sha(p.ROOT/'lib/material_test_spaces.py'),
        selection_uses_380N_or_280N=False)
    if selection['optimal_index'] is not None:
        source=folder/f'gamma_{selection["optimal_index"]:02d}.npz'
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
        np.savez_compressed(folder/'selected.npz',**fields)
    p.save_json(folder/'summary.json',row)
    print(json.dumps(row,default=p.plain),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('method',choices=p.METHODS)
    run(parser.parse_args().method)
