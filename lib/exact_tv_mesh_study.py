"""Mesh refinement with frozen observations, independent 2D nodal moduli.

Preparation freezes all data and the protocol before any new solves. Each
mesh/kind/method scan is an independently executable, resumable experiment.
No default DIC configuration or finite-element/adjoint kernel is modified.
"""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
from scipy.sparse import save_npz

from fgm_asm.exact_tv import ExactTVOptions, QuadratureGradient, solve_exact_tv
from fgm_asm.l_curve import select_l_curve_corner
from fgm_asm.mesh import MeshInfo
from fgm_asm.observation import build_observation_package
from fgm_asm.research import make_observation_case
from lib.exact_tv_study import (DEFAULT_OUTPUT as BASELINE, dataset, field_metrics,
                                save_json, solve_h1, truth_function)


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT/'results/dic_full/exact_tv_mesh_20261021'
MESHES = {12:(12,4), 24:(24,8), 48:(48,16)}
OPTIONS = ExactTVOptions(max_outer=1200)


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(root):
    root.mkdir(parents=True, exist_ok=False)
    data_dir = root/'data'
    data_dir.mkdir()
    protocol = dict(created_utc=datetime.now(timezone.utc).isoformat(),
        meshes=list(MESHES.values()), nodal_parameters=[65,225,833],
        fixed_truth_forward_mesh=[48,16], fixed_observation_count=833,
        fixed_observation_coordinates=True, fixed_noise_across_meshes=True,
        identification_seed=20261022, validation_seeds=[20261023,20261024],
        noise_std_mm=.0002, kinds=['smooth','step'], nu=.26,
        objectives=dict(h1='0.5*r.T@W_area@r + 0.5*gamma*integral(|grad E|^2)',
                        exact_tv='0.5*r.T@W_area@r + gamma*integral(|grad E|)'),
        gamma_count=50, gamma_ranges=dict(h1=[1e-14,1e-7],exact_tv=[1e-12,1e-6]),
        options=asdict(OPTIONS), independent_y_parameters=True,
        true_interface_not_used_in_inverse=True, initial='same scalar per data sample, fitted on 12x4',
        selection='existing L-curve; only KKT/stationarity-qualified points; no truth-based tuning',
        metrics=['same-833-point area-relative RMS','independent dense quadrature RMS',
                 'midline 20%-80% transition width and crossings','actual evaluations and elapsed time'],
        stage_gate='No real DIC. Failed selection remains a result, never replaced by best truth error.',
        reference='Reuse 12x4 scans only after exact frozen-data identity and objective checks.',
        source_sha256={p:file_hash(PROJECT/p) for p in [
            'fgm_asm/exact_tv.py','lib/exact_tv_study.py','lib/exact_tv_mesh_study.py']})
    save_json(root/'protocol.json', protocol)
    for kind in protocol['kinds']:
        for seed in [protocol['identification_seed'],*protocol['validation_seeds']]:
            mesh, case, initial, _, meta = dataset(kind, seed=seed)
            obs=case.observation
            if seed == protocol['identification_seed']:
                with np.load(BASELINE/f'selection_pilot/{kind}_observations.npz') as old:
                    for key, value in [('coordinates',np.column_stack((obs.x_meas,obs.y_meas))),
                                       ('observed',obs.u_measured),('weights',obs.weight_vector),
                                       ('initial',initial)]:
                        if not np.array_equal(old[key],value):
                            raise ValueError(f'Baseline data identity failed: {kind} {key}')
            destination=data_dir/f'{kind}_{seed}.npz'
            np.savez_compressed(destination, coordinates=np.column_stack((obs.x_meas,obs.y_meas)),
                observed=obs.u_measured, point_weights=obs.point_weights,
                initial_scalar=initial[0], force=case.bc_info['tensile_end_force'])
            save_json(data_dir/f'{kind}_{seed}.json', dict(metadata=meta,sha256=file_hash(destination)))
    print(root/'protocol.json',flush=True)


def fixed_case(root, nx, kind, seed=20261022):
    mesh=MeshInfo(30.,8.,*MESHES[nx])
    with np.load(root/f'data/{kind}_{seed}.npz') as arrays:
        case=make_observation_case(mesh,'180N_synthetic',float(arrays['force']),3.2,
            arrays['coordinates'],arrays['observed'].reshape(-1,2),arrays['point_weights'])
        initial=np.full(mesh.n_nod,float(arrays['initial_scalar']))
    return mesh,case,initial


def solve(mesh, case, initial, method, gamma):
    return (solve_h1(mesh,case,initial,gamma) if method=='h1' else
            solve_exact_tv(mesh,[case],initial,gamma,options=OPTIONS))


def dense_metrics(mesh, field, kind):
    """Independent common 96x32 grid quadrature, not used in selection."""
    evaluation=MeshInfo(30.,8.,96,32)
    nodes=evaluation.ele_nods_id-1
    x=np.einsum('ga,ea->ge',evaluation.gauss_N,evaluation.X[nodes]).ravel()
    y=np.einsum('ga,ea->ge',evaluation.gauss_N,evaluation.Y[nodes]).ravel()
    weights=(evaluation.gauss_abs_det_j*evaluation.gauss_w[:,None]).ravel()
    scalar=build_observation_package(mesh,x,y,np.zeros_like(x),np.zeros_like(y),
                                    weights,sampling_mode='physical_points').H[::2,::2]
    reconstructed=np.asarray(scalar@field).ravel()
    truth=truth_function(kind)(x,y)
    relative=float(np.sqrt(weights@(reconstructed-truth)**2/(weights@truth**2)))
    xx=np.linspace(0,30,1201)
    line=build_observation_package(mesh,xx,np.full_like(xx,4.),np.zeros_like(xx),
        np.zeros_like(xx),np.ones_like(xx),sampling_mode='physical_points').H[::2,::2]@field
    result=dict(dense_area_relative_rms=relative)
    if kind=='step':
        locations=[]
        for level in (2200.,1600.):
            intersections=[]
            for i in np.flatnonzero((line[:-1]-level)*(line[1:]-level)<0):
                intersections.append(float(xx[i]+(level-line[i])*(xx[i+1]-xx[i])/(line[i+1]-line[i])))
            locations.append(intersections)
        result['transition_crossings']=locations
        result['midline_transition_width_mm']=(locations[1][0]-locations[0][0]
            if all(len(v)==1 for v in locations) and locations[1][0]>=locations[0][0] else None)
    return result


def scan(root, nx, kind, method):
    protocol=read_json(root/'protocol.json')
    directory=root/f'{nx}x{MESHES[nx][1]}/{kind}_{method}'
    directory.mkdir(parents=True,exist_ok=True)
    mesh,case,initial=fixed_case(root,nx,kind)
    op=QuadratureGradient(mesh)
    gamma=np.geomspace(*protocol['gamma_ranges'][method],protocol['gamma_count'])
    if nx==12 and not (directory/'scan.json').exists():
        source=BASELINE/f'selection_pilot/{kind}_{method}'
        for name in ('scan.json','fields.npz'):
            shutil.copy2(source/name,directory/name)
        for path in source.glob('history_*.json'):
            shutil.copy2(path,directory/path.name)
        save_json(directory/'reuse.json',dict(source=str(source),
            original_scan_sha256=file_hash(source/'scan.json'), frozen_data_identity=True))
    rows=read_json(directory/'scan.json') if (directory/'scan.json').exists() else []
    fields=[]
    if rows:
        with np.load(directory/'fields.npz') as previous:
            fields=list(previous['E'])
        if len(fields)!=len(rows):
            raise ValueError('Saved fields and scan rows disagree')
    # A larger outer cap cannot change a solve that already stopped by KKT.
    # Recompute old budget failures rather than inheriting the smaller cap.
    pending=[i for i,r in enumerate(rows) if not r['converged']]
    pending.extend(range(len(rows),len(gamma)))
    for index in pending:
        weight=float(gamma[index])
        start=time.perf_counter()
        result=solve(mesh,case,initial,method,weight)
        if method=='exact_tv':
            reg=op.value(result['E']); eta=reg
            resolved=op.rms(op.apply(result['E']))>1e-5
            save_json(directory/f'history_{index:02d}.json',result['history'])
        else:
            reg=result['regularization']; eta=np.sqrt(reg); resolved=True
        row=dict(index=index,gamma=weight,method=method,kind=kind,nx=nx,
            status=result['status'],converged=result['converged'],norm_resolved=bool(resolved),
            evaluations=result['evaluations'],elapsed_s=time.perf_counter()-start,
            rho=float(np.sqrt(2*result['state']['cost_tar'])),eta=float(eta),regularization=float(reg))
        if method=='exact_tv': row['certificate']=result['history'][-1]
        else: row['stationarity']=result['stationarity']
        if index < len(rows):
            rows[index]=row;fields[index]=result['E']
        else:
            rows.append(row);fields.append(result['E'])
        np.savez_compressed(directory/'fields.npz',E=np.asarray(fields),initial=initial,coordinates=mesh.coord)
        save_json(directory/'scan.json',rows)
        print(nx,kind,method,index,result['status'],result['evaluations'],round(row['elapsed_s'],2),flush=True)
    selection=select_l_curve_corner(gamma,[r['rho'] for r in rows],[r['eta'] for r in rows],
                                   [r['converged'] and r['norm_resolved'] for r in rows])
    save_json(directory/'selection.json',selection)
    selected=dict(nx=nx,ny=MESHES[nx][1],parameters=mesh.n_nod,kind=kind,method=method,
                  status=selection['status'],selected_gamma=selection['selected_gamma'],
                  converged_points=sum(r['converged'] for r in rows),eligible_points=int(np.sum(selection['eligible'])))
    if selection['selected_gamma'] is not None:
        index=selection['optimal_index']
        selected.update(field_metrics(mesh,case,fields[index],kind))
        selected.update(dense_metrics(mesh,fields[index],kind))
        selected.update(evaluations=rows[index]['evaluations'],elapsed_s=rows[index]['elapsed_s'])
    save_json(directory/'selected_metrics.json',selected)
    print('SELECTED',selected,flush=True)


def validate(root, nx, kind, method):
    directory=root/f'{nx}x{MESHES[nx][1]}/{kind}_{method}'
    selected=read_json(directory/'selected_metrics.json')
    if selected['status']!='selected':
        print('No valid gamma; validation skipped.',flush=True)
        return
    rows=[]
    for seed in read_json(root/'protocol.json')['validation_seeds']:
        path=directory/f'validation_{seed}.json'
        if path.exists():
            rows.append(read_json(path));continue
        mesh,case,initial=fixed_case(root,nx,kind,seed)
        start=time.perf_counter()
        result=solve(mesh,case,initial,method,selected['selected_gamma'])
        row=dict(nx=nx,ny=MESHES[nx][1],kind=kind,method=method,seed=seed,
            gamma=selected['selected_gamma'],converged=result['converged'],status=result['status'],
            evaluations=result['evaluations'],elapsed_s=time.perf_counter()-start,
            **field_metrics(mesh,case,result['E'],kind),**dense_metrics(mesh,result['E'],kind))
        np.savez_compressed(directory/f'validation_{seed}.npz',E=result['E'],coordinates=mesh.coord)
        if method=='exact_tv': save_json(directory/f'validation_{seed}_history.json',result['history'])
        save_json(path,row); rows.append(row)
        print('VALIDATION',row,flush=True)
    save_json(directory/'validation_metrics.json',rows)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','scan','validate'])
    parser.add_argument('--root',type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument('--nx',type=int,choices=MESHES,default=12)
    parser.add_argument('--kind',choices=['smooth','step'],default='step')
    parser.add_argument('--method',choices=['h1','exact_tv'],default='exact_tv')
    args=parser.parse_args()
    if args.action=='prepare':prepare(args.root.resolve())
    else:(scan if args.action=='scan' else validate)(args.root.resolve(),args.nx,args.kind,args.method)
