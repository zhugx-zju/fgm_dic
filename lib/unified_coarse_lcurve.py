"""Independent coarse simulation stages with intra-region H1 and L-curve.

No experimental inversion is started by this module. A saved simulation gate
must pass before a separate experimental protocol is prepared.
"""
from __future__ import annotations
import argparse
import gc
import hashlib
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np

from fgm_asm.fixed_region_material import FixedRegionMaterialMap
from fgm_asm.fem_forward import fem_assemble, forward_solver
from fgm_asm.integration import rectangular_quadrature
from fgm_asm.l_curve import select_l_curve_corner
from fgm_asm.load_case import ObservationCase, evaluate_observation
from fgm_asm.material import MaterialInfo
from fgm_asm.mesh import MeshInfo, setup_boundary_conditions
from fgm_asm.observation import build_observation_package, build_observation_operator, compute_axis_cell_widths
from fgm_asm.unified_inverse import fit_uniform_initial_modulus, solve_unified
from fgm_asm.unified_material import RegionMaterialMap
from .config_utils import load_python_configuration, configuration_fingerprint
from .dic_preprocessing import plot_grid_fields
from .research_reporting import save_run, plot_l_curve
from .unified_gradient_layered_study import read_json, write_json, write_csv

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ('fgm_asm/mesh.py','fgm_asm/material.py','fgm_asm/fem_forward.py',
           'fgm_asm/integration.py','fgm_asm/unified_material.py','fgm_asm/fixed_region_material.py',
           'fgm_asm/unified_inverse.py','fgm_asm/inverse_solver.py','fgm_asm/load_case.py',
           'fgm_asm/observation.py','fgm_asm/l_curve.py','lib/unified_coarse_lcurve.py')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def hashes():
    return {p:digest(ROOT/p) for p in SOURCES}


def prepare(cfg):
    output = Path(cfg.CASE_OUTPUT)/'01_prepared'
    if Path(cfg.CASE_OUTPUT).exists():
        raise FileExistsError('Run directory exists; use a new RUN_NAME')
    source = Path(cfg.REFERENCE_PREPARED)
    metadata = read_json(source/'run_metadata.json')
    cfg.GEO_L,cfg.GEO_H,cfg.THICKNESS = metadata['physical_model_mm']
    cfg.BOUNDARIES = metadata['configuration']['REGION_BOUNDARIES_X']
    cfg.BEARING_WIDTH = metadata['domain_geometry']['load_section_width_mm']
    cfg.FORCE = metadata['force_pairing']['180N']['mean_corrected_force_N']
    requested = load_python_configuration(cfg.__file__)
    with np.load(source/'fields.npz') as fields:
        arrays = dict(xy=fields['180N_coordinates'],areas=fields['180N_point_area_mm2'])
    save_run(output,cfg,dict(status='prepared',scope='coarse simulation L-curve, fixed known interfaces',
             requested_configuration_sha256=configuration_fingerprint(requested),
             kernel_sha256=hashes(),geometry_source_sha256=digest(source/'run_metadata.json'),
             coordinate_source_sha256=digest(source/'fields.npz'),
             truth_used_in_initialization=False,truth_used_in_selection=False),arrays)
    write_json(output/'inputs.json',dict(fields_sha256=digest(output/'fields.npz')))
    publish(Path(cfg.CASE_OUTPUT))


def load(config):
    requested=load_python_configuration(config)
    root=Path(requested.CASE_OUTPUT)
    meta=read_json(root/'01_prepared/run_metadata.json')
    if configuration_fingerprint(requested)!=meta['requested_configuration_sha256'] or hashes()!=meta['kernel_sha256']:
        raise ValueError('Configuration/kernel differs from the frozen preparation')
    if digest(root/'01_prepared/fields.npz')!=read_json(root/'01_prepared/inputs.json')['fields_sha256']:
        raise ValueError('Frozen coordinates/weights identity mismatch')
    return root,SimpleNamespace(**meta['configuration'])


def mesh(cfg,refinement=1,truth=False,kind=None):
    m=MeshInfo(cfg.GEO_L,cfg.GEO_H,cfg.ELEMENTS[0]*refinement,cfg.ELEMENTS[1]*refinement)
    cuts=tuple(cfg.BOUNDARIES) if truth and kind!='continuous_2d' else ()
    return rectangular_quadrature(m,3,1 if truth else cfg.QUADRATURE_SUBDIVISIONS,cuts)


def bc(m,cfg):
    return setup_boundary_conditions(m,cfg.GEO_L,cfg.GEO_H,cfg.FORCE*cfg.GEO_H/cfg.BEARING_WIDTH,cfg.THICKNESS)


def truth_function(cfg,kind):
    boundaries=np.array(cfg.BOUNDARIES)
    values=np.linspace(2400,1200,7)
    def field(x,y):
        xi,eta=x/cfg.GEO_L,y/cfg.GEO_H
        if kind=='continuous_2d':
            return 2350-1000*xi+140*(eta-.5)+80*np.sin(2*np.pi*xi)*np.sin(np.pi*eta)
        region=np.searchsorted(boundaries,x,side='right')
        if kind=='layered':return values[region]
        return values[region]+180*(xi-.5)+140*(eta-.5)+60*np.sin(2*np.pi*xi)*np.sin(np.pi*eta)
    return field


def case_from_arrays(m,cfg,xy,data,areas,name):
    obs=build_observation_package(m,xy[:,0],xy[:,1],data[0::2],data[1::2],areas)
    return ObservationCase(name,bc(m,cfg),obs)


def forward(root,cfg,kind):
    out=root/'02_forward'/kind
    out.mkdir(parents=True,exist_ok=False)
    with np.load(root/'01_prepared/fields.npz') as f:xy,areas=f['xy'],f['areas']
    responses=[]
    for factor in (cfg.TRUTH_REFINEMENT,cfg.TRUTH_CHECK_REFINEMENT):
        m=mesh(cfg,factor,truth=True,kind=kind)
        material=MaterialInfo(cfg.NU,modulus_function=truth_function(cfg,kind))
        fem=fem_assemble(m,material,bc(m,cfg));u=forward_solver(fem)
        responses.append(np.asarray(build_observation_operator(m,xy[:,0],xy[:,1])@u))
        del fem,m,material,u
        gc.collect()
    delta=(responses[0]-responses[1]).reshape(-1,2)
    seed=cfg.SEED+list(cfg.CASES).index(kind)
    rng=np.random.default_rng(seed)
    observed=responses[0]+rng.normal(0,cfg.NOISE_STD_MM,len(responses[0]))
    validation=responses[0]+np.random.default_rng(cfg.VALIDATION_SEED+list(cfg.CASES).index(kind)).normal(0,cfg.NOISE_STD_MM,len(observed))
    np.savez_compressed(out/'observations.npz',xy=xy,areas=areas,data=observed,validation_data=validation)
    truth_dir=out/'truth';truth_dir.mkdir()
    np.savez_compressed(truth_dir/'reference.npz',clean_displacement=responses[0],refined_displacement=responses[1])
    summary=dict(case=kind,source_sha256=hashes(),observation_sha256=digest(out/'observations.npz'),
                 seed=seed,noise_std_mm=cfg.NOISE_STD_MM,
                 refinement_component_rmse_mm=np.sqrt((areas[:,None]*delta**2).sum(axis=0)/areas.sum()),
                 forward_evaluations=2,independent_truth_mesh=True)
    write_json(out/'summary.json',summary)
    print('Forward',kind,summary,flush=True)
    plot_observations(out,observed,xy,cfg)
    publish(root)


def plot_observations(out,observed,xy,cfg):
    from .dic_preprocessing import grid_fields_from_points
    X,Y,fields=grid_fields_from_points(xy,dict(ux=observed[0::2],uy=observed[1::2]))
    plot_grid_fields(out,'synthetic',X,Y,[(k,v,k) for k,v in fields.items()],
                     cmap='jet',unit='mm',title_prefix='Synthetic ',bounds=(0,cfg.GEO_L,0,cfg.GEO_H))


def model(m,cfg,kind='q4'):
    return FixedRegionMaterialMap(m,n_regions=cfg.N_REGIONS,branch_kind=kind,
            interfaces=np.array(cfg.BOUNDARIES)[:,None],epsilon=cfg.EPSILON_MM)


def solve(p,case,cfg,initial,gamma,theta=None,label=''):
    def progress(row):
        print(f'{label}: FE={row["evaluation"]}, J={row["value"]:.6g}, '
              f'|g|inf={row["gradient_inf"]:.3g}',flush=True)
    return solve_unified(p,case,initial,nu=cfg.NU,gamma=gamma,
             modulus_bounds=cfg.MODULUS_BOUNDS_MPA,max_iterations=cfg.MAX_ITERATIONS,
             max_evaluations=cfg.MAX_EVALUATIONS,ftol=cfg.FTOL,gtol=cfg.GTOL,
             initial_parameters=theta,progress=progress)


def initial_from_data(m,cfg,case,out):
    p0=model(m,cfg,'constant')
    initial=fit_uniform_initial_modulus(p0,case,cfg.NU)
    state,evidence,history=solve(p0,case,cfg,initial,0,label='constant-region initialization')
    write_json(out/'initialization.json',dict(uniform_modulus_MPa=initial,source='observations only',
               subspace='region constants, no independent material calibration',**evidence))
    write_csv(out/'initialization_history.csv',history)
    np.savez_compressed(out/'initialization.npz',theta=state['theta'])
    theta=np.repeat(state['theta'],m.n_nod)
    return initial,theta,evidence


def inverse(root,cfg,kind):
    source=root/'02_forward'/kind
    meta=read_json(source/'summary.json')
    if meta['source_sha256']!=hashes() or digest(source/'observations.npz')!=meta['observation_sha256']:
        raise ValueError('Simulation source identity mismatch')
    out=root/'03_inverse'/kind
    out.mkdir(parents=True,exist_ok=True)
    if (out/'selection.json').exists():raise FileExistsError('Completed scan exists')
    m=mesh(cfg);p=model(m,cfg)
    with np.load(source/'observations.npz') as f:case=case_from_arrays(m,cfg,f['xy'],f['data'],f['areas'],kind)
    if (out/'initialization.json').exists():
        init=read_json(out/'initialization.json');initial=init['uniform_modulus_MPa']
        with np.load(out/'initialization.npz') as f:theta=np.repeat(f['theta'],m.n_nod)
    else:initial,theta,_=initial_from_data(m,cfg,case,out)
    rows=[];initial_source='data-fitted region constant subspace'
    start=time.perf_counter()
    for index,gamma in enumerate(sorted(cfg.GAMMA_CANDIDATES,reverse=True)):
        point=out/f'gamma_{index:02d}'
        if (point/'summary.json').exists():
            record=read_json(point/'summary.json')
            if record['gamma']!=gamma:raise ValueError('Scan gamma identity mismatch')
        else:
            point.mkdir(exist_ok=True)
            print(f'{kind}: L-curve {index+1}/50 gamma={gamma:.12g}',flush=True)
            state,evidence,history=solve(p,case,cfg,initial,gamma,theta,label=f'{kind}/{index:02d}')
            np.savez_compressed(point/'fields.npz',theta=state['theta'],U=state['U'],
                                prediction=state['prediction'],residual=state['residual'],gradient=state['gradient'])
            write_csv(point/'history.csv',history)
            record=dict(gamma=gamma,residual_norm=float(np.sqrt(2*state['data_cost'])),
                  regularization_norm=float(np.sqrt(state['regularization'])),
                  converged=evidence['first_order_passed'],initial_source=initial_source,**evidence)
            write_json(point/'summary.json',record)
        rows.append(record)
        if record['converged']:
            with np.load(point/'fields.npz') as saved:theta=saved['theta'].copy()
            initial_source=point.name
        write_csv(out/'scan.csv',rows)
    selection=select_l_curve_corner([r['gamma'] for r in rows],[r['residual_norm'] for r in rows],
                     [r['regularization_norm'] for r in rows],[r['converged'] for r in rows])
    for i,row in enumerate(rows):
        row['curvature']=float(selection['curvature'][i]) if np.isfinite(selection['curvature'][i]) else None
        row['selected']=i==selection['optimal_index']
    write_csv(out/'scan.csv',rows)
    record={k:v for k,v in selection.items() if k not in ('curvature','eligible')}
    record.update(case=kind,valid_points=sum(r['converged'] for r in rows),
                  forward_evaluations=sum(r['forward_evaluations'] for r in rows),
                  elapsed_seconds=time.perf_counter()-start,source_sha256=hashes())
    write_json(out/'selection.json',record)
    plot_l_curve(out,rows)
    print('Selection',record,flush=True)
    publish(root)


def validate_selected(root,cfg,kind):
    """Fixed gamma robustness and unknown-geometry diagnostic, no reselection."""
    folder=root/'03_inverse'/kind
    selection=read_json(folder/'selection.json')
    if selection['selected_gamma'] is None:raise ValueError('No selected gamma to validate')
    out=folder/'validation';out.mkdir(exist_ok=False)
    m=mesh(cfg);p=model(m,cfg)
    with np.load(folder/f'gamma_{selection["optimal_index"]:02d}'/'fields.npz') as saved:
        selected=saved['theta']
    with np.load(root/'02_forward'/kind/'observations.npz') as saved:
        xy,areas=saved['xy'],saved['areas']
        case=case_from_arrays(m,cfg,xy,saved['data'],areas,kind)
        perturbed=case_from_arrays(m,cfg,xy,saved['validation_data'],areas,kind+'_validation_noise')
    gamma=selection['selected_gamma']
    initial=read_json(folder/'initialization.json')['uniform_modulus_MPa']
    records=[]
    # Same observations, fresh data-based start instead of L-curve continuation.
    constants=model(m,cfg,'constant')
    sub,sub_evidence,sub_history=solve(constants,case,cfg,initial*1.2,0,label=kind+'/multistart constants')
    write_csv(out/'multistart_initialization_history.csv',sub_history)
    theta0=np.repeat(sub['theta'],m.n_nod)
    for label,current,start in [('multistart',case,theta0),('independent_noise',perturbed,selected)]:
        state,evidence,history=solve(p,current,cfg,initial,gamma,start,label=kind+'/'+label)
        np.savez_compressed(out/f'{label}.npz',theta=state['theta'])
        write_csv(out/f'{label}_history.csv',history)
        records.append(dict(label=label,**evidence))
    spacing=cfg.GEO_L/cfg.N_REGIONS
    centers=np.arange(1,cfg.N_REGIONS)*spacing
    limits=np.column_stack([centers-.4*spacing,centers+.4*spacing])
    for shift in cfg.UNKNOWN_START_SHIFTS:
        positions=(centers+shift*spacing)[:,None]
        constant_unknown=RegionMaterialMap(m,n_regions=cfg.N_REGIONS,branch_kind='constant',
                         interfaces=positions,epsilon=cfg.EPSILON_MM,optimize_geometry=True)
        uniform=fit_uniform_initial_modulus(constant_unknown,case,cfg.NU)
        sub,sub_evidence,sub_history=solve_unified(constant_unknown,case,uniform,nu=cfg.NU,gamma=0,
                modulus_bounds=cfg.MODULUS_BOUNDS_MPA,geometry_bounds=limits,
                max_iterations=cfg.MAX_ITERATIONS,max_evaluations=cfg.MAX_EVALUATIONS,
                ftol=cfg.FTOL,gtol=cfg.GTOL)
        moving=RegionMaterialMap(m,n_regions=cfg.N_REGIONS,interfaces=positions,epsilon=cfg.EPSILON_MM,
                                 optimize_geometry=True)
        t0=np.r_[np.repeat(sub['theta'][:cfg.N_REGIONS],m.n_nod),sub['theta'][cfg.N_REGIONS:]]
        state,evidence,history=solve_unified(moving,case,uniform,nu=cfg.NU,gamma=gamma,
                initial_parameters=t0,modulus_bounds=cfg.MODULUS_BOUNDS_MPA,geometry_bounds=limits,
                max_iterations=cfg.MAX_ITERATIONS,max_evaluations=cfg.MAX_EVALUATIONS,
                ftol=cfg.FTOL,gtol=cfg.GTOL,
                progress=lambda r:print(f'{kind}/moving {shift}: {r}',flush=True))
        label=f'unknown_{shift:g}'
        np.savez_compressed(out/f'{label}.npz',theta=state['theta'],initial_positions=positions)
        write_csv(out/f'{label}_history.csv',history)
        write_csv(out/f'{label}_initialization_history.csv',sub_history)
        records.append(dict(label=label,initialization_evidence=sub_evidence,**evidence))
    write_json(out/'summary.json',dict(records=records,selected_gamma=gamma,
              geometry_gamma_protocol='Known-geometry selected gamma fixed for sensitivity; no unknown-geometry reselection',
              noise_protocol='Conditional perturbation from selected checkpoint',source_sha256=hashes()))


def compare(root,cfg):
    ny,nx=cfg.EVALUATION_SHAPE
    X,Y=np.meshgrid(np.linspace(0,cfg.GEO_L,nx),np.linspace(0,cfg.GEO_H,ny))
    xy=np.column_stack([X.ravel(),Y.ravel()])
    weights=np.outer(compute_axis_cell_widths(Y[:,0]),compute_axis_cell_widths(X[0]))
    m=mesh(cfg);p=model(m,cfg);rows=[]
    out=root/'04_comparison';out.mkdir(exist_ok=True)
    for kind in cfg.CASES:
        path=root/'03_inverse'/kind
        if not (path/'selection.json').exists():
            rows.append(dict(case=kind,status='not_computed',passed=False));continue
        selection=read_json(path/'selection.json')
        if selection['selected_gamma'] is None:
            rows.append(dict(case=kind,status=selection['status'],passed=False));continue
        index=selection['optimal_index']
        with np.load(path/f'gamma_{index:02d}'/'fields.npz') as saved:theta=saved['theta']
        E=p.sample(theta,xy).reshape(X.shape)
        truth=truth_function(cfg,kind)(X,Y)
        field_error=float(np.sqrt(np.sum(weights*(E-truth)**2)/np.sum(weights*truth**2)))
        region=np.searchsorted(cfg.BOUNDARIES,X,side='right')
        regional=[]
        for j in range(cfg.N_REGIONS):
            w=weights*(region==j)
            regional.append(float(abs(np.sum(w*(E-truth)))/np.sum(w*truth)))
        # Contrast across a declared fixed-width physical window, with the same
        # window applied to truth and reconstruction; continuous truth included.
        yy=np.linspace(0,cfg.GEO_H,65)
        contrasts=[]
        for s in cfg.BOUNDARIES:
            left=np.column_stack([np.full(len(yy),s-2*cfg.EPSILON_MM),yy])
            right=np.column_stack([np.full(len(yy),s+2*cfg.EPSILON_MM),yy])
            identified=p.sample(theta,right)-p.sample(theta,left)
            target=truth_function(cfg,kind)(right[:,0],yy)-truth_function(cfg,kind)(left[:,0],yy)
            contrasts.append(float(np.sqrt(np.mean((identified-target)**2))/np.mean(truth)))
        preliminary=(field_error<=cfg.GATE_MAX_FIELD_RELATIVE_RMS and max(regional)<=cfg.GATE_MAX_REGION_MEAN_RELATIVE_ERROR
                     and np.mean(regional)<=cfg.GATE_MAX_MEAN_REGION_RELATIVE_ERROR
                     and max(contrasts)<=cfg.GATE_MAX_INTERFACE_CONTRAST_ERROR)
        row=dict(case=kind,status='selected',selected_gamma=selection['selected_gamma'],
                 field_relative_rms=field_error,region_errors=regional,mean_region_error=float(np.mean(regional)),
                 max_region_error=max(regional),max_interface_contrast_error=max(contrasts),
                 preliminary_recovery_passed=bool(preliminary),passed=False,
                 reason='Independent-noise and initialization stability still required before experiments')
        validation=path/'validation/summary.json'
        if preliminary and validation.exists():
            robustness=read_json(validation)
            differences=[];positions=[]
            for record in robustness['records']:
                with np.load(validation.parent/f'{record["label"]}.npz') as saved:t=saved['theta']
                if record['label'].startswith('unknown'):
                    moving=RegionMaterialMap(m,n_regions=cfg.N_REGIONS,interfaces=np.array(cfg.BOUNDARIES)[:,None],
                          epsilon=cfg.EPSILON_MM,optimize_geometry=True)
                    other=moving.sample(t,xy).reshape(X.shape)
                    _,g=moving.unpack(t)
                    if kind!='continuous_2d':positions.append(float(abs(g.ravel()-cfg.BOUNDARIES).max()))
                else:other=p.sample(t,xy).reshape(X.shape)
                differences.append(float(np.sqrt(np.sum(weights*(other-E)**2)/np.sum(weights*truth**2))))
            row['max_robustness_field_difference']=max(differences)
            row['max_interface_position_error_mm']=max(positions,default=0.)
            row['passed']=bool(all(r['first_order_passed'] for r in robustness['records'])
                 and max(differences)<=cfg.GATE_MAX_MULTISTART_FIELD_DIFFERENCE
                 and max(positions,default=0.)<=cfg.GATE_MAX_INTERFACE_POSITION_ERROR_MM)
            row['reason']='Recovery plus fixed-gamma perturbation diagnostics completed'
        rows.append(row)
        np.savez_compressed(out/f'{kind}.npz',X=X,Y=Y,E=E,truth=truth,weights=weights)
        low,high=min(E.min(),truth.min()),max(E.max(),truth.max())
        plot_grid_fields(out,kind,X,Y,[('truth',truth,'Truth'),('identified',E,'Identified')],cmap='jet',unit='MPa',
                    title_prefix='',limits={'truth':(low,high),'identified':(low,high)},bounds=(0,cfg.GEO_L,0,cfg.GEO_H))
        plot_grid_fields(out,kind,X,Y,[('error',E-truth,'Identified - truth')],cmap='coolwarm',unit='MPa',title_prefix='',
                         bounds=(0,cfg.GEO_L,0,cfg.GEO_H))
    gate=dict(experiment_authorized_by_simulation=bool(all(r['passed'] for r in rows)),records=rows,
              criterion='All cases must select gamma, pass recovery, then pass independent-noise/multistart and unknown-geometry checks',
              status='passed' if all(r['passed'] for r in rows) else 'not_passed',source_sha256=hashes())
    write_json(out/'gate.json',gate)
    print(gate,flush=True)
    publish(root)


def publish(root):
    style='<style>body{max-width:1200px;margin:30px auto;padding:20px;font:16px/1.7 "Microsoft YaHei",sans-serif}img{max-width:100%}a{color:#075fa2}</style>'
    parts=['<!doctype html><meta charset="utf-8">'+style+'<h1>121×19 粗网格区内正则／L-curve</h1>',
           '<p>七区自由二维 Q4 分支；已知界面协议。gamma 只按观测残差／正则范数选取，不使用真值误差。</p>']
    for path in sorted(root.rglob('*.png')):
        rel=path.relative_to(root).as_posix()
        parts.append(f'<p>{rel}</p><a href="{Path(rel).with_suffix(".svg").as_posix()}"><img src="{rel}" loading="lazy"></a>')
    for path in sorted(root.rglob('selection.json'))+sorted(root.rglob('gate.json')):
        rel=path.relative_to(root).as_posix();parts.append(f'<p><a href="{rel}">{rel}</a></p>')
    (root/'结果总览.html').write_text('\n'.join(parts),encoding='utf-8')
    research=root.parent
    (research/'结果总览.html').write_text('<meta charset="utf-8">'+style+'<h1>统一材料算法</h1>'+''.join(
        f'<p><a href="{p.parent.name}/结果总览.html">{p.parent.name}</a></p>'
        for p in sorted(research.glob('*/结果总览.html'))),encoding='utf-8')
    print('保存位置：',root,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('prepare','forward','inverse','compare'))
    parser.add_argument('--config',default='unified_coarse_lcurve_config')
    parser.add_argument('--case',choices=('layered','continuous_2d','mixed_2d'))
    parser.add_argument('--validate',action='store_true',help='fixed selected gamma robustness diagnostics')
    parser.add_argument('--plot-only',action='store_true')
    args=parser.parse_args()
    if args.stage=='prepare' and not args.plot_only:prepare(load_python_configuration(args.config));return
    root,cfg=load(args.config)
    if args.stage in ('forward','inverse') and args.case is None:parser.error('--case required')
    if args.plot_only:
        if args.stage=='inverse':
            import pandas as pd
            plot_l_curve(root/'03_inverse'/args.case,pd.read_csv(root/'03_inverse'/args.case/'scan.csv').to_dict('records'))
        elif args.stage=='forward':
            with np.load(root/'02_forward'/args.case/'observations.npz') as saved:
                plot_observations(root/'02_forward'/args.case,saved['data'],saved['xy'],cfg)
        elif args.stage=='compare':compare(root,cfg)
        publish(root)
    elif args.stage=='forward':forward(root,cfg,args.case)
    elif args.stage=='inverse' and args.validate:validate_selected(root,cfg,args.case)
    elif args.stage=='inverse':inverse(root,cfg,args.case)
    else:compare(root,cfg)


if __name__=='__main__':main()
