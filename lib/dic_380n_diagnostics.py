"""Inspect the 380 N exports without treating temporal variation as pure noise."""
from pathlib import Path
import argparse
import json
import platform
import sys
import numpy as np
import pandas as pd
from lib.experiment_data import load_dic_frame_series, file_sha256
from lib.config_utils import load_python_configuration
from lib.dic_case_workflow import load_prepared
from fgm_asm.load_case import evaluate_observation

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT/'results/dic_full/load380_20261022'

def save_json(path, value):
    def convert(x):
        if isinstance(x,np.ndarray): return x.tolist()
        if isinstance(x,np.generic): return x.item()
        if isinstance(x,Path): return str(x)
        raise TypeError(type(x).__name__)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=convert,allow_nan=False),encoding='utf-8')

def audit():
    destination=OUTPUT/'input_audit'
    destination.mkdir(exist_ok=False)
    cfg,mesh,cases,meta,_=load_prepared(OUTPUT/'h1/01_prepared')
    with np.load(OUTPUT/'h1/01_prepared/fields.npz') as z:
        frozen={k:z[k].copy() for k in z.files}
    matrix=np.array(meta['matrix'])
    x=frozen['380N_coordinates']
    rigid=np.zeros((2*len(x),3))
    rigid[0::2,0]=1;rigid[1::2,1]=1
    rigid[0::2,2]=-(x[:,1]-x[:,1].mean())
    rigid[1::2,2]=(x[:,0]-x[:,0].mean())
    weights=np.repeat(frozen['380N_point_area_mm2'],2)
    inverse=np.linalg.inv(rigid.T@(weights[:,None]*rigid))
    def project(value):
        return value-(value*weights)@rigid@inverse@rigid.T
    rows=[];mean_fields={};arrays={};sources={}
    for name,item in cfg.PLATFORMS.items():
        series=load_dic_frame_series(Path(item['folder']),*item['stages'],coordinate_state='deformed')
        index=pd.MultiIndex.from_arrays(series.grid_ids.T)
        wanted=pd.MultiIndex.from_arrays(frozen[name+'_grid_ids'].T)
        match=index.get_indexer(wanted)
        assert (match>=0).all()
        values=series.frame_displacements[:,match,:]@matrix.T
        np.testing.assert_allclose(values.mean(0),frozen[name+'_observed'],atol=2e-16,rtol=0)
        projected=project(values.reshape(15,-1)).reshape(values.shape)
        mean=projected.mean(0)
        centered=projected-mean
        t=np.arange(15,dtype=float);t-=t.mean()
        slope=np.einsum('t,tpc->pc',t,centered)/(t@t)
        detrended=centered-t[:,None,None]*slope
        area=frozen[name+'_point_area_mm2'];area=area/area.sum()
        def rms(v):return np.sqrt(np.sum(np.mean(v*v,axis=0)*area[:,None],axis=0))*1000
        signal=np.sqrt(np.sum(mean*mean*area[:,None],axis=0))*1000
        sd=np.sqrt(np.sum(projected.var(0,ddof=1)*area[:,None],axis=0))*1000
        fluct=rms(detrended)
        drift=np.sqrt(np.sum((14*slope)**2*area[:,None],axis=0))*1000
        force=meta['force_pairing'][name]['mean_corrected_force_N']
        entry=dict(platform=name,frames=15,points=len(x),force_N=force,
            force_is_assumed=name=='380N',signal_rms_um=signal,
            temporal_sd_rms_um=sd,detrended_temporal_rms_um=fluct,
            fitted_drift_across_window_rms_um=drift,
            signal_to_detrended_variation=signal/fluct)
        rows.append(entry);print(name,entry,flush=True)
        mean_fields[name]=mean
        arrays[name+'_mean_deformation_mm']=mean
        arrays[name+'_detrended_temporal_rms_mm']=np.sqrt(np.mean(detrended*detrended,axis=0))
        for record in series.metadata['sources']:sources[record['path']]=record['sha256']
    differences=[]
    for name in ('180N','280N'):
        fa=meta['force_pairing'][name]['mean_corrected_force_N'];fb=319.9
        delta=mean_fields['380N']-mean_fields[name]*fb/fa
        differences.append(dict(reference_platform=name,compare_platform='380N',
            force_ratio=fb/fa,force_scaled_difference_rmse_um=np.sqrt(np.mean(delta**2,axis=0))*1000,
            note='Projected displacement difference at assumed 319.9 N; includes noise, time effects, force error and model mismatch.'))
    for case in cases:
        expected=-project(frozen[case.name+'_observed'].ravel())
        actual=evaluate_observation(case,np.zeros(mesh.n_dof))['residual']
        # The residual convention is measured minus prediction in the core.
        error=min(np.max(abs(actual-expected)),np.max(abs(actual+expected)))
        assert error<1e-10,(case.name,error)
    save_json(destination/'summary.json',dict(platforms=rows,force_scaled_comparisons=differences,
        force_assumption='380 N nominal minus 60.1 N offset = 319.9 N; not measured',
        noise_note='Temporal fluctuations contain measurement variation, creep and changing deformation. No division by sqrt(15); no claim of absolute DIC uncertainty.',
        selection='Only 380N selects gamma and initialization. 180N and 280N are previously inspected cross-load diagnostics.',
        sources_sha256=sources,code_sha256=file_sha256(Path(__file__)),python=sys.version,platform=platform.platform()))
    np.savez_compressed(destination/'fields.npz',coordinates=x,**arrays)
    print('input audit:',destination,flush=True)

if __name__=='__main__':
    audit()
