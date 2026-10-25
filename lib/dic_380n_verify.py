"""Fresh FE/adjoint checks of saved 380N scans, selected fields and KKT flags."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import diags
from fgm_asm.exact_tv import QuadratureGradient
from fgm_asm.inverse_solver import _evaluate_inverse_state
from fgm_asm.material import MaterialInfo
from fgm_asm.load_case import evaluate_observation
from fgm_asm.research import solve_forward
from fgm_asm.l_curve import select_l_curve_corner
from lib.dic_case_workflow import load_prepared,_metrics
from lib.dic_exact_tv_report import _audit_field
from lib.experiment_data import file_sha256

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/'results/dic_full/load380_20261022'

def verify(method, root=None):
    root=Path(root) if root is not None else OUTPUT/method
    cfg,mesh,cases,prepared,_=load_prepared(root/'01_prepared')
    inverse=json.loads((root/'03_inverse/run_metadata.json').read_text(encoding='utf-8'))
    scan=pd.read_csv(root/'03_inverse/selection.csv',float_precision='round_trip')
    assert len(scan)==50 and scan.gamma.nunique()==50
    assert tuple(cfg.FIT_CASES)==('380N',)
    np.testing.assert_allclose(sorted(scan.gamma),sorted(cfg.GAMMA_CANDIDATES),rtol=1e-14,atol=0)
    corner=select_l_curve_corner(scan.gamma,scan.residual_norm,scan.regularization_norm,
        scan.converged & scan.get('regularization_resolved',True))
    assert corner['selected_gamma']==inverse.get('selected_gamma')
    fit=[c for c in cases if c.name=='380N'];audits=[];missing_fields=[]
    for row in scan.itertuples():
        directory=getattr(row,'solution_directory',None)
        directory=root/'03_inverse'/directory if isinstance(directory,str) else root/'03_inverse/l_curve'/f'gamma_{row.gamma:g}'
        if row.stop_message == 'numerical_error_no_returned_field':
            failure=json.loads((directory/'exception.json').read_text(encoding='utf-8'))
            assert not row.converged and failure['excluded_from_selection']
            assert not failure['returned_field_available'] and not (directory/'fields.npz').exists()
            np.testing.assert_allclose(failure['gamma'],row.gamma,rtol=1e-14,atol=0)
            assert 0 < failure['FE_lower_bound'] <= failure['FE_upper_bound'] <= cfg.MAX_FUN
            if 'original_log_sha256' in failure:
                assert file_sha256(root/'step3.log') == failure['original_log_sha256']
            else:
                trace=pd.read_csv(directory/'history_until_exception.csv')
                assert int(trace.evaluations.iloc[-1]) == failure['FE_lower_bound'] == failure['FE_upper_bound']
            missing_fields.append(dict(failure))
            audits.append(dict(gamma=row.gamma,saved_converged=False,
                returned_field_available=False,numerical_failure_record_audited=True,
                norms_and_gradient_reproduced=False))
            continue
        with np.load(directory/'fields.npz') as z:
            E=z['E_inverse'].copy()
            assert np.isfinite(E).all() and (E>0).all()
            saved_gradient=z['final_gradient'].copy()
        if cfg.REGULARIZATION_MODE=='exact_tv':
            record=_audit_field(mesh,fit,cfg,QuadratureGradient(mesh),directory,row.gamma,row.nu,row.converged)
            assert record['certificate_reproduced']
        else:
            state=_evaluate_inverse_state(mesh,fit[0].bc_info,None,MaterialInfo(nu=.26),row.gamma,E,0,
                regularization_mode=cfg.REGULARIZATION_MODE,observation_cases=fit,
                regularization_options={'delta':cfg.TV_DELTA} if cfg.REGULARIZATION_MODE=='huber_tv' else None)
            np.testing.assert_allclose(state['residual_norm'],row.residual_norm,rtol=1e-8,atol=1e-12)
            np.testing.assert_allclose(state['regularization_norm'],row.regularization_norm,rtol=1e-10,atol=1e-9)
            np.testing.assert_allclose(state['grad_m'],saved_gradient,rtol=1e-5,atol=2e-11)
            record=dict(gamma=row.gamma,saved_converged=bool(row.converged),
                gradient_inf=float(np.max(abs(state['grad_m']))),
                gradient_tolerance_satisfied=bool(np.max(abs(state['grad_m']))<=cfg.GTOL),
                norms_and_gradient_reproduced=True)
        audits.append(record)
    destination=root/'verification';destination.mkdir(exist_ok=True)
    pd.DataFrame(audits).to_csv(destination/'scan_audit.csv',index=False)
    common=[];checks=[];gradient_checks=[]
    if inverse.get('selected_gamma') is not None and 'solvers' in inverse:
        saved_metrics=pd.read_csv(root/'03_inverse/comparison.csv')
        for solver in inverse['solvers']:
            nu=solver['nu'];directory=root/'03_inverse'/f'nu_{nu:.2f}'
            with np.load(directory/'fields.npz') as z:
                E=z['E_inverse'].copy();stored={k:z[k].copy() for k in z.files}
            if cfg.REGULARIZATION_MODE=='exact_tv':
                result=_audit_field(mesh,fit,cfg,QuadratureGradient(mesh),directory,inverse['selected_gamma'],nu,solver['converged'])
                assert result['certificate_reproduced'];checks.append(result)
            else:
                state=_evaluate_inverse_state(mesh,fit[0].bc_info,None,MaterialInfo(nu=nu),
                    inverse['selected_gamma'],E,0,regularization_mode=cfg.REGULARIZATION_MODE,
                    observation_cases=fit,
                    regularization_options={'delta':cfg.TV_DELTA} if cfg.REGULARIZATION_MODE=='huber_tv' else None)
                np.testing.assert_allclose(state['grad_m'],stored['final_gradient'],rtol=1e-5,atol=2e-11)
                np.testing.assert_allclose(state['residual_norm'],solver['residual_norm'],rtol=1e-8,atol=1e-12)
                np.testing.assert_allclose(state['regularization_norm'],solver['regularization_norm'],rtol=1e-10,atol=1e-9)
                gradient_checks.append(dict(nu=nu,gradient_inf=float(np.max(abs(state['grad_m']))),
                    gradient_tolerance_satisfied=bool(np.max(abs(state['grad_m']))<=cfg.GTOL),
                    norms_and_gradient_reproduced=True))
            for case in cases:
                u,_=solve_forward(mesh,case.bc_info['tensile_end_force'],cfg.THICKNESS,nu,E)
                fresh=evaluate_observation(case,u)
                np.testing.assert_allclose(fresh['prediction'],stored[case.name+'_prediction'],rtol=1e-8,atol=2e-11)
                record=saved_metrics[(saved_metrics.case==case.name)&np.isclose(saved_metrics.nu,nu)].iloc[0]
                for key,val in _metrics(fresh).items():np.testing.assert_allclose(val,record[key],rtol=1e-8,atol=2e-12)
                obs=case.observation;weights=np.repeat(obs.point_weights,2)
                equal=replace(case,observation=replace(obs,weight_vector=weights,W=diags(weights,format='csr')))
                state=evaluate_observation(equal,u)
                common.append(dict(method=method,nu=nu,platform=case.name,
                    role='fit' if case.name=='380N' else 'previously_inspected_cross_load_diagnostic',
                    **_metrics(state)))
    pd.DataFrame(common).to_csv(destination/'common_equal_weight_metrics.csv',index=False)
    summary=dict(method=method,scan_points=50,scan_reproduced=not missing_fields,
        scan_outcomes_audited=True,missing_field_numerical_exceptions=missing_fields,
        saved_field_certificates_reproduced=True,selected_gamma=inverse.get('selected_gamma'),
        final_kkt_audits=checks,final_gradient_audits=gradient_checks,
        all_saved_predictions_reproduced=True if common else None,
        prepared_metadata_sha256=file_sha256(root/'01_prepared/run_metadata.json'),
        scan_csv_sha256=file_sha256(root/'03_inverse/selection.csv'),
        source_sha256=file_sha256(Path(__file__)))
    (destination/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('method')
    parser.add_argument('--root', type=Path, help='Explicit four-step case root for a cropped 380N run')
    args=parser.parse_args()
    verify(args.method, args.root)
