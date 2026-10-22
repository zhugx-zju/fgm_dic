"""Fresh norms, gradients, selections and predictions for all saved space probes."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from lib.dic_380n_spaces import inputs,regional_means,p,OUTPUT
from lib.experiment_data import file_sha256


def verify():
    mesh=p.MeshInfo(p.L,p.H,42,8);data,truth,metadata,_=inputs(mesh)
    audits=[];selection_audits=[]
    for fit_name,folder in [('380N',OUTPUT),('180N',OUTPUT.parent/'spaces_180N_control')]:
        for method in p.METHODS:
            directory=folder/method;model=p.Model(mesh,method,data[fit_name].bc)
            original=pd.read_csv(directory/'scan.csv',float_precision='round_trip')
            for index,row in original.iterrows():
                with np.load(directory/f'gamma_{index:02d}.npz') as z:
                    E=z['E'].copy();saved_gradient=z['gradient'].copy()
                state=model.value_gradient(np.log(E),data[fit_name],row.gamma)
                rho=float(np.sqrt(2*state['data_cost']));eta=float(np.sqrt(max(0,state['reg'])))
                np.testing.assert_allclose(rho,row.rho,atol=2e-11,rtol=1e-9)
                np.testing.assert_allclose(eta,row.eta,atol=2e-8,rtol=1e-9)
                np.testing.assert_allclose(state['gradient'],saved_gradient,atol=2e-10,rtol=1e-4)
                assert np.isfinite(E).all() and (E>0).all()
                audits.append(dict(fit_platform=fit_name,method=method,index=int(index),rho_error=abs(rho-row.rho),eta_error=abs(eta-row.eta)))
            if fit_name=='380N':
                combined=directory/'scan_after_recovery.csv';selection_path=directory/'selection_after_recovery.json';result_dir=directory
            else:
                result_dir=OUTPUT.parent/'spaces_180N_recovered'/method
                combined=result_dir/'scan_after_recovery.csv';selection_path=result_dir/'selection.json'
            scan=pd.read_csv(combined,float_precision='round_trip')
            selection=p.select_l_curve_corner(scan.gamma,scan.rho,scan.eta,scan.success)
            saved=json.loads(selection_path.read_text(encoding='utf-8'))
            assert selection['status']==saved['status'] and selection['selected_gamma']==saved['selected_gamma']
            result=json.loads((result_dir/'summary.json').read_text(encoding='utf-8'))
            if result.get('gamma') is not None:
                with np.load(result_dir/'selected.npz') as z:
                    E=z['E'].copy();stored={k:z[k].copy() for k in z.files}
                means=regional_means(mesh,E,method)
                np.testing.assert_allclose(means,result['regional_means_MPa'],atol=1e-9,rtol=1e-11)
                for name,case in data.items():
                    current=p.Model(mesh,method,case.bc);u,_,_=current.forward(E)
                    _,_,res,c=case.evaluate(u);pred=case.H@u+case.R@c
                    np.testing.assert_allclose(pred,stored[name+'_prediction'],atol=2e-11,rtol=1e-8)
            selection_audits.append(dict(fit_platform=fit_name,method=method,status=saved['status'],selected_gamma=saved['selected_gamma'],reproduced=True))
    pd.DataFrame(audits).to_csv(OUTPUT.parent/'space_scan_audit.csv',index=False)
    summary=dict(original_scan_fields_recomputed=len(audits),all_norms_and_gradients_reproduced=True,
        selection_and_prediction_checks=selection_audits,source_sha256=file_sha256(Path(__file__)),
        packaged_core_sha256=file_sha256(Path(p.__file__)))
    (OUTPUT.parent/'space_verification.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary),flush=True)

if __name__=='__main__':verify()
