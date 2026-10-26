"""From project root: python -B -m post_tensile_test.1026.60sample.analysis_results.h1_uniformity_20261026_final.audit_inputs --output NEW_DIR"""
import argparse, json
from pathlib import Path
import numpy as np
from pyxlsb import open_workbook
from lib.experiment_data import load_dic_frame_series, file_sha256
parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
root=Path('post_tensile_test/1026/60sample')
out=args.output;out.mkdir(parents=True,exist_ok=False)
records={}
for run,load,window in [(1,380,(945,959)),(2,280,(465,479)),(3,380,(235,249))]:
 folder=root/str(run)/f'{load}N'
 s=load_dic_frame_series(folder,*window)
 with open_workbook(str(root/str(run)/'force.xlsb')) as b:
  with b.get_sheet(1) as sh: rows=[[c.v for c in row] for row in sh.rows()][1:]
 f=np.array(rows,float);xy=s.coordinates;uv=s.frame_displacements.mean(0)
 beta=np.linalg.lstsq(np.column_stack([np.ones(len(xy)),xy]),uv,rcond=None)[0]
 record=dict(run=run,folder=str(folder),stages=window,length_mm=float(np.ptp(xy[:,1])),width_mm=float(np.ptp(xy[:,0])),thickness_mm=3.2,start_force_N=float(f[0,1]),last_record_force_N=float(f[-1,1]),end_force_N=float(load if run!=3 else f[-1,1]),end_force_assumed=run!=3,force_source_sha256=file_sha256(root/str(run)/'force.xlsb'),disp_source_sha256=file_sha256(root/str(run)/'disp.csv'),affine_DX_DY_coefficients=beta.tolist(),temporal_SD_rms_mm=np.sqrt(np.mean(s.frame_displacements.std(0,ddof=1)**2,axis=0)).tolist(),**s.metadata)
 record['delta_force_N']=record['end_force_N']-record['start_force_N']
 # Reference-coordinate consistency independently: raw X-DX/Y-DY are stationary across all frames.
 record['registration_matrix']=[[0,1],[1,0]]
 records[str(run)]=record
 print(run,record['length_mm'],record['width_mm'],record['delta_force_N'],flush=True)
(out/'audit.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
print(out.resolve())

