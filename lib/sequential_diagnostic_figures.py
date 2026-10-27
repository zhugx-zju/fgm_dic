"""Plot saved sequential diagnostic fields when no L-curve point was selected."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .dic_calibration_curve_comparison import width_profile
from .experiment_data import file_sha256
from .joint_reference_diagnostic import build_common_cases
from .sequential_lcurve_scan import verify_sources
from .smooth_calibration_reference import make_reference


def save(fig,path):
    fig.savefig(path.with_suffix('.png'),dpi=300)
    fig.savefig(path.with_suffix('.svg'))
    plt.close(fig)


def plot(study:Path,output:Path,gamma_target:float):
    if output.exists():
        raise FileExistsError('Choose a new figure directory; earlier figures are preserved.')
    protocol=json.loads((study/'protocol.json').read_text(encoding='utf-8'))
    cfg,mesh,cases,_,_,calibration=build_common_cases([Path(row['source']) for row in protocol['source_cases']])
    verify_sources(protocol['source_cases'])
    records=[json.loads(path.read_text(encoding='utf-8')) for path in (study/'scan').glob('*/endpoint.json')]
    valid=[row for row in records if row['converged']]
    chosen=min(valid,key=lambda row:abs(np.log(row['gamma']/gamma_target)))
    gamma=chosen['gamma']
    root=study/'scan'/f'gamma_{gamma:.12g}'
    with np.load(root/'fields.npz') as saved:
        field=saved['E_inverse'].copy()
    values=pd.DataFrame(calibration).mean_E_MPa.to_numpy()
    calibration_infill=pd.DataFrame(calibration).infill_percent.to_numpy()
    smooth_reference,minimum_derivative=make_reference(calibration_infill,values)
    reference_x=np.linspace(-48.,48.,1001)
    x=mesh.plot_x[0]-mesh.geo_l/2
    output.mkdir(parents=True)
    plt.rcParams.update({'font.family':'Microsoft YaHei','font.size':11,'svg.fonttype':'path',
        'axes.unicode_minus':False,'axes.spines.top':False,'axes.spines.right':False})

    def references(ax):
        ax.plot(reference_x,smooth_reference(70+.625*reference_x),color='black',lw=1.5,label='连续平滑标定分布 E(φ)')
        ax.stairs(values,np.r_[x[0],np.arange(-40.,41.,16.),x[-1]],baseline=None,
                  color='#777777',ls='--',lw=1.3,label='分层标定阶梯线')
        ax.plot(np.arange(-48.,49.,16.),values,'o',color='black',ms=3,label='标定均值')
        ax.set(xlim=(x[0],x[-1]),ylabel='E (MPa)')

    fig=plt.figure(figsize=(10,7),constrained_layout=True)
    grid=fig.add_gridspec(2,1,height_ratios=(2.5,1))
    ax=fig.add_subplot(grid[0])
    ax.plot(x,width_profile(mesh,field),color='#D55E00',lw=1.8,label='顺序反演最终场：宽度均值')
    references(ax)
    ax.set_title(f'γ={gamma:.3g}：1→2→3最终诊断结果（非L-curve选定）')
    ax.legend(frameon=False,fontsize=9,loc='upper left')
    map_ax=fig.add_subplot(grid[1],sharex=ax)
    drawing=map_ax.pcolormesh(mesh.plot_x-mesh.geo_l/2,mesh.plot_y,
        field.reshape(mesh.nods_y,mesh.nods_x),shading='auto',cmap='viridis')
    map_ax.set(xlabel='制造坐标 ξ (mm)',ylabel='y (mm)',ylim=(0,mesh.geo_h),aspect='equal',title='保留全部独立y行的二维模量场')
    fig.colorbar(drawing,ax=map_ax,label='E (MPa)',shrink=.9)
    save(fig,output/'final_field')

    fig,ax=plt.subplots(figsize=(10,5),constrained_layout=True)
    sources=[root/'fields.npz']
    for index,(case,color) in enumerate(zip(cases,('#0072B2','#009E73','#D55E00')),start=1):
        source=root/f'stage{index}_{case.name}/fields.npz'
        with np.load(source) as saved:
            profile=width_profile(mesh,saved['E_inverse'])
        ax.plot(x,profile,color=color,lw=1.6,label=f'阶段{index}：拟合{case.name}后')
        sources.append(source)
    references(ax)
    ax.set(xlabel='制造坐标 ξ (mm)',title=f'γ={gamma:.3g}：各阶段模量曲线（诊断）')
    ax.legend(frameon=False,ncol=2,fontsize=9)
    save(fig,output/'stage_profiles')

    fig,ax=plt.subplots(figsize=(10,5),constrained_layout=True)
    diagnostics=[]
    for target,color in zip((1e-8,gamma_target,1e-10),('#0072B2','#009E73','#D55E00')):
        row=min(valid,key=lambda item:abs(np.log(item['gamma']/target)))
        source=study/'scan'/f'gamma_{row["gamma"]:.12g}'/'fields.npz'
        with np.load(source) as saved:
            profile=width_profile(mesh,saved['E_inverse'])
        ax.plot(x,profile,color=color,lw=1.6,label=f'γ={row["gamma"]:.3g}')
        diagnostics.append(row['gamma'])
        sources.append(source)
    references(ax)
    ax.set(xlabel='制造坐标 ξ (mm)',title='新范围扫描：不同γ的最终诊断场比较（均非L-curve选定）')
    ax.legend(frameon=False,ncol=2,fontsize=9)
    save(fig,output/'gamma_comparison')
    (output/'provenance.json').write_text(json.dumps(dict(gamma_target=gamma_target,actual_gamma=gamma,
        role='Diagnostic only; no L-curve selection and no reoptimization.',
        calibration_reference='C2 natural cubic E(infill), through calibration means; no extrapolation.',
        minimum_analytic_reference_derivative=minimum_derivative,
        gamma_comparison=diagnostics,source_protocol_sha256=file_sha256(study/'protocol.json'),
        fields_sha256={str(path.resolve()):file_sha256(path) for path in dict.fromkeys(sources)},
        implementation_sha256=file_sha256(Path(__file__))),ensure_ascii=False,indent=2),encoding='utf-8')
    print('Saved diagnostic curves and 2D field:',output.resolve())


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--gamma',type=float,default=1.2648552168552956e-9)
    args=parser.parse_args()
    plot(args.study,args.output,args.gamma)
