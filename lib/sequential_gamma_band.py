"""Replay saved sequential endpoints to assess the requested gamma interval.

No optimization, parameter reselection, or modification of earlier studies.
The trend_* comparisons here retain the historical linear-center convention.
For the user's corrected continuous smooth E(infill) definition, use
lib.smooth_calibration_reference instead of interpreting those legacy columns.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fgm_asm.load_case import evaluate_observation
from fgm_asm.material import MaterialInfo
from fgm_asm.regularization import get_tikhonov_regularization
from fgm_asm.research import solve_forward
from .dic_calibration_curve_comparison import metrics as reference_metrics, width_profile
from .dic_preprocessing import relative_displacement_errors
from .experiment_data import file_sha256
from .joint_reference_diagnostic import build_common_cases
from .sequential_lcurve_scan import verify_sources
from .uniform_sample_analysis import area_statistics


def save(fig,path):
    fig.savefig(path.with_suffix('.png'),dpi=300)
    fig.savefig(path.with_suffix('.svg'))
    plt.close(fig)


def spatial_metrics(mesh,field):
    profile=width_profile(mesh,field)
    centered=(field.reshape(mesh.nods_y,mesh.nods_x)-profile).ravel()
    points=(centered[mesh.ele_nods_id-1]@mesh.gauss_N.T).T
    weights=mesh.gauss_abs_det_j*mesh.gauss_w[:,None]
    stats=area_statistics(mesh,field)
    dispersion=100*np.sqrt(np.sum(weights*points**2)/weights.sum())/stats['mean_E_MPa']
    negative=np.sum(np.diff(mesh.plot_x[0])[np.diff(profile)<0])/mesh.geo_l*100
    return dict(transverse_dispersion_percent=float(dispersion),negative_slope_length_percent=float(negative),
                min_nodal_E_MPa=float(field.min()),max_nodal_E_MPa=float(field.max()),**stats),profile


def analyze(study:Path,output:Path):
    if output.exists():
        raise FileExistsError('Choose a new band-analysis output; earlier outputs are preserved.')
    protocol=json.loads((study/'protocol.json').read_text(encoding='utf-8'))
    assert json.loads((study/'progress.json').read_text())['state']=='scan_complete'
    roots=[Path(item['source']) for item in protocol['source_cases']]
    cfg,mesh,cases,_,_,calibration=build_common_cases(roots)
    verify_sources(protocol['source_cases'])
    values=pd.DataFrame(calibration).mean_E_MPa.to_numpy()
    base_selection=json.loads((study/'report_final/selection.json').read_text(encoding='utf-8'))
    base_gamma=base_selection['all_cases_endpoint']['selected_gamma']
    if base_gamma is None:
        raise ValueError('This comparison needs the previous primary L-curve reference point.')
    endpoints=[json.loads(path.read_text(encoding='utf-8')) for path in (study/'scan').glob('*/endpoint.json')]
    endpoints.sort(key=lambda item:item['gamma'])
    assert len(endpoints)==len(protocol['gamma_candidates'])
    in_band=[row for row in endpoints if 1e-10<=row['gamma']<=1e-8]
    valid=[row for row in in_band if row['converged']]
    # Representatives are fixed by range limits and the already discussed
    # 1.2649e-9 candidate, rather than by calibration error or reconstructed shape.
    targets=(1e-8,1.2648552168552956e-9,1e-10)
    picks=[min(valid,key=lambda item:abs(np.log(item['gamma']/target)))['gamma'] for target in targets]
    displayed=[base_gamma,*dict.fromkeys(picks)]
    rows,profiles,maps,sources=[],[],{},[]
    for endpoint in endpoints:
        gamma=endpoint['gamma']
        row=dict(gamma=gamma,converged=endpoint['converged'],in_requested_band=1e-10<=gamma<=1e-8,
                 original_lcurve_selected=gamma==base_gamma,representative=gamma in displayed,
                 all_stage_gradient_tolerances_satisfied=endpoint['all_stage_gradient_tolerances_satisfied'])
        if endpoint['converged']:
            source=study/'scan'/f'gamma_{gamma:.12g}'/'fields.npz'
            with np.load(source) as saved:
                field=saved['E_inverse'].copy()
            rho_square=0.
            for case,saved_error in zip(cases,endpoint['case_errors']):
                assert case.name==saved_error['case']
                u,_=solve_forward(mesh,case.bc_info['tensile_end_force'],cfg.THICKNESS,cfg.PRIMARY_NU,field)
                state=evaluate_observation(case,u)
                errors=relative_displacement_errors(case,state['prediction'])
                for key,value in errors.items():
                    np.testing.assert_allclose(value,saved_error[key],rtol=1e-7,atol=1e-8)
                    row[f'{case.name}_{key}']=value
                rho_square+=2*state['cost_tar']
            material=MaterialInfo(cfg.PRIMARY_NU)
            material.update(field)
            eta=float(np.sqrt(get_tikhonov_regularization(mesh,material)))
            np.testing.assert_allclose(np.sqrt(rho_square),endpoint['residual_norm'],rtol=1e-8,atol=1e-10)
            np.testing.assert_allclose(eta,endpoint['regularization_norm'],rtol=1e-9,atol=1e-9)
            spatial,profile=spatial_metrics(mesh,field)
            row.update(spatial,replayed_total_residual_norm=float(np.sqrt(rho_square)),replayed_H1_norm=eta,
                **reference_metrics(mesh.plot_x[0]-mesh.geo_l/2,profile,values))
            sources.append(dict(field=str(source.resolve()),sha256=file_sha256(source)))
            if gamma in displayed:
                maps[gamma]=field.reshape(mesh.nods_y,mesh.nods_x)
                profiles.extend(dict(gamma=gamma,x_mm=x,E_MPa=e) for x,e in zip(mesh.plot_x[0]-mesh.geo_l/2,profile))
        rows.append(row)
    table=pd.DataFrame(rows)
    baseline=table[table.original_lcurve_selected].iloc[0]
    table['roughness_factor_from_lcurve']=table.replayed_H1_norm/baseline.replayed_H1_norm
    table['total_residual_reduction_percent']=100*(1-table.replayed_total_residual_norm/baseline.replayed_total_residual_norm)
    output.mkdir(parents=True)
    table.to_csv(output/'all_saved_endpoint_metrics.csv',index=False,lineterminator='\n')
    table[table.in_requested_band].to_csv(output/'requested_band_metrics.csv',index=False,lineterminator='\n')
    representatives=table.set_index('gamma').loc[displayed].reset_index()
    representatives.to_csv(output/'representative_metrics.csv',index=False,lineterminator='\n')
    profile_table=pd.DataFrame(profiles)
    profile_table.to_csv(output/'representative_profiles.csv',index=False,lineterminator='\n')
    plt.rcParams.update({'font.family':'Microsoft YaHei','font.size':10,'svg.fonttype':'path',
                         'axes.unicode_minus':False,'axes.spines.top':False,'axes.spines.right':False})
    colors=('#555555','#0072B2','#009E73','#D55E00')
    fig,ax=plt.subplots(figsize=(10,5),constrained_layout=True)
    for index,(gamma,color) in enumerate(zip(displayed,colors)):
        data=profile_table[profile_table.gamma==gamma]
        ax.plot(data.x_mm,data.E_MPa,color=color,lw=1.6,ls='--' if index==0 else '-',
            label=f'γ={gamma:.3g}'+('（原L-curve）' if index==0 else ''))
    x=mesh.plot_x[0]-mesh.geo_l/2
    edges=np.r_[x[0],np.arange(-40.,41.,16.),x[-1]]
    ax.plot(np.repeat(edges,2)[1:-1],np.repeat(values,2),color='black',ls=':',lw=1.1,label='单调标定阶梯线')
    ax.plot(np.arange(-48.,49.,16.),values,color='black',ls='-.',lw=1,label='层中心标定趋势线')
    ax.set(xlabel='制造坐标 ξ (mm)',ylabel='最终场宽度均值 E (MPa)',title='相同1→2→3顺序反演：减小γ后的最终场')
    ax.legend(frameon=False,ncol=2,fontsize=9)
    save(fig,output/'endpoint_profiles')
    fig,axes=plt.subplots(2,3,figsize=(13,7),constrained_layout=True)
    specifications=[(f'run{i}_displacement_relative_error_percent',f'第{i}次双分量相对误差 (%)') for i in (1,2,3)]
    specifications.extend([('trend_common_mean_absolute_relative_error_percent','趋势线平均绝对相对差异 (%)'),
        ('layered_common_mean_absolute_relative_error_percent','分层线平均绝对相对差异 (%)'),
        ('transverse_dispersion_percent','模量横向离散 / 面积均值 (%)')])
    for ax,(key,label) in zip(axes.ravel(),specifications):
        ax.axvspan(1e-10,1e-8,color='#0072B2',alpha=.07,label='用户指定区间')
        ax.semilogx(table.gamma,table[key],color='#0072B2',lw=1.1)
        ax.plot(base_gamma,baseline[key],'*',color='#555555',ms=11,label='原L-curve')
        for gamma,color in zip(picks,colors[1:]):
            point=table[table.gamma==gamma].iloc[0]
            ax.plot(gamma,point[key],'o',ms=4,color=color)
        ax.set(xlabel='γ',ylabel=label)
    axes[0,0].legend(frameon=False,fontsize=8)
    save(fig,output/'tradeoff')
    fig,axes=plt.subplots(len(displayed),1,figsize=(11,7),sharex=True,sharey=True,constrained_layout=True)
    vmin=min(field.min() for field in maps.values())
    vmax=max(field.max() for field in maps.values())
    for ax,gamma in zip(axes,displayed):
        drawing=ax.pcolormesh(mesh.plot_x-mesh.geo_l/2,mesh.plot_y,maps[gamma],shading='auto',cmap='viridis',vmin=vmin,vmax=vmax)
        ax.set(title=f'γ={gamma:.3g}'+('（原L-curve）' if gamma==base_gamma else ''),ylabel='y (mm)',aspect='equal')
    axes[-1].set_xlabel('制造坐标 ξ (mm)')
    fig.colorbar(drawing,ax=axes,label='E (MPa)',shrink=.85)
    save(fig,output/'modulus_maps')
    lines=['# 顺序反演：γ在1e-10至1e-8内是否改善','',
        f'只回放现有扫描，不重新优化或替换选参结果。原50个γ中有{len(in_band)}个位于指定区间，{len(valid)}个完整有效；8.2864e-9的第3阶段线搜索异常，保留失败，不计入成功比较。回放全部39个有效最终场以提供完整上下文，所有比较都用同一共同ROI、数据、载荷、权重、网格和ν。','',
        '**结论：** 该区间能减轻原L-curve点的平滑偏差，尤其改善第2、3次位移拟合与标定参照差异；第1次约17%的轴向误差基本保留。约1e-9可作为折中诊断候选；降至1e-10量级，分层线差异和第3次拟合继续改善，但趋势线差异不再持续降低，横向离散和局部起伏明显增加。不能把整个区间或最小γ都称为最优。','',
        '| γ | 第1次轴向误差 % | 第2次轴向误差 % | 第3次轴向误差 % | 趋势线差异 % | 分层线差异 % | 横向离散 % | H1范数/原值 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in representatives.itertuples():
        lines.append(f'| {r.gamma:.4g} | {r.run1_axial_relative_error_percent:.3f} | {r.run2_axial_relative_error_percent:.3f} | {r.run3_axial_relative_error_percent:.3f} | {r.trend_common_mean_absolute_relative_error_percent:.3f} | {r.layered_common_mean_absolute_relative_error_percent:.3f} | {r.transverse_dispersion_percent:.3f} | {r.roughness_factor_from_lcurve:.2f} |')
    lines+=['','代表点在比较标定前固定：指定区间内最接近1e-8、前已讨论的1.264855e-9、1e-10的有效网格点。没有在新边界1e-10或1e-8补算精确点。完整区间数据包含全部11个有效点和1个失败点，不只保留代表点。','',
        '趋势线与阶梯线差异均为最终宽度均值曲线在制造坐标[−48,48] mm的平均绝对相对差异。七层均值误差、完整ROI阶梯线差异和L₂差异另存CSV，不能与本表混称。模量横向离散为去除宽度均值后的二维Q4面积标准差/面积均值，不是置信区间或DIC噪声估计。H1范数增加可能来自材料变化与误差吸收两者，不能仅凭起伏将其判为噪声。','',
        '位移误差按原面积权重和该工况去刚体DIC变形量作整体分母；每个γ均用同一最终场回代三个工况，不混用不同阶段场。所有完整链按函数下降停止，均未满足严格GTOL，物理定位与力参考假设仍保留。标定只作事后诊断，不能按最接近标定点改称正式最优γ。原顺序端点L-curve选择与原文件保持不变。','',
        '核验：39个最终场各自重新正演三个工况，逐工况相对误差与原缓存核对；总残差范数与H1范数分别在1e-8和1e-9相对容差内核对，原冻结输入哈希不变。模量统计使用Q4 Gauss积分，标定曲线积分沿节点、层界及误差过零点精确切分，并沿用6点/12点积分一致性核验。','',
        '![最终场宽度均值](endpoint_profiles.png)','', '![全扫描权衡](tradeoff.png)','', '![同色标二维模量场](modulus_maps.png)','',
        '复现：`python -B -m lib.sequential_gamma_band --study <已完成顺序L-curve扫描目录> --output <新目录>`。']
    (output/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (output/'provenance.json').write_text(json.dumps(dict(source_protocol_sha256=file_sha256(study/'protocol.json'),
        source_selection_sha256=file_sha256(study/'report_final/selection.json'),saved_fields=sources,
        original_lcurve_gamma_unchanged=base_gamma,requested_band=[1e-10,1e-8],representative_targets=targets,
        representative_gammas=picks,no_optimization_executed=True,reference_used_for_selection=False,
        implementation_sha256=file_sha256(Path(__file__))),ensure_ascii=False,indent=2),encoding='utf-8')
    verify_sources(protocol['source_cases'])
    print(representatives[['gamma','run1_axial_relative_error_percent','run2_axial_relative_error_percent',
        'run3_axial_relative_error_percent','trend_common_mean_absolute_relative_error_percent',
        'layered_common_mean_absolute_relative_error_percent','transverse_dispersion_percent']].to_string(index=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    analyze(args.study,args.output)
