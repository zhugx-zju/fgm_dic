"""Select and inspect saved sequential-per-gamma L-curves without optimizing."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
import numpy as np
import pandas as pd
import scipy
from scipy.interpolate import CubicSpline

from fgm_asm.l_curve import select_l_curve_corner
from .dic_calibration_curve_comparison import CENTERS, width_profile, metrics as reference_metrics
from .experiment_data import file_sha256
from .joint_reference_diagnostic import build_common_cases
from .sequential_lcurve_scan import verify_sources
from .smooth_calibration_reference import make_reference, smooth_curve_errors


def clean(value):
    if isinstance(value, dict):
        return {key: clean(item) for key,item in value.items()}
    if isinstance(value, (list,tuple,np.ndarray)):
        return [clean(item) for item in value]
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value,float) and not np.isfinite(value):
        return None
    return value


def save(fig,path):
    fig.savefig(path.with_suffix('.png'),dpi=300)
    fig.savefig(path.with_suffix('.svg'))
    plt.close(fig)


def analyze(study: Path, output: Path):
    if output.exists():
        raise FileExistsError('Choose a new report output; previous reports are preserved.')
    protocol=json.loads((study/'protocol.json').read_text(encoding='utf-8'))
    progress=json.loads((study/'progress.json').read_text(encoding='utf-8'))
    if progress['state']!='scan_complete':
        raise ValueError('Finish the complete gamma scan before selecting a corner.')
    records=[json.loads(path.read_text(encoding='utf-8')) for path in (study/'scan').glob('*/endpoint.json')]
    if len(records)!=len(protocol['gamma_candidates']):
        raise ValueError('Missing scan endpoints; no partial-scan corner selection.')
    records.sort(key=lambda row:row['gamma'])
    np.testing.assert_allclose([row['gamma'] for row in records],sorted(protocol['gamma_candidates']),rtol=0,atol=0)
    roots=[Path(row['source']) for row in protocol['source_cases']]
    cfg,mesh,cases,audit,_,calibration=build_common_cases(roots)
    verify_sources(protocol['source_cases'])
    assert file_sha256(study/'implementation_snapshot.py')==protocol['implementation_sha256']
    output.mkdir(parents=True)
    scan=pd.DataFrame([{key:value for key,value in row.items() if key not in ('stage_records','case_errors')} for row in records])
    for key in ('residual_norm','regularization_norm','terminal_case_residual_norm'):
        scan[key]=scan[key].astype(float)
    selectors={}
    for label,key in [('all_cases_endpoint','residual_norm'),('terminal_case','terminal_case_residual_norm')]:
        corner=select_l_curve_corner(scan.gamma,scan[key],scan.regularization_norm,scan.converged)
        selectors[label]=corner
        scan[f'{label}_curvature']=corner['curvature']
        scan[f'{label}_eligible']=corner['eligible']
        scan[f'{label}_selected']=False
        if corner['optimal_index'] is not None:
            scan.loc[corner['optimal_index'],f'{label}_selected']=True
    scan.to_csv(output/'selection.csv',index=False,lineterminator='\n')
    (output/'selection.json').write_text(json.dumps(clean(selectors),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    stage_rows=[]
    errors=[]
    for row in records:
        for stage in row['stage_records']:
            stage_rows.append({key:value for key,value in stage.items() if key!='attempts'} |
                dict(attempt_count=len(stage['attempts']),first_attempt_success=stage['attempts'][0]['converged']))
        errors.extend(dict(gamma=row['gamma'],**error) for error in row.get('case_errors',[]))
    stages=pd.DataFrame(stage_rows)
    stages.to_csv(output/'all_stage_metrics.csv',index=False,lineterminator='\n')
    errors=pd.DataFrame(errors)
    errors.to_csv(output/'all_endpoint_relative_errors.csv',index=False,lineterminator='\n')
    stage_selections={}
    for stage_number in (1,2,3):
        part=stages[stages.stage==stage_number].set_index('gamma').reindex(scan.gamma)
        stage_selections[f'stage{stage_number}']=select_l_curve_corner(scan.gamma,
            part.residual_norm.to_numpy(),part.regularization_norm.to_numpy(),part.converged.eq(True).to_numpy(dtype=bool))
    (output/'stage_curve_diagnostics.json').write_text(json.dumps(clean(stage_selections),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    plt.rcParams.update({'font.family':'Microsoft YaHei','font.size':10,'svg.fonttype':'path',
                         'axes.unicode_minus':False,'axes.spines.top':False,'axes.spines.right':False})
    status_grid=np.zeros((3,len(scan)),dtype=int)
    gamma_index={value:index for index,value in enumerate(scan.gamma)}
    for row in stages.itertuples():
        status_grid[row.stage-1,gamma_index[row.gamma]]=1 if row.converged else 2
    status_colors=('#E5E5E5','#0072B2','#D55E00')
    fig,ax=plt.subplots(figsize=(11,3),constrained_layout=True)
    ax.imshow(status_grid,aspect='auto',interpolation='nearest',cmap=ListedColormap(status_colors),vmin=0,vmax=2)
    ticks=np.unique(np.linspace(0,len(scan)-1,7,dtype=int))
    ax.set_xticks(ticks,[f'{scan.gamma.iloc[index]:.1e}' for index in ticks])
    ax.set_yticks([0,1,2],['阶段1 / run1','阶段2 / run2','阶段3 / run3'])
    ax.set(xlabel='γ（等距对数网格）',title='逐γ的顺序链停止状态；成功停止不代表满足严格GTOL')
    ax.legend(handles=[Patch(facecolor=color,label=label) for color,label in zip(status_colors,
        ('未执行','成功停止（含预算内恢复）','失败或超预算'))],frameon=False,ncol=3,loc='upper center',bbox_to_anchor=(.5,-.24),fontsize=9)
    save(fig,output/'stage_status')
    fig,axes=plt.subplots(2,2,figsize=(11,8),constrained_layout=True)
    for column,(label,key,title,color) in enumerate([
        ('all_cases_endpoint','residual_norm','最终场回代三个工况（主曲线）','#0072B2'),
        ('terminal_case','terminal_case_residual_norm','最终场回代末工况（对照）','#009E73')]):
        ax=axes[0,column]
        good=scan[f'{label}_eligible']
        ax.loglog(scan[key].where(good),scan.regularization_norm.where(good),'o-',color=color,ms=3,lw=1)
        for idx in np.flatnonzero(good.to_numpy())[::7]:
            point=scan.iloc[idx]
            ax.annotate(f'{point.gamma:.1g}',(point[key],point.regularization_norm),fontsize=7,xytext=(3,3),textcoords='offset points')
        corner=selectors[label]
        if corner['optimal_index'] is not None:
            point=scan.loc[corner['optimal_index']]
            ax.plot(point[key],point.regularization_norm,'*',ms=13,color='#D55E00',label=f'γ={point.gamma:.3g}')
            ax.legend(frameon=False,fontsize=8)
        visible=scan.loc[good,key]
        if len(visible) and visible.min()<visible.max():
            ax.xaxis.set_major_locator(FixedLocator(np.geomspace(visible.min(),visible.max(),5)))
            decimals=min(10,max(0,int(np.ceil(-np.log10(visible.max()-visible.min())))+1))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda value,position,places=decimals:f'{value:.{places}f}'))
            ax.xaxis.set_minor_locator(NullLocator())
        ax.set(title=title+'\n'+('已选定' if corner['status']=='selected' else corner['status']),xlabel=r'面积加权残差范数 ρ (mm$^2$)',ylabel='H1梯度范数 η (MPa)')
        axes[1,column].semilogx(scan.gamma,scan[f'{label}_curvature'],'o-',color=color,ms=3,lw=1)
        axes[1,column].axhline(0,color='.6',lw=.7)
        axes[1,column].set(xlabel='γ',ylabel='有向离散曲率',title='沿用原曲率、单调性与两侧支撑检查')
    save(fig,output/'l_curves')
    fig,axes=plt.subplots(2,1,figsize=(10,7),sharex=True,constrained_layout=True)
    if len(errors):
        for name,color in zip(('run1','run2','run3'),('#0072B2','#009E73','#D55E00')):
            part=errors[errors.case==name].sort_values('gamma')
            axes[0].semilogx(part.gamma,part.axial_relative_error_percent,'o-',color=color,ms=3,lw=1,label=name)
            axes[1].semilogx(part.gamma,part.displacement_relative_error_percent,'o-',color=color,ms=3,lw=1,label=name)
    axes[0].set(ylabel='轴向位移相对误差 (%)',title='每个γ的同一最终模量场回代三个工况')
    axes[1].set(xlabel='γ',ylabel='双分量位移相对误差 (%)')
    for ax in axes:
        ax.legend(frameon=False,ncol=3,fontsize=9)
    save(fig,output/'endpoint_errors')
    selected_metrics=[]
    plotted=[]
    for label,corner in selectors.items():
        if corner['selected_gamma'] is None:
            continue
        gamma=corner['selected_gamma']
        source=study/'scan'/f'gamma_{gamma:.12g}'
        endpoint=json.loads((source/'endpoint.json').read_text(encoding='utf-8'))
        with np.load(source/'fields.npz') as saved:
            terminal=saved['E_inverse'].copy()
        calibration_table=pd.DataFrame(calibration)
        values=calibration_table.mean_E_MPa.to_numpy()
        make_reference(calibration_table.infill_percent.to_numpy(),values)
        smooth_reference=CubicSpline(CENTERS,values,bc_type='natural',extrapolate=False)
        legacy_reference=reference_metrics(mesh.plot_x[0]-mesh.geo_l/2,width_profile(mesh,terminal),values)
        reference={('historical_linear_'+key[6:] if key.startswith('trend_') else key):value
                   for key,value in legacy_reference.items()}
        reference.update(smooth_curve_errors(mesh.plot_x[0]-mesh.geo_l/2,width_profile(mesh,terminal),smooth_reference))
        selected_metrics.extend(dict(selection=label,gamma=gamma,**error,**reference) for error in endpoint['case_errors'])
        target=output/f'selected_{label}'
        target.mkdir()
        np.savez_compressed(target/'fields.npz',E_inverse=terminal,coordinates=mesh.coord)
        profiles=[]
        for index in (1,2,3):
            with np.load(source/f'stage{index}_run{index}/fields.npz') as saved:
                profiles.append(width_profile(mesh,saved['E_inverse']))
        plotted.append((label,gamma,terminal,profiles))
    selected=pd.DataFrame(selected_metrics)
    selected.to_csv(output/'selected_relative_errors_and_reference.csv',index=False,lineterminator='\n')
    if plotted:
        fig,axes=plt.subplots(len(plotted),1,figsize=(10,3.5*len(plotted)),sharex=True,sharey=True,constrained_layout=True)
        fig_map,map_axes=plt.subplots(len(plotted),1,figsize=(10,2.5*len(plotted)),sharex=True,sharey=True,constrained_layout=True)
        vmin=min(item[2].min() for item in plotted)
        vmax=max(item[2].max() for item in plotted)
        values=pd.DataFrame(calibration).mean_E_MPa.to_numpy()
        for ax,map_ax,(label,gamma,field,profiles) in zip(np.atleast_1d(axes),np.atleast_1d(map_axes),plotted):
            for index,(profile,color) in enumerate(zip(profiles,('#0072B2','#009E73','#D55E00')),start=1):
                ax.plot(mesh.plot_x[0]-mesh.geo_l/2,profile,color=color,lw=1.4,label=f'阶段{index}结果')
            reference_x=np.linspace(-48.,48.,1001)
            ax.plot(reference_x,smooth_reference(reference_x),color='black',lw=1.3,label='连续平滑标定E(φ)，不参与选γ')
            ax.stairs(values,np.r_[-mesh.geo_l/2,np.arange(-40.,41.,16.),mesh.geo_l/2],baseline=None,
                      color='#777777',ls='--',lw=1.2,label='分层标定阶梯线')
            ax.set(title=f'{label}，γ={gamma:.3g}',ylabel='宽度均值 E (MPa)')
            ax.legend(frameon=False,ncol=4,fontsize=8)
            plotted_map=map_ax.pcolormesh(mesh.plot_x-mesh.geo_l/2,mesh.plot_y,field.reshape(mesh.nods_y,mesh.nods_x),
                shading='auto',cmap='viridis',vmin=vmin,vmax=vmax)
            map_ax.set(title=f'{label}：最终自由二维场，γ={gamma:.3g}',ylabel='y (mm)',aspect='equal')
        np.atleast_1d(axes)[-1].set_xlabel('制造坐标 ξ (mm)')
        np.atleast_1d(map_axes)[-1].set_xlabel('制造坐标 ξ (mm)')
        fig_map.colorbar(plotted_map,ax=np.atleast_1d(map_axes).tolist(),label='E (MPa)',shrink=.85)
        save(fig,output/'selected_profiles')
        save(fig_map,output/'selected_modulus_maps')
    lines=['# 每个γ内执行完整1→2→3的顺序反演与L-curve','',
        f'{len(scan)}个γ，范围{scan.gamma.min():.6g}–{scan.gamma.max():.6g}。共{len(stages)}个已执行阶段，{int(scan.converged.sum())}条完整成功顺序链，{int((~scan.converged).sum())}个无效γ。每个γ从相同的工况1 DIC均匀初值出发，无跨γ续算；同γ内逐元素传递E1→E2→E3，每阶段只有当前工况目标。','',
        '**主选点口径：** 取同一个最终场E3(γ)，分别正演并配准三个工况，ρ=sqrt(Σ r_c(E3)ᵀW_c r_c(E3))；η=sqrt(∫|∇E3|²)，不把γ乘进范数。没有混用E1拟合工况1、E2拟合工况2、E3拟合工况3的残差。','',
        '主曲线是顺序端点的多工况几何诊断：E3只最小化最后工况的目标，没有最小化总残差目标。因此它不是同时联合目标的标准Tikhonov解路径，也不保证沿γ单调。另保存末工况残差曲线与三个阶段各自曲线以区分选点定义。所有曲线使用原离散正曲率、相邻有效点、两侧各两点支撑与单调性规则；没有有效拐点时保留失败，不按标定补γ。','',
        '| 曲线 | 选点状态 | γ |', '|---|---|---:|']
    for label,corner in selectors.items():
        lines.append(f'| {label} | {corner["status"]} | {corner["selected_gamma"] if corner["selected_gamma"] is not None else "未选定"} |')
    lines+=['','主曲线未选定时，不用末工况对照结果冒充三工况选点。末工况曲线若有拐点，仅说明顺序末阶段的拟合与H1之间有该几何折中。','',
        f'每阶段最多1500迭代/12000评价，FTOL=1e-30、GTOL=1e-12。ABNORMAL最多用原剩余预算重置历史两次，首次异常、重启和总开销均记录。最终阶段失败或超预算的γ不进入端点选点，前一阶段失败则停止该链。{int(stages.converged.sum())}/{len(stages)}个阶段报告成功，严格GTOL通过{int(stages.gradient_tolerance_satisfied.sum())}个；函数停止不是严格一阶最优性证明。','',
        '单调约束标定仅在选点后用于趋势线与分层线相对差异比较。它不参与初值、阶段目标、γ或拐点选择。位移报告面积加权相对误差，分母为该工况DIC去刚体变形量，非逐点位移相除。','',
        '共同域105.774937554×18.435507949 mm，121×19自由二维H1，厚度3.2 mm、ν=0.26；沿用前次共同中心定位与各次起末载荷假设。独立参考图未被当作同一参考图，分别去除三刚体模式；材料点对应仍未独立核实，回代也不是独立验证。','',
        '核验：每γ相同初值、同γ阶段间初值逐元素一致；逐阶段正演ρ与优化器值相对容差1e-8核对，η相对容差1e-9核对；同一最终场的三工况ρ同时以代价之和与加权残差拼接范数独立核对；原冻结输入哈希不变。','',
        '![逐γ阶段停止状态](stage_status.png)','', '![两种L-curve及曲率](l_curves.png)','', '![最终场的逐工况误差](endpoint_errors.png)','']
    if len(selected):
        lines+=['| 选点曲线 | γ | 回代工况 | 轴向误差 % | 横向误差 % | 双分量误差 % | 连续光滑参照差异 % | 分层线差异 % |',
                '|---|---:|---|---:|---:|---:|---:|---:|']
        for r in selected.itertuples():
            lines.append(f'| {r.selection} | {r.gamma:.4g} | {r.case} | {r.axial_relative_error_percent:.3f} | {r.transverse_relative_error_percent:.3f} | {r.displacement_relative_error_percent:.3f} | {r.smooth_mean_absolute_relative_error_percent:.3f} | {r.layered_common_mean_absolute_relative_error_percent:.3f} |')
        lines+=['','标定曲线差异统一为宽度均值曲线在[−48,48] mm的平均绝对相对差异。连续光滑参照由七个标定均值构造C²自然三次E(填充率)样条并核验单调不减，按既有层中心对应关系映射，不视为实测连续材料真值。历史折线结果仅保留在historical_linear_*列。','',
                '![选定γ的阶段曲线](selected_profiles.png)','', '![选定γ的二维场](selected_modulus_maps.png)','']
    bounds=protocol.get('gamma_log10_range')
    range_arguments=f' --gamma-log10-range {bounds[0]:g} {bounds[1]:g}' if protocol.get('gamma_grid_source')=='explicit_experiment_log10_bounds' else ''
    lines+=[f'复现扫描：`python -u -B -m lib.sequential_lcurve_scan --cases <三个完整ROI冻结目录> --output <新目录>{range_arguments}`；完全相同协议的完整缓存可用`--resume`。后处理：`python -B -m lib.sequential_lcurve_report --study <扫描目录> --output <新的报告目录>`。']
    (output/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    import fgm_asm.l_curve as selector_module
    evidence=dict(source_protocol_sha256=file_sha256(study/'protocol.json'),
        selection_script_sha256=file_sha256(Path(__file__)),selector_sha256=file_sha256(Path(selector_module.__file__)),
        no_calibration_in_selection=True,original_sources_verified_unchanged=True,
        endpoint_metadata_sha256={str(p.relative_to(study)):file_sha256(p) for p in (study/'scan').glob('*/endpoint.json')},
        result_fields_sha256={str(p.relative_to(study)):file_sha256(p) for p in (study/'scan').glob('**/fields.npz')},
        environment=dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,pandas=pd.__version__,platform=platform.platform()))
    if (study/'fixed_gamma_crosscheck.json').exists():
        crosscheck=json.loads((study/'fixed_gamma_crosscheck.json').read_text(encoding='utf-8'))
        assert crosscheck['terminal_field_relative_difference_percent']<crosscheck['tolerance_percent']
        evidence['independent_fixed_gamma_crosscheck']=crosscheck
    (output/'verification.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({label:dict(status=corner['status'],selected_gamma=corner['selected_gamma']) for label,corner in selectors.items()},ensure_ascii=False))
    print(f'{int(scan.converged.sum())}/{len(scan)} complete chains, {len(stages)} stages.')
    if len(selected):
        print(selected[['selection','gamma','case','axial_relative_error_percent','displacement_relative_error_percent']].to_string(index=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    analyze(args.study,args.output)
