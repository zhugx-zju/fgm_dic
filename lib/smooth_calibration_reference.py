"""Continuous C2 calibration E(infill), used only for post-hoc comparison.

Historical trend_* outputs are piecewise-linear center connections. This module
keeps those historical numbers separate from the corrected smooth reference.
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
from numpy.polynomial.legendre import leggauss
from scipy.integrate import quad
from scipy.interpolate import CubicSpline

from .dic_calibration_curve_comparison import CENTERS, curve_errors, width_profile
from .experiment_data import file_sha256
from .joint_reference_diagnostic import build_common_cases
from .sequential_lcurve_scan import verify_sources


def make_reference(infill,modulus):
    """Interpolate the calibration means, checking monotonicity analytically."""
    infill,modulus=np.asarray(infill,dtype=float),np.asarray(modulus,dtype=float)
    if (infill.ndim!=1 or infill.shape!=modulus.shape or len(infill)<3 or
            not np.isfinite(infill).all() or not np.isfinite(modulus).all() or
            np.any(np.diff(infill)<=0) or np.any(np.diff(modulus)<0) or np.any(modulus<=0)):
        raise ValueError('Use increasing infill percentages and positive nondecreasing calibration means.')
    spline=CubicSpline(infill,modulus,bc_type='natural',extrapolate=False)
    minima=[]
    for index,h in enumerate(np.diff(infill)):
        a,b,c,_=spline.c[:,index]
        points=[0.,float(h)]
        if a!=0 and 0 < -b/(3*a) < h:
            points.append(float(-b/(3*a)))
        minima.extend((3*a*np.asarray(points)**2+2*b*np.asarray(points)+c).tolist())
        if index<len(infill)-2:
            np.testing.assert_allclose(3*a*h*h+2*b*h+c,spline.c[2,index+1],rtol=1e-10,atol=1e-10)
            np.testing.assert_allclose(6*a*h+2*b,2*spline.c[1,index+1],rtol=1e-10,atol=1e-10)
    if min(minima)<-1e-10:
        raise ValueError('Natural cubic interpolation is not monotone for these anchors; use a constrained C2 model instead of silently clipping.')
    np.testing.assert_allclose(spline(infill),modulus,rtol=0,atol=1e-10)
    return spline,float(min(minima))


def smooth_curve_errors(x,profile,spatial_reference,order=16):
    """Integrate curve-relative differences, splitting all cubic/linear roots."""
    lo,hi=float(CENTERS[0]),float(CENTERS[-1])
    if x[0]>lo or x[-1]<hi:
        raise ValueError('The field must cover the established -48 to 48 mm comparison support.')
    edges=np.unique(np.r_[lo,hi,x[(x>lo)&(x<hi)],CENTERS])
    nodes,weights=leggauss(order)
    absolute,square,reference_square=0.,0.,0.
    for left,right in zip(edges[:-1],edges[1:]):
        region=int(np.clip(np.searchsorted(CENTERS,(left+right)/2)-1,0,len(CENTERS)-2))
        center=CENTERS[region]
        predicted=np.interp([left,right],x,profile)
        slope=(predicted[1]-predicted[0])/(right-left)
        a,b,c,d=spatial_reference.c[:,region]
        polynomial=[predicted[0]+slope*(center-left)-d,slope-c,-b,-a]
        roots=np.polynomial.polynomial.polyroots(polynomial)
        crossings=[float(root.real+center) for root in roots if abs(root.imag)<1e-9 and left+1e-10<root.real+center<right-1e-10]
        intervals=np.unique(np.r_[left,crossings,right])
        for start,end in zip(intervals[:-1],intervals[1:]):
            points=(start+end)/2+(end-start)/2*nodes
            measure=weights*(end-start)/2
            ref=spatial_reference(points)
            difference=np.interp(points,x,profile)-ref
            absolute+=np.sum(measure*abs(difference/ref))
            square+=np.sum(measure*difference**2)
            reference_square+=np.sum(measure*ref**2)
    return dict(smooth_mean_absolute_relative_error_percent=float(100*absolute/(hi-lo)),
                smooth_l2_relative_error_percent=float(100*np.sqrt(square/reference_square)))


def save(fig,path):
    fig.savefig(path.with_suffix('.png'),dpi=300,bbox_inches='tight')
    fig.savefig(path.with_suffix('.svg'),bbox_inches='tight')
    plt.close(fig)


def analyze(study:Path,output:Path):
    if output.exists():
        raise FileExistsError('Choose a new smooth-reference output; historical artifacts are preserved.')
    protocol=json.loads((study/'protocol.json').read_text(encoding='utf-8'))
    _,mesh,cases,_,_,records=build_common_cases([Path(row['source']) for row in protocol['source_cases']])
    verify_sources(protocol['source_cases'])
    calibration=pd.DataFrame(records).sort_values('infill_percent')
    infill,values=calibration.infill_percent.to_numpy(),calibration.mean_E_MPa.to_numpy()
    np.testing.assert_array_equal(infill,np.arange(40,101,10))
    density_reference,min_derivative=make_reference(infill,values)
    spatial_reference=CubicSpline(CENTERS,values,bc_type='natural',extrapolate=False)
    xi=np.linspace(-48,48,1001)
    np.testing.assert_allclose(density_reference(70+.625*xi),spatial_reference(xi),rtol=1e-12,atol=1e-10)
    # Independent analytic constant-error check of the relative-curve integral.
    check=smooth_curve_errors(np.array([-48.,48.]),np.array([110.,110.]),CubicSpline(CENTERS,np.full(7,100.)))
    np.testing.assert_allclose(list(check.values()),[10.,10.],rtol=1e-12,atol=1e-12)
    endpoints=[json.loads(path.read_text(encoding='utf-8')) for path in (study/'scan').glob('*/endpoint.json')]
    valid=[row for row in endpoints if row['converged']]
    targets=(1e-8,1.2648552168552956e-9,1e-10)
    picked=[min(valid,key=lambda row:abs(np.log(row['gamma']/target)))['gamma'] for target in targets]
    x=mesh.plot_x[0]-mesh.geo_l/2
    rows,profiles,sources,quad_checks=[],{},[],[]
    for endpoint in sorted(valid,key=lambda row:row['gamma']):
        gamma=endpoint['gamma']
        source=study/'scan'/f'gamma_{gamma:.12g}'/'fields.npz'
        with np.load(source) as saved:
            field=saved['E_inverse'].copy()
        profile=width_profile(mesh,field)
        errors=smooth_curve_errors(x,profile,spatial_reference,order=16)
        coarse=smooth_curve_errors(x,profile,spatial_reference,order=8)
        np.testing.assert_allclose(list(errors.values()),list(coarse.values()),rtol=1e-10,atol=1e-10)
        old=curve_errors(x,profile,values,'trend',(-48.,48.))
        layered=curve_errors(x,profile,values,'layered',(-48.,48.))
        rows.append(dict(gamma=gamma,representative=gamma in picked,**errors,
            historical_linear_mean_absolute_relative_error_percent=old['mean_absolute_relative_error_percent'],
            layered_mean_absolute_relative_error_percent=layered['mean_absolute_relative_error_percent']))
        sources.append(dict(path=str(source.resolve()),sha256=file_sha256(source)))
        if gamma in picked:
            points=np.unique(np.r_[x[(x>-48)&(x<48)],CENTERS[1:-1]])
            independent,estimate=quad(lambda t:abs(np.interp(t,x,profile)/spatial_reference(t)-1),
                -48,48,points=points,epsabs=1e-11,epsrel=1e-11,limit=1000)
            independent=100*independent/96
            np.testing.assert_allclose(errors['smooth_mean_absolute_relative_error_percent'],independent,rtol=1e-8,atol=1e-8)
            quad_checks.append(dict(gamma=gamma,adaptive_quad_relative_error_percent=independent,adaptive_quad_absolute_error_estimate=estimate))
            profiles[gamma]=(profile,field)
    output.mkdir(parents=True)
    table=pd.DataFrame(rows)
    table.to_csv(output/'all_smooth_reference_errors.csv',index=False,lineterminator='\n')
    table[table.representative].to_csv(output/'representative_errors.csv',index=False,lineterminator='\n')
    dense_fill=np.linspace(40,100,1001)
    pd.DataFrame(dict(infill_percent=dense_fill,smooth_E_MPa=density_reference(dense_fill))).to_csv(output/'continuous_E_of_infill.csv',index=False,lineterminator='\n')
    calibration.to_csv(output/'calibration_anchors.csv',index=False,lineterminator='\n')
    plt.rcParams.update({'font.family':'Microsoft YaHei','font.size':11,'svg.fonttype':'path',
        'axes.unicode_minus':False,'axes.spines.top':False,'axes.spines.right':False})
    colors=('#0072B2','#009E73','#D55E00')
    fig,ax=plt.subplots(figsize=(10,5),constrained_layout=True)
    ax.plot(dense_fill,density_reference(dense_fill),color='black',lw=1.8,label='连续平滑标定分布 E(φ)，C²三次样条')
    infill_edges=np.r_[infill[0],(infill[:-1]+infill[1:])/2,infill[-1]]
    ax.stairs(values,infill_edges,baseline=None,color='#777777',ls='--',lw=1.3,label='分层标定阶梯线')
    ax.errorbar(infill,values,yerr=calibration.sd_E_MPa.to_numpy(),fmt='o',color='black',ms=4,capsize=3,lw=.9,label='标定均值 ± 组内SD')
    for gamma,color in zip(picked,colors):
        profile,_=profiles[gamma]
        ax.plot(dense_fill,np.interp((dense_fill-70)/.625,x,profile),color=color,lw=1.5,label=f'顺序反演 γ={gamma:.3g}')
    ax.set(xlabel='连续化填充率 φ (%)',ylabel='E (MPa)',xlim=(40,100),title='连续光滑分布、分层标定与顺序反演结果')
    ax.legend(frameon=False,fontsize=9,ncol=2)
    save(fig,output/'smooth_E_of_infill')
    fig,ax=plt.subplots(figsize=(10,5),constrained_layout=True)
    ax.plot(xi,spatial_reference(xi),color='black',lw=1.8,label='连续平滑标定参照 E[φ(ξ)]')
    spatial_edges=np.r_[x[0],np.arange(-40.,41.,16.),x[-1]]
    ax.stairs(values,spatial_edges,baseline=None,color='#777777',ls='--',lw=1.3,label='分层标定阶梯线')
    for gamma,color in zip(picked,colors):
        ax.plot(x,profiles[gamma][0],color=color,lw=1.5,label=f'最终诊断场 γ={gamma:.3g}')
    ax.set(xlabel='制造坐标 ξ (mm)',ylabel='E (MPa)',xlim=(x[0],x[-1]),title='同一空间对应下，与连续平滑模量分布比较')
    ax.legend(frameon=False,fontsize=9,ncol=2)
    save(fig,output/'smooth_spatial_comparison')
    gamma=picked[1]
    profile,field=profiles[gamma]
    fig=plt.figure(figsize=(10,7),constrained_layout=True)
    grid=fig.add_gridspec(2,1,height_ratios=(2.5,1))
    ax=fig.add_subplot(grid[0])
    ax.plot(x,profile,color='#009E73',lw=1.8,label='顺序反演最终场：宽度均值')
    ax.plot(xi,spatial_reference(xi),color='black',lw=1.8,label='连续平滑标定参照')
    ax.stairs(values,spatial_edges,baseline=None,color='#777777',ls='--',lw=1.3,label='分层标定阶梯线')
    ax.set(ylabel='E (MPa)',title=f'γ={gamma:.3g}诊断结果：修正后的连续平滑参照',xlim=(x[0],x[-1]))
    ax.legend(frameon=False,fontsize=9)
    map_ax=fig.add_subplot(grid[1],sharex=ax)
    drawing=map_ax.pcolormesh(mesh.plot_x-mesh.geo_l/2,mesh.plot_y,field.reshape(mesh.nods_y,mesh.nods_x),shading='auto',cmap='viridis')
    map_ax.set(xlabel='制造坐标 ξ (mm)',ylabel='y (mm)',ylim=(0,mesh.geo_h),aspect='equal',title='已保存的二维反演场（数值未改动）')
    fig.colorbar(drawing,ax=map_ax,label='E (MPa)',shrink=.9)
    save(fig,output/'corrected_final_field')
    metadata=dict(definition='C2 natural cubic E(infill), through seven calibration means, no extrapolation.',
        infill_percent=infill.tolist(),E_MPa=values.tolist(),cubic_coefficients=density_reference.c.tolist(),
        minimum_analytic_derivative_MPa_per_percentage_point=min_derivative,
        spatial_mapping='phi_percent = 70 + 0.625*manufacturing_x_mm, only -48 <= x <= 48; inherited layer-center correspondence, not coordinate rescaling.',
        spatial_mapping_role='Auxiliary continuous comparison assumption, not verified actual continuous infill.',
        displayed_references='Black solid: smooth E(infill); grey dashed: original layered calibration.',
        no_optimization_or_gamma_reselection=True,source_protocol_sha256=file_sha256(study/'protocol.json'),
        source_calibration=records,source_fields=sources,independent_adaptive_quad_checks=quad_checks,
        implementation_sha256=file_sha256(Path(__file__)))
    (output/'provenance.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['# 修正：连续光滑的随填充率变化模量分布','',
        '此前的层中心分段直线不是用户指定的连续光滑参照。本报告改用七个1023单调约束标定均值构造E(φ)自然三次样条：通过所有均值，整段C²连续；解析核验最小导数为'+f'{min_derivative:.6f} MPa/填充率百分点，保持单调不减。没有为贴近反演调节标定均值或曲线参数。','',
        '先在填充率40%–100%上定义连续E(φ)，再沿既有层中心位置关系φ=70+0.625ξ映射到制造坐标[-48,48] mm。这是把分层参照连续化的比较约定，不把实物分层直接宣称为连续真值；不更改坐标尺度，也不在40%–100%之外外推。','',
        '新的平均绝对相对差异为100/96×∫[-48,48]|Ē_inverse(ξ)/E_smooth[φ(ξ)]−1|dξ。与旧线性参照、原阶梯线采用相同支持区间和最终模量场。反演、位移误差、γ、L-curve失败状态均未重算或覆盖。','',
        '| γ | 连续光滑参照差异 % | 旧层中心折线差异 %（历史口径） | 原阶梯线差异 % |',
        '|---|---:|---:|---:|']
    for gamma in picked:
        row=table[table.gamma==gamma].iloc[0]
        lines.append(f'| {gamma:.4g} | {row.smooth_mean_absolute_relative_error_percent:.4f} | {row.historical_linear_mean_absolute_relative_error_percent:.4f} | {row.layered_mean_absolute_relative_error_percent:.4f} |')
    lines+=['','原30个有效最终场均按新参照重算，见all_smooth_reference_errors.csv。图中同时保留黑色实线的连续光滑E(φ)和灰色虚线的分层标定阶梯线；阶梯误差作为分层表示的独立对照，旧层中心折线不再冒充光滑参照。','',
        '核验：标定均值逐点通过，逐区间导数最小值与C²拼接解析检查；相对差异积分在节点、样条结点与误差多项式实根处切分，8点/16点Gauss一致；三个展示点另以独立自适应quad核对。常值场10%差异解析检查通过。原准备数据哈希不变。','',
        '![连续E随填充率分布](smooth_E_of_infill.png)','', '![连续参照空间比较](smooth_spatial_comparison.png)','', '![修正后的结果图](corrected_final_field.png)','',
        '复现：`python -B -m lib.smooth_calibration_reference --study <新范围顺序扫描目录> --output <新输出目录>`。']
    (output/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    verify_sources(protocol['source_cases'])
    print(table[table.representative].to_string(index=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    analyze(args.study,args.output)
