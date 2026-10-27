"""Replay saved cropped H1 fields in a prespecified gamma interval; never optimize.

python -B -m lib.material_window_gamma_check --cases CROP1 CROP2 CROP3 --output NEW_DIR
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .dic_case_workflow import load_prepared
from .dic_gamma_sensitivity import inspect_field
from .experiment_data import file_sha256
from .material_window_analysis import common_quadrature, scalar_values
from .sample2_result_analysis import save_figure

# Fixed from the existing logarithmic grid before looking at calibration scores.
DISPLAY_TARGETS = (8e-9, 4e-9, 1.3e-9)
COLORS = ('#0072B2', '#D55E00', '#009E73', '#CC79A7')


def profile_distance(x1, f1, x2, f2):
    """Symmetric L2 difference (%) over overlapping physical manufacturing x.

    Both profiles are continuous piecewise linear width means. Splitting at all
    knots makes two-point Gauss integration of the squared difference exact.
    """
    low, high = max(x1[0], x2[0]), min(x1[-1], x2[-1])
    if high <= low:
        raise ValueError('Profiles have no common physical interval')
    edges = np.unique(np.r_[low, high, x1[(x1>low)&(x1<high)], x2[(x2>low)&(x2<high)]])
    half = np.diff(edges)/2
    points = ((edges[:-1]+edges[1:])[:, None]/2 + half[:, None]*[-1/np.sqrt(3), 1/np.sqrt(3)]).ravel()
    weights = np.repeat(half, 2)
    first, second = np.interp(points, x1, f1), np.interp(points, x2, f2)
    denominator = np.sum(weights*((first+second)/2)**2)
    if denominator <= 0:
        raise ValueError('Nonpositive profile normalization')
    return float(100*np.sqrt(np.sum(weights*(first-second)**2)/denominator)), (low, high)


def analyze(roots: list[Path], output: Path):
    output.mkdir(parents=True, exist_ok=False)
    rows, regions, profiles, records, models = [], [], [], [], []
    for root in roots:
        cfg, mesh, cases, meta, _ = load_prepared(root/'01_prepared')
        if len(cases) != 1 or list(cfg.REGION_INFILL_PERCENT) != [40, 50, 60, 70]:
            raise ValueError('This report is scoped to independent 40%–70% cropped cases')
        case = cases[0]
        inverse = root/'03_inverse'
        original = json.loads((inverse/'run_metadata.json').read_text(encoding='utf-8'))
        scan = pd.read_csv(inverse/'selection.csv')
        chosen = scan.index[scan.selected].tolist()
        if len(chosen) != 1:
            raise ValueError('Expected one original selected L-curve point')
        interval = scan.index[(scan.gamma>=1e-9)&(scan.gamma<=1.21e-8)].tolist()
        display = {chosen[0]: 0}
        for level, target in enumerate(DISPLAY_TARGETS, 1):
            eligible = scan.loc[interval]
            index = int(eligible.index[np.argmin(abs(np.log(eligible.gamma.to_numpy()/target)))])
            display[index] = level
        points, weights = common_quadrature(mesh, mesh, np.zeros(2), cfg.REGION_BOUNDARIES_X)
        groups = np.searchsorted(cfg.REGION_BOUNDARIES_X, points[:, 0], side='right')
        calibration = pd.DataFrame(meta['calibration'])
        evidence = []
        for index in sorted(set(interval+chosen)):
            item = scan.loc[index]
            trial = inverse/'l_curve'/f'gamma_{item.gamma:g}'
            if isinstance(item.get('solution_directory'), str):
                trial = inverse/item.solution_directory
            with np.load(trial/'fields.npz') as saved:
                field = saved['E_inverse'].copy()
            result, _, mean, sd = inspect_field(cfg, mesh, case, field, calibration)
            np.testing.assert_allclose(result['replay_residual_norm'], item.residual_norm, rtol=1e-7, atol=1e-10)
            np.testing.assert_allclose(result['replay_regularization_norm'], item.regularization_norm, rtol=1e-8, atol=1e-10)
            # Exact partition integration, matching the preceding cropped report.
            values = scalar_values(mesh, field, points)
            differences = []
            for group, record in enumerate(calibration.to_dict(orient='records')):
                mask = groups == group
                average = float(np.average(values[mask], weights=weights[mask]))
                difference = 100*(average/record['mean_E_MPa']-1)
                differences.append(difference)
                regions.append(dict(run=cfg.SAMPLE_RUN, gamma=item.gamma, **record,
                                    inverse_mean_E_MPa=average, difference_percent=difference))
            result.update(regional_mean_absolute_relative_error_percent=float(np.mean(np.abs(differences))),
                          high_infill_region_relative_error_percent=differences[-1])
            valid = bool(item.converged and item.get('regularization_resolved', True))
            selected_final_maxdiff = None
            if index == chosen[0]:
                with np.load(inverse/f'nu_{cfg.PRIMARY_NU:.2f}'/'fields.npz') as final:
                    selected_final_maxdiff = float(np.max(abs(field-final['E_inverse'])))
            rows.append(dict(run=cfg.SAMPLE_RUN, gamma=item.gamma, display_level=display.get(index, -1),
                original_selected_gamma=original['selected_gamma'], original_selected=bool(item.selected),
                original_final_primary_converged=next(s['converged'] for s in original['solvers'] if s['nu']==cfg.PRIMARY_NU),
                scan_solver_converged=bool(item.converged), qualified=valid,
                final_gradient_inf=item.final_gradient_inf, gradient_tolerance_satisfied=bool(item.gradient_tolerance_satisfied),
                stop_message=item.stop_message, iterations=item.iterations, evaluations=item.evaluations, **result))
            x = mesh.plot_x[0] + cfg.MANUFACTURING_X_OFFSET_MM
            profiles.extend(dict(run=cfg.SAMPLE_RUN, gamma=item.gamma, manufacturing_x_mm=a,
                                 mean_E_MPa=b, transverse_spatial_sd_MPa=c) for a,b,c in zip(x,mean,sd))
            if index in display:
                models.append(dict(run=cfg.SAMPLE_RUN, level=display[index], gamma=item.gamma, x=x,
                    mean=mean, sd=sd, mesh=mesh, cfg=cfg, field=field.reshape(mesh.nods_y, mesh.nods_x),
                    qualified=valid, calibration=calibration))
            evidence.append(dict(gamma=item.gamma, field_path=str((trial/'fields.npz').resolve()),
                                 fields_sha256=file_sha256(trial/'fields.npz'), norms_verified=True,
                                 selected_scan_vs_final_maxdiff_MPa=selected_final_maxdiff))
        sources = [root/'01_prepared/run_metadata.json', root/'01_prepared/fields.npz',
                   inverse/'selection.csv', inverse/'run_metadata.json']
        records.append(dict(run=cfg.SAMPLE_RUN, root=str(root.resolve()),
            source_sha256={str(p.resolve()): file_sha256(p) for p in sources}, replayed_fields=evidence,
            nu=cfg.PRIMARY_NU, max_iter=cfg.MAX_ITER, max_fun=cfg.MAX_FUN,
            gamma_selection_unchanged=True, no_optimization_executed=True))
    table = pd.DataFrame(rows)
    reps = table[table.display_level>=0].sort_values(['run','display_level'])
    profile_table = pd.DataFrame(profiles)
    # Repeated loading comparison uses identical gamma, overlap and metric.
    pairs = []
    for level in range(4):
        selected_models = [m for m in models if m['level']==level]
        for i, first in enumerate(selected_models):
            for second in selected_models[i+1:]:
                distance, bounds = profile_distance(first['x'], first['mean'], second['x'], second['mean'])
                pairs.append(dict(level=level, first_run=first['run'], second_run=second['run'],
                    first_gamma=first['gamma'], second_gamma=second['gamma'],
                    symmetric_profile_difference_percent=distance, lower_x_mm=bounds[0],upper_x_mm=bounds[1]))
    pair_table = pd.DataFrame(pairs)
    for name, data in (('all_metrics',table),('representative_metrics',reps),
                       ('regional_comparison',pd.DataFrame(regions)),('profiles',profile_table),('repeat_comparison',pair_table)):
        data.to_csv(output/f'{name}.csv',index=False,lineterminator='\n')
    (output/'provenance.json').write_text(json.dumps(dict(cases=records,python=platform.python_version(),
        display_targets=DISPLAY_TARGETS,script_sha256=file_sha256(Path(__file__)),
        helpers_sha256={name:file_sha256(Path(__file__).with_name(name)) for name in
                        ('dic_gamma_sensitivity.py','material_window_analysis.py')},
        interval='1e-9 to 1e-8, plus next upper scan point 1.20679e-8',
        calibration_used_for_selection=False),ensure_ascii=False,indent=2),encoding='utf-8')
    draw(models,table,output)
    report(reps,pair_table,records,output)
    print(reps[['run','gamma','axial_relative_error_percent','displacement_relative_error_percent',
                'regional_mean_absolute_relative_error_percent','transverse_dispersion_percent']].to_string(index=False))


def draw(models,table,output):
    plt.rcParams.update({'font.family':'Microsoft YaHei','font.size':10,'axes.unicode_minus':False,
                         'svg.fonttype':'path','axes.spines.top':False,'axes.spines.right':False})
    runs=sorted(table.run.unique())
    fig,axes=plt.subplots(len(runs),1,figsize=(10,8),sharex=True,sharey=True,constrained_layout=True)
    for ax,run in zip(np.atleast_1d(axes),runs):
        for level,color in enumerate(COLORS):
            m=next(m for m in models if m['run']==run and m['level']==level)
            ax.plot(m['x'],m['mean'],color=color,lw=1.5,label=f'γ={m["gamma"]:.3g}'+('（原L-curve扫描点）' if level==0 else ''))
        edges=np.r_[m['x'][0],m['cfg'].REGION_BOUNDARIES_MANUFACTURING_X,m['x'][-1]]
        ax.plot(np.repeat(edges,2)[1:-1],np.repeat(m['calibration'].mean_E_MPa,2),color='.3',lw=1.1,ls='--',label='外部标定参照')
        for edge in edges[1:-1]:ax.axvline(edge,color='.8',lw=.6,ls=':')
        ax.set(title=f'第{run}次：40%–70%区段，ν=0.26',ylabel='E (MPa)')
        ax.legend(frameon=False,fontsize=8,ncol=3)
    np.atleast_1d(axes)[-1].set_xlabel('制造坐标 ξ (mm)，沿用各次原定位')
    save_figure(fig,output/'axial_profiles')
    fig,axes=plt.subplots(len(runs),3,figsize=(12,8),constrained_layout=True,squeeze=False)
    for row,run in enumerate(runs):
        data=table[table.run==run].sort_values('gamma')
        for ax,key,label in zip(axes[row],('displacement_relative_error_percent','regional_mean_absolute_relative_error_percent','transverse_dispersion_percent'),
                                ('双分量位移相对误差 (%)','四区模量平均绝对参照差 (%)','模量横向离散 / 面积均值 (%)')):
            ax.semilogx(data.gamma,data[key],'-o',lw=1.2,ms=3,color=COLORS[0])
            base=data[data.original_selected].iloc[0]
            ax.plot(base.gamma,base[key],'*',color=COLORS[1],ms=10)
            ax.axvspan(1e-9,1e-8,color='.8',alpha=.2)
            ax.set(xlabel='γ',ylabel=label,title=f'第{run}次')
    save_figure(fig,output/'tradeoff')
    fig,axes=plt.subplots(len(runs),4,figsize=(14,6),constrained_layout=True,squeeze=False)
    low,high=min(m['field'].min() for m in models),max(m['field'].max() for m in models)
    for m in models:
        ax=axes[runs.index(m['run']),m['level']];mesh=m['mesh']
        im=ax.pcolormesh(mesh.plot_x+m['cfg'].MANUFACTURING_X_OFFSET_MM,mesh.plot_y,m['field'],
                         shading='auto',cmap='viridis',vmin=low,vmax=high,rasterized=True)
        for edge in m['cfg'].REGION_BOUNDARIES_MANUFACTURING_X:ax.axvline(edge,color='white',lw=.5,ls='--')
        ax.set(title=f'第{m["run"]}次，γ={m["gamma"]:.2g}',xlabel='ξ (mm)',ylabel='y (mm)',aspect='equal')
    fig.colorbar(im,ax=axes,label='E (MPa)，共享色标',shrink=.8)
    save_figure(fig,output/'modulus_maps')


def report(reps,pairs,records,output):
    lines=['# 40%–70%区段：γ在10⁻⁹至10⁻⁸量级的核验','',
        '**结论：** 减小γ可改善拟合并减轻70%端的平滑偏差，但改善有限，曲线起伏和横向离散同时增大。建议先保留约8.29e-9与3.91e-9作稳定性核验候选，不直接采用最低1.26e-9，更不向1e-9以下扩展。本次未修改正式选γ规则，未确认这些候选为最优值。', '',
        '只回放三次裁剪运行已保存的扫描场。固定同一DIC、ROI、网格、载荷、参考图、面积权重及ν=0.26；不重新优化，不改原γ或根配置。候选为原扫描中1e-9至1e-8的6点，加上上邻点1.20679e-8，每次再加原L-curve点，共24场。', '',
        '图示代表点预先按目标8e-9、4e-9、1.3e-9取已有扫描中最近点，不按标定或图形挑选；区间内所有点均保留。标定仅用于事后比较，未引入新的正式选参规则。', '',
        '| 次数 | γ | 轴向位移误差 % | 双分量位移误差 % | 四区平均绝对参照差 % | 70%区参照差 % | 横向离散 % |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in reps.itertuples():
        lines.append(f'| {r.run} | {r.gamma:.3g} | {r.axial_relative_error_percent:.3f} | {r.displacement_relative_error_percent:.3f} | {r.regional_mean_absolute_relative_error_percent:.3f} | {r.high_infill_region_relative_error_percent:+.3f} | {r.transverse_dispersion_percent:.3f} |')
    lines+=['','较小γ使70%端模量上升、双分量位移拟合改善，同时横向变化增加。后者是空间离散，不能直接称为噪声。四区参照差不是材料真值误差。制造界面仅用于事后积分和绘图，不限制自由二维反演。', '',
        '## 第2、3次重复加载的宽度均值一致性','',
        '| γ方案 | 第2/3次曲线对称L₂差 % |','|---|---:|']
    for p in pairs[(pairs.first_run==2)&(pairs.second_run==3)].itertuples():
        label='各自原选γ' if p.level==0 else f'{p.first_gamma:.3g}'
        lines.append(f'| {label} | {p.symmetric_profile_difference_percent:.3f} |')
    lines+=['','曲线差在相同制造坐标重叠范围积分，分母为两曲线均值的L₂范数；按全部曲线节点分段的Gauss积分对线性宽度均值精确。不同参考图的加载记录不是独立试样。重复加载一致性不是时间扰动或初值稳定性试验。', '',
        '## 核验范围和限制','',
        '24场的面积加权残差范数及H1梯度范数分别与原扫描记录在相对1e-7及1e-8容差下重现。标定比较沿网格边和制造界面细分，以2×2 Gauss精确积分各区均值。正模量极值、空间离散、全部梯度∞范数和停止信息见all_metrics.csv。', '',
        '原选γ在本报告采用原扫描中正常停止的候选场。逐元素核验第1次最终场与该扫描场完全相同，第2/3次最大差分别为7.05e-12/2.27e-12 MPa。第1次最终重复优化曾线搜索异常，仍保留原失败记录，不能将本次回放改称修复。24个扫描场均满足函数下降停止，但均未满足严格GTOL；函数停止不等于参数稳定或物理正确。', '',
        '本次没有重做早/晚帧、初值扰动或噪声扰动下的反演，因而不能据此确认更细的模量起伏可重复。小γ还可能吸收边界、各向异性、DIC或力参考误差。保持原厚度/载荷/材料定位假设，以及裁剪端面均匀牵引近似。没有核验小于1e-9的更低系数。', '',
        '![轴向曲线](axial_profiles.png)','','![拟合与离散权衡](tradeoff.png)','','![二维场](modulus_maps.png)','',
        '## 复现','', '命令：`python -B -m lib.material_window_gamma_check --cases <三次裁剪运行目录> --output <新目录>`。已有输出拒绝覆盖；字段和冻结输入哈希见provenance.json。', '']
    lines += [f'- {r["root"]}' for r in records]
    (output/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases',nargs='+',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    analyze(args.cases,args.output)
