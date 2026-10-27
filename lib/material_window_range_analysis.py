"""Compare completed cropped H1 scans after changing the upper gamma bound.

This only reads saved solves and independently replays selected/diagnostic fields.
No failed L-curve selection is replaced by a diagnostic point.
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


def analyze(roots: list[Path], baselines: list[Path], output: Path):
    if len(roots) != len(baselines):
        raise ValueError('Each new scan needs its original-range baseline')
    output.mkdir(parents=True, exist_ok=True)
    if (output/'report.md').exists() or (output/'metrics.csv').exists():
        raise FileExistsError('Choose a new report directory')
    rows, regions, statuses, plots, provenance = [], [], [], [], []
    profile_rows = []
    for root, base in zip(roots, baselines):
        cfg, mesh, cases, prepared, _ = load_prepared(root/'01_prepared')
        old_cfg, _, _, old_prepared, _ = load_prepared(base/'01_prepared')
        if len(cases) != 1 or not hasattr(cfg, 'MATERIAL_WINDOW_MM'):
            raise ValueError('Expected one cropped observation case')
        case, run = cases[0], cfg.SAMPLE_RUN
        new_meta = json.loads((root/'03_inverse/run_metadata.json').read_text(encoding='utf-8'))
        old_meta = json.loads((base/'03_inverse/run_metadata.json').read_text(encoding='utf-8'))
        if new_meta['status'] == 'running':
            raise ValueError('Wait for the entire scan to finish')
        changed = {k for k in old_prepared['configuration']
                   if old_prepared['configuration'][k] != prepared['configuration'][k]}
        if not changed <= {'RUN_NAME', 'CASE_ID', 'CASE_OUTPUT', 'GAMMA_LOG10_RANGE', 'GAMMA_CANDIDATES'}:
            raise ValueError(f'Unmatched configuration settings: {changed}')
        with np.load(root/'01_prepared/fields.npz') as new, np.load(base/'01_prepared/fields.npz') as old:
            if new.files != old.files:
                raise ValueError('Different preparation arrays')
            for name in new.files:
                np.testing.assert_array_equal(new[name], old[name])
        scan = pd.read_csv(root/'03_inverse/selection.csv')
        old_scan = pd.read_csv(base/'03_inverse/selection.csv')
        if len(scan) != cfg.GAMMA_COUNT:
            raise ValueError('Incomplete gamma scan')
        if prepared['calibration'] != old_prepared['calibration']:
            raise ValueError('External calibration reference changed')
        positive = scan[np.isfinite(scan.curvature) & (scan.curvature > 0)]
        peak_gamma = float(positive.loc[positive.curvature.idxmax(), 'gamma']) if len(positive) else None
        statuses.append(dict(run=run, status=new_meta['status'], l_curve_status=new_meta['l_curve_status'],
            l_curve_message=new_meta['l_curve_message'], selected_gamma=new_meta['selected_gamma'],
            curvature_peak_gamma=peak_gamma,
            original_selected_gamma=old_meta['selected_gamma'], scan_count=len(scan),
            scan_success_count=int(scan.converged.sum()), old_scan_success_count=int(old_scan.converged.sum()),
            gamma_min=float(scan.gamma.min()), gamma_max=float(scan.gamma.max()),
            log10_spacing=float(np.diff(np.sort(np.log10(scan.gamma)))[0]),
            old_log10_spacing=float(np.diff(np.sort(np.log10(old_scan.gamma)))[0])))
        choices = [('original_selected', base, old_scan[old_scan.selected].iloc[0])]
        if new_meta['selected_gamma'] is not None:
            choices.append(('new_selected', root, scan[scan.selected].iloc[0]))
        # Fixed diagnostic targets are independent of all evaluation results.
        for target in (1e-8, 1e-9):
            item = scan.iloc[int(np.argmin(abs(np.log(scan.gamma.to_numpy()/target))))]
            choices.append((f'diagnostic_target_{target:g}', root, item))
        calibration = pd.DataFrame(prepared['calibration'])
        points, weights = common_quadrature(mesh, mesh, np.zeros(2), cfg.REGION_BOUNDARIES_X)
        groups = np.searchsorted(cfg.REGION_BOUNDARIES_X, points[:, 0], side='right')
        curves, evidence = [], []
        for role, source, item in choices:
            directory = source/'03_inverse/l_curve'/f'gamma_{item.gamma:g}'
            if isinstance(item.get('solution_directory'), str):
                directory = source/'03_inverse'/item.solution_directory
            fields = directory/'fields.npz'
            with np.load(fields) as saved:
                field = saved['E_inverse'].copy()
            result, _, mean, sd = inspect_field(cfg, mesh, case, field, calibration)
            np.testing.assert_allclose(result['replay_residual_norm'], item.residual_norm, rtol=1e-7, atol=1e-10)
            np.testing.assert_allclose(result['replay_regularization_norm'], item.regularization_norm, rtol=1e-8, atol=1e-10)
            values = scalar_values(mesh, field, points)
            differences = []
            for group, record in enumerate(calibration.to_dict(orient='records')):
                mask = groups == group
                average = float(np.average(values[mask], weights=weights[mask]))
                difference = 100*(average/record['mean_E_MPa']-1)
                differences.append(difference)
                regions.append(dict(run=run, role=role, gamma=item.gamma, **record,
                                    inverse_mean_E_MPa=average, difference_percent=difference))
            result['regional_mean_absolute_relative_error_percent'] = float(np.mean(np.abs(differences)))
            result['high_infill_region_relative_error_percent'] = differences[-1]
            rows.append(dict(run=run, role=role, gamma=item.gamma, scan_converged=bool(item.converged),
                final_gradient_inf=item.final_gradient_inf, gradient_tolerance_satisfied=bool(item.gradient_tolerance_satisfied),
                stop_message=item.stop_message, **result))
            x = mesh.plot_x[0]+cfg.MANUFACTURING_X_OFFSET_MM
            curves.append(dict(role=role, gamma=item.gamma, x=x, mean=mean))
            profile_rows.extend(dict(run=run, role=role, gamma=item.gamma, manufacturing_x_mm=a,
                                     mean_E_MPa=b, transverse_spatial_sd_MPa=c) for a,b,c in zip(x,mean,sd))
            evidence.append(dict(role=role, fields_path=str(fields.resolve()), sha256=file_sha256(fields), replay_verified=True))
        plots.append(dict(run=run, scan=scan, old_scan=old_scan, curves=curves, cfg=cfg,
                          calibration=calibration, status=new_meta['l_curve_status']))
        evidence_dir = output/f'run{run}_evidence'
        evidence_dir.mkdir(exist_ok=False)
        for source, name in ((root/'01_prepared/configuration_snapshot.py', 'configuration_snapshot.py'),
                             (root/'01_prepared/run_metadata.json', 'prepared_metadata.json'),
                             (root/'03_inverse/run_metadata.json', 'inverse_metadata.json')):
            (evidence_dir/name).write_bytes(source.read_bytes())
        scan.to_csv(evidence_dir/'selection.csv', index=False, lineterminator='\n')
        source_files = [root/'01_prepared/run_metadata.json', root/'01_prepared/fields.npz',
                        root/'03_inverse/run_metadata.json', root/'03_inverse/selection.csv',
                        base/'03_inverse/run_metadata.json', base/'03_inverse/selection.csv']
        provenance.append(dict(run=run, root=str(root.resolve()), baseline=str(base.resolve()),
            input_arrays_identical=True, changed_configuration=sorted(changed),
            source_sha256={str(p.resolve()): file_sha256(p) for p in source_files}, replayed_fields=evidence))
    table = pd.DataFrame(rows)
    status_table = pd.DataFrame(statuses)
    for name,data in (('metrics',table),('statuses',status_table),('regional_comparison',pd.DataFrame(regions)),
                      ('profiles',pd.DataFrame(profile_rows))):
        data.to_csv(output/f'{name}.csv',index=False,lineterminator='\n')
    (output/'provenance.json').write_text(json.dumps(dict(cases=provenance,
        python=platform.python_version(), script_sha256=file_sha256(Path(__file__)),
        helper_sha256={name:file_sha256(Path(__file__).with_name(name)) for name in
                       ('dic_gamma_sensitivity.py','material_window_analysis.py')},
        calibration_used_for_selection=False, no_optimization_in_postprocessing=True),
        ensure_ascii=False,indent=2),encoding='utf-8')
    draw(plots,output)
    write_report(status_table,table,provenance,output)
    print(status_table.to_string(index=False))
    print(table[['run','role','gamma','displacement_relative_error_percent',
                 'regional_mean_absolute_relative_error_percent']].to_string(index=False))


def draw(plots,output):
    plt.rcParams.update({'font.family':'Microsoft YaHei','font.size':10,'axes.unicode_minus':False,
                         'svg.fonttype':'path','axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(len(plots),2,figsize=(11,8),constrained_layout=True,squeeze=False)
    for row,p in enumerate(plots):
        for data,color,label in ((p['old_scan'],'.6','原范围1e-12～1e-4'),(p['scan'],'#0072B2','新范围1e-12～1e-7')):
            good=data.converged
            axes[row,0].loglog(data.residual_norm.where(good),data.regularization_norm.where(good),'-o',color=color,ms=2,label=label)
            axes[row,1].semilogx(data.gamma,data.curvature,'-o',color=color,ms=2,label=label)
            selected=data[data.selected]
            if len(selected):
                axes[row,0].plot(selected.residual_norm,selected.regularization_norm,'*',color=color,ms=10)
                axes[row,1].plot(selected.gamma,selected.curvature,'*',color=color,ms=10)
        axes[row,0].set(xlabel='位移残差范数 ρ',ylabel='H1梯度范数 η',title=f'第{p["run"]}次 L-curve')
        axes[row,1].set(xlabel='γ',ylabel='离散正曲率',title=f'新扫描状态：{p["status"]}')
        axes[row,1].axvline(1e-7,color='.5',ls=':',lw=.7)
        axes[row,0].legend(frameon=False,fontsize=8)
    save_figure(fig,output/'lcurve_comparison')
    fig,axes=plt.subplots(len(plots),1,figsize=(10,8),constrained_layout=True,sharex=True,sharey=True)
    colors={'original_selected':'#0072B2','new_selected':'#D55E00',
            'diagnostic_target_1e-08':'#009E73','diagnostic_target_1e-09':'#CC79A7'}
    for ax,p in zip(np.atleast_1d(axes),plots):
        for c in p['curves']:
            label={'original_selected':'原选点','new_selected':'新选点'}.get(c['role'],'新扫描诊断点')
            ax.plot(c['x'],c['mean'],color=colors[c['role']],lw=1.5,label=f'{label} γ={c["gamma"]:.3g}')
        edges=np.r_[c['x'][0],p['cfg'].REGION_BOUNDARIES_MANUFACTURING_X,c['x'][-1]]
        ax.plot(np.repeat(edges,2)[1:-1],np.repeat(p['calibration'].mean_E_MPa,2),'--',color='.3',lw=1.1,label='标定参照')
        for edge in edges[1:-1]:ax.axvline(edge,color='.8',lw=.6,ls=':')
        ax.set(title=f'第{p["run"]}次：{p["status"]}',ylabel='E (MPa)')
        ax.legend(frameon=False,fontsize=8,ncol=2)
    np.atleast_1d(axes)[-1].set_xlabel('制造坐标 ξ (mm)')
    save_figure(fig,output/'profiles')


def write_report(statuses,metrics,provenance,output):
    lines=['# 40%–70%区段：将γ上限从1e-4降至1e-7重新扫描','',
        '**结论：** 上限降低三个数量级后，三次主曲率峰仍在约5e-8至8e-8。第1次选得6.25055e-8，位移及参照差变化很小；第2次因峰邻近候选线搜索失败而缺少完整支撑，第3次因峰过近上界而未选参。此次重算没有得到10⁻⁹至10⁻⁸量级的正式L-curve选值。', '',
        '三次独立参考图分别重新计算。仅修改扫描上限和运行身份；下限1e-12、50个候选、121×19单元、目标、ν、初值规则、容差和1500/12000预算保持不变。准备数组逐元素完全一致，原始数据及旧结果保留。运行过程中未调整参数或按中间结果重启。', '',
        '| 次数 | 新扫描状态 | 原选γ | 新选γ | 曲率峰γ（诊断） | 函数停止成功点/50 |','|---|---|---:|---:|---:|---:|']
    for r in statuses.itertuples():
        gamma=f'{r.selected_gamma:.6g}' if pd.notna(r.selected_gamma) else '未选出'
        peak=f'{r.curvature_peak_gamma:.6g}' if pd.notna(r.curvature_peak_gamma) else '无'
        lines.append(f'| {r.run} | {r.l_curve_status} | {r.original_selected_gamma:.6g} | {gamma} | {peak} | {r.scan_success_count}/50 |')
    lines += ['', 'L-curve必须有受支持的内部正曲率峰；若最大曲率靠近上界或邻近候选失败，则保存失败，不强制选择边界或残差最小点。没有选出γ的运行不能称为完成了正式反演；下表固定γ场只作诊断。', '',
        '| 次数 | 场的身份 | γ | 轴向位移误差 % | 双分量位移误差 % | 四区平均绝对参照差 % | 横向离散 % |',
        '|---|---|---:|---:|---:|---:|---:|']
    for r in metrics.itertuples():
        role={'original_selected':'原扫描选点','new_selected':'新扫描选点'}.get(r.role,'新扫描固定γ诊断')
        lines.append(f'| {r.run} | {role} | {r.gamma:.6g} | {r.axial_relative_error_percent:.3f} | {r.displacement_relative_error_percent:.3f} | {r.regional_mean_absolute_relative_error_percent:.3f} | {r.transverse_dispersion_percent:.3f} |')
    lines += ['', '位移分母为同一DIC去刚体变形量；模量参照差以四区面积均值相对1023约束标定参照的绝对差求均值。参照不进入目标、初值或选γ。横向离散是去除宽度均值后的面积RMS/模量面积均值，不是噪声或置信区间。', '',
        '50点覆盖范围变窄后，对数间距由8/49变为5/49，实际候选γ网格也改变；首次求解的γ从1e-4变成1e-7，收敛延续路径随之改变。这是改变上限的实际重算，不能将所有差异只归因于截断原曲线。', '',
        '各图及统计中的诊断点均按预设目标1e-8/1e-9取最近候选，不按标定或重构形状选点。原选点与新选点采用相应扫描场；最终各ν优化的停止信息保存在逆问题元数据中。严格GTOL与函数下降停止分别保留，原失败不会因重绘被改称成功。', '',
        '代表场重新正演及重新积分H1梯度范数，分别在相对1e-7及1e-8容差下与扫描记录吻合。输入、字段和元数据哈希见provenance.json；全部新50点扫描及冻结配置见run*_evidence。原载荷参考、厚度、材料定位及截取端均匀牵引假设保留；本平台回代是拟合检查。', '',
        '![L-curve及曲率](lcurve_comparison.png)','','![选点与诊断场](profiles.png)','',
        '## 运行目录与复现','',
        '四步仍分别运行项目原入口。后处理：`python -B -m lib.material_window_range_analysis --cases <三个新运行目录> --baselines <三个原裁剪目录> --output <新报告目录>`。', '']
    lines += [f'- {p["root"]}' for p in provenance]
    (output/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases',type=Path,nargs='+',required=True)
    parser.add_argument('--baselines',type=Path,nargs='+',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    analyze(args.cases,args.baselines,args.output)
