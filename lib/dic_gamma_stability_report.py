"""Postprocess the fixed-gamma frame/initialization study without optimizing."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy

from .dic_case_workflow import load_prepared
from .dic_gamma_stability import GAMMAS, field_distance
from .experiment_data import file_sha256


def save(fig, path):
    fig.savefig(path.with_suffix('.png'), dpi=300)
    fig.savefig(path.with_suffix('.svg'))
    plt.close(fig)


def analyze(study: Path, output: Path):
    output.mkdir(parents=True, exist_ok=False)
    rows, profiles, checks, protocols = [], [], [], []
    for run in (1, 2, 3):
        source = study/f'run{run}'
        assert json.loads((source/'progress.json').read_text())['state'] == 'complete'
        protocol = json.loads((source/'protocol.json').read_text(encoding='utf-8'))
        root = Path(protocol['source_case'])
        cfg, mesh, cases, _, _ = load_prepared(root/'01_prepared')
        for name, key in [('fields.npz', 'prepared_fields_sha256'), ('run_metadata.json', 'prepared_metadata_sha256')]:
            assert file_sha256(root/'01_prepared'/name) == protocol[key]
        selected = json.loads((root/'03_inverse/run_metadata.json').read_text(encoding='utf-8'))['selected_gamma']
        assert selected == protocol['original_selected_gamma']
        # Independent limiting-case checks of the integration-based relative norm.
        np.testing.assert_allclose(field_distance(mesh, np.ones(mesh.n_nod)*110, np.ones(mesh.n_nod)*100), 10.)
        np.testing.assert_allclose(field_distance(mesh, np.ones(mesh.n_nod)*100, np.ones(mesh.n_nod)*100), 0.)
        for index, gamma in enumerate(GAMMAS):
            full_path = source/f'g{index}_full_uniform/fields.npz'
            with np.load(full_path) as saved:
                full = saved['E_inverse'].copy()
            for name in ('full_uniform', 'early_uniform', 'late_uniform', 'full_perturbed'):
                destination = source/f'g{index}_{name}'
                if name == 'full_perturbed' and index != 1:
                    continue
                metric_path = destination/'metrics.json'
                metric = json.loads(metric_path.read_text(encoding='utf-8'))
                assert metric['run'] == run and metric['gamma'] == gamma
                with np.load(destination/'fields.npz') as saved:
                    field = saved['E_inverse'].copy()
                row = {k: v for k, v in metric.items() if k not in ('errors_on_splits', 'line_search_restarts')}
                row['field_change_from_full_uniform_percent'] = field_distance(mesh, field, full)
                # Objective values are comparable only for the identical full target.
                row['objective_change_from_saved_percent'] = (100*(metric['final_cost']/metric['saved_continuation_cost']-1)
                                                            if metric['split'] == 'full' else np.nan)
                for split, errors in metric['errors_on_splits'].items():
                    row.update({f'evaluation_{split}_{key}': value for key, value in errors.items()})
                rows.append(row)
                weights = np.ones(mesh.nods_y)
                weights[[0, -1]] = .5
                width_mean = np.average(field.reshape(mesh.nods_y, mesh.nods_x), axis=0, weights=weights)
                profiles.extend(dict(run=run, gamma=gamma, variant=name, manufacturing_x_mm=x, width_mean_E_MPa=y)
                                for x, y in zip(mesh.plot_x[0]-cfg.GEO_L/2, width_mean))
                checks.append(dict(metrics=str(metric_path.resolve()), metrics_sha256=file_sha256(metric_path),
                                   fields=str((destination/'fields.npz').resolve()), fields_sha256=file_sha256(destination/'fields.npz'),
                                   solver_log_sha256=file_sha256(destination/'solver.log')))
        protocols.append(protocol)
    table = pd.DataFrame(rows).sort_values(['run', 'gamma', 'split', 'initialization'])
    assert len(table) == 30
    table.to_csv(output/'all_fit_metrics.csv', index=False, lineterminator='\n')
    profile = pd.DataFrame(profiles)
    profile.to_csv(output/'width_mean_profiles.csv', index=False, lineterminator='\n')
    summaries = []
    for (run, gamma), data in table.groupby(['run', 'gamma'], sort=True):
        full = data[(data.split == 'full') & (data.initialization == 'uniform')].iloc[0]
        temporal = data[data.split.isin(['early', 'late'])]
        summaries.append(dict(run=run, gamma=gamma,
            axial_relative_error_percent=full.evaluation_full_axial_relative_error_percent,
            displacement_relative_error_percent=full.evaluation_full_displacement_relative_error_percent,
            temporal_target_max_change_percent=temporal.target_change_from_full_percent.max(),
            temporal_field_max_change_percent=temporal.field_change_from_full_uniform_percent.max(),
            initial_path_field_change_percent=full.field_change_from_saved_continuation_percent,
            full_objective_change_from_saved_percent=full.objective_change_from_saved_percent,
            all_stopping_flags_passed=bool(data.converged.all()),
            all_gtol_satisfied=bool(data.gradient_tolerance_satisfied.all())))
    summary = pd.DataFrame(summaries)
    summary.to_csv(output/'gamma_summary.csv', index=False, lineterminator='\n')
    plt.rcParams.update({'font.family': 'Microsoft YaHei', 'font.size': 10, 'svg.fonttype': 'path',
                         'axes.unicode_minus': False, 'axes.spines.top': False, 'axes.spines.right': False})
    colors = {'full_uniform': '#0072B2', 'early_uniform': '#D55E00', 'late_uniform': '#009E73', 'full_perturbed': '#CC79A7'}
    labels = {'full_uniform': '15帧，均匀初值', 'early_uniform': '前5帧', 'late_uniform': '后5帧', 'full_perturbed': '15帧，扰动初值'}
    styles = {'full_uniform': '-', 'early_uniform': '--', 'late_uniform': ':', 'full_perturbed': '-.'}
    fig, axes = plt.subplots(3, 3, figsize=(13, 8), sharex=True, sharey=True, constrained_layout=True)
    for row, run in enumerate((1, 2, 3)):
        for col, gamma in enumerate(GAMMAS):
            ax = axes[row, col]
            for variant in colors:
                part = profile[(profile.run == run) & (profile.gamma == gamma) & (profile.variant == variant)]
                if len(part):
                    fit = table[(table.run == run) & (table.gamma == gamma) &
                                (table.split == variant.rsplit('_', 1)[0]) &
                                (table.initialization == variant.rsplit('_', 1)[1])].iloc[0]
                    ax.plot(part.manufacturing_x_mm, part.width_mean_E_MPa, color=colors[variant],
                            ls=styles[variant], lw=1.4, label=labels[variant]+('（异常停止）' if not fit.converged else ''))
                    if not fit.converged:
                        ax.text(.02, .92, '前5帧：异常停止，结果仅作诊断', color='#D55E00',
                                transform=ax.transAxes, fontsize=8)
            ax.set(title=f'第{run}次，γ={gamma:.2g}', xlim=(-57, 57))
            if col == 0:
                ax.set_ylabel('宽度均值 E (MPa)')
            if row == 2:
                ax.set_xlabel('制造坐标 ξ (mm)')
    axes[0, 1].legend(frameon=False, fontsize=8, ncol=2)
    save(fig, output/'frame_initial_profiles')
    fig, axes = plt.subplots(3, 2, figsize=(10, 8), constrained_layout=True)
    for row, run in enumerate((1, 2, 3)):
        data = summary[summary.run == run].sort_values('gamma')
        ax = axes[row, 0]
        ax.semilogx(data.gamma, data.temporal_field_max_change_percent, 'o-', color='#D55E00', label='模量场变化')
        ax.semilogx(data.gamma, data.temporal_target_max_change_percent, 's--', color='#0072B2', label='DIC目标变化')
        failed = data[~data.all_stopping_flags_passed]
        if len(failed):
            ax.semilogx(failed.gamma, failed.temporal_field_max_change_percent, 'x', color='black', ms=9,
                        label='含异常停止，仅诊断')
        ax.set(title=f'第{run}次：前/后5帧相对15帧的最大变化', ylabel=r'面积加权 $L_2$ 相对变化 (%)', xlabel='γ')
        ax.legend(frameon=False, fontsize=8)
        ax = axes[row, 1]
        ax.semilogx(data.gamma, data.displacement_relative_error_percent, 'o-', color='#0072B2', label='双分量')
        ax.semilogx(data.gamma, data.axial_relative_error_percent, 's--', color='#009E73', label='轴向')
        ax.set(title=f'第{run}次：15帧回代拟合', ylabel='位移相对误差 (%)', xlabel='γ')
        ax.legend(frameon=False, fontsize=8)
    save(fig, output/'stability_tradeoff')
    lines = ['# sample_2：小γ的帧内与初值稳定性检查', '',
        '**结论：** 在本次固定模型与原完整ROI内，降低γ到1e-9量级后，拟合改善未伴随明显的帧内或初值不稳定。三次在γ=1.26e-9下，轴向位移相对误差为1.185/0.857/0.763%，前后帧导致的模量场相对变化为0.144/0.211/0.090%。这些是二维积分场差异，不是仅凭曲线重合判断。', '',
        '**建议：** 展示折中结果可采用γ=3.91e-9，同时保留1.26e-9的细节敏感性对照；前者比原约1e-7更少平滑，后者拟合误差更低但起伏更明显。本检查没有给出正式最优γ。局部起伏在不同帧中重现，只能排除本窗口内较大的帧敏感性，仍可能来自共同的观测或模型偏差，不能认定是真实材料细节。', '',
        '固定原完整ROI、121×19二维H1自由度、ν=0.26、载荷与求解预算。三次试验分别比较三个事先固定的γ；每个γ用15帧、前5帧、后5帧各求解一次，中间γ再以平滑扰动初值求解。共30次，未用标定值参与初值、目标、排序或选γ。', '',
        '这是原完整ROI结果的稳定性研究，未采用当前日常配置中的40%–70%裁剪ROI；不混合两个几何域的γ和误差。正式L-curve选择与原数值结果保持不变。', '',
        '| 次数 | γ | 轴向位移相对误差 % | 双分量位移相对误差 % | 帧变化最大 % | 模量场变化最大 % | 均匀初值与原延续路径的场差异 % |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in summary.sort_values(['run', 'gamma'], ascending=[True, False]).itertuples():
        lines.append(f'| {r.run} | {r.gamma:.3g} | {r.axial_relative_error_percent:.3f} | {r.displacement_relative_error_percent:.3f} | {r.temporal_target_max_change_percent:.4f} | {r.temporal_field_max_change_percent:.4f} | {r.initial_path_field_change_percent:.6f} |')
    perturb = table[table.initialization == 'perturbed']
    lines += ['', '中间γ的扰动初值与均匀初值模量场相对差异（第1/2/3次）：'+
              ' / '.join(f'{value:.6f}%' for value in perturb.sort_values('run').field_change_from_full_uniform_percent)+ '。', '',
        f'30次中优化停止标志通过{int(table.converged.sum())}次；严格梯度阈值通过{int(table.gradient_tolerance_satisfied.sum())}次。梯度无穷范数范围为{table.final_gradient_inf.min():.3g}–{table.final_gradient_inf.max():.3g}，配置GTOL=1e-12。函数下降停止不能解释为严格一阶最优性或真实材料恢复。', '',
        '异常停止为第2次、γ=1.21e-8、前5帧组：175次迭代、313次目标评价后返回ABNORMAL，未提高预算或替换结果；该组仅作诊断。其余29组均因函数相对下降停止。三个γ的15帧均匀初值解与原延续路径场差异均小于0.000036%，同目标函数值差异仅处于约1e-10%的数值量级。', '',
        '## 定义与解释范围', '',
        '位移相对误差为面积加权残差L₂范数/去刚体运动后的DIC变形L₂范数，轴向、横向、双分量分别计算；不逐点相除。模量场差异为完整二维Q4场的面积积分L₂差异/基准场L₂范数，不先进行一维化。前后帧场差异以同γ的15帧均匀初值场为基准，初值路径差异以已有扫描场为基准。', '',
        '前/后5帧均来自原15帧，具有时间相关性，也都已参与过原识别。本检查只能评价帧内重复性与初值敏感性，不能称为独立留出验证。帧间漂移还可能包含真实蠕变或载荷变化；各窗口使用相同假设力，不能将其完全解释为DIC噪声。独立y行自由度始终保留；图中仅展示宽度均值。', '',
        '厚度3.2 mm、名义末载减记录首载、首载对应参考图及标线中点定位等原假设仍有效。稳定性不能排除共同存在的力、尺度、边界或本构偏差。未更改正式选参规则，也未将这三个候选之一宣称为已验证最优γ。', '',
        '![帧与初值下的宽度均值](frame_initial_profiles.png)', '', '![稳定性和拟合误差](stability_tradeoff.png)', '',
        '## 核验与复现', '',
        '逐次从原始15帧重建坐标和平均位移，与冻结输入在1e-10 mm和1e-12 mm绝对容差内核对；前后5帧使用相同点集、权重、观测算子及统一初值。每个优化结果重新正演，残差范数与求解器结果在1e-8相对容差内核对。模量积分差异通过常值场10%差异和同场零差异两项解析核验。原冻结输入哈希及原选定γ再次核对未变。', '',
        '运行模块：`python -u -B -m lib.dic_gamma_stability --case <冻结完整ROI目录> --output <新的单次目录>`；三次各自执行。后处理：`python -B -m lib.dic_gamma_stability_report --study <本研究目录> --output <新的汇总目录>`。', '',
        '逐次protocol.json记录帧源哈希、初值和预算，逐优化metrics.json/fields.npz/solver.log保留停止状态、历史与数值场。all_fit_metrics.csv包含所有交叉帧误差与初值差异，gamma_summary.csv为本表数据。']
    (output/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    provenance = dict(protocols=protocols, results=checks, calibration_used=False,
        frozen_inputs_and_selected_gamma_verified_unchanged=True,
        environment=dict(python=sys.version, numpy=np.__version__, scipy=scipy.__version__, pandas=pd.__version__,
                         matplotlib=matplotlib.__version__, platform=platform.platform()),
        reproduction_commands=[f'python -u -B -m lib.dic_gamma_stability --case "{p["source_case"]}" --output <NEW_STUDY>/run{p["run"]}' for p in protocols],
        report_script_sha256=file_sha256(Path(__file__)))
    (output/'provenance.json').write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding='utf-8')
    print(summary.to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    analyze(args.study, args.output)
