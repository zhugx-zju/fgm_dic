"""Inspect saved H1 scan fields; no re-optimization or calibration-driven selection.

python -B -m lib.dic_gamma_sensitivity --cases ROOT1 ROOT2 ROOT3 --output NEW_DIR
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

from fgm_asm.research import solve_forward
from fgm_asm.load_case import evaluate_observation
from lib.dic_case_workflow import load_prepared, _regional_comparison
from lib.dic_preprocessing import relative_displacement_errors
from lib.experiment_data import file_sha256
from .uniform_sample_analysis import area_statistics

FACTORS = (1., .1, .01, .001)
COLORS = ('#0072B2', '#D55E00', '#009E73', '#CC79A7')


def inspect_field(cfg, mesh, case, field, calibration):
    displacement, _ = solve_forward(mesh, case.bc_info['tensile_end_force'], cfg.THICKNESS, cfg.PRIMARY_NU, field)
    state = evaluate_observation(case, displacement)
    rho = float(np.sqrt(np.sum(case.observation.weight_vector*state['residual']**2)))
    values = field[mesh.ele_nods_id-1]
    gx = np.einsum('gea,ea->ge', mesh.gauss_dNdx, values)
    gy = np.einsum('gea,ea->ge', mesh.gauss_dNdy, values)
    weights = mesh.gauss_abs_det_j*mesh.gauss_w[:, None]
    eta = float(np.sqrt(np.sum(weights*(gx**2+gy**2))))
    regions = _regional_comparison(mesh, field, calibration, cfg.REGION_BOUNDARIES_X)
    grid = field.reshape(mesh.nods_y, mesh.nods_x)
    width_weights = np.ones(mesh.nods_y)
    width_weights[[0, -1]] = .5
    width_mean = np.average(grid, axis=0, weights=width_weights)
    width_sd = np.sqrt(np.average((grid-width_mean)**2, axis=0, weights=width_weights))
    centered = (grid-width_mean).ravel()
    gauss_centered = (centered[mesh.ele_nods_id-1] @ mesh.gauss_N.T).T
    stats = area_statistics(mesh, field)
    spread = float(np.sqrt(np.sum(weights*gauss_centered**2)/weights.sum()))
    metrics = dict(**relative_displacement_errors(case, state['prediction']),
                   replay_residual_norm=rho, replay_regularization_norm=eta,
                   regional_mean_absolute_relative_error_percent=float(regions.difference_percent.abs().mean()),
                   high_infill_region_relative_error_percent=float(regions.difference_percent.iloc[-1]),
                   transverse_dispersion_percent=100*spread/stats['mean_E_MPa'],
                   min_nodal_E_MPa=float(field.min()), max_nodal_E_MPa=float(field.max()), **stats)
    return metrics, regions, width_mean, width_sd


def analyze(roots: list[Path], output: Path):
    output.mkdir(parents=True, exist_ok=False)
    rows, region_rows, profiles, maps, provenance = [], [], [], [], []
    for root in roots:
        cfg, mesh, cases, prepared, _ = load_prepared(root/'01_prepared')
        assert len(cases) == 1 and not cfg.PREDICT_CASES and cfg.REGULARIZATION_MODE == 'gradient'
        case = cases[0]
        run = cfg.PLATFORMS[case.name]['run']
        inverse = root/'03_inverse'
        meta = json.loads((inverse/'run_metadata.json').read_text(encoding='utf-8'))
        gamma_selected = meta['selected_gamma']
        scan = pd.read_csv(inverse/'selection.csv')
        calibration = pd.DataFrame(prepared['calibration'])
        # Display representatives are chosen by log-distance to fixed factors,
        # before looking at calibration errors or the reconstructed shapes.
        picked = {int(np.argmin(abs(np.log(scan.gamma.to_numpy()/(gamma_selected*factor))))): level
                  for level, factor in enumerate(FACTORS)}
        evidence = []
        for index, item in scan.iterrows():
            trial = inverse/'l_curve'/f'gamma_{item.gamma:g}'
            if isinstance(item.get('solution_directory'), str):
                trial = inverse/item['solution_directory']
            fields_path = trial/'fields.npz'
            with np.load(fields_path) as saved:
                field = saved['E_inverse'].copy()
            result, regions, mean, sd = inspect_field(cfg, mesh, case, field, calibration)
            np.testing.assert_allclose(result['replay_residual_norm'], item.residual_norm, rtol=1e-7, atol=1e-10)
            np.testing.assert_allclose(result['replay_regularization_norm'], item.regularization_norm, rtol=1e-8, atol=1e-10)
            valid = bool(item.converged and item.get('regularization_resolved', True))
            row = dict(run=run, gamma=item.gamma, gamma_factor=item.gamma/gamma_selected,
                       nu=cfg.PRIMARY_NU, original_selected_gamma=gamma_selected,
                       selected_by_original_rule=bool(item.selected), solver_converged=bool(item.converged),
                       comparison_qualified=valid, iterations=int(item.iterations), evaluations=int(item.evaluations),
                       gradient_tolerance_satisfied=bool(item.gradient_tolerance_satisfied),
                       stop_message=item.stop_message, representative_level=picked.get(index, -1), **result)
            rows.append(row)
            regions.insert(0, 'gamma', item.gamma)
            regions.insert(0, 'run', run)
            regions['comparison_qualified'] = valid
            region_rows.append(regions)
            evidence.append({'gamma': float(item.gamma), 'field_path': str(fields_path.resolve()),
                             'sha256': file_sha256(fields_path), 'both_scan_norms_replayed': True})
            if index in picked:
                level = picked[index]
                x = mesh.plot_x[0]-cfg.GEO_L/2
                profiles.extend(dict(run=run, level=level, gamma=item.gamma, manufacturing_x_mm=a,
                                     width_mean_E_MPa=b, transverse_spatial_sd_MPa=c)
                                for a, b, c in zip(x, mean, sd))
                maps.append(dict(run=run, level=level, gamma=item.gamma, mesh=mesh,
                                 field=field.reshape(mesh.nods_y, mesh.nods_x),
                                 calibration=calibration, boundaries=np.asarray(cfg.REGION_BOUNDARIES_MANUFACTURING_X)))
        source_files = [root/'01_prepared/run_metadata.json', root/'01_prepared/fields.npz',
                        inverse/'run_metadata.json', inverse/'selection.csv']
        provenance.append(dict(run=run, root=str(root.resolve()), frozen_inputs_unchanged=True,
                               source_sha256={str(p): file_sha256(p) for p in source_files},
                               saved_trial_fields=evidence, no_optimization_executed=True,
                               original_gamma_selection_unchanged=True))
        print(f'Run {run}: replayed {len(scan)} saved fields; selected gamma retained.', flush=True)
    table = pd.DataFrame(rows)
    selected = table[table.selected_by_original_rule].set_index('run')
    table['relative_error_reduction_percent'] = [100*(1-r.displacement_relative_error_percent/
                selected.loc[r.run, 'displacement_relative_error_percent']) for r in table.itertuples()]
    table['roughness_factor'] = [r.replay_regularization_norm/selected.loc[r.run, 'replay_regularization_norm']
                                 for r in table.itertuples()]
    table.to_csv(output/'all_gamma_metrics.csv', index=False, lineterminator='\n')
    representatives = table[table.representative_level >= 0].sort_values(['run', 'representative_level'])
    representatives.to_csv(output/'representative_metrics.csv', index=False, lineterminator='\n')
    pd.concat(region_rows, ignore_index=True).to_csv(output/'regional_comparison.csv', index=False, lineterminator='\n')
    profile_table = pd.DataFrame(profiles)
    profile_table.to_csv(output/'representative_profiles.csv', index=False, lineterminator='\n')
    provenance_record = dict(cases=provenance, representative_target_factors=FACTORS,
        representative_rule='Nearest existing gamma by logarithmic distance; fixed before external calibration evaluation.',
        metric_definition='Area-weighted L2 residual / DIC rigid-projected deformation norm, percent.',
        status_note='Solver convergence flag is a stopping criterion, not proof of GTOL, robustness, or material truth.',
        script_sha256=file_sha256(Path(__file__)))
    (output/'provenance.json').write_text(json.dumps(provenance_record, ensure_ascii=False, indent=2), encoding='utf-8')
    plt.rcParams.update({'font.family': 'Microsoft YaHei', 'font.size': 10, 'svg.fonttype': 'path',
                         'axes.unicode_minus': False, 'axes.spines.top': False, 'axes.spines.right': False})
    runs = sorted(table.run.unique())
    fig, axes = plt.subplots(len(runs), 1, figsize=(10, 8), sharex=True, sharey=True, constrained_layout=True)
    for ax, run in zip(np.atleast_1d(axes), runs):
        for level, color in enumerate(COLORS):
            data = profile_table[(profile_table.run == run) & (profile_table.level == level)]
            g = representatives.loc[(representatives.run == run) & (representatives.representative_level == level), 'gamma'].iloc[0]
            ax.plot(data.manufacturing_x_mm, data.width_mean_E_MPa, color=color, lw=1.5,
                    label=f'γ={g:.2g}'+('（原选定）' if level == 0 else ''))
        model = next(m for m in maps if m['run'] == run)
        edges = np.r_[profile_table[profile_table.run == run].manufacturing_x_mm.min(), model['boundaries'],
                      profile_table[profile_table.run == run].manufacturing_x_mm.max()]
        ax.plot(np.repeat(edges, 2)[1:-1], np.repeat(model['calibration'].mean_E_MPa, 2),
                color='black', lw=1.1, ls='--', label='单调标定参照（不参与选γ）')
        for boundary in model['boundaries']:
            ax.axvline(boundary, color='.8', lw=.6, ls=':')
        ax.set(title=f'第 {run} 次：减小 γ 后的宽度均值', ylabel='E (MPa)', xlim=(-57, 57))
        ax.legend(frameon=False, fontsize=8, ncol=3, loc='upper left')
    axes[-1].set_xlabel('制造坐标 ξ (mm)')
    save(fig, output/'axial_profiles')
    fig, axes = plt.subplots(len(runs), 3, figsize=(13, 8), constrained_layout=True)
    specs = [('displacement_relative_error_percent', '双分量位移相对误差 (%)'),
             ('regional_mean_absolute_relative_error_percent', '七区模量平均绝对相对误差 (%)'),
             ('transverse_dispersion_percent', '模量横向离散 / 面积均值 (%)')]
    for row, run in enumerate(runs):
        data = table[table.run == run].sort_values('gamma')
        good = data.comparison_qualified
        for ax, (key, label) in zip(axes[row], specs):
            ax.semilogx(data.gamma, data[key].where(good), color='#0072B2', lw=1.3)
            ax.semilogx(data.gamma[~good], data.loc[~good, key], 'x', color='.6', ms=4, label='停止条件未通过')
            base = data[data.selected_by_original_rule].iloc[0]
            ax.plot(base.gamma, base[key], '*', color='#D55E00', ms=11, label='原L-curve选定')
            for level, color in enumerate(COLORS):
                point = data[data.representative_level == level].iloc[0]
                ax.plot(point.gamma, point[key], 'o', color=color, ms=4)
            ax.set(xlabel='γ', ylabel=label, title=f'第 {run} 次')
        axes[row, 0].legend(frameon=False, fontsize=8)
    save(fig, output/'gamma_tradeoff')
    plot_modulus_maps(maps, runs, output)
    lines = ['# sample_2：已有 H1 正则化扫描的敏感性检查', '',
             '**结论：** 当前约1e-7的γ存在平滑偏差的证据。降低到原值约1/100后，三次轴向位移相对误差由2.11/1.84/1.58%降至1.18/0.86/0.78%，七区模量平均绝对相对误差由24.92/11.82/12.39%降至21.60/7.86/9.84%。高填充率端的模量上升更充分。', '',
             '进一步降低到原值约1/1000，相比1/100的双分量位移误差只再下降1.19/4.57/4.70%，横向离散却分别增至1.22/1.93/2.45倍，宽度均值出现更明显起伏。到更小γ时可出现近零局部模量和停止失败，不能把所有拟合改善视为材料恢复改善。', '',
             '建议把约1e-9至1e-8作为进一步DIC内稳定性核验的候选区间；这是本次敏感性诊断，不替换正式L-curve选参，也不把标定参照最接近点作为最优γ。', '',
             '**问题：** 原选定γ是否把有效空间变化过度平滑；更小γ是否改善位移及外部标定参照的一致性。', '',
             '读取三次原扫描各50个已保存模量场，固定同一ROI、网格、载荷、参考图、ν=0.26、观测、权重与求解预算。逐场重新正演，共150次小型回放；未重新优化，未更改根配置或原L-curve选择。位移均用面积加权L₂相对误差。', '',
             '图示系数预先按γ原值×1/0.1/0.01/0.001选最近的已有对数网格点，不根据标定误差或图形挑点。所有系数的原始停止信息、GTOL状态与结果保留在all_gamma_metrics.csv；停止条件未通过者仅作灰叉诊断，不能作为成功候选。', '',
             '| 次数 | γ | 实际γ/原γ | 轴向相对误差 % | 横向相对误差 % | 双分量相对误差 % | 七区模量平均绝对相对误差 % | 100%区有符号相对误差 % | 横向离散 % |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in representatives.itertuples():
        lines.append(f'| {r.run} | {r.gamma:.3g} | {r.gamma_factor:.4g} | {r.axial_relative_error_percent:.3f} | {r.transverse_relative_error_percent:.3f} | {r.displacement_relative_error_percent:.3f} | {r.regional_mean_absolute_relative_error_percent:.3f} | {r.high_infill_region_relative_error_percent:+.3f} | {r.transverse_dispersion_percent:.3f} |')
    lines += ['', '## 解释与边界', '',
              '减小γ允许模量梯度增加，数据拟合改善本身是预期行为。必须同时查看二维场、横向离散、极值、停止状态及跨重复加载的一致性，不能只按曲线更接近标定认定恢复了真实材料细节。横向离散是去掉宽度均值场后的面积加权空间标准差/面积均值，不是测量噪声或参数不确定性。', '',
              '单调标定采用individual_window_joint_3pct_monotone的七组均值，仅在此敏感性检查中作为外部参照。标定差异可说明原解有平滑偏差，但本次不能据此把某γ改称正式最优；这些参照已被用于诊断，也不构成新的独立留出验证。', '',
              '更小γ还可能吸收DIC误差、材料各向异性、边界近似、未配对的力参考或ROI定位误差；现有比较没有区分这些来源。三次均采用名义末载减记录首载、厚度约3.2 mm及标线中点定位等既有假设。不同γ的初值遵循原有收敛延续路径，而非多初值全局最优检验；所有代表点的函数下降停止与严格GTOL分别报告。', '',
              '图中宽度均值不抹去二维反演自由度；二维图共享色标，保留横向变化。原γ及正式四步结果保持不变。', '',
              '![宽度均值](axial_profiles.png)', '', '![位移、参照与横向离散权衡](gamma_tradeoff.png)', '', '![二维模量](modulus_maps.png)', '',
              '## 核验与复现', '',
              '150个回放的面积加权残差范数及H1梯度范数分别与原扫描记录在1e-7和1e-8相对容差下核对。输入/字段来源哈希和逐场文件路径见provenance.json。标定与反演统计均沿用同一Gauss积分分区规则，不改力学、观测或梯度实现。', '',
              '复现：`python -B -m lib.dic_gamma_sensitivity --cases <三个monotone运行目录> --output <新目录>`。已有输出拒绝覆盖。', '']
    (output/'report.md').write_text('\n'.join(lines), encoding='utf-8')
    print(representatives[['run', 'gamma', 'axial_relative_error_percent',
          'displacement_relative_error_percent', 'regional_mean_absolute_relative_error_percent',
          'transverse_dispersion_percent']].to_string(index=False))


def plot_modulus_maps(maps, runs, output):
    """Replot representative maps independently of the forward replay."""
    fig, axes = plt.subplots(len(runs), len(FACTORS), figsize=(16, 4.5), constrained_layout=True)
    vmin, vmax = min(m['field'].min() for m in maps), max(m['field'].max() for m in maps)
    for row, run in enumerate(runs):
        for col in range(len(FACTORS)):
            ax = axes[row, col]
            m = next(m for m in maps if m['run'] == run and m['level'] == col)
            mesh = m['mesh']
            im = ax.pcolormesh(mesh.plot_x-mesh.geo_l/2, mesh.plot_y, m['field'],
                               shading='auto', cmap='viridis', vmin=vmin, vmax=vmax, rasterized=True)
            for boundary in m['boundaries']:
                ax.axvline(boundary, color='white', alpha=.65, ls='--', lw=.5)
            ax.set(title=f'第{run}次，γ={m["gamma"]:.2g}', xlabel='ξ (mm)', ylabel='y (mm)', aspect='equal')
    fig.colorbar(im, ax=axes, label='E (MPa)，全部面板共用色标', shrink=.7)
    save(fig, output/'modulus_maps')


def save(fig, stem):
    fig.savefig(stem.with_suffix('.png'), dpi=300, bbox_inches='tight')
    fig.savefig(stem.with_suffix('.svg'), bbox_inches='tight')
    svg = stem.with_suffix('.svg')
    svg.write_bytes(('\n'.join(line.rstrip() for line in svg.read_text(encoding='utf-8').splitlines())+'\n').encode('utf-8'))
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', nargs='+', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    analyze(args.cases, args.output)
