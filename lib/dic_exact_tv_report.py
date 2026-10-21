"""Audit saved exact-TV fields and compare them with the unchanged H1 baseline.

Post-processing only: no gamma selection, optimization or calibration tuning.
Forward/adjoint evaluations below independently recheck saved certificates.
"""

import argparse
import html
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fgm_asm.exact_tv import FEDataOracle, QuadratureGradient
from fgm_asm.load_case import evaluate_observation
from fgm_asm.research import solve_forward
from .dic_case_workflow import load_prepared, _metrics, _regional_comparison
from .experiment_data import file_sha256


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _profile(mesh, field):
    values = field.reshape(mesh.nods_y, mesh.nods_x)
    midpoint = .5*(values[:-1]+values[1:])
    offset = (values[1:]-values[:-1])/(2*np.sqrt(3))
    quadrature = np.stack([midpoint-offset, midpoint+offset])
    mean = quadrature.mean(axis=(0, 1))
    sd = np.sqrt(np.mean((quadrature-mean)**2, axis=(0, 1)))
    return mean, sd


def report(root, baseline):
    root, baseline = Path(root), Path(baseline)
    cfg, mesh, cases, metadata, _ = load_prepared(root/'01_prepared')
    inverse = _read(root/'03_inverse/run_metadata.json')
    previous = _read(baseline/'03_inverse/run_metadata.json')
    baseline_prepared = _read(baseline/'01_prepared/run_metadata.json')
    if inverse['status'] not in ('computed', 'inverse_not_converged'):
        raise ValueError('A selected gamma and completed final solves are required')
    if cfg.REGULARIZATION_MODE != 'exact_tv' or previous['configuration']['REGULARIZATION_MODE'] != 'gradient':
        raise ValueError('Expected exact TV versus the original gradient-square baseline')
    if cfg.DATA_MODE != 'dic':
        raise ValueError('This comparison requires experimental DIC observations')
    if (list(getattr(cfg, 'DISPLACEMENT_COMPONENT_WEIGHTS', (1., 1.))) !=
            list(previous['configuration'].get('DISPLACEMENT_COMPONENT_WEIGHTS', (1., 1.)))):
        raise ValueError('Displacement component weights differ')
    if metadata['calibration'] != baseline_prepared['calibration']:
        raise ValueError('Independent calibration definitions differ')
    for name in cfg.PLATFORMS:
        for key in ('mean_corrected_force_N', 'model_end_force_N'):
            if metadata['force_pairing'][name][key] != baseline_prepared['force_pairing'][name][key]:
                raise ValueError(f'{name}: force definition differs')
    with np.load(root/'01_prepared/fields.npz') as current, np.load(baseline/'01_prepared/fields.npz') as old:
        if current.files != old.files:
            raise ValueError('Prepared array identities differ')
        for key in current.files:
            np.testing.assert_array_equal(current[key], old[key])
    for key in ('GEO_L', 'GEO_H', 'THICKNESS', 'NEL_X', 'NEL_Y', 'NU_VALUES',
                'PRIMARY_NU', 'FIT_CASES', 'PREDICT_CASES', 'REGION_BOUNDARIES_X'):
        if inverse['configuration'][key] != previous['configuration'][key]:
            raise ValueError(f'Mismatched comparison setting: {key}')
    pd.testing.assert_frame_equal(pd.read_csv(root/'02_forward/comparison.csv'),
                                  pd.read_csv(baseline/'02_forward/comparison.csv'), check_exact=True)
    output = root/'method_comparison'
    output.mkdir(exist_ok=True)
    fit = [case for case in cases if case.name in cfg.FIT_CASES]
    op = QuadratureGradient(mesh)
    scan = pd.read_csv(root/'03_inverse/selection.csv')
    if len(scan) != len(cfg.GAMMA_CANDIDATES):
        raise ValueError('The declared gamma scan is incomplete')
    audit = []
    for row in scan.itertuples():
        directory = root/'03_inverse/l_curve'/f'gamma_{row.gamma:g}'
        audit.append(_audit_field(mesh, fit, cfg, op, directory, row.gamma, row.nu, row.converged))
    for solver in inverse['solvers']:
        audit.append(_audit_field(mesh, fit, cfg, op, root/'03_inverse'/f"nu_{solver['nu']:.2f}",
                                  inverse['selected_gamma'], solver['nu'], solver['converged']))
    pd.DataFrame(audit).to_csv(output/'independent_kkt_audit.csv', index=False)
    if not all(row['certificate_reproduced'] for row in audit):
        raise AssertionError('A saved first-order certificate was not independently reproduced')
    names = ('原方案：节点Q4＋梯度平方', '本次：节点Q4＋梯度幅值TV')
    colors = ('#0072B2', '#D55E00')
    metric_frames, region_frames, fields = [], [], []
    calibration = pd.DataFrame(metadata['calibration'])
    for directory, name in zip((baseline, root), names):
        regions = pd.read_csv(directory/'03_inverse/regional_comparison.csv')
        metrics = pd.read_csv(directory/'03_inverse/comparison.csv')
        metrics.insert(0, 'method', name)
        metric_frames.append(metrics)
        regions.insert(0, 'method', name)
        region_frames.append(regions)
        with np.load(directory/'03_inverse'/f'nu_{cfg.PRIMARY_NU:.2f}'/'fields.npz') as saved:
            field = saved['E_inverse'].copy()
        fields.append(field)
        recomputed = _regional_comparison(mesh, field, calibration, cfg.REGION_BOUNDARIES_X)
        frozen = regions[np.isclose(regions.nu, cfg.PRIMARY_NU)]
        np.testing.assert_allclose(recomputed.inverse_mean_E_MPa, frozen.inverse_mean_E_MPa, rtol=1e-12)
        for case in cases:
            u, _ = solve_forward(mesh, case.bc_info['tensile_end_force'], cfg.THICKNESS, cfg.PRIMARY_NU, field)
            fresh = _metrics(evaluate_observation(case, u))
            record = metrics[(metrics.case == case.name) & np.isclose(metrics.nu, cfg.PRIMARY_NU)].iloc[0]
            for key, value in fresh.items():
                np.testing.assert_allclose(record[key], value, rtol=1e-9, atol=1e-13)
    metrics = pd.concat(metric_frames, ignore_index=True)
    regions = pd.concat(region_frames, ignore_index=True)
    metrics.to_csv(output/'displacement_comparison.csv', index=False)
    regions.to_csv(output/'regional_comparison.csv', index=False)
    primary_regions = regions[np.isclose(regions.nu, cfg.PRIMARY_NU)]
    primary_metrics = metrics[np.isclose(metrics.nu, cfg.PRIMARY_NU)].copy()
    for key in ('rmse_ux_mm', 'rmse_uy_mm'):
        primary_metrics[key.replace('_mm', '_um')] = primary_metrics[key]*1000
    summaries = []
    for name, field in zip(names, fields):
        subset = primary_regions[primary_regions.method == name]
        mean, sd = _profile(mesh, field)
        summaries.append(dict(method=name,
            mean_absolute_regional_difference_percent=float(subset.difference_percent.abs().mean()),
            minimum_regional_difference_percent=float(subset.difference_percent.min()),
            maximum_regional_difference_percent=float(subset.difference_percent.max()),
            width_spatial_sd_rms_MPa=float(np.sqrt(np.trapezoid(sd**2, mesh.plot_x[0])/cfg.GEO_L))))
    pd.DataFrame(summaries).to_csv(output/'material_summary.csv', index=False)
    plt.rcParams.update({'font.family': 'Microsoft YaHei', 'font.size': 10, 'axes.unicode_minus': False})
    fig, axes = plt.subplots(2, 2, figsize=(13, 7.6), layout='constrained')
    low, high = min(field.min() for field in fields), max(field.max() for field in fields)
    for axis, field, name in zip(axes[0], fields, names):
        image = axis.pcolormesh(mesh.plot_x, mesh.plot_y, field.reshape(mesh.nods_y, mesh.nods_x),
                                cmap='viridis', vmin=low, vmax=high, shading='gouraud')
        axis.set(title=name, xlabel='x (mm)', ylabel='y (mm)', aspect='equal')
    fig.colorbar(image, ax=list(axes[0]), label='杨氏模量 E (MPa)', shrink=.7)
    edges = np.r_[0., cfg.REGION_BOUNDARIES_X, cfg.GEO_L]
    axis = axes[1, 0]
    axis.stairs(calibration.mean_E_MPa, edges, baseline=None, color='black', linestyle='--', label='独立标定均值（仅事后对照）')
    axis.fill_between(np.repeat(edges, 2)[1:-1], np.repeat(calibration.mean_E_MPa-calibration.sd_E_MPa, 2),
                      np.repeat(calibration.mean_E_MPa+calibration.sd_E_MPa, 2), color='gray', alpha=.18)
    profiles = {'x_mm': mesh.plot_x[0]}
    for field, name, color, key in zip(fields, names, colors, ('h1', 'tv')):
        mean, sd = _profile(mesh, field)
        profiles[f'{key}_mean_E_MPa'], profiles[f'{key}_width_spatial_sd_MPa'] = mean, sd
        axis.plot(mesh.plot_x[0], mean, color=color, label=name)
        axis.fill_between(mesh.plot_x[0], mean-sd, mean+sd, color=color, alpha=.13)
    axis.set(xlabel='x (mm)', ylabel='宽度平均 E (MPa)')
    axis.set_title('实线±色带：宽度均值±横向空间SD\n虚线±灰带：独立标定均值±组内SD', fontsize=10)
    axis.legend(fontsize=8)
    axis = axes[1, 1]
    for index, (name, color) in enumerate(zip(names, colors)):
        subset = primary_regions[primary_regions.method == name]
        axis.bar(np.arange(len(calibration))+(index-.5)*.36, subset.difference_percent,
                 width=.36, color=color, label=name)
    axis.axhline(0, color='black', linewidth=.7)
    axis.set(xticks=np.arange(len(calibration)), xticklabels=[f'{int(v)}%' for v in calibration.infill_percent],
             xlabel='制造填充率（仅用于事后统计）', ylabel='区域均值相对独立标定差异 (%)', title=f'主ν={cfg.PRIMARY_NU:g}，γ各自仅由180N选定')
    axis.legend(fontsize=8)
    for ext in ('png', 'svg'):
        fig.savefig(output/f'method_comparison.{ext}', dpi=300)
    plt.close(fig)
    pd.DataFrame(profiles).to_csv(output/'width_profiles.csv', index=False)
    summary = dict(primary_nu=cfg.PRIMARY_NU, exact_tv_status=inverse['status'],
        selected_gamma_h1=previous['selected_gamma'], selected_gamma_tv=inverse['selected_gamma'],
        scan_points=len(scan), converged_scan_points=int(scan.converged.sum()),
        resolved_converged_scan_points=int((scan.converged & scan.regularization_resolved).sum()),
        scan_FE_evaluations_tv=int(scan.evaluations.sum()),
        scan_FE_evaluations_h1=int(pd.read_csv(baseline/'03_inverse/selection.csv').evaluations.sum()),
        all_saved_certificates_independently_reproduced=True, material_summary=summaries,
        source_sha256={str(p.resolve()): file_sha256(p) for p in (
            root/'01_prepared/fields.npz', root/'03_inverse/run_metadata.json',
            baseline/'01_prepared/fields.npz', baseline/'03_inverse/run_metadata.json')})
    (output/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    columns = ['method', 'case', 'role', 'rmse_ux_um', 'rmse_uy_um']
    metric_table = primary_metrics[columns].replace({'fit': '180N识别', 'prediction': '280N留出预测'}).rename(columns={
        'method': '方案', 'case': '平台', 'role': '用途', 'rmse_ux_um': '轴向RMSE (μm)', 'rmse_uy_um': '横向RMSE (μm)'})
    material_table = pd.DataFrame(summaries).rename(columns={
        'method': '方案', 'mean_absolute_regional_difference_percent': '七区均值差异绝对值的平均 (%)',
        'minimum_regional_difference_percent': '最小区域有符号差异 (%)',
        'maximum_regional_difference_percent': '最大区域有符号差异 (%)',
        'width_spatial_sd_rms_MPa': '横向空间SD沿长度的RMS (MPa)'})
    material_note = (f"七区均值差异绝对值的平均：{summaries[0]['mean_absolute_regional_difference_percent']:.3f}% → "
                     f"{summaries[1]['mean_absolute_regional_difference_percent']:.3f}%；最大区域有符号差异："
                     f"{summaries[0]['maximum_regional_difference_percent']:.3f}% → {summaries[1]['maximum_regional_difference_percent']:.3f}%。")
    held_out = primary_metrics[primary_metrics.role == 'prediction']
    prediction_note = '；'.join(
        f"{case}轴向预测RMSE：{group[group.method == names[0]].rmse_ux_um.iloc[0]:.3f} → "
        f"{group[group.method == names[1]].rmse_ux_um.iloc[0]:.3f} μm"
        for case, group in held_out.groupby('case'))
    page = ['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>实验DIC：精确TV与原方案对照</title>',
        '<style>body{max-width:1300px;margin:32px auto;padding:0 24px;font:16px/1.7 "Microsoft YaHei",sans-serif}table{border-collapse:collapse}th,td{padding:7px;border:1px solid #ddd}img{width:100%}a{color:#075fa2}</style>',
        '<h1>实验DIC：精确TV与原方案对照</h1>',
        '<p><strong>本轮结果：平台更明显不等于识别更准确。</strong>'+material_note+prediction_note+'。具体局部差异与预测误差如下；独立标定未参与选参。</p>',
        '<p><a href="../结果总览.html">四步结果</a> · <a href="../03_inverse/云图.html">L-curve及停止证据</a> · <a href="../04_comparison/打开查看结果.html">完整正反演报告</a></p>',
        f'<p>两种方法均为{mesh.n_nod}个独立节点模量、自由二维Q4插值，保留{len(cases[0].observation.x_meas)}个DIC测点；位移试函数和测试函数仍为Q4。没有分层先验、SUPG或单元常数模量。</p>',
        '<table><tr><th>图中名称</th><th>目标函数</th><th>仅180N选定的γ</th></tr>',
        f'<tr><td>{names[0]}</td><td>½rᵀWAr＋½γ∫[(∂E/∂x)²＋(∂E/∂y)²]dA</td><td>{previous["selected_gamma"]:.9g}</td></tr>',
        f'<tr><td>{names[1]}</td><td>½rᵀWAr＋γ∫√[(∂E/∂x)²＋(∂E/∂y)²]dA</td><td>{inverse["selected_gamma"]:.9g}</td></tr></table>',
        '<p>TV无Huber阈值或平滑ε，L-curve纵坐标直接取梯度幅值积分。γ单位不同，不能比较数值大小。原H1按函数下降停止、未达原GTOL；本次按保存的一阶驻点及对偶间隙条件判定。停止标准的差异不等于物理正确性差异。</p>',
        f'<p>TV状态：{html.escape(inverse["status"])}；{len(scan)}点中{summary["converged_scan_points"]}点通过一阶证据，{summary["resolved_converged_scan_points"]}点同时具有非零可分辨TV。所有保存场的证据已独立重新正演/伴随核验。</p>',
        f'<p>{len(scan)}点扫描实际FE函数计算次数：原方案{summary["scan_FE_evaluations_h1"]}，本次TV{summary["scan_FE_evaluations_tv"]}。两者停止标准不同，此处报告计算量，不据此作严格耗时排名。</p>',
        '<img src="method_comparison.png" alt="两种目标函数得到的二维模量、宽度均值及标定差异">',
        f'<h2>主ν={cfg.PRIMARY_NU:g}：位移分量RMSE（μm）</h2>', metric_table.to_html(index=False, float_format=lambda v:f'{v:.6g}'),
        f'<h2>主ν={cfg.PRIMARY_NU:g}：材料差异</h2>', material_table.to_html(index=False, float_format=lambda v:f'{v:.6g}'),
        '<p>材料差异是与独立标定的比较，不是对未知实验真值的误差。横向空间SD不是置信区间；分区位置仅用于事后统计。280N未用于选γ或ν。物理配准、内部侧边界等效及跨批标定转移限制仍保留。</p>',
        '<p><a href="independent_kkt_audit.csv">逐场独立证据</a> · <a href="displacement_comparison.csv">全部ν位移误差</a> · <a href="regional_comparison.csv">全部ν区域差异</a> · <a href="method_comparison.svg">矢量图</a></p></html>']
    (output/'index.html').write_text('\n'.join(page), encoding='utf-8')
    overview = root/'结果总览.html'
    text = overview.read_text(encoding='utf-8')
    if 'id="exact-tv-method-comparison"' not in text:
        link = '<p id="exact-tv-method-comparison"><strong><a href="method_comparison/index.html">查看：精确TV与原梯度平方方案的同条件对照</a></strong></p>'
        overview.write_text(text.replace('<div class="cards">', link+'\n<div class="cards">', 1), encoding='utf-8')
    return summary


def _audit_field(mesh, fit, cfg, op, directory, gamma, nu, converged):
    with np.load(directory/'fields.npz') as saved:
        field, dual = saved['E_inverse'], saved['tv_dual']
        split = saved['tv_split_gradient']
    value, grad = FEDataOracle(mesh, fit, nu)(field)
    gradient = op.apply(field)
    data, regularizer = grad*field, op.adjoint(dual)*field
    stationarity = float(np.max(np.abs(data+regularizer)))
    options = cfg.EXACT_TV_OPTIONS
    tolerance = options['stationarity_atol']+options['stationarity_rtol']*max(np.max(np.abs(data)), np.max(np.abs(regularizer)))
    tv = op.value(field)
    gap = float(op.area@(gamma*np.linalg.norm(gradient, axis=1)-np.sum(dual*gradient, axis=1)))
    gap_tolerance = options['gap_atol']+options['gap_rtol']*gamma*tv
    primal = op.rms(gradient-split)
    primal_tolerance = options['primal_atol']+options['primal_rtol']*max(op.rms(gradient), op.rms(split))
    excess = max(0., float(np.max(np.linalg.norm(dual, axis=1))-gamma))
    passed = bool(stationarity <= tolerance and abs(gap) <= gap_tolerance and
                  primal <= primal_tolerance and excess <= 1e-12*gamma)
    solver = _read(directory/'run_metadata.json')['solver']
    np.testing.assert_allclose(value+gamma*tv, solver['final_cost'], rtol=1e-9, atol=1e-12)
    return dict(directory=str(directory.resolve()), gamma=gamma, nu=nu, saved_converged=bool(converged),
        independent_converged=passed, certificate_reproduced=passed == bool(converged),
        stationarity_ratio=stationarity/tolerance, gap_ratio=abs(gap)/gap_tolerance,
        primal_ratio=primal/primal_tolerance, dual_excess=excess)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(report(args.root, args.baseline), ensure_ascii=False, indent=2))
