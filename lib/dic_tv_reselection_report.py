"""Compare independently selected real-DIC TV results on two mesh levels.

Read-only with respect to computed fields, selection and solver metadata.
Manufacturing regions and the held-out load are used for evaluation only.
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fgm_asm.exact_tv import QuadratureGradient
from fgm_asm.load_case import evaluate_observation
from fgm_asm.research import solve_forward
from .dic_case_workflow import _metrics, load_prepared
from .dic_exact_tv_report import _audit_field, _profile
from .dic_tv_refinement import _context, _read, prolong_field
from .dic_tv_refinement_report import _common_integration
from .experiment_data import file_sha256


def report(root, coarse):
    root, coarse = Path(root), Path(coarse)
    cfg, mesh, cases, prepared, _, old_mesh, old_field, _, old_gamma, _ = _context(root, coarse)
    inverse = _read(root/'03_inverse/run_metadata.json')
    previous = _read(coarse/'03_inverse/run_metadata.json')
    if inverse['status'] not in ('computed', 'inverse_not_converged'):
        raise ValueError('A complete scan, selected gamma and final solves are required')
    for key in ('NU_VALUES', 'GAMMA_CANDIDATES', 'EXACT_TV_SOLVER', 'EXACT_TV_COUPLING'):
        if inverse['configuration'][key] != previous['configuration'][key]:
            raise ValueError(f'Different comparison setting: {key}')
    gamma = inverse['selected_gamma']
    output = root/'mesh_reselected'
    output.mkdir(exist_ok=True)
    scan = pd.read_csv(root/'03_inverse/selection.csv', float_precision='round_trip')
    old_scan = pd.read_csv(coarse/'03_inverse/selection.csv', float_precision='round_trip')
    expected = np.asarray(cfg.GAMMA_CANDIDATES)
    for current in (old_scan, scan):
        np.testing.assert_allclose(np.sort(current.gamma), np.sort(expected), rtol=1e-14, atol=0)
        if len(current) != len(expected) or current.gamma.duplicated().any():
            raise ValueError('Each mesh needs one complete independent gamma scan')
        chosen = current[current.selected]
        if len(chosen) != 1 or not bool(chosen.eligible.iloc[0]):
            raise ValueError('Need exactly one admissible selected corner per mesh')
    fit = [case for case in cases if case.name in cfg.FIT_CASES]
    op = QuadratureGradient(mesh)
    audits = []
    for row in scan.itertuples():
        directory = getattr(row, 'solution_directory', None)
        directory = root/'03_inverse'/directory if isinstance(directory, str) else (
            root/'03_inverse/l_curve'/f'gamma_{row.gamma:g}')
        audits.append(_audit_field(mesh, fit, cfg, op, directory, row.gamma, row.nu, row.converged))
        if isinstance(getattr(row, 'solution_directory', None), str):
            original = root/'03_inverse/l_curve'/f'gamma_{row.gamma:g}'
            old_flag = _read(original/'run_metadata.json')['solver']['converged']
            audits.append(_audit_field(mesh, fit, cfg, op, original, row.gamma, row.nu, old_flag))
    audits.extend(_audit_field(mesh, fit, cfg, op, root/'03_inverse'/f"nu_{s['nu']:.2f}",
        gamma, s['nu'], s['converged']) for s in inverse['solvers'])
    if not all(record['certificate_reproduced'] for record in audits):
        raise AssertionError('Saved KKT flags differ from independently recomputed evidence')
    pd.DataFrame(audits).to_csv(output/'independent_kkt_audit.csv', index=False)
    final = root/'03_inverse'/f'nu_{cfg.PRIMARY_NU:.2f}'/'fields.npz'
    with np.load(final) as saved:
        field = saved['E_inverse'].copy()
    fixed_path = root/'mesh_diagnostic/solution/fields.npz'
    with np.load(fixed_path) as saved:
        fixed = saved['E_inverse'].copy()
    prolonged = prolong_field(old_mesh, mesh, old_field)
    nodes = mesh.ele_nods_id-1
    area = mesh.gauss_abs_det_j*mesh.gauss_w[:, None]
    def field_difference(reference, current):
        reference_q = (reference[nodes]@mesh.gauss_N.T).T
        difference_q = ((current-reference)[nodes]@mesh.gauss_N.T).T
        square = float(np.sum(area*difference_q**2))
        return dict(rms_MPa=float(np.sqrt(square/area.sum())),
            relative_rms_percent=float(100*np.sqrt(square/np.sum(area*reference_q**2))))
    field_differences = dict(coarse_to_refined_reselected=field_difference(prolonged, field),
        refined_fixed_to_refined_reselected=field_difference(fixed, field),
        refined_fixed_and_reselected_gamma_equal=bool(np.isclose(gamma, old_gamma, rtol=1e-12, atol=0)))
    calibration = pd.DataFrame(prepared['calibration'])
    regions, spatial_sd, quadrature = _common_integration(
        (old_mesh, mesh), (old_field, field), calibration, cfg.REGION_BOUNDARIES_X)
    fixed_regions, fixed_sd, fixed_quadrature = _common_integration(
        (old_mesh, mesh), (old_field, fixed), calibration, cfg.REGION_BOUNDARIES_X)
    for key in quadrature:
        np.testing.assert_array_equal(quadrature[key], fixed_quadrature[key])
    fixed_regions = fixed_regions[fixed_regions.mesh == 'refined'].copy()
    fixed_regions['mesh'] = 'refined_fixed_gamma'
    regions = pd.concat([regions, fixed_regions], ignore_index=True)
    regions.to_csv(output/'common_regional_comparison.csv', index=False)
    np.savez_compressed(output/'common_evaluation_quadrature.npz', **quadrature)
    keys = ('coarse', 'refined', 'refined_fixed_gamma')
    coarse_size, fine_size = f'{old_mesh.nel_x}×{old_mesh.nel_y}', f'{mesh.nel_x}×{mesh.nel_y}'
    labels = (f'{coarse_size}单元，节点Q4＋TV\n独立选γ={old_gamma:.3g}',
              f'{fine_size}单元，节点Q4＋TV\n独立选γ={gamma:.3g}',
              f'{fine_size}单元，节点Q4＋TV\n沿用粗网格γ={old_gamma:.3g}')
    color = ('#D55E00', '#7B3294', '#7B3294')
    all_meshes, all_fields = (old_mesh, mesh, mesh), (old_field, field, fixed)
    summary_rows, profiles = [], []
    for key, current_mesh, values, label, current_gamma, sd in zip(keys, all_meshes, all_fields,
            labels, (old_gamma, gamma, old_gamma),
            (spatial_sd['coarse'], spatial_sd['refined'], fixed_sd['refined'])):
        mean, width_sd = _profile(current_mesh, values)
        profiles.append(pd.DataFrame(dict(mesh=key, x_mm=current_mesh.plot_x[0],
            mean_E_MPa=mean, width_spatial_sd_MPa=width_sd)))
        subset = regions[regions.mesh == key]
        summary_rows.append(dict(mesh=key, label=label, gamma=current_gamma,
            independent_parameters=current_mesh.n_nod,
            average_absolute_regional_difference_percent=float(subset.difference_percent.abs().mean()),
            maximum_absolute_regional_difference_percent=float(subset.difference_percent.abs().max()),
            width_spatial_sd_rms_MPa=sd))
    pd.concat(profiles).to_csv(output/'width_profiles.csv', index=False)
    pd.DataFrame(summary_rows).to_csv(output/'material_summary.csv', index=False)
    metrics = []
    for key, directory, current_mesh, values in zip(keys[:2], (coarse, root), all_meshes, all_fields):
        current = pd.read_csv(directory/'03_inverse/comparison.csv')
        current.insert(0, 'mesh', key)
        primary = current[np.isclose(current.nu, cfg.PRIMARY_NU)]
        # Independent fresh forward predictions, using each mesh's own H.
        if key == 'coarse':
            _, _, current_cases, _, _ = load_prepared(coarse/'01_prepared')
        else:
            current_cases = cases
        for case in current_cases:
            u, _ = solve_forward(current_mesh, case.bc_info['tensile_end_force'],
                                  cfg.THICKNESS, cfg.PRIMARY_NU, values)
            fresh = _metrics(evaluate_observation(case, u))
            record = primary[primary.case == case.name].iloc[0]
            for name, value in fresh.items():
                np.testing.assert_allclose(record[name], value, rtol=1e-9, atol=1e-13)
        metrics.append(current)
    fixed_metrics = pd.read_csv(root/'mesh_diagnostic/displacement_comparison.csv')
    fixed_metrics = fixed_metrics[fixed_metrics.mesh == 'refined'].copy()
    fixed_metrics['mesh'], fixed_metrics['nu'] = keys[2], cfg.PRIMARY_NU
    fixed_metrics['role'] = np.where(fixed_metrics.case.isin(cfg.FIT_CASES), 'fit', 'prediction')
    metrics.append(fixed_metrics)
    metrics = pd.concat(metrics, ignore_index=True)
    metrics.to_csv(output/'displacement_comparison.csv', index=False)
    main_metrics = metrics[np.isclose(metrics.nu, cfg.PRIMARY_NU)].copy()
    main_metrics['轴向RMSE (μm)'] = 1000*main_metrics.rmse_ux_mm
    main_metrics['横向RMSE (μm)'] = 1000*main_metrics.rmse_uy_mm
    plt.rcParams.update({'font.family':'Microsoft YaHei', 'font.size':10, 'axes.unicode_minus':False})
    fig, axes = plt.subplots(2, 2, figsize=(13, 7.8), layout='constrained')
    lo, hi = min(old_field.min(), field.min()), max(old_field.max(), field.max())
    for ax, current_mesh, values, label in zip(axes[0], all_meshes, all_fields, labels):
        im = ax.pcolormesh(current_mesh.plot_x, current_mesh.plot_y,
            values.reshape(current_mesh.nods_y, current_mesh.nods_x),
            cmap='viridis', vmin=lo, vmax=hi, shading='gouraud')
        ax.set(title=label, xlabel='x (mm)', ylabel='y (mm)', aspect='equal')
    fig.colorbar(im, ax=list(axes[0]), label='杨氏模量 E (MPa)', shrink=.7)
    edges = np.r_[0., cfg.REGION_BOUNDARIES_X, cfg.GEO_L]
    ax = axes[1, 0]
    ax.stairs(calibration.mean_E_MPa, edges, baseline=None, color='black', linestyle='--',
              label='独立标定均值（仅事后比较）')
    ax.fill_between(np.repeat(edges, 2)[1:-1], np.repeat(calibration.mean_E_MPa-calibration.sd_E_MPa, 2),
        np.repeat(calibration.mean_E_MPa+calibration.sd_E_MPa, 2), color='gray', alpha=.15)
    for profile, label, c, line in zip(profiles, labels, color, ('-', '-', '--')):
        ax.plot(profile.x_mm, profile.mean_E_MPa, color=c, linestyle=line, label=label)
    ax.set(xlabel='x (mm)', ylabel='宽度平均 E (MPa)', title='重新选参后的模量分布；虚线为独立标定')
    ax.legend(fontsize=7.5)
    ax = axes[1, 1]
    for index, (key, label, c) in enumerate(zip(keys, labels, color)):
        subset = regions[regions.mesh == key]
        ax.bar(np.arange(len(calibration))+(index-1)*.25, subset.difference_percent,
               width=.25, color=c, alpha=.4 if index == 2 else 1.,
               hatch='///' if index == 2 else None, label=label)
    ax.axhline(0, color='black', linewidth=.7)
    ax.set(xticks=np.arange(len(calibration)), xticklabels=[f'{v:g}%' for v in calibration.infill_percent],
        xlabel='制造填充率（仅事后统计）', ylabel='区域均值相对独立标定差异 (%)',
        title=f'相同物理积分点评价，主ν={cfg.PRIMARY_NU:g}')
    ax.legend(fontsize=7.5)
    for extension in ('png', 'svg'):
        fig.savefig(output/f'mesh_reselected.{extension}', dpi=300)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8.5, 5.8), layout='constrained')
    for current, label, c in zip((old_scan, scan), labels[:2], color):
        valid = current.converged & current.regularization_resolved
        # Preserve failed-point holes instead of connecting across them.
        rho = current.residual_norm.where(valid)
        eta = current.regularization_norm.where(valid)
        ax.loglog(rho, eta, '.-', color=c, label=label)
        chosen = current[current.selected].iloc[0]
        ax.scatter(chosen.residual_norm, chosen.regularization_norm, s=100,
                   facecolors='none', edgecolors=c, linewidth=1.8)
    ax.set(xlabel='180N面积加权残差范数 (mm²)', ylabel='TV(E) (MPa·mm)',
           title='两网格各自完整50点扫描；仅合格非零TV点用于曲率')
    ax.legend(fontsize=9)
    for extension in ('png', 'svg'):
        fig.savefig(output/f'l_curve_comparison.{extension}', dpi=300)
    plt.close(fig)
    scan_records = []
    for name, current in zip(keys, (old_scan, scan)):
        chosen = current[current.selected].iloc[0]
        scan_records.append(dict(mesh=name, scan_points=len(current), selected_gamma=float(chosen.gamma),
            converged_scan_points=int(current.converged.sum()),
            resolved_converged_points=int((current.converged & current.regularization_resolved).sum()),
            selected_curvature=float(chosen.curvature), FE_evaluations=int(current.evaluations.sum())))
    summary = dict(status=inverse['status'], gamma_reselected_on_refined_mesh=True,
        gamma_selection_uses='180N only, nu=0.26, independent 50-point L-curve per mesh',
        primary_nu=cfg.PRIMARY_NU, scans=scan_records, material_summary=summary_rows,
        evaluation='common physical quadrature split at both meshes and manufacturing boundaries',
        certificates_reproduced=len(audits), independently_converged_certificates=sum(a['independent_converged'] for a in audits),
        final_solvers=inverse['solvers'], scan_recovery=inverse.get('scan_recovery'),
        scan_completion=inverse.get('scan_completion'),
        field_differences=field_differences,
        source_sha256={str(p.resolve()):file_sha256(p) for p in (
            root/'01_prepared/fields.npz', root/'03_inverse/run_metadata.json',
            root/'03_inverse/selection.csv', final, fixed_path,
            coarse/'03_inverse/run_metadata.json', coarse/'03_inverse/selection.csv', Path(__file__))})
    (output/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    label_map = dict(zip(keys, labels))
    material_table = pd.DataFrame(summary_rows).drop(columns=['label', 'gamma']).replace({'mesh':label_map}).rename(columns={
        'mesh':'方案', 'independent_parameters':'独立节点模量数',
        'average_absolute_regional_difference_percent':'七区平均绝对差异 (%)',
        'maximum_absolute_regional_difference_percent':'最差区域绝对差异 (%)',
        'width_spatial_sd_rms_MPa':'横向空间SD的RMS (MPa)'})
    metric_table = main_metrics[['mesh','case','role','轴向RMSE (μm)','横向RMSE (μm)']].replace(
        {'mesh':label_map, 'role':{'fit':'识别', 'prediction':'留出预测'}}).rename(
        columns={'mesh':'方案', 'case':'平台', 'role':'用途'})
    scan_table = pd.DataFrame(scan_records).replace({'mesh':dict(zip(keys[:2],labels[:2]))}).rename(columns={
        'mesh':'网格', 'scan_points':'扫描点数', 'selected_gamma':'新选γ',
        'converged_scan_points':'通过一阶条件点数', 'resolved_converged_points':'其中非零TV点数',
        'selected_curvature':'所选曲率', 'FE_evaluations':'FE计算次数'})
    old_link = Path(os.path.relpath(coarse/'method_comparison/index.html', output)).as_posix()
    page = ['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>真实DIC：网格加密并重新选γ</title>',
        '<style>body{max-width:1350px;margin:32px auto;padding:0 24px;font:16px/1.7 "Microsoft YaHei",sans-serif}table{border-collapse:collapse;display:block;overflow:auto}th,td{padding:7px;border:1px solid #ddd}img{width:100%}a{color:#075fa2}</style>',
        '<h1>真实DIC：242×38单元重新选γ后的结果</h1>',
        f'<p><strong>细网格已独立重算50个γ，由180N的L-curve选得γ={gamma:.12g}；粗网格原选γ={old_gamma:.12g}。最终状态：{inverse["status"]}。</strong></p>',
        f'<p>共同积分口径下，七区均值相对独立标定的平均绝对差异：粗网格{summary_rows[0]["average_absolute_regional_difference_percent"]:.4f}% → '
        f'细网格重新选参{summary_rows[1]["average_absolute_regional_difference_percent"]:.4f}%；上次细网格固定γ为{summary_rows[2]["average_absolute_regional_difference_percent"]:.4f}%。留出预测及局部偏差见下表，不用这些指标反选γ。</p>',
        f'<p>各自选参后的细粗模量场间相对RMS差异为{field_differences["coarse_to_refined_reselected"]["relative_rms_percent"]:.4f}%'
        f'（{field_differences["coarse_to_refined_reselected"]["rms_MPa"]:.4f} MPa）。细网格重新选参与此前固定γ解的场间相对RMS差异为'
        f'{field_differences["refined_fixed_to_refined_reselected"]["relative_rms_percent"]:.4f}%。这些是解之间的差别，不是对未知真值的误差；两次细网格γ'
        f'{"相同" if field_differences["refined_fixed_and_reselected_gamma_equal"] else "不同"}。</p>',
        '<p><a href="../结果总览.html">四步结果</a> · <a href="../03_inverse/云图.html">细网格L-curve与求解证据</a> · <a href="../04_comparison/打开查看结果.html">DIC、独立正演和反演的完整对照</a></p>',
        '<p>两组使用相同23920个DIC测点、毫米坐标、力、面积权重及三个刚体规范；每个节点模量独立，各y行不绑定。位移及模量均采用连续Q4插值，测试函数仍为Q4。没有SUPG、单元常数模量或分层约束。</p>',
        '<p>目标函数完全相同：½rᵀWAr＋γ∫Ω√[(∂E/∂x)²＋(∂E/∂y)²]dA。没有Huber阈值、平滑ε或参考模量。两网格各从仅用180N求得的均匀模量出发，按已收敛点延续，独立完成相同的50个γ（1e-10—1e-2）。细网格新扫描不读取固定γ诊断解。</p>',
        '<p>仅在ν=0.26选γ；固定新γ后计算ν=0.24/0.26/0.28/0.33并预测280N。此为ν敏感性比较，不是ν和γ联合识别。独立标定与280N均未参与选参。</p>',
        '<h2>各自选参的扫描证据</h2>', scan_table.to_html(index=False, float_format=lambda v:f'{v:.8g}'),
        '<p>零TV均匀场可通过一阶条件，但不参与对数曲率；未通过条件的点保留并排除。所选点必须具有足够合格邻点。基础50点扫描每点预算均为12000次FE，驻点/间隙容差一致；细网格数值步长为粗网格的4倍以对应单元面积变化，不缩放目标或γ。</p>',
        '<img src="l_curve_comparison.png" alt="两个网格各自重新计算的L-curve和所选拐点">',
        '<img src="mesh_reselected.png" alt="重新选参的二维模量、宽度均值和区域标定差异">',
        '<h2>主ν=0.26：材料与横向空间变化</h2>', material_table.to_html(index=False,float_format=lambda v:f'{v:.6g}'),
        '<p>两网格的区域均值和空间SD使用共同物理积分点，积分在两网格单元边界和制造界面处分割。它与旧页按各自原生Gauss点分区的统计口径有小差别。分区只用于事后评价。空间SD不是反演置信区间。</p>',
        '<h2>主ν=0.26：180N拟合与280N固定参数预测</h2>', metric_table.to_html(index=False,float_format=lambda v:f'{v:.7g}'),
        f'<p>细网格50个当前扫描场、原失败场（如有补算）和4个最终ν场均已独立重新计算正演/伴随证据，{len(audits)}个保存的通过/失败标记均得到复现；其中{summary["independently_converged_certificates"]}个通过，不把失败点称为收敛。主ν两网格保存的180N/280N误差也已重新正演复核。</p>',
        '<p>局部一阶条件不能证明全局最优。物理配准尚未完成、内部侧面自由边界近似和跨批标定转移限制保留；材料差异是与独立标定的比较，不是对未知实验真值的误差。</p>',
        f'<p><a href="../mesh_diagnostic/index.html">此前固定γ诊断（单独保留）</a> · <a href="{old_link}">原粗网格TV与梯度平方对照</a> · <a href="common_regional_comparison.csv">共同积分区域统计</a> · <a href="displacement_comparison.csv">全部ν位移误差</a> · <a href="independent_kkt_audit.csv">逐场独立一阶核验</a> · <a href="summary.json">配置与数值摘要</a> · <a href="mesh_reselected.svg">模量矢量图</a> · <a href="l_curve_comparison.svg">L-curve矢量图</a></p></html>']
    if inverse.get('scan_recovery'):
        recovery = inverse['scan_recovery']
        notice = (f'<p><strong>细网格基础扫描后，按正曲率峰两侧缺失的计算证据补算，额外使用'
            f'{recovery["actual_extra_FE_evaluations"]}次FE。每个相关γ最多追加'
            f'{recovery["additional_evaluation_budget_per_attempt"]}次，保持相同γ、目标及容差；'
            '从该γ保存的失败E和对偶场重启。此项额外预算不是两网格相同预算的效率比较。'
            '补算对象只由180N曲率和收敛状态确定，未读取标定差异或280N误差。</strong> '
            '<a href="../03_inverse/tv_recovery_protocol.json">冻结补算规则</a> · '
            '<a href="../03_inverse/selection_before_tv_recovery.csv">原50点状态</a></p>')
        page.insert(page.index('<h2>各自选参的扫描证据</h2>')+1, notice)
    if inverse.get('scan_completion'):
        c = inverse['scan_completion']
        page.insert(page.index('<h2>各自选参的扫描证据</h2>')+1,
            f'<p>最弱γ={c["interrupted_gamma"]:g}的原尝试出现正模量步长下溢，日志可确认'
            f'{c["interrupted_FE_lower_bound"]}—{c["interrupted_FE_upper_bound"]}次FE，确数不可恢复。'
            f'以步长{c["retry_primal_step"]:g}另算{c["retry_actual_evaluations"]}次得到保存的诊断场，'
            f'通过条件：{c["retry_converged"]}。该场按实际状态参与或排除；表内计算量不含中断尝试，'
            '<a href="../03_inverse/interrupted_scan_protocol.json">完整中断与补算记录</a>另存。</p>')
    (output/'index.html').write_text('\n'.join(page), encoding='utf-8')
    # Add a new entrance while preserving the earlier fixed-gamma page and arrays.
    for path, marker, anchor, insertion in (
        (root/'结果总览.html', 'tv-mesh-reselected', 'mesh_reselected/index.html', '<div class="cards">'),
        (root/'mesh_diagnostic/index.html', 'tv-mesh-reselected', '../mesh_reselected/index.html', '<h1>')):
        text = path.read_text(encoding='utf-8')
        if f'id="{marker}"' not in text:
            banner = f'<p id="{marker}"><strong><a href="{anchor}">查看细网格完整50点扫描并重新选γ的结果</a></strong></p>'
            path.write_text(text.replace(insertion, banner+'\n'+insertion, 1), encoding='utf-8')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--coarse', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(report(args.root, args.coarse), ensure_ascii=False, indent=2))
