"""Audit and plot the fixed-gamma coarse/refined real-DIC TV diagnostic."""

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

from fgm_asm.exact_tv import QuadratureGradient
from .dic_exact_tv_report import _audit_field, _profile
from .dic_tv_refinement import _context
from .experiment_data import file_sha256
from .calibration_trend import add_calibration_trend


def _common_integration(meshes, fields, calibration, boundaries):
    """Identical physical quadrature for both Q4 fields, cut at region edges.

    The union of cell edges and manufacturing edges splits every field into
    bilinear patches. Two Gauss points per axis exactly integrate E and E^2
    on these patches. Region locations affect evaluation only, never inversion.
    """
    x_edges = np.unique(np.r_[boundaries, *[m.plot_x[0] for m in meshes]])
    y_edges = np.unique(np.concatenate([m.plot_y[:, 0] for m in meshes]))
    def axis_rule(edges):
        half = np.diff(edges)/2
        midpoint = (edges[:-1]+edges[1:])/2
        return (midpoint[:, None]+half[:, None]*np.array([-1.,1.])/np.sqrt(3)).ravel(), np.repeat(half,2)
    x, wx = axis_rule(x_edges)
    y, wy = axis_rule(y_edges)
    xx, yy = np.meshgrid(x,y)
    points = np.column_stack([yy.ravel(),xx.ravel()])
    areas = wy[:,None]*wx[None,:]
    groups = np.searchsorted(boundaries,x,side='right')
    rows, spatial_sd = [], {}
    for name, mesh, field in zip(('coarse','refined'), meshes, fields):
        values = RegularGridInterpolator((mesh.plot_y[:,0],mesh.plot_x[0]),field.reshape(mesh.nods_y,mesh.nods_x))(points).reshape(len(y),len(x))
        mean_y = (wy[:,None]*values).sum(axis=0)/wy.sum()
        variance_y = (wy[:,None]*(values-mean_y)**2).sum(axis=0)/wy.sum()
        spatial_sd[name] = float(np.sqrt(wx@variance_y/wx.sum()))
        for index, record in enumerate(calibration.to_dict(orient='records')):
            mask = groups == index
            weight, local = areas[:,mask], values[:,mask]
            average = float(np.sum(weight*local)/weight.sum())
            rows.append(dict(mesh=name,**record,inverse_mean_E_MPa=average,
                inverse_spatial_sd_MPa=float(np.sqrt(np.sum(weight*(local-average)**2)/weight.sum())),
                difference_percent=100*(average/record['mean_E_MPa']-1)))
    return pd.DataFrame(rows),spatial_sd,dict(x_mm=x,y_mm=y,x_weights_mm=wx,y_weights_mm=wy)


def report(root):
    output = root/'mesh_diagnostic'
    protocol = json.loads((output/'run_metadata.json').read_text(encoding='utf-8'))
    coarse = Path(protocol['coarse_root'])
    cfg, mesh, cases, meta, _, old_mesh, old_field, _, gamma, source = _context(root, coarse)
    destination = output/'solution'
    solved = json.loads((destination/'run_metadata.json').read_text(encoding='utf-8'))
    with np.load(destination/'fields.npz') as saved:
        field = saved['E_inverse'].copy()
    with np.load(output/'fields.npz') as saved:
        initial = saved['initial_E'].copy()
    fit = [case for case in cases if case.name in cfg.FIT_CASES]
    audit = _audit_field(mesh, fit, cfg, QuadratureGradient(mesh), destination, gamma, cfg.PRIMARY_NU,
                         solved['solver']['converged'])
    assert audit['certificate_reproduced']
    calibration = pd.DataFrame(meta['calibration'])
    regions, common_sd, quadrature = _common_integration((old_mesh,mesh),(old_field,field),calibration,cfg.REGION_BOUNDARIES_X)
    regions.to_csv(output/'common_regional_comparison.csv',index=False)
    np.savez_compressed(output/'common_evaluation_quadrature.npz',**quadrature)
    metrics = pd.read_csv(output/'displacement_comparison.csv')
    summary_rows = []
    profiles = []
    labels = [f'TV：{old_mesh.nel_x}×{old_mesh.nel_y}单元（{old_mesh.n_nod}节点模量）',
              f'TV：{mesh.nel_x}×{mesh.nel_y}单元（{mesh.n_nod}节点模量）']
    if not solved['solver']['converged']:
        labels[1] += '，未通过一阶条件'
    for name, current_mesh, values, label in zip(('coarse', 'refined'), (old_mesh, mesh), (old_field, field), labels):
        mean, sd = _profile(current_mesh, values)
        profiles.append(pd.DataFrame(dict(mesh=name, x_mm=current_mesh.plot_x[0], mean_E_MPa=mean, width_spatial_sd_MPa=sd)))
        subset = regions[regions.mesh == name]
        summary_rows.append(dict(mesh=name, elements_x=current_mesh.nel_x, elements_y=current_mesh.nel_y,
            parameters=current_mesh.n_nod,
            average_absolute_regional_difference_percent=float(subset.difference_percent.abs().mean()),
            maximum_absolute_regional_difference_percent=float(subset.difference_percent.abs().max()),
            width_spatial_sd_rms_MPa=common_sd[name],
            maximum_axial_profile_slope_MPa_per_mm=float(np.max(np.abs(np.diff(mean)/np.diff(current_mesh.plot_x[0]))))))
    summary_frame = pd.DataFrame(summary_rows)
    summary_frame.to_csv(output/'material_summary.csv', index=False)
    pd.concat(profiles).to_csv(output/'width_profiles.csv', index=False)
    nodes = mesh.ele_nods_id-1
    new_values = (field[nodes] @ mesh.gauss_N.T).T
    old_values = (initial[nodes] @ mesh.gauss_N.T).T
    areas = mesh.gauss_abs_det_j*mesh.gauss_w[:, None]
    field_rms = float(np.sqrt(np.sum(areas*(new_values-old_values)**2)/areas.sum()))
    relative_rms = float(100*np.sqrt(np.sum(areas*(new_values-old_values)**2)/np.sum(areas*old_values**2)))
    preflight = json.loads((output/'preflight.json').read_text(encoding='utf-8'))
    data = dict(status=solved['status'], gamma_fixed=gamma, gamma_reselected_on_refined_mesh=False,
        nu=cfg.PRIMARY_NU, coarse_parameters=old_mesh.n_nod, refined_parameters=mesh.n_nod,
        regional_evaluation='common physical quadrature, split at both meshes and manufacturing edges; E and E^2 integrated exactly up to roundoff',
        field_difference_rms_MPa=field_rms, field_difference_relative_rms_percent=relative_rms,
        maximum_nodal_field_difference_MPa=float(np.max(np.abs(field-initial))),
        initial_refined_objective=preflight['initial_data_cost']+gamma*preflight['prolonged_TV'],
        final_refined_objective=solved['solver']['final_cost'],
        refined_evaluations=solved['solver']['n_evaluations'], elapsed_seconds=solved['elapsed_seconds'],
        independent_audit=audit, material_summary=summary_rows,
        source_sha256={str(p.resolve()):file_sha256(p) for p in (source, destination/'fields.npz',
            root/'01_prepared/run_metadata.json', output/'run_metadata.json', Path(__file__))})
    (output/'summary.json').write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    colors = ('#D55E00', '#7B3294')
    plt.rcParams.update({'font.family':'Microsoft YaHei', 'font.size':10, 'axes.unicode_minus':False})
    fig, axes = plt.subplots(2, 2, figsize=(13, 7.6), layout='constrained')
    low, high = min(old_field.min(), field.min()), max(old_field.max(), field.max())
    for ax, current_mesh, values, label in zip(axes[0], (old_mesh, mesh), (old_field, field), labels):
        im = ax.pcolormesh(current_mesh.plot_x, current_mesh.plot_y,
                           values.reshape(current_mesh.nods_y, current_mesh.nods_x),
                           vmin=low, vmax=high, cmap='viridis', shading='gouraud')
        ax.set(title=label, xlabel='x (mm)', ylabel='y (mm)', aspect='equal')
    fig.colorbar(im, ax=list(axes[0]), label='杨氏模量 E (MPa)', shrink=.7)
    edges = np.r_[0., cfg.REGION_BOUNDARIES_X, cfg.GEO_L]
    ax = axes[1, 0]
    ax.stairs(calibration.mean_E_MPa, edges, baseline=None, color='black', linestyle='--', label='独立标定均值（仅事后对照）')
    add_calibration_trend(ax, calibration, cfg.REGION_BOUNDARIES_X, cfg.GEO_L, output=output)
    ax.fill_between(np.repeat(edges,2)[1:-1], np.repeat(calibration.mean_E_MPa-calibration.sd_E_MPa,2),
                    np.repeat(calibration.mean_E_MPa+calibration.sd_E_MPa,2), color='gray', alpha=.18)
    for profile, color, label in zip(profiles, colors, labels):
        ax.plot(profile.x_mm, profile.mean_E_MPa, color=color, label=label)
        ax.fill_between(profile.x_mm, profile.mean_E_MPa-profile.width_spatial_sd_MPa,
                        profile.mean_E_MPa+profile.width_spatial_sd_MPa, color=color, alpha=.12)
    ax.set(xlabel='x (mm)', ylabel='宽度平均 E (MPa)')
    ax.set_title('实线±色带：宽度均值±横向空间SD\n虚线±灰带：独立标定均值±组内SD', fontsize=10)
    ax.legend(fontsize=7.5)
    ax = axes[1, 1]
    for index, (name, color, label) in enumerate(zip(('coarse','refined'), colors, labels)):
        subset = regions[regions.mesh == name]
        ax.bar(np.arange(len(calibration))+(index-.5)*.36, subset.difference_percent,
               width=.36, color=color, label=label)
    ax.axhline(0, color='black', linewidth=.7)
    ax.set(xticks=np.arange(len(calibration)), xticklabels=[f'{v:g}%' for v in calibration.infill_percent],
           xlabel='制造填充率（仅事后统计）', ylabel='区域均值相对独立标定差异 (%)',
           title=f'两组均使用粗网格所选γ={gamma:.3g}，ν={cfg.PRIMARY_NU:g}')
    ax.legend(fontsize=7.5)
    for extension in ('png','svg'):
        fig.savefig(output/f'mesh_comparison.{extension}', dpi=300)
    plt.close(fig)
    display_metrics = metrics.copy()
    display_metrics['轴向RMSE (μm)'] = display_metrics.rmse_ux_mm*1000
    display_metrics['横向RMSE (μm)'] = display_metrics.rmse_uy_mm*1000
    replace_names = dict(zip(('coarse','refined'), labels))
    display_metrics = display_metrics[['mesh','case','轴向RMSE (μm)','横向RMSE (μm)']].replace({'mesh':replace_names}).rename(columns={'mesh':'网格','case':'平台'})
    material = summary_frame.drop(columns=['elements_x','elements_y']).replace({'mesh':replace_names}).rename(columns={
        'mesh':'网格','parameters':'独立模量数','average_absolute_regional_difference_percent':'七区平均绝对差异 (%)',
        'maximum_absolute_regional_difference_percent':'最差区域绝对差异 (%)',
        'width_spatial_sd_rms_MPa':'横向空间SD的RMS (MPa)',
        'maximum_axial_profile_slope_MPa_per_mm':'宽度均值最大坡度 (MPa/mm)'})
    coarse_link = Path(os.path.relpath(coarse/'method_comparison/index.html', output)).as_posix()
    page = ['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>真实DIC：固定γ的TV网格加密</title>',
        '<style>body{max-width:1350px;margin:32px auto;padding:0 24px;font:16px/1.7 "Microsoft YaHei",sans-serif}table{border-collapse:collapse;display:block;overflow:auto}th,td{padding:7px;border:1px solid #ddd}img{width:100%}a{color:#075fa2}</style>',
        '<h1>真实DIC：固定γ的TV网格加密</h1>',
        '<p><strong>本轮仅检验固定原γ后的网格影响，细网格没有重新扫描L-curve，不能称为细网格独立选参后的最优结果。</strong></p>',
        f'<p>本轮同一γ下的场间相对RMS差异为{relative_rms:.3f}%；七区均值相对独立标定的平均绝对差异为'
        f'{summary_rows[0]["average_absolute_regional_difference_percent"]:.4f}% → '
        f'{summary_rows[1]["average_absolute_regional_difference_percent"]:.4f}%。宽度均值最大坡度为'
        f'{summary_rows[0]["maximum_axial_profile_slope_MPa_per_mm"]:.1f} → '
        f'{summary_rows[1]["maximum_axial_profile_slope_MPa_per_mm"]:.1f} MPa/mm；过渡变陡与材料识别更准是不同判断。</p>',
        f'<p>求解状态：{solved["status"]}；细网格使用{solved["solver"]["n_evaluations"]}次FE计算，独立一阶核验与保存标记一致。</p>',
        f'<p>{old_mesh.nel_x}×{old_mesh.nel_y} → {mesh.nel_x}×{mesh.nel_y}单元；{old_mesh.n_nod} → {mesh.n_nod}个独立节点模量；全部{len(cases[0].observation.x_meas)}个DIC测点、载荷、位移、面积权重和ν={cfg.PRIMARY_NU:g}保持一致。各y行仍独立，不指定层数、界面或模量。</p>',
        f'<p>两组目标均为 ½rᵀWAr＋γ∫Ω√[(∂E/∂x)²＋(∂E/∂y)²]dA，固定γ={gamma:.12g}，来源仅为粗网格180N的L-curve。280N和独立标定不参与本轮求解设置选择。</p>',
        '<p>位移和模量的Q4网格同时加密；试函数和测试函数仍取Q4，不使用SUPG。初值由粗网格场作Q4插值、对偶场作Gauss点插值及幅值投影；未引入分层约束。数值模量步长×4以抵消单元面积缩小，目标、γ、预算和通过条件不变。</p>',
        f'<p>两网格反演场的面积加权相对RMS差异为{relative_rms:.6g}%（{field_rms:.6g} MPa），最大节点差异为{data["maximum_nodal_field_difference_MPa"]:.6g} MPa；这是场间差异，不是对未知真值的误差。平台、界面位置和独立标定差异需一起看，过渡更陡不能单独证明识别更准。</p>',
        '<img src="mesh_comparison.png" alt="相同精确TV目标在两种网格下的模量与标定差异">',
        '<h2>位移误差：180N识别，280N固定参数预测</h2>',display_metrics.to_html(index=False,float_format=lambda v:f'{v:.6g}'),
        '<h2>材料与空间变化</h2>',material.to_html(index=False,float_format=lambda v:f'{v:.6g}'),
        '<p>两组区域均值和空间SD都用相同物理积分点重算，并在材料区边界处分开积分，避免统计网格改变被误当作恢复改善。因此与旧页按各自原生Gauss点分区统计的数值会有小差别。分区位置仅用于这项事后评价。</p>',
        f'<p>本轮只计算主ν={cfg.PRIMARY_NU:g}，未进行细网格新γ扫描或独立初值重复试验；延续得到的局部驻点不能证明全局最优。实验物理配准、内部自由边界近似及跨批标定转移限制仍保留。SD为横向空间离散或标定组内离散，不是反演置信区间。</p>',
        f'<p><a href="{coarse_link}">原粗网格TV与梯度平方正则对照</a> · <a href="../02_forward/云图.html">本细网格独立标定正演</a> · <a href="common_regional_comparison.csv">共同积分区域统计</a> · <a href="solution/tv_history.csv">迭代与证据</a> · <a href="summary.json">完整数值摘要</a> · <a href="mesh_comparison.svg">矢量图</a></p></html>']
    (output/'index.html').write_text('\n'.join(page), encoding='utf-8')
    overview = root/'结果总览.html'
    text = overview.read_text(encoding='utf-8')
    if 'id="tv-mesh-diagnostic"' not in text:
        link = '<p id="tv-mesh-diagnostic"><strong><a href="mesh_diagnostic/index.html">查看固定γ的TV加密诊断（尚未重新选γ）</a></strong></p>'
        overview.write_text(text.replace('<div class="cards">',link+'\n<div class="cards">',1),encoding='utf-8')
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    args = parser.parse_args()
    print(json.dumps(report(args.root),ensure_ascii=False,indent=2))
