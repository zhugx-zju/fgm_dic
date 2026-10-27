"""Post-process independent uniform-sample runs without selecting or tuning E/gamma.

python -B -m lib.uniform_sample_analysis --cases RUN1 RUN2 RUN3 --audit AUDIT --output NEW_DIR
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fgm_asm.load_case import evaluate_observation
from fgm_asm.research import solve_forward
from lib.dic_case_workflow import homogeneous_dic_initial, load_prepared
from lib.dic_preprocessing import dic_relative_error_scale
from lib.experiment_data import file_sha256


def area_statistics(mesh, field: np.ndarray, *, interior: bool = False) -> dict:
    """Q4 moments by 2x2 Gauss integration, exact for E and E**2 on each cell.

    Interior statistics include quadrature points in the central 80% of each
    coordinate span. They diagnose boundary sensitivity, not a new selection set.
    """
    values = (field[mesh.ele_nods_id - 1] @ mesh.gauss_N.T).T
    weights = mesh.gauss_abs_det_j * mesh.gauss_w[:, None]
    if interior:
        x = np.einsum('en,gn->ge', mesh.x_ele, mesh.gauss_N)
        y = np.einsum('en,gn->ge', mesh.y_ele, mesh.gauss_N)
        mask = ((x >= .1*mesh.geo_l) & (x <= .9*mesh.geo_l) &
                (y >= .1*mesh.geo_h) & (y <= .9*mesh.geo_h))
        values, weights = values[mask], weights[mask]
    mean = float(np.average(values, weights=weights))
    sd = float(np.sqrt(np.average((values-mean)**2, weights=weights)))
    return dict(mean_E_MPa=mean, spatial_sd_MPa=sd, spatial_CV_percent=100*sd/mean)


def error_statistics(case, mesh, field, thickness, nu):
    u, _ = solve_forward(mesh, case.bc_info['tensile_end_force'], thickness, nu, field)
    state = evaluate_observation(case, u)
    residual = state['residual'].reshape(-1, 2)
    weights = case.observation.weight_vector.reshape(-1, 2)
    rmse = np.sqrt(np.sum(weights*residual**2, axis=0)/weights.sum(axis=0))
    scale = dic_relative_error_scale(case)
    return dict(rms_mm=float(state['weighted_rms']), axial_rmse_mm=float(rmse[0]),
                transverse_rmse_mm=float(rmse[1]), axial_relative_RMSE_percent=float(100*rmse[0]/scale[0]),
                transverse_relative_RMSE_percent=float(100*rmse[1]/scale[1]),
                axial_DIC_deformation_RMS_mm=float(scale[0]), transverse_DIC_deformation_RMS_mm=float(scale[1]))


def analyze(case_roots: list[Path], audit_path: Path, output: Path):
    output.mkdir(parents=True, exist_ok=False)
    audit = json.loads(audit_path.read_text(encoding='utf-8'))
    rows, scans, maps, provenance = [], [], [], []
    profiles = []
    for root in case_roots:
        cfg, mesh, cases, _, _ = load_prepared(root/'01_prepared')
        assert len(cases) == 1 and not cfg.PREDICT_CASES
        case = cases[0]
        meta_path = root/'03_inverse'/'run_metadata.json'
        meta = json.loads(meta_path.read_text(encoding='utf-8'))
        scan = pd.read_csv(root/'03_inverse'/'selection.csv')
        run = cfg.PLATFORMS[case.name]['run']
        common = dict(run=run, nu=cfg.PRIMARY_NU, delta_force_N=case.metadata['measured_force_N'],
                      length_mm=mesh.geo_l, width_mm=mesh.geo_h,
                      selected_gamma=meta.get('selected_gamma'), l_curve_status=meta.get('l_curve_status'),
                      inverse_status=meta['status'])
        provenance.append(dict(case_directory=str(root.resolve()),
                               inverse_metadata_sha256=file_sha256(meta_path),
                               prepared_fields_sha256=file_sha256(root/'01_prepared'/'fields.npz')))
        # Independent check: closed-form uniaxial least-squares constant E,
        # using a separate analytical compliance rather than an FE unit solution.
        xy = np.column_stack([case.observation.x_meas, case.observation.y_meas])
        strain_shape = np.column_stack([xy[:, 0], -cfg.PRIMARY_NU*xy[:, 1]])
        target = evaluate_observation(case, np.zeros(mesh.n_dof))['residual']
        obs = case.observation
        analytical_case = replace(case, observation=replace(obs, u_measured=strain_shape.ravel()))
        compliance_shape = evaluate_observation(analytical_case, np.zeros(mesh.n_dof))['residual']
        axial_strain = float(compliance_shape @ (obs.weight_vector*target) /
                             (compliance_shape @ (obs.weight_vector*compliance_shape)))
        analytic_E = common['delta_force_N']/(cfg.LOAD_SECTION_WIDTH*cfg.THICKNESS*axial_strain)
        baseline, info = homogeneous_dic_initial(mesh, cases, cfg.THICKNESS, cfg.PRIMARY_NU)
        np.testing.assert_allclose(info['uniform_E_MPa'], analytic_E, rtol=1e-7)
        rows.append(dict(common, model='uniform_baseline', **area_statistics(mesh, baseline),
                         analytic_uniform_E_MPa=analytic_E,
                         **error_statistics(case, mesh, baseline, cfg.THICKNESS, cfg.PRIMARY_NU)))
        for record in scan.to_dict('records'):
            recovered = record.get('solution_directory')
            trial = (root/'03_inverse'/recovered if isinstance(recovered, str)
                     else root/'03_inverse'/'l_curve'/f"gamma_{record['gamma']:g}")
            p = trial/'fields.npz'
            with np.load(p) as fields:
                stats = area_statistics(mesh, fields['E_inverse'])
            scans.append(dict(run=run, **record, **stats))
        if meta.get('selected_gamma') is None:
            continue
        for solver in meta['solvers']:
            nu = solver['nu']
            p = root/'03_inverse'/f'nu_{nu:.2f}'/'fields.npz'
            with np.load(p) as fields:
                field = fields['E_inverse'].copy()
                saved_residual = fields[f'{case.name}_residual'].copy()
            stats = area_statistics(mesh, field)
            interior = area_statistics(mesh, field, interior=True)
            errors = error_statistics(case, mesh, field, cfg.THICKNESS, nu)
            np.testing.assert_allclose(errors['axial_rmse_mm'],
                                       np.sqrt(np.mean(saved_residual.reshape(-1, 2)[:, 0]**2)), rtol=1e-8)
            rows.append(dict({**common, 'nu': nu}, model='H1', **stats, **errors,
                             interior_CV_percent=interior['spatial_CV_percent'],
                             min_nodal_E_MPa=float(field.min()), max_nodal_E_MPa=float(field.max()),
                             solver_converged=solver['converged'], stop_message=solver['message'],
                             gradient_tolerance_satisfied=solver['gradient_tolerance_satisfied']))
            if nu == cfg.PRIMARY_NU:
                maps.append((run, mesh, field, stats))
                grid = field.reshape(mesh.nods_y, mesh.nods_x)
                w = np.ones(mesh.nods_y); w[[0, -1]] = .5
                mean = np.average(grid, axis=0, weights=w)
                sd = np.sqrt(np.average((grid-mean)**2, axis=0, weights=w))
                profiles.extend(dict(run=run, x_mm=x, mean_E_MPa=m, transverse_sd_MPa=s)
                                for x, m, s in zip(mesh.plot_x[0], mean, sd))
    table, scan_table = pd.DataFrame(rows), pd.DataFrame(scans)
    table.to_csv(output/'uniformity_summary.csv', index=False)
    scan_table.to_csv(output/'gamma_uniformity.csv', index=False)
    pd.DataFrame(profiles).to_csv(output/'axial_profiles.csv', index=False)
    (output/'input_audit.json').write_bytes(audit_path.read_bytes())
    (output/'provenance.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
    colors = {1:'#0072B2', 2:'#D55E00', 3:'#009E73'}
    if maps:
        fig, axes = plt.subplots(len(maps), 1, figsize=(10, 2.1*len(maps)), layout='constrained', squeeze=False)
        low, high = min(f.min() for _, _, f, _ in maps), max(f.max() for _, _, f, _ in maps)
        for ax, (run, mesh, field, stats) in zip(axes[:, 0], maps):
            im = ax.pcolormesh(mesh.plot_x, mesh.plot_y, field.reshape(mesh.nods_y, mesh.nods_x),
                               shading='gouraud', cmap='viridis', vmin=low, vmax=high, rasterized=True)
            ax.set(xlabel='Axial x (mm)', ylabel='Width y (mm)', aspect='equal',
                   title=f"Run {run}: mean {stats['mean_E_MPa']:.1f} MPa; spatial CV {stats['spatial_CV_percent']:.2f}%")
        fig.colorbar(im, ax=axes[:, 0].tolist(), label='E (MPa)', shrink=.8)
        _save_figure(fig, output/'modulus_fields')
        fig, ax = plt.subplots(figsize=(9, 4), layout='constrained')
        for run, group in pd.DataFrame(profiles).groupby('run'):
            x, mean, sd = [group[k].to_numpy() for k in ('x_mm', 'mean_E_MPa', 'transverse_sd_MPa')]
            ax.plot(x, mean, color=colors[run], label=f'Run {run}')
            ax.fill_between(x, mean-sd, mean+sd, color=colors[run], alpha=.15)
        ax.set(xlabel='Axial x (mm)', ylabel='Width mean E (MPa)', title='Shading: transverse spatial SD (not uncertainty)')
        ax.legend(); _save_figure(fig, output/'axial_profiles')
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
    for run, group in scan_table.groupby('run'):
        group = group.sort_values('gamma')
        valid = group.converged.astype(bool)
        for ax, y in zip(axes, ('residual_norm', 'spatial_CV_percent')):
            ax.semilogx(group.gamma, group[y].where(valid), color=colors[run], label=f'Run {run}')
            ax.semilogx(group.gamma[~valid], group.loc[~valid, y], 'x', color=colors[run],
                        markersize=4, alpha=.45)
            selected = group[group.get('selected', False) == True]
            if not selected.empty:
                ax.plot(selected.gamma, selected[y], '*', color=colors[run], markersize=11)
            ax.set_xlabel(r'$\gamma$')
    axes[0].set_ylabel('Area-weighted displacement residual norm (mm$^2$)')
    axes[1].set_ylabel('Spatial CV of E (%)'); axes[0].legend()
    fig.suptitle('Stars: selected gamma; crosses: failed solves, excluded from selection', fontsize=11)
    _save_figure(fig, output/'gamma_tradeoff')
    _write_report(table, audit, provenance, output)
    print(table.to_string(index=False))


def _save_figure(fig, path):
    for extension in ('png', 'svg'):
        fig.savefig(path.with_suffix('.'+extension), dpi=300)
    plt.close(fig)


def _write_report(table, audit, provenance, output):
    primary = table[(table.model == 'H1') & (table.nu == .26)]
    conclusions = ['三次均获得L-curve所选解，但不能认定三次稳定恢复了均匀模量场。第3次最接近均匀，仍存在沿轴向的平缓趋势；第1、2次非均匀程度更大。未事先规定均匀性验收阈值，以下为定量描述。']
    for row in primary.to_dict('records'):
        conclusions.append(f"第{row['run']}次：面积均值{row['mean_E_MPa']:.2f} MPa，空间CV={row['spatial_CV_percent']:.2f}%，内部CV={row['interior_CV_percent']:.2f}%；轴向/横向相对RMSE分别{row['axial_relative_RMSE_percent']:.2f}%/{row['transverse_relative_RMSE_percent']:.2f}%。")
    lines = ['# 1026 均匀 60% 试样：自由二维 H1 反演', '', *conclusions, '',
             '三次独立参考图分别反演，不合并为多载荷同参考数据。厚度约3.2 mm，长宽取各自DIC参考坐标跨度；原Y换为FEM轴向x，原X换为横向y，坐标和位移保持同一毫米尺度。', '',
             '模型为平面应力、两端均匀牵引及侧面自由；主ν=0.26，ν=0.24/0.28/0.33为同γ敏感性比较。每次121×19单元、2440个独立节点模量；目标为0.5*r.T@W_area@r+0.5*γ*∫|∇E|²，不强制均匀。', '',
             'γ只用该次DIC的原有50点L-curve规则选择，范围1e-12—1e-4；无有效拐点时不补选。初值只来自该次DIC的均匀拟合，独立标定未进入初值、目标或选参。', '',
             '| 次数 | 长×宽 mm | 起载 N | 末载 N | 增量 N | 末载来源 |', '|---|---:|---:|---:|---:|---|']
    for r, a in audit.items():
        lines.append(f"| {r} | {a['length_mm']:.3f}×{a['width_mm']:.3f} | {a['start_force_N']:.5f} | {a['end_force_N']:.5f} | {a['delta_force_N']:.5f} | {'用户假定平台末载，无同期载荷' if a['end_force_assumed'] else '记录最后一条'} |")
    lines += ['', '**参考状态限制：** 始末差值按用户指定口径计算。尤其第3次起始记录已为250.72479 N，早于该记录起点的DIC参考帧载荷尚未同步核实。因此绝对模量依赖此参考对应关系、DIC覆盖宽度等于承载宽度以及厚度近似值。第3次窗口均值379.6495967 N与末记录379.78479 N相差0.13519 N，相当于本次增量的0.105%。', '',
              '扫描若仅因L-BFGS线搜索异常缺少拐点支撑，使用项目既有--resume恢复接口：只重置该候选的L-BFGS历史，迭代与函数调用累计仍限制在原1500/12000预算内。原候选、恢复候选和恢复前选参表分别保留；不追加预算，不依据均匀程度或标定结果补选γ。', '',
              '| 次数 | 模型 | ν | γ | 面积均值 MPa | 空间CV % | 内部CV % | 轴向相对RMSE % | 横向相对RMSE % |', '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in table.to_dict('records'):
        if row['nu'] != .26:
            continue
        gamma = f"{row['selected_gamma']:.3g}" if row['model'] == 'H1' else '—'
        lines.append(f"| {row['run']} | {row['model']} | {row['nu']:.2f} | {gamma} | {row['mean_E_MPa']:.2f} | {row['spatial_CV_percent']:.2f} | {row.get('interior_CV_percent',float('nan')):.2f} | {row['axial_relative_RMSE_percent']:.2f} | {row['transverse_relative_RMSE_percent']:.2f} |")
    lines += ['', 'CV=面积加权空间标准差/面积均值，不是参数不确定性。内部统计仅用轴向与横向各10%—90%范围内的Gauss点；全域一、二阶矩使用Q4的2×2 Gauss积分。未预设通过/失败阈值。横向阴影是空间SD，不是置信区间。', '',
              '相对位移误差的分母为DIC去掉两平移和一微小转动后的相应分量RMS，全部测点共用，不逐点除以过零位移。这里只移除刚体模式，不删除仿射伸长/收缩。', '',
              '均匀基线只有一个E参数，采用相同几何、载荷、ν、观测位置和面积权重。其拟合模量同时用均匀有限元解和解析单轴柔度的独立计算核验到相对1e-7。保存反演场重新正演得到的残差与已保存轴向RMSE核验一致；全部ν的停止信息和GTOL状态见uniformity_summary.csv。函数下降停止不等于满足梯度阈值。', '',
              '同一数据用于反演和回代，以上是拟合检查，没有独立物理验证。较大γ必然更偏好均匀场，因此应结合gamma_tradeoff与均匀基线判断；不能通过挑选最均匀的候选宣称恢复成功。正反演使用同一标量各向同性模型，空间变化也可能吸收边界、各向异性、参考状态或DIC系统误差，不能直接认定为材料真实非均匀。', '',
              '全域仿射应变诊断（原DIC轴，不作为材料泊松比测量）：']
    for r, a in audit.items():
        b = np.array(a['affine_DX_DY_coefficients'])
        lines.append(f"- 第{r}次：εXX={b[1,0]:.7g}，εYY={b[2,1]:.7g}，−εXX/εYY={-b[1,0]/b[2,1]:.4f}。该比值与预设ν差异较大，提示均匀各向同性单轴模型的适用性需另验。")
    lines += ['', '![二维模量场](modulus_fields.png)', '', '![沿长度统计](axial_profiles.png)', '', '![γ与均匀性权衡](gamma_tradeoff.png)', '',
              '复现：各次源配置已冻结在对应01_prepared/configuration_snapshot.py；对各冻结输入分别执行script.step3_inverse_dic，已有目录拒绝覆盖。后处理命令见lib/uniform_sample_analysis.py，使用--cases、--audit和新的--output。所有原始输入SHA256见input_audit.json，上游结果身份见provenance.json；详细数值场和全部50点候选在以下本地目录：']
    lines += [f"- {p['case_directory']}" for p in provenance]
    (output/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', type=Path, nargs='+', required=True)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    analyze(args.cases, args.audit, args.output)
