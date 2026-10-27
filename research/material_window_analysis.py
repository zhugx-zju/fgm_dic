"""Audit and compare a cropped inversion with the full-domain fit on identical data.

python -B -m lib.material_window_analysis --cases CROP_ROOTS --baselines FULL_ROOTS --output NEW_DIR
Only saved fields are evaluated; this module never selects parameters or optimizes.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import platform
import subprocess

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fgm_asm.load_case import evaluate_observation
from fgm_asm.observation import build_observation_operator
from fgm_asm.research import solve_forward
from lib.dic_case_workflow import load_prepared
from lib.dic_preprocessing import relative_displacement_errors
from lib.experiment_data import file_sha256
from .sample2_result_analysis import save_figure


def scalar_values(mesh, field, xy):
    """Evaluate the saved continuous Q4 modulus at physical xy (mm)."""
    operator = build_observation_operator(mesh, xy[:, 0], xy[:, 1])
    return np.asarray(operator[0::2, 0::2] @ field).ravel()


def common_quadrature(mesh, old_mesh, shift, boundaries):
    """Exact Q4 moments on rectangles split at both meshes and material interfaces."""
    axes = []
    for new_axis, old_axis, delta, span, interfaces in (
            (mesh.plot_x[0], old_mesh.plot_x[0], shift[0], mesh.geo_l, boundaries),
            (mesh.plot_y[:, 0], old_mesh.plot_y[:, 0], shift[1], mesh.geo_h, ())):
        old_inside = old_axis - delta
        old_inside = old_inside[(old_inside > 0) & (old_inside < span)]
        edges = np.unique(np.r_[new_axis, old_inside, interfaces])
        if np.any(np.diff(edges) <= 0) or edges[0] != 0 or edges[-1] != span:
            raise ValueError('Common quadrature must partition the cropped rectangle')
        half = np.diff(edges) / 2
        mid = (edges[1:] + edges[:-1]) / 2
        axes.append(((mid[:, None]+half[:, None]*np.array([-1, 1])/np.sqrt(3)).ravel(),
                     np.repeat(half, 2)))
    x, y = np.meshgrid(axes[0][0], axes[1][0])
    weights = np.outer(axes[1][1], axes[0][1]).ravel()
    return np.column_stack([x.ravel(), y.ravel()]), weights


def analyze(roots: list[Path], baselines: list[Path], output: Path) -> None:
    if len(roots) != len(baselines) or not roots:
        raise ValueError('Supply one corresponding full-domain baseline per crop')
    output.mkdir(parents=True, exist_ok=True)
    if any((output / name).exists() for name in ('report.md', 'summary.csv', 'profiles.png')):
        raise FileExistsError('Choose a new report directory')
    summaries, regions, profiles, audits, plots, stopping = [], [], [], [], [], []
    for root, baseline in zip(roots, baselines):
        cfg, mesh, cases, meta, _ = load_prepared(root / '01_prepared')
        old_cfg, old_mesh, old_cases, old_meta, _ = load_prepared(baseline / '01_prepared')
        case, old_case = cases[0], old_cases[0]
        run = cfg.SAMPLE_RUN
        if (len(cases) != 1 or len(old_cases) != 1 or cfg.PREDICT_CASES or
                cfg.SAMPLE_RUN != old_cfg.SAMPLE_RUN):
            raise ValueError('Compare matching independent single-reference runs')
        # Every retained observation must be exactly the same physical datum.
        with np.load(root / '01_prepared/fields.npz') as new, np.load(baseline / '01_prepared/fields.npz') as old:
            ids = {tuple(pair): i for i, pair in enumerate(old[f'{case.name}_raw_grid_ids'])}
            indices = np.array([ids[tuple(pair)] for pair in new[f'{case.name}_raw_grid_ids']])
            for suffix in ('raw_reference_coordinates', 'raw_mean_displacement'):
                np.testing.assert_array_equal(new[f'{case.name}_{suffix}'], old[f'{case.name}_{suffix}'][indices])
            np.testing.assert_array_equal(case.observation.u_measured.reshape(-1, 2), old[f'{case.name}_observed'][indices])
        np.testing.assert_allclose(cfg.REGISTRATION['matrix'], old_cfg.REGISTRATION['matrix'], rtol=0, atol=0)
        np.testing.assert_allclose(case.observation.weight_vector.reshape(-1, 2),
                                   old_case.observation.weight_vector.reshape(-1, 2)[indices], rtol=0, atol=0)
        for key in ('THICKNESS', 'LOAD_SECTION_WIDTH', 'PRIMARY_NU', 'NU_VALUES', 'MAX_ITER', 'MAX_FUN',
                    'FTOL', 'GTOL', 'GAMMA_CANDIDATES', 'DISPLACEMENT_COMPONENT_WEIGHTS'):
            if getattr(cfg, key) != getattr(old_cfg, key):
                raise ValueError(f'Unexpected change in comparison setting: {key}')
        np.testing.assert_allclose(case.bc_info['tensile_end_force'], old_case.bc_info['tensile_end_force'], rtol=0, atol=0)
        # Translate only; the material map is inherited before cropping.
        shift = np.asarray(old_meta['offset_mm']) - np.asarray(meta['offset_mm'])
        xy = np.column_stack([case.observation.x_meas, case.observation.y_meas])
        old_xy = xy + shift
        np.testing.assert_allclose(old_xy, np.column_stack([old_case.observation.x_meas, old_case.observation.y_meas])[indices], atol=1e-12, rtol=0)
        np.testing.assert_allclose(xy[:, 0] + cfg.MANUFACTURING_X_OFFSET_MM,
                                   old_xy[:, 0] + old_cfg.MANUFACTURING_X_OFFSET_MM, atol=1e-12, rtol=0)
        old_H = build_observation_operator(old_mesh, old_xy[:, 0], old_xy[:, 1])
        restricted_case = replace(case, observation=replace(case.observation, H=old_H))
        inv_meta = json.loads((root / '03_inverse/run_metadata.json').read_text(encoding='utf-8'))
        old_inv_meta = json.loads((baseline / '03_inverse/run_metadata.json').read_text(encoding='utf-8'))
        if inv_meta['status'] not in ('computed', 'inverse_not_converged') or inv_meta['selected_gamma'] is None:
            raise ValueError(f'{root}: selected inversion not computed: {inv_meta["status"]}')
        if not (root / '04_comparison/run_metadata.json').exists():
            raise ValueError(f'{root}: complete step 4 first')
        scan = pd.read_csv(root / '03_inverse/selection.csv')
        # Shared partition makes means and second moments exact for both Q4 fields.
        points, weights = common_quadrature(mesh, old_mesh, shift, cfg.REGION_BOUNDARIES_X)
        groups = np.searchsorted(cfg.REGION_BOUNDARIES_X, points[:, 0], side='right')
        calibration = pd.DataFrame(meta['calibration'])
        for nu in cfg.NU_VALUES:
            with np.load(root / f'03_inverse/nu_{nu:.2f}/fields.npz') as saved:
                field = saved['E_inverse'].copy()
                recorded = saved[f'{case.name}_residual'].copy()
            with np.load(baseline / f'03_inverse/nu_{nu:.2f}/fields.npz') as saved:
                old_field = saved['E_inverse'].copy()
                old_u = saved[f'{case.name}_U'].copy()
            # A fresh forward solve independently reproduces the crop prediction.
            u, _ = solve_forward(mesh, case.bc_info['tensile_end_force'], cfg.THICKNESS, nu, field)
            state = evaluate_observation(case, u)
            np.testing.assert_allclose(state['residual'], recorded, rtol=1e-8, atol=1e-10)
            full_state = evaluate_observation(restricted_case, old_u)
            with np.load(root / f'02_forward/nu_{nu:.2f}/fields.npz') as saved:
                forward_prediction = saved[f'{case.name}_prediction'].copy()
            values = {'crop_inverse': scalar_values(mesh, field, points),
                      'full_inverse_same_roi': scalar_values(old_mesh, old_field, points + shift)}
            differences = {}
            for model, gauss_values in values.items():
                local = []
                for group, record in enumerate(calibration.to_dict(orient='records')):
                    mask = groups == group
                    mean = float(np.average(gauss_values[mask], weights=weights[mask]))
                    sd = float(np.sqrt(np.average((gauss_values[mask]-mean)**2, weights=weights[mask])))
                    difference = 100*(mean / record['mean_E_MPa'] - 1)
                    local.append(abs(difference))
                    regions.append(dict(run=run, nu=nu, model=model, **record,
                                        inverse_mean_E_MPa=mean, inverse_spatial_sd_MPa=sd,
                                        difference_percent=difference))
                differences[model] = float(np.mean(local))
            solver = next(s for s in inv_meta['solvers'] if np.isclose(s['nu'], nu))
            stopping.append(dict(run=run, **solver))
            for model, prediction in (('crop_inverse', state['prediction']),
                                      ('full_inverse_same_roi', full_state['prediction']),
                                      ('calibrated_forward', forward_prediction)):
                summaries.append(dict(run=run, nu=nu, model=model,
                    gamma=(inv_meta['selected_gamma'] if model == 'crop_inverse' else
                           old_inv_meta['selected_gamma'] if model == 'full_inverse_same_roi' else np.nan),
                    **relative_displacement_errors(case, prediction),
                    four_region_mean_absolute_difference_percent=differences.get(model, np.nan),
                    solver_converged=(solver['converged'] if model == 'crop_inverse' else None),
                    points=len(xy), length_mm=cfg.GEO_L, width_mm=cfg.GEO_H,
                    scan_success_count=int(scan.converged.sum()), scan_count=len(scan)))
            if np.isclose(nu, cfg.PRIMARY_NU):
                # Dense shared physical profile grid for comparison, not inversion.
                px, py = np.meshgrid(np.linspace(0, cfg.GEO_L, 400), np.linspace(0, cfg.GEO_H, 81))
                pxy = np.column_stack([px.ravel(), py.ravel()])
                transverse_weights = np.ones(81); transverse_weights[[0, -1]] = .5
                pdata = {}
                for model, pm, pf, delta in (('crop_inverse', mesh, field, np.zeros(2)),
                                            ('full_inverse_same_roi', old_mesh, old_field, shift)):
                    array = scalar_values(pm, pf, pxy + delta).reshape(px.shape)
                    mean = np.average(array, weights=transverse_weights, axis=0)
                    sd = np.sqrt(np.average((array-mean)**2, weights=transverse_weights, axis=0))
                    pdata[model] = (mean, sd)
                    profiles.extend(dict(run=run, model=model, manufacturing_x_mm=x,
                                         mean_E_MPa=e, transverse_spatial_sd_MPa=s)
                                    for x, e, s in zip(px[0]+cfg.MANUFACTURING_X_OFFSET_MM, mean, sd))
                plots.append(dict(run=run, cfg=cfg, mesh=mesh, field=field.reshape(mesh.nods_y, mesh.nods_x),
                                  x=px[0]+cfg.MANUFACTURING_X_OFFSET_MM, profiles=pdata, calibration=calibration,
                                  converged=solver['converged']))
        for source in meta['platforms'][case.name]['sources']:
            if file_sha256(Path(source['path'])) != source['sha256']:
                raise ValueError(f'Input hash changed: {source["path"]}')
        anchor = baseline / '01_prepared/run_metadata.json'
        if file_sha256(anchor) != cfg.REGISTRATION['anchor_prepared_metadata_sha256']:
            raise ValueError('Material anchor changed')
        audits.append(dict(run=run, root=str(root.resolve()), baseline=str(baseline.resolve()),
            extraction=meta['platforms'][case.name]['extraction'],
            exact_raw_data_subset_verified=True, coordinate_map_verified=True,
            matched_force_weights_settings_verified=True, all_raw_source_hashes_verified=True,
            fresh_forward_replay_verified_all_nu=True, residual_atol_mm=1e-10,
            anchor_metadata_sha256=file_sha256(anchor),
            inverse_metadata_sha256=file_sha256(root / '03_inverse/run_metadata.json'),
            l_curve_status=inv_meta['l_curve_status'], inverse_status=inv_meta['status']))
    table, reg = pd.DataFrame(summaries), pd.DataFrame(regions)
    table.to_csv(output / 'summary.csv', index=False, lineterminator='\n')
    reg.to_csv(output / 'regional_comparison.csv', index=False, lineterminator='\n')
    pd.DataFrame(profiles).to_csv(output / 'axial_profiles.csv', index=False, lineterminator='\n')
    pd.DataFrame(stopping).to_csv(output / 'solver_stopping.csv', index=False, lineterminator='\n')
    (output / 'audit.json').write_text(json.dumps(audits, ensure_ascii=False, indent=2), encoding='utf-8')
    environment = dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
        git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        source_sha256=file_sha256(Path(__file__)),
        command='python -B -m lib.material_window_analysis --cases ' + ' '.join(map(str, roots)) +
                ' --baselines ' + ' '.join(map(str, baselines)) + ' --output ' + str(output))
    (output / 'environment.json').write_text(json.dumps(environment, ensure_ascii=False, indent=2), encoding='utf-8')
    draw(plots, output)
    write_report(table, reg, stopping, audits, output)
    print(table[np.isclose(table.nu, plots[0]['cfg'].PRIMARY_NU)].to_string(index=False))


def draw(plots, output):
    plt.rcParams.update({'font.family': 'Microsoft YaHei', 'font.size': 10, 'axes.unicode_minus': False,
                         'svg.fonttype': 'path', 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(len(plots), 1, figsize=(10, 8), sharex=True, sharey=True, constrained_layout=True)
    for ax, p in zip(np.atleast_1d(axes), plots):
        for model, color, label, style in (('crop_inverse', '#0072B2', '仅40%–70%反演', '-'),
                                            ('full_inverse_same_roi', '#009E73', '全段反演在同一区域', '--')):
            mean, sd = p['profiles'][model]
            ax.plot(p['x'], mean, color=color, ls=style, lw=1.6, label=label)
            if model == 'crop_inverse':
                ax.fill_between(p['x'], mean-sd, mean+sd, color=color, alpha=.15, label='裁剪反演横向空间SD')
        edges = np.r_[p['x'][0], p['cfg'].REGION_BOUNDARIES_MANUFACTURING_X, p['x'][-1]]
        step_x = np.repeat(edges, 2)[1:-1]
        cal = p['calibration']
        ax.plot(step_x, np.repeat(cal.mean_E_MPa, 2), color='#D55E00', lw=1.4, label='1023约束标定参照')
        for edge in edges[1:-1]:
            ax.axvline(edge, color='.7', ls=':', lw=.7)
        suffix = '' if p['converged'] else '（未收敛，仅诊断）'
        ax.set(title=f'第{p["run"]}次，ν={p["cfg"].PRIMARY_NU}{suffix}', ylabel='E (MPa)')
    np.atleast_1d(axes)[0].legend(ncol=2, frameon=False, fontsize=9)
    np.atleast_1d(axes)[-1].set_xlabel('制造坐标 ξ (mm)，各次沿用原参考图定位')
    save_figure(fig, output / 'profiles')
    fig, axes = plt.subplots(len(plots), 1, figsize=(10, 6.5), constrained_layout=True)
    low, high = min(p['field'].min() for p in plots), max(p['field'].max() for p in plots)
    for ax, p in zip(np.atleast_1d(axes), plots):
        mesh, cfg = p['mesh'], p['cfg']
        im = ax.pcolormesh(mesh.plot_x+cfg.MANUFACTURING_X_OFFSET_MM, mesh.plot_y, p['field'],
                           shading='auto', cmap='viridis', vmin=low, vmax=high)
        for edge in cfg.REGION_BOUNDARIES_MANUFACTURING_X:
            ax.axvline(edge, color='white', ls='--', lw=.6)
        suffix = '' if p['converged'] else '（未收敛，仅诊断）'
        ax.set(title=f'第{p["run"]}次：仅40%–70%自由二维H1反演{suffix}', xlabel='制造坐标 ξ (mm)', ylabel='y (mm)', aspect='equal')
    fig.colorbar(im, ax=axes, label='E (MPa)', shrink=.85)
    save_figure(fig, output / 'modulus_fields')


def write_report(table, regions, stopping, audits, output):
    main = table[np.isclose(table.nu, .26)]
    names = {'crop_inverse': '仅40%–70%反演', 'full_inverse_same_roi': '全段反演在同区评价', 'calibrated_forward': '四区标定正演'}
    lines = ['# 1026 sample_2：仅40%–70%区段的自由二维H1反演', '',
        '本次从三次独立参考图中分别裁剪40%、50%、60%、70%四区，仅使用裁剪后的DIC重新求初值、扫描γ并反演。全段反演只用于计算后的比较，未用于初始化或选参。', '',
        '| 次数 | 方法 | γ | 轴向位移相对误差 % | 横向位移相对误差 % | 双分量相对误差 % | 四区模量平均绝对参照差 % |',
        '|---|---|---:|---:|---:|---:|---:|']
    for r in main.itertuples():
        label = names[r.model]
        if r.model == 'crop_inverse' and not r.solver_converged:
            label += '（未收敛，仅诊断）'
        gamma = f'{r.gamma:.3g}' if np.isfinite(r.gamma) else '—'
        difference = f'{r.four_region_mean_absolute_difference_percent:.3f}' if np.isfinite(r.four_region_mean_absolute_difference_percent) else '—'
        lines.append(f'| {r.run} | {label} | {gamma} | {r.axial_relative_error_percent:.3f} | {r.transverse_relative_error_percent:.3f} | {r.displacement_relative_error_percent:.3f} | {difference} |')
    conclusion = []
    for r in main[main.model == 'crop_inverse'].itertuples():
        old = main[(main.run == r.run) & (main.model == 'full_inverse_same_roi')].iloc[0]
        status = '函数下降停止正常' if r.solver_converged else '线搜索异常，以下仅为诊断场'
        conclusion.append(f'第{r.run}次{status}：四区平均绝对参照差由{old.four_region_mean_absolute_difference_percent:.2f}%变为{r.four_region_mean_absolute_difference_percent:.2f}%，双分量位移误差由{old.displacement_relative_error_percent:.2f}%变为{r.displacement_relative_error_percent:.2f}%。')
    failed = [f'第{s["run"]}次ν={s["nu"]:.2f}' for s in stopping if not s['converged']]
    conclusion += ['异常计算：' + ('、'.join(failed) if failed else '无') + '。未重启优化、增加预算或按标定调整γ。']
    lines[2:2] = ['**结果：** ' + ' '.join(conclusion), '']
    lines += ['', '所有位移比较均在同一裁剪DIC点、同一面积权重与同一去刚体DIC变形分母下评价，全段场在这些点重新拟合三刚体自由度。分量误差不是逐点相除。四区指标为各区面积均值相对参照的绝对差的算术均值；共同积分网格沿两种有限元网格的全部单元边及制造界面细分，每个子矩形采用2×2 Gauss积分，对两种连续Q4场的均值及二阶矩均精确。', '',
        '## 四区模量（主ν=0.26）', '', '| 次数 | 填充率 % | 标定均值 MPa | 裁剪反演 MPa | 全段反演同区 MPa | 裁剪参照差 % |', '|---|---:|---:|---:|---:|---:|']
    for r in regions[(regions.model == 'crop_inverse') & np.isclose(regions.nu, .26)].itertuples():
        old = regions[(regions.run == r.run) & (regions.infill_percent == r.infill_percent) &
                      (regions.model == 'full_inverse_same_roi') & np.isclose(regions.nu, .26)].iloc[0]
        lines.append(f'| {r.run} | {r.infill_percent} | {r.mean_E_MPa:.2f} | {r.inverse_mean_E_MPa:.2f} | {old.inverse_mean_E_MPa:.2f} | {r.difference_percent:+.2f} |')
    lines += ['', '## 输入、方法和限制', '',
        '裁剪窗口为制造坐标[-57,8] mm。各次坐标锚点仅来自本次同一帧组的原全段冻结输入；不重新居中、不缩放、不在不同参考图之间共享锚点。40%区受原DIC覆盖限制，只比较现有覆盖部分；70%区末端不足一行采样间距。原始坐标、位移、网格ID均逐项核验为原全段的严格子集。', '',
        '| 次数 | 实际制造范围 mm | DIC点数 | 原点数 | 裁剪长×宽 mm |', '|---|---|---:|---:|---:|']
    for a in audits:
        e = a['extraction']; r = main[(main.run == a['run']) & (main.model == 'crop_inverse')].iloc[0]
        lines.append(f'| {a["run"]} | {e["selected_manufacturing_bounds_mm"][0]:.4f} 至 {e["selected_manufacturing_bounds_mm"][1]:.4f} | {e["selected_points"]} | {e["input_points"]} | {r.length_mm:.4f}×{r.width_mm:.4f} |')
    lines += ['', '保持121×19单元、2440个独立节点模量和独立y行；裁剪后轴向单元变细，因此本实验同时改变域长度、观测范围、端面位置及轴向网格间距，不能将差异单独归因于去掉高填充区。原载荷增量229.2224/309.3808/397.7152 N、厚度3.2 mm、承载宽度及各次参考对应假设保留。新截取端面采用均匀牵引近似，未测得该内部截面的实际牵引分布。', '',
        'H1目标、50点γ=1e-12至1e-4、1500迭代/12000函数预算、FTOL/GTOL与原全段一致。主ν=0.26；0.24/0.28/0.33仅作固定γ敏感性分析。使用1023的individual_window_joint_3pct_monotone外部参照；组内3%及单调性为标定选窗约束，非独立验证性质。参照不进入反演目标、初值和选参。', '',
        '## 求解停止与核验', '', '| 次数 | ν | 函数停止成功 | GTOL满足 | 梯度∞范数 | 停止信息 |', '|---|---:|---|---|---:|---|']
    for s in stopping:
        lines.append(f'| {s["run"]} | {s["nu"]:.2f} | {s["converged"]} | {s["gradient_tolerance_satisfied"]} | {s["final_gradient_inf"]:.3g} | {s["message"]} |')
    lines += ['', '三次均完成各自50点扫描；扫描成功数见summary.csv，完整失败候选保留在各运行目录。每个ν均额外重做一次正演，保存残差在绝对1e-10 mm容差下重现；核验原始CSV哈希、坐标锚点哈希、同区原始数据、载荷、权重及求解设置。函数下降停止与GTOL满足分别报告。', '',
        '位移回代是使用过的数据的拟合检查；较好的拟合或较小的参照差不能证明绝对模量正确。载荷参考未同步核实、厚度/尺度假设、条件材料定位和截面边界等效限制仍然成立。三次来自同一试样，不作为三根独立样本。', '',
        '![同区轴向比较](profiles.png)', '', '![裁剪二维模量](modulus_fields.png)', '',
        '## 复现', '', '四步分别运行项目script的step1至step4入口；第2至4步显式--input指向各01_prepared。冻结配置、输入哈希、全量50点扫描及原始数组保存在下列运行目录：', '']
    lines += [f'- {a["root"]}' for a in audits]
    lines += ['', '后处理命令与环境见environment.json；同区对照数据见summary.csv、regional_comparison.csv、axial_profiles.csv；12组停止证据见solver_stopping.csv；审计见audit.json。已有数值结果与原始数据未覆盖。', '']
    (output / 'report.md').write_text('\n'.join(lines), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', nargs='+', type=Path, required=True)
    parser.add_argument('--baselines', nargs='+', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    analyze(args.cases, args.baselines, args.output)


if __name__ == '__main__':
    main()
