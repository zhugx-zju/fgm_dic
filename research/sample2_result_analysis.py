"""Compare saved four-step gradient-specimen runs without tuning or re-identifying.

python -B -m lib.sample2_result_analysis --cases ROOT1 ROOT2 ROOT3 --audit INPUT_JSON --output NEW_REPORT_DIR
The output may already contain this run's input audit/logs; result files cannot exist.
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
from lib.dic_case_workflow import load_prepared
from lib.dic_preprocessing import relative_displacement_errors
from lib.experiment_data import file_sha256
from .uniform_sample_analysis import area_statistics


def plot_displacement_errors(table, output):
    """Redraw percent errors, also accepting historical percent-column names."""
    plt.rcParams.update({'font.family': 'Microsoft YaHei', 'font.size': 10,
                         'axes.unicode_minus': False, 'svg.fonttype': 'path'})
    table = table.copy()
    for component in ('axial', 'transverse'):
        key = f'{component}_relative_error_percent'
        if key not in table:
            table[key] = table[f'{component}_relative_RMSE_percent']
    main = table[np.isclose(table.nu, table.primary_nu)]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)
    for ax, key, title in zip(axes, ('axial_relative_error_percent', 'transverse_relative_error_percent'),
                              ('轴向位移相对误差', '横向位移相对误差')):
        for offset, model, label, color in ((-.18, 'forward', '标定正演', '#D55E00'), (.18, 'inverse', '反演回代', '#0072B2')):
            sub = main[main.model == model].sort_values('run')
            ax.bar(sub.run+offset, sub[key], width=.33, label=label, color=color)
        ax.set(xticks=sorted(main.run.unique()), xlabel='加载次数', ylabel='相对误差 (%)', title=title)
    axes[0].legend(frameon=False, fontsize=9)
    save_figure(fig, Path(output)/'displacement_errors')


def analyze(roots: list[Path], audit_path: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    if any((output/name).exists() for name in ('report.md', 'summary.csv', 'modulus_fields.png')):
        raise FileExistsError('Choose a new report directory; established summaries are protected.')
    audit = json.loads(audit_path.read_text(encoding='utf-8'))
    rows, regional, profiles, maps, provenance = [], [], [], [], []
    for root in roots:
        cfg, mesh, cases, prepared, _ = load_prepared(root/'01_prepared')
        assert len(cases) == 1 and not cfg.PREDICT_CASES
        case = cases[0]
        run = cfg.PLATFORMS[case.name]['run']
        inv_path = root/'03_inverse/run_metadata.json'
        meta = json.loads(inv_path.read_text(encoding='utf-8'))
        if meta.get('selected_gamma') is None:
            raise ValueError(f'{root}: no selected L-curve corner; cannot report a selected inverse field.')
        assert (root/'04_comparison/run_metadata.json').exists()
        comparisons = pd.read_csv(root/'04_comparison/comparison.csv')
        reg = pd.read_csv(root/'04_comparison/regional_comparison.csv')
        reg.insert(0, 'run', run)
        regional.append(reg)
        scan = pd.read_csv(root/'03_inverse/selection.csv')
        point_count = len(case.observation.u_measured)//2
        baseline_fields = root/f'02_forward/nu_{cfg.PRIMARY_NU:.2f}/fields.npz'
        inverse_fields = root/f'03_inverse/nu_{cfg.PRIMARY_NU:.2f}/fields.npz'
        with np.load(inverse_fields) as saved:
            field = saved['E_inverse'].copy()
            recorded_residual = saved[f'{case.name}_residual'].copy()
        # One independent fresh forward replay checks saved predictions and metrics.
        displacement, _ = solve_forward(mesh, case.bc_info['tensile_end_force'], cfg.THICKNESS, cfg.PRIMARY_NU, field)
        state = evaluate_observation(case, displacement)
        np.testing.assert_allclose(state['residual'], recorded_residual, rtol=1e-8, atol=1e-10)
        recorded_rms = comparisons.loc[(comparisons.model == 'inverse') &
                                      np.isclose(comparisons.nu, cfg.PRIMARY_NU), 'rms_mm'].iloc[0]
        np.testing.assert_allclose(state['weighted_rms'], recorded_rms, rtol=1e-8, atol=1e-12)
        stats = area_statistics(mesh, field)
        main_regions = reg.loc[np.isclose(reg.nu, cfg.PRIMARY_NU)]
        region_difference = float(main_regions.difference_percent.abs().mean())
        for item in comparisons.to_dict(orient='records'):
            solver = next((s for s in meta.get('solvers', []) if np.isclose(s['nu'], item['nu'])), {})
            model_fields = root/f"{'03_inverse' if item['model'] == 'inverse' else '02_forward'}/nu_{item['nu']:.2f}/fields.npz"
            with np.load(model_fields) as values:
                errors = relative_displacement_errors(case, values[f'{case.name}_prediction'])
            row = dict(run=run, model=item['model'], nu=item['nu'], primary_nu=cfg.PRIMARY_NU,
                       gamma=meta['selected_gamma'] if item['model'] == 'inverse' else np.nan,
                       **errors,
                       length_mm=cfg.GEO_L, width_mm=cfg.GEO_H, thickness_mm=cfg.THICKNESS,
                       force_increment_N=case.metadata['measured_force_N'], observation_points=point_count,
                       temporal_SD_axial_mm=audit[str(run)]['temporal_SD_rms_raw_DX_DY_mm'][1],
                       temporal_SD_transverse_mm=audit[str(run)]['temporal_SD_rms_raw_DX_DY_mm'][0],
                       l_curve_status=meta['l_curve_status'], scan_converged_count=int(scan.converged.sum()),
                       scan_count=len(scan), inverse_status=meta['status'])
            if item['model'] == 'inverse':
                row.update(solver_converged=solver.get('converged'), stop_message=solver.get('message'),
                           final_gradient_inf=solver.get('final_gradient_inf'),
                           gradient_tolerance_satisfied=solver.get('gradient_tolerance_satisfied'))
                if np.isclose(item['nu'], cfg.PRIMARY_NU):
                    row.update(stats, regional_mean_absolute_relative_error_percent=region_difference)
            rows.append(row)
        # Trap geometry only locates reporting regions; each y row retains its own E.
        array = field.reshape(mesh.nods_y, mesh.nods_x)
        weights = np.ones(mesh.nods_y)
        weights[[0, -1]] = .5
        mean = np.average(array, axis=0, weights=weights)
        sd = np.sqrt(np.average((array-mean)**2, axis=0, weights=weights))
        xi = mesh.plot_x[0]-cfg.GEO_L/2
        for x, value, spread in zip(xi, mean, sd):
            profiles.append(dict(run=run, manufacturing_x_mm=x, width_mean_E_MPa=value, transverse_spatial_sd_MPa=spread))
        maps.append(dict(run=run, mesh=mesh, field=array, xi=xi, mean=mean, sd=sd,
                         calibration=pd.DataFrame(prepared['calibration']), boundaries=np.asarray(cfg.REGION_BOUNDARIES_MANUFACTURING_X),
                         gamma=meta['selected_gamma']))
        # Confirm frozen input CSV hashes and the independent machine/point records.
        for source in prepared['platforms'][case.name]['sources']:
            assert file_sha256(Path(source['path'])) == source['sha256']
        source_folder = Path(cfg.PLATFORMS[case.name]['folder']).parent
        assert file_sha256(source_folder/'force.xlsb') == audit[str(run)]['force_csv_sha256']
        assert file_sha256(source_folder/'disp.csv') == audit[str(run)]['disp_csv_sha256']
        provenance.append(dict(run=run, root=str(root.resolve()),
                               prepared_metadata_sha256=file_sha256(root/'01_prepared/run_metadata.json'),
                               inverse_metadata_sha256=file_sha256(inv_path),
                               comparison_metadata_sha256=file_sha256(root/'04_comparison/run_metadata.json'),
                               inverse_fields_sha256=file_sha256(inverse_fields),
                               forward_fields_sha256=file_sha256(baseline_fields),
                               saved_inverse_replay_verified=True, all_input_hashes_verified=True,
                               inverse_reuse=meta.get('inverse_reuse'),
                               material_registration=prepared['material_registration']))
    table = pd.DataFrame(rows)
    regions = pd.concat(regional, ignore_index=True)
    table.to_csv(output/'summary.csv', index=False, lineterminator='\n')
    regions.to_csv(output/'regional_comparison.csv', index=False, lineterminator='\n')
    pd.DataFrame(profiles).to_csv(output/'axial_profiles.csv', index=False, lineterminator='\n')
    (output/'provenance.json').write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding='utf-8')
    plt.rcParams.update({'font.family': 'Microsoft YaHei', 'font.size': 10, 'axes.unicode_minus': False,
                         'svg.fonttype': 'path', 'axes.spines.top': False, 'axes.spines.right': False})
    low = min(m['field'].min() for m in maps)
    high = max(m['field'].max() for m in maps)
    fig, axes = plt.subplots(len(maps), 1, figsize=(10, 6.8), constrained_layout=True)
    for ax, m in zip(np.atleast_1d(axes), maps):
        mesh = m['mesh']
        image = ax.pcolormesh(mesh.plot_x-mesh.geo_l/2, mesh.plot_y, m['field'],
                              shading='auto', cmap='viridis', vmin=low, vmax=high)
        for boundary in m['boundaries']:
            ax.axvline(boundary, color='white', lw=.55, ls='--', alpha=.8)
        ax.set(xlabel='制造坐标 ξ (mm)', ylabel='横向 y (mm)',
               title=f'第 {m["run"]} 次：自由二维 H1，γ={m["gamma"]:.3g}', aspect='equal')
    fig.colorbar(image, ax=axes, label='反演 E (MPa)', shrink=.85)
    save_figure(fig, output/'modulus_fields')
    fig, axes = plt.subplots(len(maps), 1, figsize=(10, 8), sharex=True, sharey=True, constrained_layout=True)
    for ax, m in zip(np.atleast_1d(axes), maps):
        xi, mean, spread = m['xi'], m['mean'], m['sd']
        cal = m['calibration']
        ax.fill_between(xi, mean-spread, mean+spread, color='#0072B2', alpha=.15, label='横向空间 SD')
        ax.plot(xi, mean, color='#0072B2', lw=1.6, label='反演宽度均值')
        edges = np.r_[xi[0], m['boundaries'], xi[-1]]
        step_x = np.repeat(edges, 2)[1:-1]
        step_mean = np.repeat(cal.mean_E_MPa.to_numpy(), 2)
        step_sd = np.repeat(cal.sd_E_MPa.to_numpy(), 2)
        ax.plot(step_x, step_mean, color='#D55E00', lw=1.4, label='1023 单调约束标定均值')
        ax.fill_between(step_x, step_mean-step_sd, step_mean+step_sd, color='#D55E00', alpha=.13, label='标定组内 SD')
        for boundary in m['boundaries']:
            ax.axvline(boundary, color='.75', lw=.6, ls='--')
        ax.set(title=f'第 {m["run"]} 次', ylabel='E (MPa)', xlim=(-57, 57))
    axes[0].legend(ncol=4, fontsize=9, frameon=False, loc='upper left')
    axes[-1].set_xlabel('制造坐标 ξ (mm)；两标线名义位置 ±57 mm，DIC 保持实际跨度')
    save_figure(fig, output/'axial_profiles')
    plot_displacement_errors(table, output)
    lines = ['# 1026 sample_2：三次独立参考图的四步自由二维 H1 反演', '',
             '**结论：** 使用随填充率不减的标定分布重新正演及比较。三次反演得到沿高填充率端增大的模量趋势，但绝对模量与约束标定参照仍有差异，尤其100%区偏低。第2、3次宽度均值曲线较接近，第1次整体偏低。', '',
             '标定来源为 individual_3pct_monotone_20261026_final/group_summary.csv，fit_method=individual_window_joint_3pct_monotone。40%至100%均值依次为985.32/1119.96/1276.05/1472.30/1623.17/2055.18/2566.71 MPa。每组4根，保留各组SD；组内3%及随填充率不减是标定选窗的显式约束，未施加到空间反演。', '',
             '此次仅更换外部标定参照：重新冻结输入、重算标定正演、更新区域相对误差与图件。已核验原反演的全部准备数组、观测算子、权重、载荷、目标和停止设置一致，复用原模量场、位移、γ及扫描记录，未重新优化；12组ν模量/位移数组逐元素完全一致。原始优化试验元数据保留在复制的l_curve目录，来源与哈希见 inverse_reuse_checks.json 和各次03_inverse/run_metadata.json。', '',
             '厚度约 3.2 mm；长宽与承载宽度使用各次 DIC 参考跨度，原 Y 为轴向 x。用户明确将几何/载荷口径及力参考对应视为假设。两标线距 220 mm 试样两端各 53 mm，制造坐标为 ±57 mm；按中点定位，不缩放 DIC 实际跨度。七区填充率沿原 +Y 由 40% 增至 100%，界面为 −40/−24/−8/8/24/40 mm。', '',
             '每次单独冻结输入、单独标定正演、单独 L-curve 选参及反演，PREDICT_CASES=()。121×19 单元、2440 个独立节点模量，保留独立 y 行。H1 目标为 0.5*r.T@W_area@r+0.5*γ*∫|∇E|²；γ 为 1e-12–1e-4 共 50 点，初值和选参均只用本次 DIC。七区标定及其批次、n、SD 只作外部参照，不进入初值、目标或选参。主 ν=0.26；0.24/0.28/0.33 为同 γ 敏感性比较。', '',
             '| 次数 | 长×宽 mm | 起载 N | 假定末载 N | 增量 N | 标线间距−DIC跨度 mm |',
             '|---|---:|---:|---:|---:|---:|']
    for m in maps:
        r = main[(main.run == m['run']) & (main.model == 'inverse')].iloc[0]
        a = audit[str(m['run'])]
        lines.append(f'| {m["run"]} | {r.length_mm:.3f}×{r.width_mm:.3f} | {a["force_start_N"]:.5f} | {a["setting_N"]} | {r.force_increment_N:.5f} | {114-r.length_mm:.3f} |')
    lines += ['', '**参考限制：** 三次窗口均无同期力记录。末载假定为 280/380/480 N，减去各自记录首载；假定该首载对应 DIC 第0帧。这些不是实测配对的窗口均值。实际DIC跨度与名义114 mm不一致，采取中点对齐而不缩放，因此分区位置是条件定位。模型采用平面应力、各向同性、端部均匀牵引和侧面自由近似；实际边界等效与尺度仍须物理核验。', '',
              '| 次数 | 模型 | γ | 轴向位移相对误差 % | 横向位移相对误差 % | 双分量位移相对误差 % | 七区模量平均绝对相对误差 % |',
              '|---|---|---:|---:|---:|---:|---:|']
    for r in main.sort_values(['run', 'model']).itertuples():
        inv = r.model == 'inverse'
        lines.append(f'| {r.run} | {"反演回代" if inv else "单调约束标定正演"} | {f"{r.gamma:.3g}" if inv else "—"} | {r.axial_relative_error_percent:.3f} | {r.transverse_relative_error_percent:.3f} | {r.displacement_relative_error_percent:.3f} | {f"{r.regional_mean_absolute_relative_error_percent:.3f}" if inv else "—"} |')
    lines += ['', '位移采用面积加权L₂相对误差：ε_c=100×√Σ[w_i(u_model,c−u_DIC,c)²]/√Σ[w_i(u_DIC,c的变形分量)²]。分母由DIC去除两平移和一微小转动后的变形量确定；双分量形式对两个分量一同求和。所有模型共用同一DIC分母，不逐点除以过零位移。该范数比与此前分量百分比在数学上等价，因此反演位移百分比不因参照表改变而变化；本报告统一采用相对误差名称及百分数。', '',
              '区域模量有符号相对误差为100×(E_inverse_mean−E_calibration_mean)/E_calibration_mean；汇总为七区绝对相对误差的算术均值。分母使用单调约束标定参照，不是已知材料真值。完整逐区数据、n、标定SD和反演空间SD见 regional_comparison.csv。横向空间SD与标定组内SD不是参数置信区间。', '',
              '## 求解与核验', '', '| 次数 | ν | 停止 | GTOL满足 | 梯度∞范数 | 扫描成功数/50 |', '|---|---:|---|---|---:|---:|']
    for r in table[table.model == 'inverse'].sort_values(['run', 'nu']).itertuples():
        lines.append(f'| {r.run} | {r.nu:.2f} | {r.stop_message} | {r.gradient_tolerance_satisfied} | {r.final_gradient_inf:.3g} | {r.scan_converged_count}/{r.scan_count} |')
    lines += ['', '函数下降停止不等于 GTOL 满足。本次三次均由原50点扫描选出有效拐点，未使用 --resume、未增加1500/12000预算、未按标定或期望梯度形状补选 γ。未通过停止条件的候选被排除，其原始场和诊断保留在各03_inverse/l_curve目录。', '',
              '第4步已用同一冻结输入核对正演、反演与回代记录；此后另对主ν保存模量正演，逐测点残差在1e-10 mm绝对容差下复核。全部45份全场、三份加载和三份单点CSV哈希复核一致。仅更新参照及误差表达，未改力学、观测或梯度实现。', '',
              '三个结果来自同一试样不同加载/参考图，不能当作三根独立材料样本。回代使用过的位移属于拟合检查，无独立留出物理验证。较好位移拟合不独自证明模量正确；空间差异可能吸收载荷参考、厚度、材料定位、边界、各向异性或DIC系统误差。', '',
              '![二维模量](modulus_fields.png)', '', '![轴向统计与独立标定](axial_profiles.png)', '', '![位移误差](displacement_errors.png)', '',
              '## 输出与复现', '', '三次四步结果：']
    lines += [f'- {root.resolve()}' for root in roots]
    lines += ['', '各次01_prepared保存 configuration_snapshot.py 与输入哈希；各步骤分别运行项目四个入口。已有结果拒绝覆盖，其他步骤使用 --input 显式指定冻结01_prepared，不依赖当前根配置已切换到第3次。原始数值结果及50点扫描保存在results/dic_full；本目录仅保存可审阅摘要、图件、输入审计与执行日志。', '',
              '后处理复现：`python -B -m lib.sample2_result_analysis --cases <三个运行目录> --audit <input_audit.json> --output <新报告目录>`。执行环境、Git版本、命令与源文件哈希保存在四步元数据；本次后处理源文件见项目lib/sample2_result_analysis.py。', '']
    (output/'report.md').write_text('\n'.join(lines), encoding='utf-8')
    print(main[['run', 'model', 'gamma', 'axial_relative_error_percent', 'transverse_relative_error_percent',
                'regional_mean_absolute_relative_error_percent']].to_string(index=False))


def save_figure(fig, stem):
    for ext in ('png', 'svg'):
        fig.savefig(stem.with_suffix('.'+ext), dpi=300, bbox_inches='tight')
    svg = stem.with_suffix('.svg')
    svg.write_bytes(('\n'.join(line.rstrip() for line in svg.read_text(encoding='utf-8').splitlines())+'\n').encode('utf-8'))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', type=Path, nargs='+', required=True)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    analyze(args.cases, args.audit, args.output)


if __name__ == '__main__':
    main()
