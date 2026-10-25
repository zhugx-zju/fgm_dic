"""380N-only post-processing of axial versus free two-dimensional H1 fields.

Run after the ordinary four steps. --fixed-gamma computes one separately
labelled diagnostic at the original 2D gamma; the default only reads fields.
"""

import argparse
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fgm_asm.axial_inverse import pullback_axial
from fgm_asm.inverse_solver import _evaluate_inverse_state
from fgm_asm.material import MaterialInfo
from fgm_asm.load_case import evaluate_observation
from fgm_asm.research import solve_forward
from .calibration_trend import calibration_trend
from .calibration_trend_error import difference_summary, plot_samples
from .dic_case_workflow import load_prepared, _optimize, _solver_summary, _regional_comparison
from .experiment_data import file_sha256
from .research_reporting import save_run, _json_value


ROOT = Path(__file__).resolve().parents[1]
CASE = ROOT/'results/dic_full/h1_380N_axial_20261025'
BASELINE = ROOT/'results/dic_full/load380_20261022/h1'


def fixed_gamma():
    """One prespecified common-gamma diagnostic, never a calibration selection."""
    cfg, mesh, cases, _, _ = load_prepared(CASE/'01_prepared')
    source = CASE/'03_inverse/nu_0.26'
    baseline = json.loads((BASELINE/'03_inverse/run_metadata.json').read_text(encoding='utf-8'))
    gamma = float(baseline['selected_gamma'])
    target = CASE/'fixed_original_gamma'
    if target.exists():
        raise FileExistsError(target)
    with np.load(source/'fields.npz') as saved:
        initial = saved['E_inverse'].copy()
    result = _optimize(cfg, mesh, cases, gamma, .26, initial)
    save_run(target, cfg, dict(workflow='axial_common_gamma_diagnostic',
        gamma=gamma, nu=.26, uses_calibration=False,
        initialization='selected axial field, without calibration',
        selection_role='diagnostic only; original 2D gamma fixed before comparison',
        solver=_solver_summary(result)),
        dict(E_inverse=result['E_final'], final_gradient=result['final_gradient'],
             cost_history=result['cost_history'], gradient_norms=result['gradient_norms']))
    print(json.dumps(_json_value(_solver_summary(result))), flush=True)


def exact_regional_means(x, profile, boundaries):
    """Integrate piecewise-linear width means with exact interface clipping."""
    limits = np.r_[x[0], boundaries, x[-1]]
    means = []
    for left, right in zip(limits[:-1], limits[1:]):
        points = np.r_[left, x[(x > left) & (x < right)], right]
        means.append(np.trapezoid(np.interp(points, x, profile), points)/(right-left))
    return np.asarray(means)


def clipped_area_means(mesh, field, boundaries):
    """Independent Q4 cell-area integration, clipping cells at each interface."""
    grid = field.reshape(mesh.nods_y, mesh.nods_x)
    x, y = mesh.plot_x[0], mesh.plot_y[:, 0]
    limits = np.r_[x[0], boundaries, x[-1]]
    means = []
    left_ymean = (grid[:-1,:-1]+grid[1:,:-1])/2
    right_ymean = (grid[:-1,1:]+grid[1:,1:])/2
    for left, right in zip(limits[:-1], limits[1:]):
        lo, hi = np.maximum(x[:-1],left), np.minimum(x[1:],right)
        widths = np.maximum(hi-lo,0)
        fraction = ((lo+hi)/2-x[:-1])/np.diff(x)
        values = left_ymean*(1-fraction)+right_ymean*fraction
        weights = np.diff(y)[:,None]*widths[None,:]
        means.append(float(np.sum(weights*values)/(mesh.geo_h*(right-left))))
    return np.asarray(means)


def compare():
    cfg, mesh, cases, meta, _ = load_prepared(CASE/'01_prepared')
    case = cases[0]
    assert len(cases) == 1 and case.name == '380N'
    verification = json.loads((CASE/'implementation_verification.json').read_text(encoding='utf-8'))
    assert verification['passed']
    selected = json.loads((CASE/'03_inverse/run_metadata.json').read_text(encoding='utf-8'))
    old = json.loads((BASELINE/'03_inverse/run_metadata.json').read_text(encoding='utf-8'))
    specifications = [
        ('free_2d', '原自由二维H1', BASELINE/'03_inverse/nu_0.26', old['selected_gamma']),
        ('axial', '轴向E(x) H1（独立选γ）', CASE/'03_inverse/nu_0.26', selected['selected_gamma'])]
    if (CASE/'fixed_original_gamma/fields.npz').exists():
        specifications.append(('axial_fixed_gamma', '轴向E(x)（固定原二维γ）',
                               CASE/'fixed_original_gamma', old['selected_gamma']))
    calibration = pd.DataFrame(meta['calibration'])
    reference = calibration.mean_E_MPa.to_numpy()
    x = mesh.plot_x[0]
    _, _, trend = calibration_trend(calibration, cfg.REGION_BOUNDARIES_X, x)
    rows, regional, profiles, curves, audits = [], [], [], [], []
    protected = {}
    for key, label, path, gamma in specifications:
        for name in ('fields.npz', 'run_metadata.json'):
            protected[str((path/name).relative_to(ROOT))] = file_sha256(path/name)
        with np.load(path/'fields.npz') as saved:
            E = saved['E_inverse'].copy()
        grid = E.reshape(mesh.nods_y, mesh.nods_x)
        mean = np.trapezoid(grid, x=mesh.plot_y[:, 0], axis=0)/mesh.geo_h
        means = exact_regional_means(x, mean, cfg.REGION_BOUNDARIES_X)
        np.testing.assert_allclose(means, clipped_area_means(mesh,E,cfg.REGION_BOUNDARIES_X),rtol=1e-12,atol=1e-9)
        difference = 100*(means/reference-1)
        curve = dict(label=label, x=x, mean=mean)
        fine = difference_summary(curve, trend, order=16)
        coarse = difference_summary(curve, trend, order=8)
        assert max(abs(fine[k]-coarse[k]) for k in fine) < 1e-7
        px, pe, delta = plot_samples(curve, trend)
        u, _ = solve_forward(mesh, case.bc_info['tensile_end_force'], cfg.THICKNESS, .26, E)
        state = evaluate_observation(case, u)
        rmse = np.sqrt(np.mean(state['residual'].reshape(-1, 2)**2, axis=0))*1000
        fresh = _evaluate_inverse_state(mesh, case.bc_info, None, MaterialInfo(nu=.26),
                                         gamma, E, 1, observation_cases=cases)
        gradient = fresh['grad_m'] if key == 'free_2d' else pullback_axial(mesh, fresh['grad_m'])
        info = json.loads((path/'run_metadata.json').read_text(encoding='utf-8'))
        solver = info['solver']
        np.testing.assert_allclose(fresh['objective'], solver['final_cost'], rtol=2e-9, atol=1e-11)
        with np.load(path/'fields.npz') as saved:
            np.testing.assert_allclose(gradient, saved['final_gradient'], rtol=1e-4, atol=2e-10)
        if key != 'free_2d':
            assert np.ptp(grid, axis=0).max() == 0
        audits.append(dict(method=key, objective_recomputed=True,
            gradient_recomputed=True, width_tying_exact=key != 'free_2d'))
        row = dict(method=key, label=label, parameters=mesh.n_nod if key == 'free_2d' else mesh.nods_x,
            gamma=gamma, regional_mean_absolute_difference_percent=float(np.mean(abs(difference))),
            regional_max_absolute_difference_percent=float(np.max(abs(difference))),
            trend_max_absolute_difference_percent=float(np.max(abs(delta))),
            ux_rmse_um=float(rmse[0]), uy_rmse_um=float(rmse[1]),
            solver_converged=solver['converged'], stop_message=solver['message'],
            gradient_inf=float(np.max(abs(gradient))), gradient_tolerance_satisfied=bool(np.max(abs(gradient)) <= cfg.GTOL),
            **fine)
        rows.append(row)
        regional.extend(dict(method=key, infill_percent=int(record.infill_percent),
            calibration_E_MPa=float(record.mean_E_MPa), calibration_sd_MPa=float(record.sd_E_MPa),
            inverse_mean_E_MPa=float(value), difference_percent=float(error))
            for record, value, error in zip(calibration.itertuples(), means, difference))
        profiles.extend(dict(method=key, x_mm=float(a), inverse_mean_E_MPa=float(b),
            relative_trend_difference_percent=float(c)) for a, b, c in zip(px, pe, delta))
        curves.append((key, label, mean, px, delta))
    # Historical CSV uses native Gauss-point grouping rather than cut cells.
    # Reproduce that statistic separately; never force it to equal exact cuts.
    original_regions = pd.read_csv(BASELINE/'04_comparison/regional_comparison.csv')
    old_errors = original_regions[np.isclose(original_regions.nu,.26)].difference_percent.to_numpy()
    with np.load(BASELINE/'03_inverse/nu_0.26/fields.npz') as saved:
        native = _regional_comparison(mesh,saved['E_inverse'],calibration,cfg.REGION_BOUNDARIES_X)
    np.testing.assert_allclose(native.difference_percent,old_errors,atol=1e-8,rtol=1e-9)
    output = CASE/'comparison'
    output.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(output/'summary.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(regional).to_csv(output/'regional_comparison.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(profiles).to_csv(output/'relative_trend_profiles.csv', index=False)
    plt.rcParams.update({'font.family':'Microsoft YaHei', 'font.size':10,
                         'axes.unicode_minus':False, 'svg.fonttype':'none'})
    colors, styles = ['#0072B2', '#D55E00', '#009E73'], ['-', '--', ':']
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout='constrained')
    edges = np.r_[0., cfg.REGION_BOUNDARIES_X, cfg.GEO_L]
    axes[0,0].stairs(reference, edges, color='#555555', linewidth=1.2, label='独立分层标定')
    for j, record in enumerate(calibration.itertuples()):
        axes[0,0].fill_between(edges[j:j+2], record.mean_E_MPa-record.sd_E_MPa,
                              record.mean_E_MPa+record.sd_E_MPa, color='#777777', alpha=.13)
    tx = curves[0][3]
    from .calibration_trend_error import trend_values
    axes[0,0].plot(tx, trend_values(tx, trend), color='#7B3294', linestyle='-.', label='辅助线性趋势')
    width = .24
    for j, (key,label,mean,px,delta) in enumerate(curves):
        axes[0,0].plot(x, mean, color=colors[j], linestyle=styles[j], label=label)
        axes[0,1].plot(px, delta, color=colors[j], linestyle=styles[j], label=label)
        errors = [r['difference_percent'] for r in regional if r['method']==key]
        axes[1,0].bar(np.arange(7)+(j-(len(curves)-1)/2)*width, errors, width, color=colors[j], label=label)
        axes[1,1].bar(np.arange(2)+(j-(len(curves)-1)/2)*width,
                       [rows[j]['ux_rmse_um'],rows[j]['uy_rmse_um']], width, color=colors[j], label=label)
    axes[0,0].set(title='(a) 宽度平均模量', xlabel='FEM x (mm)', ylabel='E (MPa)', xlim=(0,mesh.geo_l))
    axes[0,0].legend(fontsize=8)
    axes[0,1].set(title='(b) 相对辅助线性趋势', xlabel='FEM x (mm)', ylabel='有符号相对差异 (%)', xlim=trend['trend_support_x_mm'])
    axes[0,1].axhline(0, color='black', linewidth=.6)
    for y in (-10,10):
        axes[0,1].axhline(y, color='#777777', linewidth=.6, linestyle=':')
    axes[1,0].set(title='(c) 区域均值相对独立标定', xlabel='名义填充率 (%)', ylabel='有符号相对差异 (%)', xticks=np.arange(7), xticklabels=calibration.infill_percent.astype(int))
    axes[1,0].axhline(0, color='black', linewidth=.6)
    axes[1,1].set(title='(d) 同平台位移拟合', ylabel='RMSE (μm)', xticks=np.arange(2), xticklabels=['轴向 ux','横向 uy'])
    fig.suptitle('380N：自由二维模量与轴向E(x)约束对照；121×19单元，ν=0.26，等权重', fontsize=13)
    for suffix in ('png','svg'):
        fig.savefig(output/f'comparison.{suffix}', dpi=300)
    plt.close(fig)
    scan = pd.read_csv(CASE/'03_inverse/selection.csv')
    result = dict(status=selected['status'], fit_only='380N', force_N=319.9,
        force_is_assumed=True, nu=.26, mesh=[121,19], observations=len(case.observation.x_meas),
        selected_axial_gamma=selected['selected_gamma'], scan_points=len(scan),
        solver_successful_scan_points=int(scan.converged.sum()),
        gtol_scan_points=int(scan.gradient_tolerance_satisfied.sum()),
        scan_FE_evaluations=int(scan.evaluations.sum()),
        comparisons=rows, independent_recomputation=audits,
        exact_region_integration_verified_by_independent_clipped_Q4_cells=True,
        historical_region_CSV_native_Gauss_grouping_reproduced=True,
        input_arrays_bitwise_equal=True, source_sha256=protected,
        regularization='gamma/2 * H * integral(E_x^2 dx); H not dropped',
        evaluation='exact physical region clipping; fixed existing trend support; calibration post hoc only',
        scope='E(x) prior imposed; 2D spatial recovery not tested; original force/coordinate/boundary/calibration transfer limitations retained')
    (output/'summary.json').write_text(json.dumps(_json_value(result),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    display = pd.DataFrame(rows)[['label','parameters','gamma','regional_mean_absolute_difference_percent',
        'regional_max_absolute_difference_percent','mean_absolute_difference_percent',
        'trend_max_absolute_difference_percent','ux_rmse_um','uy_rmse_um']]
    display.columns=['模型','参数数','γ','七区平均绝对差异(%)','最差区域差异(%)','趋势平均绝对差异(%)','趋势最大差异(%)','轴向RMSE(μm)','横向RMSE(μm)']
    caption = ('独立选γ是主要结果；固定原二维γ仅用于分离参数约束与选参变化。'
        '区域均值均按同一物理界面精确积分，趋势差异限定在原标定支撑内。'
        '原分区CSV按原生高斯点归属统计，与本页精确切分统计有小量差异。'
        '趋势线不是连续模量真值；位移回代是同数据拟合。'
        '载荷319.9 N仍为假设净力，物理配准及内部侧边界近似等限制不变。')
    body = ('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>380N轴向模量约束对照</title>'
        '<style>body{max-width:1400px;margin:30px auto;padding:20px;font:16px/1.7 "Microsoft YaHei",sans-serif;color:#17212b}img{width:100%}table{border-collapse:collapse;font-size:13px}td,th{padding:8px;border:1px solid #ddd}a{color:#075fa2}</style>'
        '<h1>380N：原自由二维H1与轴向E(x)约束</h1>'
        '<p><a href="../结果总览.html">本组四步结果</a> · <a href="../../load380_20261022/h1/结果总览.html">原二维结果</a> · <a href="../03_inverse/云图.html">L-curve与停止状态</a></p>'
        '<p>'+html.escape(caption)+'</p>'+display.to_html(index=False,border=0,float_format=lambda v:f'{v:.7g}')+
        '<img src="comparison.png" alt="模量、相对误差及位移拟合对照">'
        '<p><a href="summary.csv">汇总CSV</a> · <a href="regional_comparison.csv">逐区CSV</a> · <a href="relative_trend_profiles.csv">趋势误差曲线CSV</a> · <a href="comparison.svg">矢量图</a> · <a href="summary.json">核验与元数据</a></p></html>')
    (output/'index.html').write_text(body,encoding='utf-8')
    assert all(file_sha256(ROOT/path)==digest for path,digest in protected.items())
    print(json.dumps(_json_value(result),ensure_ascii=False,indent=2),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixed-gamma',action='store_true')
    args=parser.parse_args()
    fixed_gamma() if args.fixed_gamma else compare()
