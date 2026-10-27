"""Post-hoc calibration curve differences for saved full-domain sample_2 fields.

No optimization, gamma reselection, or modification of the four-step results.

Historical ``trend`` means piecewise-linear layer-center connections. It is
retained for reproducibility, not the user's corrected smooth E(infill)
reference; use lib.smooth_calibration_reference for that definition.
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
from numpy.polynomial.legendre import leggauss

from .dic_case_workflow import load_prepared
from .dic_gamma_stability import GAMMAS
from .experiment_data import file_sha256


CENTERS = np.arange(-48., 49., 16.)
INTERFACES = np.arange(-40., 41., 16.)


def curve_errors(x, field, values, kind, support, order=12):
    """Length-averaged absolute relative difference; split all knots/zeroes.

    field is the exact transverse mean of a continuous Q4 field. References
    are positive piecewise linear (trend) or piecewise constant (layered).
    Splitting sign changes removes the kink in the absolute-value integrand.
    """
    lo, hi = support
    if lo < x[0]-1e-10 or hi > x[-1]+1e-10:
        raise ValueError('Support outside the saved inversion domain')
    breaks = CENTERS if kind == 'trend' else INTERFACES
    edges = np.unique(np.r_[lo, hi, x[(x > lo) & (x < hi)], breaks[(breaks > lo) & (breaks < hi)]])
    nodes, weights = leggauss(order)
    absolute, square, reference_square = 0., 0., 0.
    for left, right in zip(edges[:-1], edges[1:]):
        predicted = np.interp([left, right], x, field)
        if kind == 'trend':
            reference = np.interp([left, right], CENTERS, values)
        else:
            reference = np.repeat(values[np.searchsorted(INTERFACES, (left+right)/2)], 2)
        difference = predicted-reference
        fractions = [0., 1.]
        if difference[0]*difference[1] < 0:
            fractions.insert(1, float(-difference[0]/(difference[1]-difference[0])))
        for begin, end in zip(fractions[:-1], fractions[1:]):
            t = (begin+end)/2+(end-begin)/2*nodes
            measure = weights*(right-left)*(end-begin)/2
            ref = reference[0]+t*(reference[1]-reference[0])
            delta = difference[0]+t*(difference[1]-difference[0])
            absolute += np.sum(measure*abs(delta/ref))
            square += np.sum(measure*delta**2)
            reference_square += np.sum(measure*ref**2)
    return dict(mean_absolute_relative_error_percent=float(100*absolute/(hi-lo)),
                l2_relative_error_percent=float(100*np.sqrt(square/reference_square)))


def regional_errors(x, field, values):
    edges = np.r_[x[0], INTERFACES, x[-1]]
    means = []
    for left, right in zip(edges[:-1], edges[1:]):
        points = np.r_[left, x[(x > left) & (x < right)], right]
        means.append(np.trapezoid(np.interp(points, x, field), points)/(right-left))
    errors = 100*(np.asarray(means)/values-1)
    return means, errors


def width_profile(mesh, field):
    weights = np.ones(mesh.nods_y)
    weights[[0, -1]] = .5
    return np.average(field.reshape(mesh.nods_y, mesh.nods_x), axis=0, weights=weights)


def metrics(x, profile, values):
    row = {}
    for kind, support, prefix in [('trend', (-48., 48.), 'trend_common'),
                                  ('layered', (-48., 48.), 'layered_common'),
                                  ('layered', (x[0], x[-1]), 'layered_full')]:
        fine = curve_errors(x, profile, values, kind, support, order=12)
        check = curve_errors(x, profile, values, kind, support, order=6)
        np.testing.assert_allclose(list(fine.values()), list(check.values()), rtol=1e-10, atol=1e-10)
        row.update({f'{prefix}_{key}': value for key, value in fine.items()})
    _, errors = regional_errors(x, profile, values)
    row['seven_region_mean_absolute_relative_error_percent'] = float(np.mean(abs(errors)))
    return row


def analyze(roots, study, output):
    output.mkdir(parents=True, exist_ok=False)
    # Analytic constant-field checks and exact zero error of a matching trend.
    x_test = np.array([-57., 0., 57.])
    for kind in ('trend', 'layered'):
        result = curve_errors(x_test, np.full(3, 110.), np.full(7, 100.), kind, (-48., 48.))
        np.testing.assert_allclose(list(result.values()), [10., 10.], rtol=1e-12)
    np.testing.assert_allclose(list(curve_errors(CENTERS, CENTERS+100., CENTERS+100., 'trend', (-48., 48.)).values()), 0.)
    rows, candidate_rows, profiles, regions, sources = [], [], [], [], []
    for root in roots:
        cfg, mesh, cases, meta, _ = load_prepared(root/'01_prepared')
        for name in ('run_metadata.json', 'fields.npz'):
            source = root/'01_prepared'/name
            sources.append(dict(path=str(source.resolve()), sha256=file_sha256(source)))
        run = cfg.PLATFORMS[cases[0].name]['run']
        np.testing.assert_array_equal(cfg.REGION_BOUNDARIES_MANUFACTURING_X, INTERFACES)
        values = pd.DataFrame(meta['calibration']).mean_E_MPa.to_numpy()
        x = mesh.plot_x[0]-cfg.GEO_L/2
        scan = pd.read_csv(root/'03_inverse/selection.csv')
        for item in scan.itertuples():
            path = root/'03_inverse/l_curve'/f'gamma_{item.gamma:g}'/'fields.npz'
            with np.load(path) as saved:
                profile = width_profile(mesh, saved['E_inverse'])
            row = dict(run=run, gamma=item.gamma, original_selected=bool(item.selected),
                       solver_stopping_passed=bool(item.converged), **metrics(x, profile, values))
            rows.append(row)
            sources.append(dict(path=str(path.resolve()), sha256=file_sha256(path)))
        for index, gamma in enumerate(GAMMAS):
            path = study/f'run{run}/g{index}_full_uniform/fields.npz'
            with np.load(path) as saved:
                profile = width_profile(mesh, saved['E_inverse'])
            row = dict(run=run, gamma=gamma, **metrics(x, profile, values))
            candidate_rows.append(row)
            means, signed = regional_errors(x, profile, values)
            for infill, ref, mean, error in zip(range(40, 101, 10), values, means, signed):
                regions.append(dict(run=run, gamma=gamma, infill_percent=infill, calibration_E_MPa=ref,
                                    inverse_region_mean_E_MPa=mean, signed_relative_error_percent=error))
            display_x = np.unique(np.r_[x, CENTERS, INTERFACES])
            display_x = display_x[(display_x >= x[0]) & (display_x <= x[-1])]
            for point in display_x:
                profiles.append(dict(run=run, gamma=gamma, x_mm=point, inverse_width_mean_E_MPa=np.interp(point, x, profile),
                                     calibration_trend_E_MPa=np.interp(point, CENTERS, values) if -48 <= point <= 48 else np.nan,
                                     layered_calibration_E_MPa=values[np.searchsorted(INTERFACES, point)]))
            sources.append(dict(path=str(path.resolve()), sha256=file_sha256(path)))
    all_scan, candidates, profiles = pd.DataFrame(rows), pd.DataFrame(candidate_rows), pd.DataFrame(profiles)
    all_scan.to_csv(output/'all_scan_curve_metrics.csv', index=False, lineterminator='\n')
    candidates.to_csv(output/'candidate_curve_metrics.csv', index=False, lineterminator='\n')
    pd.DataFrame(regions).to_csv(output/'regional_metrics.csv', index=False, lineterminator='\n')
    profiles.to_csv(output/'candidate_profiles.csv', index=False, lineterminator='\n')
    plt.rcParams.update({'font.family': 'Microsoft YaHei', 'font.size': 10, 'svg.fonttype': 'path',
                         'axes.unicode_minus': False, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True, sharey=True, constrained_layout=True)
    for ax, run in zip(axes, (1, 2, 3)):
        for index, color in enumerate(('#0072B2', '#009E73', '#CC79A7')):
            data = profiles[(profiles.run == run) & (profiles.gamma == GAMMAS[index])]
            ax.plot(data.x_mm, data.inverse_width_mean_E_MPa, color=color, lw=1.5, label=f'γ={GAMMAS[index]:.3g}')
        data = data.sort_values('x_mm')
        ax.plot(data.x_mm, data.calibration_trend_E_MPa, color='black', ls='--', lw=1.2, label='层中心标定趋势线')
        edges = np.r_[data.x_mm.min(), INTERFACES, data.x_mm.max()]
        calibration = data.groupby('layered_calibration_E_MPa').first().index.to_numpy()
        ax.plot(np.repeat(edges, 2)[1:-1], np.repeat(calibration, 2), color='#D55E00', lw=1.1, label='七层标定阶梯线')
        ax.axvline(-48, color='.65', ls=':', lw=.8)
        ax.axvline(48, color='.65', ls=':', lw=.8)
        ax.set(title=f'第{run}次：完整ROI；共同趋势比较支撑 −48至48 mm', ylabel='E (MPa)')
    axes[0].legend(frameon=False, ncol=3, fontsize=8)
    axes[-1].set_xlabel('制造坐标 ξ (mm)')
    for suffix in ('.png', '.svg'):
        fig.savefig(output/f'calibration_profiles{suffix}', dpi=300)
    plt.close(fig)
    lines = ['# 小γ结果与标定趋势线、分层线的事后比较', '',
        '原完整ROI，ν=0.26。γ≈1.3e-9的精确值为1.2648552168552956e-9。所有数值为事后诊断，不按标定最小误差重新选择γ。', '',
        '趋势线定义：七层中心制造坐标−48/−32/−16/0/16/32/48 mm上的本次1023单调约束标定均值作分段线性连接，仅在[−48,48] mm使用，不外推。中心按16 mm层间距延续到两个外层，是辅助绘图/评价约定，不是真实连续材料场。分层线按已配置−40/−24/−8/8/24/40 mm界面分为七层常值。', '',
        '主指标为反演宽度均值曲线的平均绝对相对误差：100/(b−a) × ∫|Ē(x)/E_ref(x)−1|dx。两条线均在相同[−48,48] mm上评价。另列完整ROI的分层曲线误差及七层均值误差；后者为七层各自均值相对差绝对值的等权平均，与曲线逐位置误差不同。', '',
        '| 次数 | γ | 趋势线误差 % | 分层线误差 %（共同支撑） | 分层线误差 %（完整ROI） | 七层均值误差 % |',
        '|---|---:|---:|---:|---:|---:|']
    for r in candidates.itertuples():
        lines.append(f'| {r.run} | {r.gamma:.3g} | {r.trend_common_mean_absolute_relative_error_percent:.4f} | {r.layered_common_mean_absolute_relative_error_percent:.4f} | {r.layered_full_mean_absolute_relative_error_percent:.4f} | {r.seven_region_mean_absolute_relative_error_percent:.4f} |')
    lines += ['', '全扫描中各指标最接近标定参照的点（仅限原停止标志通过者；不是正式最优γ）：', '',
              '| 次数 | 指标 | γ | 相对误差 % |', '|---|---|---:|---:|']
    for run in (1, 2, 3):
        data = all_scan[(all_scan.run == run) & all_scan.solver_stopping_passed]
        for key, label in [('trend_common_mean_absolute_relative_error_percent', '趋势线'),
                           ('layered_common_mean_absolute_relative_error_percent', '分层线（共同支撑）')]:
            best = data.loc[data[key].idxmin()]
            lines.append(f'| {run} | {label} | {best.gamma:.4g} | {best[key]:.4f} |')
    lines += ['', '本次对七层边界精确切分后积分，因此七层均值误差与旧报告直接按有限元Gauss点归区的近似值可能略有差别。新数值没有覆盖旧结果。主表只评价宽度均值；二维横向起伏仍须结合已有二维场和稳定性报告，不能用曲线相近证明二维模量场正确。', '',
              '核验：常值场10%相对差及完全匹配趋势线零差异解析检查；在有限元节点、层中心、制造界面及误差过零点切分，6点和12点Gauss积分逐场核对到1e-10相对/绝对容差。未运行优化、修改参考值或正式γ。L₂相对误差另存CSV，避免将不同指标混称。', '',
              '![标定曲线比较](calibration_profiles.png)', '',
              '复现：`python -B -m lib.dic_calibration_curve_comparison --cases <三个完整ROI冻结目录> --study <小γ稳定性目录> --output <新输出目录>`。']
    (output/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    (output/'provenance.json').write_text(json.dumps(dict(source_fields=sources, script_sha256=file_sha256(Path(__file__)),
        centers_mm=CENTERS.tolist(), interfaces_mm=INTERFACES.tolist(), common_support_mm=[-48,48],
        calibration_role='Post-hoc comparison only; not a gamma-selection criterion',
        metric='Length-averaged absolute relative difference of transverse-mean curves',
        quadrature_check='Order 6 versus 12; split FE knots, reference knots, interfaces and difference zeroes'),
        ensure_ascii=False, indent=2), encoding='utf-8')
    print(candidates.to_string(index=False))
    print('\n'.join(lines[lines.index('全扫描中各指标最接近标定参照的点（仅限原停止标志通过者；不是正式最优γ）：'):][:10]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', type=Path, nargs=3, required=True)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    analyze(args.cases, args.study, args.output)
