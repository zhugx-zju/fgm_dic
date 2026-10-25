"""Compare post hoc cropped 380N H1 runs on shared physical supports.

Read saved fields only. The original seven-region run is restricted to exactly
the same DIC points, and all predictions receive the same three-mode rigid fit.
Modulus statistics use a common partition of both Q4 meshes and material zones.
"""
import html
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

from .dic_case_workflow import load_prepared
from .experiment_data import file_sha256


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results/dic_full/load380_20261022/h1'
RUNS = ('load380_subset2_60_50_20261025', 'load380_subset3_60_50_40_20261025')
OUTPUT = ROOT / 'results/dic_full/subset380_comparison_20261025'
NOTICE = ('按原380N、H1、ν=0.26的分区平均模量相对independent标定差异，'
          '事后固定连续两区60%/50%与三区60%/50%/40%。标定仅用于选区和事后对照，'
          '不进入局部反演目标、初值或γ选择；选区后符合度不作为独立验证。'
          '380N净力仍假定319.9 N，非同步实测。截取端均布牵引及内部横向自由均为近似，'
          '物理配准、边界等效和跨批标定转移限制保留。')


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _rigid_metrics(xy, observed, prediction, area):
    """Independent three-rigid-mode projection; area in mm², fields in mm."""
    centered = xy - xy.mean(axis=0)
    rigid = np.zeros((2*len(xy), 3))
    rigid[0::2, 0] = rigid[1::2, 1] = 1.
    rigid[0::2, 2], rigid[1::2, 2] = -centered[:, 1], centered[:, 0]
    residual = (observed-prediction).ravel()
    weights = np.repeat(area, 2)
    coefficients, _, rank, _ = np.linalg.lstsq(
        np.sqrt(weights)[:, None]*rigid, np.sqrt(weights)*residual, rcond=None)
    if rank != 3:
        raise ValueError('Subset cannot resolve the three rigid modes')
    residual = (residual-rigid@coefficients).reshape(-1, 2)
    return {'rmse_ux_um': float(1e3*np.sqrt(np.average(residual[:, 0]**2, weights=area))),
            'rmse_uy_um': float(1e3*np.sqrt(np.average(residual[:, 1]**2, weights=area)))}


def _interpolator(coordinates, values):
    x, y = np.unique(coordinates[:, 0]), np.unique(coordinates[:, 1])
    grid = np.full((len(y), len(x)), np.nan)
    grid[np.searchsorted(y, coordinates[:, 1]), np.searchsorted(x, coordinates[:, 0])] = values
    if not np.isfinite(grid).all():
        raise ValueError('Expected complete regular Q4 grid')
    return x, y, RegularGridInterpolator((y, x), grid, bounds_error=True)


def _common_regions(full, local, offset, boundaries, calibration, order=3):
    """Exact Q4 mean and second moment on a shared integration partition."""
    fx, fy, interp_full = _interpolator(full['coordinates'], full['E_inverse'])
    lx, ly, interp_local = _interpolator(local['coordinates'], local['E_inverse'])
    x_edges = np.unique(np.r_[lx, fx[(fx > offset) & (fx < offset+lx[-1])]-offset, boundaries])
    y_edges = np.unique(np.r_[ly, fy[(fy > ly[0]) & (fy < ly[-1])]])
    nodes, weights = np.polynomial.legendre.leggauss(order)
    def quadrature(edges):
        half = np.diff(edges)/2
        return ((edges[:-1, None]+half[:, None]*(nodes+1)).ravel(),
                (half[:, None]*weights).ravel())
    x, wx = quadrature(x_edges)
    y, wy = quadrature(y_edges)
    xx, yy = np.meshgrid(x, y)
    w = wy[:, None]*wx
    points = np.column_stack((yy.ravel(), xx.ravel()))
    values = {'full_inverse_restricted': interp_full(points+[0., offset]).reshape(xx.shape),
              'subset_inverse': interp_local(points).reshape(xx.shape)}
    group = np.searchsorted(boundaries, xx, side='right')
    rows = []
    for i, record in enumerate(calibration):
        mask = group == i
        for model, field in values.items():
            mean = float(np.average(field[mask], weights=w[mask]))
            sd = float(np.sqrt(np.average((field[mask]-mean)**2, weights=w[mask])))
            rows.append(dict(model=model, infill_percent=record['infill_percent'],
                             batch=record['batch'], n=record['n'],
                             calibration_MPa=record['mean_E_MPa'], calibration_sd_MPa=record['sd_E_MPa'],
                             mean_MPa=mean, spatial_sd_MPa=sd,
                             difference_percent=100*(mean/record['mean_E_MPa']-1)))
    return pd.DataFrame(rows)


def main():
    OUTPUT.mkdir(exist_ok=True)
    baseline = _read(BASE/'01_prepared/run_metadata.json')
    source = BASE/'03_inverse/nu_0.26/regional_modulus_comparison.csv'
    ranks = pd.read_csv(source).copy()
    ranks['absolute_difference_percent'] = abs(ranks.difference_percent)
    ranks.sort_values('absolute_difference_percent').to_csv(OUTPUT/'selection_ranking.csv', index=False)
    all_regions, all_metrics, status_rows, evidence = [], [], [], []
    fig, axes = plt.subplots(1, 2, figsize=(10., 3.8), sharey=True, layout='constrained')
    for run, ax in zip(RUNS, axes):
        root = ROOT/'results/dic_full'/run
        cfg, mesh, cases, prepared, _ = load_prepared(root/'01_prepared')
        inverse = _read(root/'03_inverse/run_metadata.json')
        scan = pd.read_csv(root/'03_inverse/selection.csv', float_precision='round_trip')
        status_rows.append(dict(run=run, regions='/'.join(map(str, cfg.REGION_INFILL_PERCENT)),
                                length_mm=mesh.geo_l, observations=prepared['coverage'][0]['points'],
                                modulus_parameters=mesh.n_nod, selected_gamma=inverse.get('selected_gamma'),
                                solver_stops=int(scan.converged.sum()), scan_gtol=int(scan.gradient_tolerance_satisfied.sum()),
                                scan_fe=int(scan.evaluations.sum()), status=inverse['status']))
        if inverse.get('selected_gamma') is None:
            ax.text(.5, .5, 'No eligible L-curve corner', transform=ax.transAxes, ha='center')
            continue
        offset = baseline['offset_mm'][0]-prepared['offset_mm'][0]
        with np.load(root/'01_prepared/fields.npz') as subset, np.load(BASE/'01_prepared/fields.npz') as original:
            masks = {}
            for case in cases:
                raw = original[f'{case.name}_raw_reference_coordinates']
                xi = 6-(raw[:, 1]+21.50749765093)
                lo, hi = cfg.MATERIAL_WINDOW_MM
                masks[case.name] = (xi >= lo-1e-7) & (xi <= hi+1e-7)
                np.testing.assert_array_equal(subset[f'{case.name}_observed'], original[f'{case.name}_observed'][masks[case.name]])
            data = {k: subset[k].copy() for k in subset.files}
        for nu in cfg.NU_VALUES:
            saved_metrics = pd.read_csv(root/'03_inverse/comparison.csv')
            with np.load(BASE/f'03_inverse/nu_{nu:.2f}/fields.npz') as full, \
                 np.load(root/f'03_inverse/nu_{nu:.2f}/fields.npz') as local, \
                 np.load(root/f'02_forward/nu_{nu:.2f}/fields.npz') as forward:
                table = _common_regions(full, local, offset, cfg.REGION_BOUNDARIES_X, prepared['calibration'])
                check = _common_regions(full, local, offset, cfg.REGION_BOUNDARIES_X, prepared['calibration'], order=2)
                error = float(np.max(abs(table.difference_percent-check.difference_percent)))
                assert error < 1e-8
                evidence.append(dict(run=run, nu=nu, common_quadrature_2_vs_3_max_difference_pp=error))
                table.insert(0, 'nu', nu); table.insert(0, 'run', run)
                all_regions.extend(table.to_dict(orient='records'))
                for case in cases:
                    predictions = {'full_inverse_restricted': full[f'{case.name}_prediction'].reshape(-1, 2)[masks[case.name]],
                                   'subset_inverse': local[f'{case.name}_prediction'].reshape(-1, 2),
                                   'subset_calibration_forward': forward[f'{case.name}_prediction'].reshape(-1, 2)}
                    for model, prediction in predictions.items():
                        result = _rigid_metrics(data[f'{case.name}_coordinates'], data[f'{case.name}_observed'],
                                                prediction, data[f'{case.name}_point_area_mm2'])
                        if model == 'subset_inverse':
                            saved = saved_metrics[(saved_metrics.case == case.name) & np.isclose(saved_metrics.nu, nu)].iloc[0]
                            np.testing.assert_allclose([result['rmse_ux_um'], result['rmse_uy_um']],
                                                       1e3*saved[['rmse_ux_mm', 'rmse_uy_mm']].to_numpy(dtype=float),
                                                       rtol=1e-8, atol=1e-10)
                        all_metrics.append(dict(run=run, nu=nu, platform=case.name, model=model, **result))
                if np.isclose(nu, .26):
                    full_profile = pd.read_csv(BASE/f'03_inverse/nu_{nu:.2f}/x_profile.csv')
                    profile = pd.read_csv(root/f'03_inverse/nu_{nu:.2f}/x_profile.csv')
                    start, end = offset, offset+mesh.geo_l
                    full_x = np.unique(np.r_[start, end, full_profile.x_mm[(full_profile.x_mm > start) & (full_profile.x_mm < end)]])
                    ax.plot(full_x, np.interp(full_x, full_profile.x_mm, full_profile.inverse_width_mean_E_MPa),
                            color='#777777', linestyle='--', label='Original full-domain inversion')
                    ax.plot(profile.x_mm+offset, profile.inverse_width_mean_E_MPa, color='#0072B2', label='Subset inversion')
                    edges = np.r_[0., cfg.REGION_BOUNDARIES_X, mesh.geo_l]+offset
                    calibration = pd.DataFrame(prepared['calibration'])
                    means, sd = calibration.mean_E_MPa, calibration.sd_E_MPa
                    steps = np.repeat(edges, 2)[1:-1]
                    ax.plot(steps, np.repeat(means, 2), color='black', linestyle=':', label='Independent calibration')
                    ax.fill_between(steps, np.repeat(means-sd, 2), np.repeat(means+sd, 2), color='gray', alpha=.15)
                    ax.set(xlabel='Original full-domain x (mm)', title=f'{len(cfg.REGION_INFILL_PERCENT)} regions: '+ '/'.join(map(str, cfg.REGION_INFILL_PERCENT)))
        evidence.append(dict(run=run, prepared_sha256=file_sha256(root/'01_prepared/fields.npz'),
                             selection_sha256=file_sha256(root/'03_inverse/selection.csv')))
    axes[0].set_ylabel('Width-mean E (MPa)')
    axes[0].legend(frameon=False, fontsize=8)
    fig.savefig(OUTPUT/'modulus_comparison.png', dpi=300)
    fig.savefig(OUTPUT/'modulus_comparison.svg')
    plt.close(fig)
    regions, metrics, status = pd.DataFrame(all_regions), pd.DataFrame(all_metrics), pd.DataFrame(status_rows)
    regions.to_csv(OUTPUT/'common_regional_modulus.csv', index=False)
    metrics.to_csv(OUTPUT/'common_displacement_metrics.csv', index=False)
    status.to_csv(OUTPUT/'status.csv', index=False)
    (OUTPUT/'verification.json').write_text(json.dumps(evidence, indent=2), encoding='utf-8')
    parts = ['<!doctype html><meta charset="utf-8"><title>380N局部两区／三区反演对照</title>',
             '<style>body{max-width:1200px;margin:30px auto;font:16px/1.65 sans-serif}table{border-collapse:collapse}td,th{padding:6px;border:1px solid #ddd}img{max-width:100%}</style>',
             '<h1>380N：连续两区／三区局部反演</h1>', '<p>'+html.escape(NOTICE)+'</p>',
             '<p>均保留原H1目标、等权重、ν=0.26下50点L-curve及四个ν敏感性。各组重新用自身局部DIC初始化与选γ。'
             '原七区场与局部场的位移指标均在同一子集重拟合三个刚体模式；模量采用共同物理积分。'
             '不同域的L-curve残差范数及γ不用于直接比较精度。两个局部算例使用当前同一内核；原七区基线读取历史保存场，未重算。</p>',
             '<p><strong>计算完成；每组四个最终ν均由函数下降停止，均未达严格GTOL。'
             '局部位移拟合略有改善，分区平均模量与标定的差异总体增大。</strong> '
             '<a href="../../../docs/dic_subset380_results_20261025.md">完整定义、数值、核验及限制</a></p>',
             status.to_html(index=False, float_format=lambda v: f'{v:.6g}'),
             '<p><a href="selection_ranking.csv">原分区差异排名</a> · <a href="common_regional_modulus.csv">共同积分模量统计</a> · '
             '<a href="common_displacement_metrics.csv">同点位移误差</a> · <a href="verification.json">核验记录</a></p>']
    for run in RUNS:
        parts.append(f'<p><a href="../{run}/结果总览.html">{run} 四步结果与全部云图</a></p>')
    parts.append('<img src="modulus_comparison.png" alt="Width-mean modulus on original physical coordinates">')
    if not regions.empty:
        parts += ['<h2>ν=0.26：共同物理积分分区模量</h2>',
                  regions[np.isclose(regions.nu, .26)].to_html(index=False, float_format=lambda v: f'{v:.5g}')]
    if not metrics.empty:
        parts += ['<h2>ν=0.26：同点、同权重、同刚体规则的位移RMSE（μm）</h2>',
                  metrics[np.isclose(metrics.nu, .26)].to_html(index=False, float_format=lambda v: f'{v:.5g}')]
    (OUTPUT/'index.html').write_text('\n'.join(parts), encoding='utf-8')
    print(status.to_string(index=False))
    print(OUTPUT/'index.html')


if __name__ == '__main__':
    main()
