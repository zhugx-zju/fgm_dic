"""Saved-field comparison of exact TV and H1 on the fixed local 380N domains."""
import html
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .dic_case_workflow import load_prepared
from .dic_subset_comparison import _common_regions, _rigid_metrics, NOTICE
from .experiment_data import file_sha256


ROOT = Path(__file__).resolve().parents[1]
RUNS = ('load380_subset2_60_50_exact_tv_20261025',
        'load380_subset3_60_50_40_exact_tv_20261025')
OUTPUT = ROOT/'results/dic_full/subset380_tv_comparison_20261025'


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    OUTPUT.mkdir(exist_ok=True)
    statuses, regions, metrics, checks = [], [], [], []
    fig, axes = plt.subplots(1, 2, figsize=(10., 3.9), sharey=True, layout='constrained')
    for name, ax in zip(RUNS, axes):
        root = ROOT/'results/dic_full'/name
        cfg, mesh, cases, prepared, _ = load_prepared(root/'01_prepared')
        baseline = ROOT/'results/dic_full'/cfg.BASELINE_CASE
        inverse = _read(root/'03_inverse/run_metadata.json')
        scan = pd.read_csv(root/'03_inverse/selection.csv', float_precision='round_trip')
        if len(scan) != 50:
            raise ValueError('Complete the declared 50-point scan before reporting')
        status = dict(run=name, regions='/'.join(map(str, cfg.REGION_INFILL_PERCENT)),
            status=inverse['status'], selected_gamma=inverse.get('selected_gamma'),
            passed=int(scan.converged.sum()),
            passed_nonzero=int((scan.converged & scan.regularization_resolved).sum()),
            scan_FE_lower=int(scan.get('evaluations_lower_bound', scan.evaluations).fillna(scan.evaluations).sum()),
            scan_FE_upper=int(scan.get('evaluations_upper_bound', scan.evaluations).fillna(scan.evaluations).sum()),
            numerical_exceptions=int((scan.stop_message == 'numerical_error_no_returned_field').sum()), pilot_FE=1800,
            final_passed=sum(s['converged'] for s in inverse.get('solvers', [])))
        statuses.append(status)
        with np.load(root/'01_prepared/fields.npz') as tv_input, np.load(baseline/'01_prepared/fields.npz') as h1_input:
            assert tv_input.files == h1_input.files
            for key in tv_input.files:
                np.testing.assert_array_equal(tv_input[key], h1_input[key])
            data = {k: tv_input[k].copy() for k in tv_input.files}
        pd.testing.assert_frame_equal(pd.read_csv(root/'02_forward/comparison.csv'),
                                      pd.read_csv(baseline/'02_forward/comparison.csv'), check_exact=True)
        checks.append(dict(run=name, all_input_arrays_identical=True, forward_metrics_identical=True,
            input_sha256=file_sha256(root/'01_prepared/fields.npz'),
            selection_sha256=file_sha256(root/'03_inverse/selection.csv')))
        h1_profile = pd.read_csv(baseline/'03_inverse/nu_0.26/x_profile.csv')
        ax.plot(h1_profile.x_mm, h1_profile.inverse_width_mean_E_MPa,
                color='#0072B2', label='H1: locally selected gamma')
        calibration = pd.DataFrame(prepared['calibration'])
        edges = np.r_[0., cfg.REGION_BOUNDARIES_X, mesh.geo_l]
        step_x = np.repeat(edges, 2)[1:-1]
        ax.plot(step_x, np.repeat(calibration.mean_E_MPa, 2), color='black', linestyle=':',
                label='Independent calibration')
        ax.fill_between(step_x, np.repeat(calibration.mean_E_MPa-calibration.sd_E_MPa, 2),
                        np.repeat(calibration.mean_E_MPa+calibration.sd_E_MPa, 2), color='gray', alpha=.15)
        ax.set(xlabel='Local x (mm)', title=f'{len(cfg.REGION_INFILL_PERCENT)} regions: '+status['regions'])
        if inverse.get('selected_gamma') is None:
            ax.text(.5, .08, 'Exact TV: no eligible L-curve corner', transform=ax.transAxes, ha='center', fontsize=9)
            continue
        tv_profile = pd.read_csv(root/'03_inverse/nu_0.26/x_profile.csv')
        primary_passed = next(s['converged'] for s in inverse['solvers'] if np.isclose(s['nu'], .26))
        ax.plot(tv_profile.x_mm, tv_profile.inverse_width_mean_E_MPa,
                color='#D55E00', linestyle='--', label=('Exact TV: locally selected gamma' if primary_passed
                    else 'Exact TV candidate: NOT certified'))
        for nu in cfg.NU_VALUES:
            certified = next(s['converged'] for s in inverse['solvers'] if np.isclose(s['nu'], nu))
            tv_label = 'exact_TV' if certified else 'uncertified_TV_candidate'
            with np.load(baseline/f'03_inverse/nu_{nu:.2f}/fields.npz') as h1, \
                 np.load(root/f'03_inverse/nu_{nu:.2f}/fields.npz') as tv, \
                 np.load(root/f'02_forward/nu_{nu:.2f}/fields.npz') as forward:
                table = _common_regions(h1, tv, 0., cfg.REGION_BOUNDARIES_X, prepared['calibration'])
                check = _common_regions(h1, tv, 0., cfg.REGION_BOUNDARIES_X, prepared['calibration'], order=2)
                difference = float(np.max(abs(table.difference_percent-check.difference_percent)))
                assert difference < 1e-8
                checks.append(dict(run=name, nu=nu, common_quadrature_max_difference_pp=difference))
                table['model'] = table.model.replace({'full_inverse_restricted': 'H1', 'subset_inverse': tv_label})
                table.insert(0, 'nu', nu); table.insert(0, 'run', name)
                regions.extend(table.to_dict(orient='records'))
                for case in cases:
                    for model, field in [('H1', h1), (tv_label, tv), ('calibration_forward', forward)]:
                        measured = data[f'{case.name}_observed']
                        prediction = field[f'{case.name}_prediction'].reshape(-1, 2)
                        values = _rigid_metrics(data[f'{case.name}_coordinates'], measured, prediction,
                                               data[f'{case.name}_point_area_mm2'])
                        source = (baseline/'03_inverse/comparison.csv' if model == 'H1' else
                                  root/'02_forward/comparison.csv' if model == 'calibration_forward' else
                                  root/'03_inverse/comparison.csv')
                        saved_metrics = pd.read_csv(source)
                        saved = saved_metrics[(saved_metrics.case == case.name) & np.isclose(saved_metrics.nu, nu)].iloc[0]
                        np.testing.assert_allclose([values['rmse_ux_um'], values['rmse_uy_um']],
                            1e3*saved[['rmse_ux_mm', 'rmse_uy_mm']].to_numpy(dtype=float), rtol=1e-8, atol=1e-10)
                        metrics.append(dict(run=name, nu=nu, platform=case.name, model=model, **values))
    axes[0].set_ylabel('Width-mean E (MPa)')
    handles, labels = axes[-1].get_legend_handles_labels()
    if len(handles) < 3:
        handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncol=3, frameon=False, fontsize=8)
    fig.savefig(OUTPUT/'h1_tv_modulus.png', dpi=300)
    fig.savefig(OUTPUT/'h1_tv_modulus.svg')
    plt.close(fig)
    status_table, regional, displacement = pd.DataFrame(statuses), pd.DataFrame(regions), pd.DataFrame(metrics)
    status_table.to_csv(OUTPUT/'status.csv', index=False)
    regional.to_csv(OUTPUT/'common_regional_modulus.csv', index=False)
    displacement.to_csv(OUTPUT/'common_displacement_metrics.csv', index=False)
    summaries = []
    if not regional.empty:
        for (run, model), frame in regional[np.isclose(regional.nu, .26)].groupby(['run', 'model']):
            summaries.append(dict(run=run, model=model,
                mean_absolute_regional_difference_percent=float(frame.difference_percent.abs().mean()),
                max_absolute_regional_difference_percent=float(frame.difference_percent.abs().max())))
    summary_table = pd.DataFrame(summaries)
    summary_table.to_csv(OUTPUT/'material_summary.csv', index=False)
    (OUTPUT/'verification.json').write_text(json.dumps(checks, indent=2), encoding='utf-8')
    page = ['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>380N局部精确TV与H1对照</title>',
        '<style>body{max-width:1300px;margin:30px auto;font:16px/1.7 sans-serif}table{border-collapse:collapse}td,th{padding:6px;border:1px solid #ddd}img{max-width:100%}</style>',
        '<h1>固定连续两区／三区：精确TV与H1</h1>', '<p>'+html.escape(NOTICE)+'</p>',
        '<p>H1目标为f+(γ/2)∫|∇E|²，精确TV为f+γ∫|∇E|；自由二维节点Q4场及观测、网格、力、权重相同。'
        '两种γ分别由各自局部380N、ν=0.26的50点L-curve选取，单位不同。TV未使用Huber阈值或分区先验。'
        'TV必须同时通过驻点、对偶可行性和Fenchel间隙条件；零TV常量场不参与对数曲率。'
        'H1原最终场按函数下降停止，未达严格GTOL；两方法停止条件及计算成本不同。</p>',
        '<p><a href="../../../docs/dic_subset380_tv_results_20261025.md">完整定义、结果、核验与限制</a> · '
        '<a href="../subset380_comparison_20261025/index.html">先前局部H1与七区场对照</a></p>',
        status_table.to_html(index=False, float_format=lambda v: f'{v:.6g}'),
        '<p>scan_FE_lower/upper为正式扫描总FE评估的下／上界；原中断点未返回精确计数时保留区间，不伪造确数。'
        'pilot_FE为另计的步长先导；final_passed为最终四个ν通过数。'
        '选参失败的算例仅保留扫描诊断，不用手指定γ生成正式识别场。</p>']
    for name in RUNS:
        page.append(f'<p><a href="../{name}/结果总览.html">{name}：四步输出与云图</a> · '
                    f'<a href="../{name}/verification/summary.json">独立一阶证据核验</a></p>')
    page.append('<img src="h1_tv_modulus.png" alt="H1 and exact TV width-mean modulus">')
    if not summary_table.empty:
        page.extend(['<h2>ν=0.26：分区平均模量相对标定差异（%）</h2>',
                     summary_table.to_html(index=False, float_format=lambda v: f'{v:.6g}'),
                     regional[np.isclose(regional.nu, .26)].to_html(index=False, float_format=lambda v: f'{v:.6g}')])
    if not displacement.empty:
        page.extend(['<h2>ν=0.26：同点位移RMSE（μm）</h2>',
                     displacement[np.isclose(displacement.nu, .26)].to_html(index=False, float_format=lambda v: f'{v:.6g}')])
    page.extend(['<p><a href="common_regional_modulus.csv">共同积分区域统计</a> · '
                 '<a href="common_displacement_metrics.csv">同点位移误差</a> · '
                 '<a href="verification.json">输入一致性与共同积分核验</a></p>', '</html>'])
    (OUTPUT/'index.html').write_text('\n'.join(page), encoding='utf-8')
    print(status_table.to_string(index=False))
    if not summary_table.empty:
        print(summary_table.to_string(index=False))
    print(OUTPUT/'index.html')


if __name__ == '__main__':
    main()
