"""Audit saved coarse scans and render explicitly unselected diagnostic fields.

This postprocessor never solves an inverse problem or changes gamma selection.
Run with ``python -X utf8 -B -m lib.unified_coarse_diagnostics``.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np

from fgm_asm.l_curve import MIN_POSITIVE_CURVATURE, select_l_curve_corner
from fgm_asm.observation import compute_axis_cell_widths
from .dic_preprocessing import plot_grid_fields
from .unified_coarse_lcurve import digest, load, mesh, model, truth_function
from .unified_gradient_layered_study import read_json, write_json


LABELS = {'layered': '分层', 'continuous_2d': '连续二维', 'mixed_2d': '混合二维'}
STATUS = {'no_resolved_corner': '无可分辨正曲率拐点',
          'insufficient_corner_support': '曲率峰两侧有效点不足'}
# Fixed positions in the saved scan, declared before reading truth errors.
ANCHORS = (0, 24, 49)


def relative_error(value: float, reference: float) -> float:
    return float(abs(value-reference)/max(abs(reference), np.finfo(float).tiny))


def audit(root: Path, cfg, material) -> dict:
    records = []
    for kind in cfg.CASES:
        folder = root/'03_inverse'/kind
        selection = read_json(folder/'selection.json')
        with np.load(root/'02_forward'/kind/'observations.npz') as data:
            area = np.repeat(data['areas'], 2)
        rows = []
        for index in range(len(cfg.GAMMA_CANDIDATES)):
            point = folder/f'gamma_{index:02d}'
            row = read_json(point/'summary.json')
            with np.load(point/'fields.npz') as saved:
                rho = float(np.sqrt(np.dot(area, saved['residual']**2)))
                eta = float(np.sqrt(material.regularization(saved['theta'])[0]))
                gradient_inf = float(np.max(abs(saved['gradient'])))
            errors = dict(residual_relative_error=relative_error(rho, row['residual_norm']),
                          regularization_relative_error=relative_error(eta, row['regularization_norm']),
                          gradient_relative_error=relative_error(gradient_inf, row['gradient_inf']))
            assert max(errors.values()) < 1e-7, (kind, index, errors)
            assert row['converged'] == row['first_order_passed']
            assert row['first_order_passed'] == (row['projected_gradient_inf'] <= cfg.GTOL)
            assert row['forward_evaluations'] <= cfg.MAX_EVALUATIONS
            assert row['iterations'] <= cfg.MAX_ITERATIONS
            rows.append(dict(index=index, **row, **errors))
        result = select_l_curve_corner([r['gamma'] for r in rows],
                 [r['residual_norm'] for r in rows], [r['regularization_norm'] for r in rows],
                 [r['converged'] for r in rows])
        assert result['status'] == selection['status']
        assert result['selected_gamma'] == selection['selected_gamma']
        if result['status'] not in STATUS or result['selected_gamma'] is not None:
            raise ValueError('This report supports the completed, unselected coarse study only')
        assert sum(r['converged'] for r in rows) == selection['valid_points']
        assert sum(r['forward_evaluations'] for r in rows) == selection['forward_evaluations']
        positive = np.flatnonzero(np.isfinite(result['curvature']) &
                                  (result['curvature'] > MIN_POSITIVE_CURVATURE))
        peak = None
        if len(positive):
            i = int(positive[np.argmax(result['curvature'][positive])])
            peak = dict(index=i, gamma=rows[i]['gamma'], curvature=float(result['curvature'][i]),
                        accepted=False, neighbor_indices=list(range(max(0, i-2), min(len(rows), i+3))),
                        neighbor_passed=[r['converged'] for r in rows[max(0, i-2):i+3]])
        records.append(dict(**selection, unaccepted_curvature_peak=peak, points=rows))
        print('Audited', kind, selection['status'], flush=True)
    return dict(status='passed', independent_fe_evaluations=0, records=records,
                scope='Saved residual/regularization/gradient norms, budgets and selection recomputed; '
                      'not an independent forward/adjoint solve',
                total_scan_fe_evaluations=sum(r['forward_evaluations'] for r in records),
                total_first_order_passed=sum(r['valid_points'] for r in records))


def render_fields(root: Path, cfg, material) -> list[dict]:
    out = root/'04_comparison'/'unselected_diagnostics'
    ny, nx = cfg.EVALUATION_SHAPE
    X, Y = np.meshgrid(np.linspace(0, cfg.GEO_L, nx), np.linspace(0, cfg.GEO_H, ny))
    xy = np.column_stack((X.ravel(), Y.ravel()))
    weights = np.outer(compute_axis_cell_widths(Y[:, 0]), compute_axis_cell_widths(X[0]))
    wy = compute_axis_cell_widths(Y[:, 0])
    bounds = (0, cfg.GEO_L, 0, cfg.GEO_H)
    records = []
    for kind in cfg.CASES:
        truth = truth_function(cfg, kind)(X, Y)
        fields, rows = [], []
        for index in ANCHORS:
            folder = root/'03_inverse'/kind/f'gamma_{index:02d}'
            row = read_json(folder/'summary.json')
            with np.load(folder/'fields.npz') as saved:
                E = material.sample(saved['theta'], xy).reshape(X.shape)
            fields.append(E)
            rows.append(dict(case=kind, index=index, gamma=row['gamma'],
                        first_order_passed=row['first_order_passed'],
                        field_relative_rms=float(np.sqrt(np.sum(weights*(E-truth)**2)/np.sum(weights*truth**2))),
                        purpose='Unselected diagnostic only; truth error never used for gamma selection'))
        low = min(f.min() for f in [truth, *fields])
        high = max(f.max() for f in [truth, *fields])
        err = max(float(np.max(abs(f-truth))) for f in fields)
        plot_grid_fields(out, kind, X, Y, [('truth', truth, 'Synthetic truth')], 'jet',
                         limits={'truth': (low, high)}, title_prefix='', unit='MPa', bounds=bounds)
        fig, ax = plt.subplots(figsize=(8, 3.6))
        ax.plot(X[0], wy@truth/wy.sum(), 'k', lw=1.5, label='Truth')
        for E, row, color, line in zip(fields, rows, ('#0072B2', '#D55E00', '#009E73'), ('--', '-.', ':')):
            index = row['index']
            flag = 'pass' if row['first_order_passed'] else 'fail'
            label = f'Unselected #{index:02d}, gamma={row["gamma"]:.2e}, {flag}'
            suffix = f'gamma_{index:02d}'
            plot_grid_fields(out, kind, X, Y, [(suffix, E, label)], 'jet',
                             limits={suffix: (low, high)}, title_prefix='', unit='MPa', bounds=bounds)
            plot_grid_fields(out, kind, X, Y, [(suffix+'_error', E-truth, label+' error')], 'coolwarm',
                             limits={suffix+'_error': (-err, err)}, title_prefix='', unit='MPa', bounds=bounds)
            ax.plot(X[0], wy@E/wy.sum(), color=color, linestyle=line, lw=1.2, label=label)
        ax.set(xlabel='x (mm)', ylabel='Width-averaged E (MPa)', title='Unselected scan diagnostics')
        ax.legend(fontsize=8, frameon=False)
        fig.tight_layout()
        fig.savefig(out/f'{kind}_profile.png', dpi=300, bbox_inches='tight')
        fig.savefig(out/f'{kind}_profile.svg', bbox_inches='tight')
        plt.close(fig)
        records.extend(rows)
        print('Rendered diagnostic fields', kind, flush=True)
    return records


def publish(root: Path, result: dict, diagnostics: list[dict]) -> None:
    style = '<style>body{max-width:1100px;margin:25px auto;padding:20px;font:16px/1.7 "Microsoft YaHei",sans-serif}img{max-width:100%}table{border-collapse:collapse}td,th{padding:8px;border:1px solid #ccc}a{color:#075fa2}</style>'
    parts = ['<!doctype html><meta charset="utf-8">'+style,
             '<h1>121×19 粗网格仿真：L-curve 未取得有效选参</h1>',
             '<p>三类仿真共150点已完成；未选定γ，实验反演未启动。已知六个界面、七个自由二维分支，'
             '共17080系数、23920观测点；ν=0.26，两个分量各0.5 μm预设高斯噪声。</p>',
             '<table><tr><th>算例</th><th>一阶通过／50</th><th>扫描FE次数</th><th>选参状态</th></tr>']
    for record in result['records']:
        parts.append(f'<tr><td>{LABELS[record["case"]]}</td><td>{record["valid_points"]}/50</td>'
                     f'<td>{record["forward_evaluations"]}</td><td>{STATUS[record["status"]]}</td></tr>')
    parts += ['</table>', '<p>分层的区内常值属于正则零空间，残差随γ几乎不变；'
              '连续／混合场多数点未达到冻结的一阶条件，曲率峰缺少有效邻点。'
              '这说明当前求解和选参流程尚未通过，不足以否定区内正则本身。</p>',
              '<p>本轮采用GTOL一阶通过作为选参资格；原H1部分历史扫描接受函数下降停止，'
              '两者有效点比例不能直接用于比较算法优劣。未运行后续独立扰动与未知界面检验。</p>',
              '<p><a href="04_comparison/scan_audit.json">150点复核记录</a> · '
              '<a href="04_comparison/gate.json">实验进入条件</a> · '
              '<a href="04_comparison/unselected_diagnostics/metrics.json">未选定诊断点数据</a></p>']
    for record in result['records']:
        kind = record['case']
        parts.append(f'<h2>{LABELS[kind]}</h2><p><a href="03_inverse/{kind}/scan.csv">扫描数据</a> · '
                     f'<a href="03_inverse/{kind}/selection.json">选参判定</a></p>')
        for path in sorted((root/'03_inverse'/kind).glob('*.png')):
            rel = path.relative_to(root).as_posix()
            parts.append(f'<a href="{Path(rel).with_suffix(".svg").as_posix()}"><img src="{rel}" loading="lazy"></a>')
        parts.append('<p>以下固定显示扫描首点、中间点和末点（编号00、24、49），均为未选定γ的诊断场。'
                     '真值误差仅在扫描完成后描述这些预定点，不参与选参；pass/fail表示一阶条件。'
                     '同算例模量共享色标，误差共享对称色标，图像不代表正式反演结果。</p>')
        parts.append('<table><tr><th>编号</th><th>γ</th><th>一阶通过</th><th>诊断场相对RMS</th></tr>')
        for row in diagnostics:
            if row['case'] == kind:
                parts.append(f'<tr><td>{row["index"]:02d}</td><td>{row["gamma"]:.6g}</td>'
                             f'<td>{row["first_order_passed"]}</td><td>{100*row["field_relative_rms"]:.3f}%</td></tr>')
        parts.append('</table>')
        for suffix in ('truth', 'profile', *[s for i in ANCHORS for s in (f'gamma_{i:02d}', f'gamma_{i:02d}_error')]):
            rel = f'04_comparison/unselected_diagnostics/{kind}_{suffix}'
            parts.append(f'<a href="{rel}.svg"><img src="{rel}.png" loading="lazy"></a>')
    page = root/'结果总览.html'
    page.write_text('\n'.join(parts), encoding='utf-8')
    for target in re.findall(r'(?:href|src)="([^"]+)"', page.read_text(encoding='utf-8')):
        assert (root/target).is_file(), target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='unified_coarse_lcurve_config')
    args = parser.parse_args()
    root, cfg = load(args.config)
    # Numeric file hashes prove that this command changes presentation only.
    paths = sorted(p for p in root.rglob('*') if p.is_file() and p.suffix in ('.npz', '.csv'))
    before = {str(p.relative_to(root)): digest(p) for p in paths}
    material = model(mesh(cfg), cfg)
    result = audit(root, cfg, material)
    diagnostics = render_fields(root, cfg, material)
    assert before == {str(p.relative_to(root)): digest(p) for p in paths}
    result['numeric_files_unchanged'] = True
    result['numeric_files_checked'] = len(paths)
    result['postprocessor_sha256'] = digest(Path(__file__))
    write_json(root/'04_comparison/scan_audit.json', result)
    write_json(root/'04_comparison/unselected_diagnostics/metrics.json', diagnostics)
    publish(root, result, diagnostics)
    print('保存位置：', root/'结果总览.html', flush=True)


if __name__ == '__main__':
    main()
