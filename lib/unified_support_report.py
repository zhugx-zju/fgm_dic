"""Plot saved support-pilot states; never select gamma or solve FE problems."""
from __future__ import annotations

import argparse
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np

from fgm_asm.observation import compute_axis_cell_widths
from .dic_preprocessing import plot_grid_fields
from . import unified_support_pilot as pilot


COLORS = dict(smooth_full='#0072B2', sharp_full='#E69F00', sharp_local='#009E73',
              local_projected='#CC79A7', local_metric='#D55E00')
LINES = dict(smooth_full='-', sharp_full='--', sharp_local='-.', local_projected=':', local_metric='-')


def report(root, cfg):
    with np.load(root/'01_prepared/fields.npz') as saved:
        data = {name: saved[name] for name in saved.files}
    ny, nx = cfg.EVALUATION_SHAPE
    X, Y = np.meshgrid(np.linspace(0, cfg.GEO_L, nx), np.linspace(0, cfg.GEO_H, ny))
    xy = np.column_stack((X.ravel(), Y.ravel()))
    wy = compute_axis_cell_widths(Y[:, 0])
    weights = np.outer(wy, compute_axis_cell_widths(X[0]))
    values = np.array([row['mean_E_MPa'] for row in cfg.CALIBRATION])
    reference = values[np.searchsorted(cfg.BOUNDARIES, X, side='right')]
    out = root/'comparison'; out.mkdir(exist_ok=True)
    records, fields = [], {}
    import pandas as pd
    for method in cfg.METHODS:
        summary = pilot.read_json(root/method/'summary.json')
        p, _, _ = pilot.build(cfg, data, method)
        for row in summary['records']:
            i = row['index']
            with np.load(root/method/f'gamma_{i:02d}'/'fields.npz') as saved:
                E = p.sample(saved['theta'], xy).reshape(X.shape)
            fields[method, i] = E
            records.append(dict(**row, field_min_MPa=float(E.min()), field_max_MPa=float(E.max())))
    comparisons = []
    for i, gamma in enumerate(cfg.GAMMAS):
        lo = min([reference.min()]+[fields[m, i].min() for m in cfg.METHODS])
        hi = max([reference.max()]+[fields[m, i].max() for m in cfg.METHODS])
        for method in cfg.METHODS:
            label = f'{method}, gamma={gamma:.2e} (pilot, unselected)'
            plot_grid_fields(out, f'gamma_{i:02d}', X, Y, [(method, fields[method, i], label)],
                'jet', limits={method: (lo, hi)}, unit='MPa', title_prefix='',
                bounds=(0, cfg.GEO_L, 0, cfg.GEO_H))
        fig, axes = plt.subplots(2, 1, figsize=(8, 5.8), sharex=True, constrained_layout=True)
        axes[0].plot(X[0], wy@reference/wy.sum(), 'k--', label='Independent calibration')
        for method in cfg.METHODS:
            E = fields[method, i]; mean = wy@E/wy.sum()
            spread = np.sqrt(wy@(E-mean[None, :])**2/wy.sum())
            axes[0].plot(X[0], mean, color=COLORS[method], linestyle=LINES[method], label=method)
            axes[1].plot(X[0], spread, color=COLORS[method], linestyle=LINES[method])
        axes[0].set(ylabel='Width-mean E (MPa)', title=f'Unselected pilot gamma={gamma:.2e}')
        axes[1].set(xlabel='x (mm)', ylabel='Transverse spatial SD (MPa)')
        axes[0].legend(fontsize=8, frameon=False)
        fig.savefig(out/f'gamma_{i:02d}_profiles.png', dpi=300, bbox_inches='tight')
        fig.savefig(out/f'gamma_{i:02d}_profiles.svg', bbox_inches='tight'); plt.close(fig)
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)
        for method in cfg.METHODS:
            history = pd.read_csv(root/method/f'gamma_{i:02d}'/'history.csv')
            for ax, key in zip(axes, ('value', 'gradient_inf')):
                ax.semilogy(history.evaluation, history[key], color=COLORS[method], linestyle=LINES[method], label=method)
        axes[0].set(xlabel='Forward evaluations', ylabel='Objective J', title='All evaluated trials')
        axes[1].set(xlabel='Forward evaluations', ylabel='Raw log-gradient infinity norm', title='Final projected gradient is tabulated')
        axes[0].legend(fontsize=8, frameon=False)
        fig.savefig(out/f'gamma_{i:02d}_convergence.png', dpi=300, bbox_inches='tight')
        fig.savefig(out/f'gamma_{i:02d}_convergence.svg', bbox_inches='tight'); plt.close(fig)
        for first, second in (('sharp_full', 'sharp_local'), ('local_projected', 'local_metric')):
            difference = float(np.sqrt(np.sum(weights*(fields[first, i]-fields[second, i])**2)/
                                       np.sum(weights*fields[first, i]**2)))
            comparisons.append(dict(index=i, gamma=gamma, first=first, second=second,
                                    field_relative_rms_difference=difference))
    pilot.write_json(out/'metrics.json', dict(records=records, field_comparisons=comparisons,
        scope='Three fixed pilot points; no selected gamma, truth-based tuning or experimental validation',
        report_source_sha256=pilot.shared.digest(Path(__file__))))
    style = '<style>body{max-width:1250px;margin:25px auto;padding:20px;font:16px/1.7 "Microsoft YaHei",sans-serif}table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:7px}img{max-width:100%}a{color:#075fa2}</style>'
    parts = ['<!doctype html><meta charset="utf-8">'+style+'<h1>380N：局部支撑与预条件试验</h1>',
        '<p>固定界面、121×19位移单元、23920观测点，净力319.9 N仍为假设值。'
        '只比较三个预定γ，每点400迭代／600次FE；这些不是L-curve所选参数。</p>',
        '<p>smooth_full→sharp_full包含界面模型变化；sharp_full→sharp_local只消去尖锐模型无支撑系数。'
        'local_projected与local_metric采用相同投影L-BFGS步骤，后者增加均值／起伏逆度量。</p>',
        '<p>通过判据仍为原log系数坐标的投影梯度≤1e-9。预条件均值曲率准备另用14次FE，三个γ共用。'
        '下表求解FE不含该准备开销。原目标、盒约束和正则权重未改变。</p>',
        '<table><tr><th>方法</th><th>γ</th><th>参数数</th><th>一阶通过</th><th>求解FE</th>'
        '<th>投影梯度∞范数</th><th>最终目标</th><th>轴向／横向RMSE（μm）</th></tr>']
    for row in records:
        ux, uy = row['component_rmse_um']
        parts.append(f'<tr><td>{row["method"]}</td><td>{row["gamma"]:.3g}</td><td>{row["parameters"]}</td>'
             f'<td>{row["first_order_passed"]}</td><td>{row["forward_evaluations"]}</td>'
             f'<td>{row["projected_gradient_inf"]:.3g}</td><td>{row["final_value"]:.10g}</td><td>{ux:.3f}／{uy:.3f}</td></tr>')
    parts += ['</table>', '<p>全部图为预定试验点。记录复核通过不等于所有优化通过；低残差不等于模量恢复可靠。'
        '原始梯度曲线包含拒绝试探，不用它替代触界情况下的最终投影梯度。</p>',
        '<p><a href="comparison/metrics.json">物理场与指标</a> · '
        '<a href="verification/independent_checks.json">独立正演／梯度复核</a></p>']
    confirmation = root/'confirmation_legacy_index24'
    if (confirmation/'protocol.json').exists():
        parts.append('<h2>原中间点编号说明的校正与确认</h2><p>原协议文字把中间γ写成降序编号24，'
            '实际运行前冻结配置是降序编号25（8.2864e-9）。原结果保留；另在原拟测试的编号24'
            '（1.2068e-8）补做下列同预算对照，不用于选择γ。</p>')
        parts.append('<table><tr><th>方法</th><th>一阶通过</th><th>求解FE</th><th>投影梯度∞范数</th></tr>')
        for method in ('local_projected', 'local_metric'):
            row = pilot.read_json(confirmation/method/'summary.json')
            parts.append(f'<tr><td>{method}</td><td>{row["first_order_passed"]}</td>'
                         f'<td>{row["forward_evaluations"]}</td><td>{row["projected_gradient_inf"]:.3g}</td></tr>')
        parts.append('</table>')
    for i in range(len(cfg.GAMMAS)):
        parts.append(f'<h2>固定γ={cfg.GAMMAS[i]:.6g}</h2>')
        for path in [out/f'gamma_{i:02d}_convergence.png', out/f'gamma_{i:02d}_profiles.png']:
            rel = path.relative_to(root).as_posix()
            parts.append(f'<a href="{Path(rel).with_suffix(".svg").as_posix()}"><img src="{rel}" loading="lazy"></a>')
        parts.append('<details><summary>五组二维模量（同色标）</summary>')
        for method in cfg.METHODS:
            rel = f'comparison/gamma_{i:02d}_{method}.png'
            parts.append(f'<a href="{Path(rel).with_suffix(".svg").as_posix()}"><img src="{rel}" loading="lazy"></a>')
        parts.append('</details>')
    html = '\n'.join(parts)
    for target in re.findall(r'(?:href|src)="([^"]+)"', html):
        if not (root/target).is_file():
            raise FileNotFoundError(root/target)
    page = root/'结果总览.html'; page.write_text(html, encoding='utf-8')
    research = root.parent
    (research/'结果总览.html').write_text('<meta charset="utf-8">'+style+'<h1>统一材料模型研究</h1>'+''.join(
        f'<p><a href="{p.parent.name}/结果总览.html">{p.parent.name}</a></p>'
        for p in sorted(research.glob('*/结果总览.html'))), encoding='utf-8')
    print('保存位置：', page, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='unified_support_pilot_config')
    args = parser.parse_args(); root, cfg = pilot.load(args.config); report(root, cfg)


if __name__ == '__main__':
    main()
