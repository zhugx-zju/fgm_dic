"""Relative differences of saved width-mean moduli from the fixed linear trend.

Post-processing only. Reproduce with:
python -X utf8 -B -m lib.calibration_trend_comparison --relative-errors-only
"""

import html
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .experiment_data import file_sha256

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT/'results/dic_full/calibration_trend_comparison'
COLORS = ('#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00', '#332288')


def trend_values(x, definition):
    p = definition['infill_at_x_zero_percent']+definition['infill_slope_percent_per_mm']*np.asarray(x)
    return definition['intercept_MPa']+definition['slope_MPa_per_infill_percent']*p


def relative_difference(x, modulus, definition):
    """Signed percent difference; denominator is the fixed trend, not the inverse."""
    x, modulus = np.asarray(x, dtype=float), np.asarray(modulus, dtype=float)
    if x.ndim != 1 or x.shape != modulus.shape or not np.isfinite(np.r_[x, modulus]).all():
        raise ValueError('Coordinates and finite moduli must have matching one-dimensional shapes')
    reference = trend_values(x, definition)
    if np.any(reference <= 0):
        raise ValueError('Relative difference requires a strictly positive trend modulus')
    low, high = definition['trend_support_x_mm']
    return np.where((x >= low-1e-9) & (x <= high+1e-9), 100*(modulus-reference)/reference, np.nan)


def sample(curve, x):
    if curve.get('kind') == 'p0':
        ids = np.clip(np.searchsorted(curve['x'], x, side='right')-1, 0, len(curve['mean'])-1)
        return curve['mean'][ids]
    return np.interp(x, curve['x'], curve['mean'])


def evaluation_intervals(curve, definition):
    if (np.ndim(curve['x']) != 1 or np.any(np.diff(curve['x']) <= 0) or
            not np.isfinite(np.r_[curve['x'], curve['mean']]).all()):
        raise ValueError('A saved profile needs finite moduli and strictly increasing x')
    low, high = definition['trend_support_x_mm']
    low, high = max(low, curve['x'][0]), min(high, curve['x'][-1])
    if not low < high:
        raise ValueError('Saved curve does not overlap the calibrated trend range')
    edges = np.unique(np.r_[low, curve['x'][(curve['x'] > low) & (curve['x'] < high)], high])
    # Split zero crossings so the absolute-difference integral stays smooth.
    if curve.get('kind') == 'p0':
        mean = sample(curve, (edges[:-1]+edges[1:])/2)
        left, right = mean-trend_values(edges[:-1], definition), mean-trend_values(edges[1:], definition)
    else:
        difference = sample(curve, edges)-trend_values(edges, definition)
        left, right = difference[:-1], difference[1:]
    ids = np.flatnonzero(left*right < 0)
    roots = edges[ids]-left[ids]*np.diff(edges)[ids]/(right-left)[ids]
    edges = np.unique(np.r_[edges, roots])
    return edges


def difference_summary(curve, definition, order=8):
    """Length-weighted statistics, splitting every material/FE interval exactly."""
    edges = evaluation_intervals(curve, definition)
    z, w = np.polynomial.legendre.leggauss(order)
    half, middle = np.diff(edges)/2, (edges[:-1]+edges[1:])/2
    x = (middle[:, None]+half[:, None]*z).ravel()
    weights = (half[:, None]*w).ravel()
    delta = relative_difference(x, sample(curve, x), definition)
    length = edges[-1]-edges[0]
    return dict(mean_signed_difference_percent=float(weights@delta/length),
                mean_absolute_difference_percent=float(weights@np.abs(delta)/length),
                rms_relative_difference_percent=float(np.sqrt(weights@delta**2/length)),
                support_start_x_mm=float(edges[0]), support_end_x_mm=float(edges[-1]))


def plot_samples(curve, definition):
    edges = evaluation_intervals(curve, definition)
    if curve.get('kind') == 'p0':
        # Keep both values at each jump. Dividing by a varying trend means the
        # relative-difference curve itself is not constant within a P0 cell.
        x = np.concatenate([np.linspace(a, b, 12) for a, b in zip(edges[:-1], edges[1:])])
        mean = np.concatenate([np.full(12, sample(curve, np.array([(a+b)/2]))[0])
                               for a, b in zip(edges[:-1], edges[1:])])
    else:
        x = np.unique(np.r_[np.linspace(edges[0], edges[-1], 1601), edges])
        mean = sample(curve, x)
    return x, mean, relative_difference(x, mean, definition)


def render_group(curves, baseline, definition, stem, title, limits):
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, layout='constrained')
    output_rows, summaries = [], []
    for index, curve in enumerate([baseline, *curves]):
        x, mean, delta = plot_samples(curve, definition)
        color, style = ('#666666', ':') if index == 0 else (COLORS[(index-1) % len(COLORS)], '-')
        if index > 0 and ('367×64' in curve['label'] or '242×38' in curve['label']):
            style = '--'
        for ax, values in zip(axes, (delta, np.abs(delta))):
            ax.plot(x, values, color=color, linestyle=style, linewidth=1.6, label=curve['label'])
        fine = difference_summary(curve, definition, order=16)
        coarse = difference_summary(curve, definition)
        if max(abs(fine[k]-coarse[k]) for k in fine) > 1e-7:
            raise AssertionError('Relative-difference integration needs refinement')
        summaries.append(dict(group=title, method=curve['label'], role=curve.get('role', 'selected_inverse'), **fine))
        output_rows.extend(dict(group=title, method=curve['label'], x_mm=float(a),
                                mean_E_MPa=float(b), trend_E_MPa=float(c),
                                signed_relative_difference_percent=float(d),
                                absolute_relative_difference_percent=float(abs(d)))
                           for a, b, c, d in zip(x, mean, trend_values(x, definition), delta))
    axes[0].axhline(0, color='black', linewidth=.7)
    axes[0].set(ylabel='相对线性趋势的有符号差异 (%)', ylim=limits)
    axes[1].set(xlabel='x (mm)', ylabel='相对线性趋势的绝对差异 (%)',
                xlim=definition['trend_support_x_mm'], ylim=(0, max(abs(limits[0]), abs(limits[1]))))
    axes[0].legend(fontsize=8.5, ncols=2, loc='upper right')
    fig.suptitle(title+'\n100 × [宽度均值 E(x) − 线性趋势] / 线性趋势；正值偏高，负值偏低', fontsize=12)
    for extension in ('png', 'svg'):
        fig.savefig(OUTPUT/f'{stem}.{extension}', dpi=300, bbox_inches='tight')
    plt.close(fig)
    return summaries, output_rows


def build_relative_errors():
    plt.rcParams.update({'font.family': 'Microsoft YaHei', 'font.size': 10,
                         'axes.unicode_minus': False, 'svg.fonttype': 'none'})
    inputs = [OUTPUT/name for name in ('calibration_trend.json', 'calibration_trend_anchors.csv',
                                       'method_width_profiles.csv', 'verification.json')]
    protected = {p: file_sha256(p) for p in inputs}
    definition = json.loads(inputs[0].read_text(encoding='utf-8'))
    anchors = pd.read_csv(inputs[1])
    table = pd.read_csv(inputs[2])
    old_verification = json.loads(inputs[3].read_text(encoding='utf-8'))
    for path, digest in old_verification['source_sha256'].items():
        if file_sha256(ROOT/path) != digest:
            raise ValueError(f'Numerical evidence changed since trend rendering: {path}')
    anchor_x = anchors.trend_anchor_x_mm.to_numpy()
    boundaries = (anchor_x[:-1]+anchor_x[1:])/2
    low, high = definition['trend_support_x_mm']
    baseline = dict(label='分层标定相对线性趋势', role='layered_calibration_reference', kind='p0',
                    x=np.r_[low, boundaries, high], mean=anchors.mean_E_MPa.to_numpy())
    groups = []
    for platform in ('180N', '280N', '380N'):
        curves = [dict(label=label, x=g.x_mm.to_numpy(), mean=g.inverse_width_mean_E_MPa.to_numpy())
                  for label, g in table[table.platform == platform].groupby('method', sort=False)]
        if curves:
            groups.append((platform+'_relative_trend_error', platform+'：各算法，ν=0.26', curves))
    # Same five material/test-space probes as the preceding comparison figure.
    folder = ROOT/'results/dic_full/load380_20261022'
    protocol_path = folder/'spaces/protocol.json'
    space_mesh = None
    if protocol_path.exists():
        protected[protocol_path] = file_sha256(protocol_path)
        protocol = json.loads(protocol_path.read_text(encoding='utf-8'))
        space_mesh = [protocol['nx'], protocol['ny']]
    labels = {'Q1_H1': 'Q1＋H1', 'Q1_TV': 'Q1＋Huber-TV', 'P0_TV': 'P0＋Huber',
              'Q1_H1_PG': 'Q1＋H1＋bubble-PG', 'P0_TV_PG': 'P0＋Huber＋bubble-PG'}
    length = float(table.x_mm.max())
    for platform, name in [('180N', 'spaces_180N_recovered'), ('380N', 'spaces')]:
        curves = []
        for method, label in labels.items():
            directory = folder/name/method
            if not (directory/'summary.json').exists() and platform == '180N':
                directory = folder/'spaces_180N_control'/method
            summary_path, field_path = directory/'summary.json', directory/'selected.npz'
            if not summary_path.exists():
                continue
            protected[summary_path] = file_sha256(summary_path)
            record = json.loads(summary_path.read_text(encoding='utf-8'))
            if record.get('status') != 'selected' or not field_path.exists():
                continue
            protected[field_path] = file_sha256(field_path)
            if 'mesh' not in record and space_mesh is None:
                raise ValueError('Missing frozen material/test-space mesh definition')
            nx, ny = record.get('mesh', space_mesh)
            with np.load(field_path) as saved:
                field = saved['E'].copy()
            if method.startswith('P0'):
                curve = dict(label=label, kind='p0', x=np.linspace(0, length, nx+1),
                             mean=field.reshape(ny, nx).mean(axis=0))
            else:
                values = field.reshape(ny+1, nx+1)
                weights = np.r_[.5, np.ones(ny-1), .5]
                curve = dict(label=label, x=np.linspace(0, length, nx+1),
                             mean=np.average(values, axis=0, weights=weights))
            curves.append(curve)
        if curves:
            groups.append((platform+'_spaces_relative_trend_error', platform+'：42×8材料／测试空间对照', curves))
    values = np.concatenate([plot_samples(c, definition)[2] for _, _, curves in groups for c in [baseline, *curves]])
    limits = (min(-5., float(values.min())-3), max(5., float(values.max())+3))
    summaries, rows = [], []
    section = ['<section id="linear-trend-relative-errors"><h2>相对线性趋势的误差对比</h2>',
               '<p>有符号差异 = 100 × [反演宽度均值 − 线性趋势] / 线性趋势；下图同时给出绝对值。'
               '仅比较原40%—100%标定范围，所有正式对照图共用纵轴。灰色虚线表示实际分层标定自身偏离该直线的程度。'
               '这里的“误差”是对辅助趋势的差异，不是未知实验真值的误差；不同网格、权重和预算不据此作严格同条件排名。'
               '380N仍采用假定净力319.9 N。</p>']
    for stem, title, curves in groups:
        summary, points = render_group(curves, baseline, definition, stem, title, limits)
        summaries.extend(summary)
        rows.extend(points)
        section += [f'<h3>{html.escape(title)}</h3><img src="{stem}.png">',
                    f'<p><a href="{stem}.svg">矢量图 SVG</a></p>']
    stats = pd.DataFrame(summaries)
    stats.to_csv(OUTPUT/'relative_trend_error_summary.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(rows).to_csv(OUTPUT/'relative_trend_error_profiles.csv', index=False, encoding='utf-8-sig')
    section += ['<p>汇总量按相同物理长度加权，逐单元／分段积分；不把不同网格的节点数当作权重。'
                '正负偏差不相互抵消的评价见平均绝对差异和相对RMS。未选γ的统一算法失败诊断未混入此表。</p>',
                '<p><a href="relative_trend_error_summary.csv">长度加权统计 CSV</a> · '
                '<a href="relative_trend_error_profiles.csv">曲线数据 CSV</a></p>',
                stats.rename(columns={'mean_signed_difference_percent': '平均有符号差异(%)',
                                      'mean_absolute_difference_percent': '平均绝对差异(%)',
                                      'rms_relative_difference_percent': '相对RMS(%)'}).to_html(index=False, float_format=lambda v: f'{v:.4f}'),
                '</section>']
    page = OUTPUT/'index.html'
    text = page.read_text(encoding='utf-8')
    marker = '<section id="linear-trend-relative-errors">'
    if marker in text:
        start = text.index(marker)
        end = text.index('</section>', start)+len('</section>')
        text = text[:start]+text[end:]
    position = text.index('<h2>180N</h2>')
    page.write_text(text[:position]+'\n'.join(section)+text[position:], encoding='utf-8')
    assert all(file_sha256(p) == digest for p, digest in protected.items())
    for path, digest in old_verification['source_sha256'].items():
        assert file_sha256(ROOT/path) == digest, path
    verification = dict(definition='100*(width_mean_E-linear_trend_E)/linear_trend_E',
                        source_sha256={str(p.relative_to(ROOT)): digest for p, digest in protected.items()},
                        original_numerical_files_unchanged=len(old_verification['source_sha256']),
                        quadrature_order_8_vs_16='all summary differences below 1e-7 percentage points',
                        plot_groups=len(groups), summary_rows=len(stats), extrapolation=False,
                        python=sys.version.split()[0], numpy=np.__version__, matplotlib=matplotlib.__version__,
                        renderer_sha256=file_sha256(Path(__file__)))
    (OUTPUT/'relative_trend_error_verification.json').write_text(json.dumps(verification, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Relative trend differences: {len(groups)} figures, {len(stats)} summaries; numerical files unchanged.', flush=True)
    return verification
