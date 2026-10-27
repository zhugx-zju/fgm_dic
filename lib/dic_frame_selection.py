"""Read-only DIC timing analysis and the project's two-panel selection figures.

Times are seconds, displacement is mm and raw machine force is N. Stage numbers
are row indices, conditional on a confirmed zero-based, uninterrupted acquisition.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pyxlsb import open_workbook


@dataclass
class TimingData:
    machine: np.ndarray  # columns: time_s, raw_force_N, crosshead_mm, extension_mm
    points: np.ndarray  # axes: exported point, stage, [CSV time, DY_mm]
    point_labels: list[str]
    camera_time_s: np.ndarray
    fps: float

    @property
    def mean_dy_mm(self) -> np.ndarray:
        return self.points[:, :, 1].mean(axis=0)


def read_timing(folder: Path, fps: float, *, force_file='force.xlsb', displacement_file='disp.csv') -> TimingData:
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError('fps must be a confirmed positive sampling frequency.')
    force_path = folder / force_file
    if force_path.suffix.lower() == '.csv':
        import pandas as pd
        machine = pd.read_csv(force_path)[['time_s', 'force_N', 'displacement_mm', 'extension_mm']].to_numpy(float)
    else:
        with open_workbook(str(force_path)) as book:
            if len(book.sheets) != 1:
                raise ValueError(f'{folder}: expected one machine-data sheet.')
            with book.get_sheet(1) as sheet:
                rows = [[cell.v for cell in row] for row in sheet.rows()]
        machine = np.asarray(rows[1:], dtype=float)
    if machine.ndim != 2 or machine.shape[1] != 4 or not np.isfinite(machine).all():
        raise ValueError(f'{folder}: expected finite time/force/displacement/extension columns.')
    if not np.all(np.diff(machine[:, 0]) > 0):
        raise ValueError(f'{folder}: machine time is not strictly increasing.')
    displacement_path = folder / displacement_file
    header = displacement_path.read_text(encoding='utf-8-sig', errors='replace').splitlines()[0]
    if header.startswith('stage,'):
        import pandas as pd
        table = pd.read_csv(displacement_path)
        stages = table.stage.to_numpy(float)
        if len(stages) < 8 or not np.array_equal(stages, np.arange(len(stages))):
            raise ValueError('DIC stage must start at zero with no missing or duplicate frames')
        labels = ['point1-DY', 'point2-DY']
        points = np.stack([np.column_stack((stages, table[name]))
                           for name in ('point1_DY_mm', 'point2_DY_mm')])
        if not np.isfinite(points).all():
            raise ValueError('DIC timing data contain nonfinite values')
        return TimingData(machine, points, labels, stages/fps, fps)
    labels, blocks = [], []
    for row in csv.reader(displacement_path.read_text(encoding='gb18030').splitlines()):
        if row and '-DY' in row[0]:
            labels.append(row[0])
            blocks.append([])
        elif len(row) == 3 and row[0] == '':
            if not blocks:
                raise ValueError(f'{folder}: DIC rows precede their point label.')
            blocks[-1].append([float(row[1]), float(row[2])])
    if len(blocks) != 2 or len(blocks[0]) != len(blocks[1]):
        raise ValueError(f'{folder}: expected two equally long DY records.')
    points = np.asarray(blocks, dtype=float)
    if not np.isfinite(points).all() or not np.array_equal(points[0, :, 0], points[1, :, 0]):
        raise ValueError(f'{folder}: DIC points have missing values or unmatched time columns.')
    if not np.all(np.diff(points[0, :, 0]) > 0):
        raise ValueError(f'{folder}: DIC export time must be strictly increasing.')
    if not np.array_equal(points[0, :, 0], np.arange(points.shape[1])):
        raise ValueError('Legacy DIC time column must be confirmed zero-based consecutive stage numbers')
    return TimingData(machine, points, labels, np.arange(points.shape[1])/fps, fps)


def fit_transition(t: np.ndarray, y: np.ndarray, interval: tuple[float, float],
                   search: tuple[float, float], *, allow_jump: bool = False) -> dict:
    """Local two-line least squares; optional jump handles a DIC transient.

    Continuous mode searches knots at 0.01 s. Jump mode searches midpoints of
    observations because its residual is constant within an observation gap.
    The latter avoids shifting the estimated event to conceal a displacement jump.
    """
    keep = (t >= interval[0]) & (t <= interval[1])
    t, y = np.asarray(t)[keep], np.asarray(y)[keep]
    if len(t) < 8 or not (interval[0] <= search[0] < search[1] <= interval[1]):
        raise ValueError('Transition fitting requires at least eight samples and an interior search interval.')
    candidates = (t[:-1]+t[1:])/2 if allow_jump else np.arange(search[0], search[1]+.0001, .01)
    candidates = candidates[(candidates >= search[0]) & (candidates <= search[1])]
    best = None
    for knot in candidates:
        if (t < knot).sum() < 3 or (t > knot).sum() < 3:
            continue
        dt = t-knot
        design = (np.column_stack((np.ones(len(t)), dt, (t > knot), np.maximum(dt, 0)))
                  if allow_jump else np.column_stack((np.ones(len(t)), dt, np.maximum(dt, 0))))
        beta = np.linalg.lstsq(design, y, rcond=None)[0]
        sse = float(np.sum((y-design @ beta)**2))
        if best is None or sse < best['sse']:
            best = {'time_s': float(knot), 'sse': sse, 'sample_count': len(t),
                    'slope_before_mm_s': float(beta[1]), 'slope_after_mm_s': float(beta[1]+beta[-1]),
                    'jump_mm': float(beta[2]) if allow_jump else 0.0,
                    'model': 'independent_two_lines' if allow_jump else 'continuous_broken_line',
                    'fit_interval_s': list(interval), 'search_interval_s': list(search)}
    if best is None:
        raise ValueError('No valid transition candidate with three observations on each side.')
    return best


def pair_force(data: TimingData, machine_time_s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    t = np.asarray(machine_time_s)
    covered = (t >= data.machine[0, 0]) & (t <= data.machine[-1, 0])
    force = np.full(t.shape, np.nan)
    force[covered] = np.interp(t[covered], data.machine[:, 0], data.machine[:, 1])
    return force, covered


def draw_selection(data: TimingData, record: dict, output: Path, language: str,
                   specimen_name: str, force_limits: tuple[float, float]) -> None:
    """Upper force / lower mean DY; no force extrapolation into a missing tail."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    def tr(en, zh):
        return zh if language == 'zh' else en
    plt.rcParams.update({
        'font.family': 'Microsoft YaHei' if language == 'zh' else 'Arial', 'font.size': 12,
        'axes.labelsize': 13, 'axes.titlesize': 14, 'axes.linewidth': .9,
        'xtick.direction': 'out', 'ytick.direction': 'out', 'svg.fonttype': 'path',
        'axes.unicode_minus': False, 'figure.facecolor': 'white', 'savefig.facecolor': 'white',
    })
    f, mean = data.machine, data.mean_dy_mm
    t = data.camera_time_s+record['machine_time_offset_s']
    lo, hi = record['stage_first'], record['stage_last']
    center = (lo+hi)//2
    forces, covered = pair_force(data, t)
    complete = bool(covered[lo:hi+1].all())
    color = '#0072B2' if complete else '#D55E00'
    end = max(float(t[-1]), float(f[-1, 0]))
    span = float(np.ptp(mean))
    fig, (force_ax, dic_ax) = plt.subplots(2, 1, figsize=(12, 6.75), sharex=True)
    fig.subplots_adjust(left=.10, right=.975, top=.855, bottom=.235, hspace=.32)
    fig.suptitle(tr(f'{specimen_name} · Run {record["run"]}', f'{specimen_name} 试样 · 第 {record["run"]} 次加载'),
                 x=.10, y=.97, ha='left', fontsize=16, fontweight='bold')
    force_ax.plot(f[:, 0], f[:, 1], color='#292929', lw=1.7)
    dic_ax.plot(t, mean, color='#4D4D4D', lw=1.35)
    force_ax.set_ylim(*force_limits)
    force_ax.set_ylabel(tr('Machine force reading (N)', '试验机力示值 (N)'))
    dic_ax.set_ylabel(tr('Mean axial displacement (mm)', '两点平均轴向位移 (mm)'))
    dic_ax.set_xlabel(tr('Estimated machine time (s)', '估计试验机时间 (s)'))
    dic_ax.set_ylim(min(-.02, float(mean.min())-.025*span), float(mean.max())+.37*span+.025)
    dic_ax.set_xlim(0, end*1.025)
    dic_ax.set_xticks(np.arange(0, end*1.025+.01, 20))
    for ax, title in ((force_ax, tr('(a)  Loading history', '(a)  加载过程')),
                      (dic_ax, tr('(b)  DIC frame selection', '(b)  DIC 时刻选取'))):
        ax.set_title(title, loc='left', pad=11, fontweight='bold')
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(labelsize=11)
        ax.axvspan(t[lo], t[hi], color=color, alpha=.14, lw=0)
        for edge in (lo, hi):
            ax.axvline(t[edge], color=color, lw=.75, alpha=.65, ls=(0, (3, 3)))
        if end > f[-1, 0]:
            ax.axvspan(f[-1, 0], end, facecolor='.96', hatch='///', edgecolor='.8', lw=0, zorder=0)
    force_ax.tick_params(labelbottom=False)
    mask = covered[lo:hi+1]
    if mask.any():
        force_ax.plot(t[lo:hi+1][mask], forces[lo:hi+1][mask], color=color, lw=3)
    dic_ax.plot(t[lo:hi+1], mean[lo:hi+1], color=color, lw=2.2)
    dic_ax.scatter(t[center], mean[center], marker='D', s=38, color=color, edgecolors='white', linewidths=.7, zorder=5)
    role = tr('Maximum-load hold; complete force record', '最大载荷平台；载荷记录完整') if complete else tr(
        'Maximum-load end-hold candidate; force unrecorded', '最大载荷末端候选；同期载荷缺失')
    force_ax.legend(handles=[Line2D([0], [0], color=color, lw=3, label=role)],
                    loc='upper left', frameon=False, fontsize=11, handlelength=1.8)
    peak = record['raw_peak_N']
    peak_t = record['peak_machine_time_s']
    force_ax.scatter(peak_t, peak, s=20, color='#292929', zorder=5)
    label_y = force_limits[0]+.65*np.ptp(force_limits)
    force_ax.annotate(tr(f'Peak {peak:.3f} N', f'峰值 {peak:.3f} N'),
                      xy=(peak_t, peak), xytext=(peak_t-end*.20, label_y), ha='center', fontsize=10.5,
                      color='#303030', bbox={'facecolor': 'white', 'edgecolor': 'none', 'pad': 1.5},
                      arrowprops={'arrowstyle': '-', 'color': '.45', 'lw': .8})
    if not complete:
        force_ax.text((f[-1, 0]+end)/2, force_limits[0]+.25*np.ptp(force_limits),
                      tr('No recorded\nforce', '同期载荷\n未记录'), ha='center', va='center', fontsize=10.5, color='.4')
    count = hi-lo+1
    label = tr(f'Frames {lo}–{hi}: {count}-frame candidate\nMidpoint frame: {center}; force unverified',
               f'第 {lo}–{hi} 帧：{count} 帧候选\n中间帧：{center}；载荷尚未核实') if not complete else tr(
        f'Frames {lo}–{hi}: {count}-frame window\nSingle-frame option: {center}',
        f'第 {lo}–{hi} 帧：{count} 帧窗口\n可选单帧：{center}')
    dic_ax.annotate(label, xy=(t[center], mean[center]),
                    xytext=(t[center]-end*.15, mean[center]+span*.20+.015),
                    color=color, fontsize=11, ha='center', va='bottom',
                    arrowprops={'arrowstyle': '-', 'color': color, 'lw': .9, 'shrinkA': 4, 'shrinkB': 6})
    rule = record.get('selection_rule', f'6–9 s after hold entry; {data.fps:g} Hz; {count} frames')
    fig.text(.10, .115, tr(f'Selection: {rule}', f'选帧规则：{rule}'), fontsize=10.5, color='#303030')
    fig.text(.10, .070, tr(f'Force record ends at {f[-1, 0]:.2f} s. Hatching marks missing force; no tail extrapolation.',
                          f'载荷记录止于 {f[-1, 0]:.2f} s；斜线区为同期载荷缺失，未外推或填充载荷。'), fontsize=10.5, color='#303030')
    fig.text(.10, .025, tr(f'Zero-based stages. Estimated mapping: t_machine = stage / {data.fps:g} {record["machine_time_offset_s"]:+.2f} s.',
                          f'帧号从 0 开始。估计同步：试验机时间 = 帧号 / {data.fps:g} {record["machine_time_offset_s"]:+.2f} s。'),
             fontsize=10, color='#505050')
    stem = output / f'run{record["run"]}_loading_dic_selection_{language}'
    fig.savefig(stem.with_suffix('.png'), dpi=400)
    fig.savefig(stem.with_suffix('.svg'))
    svg = stem.with_suffix('.svg')
    svg.write_bytes(('\n'.join(line.rstrip() for line in svg.read_text(encoding='utf-8').splitlines())+'\n').encode('utf-8'))
    plt.close(fig)
