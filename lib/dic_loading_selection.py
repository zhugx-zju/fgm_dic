"""Loading and DIC frame selection for the three separately identified loads."""

from pathlib import Path
import csv
import hashlib
import json

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from pyxlsb import open_workbook


def plot_loading_selection(project_root: Path, output: Path) -> dict:
    """Use measured records as saved; never extrapolate a force for 380N."""
    root, output = Path(project_root).resolve(), Path(output).resolve()
    sample = root / 'post_tensile_test/1016/sample_2'
    timing_path = sample / 'analysis_results/dic_timing_20261018/timing_checks.json'
    timing = json.loads(timing_path.read_text(encoding='utf-8'))
    record = timing['records']['3']
    hashes = {}

    def digest(path):
        value = hashlib.sha256(path.read_bytes()).hexdigest()
        hashes[str(path)] = value
        return value

    digest(timing_path)
    for name in ('DY_3.csv', 'sample_2_3.xlsb'):
        expected = next(v for k, v in timing['source_sha256'].items() if Path(k).name == name)
        if digest(sample / name) != expected:
            raise ValueError(f'Timing source changed: {name}')
    blocks = []
    for row in csv.reader((sample / 'DY_3.csv').read_text(encoding='gb18030').splitlines()):
        if row and '-DY' in row[0]:
            blocks.append([])
        elif len(row) == 3 and row[0] == '':
            blocks[-1].append([float(row[1]), float(row[2])])
    points = [np.asarray(block) for block in blocks]
    if len(points) != 2:
        raise ValueError('Expected two same-section DIC timing points')
    stages = np.arange(record['n'])
    for point in points:
        np.testing.assert_array_equal(point[:, 0], stages)
    displacement = np.column_stack([point[:, 1] for point in points]).mean(axis=1)
    time = stages / record['nominal_hz'] - record['offset_median_s']
    with open_workbook(str(sample / 'sample_2_3.xlsb')) as book:
        with book.get_sheet(1) as sheet:
            rows = [[cell.v for cell in row] for row in sheet.rows()]
    machine = np.asarray(rows[1:], dtype=float)[:, :2]
    if not np.all(np.diff(machine[:, 0]) > 0):
        raise ValueError('Machine time must increase strictly')
    baseline = float(np.median(machine[machine[:, 0] <= .08, 1]))
    if baseline != record['baseline_force_N'] or baseline != 60.1:
        raise ValueError('Unexpected initial force offset')

    cases = {'180N': 'coarse', '280N': 'h1_280N_independent_coarse_20261022',
             '380N': 'load380_20261022/h1'}
    windows = []
    for name, relative in cases.items():
        path = root / 'results/dic_full' / relative / '01_prepared/run_metadata.json'
        digest(path)
        frozen = json.loads(path.read_text(encoding='utf-8'))
        if frozen['configuration']['FIT_CASES'] != [name]:
            raise ValueError(f'{name}: expected own-platform identification')
        sources = frozen['platforms'][name]['sources']
        indices = [source['stage'] for source in sources]
        if len(indices) != 15 or indices != list(range(indices[0], indices[-1] + 1)):
            raise ValueError(f'{name}: expected 15 consecutive frozen full-field stages')
        for source in sources:
            # Older frozen metadata retains the previous checkout's absolute path.
            relative_source = source['path'].replace('\\', '/').split('/post_tensile_test/', 1)[1]
            if digest(root / 'post_tensile_test' / relative_source) != source['sha256']:
                raise ValueError(f'{name}: full-field source changed')
        lo, hi = indices[0], indices[-1]
        covered = bool(time[lo] >= machine[0, 0] and time[hi] <= machine[-1, 0])
        force = frozen['force_pairing'][name]['mean_corrected_force_N']
        assumed = bool(frozen['configuration'].get('FORCE_ASSUMED', False))
        if covered:
            measured = float(np.interp(time[indices], machine[:, 0], machine[:, 1]).mean() - baseline)
            np.testing.assert_allclose(measured, force, atol=1e-10, rtol=0)
        elif not assumed:
            raise ValueError(f'{name}: uncovered force window must be explicitly assumed')
        windows.append(dict(name=name, first=lo, last=hi, center=(lo+hi)//2,
                            machine_time_s_estimated=time[[lo, hi]].tolist(),
                            force_N=force, force_assumed=assumed, measured_force_coverage=covered))

    colors = ['#0072B2', '#D55E00', '#009E73']
    with plt.rc_context({'font.family': 'Microsoft YaHei', 'font.size': 12,
                         'axes.unicode_minus': False, 'svg.fonttype': 'path'}):
        fig, (force_ax, dic_ax) = plt.subplots(2, 1, figsize=(13.5, 7.6), sharex=True)
        fig.subplots_adjust(left=.095, right=.975, top=.90, bottom=.18, hspace=.32)
        force_ax.plot(machine[:, 0], machine[:, 1], color='#292929', lw=1.7)
        nonnegative = time >= 0
        dic_ax.plot(time[nonnegative], displacement[nonnegative], color='#4D4D4D', lw=1.3)
        force_ax.set(ylim=(40, 440), yticks=[80, 180, 280, 380], ylabel='试验机力示值 (N)')
        dic_ax.set(xlim=(0, time[-1]), ylim=(-.03, displacement[nonnegative].max()+.18),
                   xticks=np.arange(0, time[-1]+.1, 20), xlabel='估计试验机时间 (s)',
                   ylabel='两点平均轴向位移 (mm)')
        for ax, title in [(force_ax, '(a)  加载过程'), (dic_ax, '(b)  DIC 时刻选取')]:
            ax.set_title(title, loc='left', pad=13, fontweight='bold', fontsize=15)
            ax.spines[['top', 'right']].set_visible(False)
        for level in (80, 180, 280):
            k1, k2 = record['machine_plateaus'][str(level)]['knots']
            force_ax.annotate('', xy=(k1, level+47), xytext=(k2, level+47),
                              arrowprops={'arrowstyle': '|-|', 'color': '.45', 'lw': .8})
            force_ax.text((k1+k2)/2, level+59, '保载 10 s', ha='center', fontsize=10, color='.35')
            force_ax.text((k1+k2)/2 if level == 80 else k1-1.3,
                          level+9 if level == 80 else level-19, f'{level} N',
                          ha='center' if level == 80 else 'right',
                          va='bottom' if level == 80 else 'top', fontsize=11)
        legend = []
        for window, color in zip(windows, colors):
            lo, hi, center = [window[key] for key in ('first', 'last', 'center')]
            for ax in (force_ax, dic_ax):
                ax.axvspan(time[lo], time[hi], color=color, alpha=.14, lw=0)
                for edge in (lo, hi):
                    ax.axvline(time[edge], color=color, lw=.75, alpha=.65, ls=(0, (3, 3)))
            if window['measured_force_coverage']:
                force_ax.plot(time[lo:hi+1], np.interp(time[lo:hi+1], machine[:, 0], machine[:, 1]),
                              color=color, lw=3)
            else:
                force_ax.text(time[center], 300, '380 N 选帧窗口\n无同步实测力',
                              ha='center', color=color, fontsize=10)
            dic_ax.plot(time[lo:hi+1], displacement[lo:hi+1], color=color, lw=2.1)
            dic_ax.scatter(time[center], displacement[center], s=37, marker='D', color=color,
                           edgecolors='white', linewidths=.7, zorder=5)
            dic_ax.annotate(f'第 {lo}–{hi} 帧（15 帧平均）',
                            xy=(time[center], displacement[center]),
                            xytext=(time[center]-12, displacement[center]+.16),
                            ha='center', color=color, fontsize=11,
                            arrowprops={'arrowstyle': '-', 'color': color, 'lw': .9})
            legend.append(Line2D([0], [0], color=color, lw=3,
                                 label=f'{window["name"][:-1]} N · 独立识别'))
        force_ax.legend(handles=legend, frameon=False, loc='upper left',
                        bbox_to_anchor=(.015, 1.03), ncol=3, fontsize=11)
        force_ax.text(.99, .97, '加载速率约 4 N/s', transform=force_ax.transAxes,
                      ha='right', va='top', fontsize=11, color='.3')
        force_ax.annotate('实测力记录终点', xy=machine[-1], xytext=(104, 400),
                          fontsize=10, color='.35', ha='center',
                          arrowprops={'arrowstyle': '-', 'color': '.45', 'lw': .8})
        fig.text(.095, .095, '各平台采用15帧平均场：180 N、280 N、380 N均用于各自独立反演。', fontsize=11)
        fig.text(.095, .055, '试样净拉力：180 N平台 119.78 N；280 N平台 219.74 N；380 N平台假定为 319.9 N。', fontsize=10.5)
        fig.text(.095, .018, '时间映射沿用既有事件同步估计；两点共同位移用于选时，不代表试样伸长。', fontsize=10, color='.35')
        output.mkdir(parents=True, exist_ok=True)
        stem = output / 'loading_dic_selection_zh'
        fig.savefig(stem.with_suffix('.png'), dpi=400)
        fig.savefig(stem.with_suffix('.svg'))
        plt.close(fig)
    if not all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == h for p, h in hashes.items()):
        raise ValueError('An input changed during plotting')
    result = dict(run=3, role='separate identification for all three loads', windows=windows,
                  frequency_Hz=record['nominal_hz'], time_offset_s=record['offset_median_s'],
                  measured_force_end_s=float(machine[-1, 0]), source_sha256=hashes,
                  full_field_source_hashes_verified=45, force_extrapolated=False)
    (output / 'figure_metadata.json').write_text(
        json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    return result
