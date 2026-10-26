"""Select full-field export frames from read-only 1026 60sample source records.

Run from the repository root with Python providing numpy, matplotlib and pyxlsb.
All outputs are exclusive creations; do not overwrite an earlier analysis.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pyxlsb import open_workbook

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
FPS = 5.0  # User confirmed 5 Hz and stage numbering starting at zero.
SELECTED = np.arange(235, 250, dtype=int)


def read_run(run: int) -> dict:
    folder = ROOT / str(run)
    with open_workbook(str(folder / 'force.xlsb')) as book:
        assert len(book.sheets) == 1
        with book.get_sheet(1) as sheet:
            raw = [[c.v for c in row] for row in sheet.rows()]
    force = np.asarray(raw[1:], dtype=float)
    assert force.ndim == 2 and force.shape[1] == 4
    assert np.isfinite(force).all() and np.all(np.diff(force[:, 0]) > 0)
    rows = list(csv.reader((folder / 'disp.csv').read_text(encoding='gb18030').splitlines()))
    blocks = []
    for source_line, row in enumerate(rows[3:], start=4):
        if len(row) < 3:
            continue
        try:
            t, u = float(row[1]), float(row[2])
        except ValueError:
            blocks.append({'label': row[0], 'rows': []})
            continue
        blocks[-1]['rows'].append([len(blocks[-1]['rows']), t, u, source_line])
    assert len(blocks) == 2
    a, b = (np.asarray(x['rows'], dtype=float) for x in blocks)
    assert np.array_equal(a[:, :2], b[:, :2])
    assert np.isfinite(a).all() and np.isfinite(b).all()
    return {'force': force, 'dic': [a, b], 'labels': [x['label'] for x in blocks],
            'time': a[:, 0] / FPS, 'mean_dy': (a[:, 2] + b[:, 2]) / 2}


def breakpoint(t: np.ndarray, y: np.ndarray, interval: tuple, search: tuple) -> dict:
    """Least-squares continuous broken line: intercept + slope*t + change*(t-k)+."""
    use = (t >= interval[0]) & (t <= interval[1])
    t, y = t[use], y[use]
    best = None
    for k in np.arange(search[0], search[1] + .0001, .01):
        design = np.column_stack((np.ones(len(t)), t-k, np.maximum(t-k, 0)))
        coefficients = np.linalg.lstsq(design, y, rcond=None)[0]
        sse = float(np.sum((y-design @ coefficients)**2))
        if best is None or sse < best['sse']:
            best = {'time_s': float(k), 'sse': sse,
                    'slope_before_mm_s': float(coefficients[1]),
                    'slope_after_mm_s': float(coefficients[1]+coefficients[2])}
    return best


def main() -> None:
    targets = ['selection.json', 'selected_frames.csv', 'analysis_report.md',
               'max_load_selection.png', 'max_load_selection.svg']
    if any((OUT / name).exists() for name in targets):
        raise FileExistsError('Existing output: use a new analysis directory / RUN_NAME.')
    runs = {run: read_run(run) for run in (1, 2, 3)}
    run = runs[3]
    force = run['force']
    machine_knot = breakpoint(force[:, 0], force[:, 2], (35, 47), (39, 43))
    point_knots = [breakpoint(run['time'], a[:, 2], (38, 53), (42, 47))
                   for a in run['dic']]
    dic_knot = breakpoint(run['time'], run['mean_dy'], (38, 53), (42, 47))
    offset = round(machine_knot['time_s']-dic_knot['time_s'], 2)
    # Sensitivity is an empirical range, not a confidence interval or trigger accuracy.
    offsets = []
    for machine_start in (33, 35, 37):
        mk = breakpoint(force[:, 0], force[:, 2], (machine_start, 47.7), (39, 43))
        for dic_start in (35, 38, 40):
            for dic_end in (50, 52, 54.6):
                for y in (run['dic'][0][:, 2], run['dic'][1][:, 2], run['mean_dy']):
                    dk = breakpoint(run['time'], y, (dic_start, dic_end), (42, 47))
                    offsets.append(mk['time_s']-dk['time_s'])
    sensitivity = [float(min(offsets)), float(max(offsets))]
    paired_time = SELECTED / FPS + offset
    for shift in (offset, *sensitivity):
        assert np.all(SELECTED / FPS + shift >= force[0, 0])
        assert np.all(SELECTED / FPS + shift <= force[-1, 0])
    paired_force = np.interp(paired_time, force[:, 0], force[:, 1])
    peaks = {}
    for number, data in runs.items():
        i = int(np.argmax(data['force'][:, 1]))
        peaks[number] = {'raw_peak_N': float(data['force'][i, 1]),
                         'machine_time_s': float(data['force'][i, 0]),
                         'workbook_row': i+2, 'dic_last_stage': len(data['time'])-1,
                         'dic_end_camera_s': float(data['time'][-1])}
    peak_stage_float = (peaks[3]['machine_time_s'] - offset)*FPS
    sources = []
    for number in runs:
        for name in ('force.xlsb', 'disp.csv'):
            path = ROOT / str(number) / name
            sources.append({'relative_path': f'{number}/{name}',
                            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    summary = {
        'scope': 'Only the highest raw recorded load level across these three runs; no modulus estimation.',
        'source_root': str(ROOT), 'sources': sources,
        'fps_user_confirmed': FPS, 'first_stage_user_confirmed': 0,
        'dic_export_time_note': 'Runs 1/2 time column is row index in seconds; use stage/5. Run 3 time column agrees within 0.0031 s.',
        'machine_time_equals_camera_time_plus_s': offset,
        'machine_knot': machine_knot, 'dic_mean_knot': dic_knot,
        'dic_point_knots': point_knots, 'offset_sensitivity_s': sensitivity,
        'sensitivity_fit_count': len(offsets), 'peaks': peaks,
        'selected_run': 3, 'stage_first': int(SELECTED[0]), 'stage_last': int(SELECTED[-1]),
        'frame_count': len(SELECTED), 'camera_time_s': [float(SELECTED[0]/FPS), float(SELECTED[-1]/FPS)],
        'paired_machine_time_s': paired_time[[0, -1]].tolist(),
        'raw_force_mean_N': float(paired_force.mean()),
        'raw_force_sd_N': float(paired_force.std(ddof=1)),
        'raw_force_min_N': float(paired_force.min()), 'raw_force_max_N': float(paired_force.max()),
        'full_force_coverage_at_nominal_and_sensitivity_offsets': True,
        'instantaneous_peak_approximate_stage': float(peak_stage_float),
        'reference_stage': 0,
        'reference_force_status': 'Not covered by force record after timing alignment; do not assign zero load or subtract first recorded force automatically.',
        'versions': {'python': platform.python_version(), **{name: importlib.metadata.version(name) for name in ('numpy', 'matplotlib', 'pyxlsb')}},
        'python_executable': sys.executable,
    }
    # Independent peak calculation on original Python lists, independent of numpy argmax.
    with open_workbook(str(ROOT / '3/force.xlsb')) as book:
        with book.get_sheet(1) as sheet:
            numeric_rows = [[c.v for c in row] for row in sheet.rows()][1:]
    assert max(row[1] for row in numeric_rows) == peaks[3]['raw_peak_N']
    assert abs(run['dic'][0][:, 1]-run['time']).max() <= .0031
    with (OUT / 'selected_frames.csv').open('x', newline='', encoding='utf-8-sig') as handle:
        writer = csv.writer(handle, lineterminator='\n')
        writer.writerow(['run', 'stage', 'expected_filename', 'camera_time_s', 'machine_time_s',
                         'raw_force_N', 'point_0_DY_mm', 'point_1_DY_mm', 'point_0_source_line', 'point_1_source_line'])
        for stage, mt, load in zip(SELECTED, paired_time, paired_force):
            a, b = (block[stage] for block in run['dic'])
            writer.writerow([3, stage, f'stage_{stage:05d}.comdat_comdata.csv', stage/FPS,
                             f'{mt:.3f}', f'{load:.6f}', a[2], b[2], int(a[3]), int(b[3])])
    (OUT / 'selection.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 1, figsize=(9, 6.4), sharex=True)
    ax = axes[0]
    ax.plot(force[:, 0], force[:, 1], color='#0072B2', lw=1.5, label='Recorded load')
    ax.scatter([peaks[3]['machine_time_s']], [peaks[3]['raw_peak_N']], s=24, color='#D55E00', label='Instantaneous peak')
    ax.set_ylabel('Raw force (N)'); ax.legend(loc='upper left', frameon=False)
    ax.text(.98, .07, f"Peak: {peaks[3]['raw_peak_N']:.3f} N at {peaks[3]['machine_time_s']:.2f} s", transform=ax.transAxes, ha='right')
    ax = axes[1]
    for block, label, color in zip(run['dic'], run['labels'], ('#0072B2', '#D55E00')):
        label_en = 'Point 0 DY' if '0-DY' in label else 'Point 1 DY'
        ax.plot(run['time']+offset, block[:, 2], lw=1.1, color=color, label=label_en)
    ax.axvspan(force[-1, 0], run['time'][-1]+offset, color='.85', alpha=.65, label='No recorded force')
    ax.set_ylabel('DIC point DY (mm)'); ax.set_xlabel('Machine time (s), aligned from 5 Hz stage index')
    ax.legend(loc='upper left', frameon=False)
    for ax in axes:
        ax.axvspan(paired_time[0], paired_time[-1], color='#009E73', alpha=.2)
        ax.axvline(machine_knot['time_s'], color='.5', ls='--', lw=.8)
        ax.set_xlim(-4, 52)
    axes[0].annotate('Export stages 235-249', xy=(paired_time.mean(), paired_force.mean()),
                     xytext=(25, 325), arrowprops={'arrowstyle': '->', 'color': '#009E73'}, color='#007D5B')
    fig.suptitle('1026 uniform 60% specimen, run 3: maximum-load frame selection', fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / 'max_load_selection.png', dpi=220)
    fig.savefig(OUT / 'max_load_selection.svg')
    svg = OUT / 'max_load_selection.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text(encoding='utf-8').splitlines())+'\n', encoding='utf-8')
    plt.close(fig)
    report = f'''# 1026 均匀 60% 试样：最大载荷全场帧提取

建议只导出第 3 次加载的 stage_00235 至 stage_00249（含两端，共 15 帧），并保留 stage_00000 参考图及参考坐标。目标是本组三次记录中的最高载荷平台，约 380 N。实际全场文件尚未提供，以下为导出清单，不代表已验证全场相关质量。

| 记录 | 原始最大载荷 N | 加载记录时刻 s | 最大载荷选择结论 |
|---|---:|---:|---|
| 1 | {peaks[1]['raw_peak_N']:.5f} | {peaks[1]['machine_time_s']:.2f} | DIC 仅到第 799 帧、相机 159.8 s；约 280 N 的最后一次停留结束转折发生在加载约 140.85 s、DIC 约 144.88 s。随后 DIC 约在加载 155.8 s 结束，早于 179.72 s 的峰值，缺少 380 N 全场。此局部配对仅用于覆盖判断。 |
| 2 | {peaks[2]['raw_peak_N']:.5f} | {peaks[2]['machine_time_s']:.2f} | 最高仅约 280 N，不属于本组最高载荷 380 N。 |
| 3 | {peaks[3]['raw_peak_N']:.5f} | {peaks[3]['machine_time_s']:.2f} | 含 380 N 停留平台，选用。 |

## 时间与帧号

- 用户确认三次采集均为 5 Hz，首帧编号为 0。因此相机时间为 stage/5。第 1、2 次 CSV 的时间列逐行增加 1，不能直接作为真实秒数；第 3 次时间列与 stage/5 的差小于 0.0031 s。所有文件逐帧连续对应仍需在导出软件中确认，CSV 本身没有显式图像编号。
- 每份 disp.csv 实际包含两个 DY 测点。本分析以两测点均值辅助同步，同时分别检查两点，不将单点位移视为试样应变。
- 第 3 次用试验机位移和 DIC 均值分别拟合连续折线，确定最后一次加载转入停留的时刻。试验机转折 {machine_knot['time_s']:.2f} s，DIC 转折 {dic_knot['time_s']:.2f} s，故加载时间 ≈ stage/5 {offset:+.2f} s。两点单独给出偏移约 −3.37、−3.51 s。
- 改变局部拟合区间并分别使用两点与均值，共 {len(offsets)} 组，偏移范围 {sensitivity[0]:.2f} 至 {sensitivity[1]:.2f} s。这是方法敏感性范围，不是触发器测得的同步精度或统计置信区间。仅有一次最终转折，未独立识别时钟漂移。

## 推荐窗口

- stage_00235 至 stage_00249：相机 47.00–49.80 s；配对加载时刻 {paired_time[0]:.2f}–{paired_time[-1]:.2f} s。
- 每帧插值原始载荷均值 {paired_force.mean():.3f} N，样本标准差 {paired_force.std(ddof=1):.3f} N，范围 {paired_force.min():.3f}–{paired_force.max():.3f} N。不扣除偏置。这里均值是 15 个配对时刻的平均，不是全部 50 Hz 载荷样本均值。
- 在全部上述偏移范围内，15 帧均位于已记录的最大载荷平台，载荷时间覆盖完整。选平台中段给同步误差和起始过渡留出余量；窗口长度为预先选定的 15 帧，不以模量标定或反演误差选帧。
- 若仅要瞬时峰值附近单帧，380.008 N 出现在加载 41.10 s，映射为约第 {peak_stage_float:.1f} 帧，即 stage_00223 附近。同步敏感性对应约 222–224 帧，无法保证某一相机帧恰好包含 50 Hz 载荷记录的瞬时最大值。用于同载荷平均时优先采用上述 235–249 帧。
- 导出各帧原始 X、Y、DX、DY、有效性／相关质量字段并保持同一 ROI、参考图、坐标尺度与计算参数。stage_00000 用作共同参考；若软件的位移已相对此图计算，无需将参考帧再次计入平台平均。

## 适用边界与核验

第 3 次载荷记录起点已为 250.725 N，相机参考帧按拟合偏移早于加载记录起点，其真实载荷未被这份文件覆盖。380 N 是绝对原始记录平台值；若后续用相对参考图的位移作增量力学识别，必须核实参考载荷，不能默认参考图为零载荷，也不能直接把首条 250.725 N 当作参考载荷扣除。

已检查两测点行数及时间一致性、加载时间单调性、有限数值、窗口在名义偏移及敏感性两端的完整覆盖；另用原始列表 max 独立复核第 3 次峰值。未检查全场相关质量、坏点、相机漏帧或材料线弹性。此分析不估计模量，也未启动反演。

复现：在项目根目录运行 `python post_tensile_test/1026/60sample/analysis_results/max_load_frames_20261026/analyze_frames.py`。已有结果拒绝覆盖；复跑须复制脚本到新的分析目录。Python 为 {platform.python_version()}；版本、六个源文件 SHA256、转折拟合参数和配对汇总见 selection.json；逐帧清单见 selected_frames.csv。图为同一最终状态的 PNG 与 SVG。
'''
    (OUT / 'analysis_report.md').write_text(report, encoding='utf-8')
    print(json.dumps({k: summary[k] for k in ('stage_first', 'stage_last', 'machine_time_equals_camera_time_plus_s', 'raw_force_mean_N', 'raw_force_sd_N', 'raw_force_min_N', 'raw_force_max_N')}, indent=2))


if __name__ == '__main__':
    main()
