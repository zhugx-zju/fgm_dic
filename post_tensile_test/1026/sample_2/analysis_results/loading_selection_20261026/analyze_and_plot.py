"""Maximum-load export requests and established-style plots for sample_2.

User confirmed 5 Hz and stage numbering from zero for all three acquisitions.
Outputs are isolated; source data and previous studies are never overwritten.
"""
from __future__ import annotations

import csv
import hashlib
import json
import platform
import sys
from pathlib import Path

import matplotlib
import numpy as np

OUT = Path(__file__).resolve().parent
SAMPLE = OUT.parents[1]
REPO = SAMPLE.parents[2]
sys.path.insert(0, str(REPO))
from lib.dic_frame_selection import draw_selection, fit_transition, pair_force, read_timing

FPS = 5.0
# Independent middle hold events: machine interval/search, DIC interval/search.
EVENTS = {
    1: [((7, 19), (9, 15), (10, 25), (12, 20)),
        ((17, 29), (19, 25), (20, 34), (24, 30)),
        ((53, 66), (57, 63), (57, 71), (61, 67)),
        ((64, 77), (66, 73), (67, 81), (70, 77))],
    2: [((34, 47), (37, 44), (39, 53), (42, 50)),
        ((45, 58), (48, 55), (49, 64), (53, 60)),
        ((69, 82), (72, 79), (75, 89), (78, 85)),
        ((80, 93), (83, 90), (85, 100), (88, 96))],
    3: [((20, 33), (23, 30), (24, 38), (27, 34)),
        ((31, 44), (34, 41), (35, 49), (38, 45)),
        ((83, 97), (87, 94), (87, 102), (91, 98)),
        ((94, 108), (97, 104), (98, 113), (101, 109))],
}
TAIL = {1: ((105, 125), (109, 119)), 2: ((112, 131), (116, 126)),
        3: ((123, 144), (128, 138))}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if (OUT / 'selection.json').exists() or any(OUT.glob('*.png')):
        raise FileExistsError('Existing output: use a new RUN_NAME/directory.')
    sources = {f'{i}/{name}': digest(SAMPLE / str(i) / name)
               for i in (1, 2, 3) for name in ('disp.csv', 'force.xlsb')}
    records, requests = {}, []
    for number in (1, 2, 3):
        data = read_timing(SAMPLE / str(number), FPS)
        f = data.machine
        assert np.array_equal(data.points[0, :, 0], np.arange(data.points.shape[1]))
        offsets, events = [], []
        for fw, fs, dw, ds in EVENTS[number]:
            machine = fit_transition(f[:, 0], f[:, 2], fw, fs, allow_jump=True)
            dic = [fit_transition(data.camera_time_s, y, dw, ds, allow_jump=True)
                   for y in (data.points[0, :, 1], data.points[1, :, 1], data.mean_dy_mm)]
            event_offsets = [machine['time_s']-r['time_s'] for r in dic]
            offsets.extend(event_offsets)
            events.append({'machine': machine, 'dic_point_blocks_and_mean': dic, 'offsets_s': event_offsets})
        offset = round(float(np.median(offsets)), 2)
        tail = fit_transition(data.camera_time_s, data.mean_dy_mm, *TAIL[number])
        start = int(np.ceil((tail['time_s']+6)*FPS))
        end = start+14
        assert end <= int(np.floor((tail['time_s']+9)*FPS))
        assert end < len(data.camera_time_s)
        assert tail['slope_before_mm_s'] > 0 and abs(tail['slope_after_mm_s']) < .25*tail['slope_before_mm_s']
        # Point/window perturbations audit final entry, without using any material calibration.
        entries = [fit_transition(data.camera_time_s, y,
                                 (TAIL[number][0][0]+shift, TAIL[number][0][1]+shift), TAIL[number][1])['time_s']
                   for shift in (-1, 0, 1) for y in (data.points[0, :, 1], data.points[1, :, 1], data.mean_dy_mm)]
        t = data.camera_time_s+offset
        paired, covered = pair_force(data, t)
        stages = np.arange(start, end+1)
        peak_idx = int(np.argmax(f[:, 1]))
        assert max(row[1] for row in f.tolist()) == f[peak_idx, 1]
        peak_stage_range = [(float(f[peak_idx, 0])-max(offsets))*FPS,
                            (float(f[peak_idx, 0])-min(offsets))*FPS]
        record = {
            'run': number, 'raw_peak_N': float(f[peak_idx, 1]),
            'peak_machine_time_s': float(f[peak_idx, 0]), 'peak_workbook_row': peak_idx+2,
            'peak_is_last_machine_row': peak_idx == len(f)-1,
            'dic_last_stage': len(data.camera_time_s)-1, 'machine_time_offset_s': offset,
            'event_offset_range_s': [float(min(offsets)), float(max(offsets))],
            'events': events, 'tail_fit': tail, 'tail_entry_sensitivity_s': [float(min(entries)), float(max(entries))],
            'stage_first': start, 'stage_last': end, 'frame_count': len(stages),
            'midpoint_stage': (start+end)//2,
            'camera_window_s': data.camera_time_s[stages][[0, -1]].tolist(),
            'estimated_machine_window_s': t[stages][[0, -1]].tolist(),
            'peak_approximate_stage': (float(f[peak_idx, 0])-offset)*FPS,
            'peak_stage_range_from_events': peak_stage_range,
            'force_coverage_count': int(covered[stages].sum()),
            'status': 'candidate_DIC_end_hold_missing_contemporaneous_force',
            'force_mean_N': float(paired[stages].mean()) if covered[stages].all() else None,
            'point_labels': data.point_labels,
        }
        # All requested end holds are beyond the force record at every fitted offset.
        assert np.all(data.camera_time_s[stages]+min(offsets) > f[-1, 0])
        assert record['force_coverage_count'] == 0
        assert data.camera_time_s[start] > max(entries)+4
        records[number] = record
        for stage in stages:
            requests.append([number, stage, f'stage_{stage:05d}.comdat_comdata.csv', data.camera_time_s[stage],
                             t[stage], False, '', 'maximum_load_end_hold_candidate_force_unverified'])
        for language in ('zh', 'en'):
            draw_selection(data, record, OUT, language, 'sample_2', (20, 525))
        with (OUT / f'run{number}_timing_curve_data.csv').open('x', encoding='utf-8-sig', newline='') as handle:
            writer = csv.writer(handle, lineterminator='\n')
            writer.writerow(['stage_0based', 'camera_time_s', 'estimated_machine_time_s',
                             'first_exported_DY_mm', 'second_exported_DY_mm', 'mean_DY_mm',
                             'raw_force_N_if_covered', 'force_record_coverage', 'selected_window'])
            for stage in range(len(t)):
                writer.writerow([stage, data.camera_time_s[stage], t[stage], *data.points[:, stage, 1],
                                 data.mean_dy_mm[stage], paired[stage] if covered[stage] else '',
                                 bool(covered[stage]), start <= stage <= end])
    assert all(digest(SAMPLE / name) == value for name, value in sources.items())
    with (OUT / 'export_requests.csv').open('x', encoding='utf-8-sig', newline='') as handle:
        writer = csv.writer(handle, lineterminator='\n')
        writer.writerow(['run', 'stage', 'expected_filename', 'camera_time_s', 'estimated_machine_time_s',
                         'force_record_covers_frame', 'paired_raw_force_N', 'status'])
        writer.writerows(requests)
    summary = {'specimen': 'sample_2', 'scope': 'Maximum recorded load separately for all three runs.',
               'fps_user_confirmed': FPS, 'first_stage_user_confirmed': 0,
               'selection_rule': 'ceil(5*(final_DIC_hold_entry_s+6)) through start+14; within entry+6 to entry+9 s.',
               'source_sha256': sources, 'runs': records,
               'environment': {'python': platform.python_version(), 'numpy': np.__version__, 'matplotlib': matplotlib.__version__},
               'implementation_sha256': {str(p.relative_to(REPO)): digest(p) for p in
                                         (Path(__file__), REPO/'lib/dic_frame_selection.py')},
               'interpretation': 'DIC end holds are export candidates, not verified constant-force platforms. No force offset subtraction or extrapolation.',
               'synchronization': 'Median of independent intermediate hold entry/exit events; jump-aware local fits accommodate DIC transients. Event ranges are sensitivity, not confidence intervals.',
               'reference': 'Keep stage_00000 independently for each run; reference load not automatically zero.',
               'material_direction_note': (SAMPLE/'说明.txt').read_text(encoding='utf-8').strip(),
               'limitations': ['No full-field quality check before export.',
                               'CSV row-index mapping does not independently establish no dropped camera images.',
                               'Two-point motion is a timing signal, not strain or constant-force evidence.']}
    (OUT / 'selection.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    lines = ['# sample_2：三次最大载荷选帧与图件', '',
             '分别对三次加载选取各自最大载荷后的 DIC 末端停留候选。用户确认均为 5 Hz、首帧为 0。每次取拟合入口后 6–9 s 内的连续 15 帧。', '',
             '| 次数 | 原始峰值 N | 估计峰值帧 | 推荐候选帧（含两端） | 中间帧 | 同期载荷覆盖 |',
             '|---|---:|---:|---|---:|---|']
    for number, r in records.items():
        lines.append(f'| {number} | {r["raw_peak_N"]:.5f} | {r["peak_approximate_stage"]:.1f} | {r["stage_first"]}–{r["stage_last"]} | {r["midpoint_stage"]} | 0/15；待补载荷 |')
    lines += ['', '三次原始载荷峰值都在文件最后一行，DIC 则继续记录了后续停留。候选窗口全部超出同期载荷记录，并且在全部拟合事件偏移范围内均如此。因此可先导出全场，不能将 280/380/480 N 或末条力值自动填充为窗口载荷；不得作为已完成力配对的正式识别输入。', '',
              '## 同步与选帧依据', '',
              'CSV 的时间列为逐行 0、1、2…，按确认的采集频率使用 stage/5 换为真实秒数。同步只用本次各自中间停留的进入／退出事件；试验机位移和 DIC 两点及均值分别作局部双直线拟合，取全部事件与测点的偏移中位数。允许两段之间有跳变，避免把 DIC 停留附近的突降／突升硬拟合成时间漂移。每次均独立估计，没有继承 60sample 的偏移。', '']
    for number, r in records.items():
        lines.append(f'- 第 {number} 次：t_machine = stage/5 {r["machine_time_offset_s"]:+.2f} s；事件偏移 {r["event_offset_range_s"][0]:.2f} 至 {r["event_offset_range_s"][1]:.2f} s。最终 DIC 停留入口 {r["tail_fit"]["time_s"]:.2f} s，换测点与拟合区间后为 {r["tail_entry_sensitivity_s"][0]:.2f}–{r["tail_entry_sensitivity_s"][1]:.2f} s；候选相机时段 {r["camera_window_s"][0]:.2f}–{r["camera_window_s"][1]:.2f} s，估计加载时段 {r["estimated_machine_window_s"][0]:.2f}–{r["estimated_machine_window_s"][1]:.2f} s。')
    lines += ['', '事件范围是方法敏感性，不是硬件同步精度或统计置信区间。估计瞬时峰值附近帧同样存在该不确定性；不保证相机某帧恰好捕获 50 Hz 载荷记录瞬时最大值。最大载荷末端候选采用 DIC 自身转折选帧，均位于全部扰动入口至少 4 s 之后；相对平缓的两点位移不能证明全场稳定或载荷恒定。', '',
              '## 图件、核验与复现', '',
              '沿用项目两幅上下图版式：上幅为原始试验机力示值，下幅为两点平均 DY；浅橙阴影、虚线、加粗线段与菱形标出候选窗口和中间帧，灰斜线区标明载荷记录之外的时段。三次力轴共用 20–525 N；位移轴按各次量级设置。中文与英文分别输出 PNG（4800×2700、400 dpi）及同一 figure 的 SVG，文字转路径。', '',
              '保留每次各自的 stage_00000 参考图与坐标；导出 X、Y、DX、DY 及有效性／相关质量字段，并保持同一 ROI、参考图与尺度。原说明“填充率大的在位移大的那头”仅记录材料方向提示，未据此设定坐标、界面或模量。参考图载荷须核实，不默认零载荷，不自动扣除首条力值。', '',
              '已核对两测点长度与时间、有限数值、时间单调性、三次峰值的独立列表 max、45 帧连续性与载荷覆盖、入口的测点／区间敏感性，以及六个原始输入在分析前后的 SHA256。未修改原始文件，未启动反演，尚未检查待导出的全场相关质量。', '',
              '- `export_requests.csv`：45 帧导出清单；未覆盖载荷留空。',
              '- `selection.json`：全部拟合区间、事件、转折、偏移、输入哈希、版本与限制。',
              '- `run1/2/3_timing_curve_data.csv`：原始两点、均值、估计机器时间、已覆盖的插值载荷及选帧标记。',
              '- `run1/2/3_loading_dic_selection_zh/en.png/.svg`：三次中文／英文图（文件名按次数与语言展开）。', '',
              '复现命令：`python post_tensile_test/1026/sample_2/analysis_results/loading_selection_20261026/analyze_and_plot.py`。共用读取、转折拟合和绘图实现位于 `lib/dic_frame_selection.py`。已有输出拒绝覆盖；复跑请复制入口脚本到同级新目录。', '']
    (OUT / 'README.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps({i: {k: r[k] for k in ('raw_peak_N', 'machine_time_offset_s', 'stage_first', 'stage_last', 'midpoint_stage', 'force_coverage_count')} for i, r in records.items()}, indent=2))


if __name__ == '__main__':
    main()
