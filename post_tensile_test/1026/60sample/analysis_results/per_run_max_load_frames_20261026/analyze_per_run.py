"""Per-run maximum-load export requests; preserve missing force coverage explicitly.

Run from any directory. Reuses the previously inspected read-only parser and
continuous broken-line fitter. Output files must not already exist.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
HELPER = OUT.parent / 'max_load_frames_20261026/analyze_frames.py'
spec = importlib.util.spec_from_file_location('prior_frame_tools', HELPER)
prior = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prior)

# Events specify machine fit interval/search, then DIC fit interval/search.
EVENTS = {
    1: [((124, 138), (127, 134), (128, 142), (132, 138)),
        ((138, 151), (140, 146), (140, 154), (142, 150))],
    2: [((39, 50), (43, 47), (43, 54), (46, 51)),
        ((49, 61), (51, 58), (53, 65), (56, 61))],
    3: [((35, 47), (39, 43), (38, 53), (42, 47))],
}
WINDOWS = {1: (785, 799), 2: (465, 479), 3: (235, 249)}
ROLES = {
    1: 'highest_available_DIC_ramp_fallback_not_run_peak',
    2: 'late_DIC_hold_candidate_force_not_recorded',
    3: 'maximum_load_hold_force_fully_recorded',
}


def main() -> None:
    names = ['analysis_report.md', 'export_requests.csv', 'selection.json',
             'per_run_selection.png', 'per_run_selection.svg']
    if any((OUT / name).exists() for name in names):
        raise FileExistsError('Use a new analysis directory; existing outputs are protected.')
    runs = {number: prior.read_run(number) for number in (1, 2, 3)}
    records, rows = {}, []
    for number, data in runs.items():
        force = data['force']
        offsets, event_records = [], []
        for machine_window, machine_search, dic_window, dic_search in EVENTS[number]:
            mk = prior.breakpoint(force[:, 0], force[:, 2], machine_window, machine_search)
            dk = [prior.breakpoint(data['time'], y, dic_window, dic_search) for y in
                  (data['dic'][0][:, 2], data['dic'][1][:, 2], data['mean_dy'])]
            values = [mk['time_s']-k['time_s'] for k in dk]
            offsets.extend(values)
            event_records.append({'machine': mk, 'dic_points_and_mean': dk,
                                  'offsets_s': values, 'machine_window': machine_window,
                                  'machine_search': machine_search, 'dic_window': dic_window,
                                  'dic_search': dic_search})
        offset = round(float(np.median(offsets)), 2)
        if number == 3:
            assert offset == -3.44
            # Include the prior wider fitting-window sensitivity audit.
            audit = json.loads((HELPER.parent / 'selection.json').read_text(encoding='utf-8'))
            offsets.extend(audit['offset_sensitivity_s'])
        sensitivity = [float(min(offsets)), float(max(offsets))]
        first, last = WINDOWS[number]
        stages = np.arange(first, last+1)
        times = stages/5 + offset
        covered = (times >= force[0, 0]) & (times <= force[-1, 0])
        paired = np.full(len(stages), np.nan)
        paired[covered] = np.interp(times[covered], force[:, 0], force[:, 1])
        # Missing force is retained as blank, never clamped or extrapolated.
        peak_index = int(np.argmax(force[:, 1]))
        peak_time = float(force[peak_index, 0])
        peak_stage = (peak_time-offset)*5
        assert len(stages) == 15 and stages[-1] < len(data['time'])
        assert max(row[1] for row in force.tolist()) == force[peak_index, 1]
        records[number] = {
            'raw_peak_N': float(force[peak_index, 1]), 'peak_machine_time_s': peak_time,
            'peak_workbook_row': peak_index+2, 'approximate_peak_stage': peak_stage,
            'peak_stage_range_from_event_offsets': [(peak_time-sensitivity[1])*5,
                                                    (peak_time-sensitivity[0])*5],
            'dic_last_stage': len(data['time'])-1,
            'stage_first': first, 'stage_last': last, 'frame_count': 15,
            'selection_role': ROLES[number], 'machine_time_offset_s': offset,
            'offset_sensitivity_s': sensitivity, 'events': event_records,
            'paired_machine_time_s': times[[0, -1]].tolist(),
            'force_record_coverage_count': int(covered.sum()),
            'force_mean_N': float(paired.mean()) if covered.all() else None,
            'force_range_N': [float(paired.min()), float(paired.max())] if covered.all() else None,
        }
        for stage, t, f, valid in zip(stages, times, paired, covered):
            rows.append([number, stage, f'stage_{stage:05d}.comdat_comdata.csv',
                         ROLES[number], stage/5, round(t, 3), bool(valid),
                         f'{f:.6f}' if valid else '',
                         data['dic'][0][stage, 2], data['dic'][1][stage, 2]])
    assert records[1]['approximate_peak_stage'] > records[1]['dic_last_stage']
    assert records[1]['force_range_N'][1] < records[1]['raw_peak_N']-50
    assert records[2]['force_record_coverage_count'] == 0
    assert records[3]['force_record_coverage_count'] == 15
    sources = [{'relative_path': f'{run}/{name}',
                'sha256': hashlib.sha256((ROOT / str(run) / name).read_bytes()).hexdigest()}
               for run in runs for name in ('disp.csv', 'force.xlsb')]
    summary = {'scope': 'Each run separately: maximum recorded load and available DIC export requests.',
               'fps_user_confirmed': 5, 'first_stage_user_confirmed': 0,
               'prior_analysis_scope_superseded': 'Prior report selected only the global maximum; this report includes all three runs.',
               'sources': sources, 'helper_sha256': hashlib.sha256(HELPER.read_bytes()).hexdigest(),
               'runs': records, 'reference_stage_each_run': 0,
               'uncertainty_note': 'Event offsets are empirical alignment sensitivity, not a confidence interval or hardware trigger measurement.'}
    with (OUT / 'export_requests.csv').open('x', encoding='utf-8-sig', newline='') as handle:
        writer = csv.writer(handle, lineterminator='\n')
        writer.writerow(['run', 'stage', 'expected_filename', 'selection_role', 'camera_time_s',
                         'estimated_machine_time_s', 'force_record_covers_frame', 'paired_raw_force_N',
                         'first_point_DY_mm', 'second_point_DY_mm'])
        writer.writerows(rows)
    (OUT / 'selection.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    fig, axes = plt.subplots(3, 1, figsize=(10, 9))
    for (number, data), ax in zip(runs.items(), axes):
        rec = records[number]
        f = data['force']
        t = data['time'] + rec['machine_time_offset_s']
        ax.plot(f[:, 0], f[:, 1], color='#0072B2', label='Recorded raw force')
        right = ax.twinx()
        right.plot(t, data['mean_dy'], color='#D55E00', lw=1, label='Mean point DY')
        ax.axvspan(*rec['paired_machine_time_s'], alpha=.2, color='#009E73')
        ax.axvline(rec['peak_machine_time_s'], ls=':', lw=1, color='#0072B2')
        ax.set_ylabel('Raw force (N)', color='#0072B2')
        right.set_ylabel('Mean point DY (mm)', color='#D55E00')
        label = {1: 'Fallback only: DIC ends before run peak',
                 2: 'Candidate hold: force record already ended',
                 3: 'Peak-load hold: complete force coverage'}[number]
        ax.set_title(f"Run {number}: export {rec['stage_first']}-{rec['stage_last']} | {label}", fontsize=10)
        ax.set_xlabel('Machine time (s), estimated from independently aligned camera stages')
        ax.spines['top'].set_visible(False)
        right.spines['top'].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / 'per_run_selection.png', dpi=220)
    fig.savefig(OUT / 'per_run_selection.svg')
    svg = OUT / 'per_run_selection.svg'
    svg.write_bytes(('\n'.join(line.rstrip() for line in svg.read_text(encoding='utf-8').splitlines())+'\n').encode('utf-8'))
    plt.close(fig)
    a, b, c = (records[i] for i in (1, 2, 3))
    report = f'''# 三次加载分别提取各自最大载荷全场

本报告按用户澄清，分别处理 1、2、3 次加载，替代此前“只选三次中的最高载荷”的范围解释。原分析数据与文件保留。

| 加载记录 | 该次原始峰值 N | 峰值附近帧号（估计） | 建议实际导出窗口 | 必须保留的解释 |
|---|---:|---:|---|---|
| 1 | {a['raw_peak_N']:.5f} | {a['approximate_peak_stage']:.1f}，超出现有 0–799 帧 | 785–799，共 15 帧，作为当前可用最高载荷末段备用 | 此窗口仍在升载，仅约 {a['force_range_N'][0]:.2f}–{a['force_range_N'][1]:.2f} N，不是约 380 N 峰值或稳定平台。真正峰值全场缺失。 |
| 2 | {b['raw_peak_N']:.5f} | {b['approximate_peak_stage']:.1f}，约 445 帧 | 465–479，共 15 帧，作为最大载荷末端停留候选 | DIC 有末端停留迹象，但此窗口没有对应载荷记录，不能确认逐帧均为约 280 N。可先导出，补齐载荷后再用于定量计算。 |
| 3 | {c['raw_peak_N']:.5f} | {c['approximate_peak_stage']:.1f}，约 223 帧 | 235–249，共 15 帧 | 最高载荷停留窗口，逐帧载荷完整，均值 {c['force_mean_N']:.3f} N。 |

三次均保留各自的 stage_00000 参考图／坐标，导出 X、Y、DX、DY 及相关质量或有效性字段，同一次使用共同参考图、ROI 与计算参数。已相对参考图计算的位移不应再将参考帧计入平台平均。

## 第 1 次：380 N 峰值没有包含在当前 DIC 中

用最后一次约 280 N 停留的进入、退出转折分别对齐试验机位移与两个 DIC 测点及均值。加载时间约为 stage/5 {a['machine_time_offset_s']:+.2f} s，事件／测点偏移范围 {a['offset_sensitivity_s'][0]:.2f} 至 {a['offset_sensitivity_s'][1]:.2f} s。此局部同步用于覆盖判断和末段近似配对，没有把其他次的偏移照搬过来。

载荷峰值在加载 {a['peak_machine_time_s']:.2f} s，按上述偏移估计需约 {a['peak_stage_range_from_event_offsets'][0]:.0f}–{a['peak_stage_range_from_event_offsets'][1]:.0f} 帧。当前 DIC 到 799 帧（相机 159.8 s），约对应加载 {159.8+a['machine_time_offset_s']:.2f} s，明显未拍到峰值。785–799 是最后 15 个现有帧，具有实际载荷配对，但负载在窗口内上升约 {a['force_range_N'][1]-a['force_range_N'][0]:.2f} N，不能按单一恒载平台平均。若软件中还有未导出的后续图像，应补充峰值附近图像与最大载荷停留段，再定正式窗口；若原始采集也在 799 帧结束，则无法从本记录恢复峰值全场。

## 第 2 次：可导出末端停留，但缺少同期载荷记录

用约 180 N 中间停留的进入、退出转折对齐，加载时间约为 stage/5 {b['machine_time_offset_s']:+.2f} s，事件／测点偏移范围 {b['offset_sensitivity_s'][0]:.2f} 至 {b['offset_sensitivity_s'][1]:.2f} s。原始载荷峰值 {b['raw_peak_N']:.5f} N 出现在 {b['peak_machine_time_s']:.2f} s，对应约 445 帧，事件偏移对应约 {b['peak_stage_range_from_event_offsets'][0]:.1f}–{b['peak_stage_range_from_event_offsets'][1]:.1f} 帧。若目标是峰值附近瞬时全场，可另导出 444–447 帧作为候选，不保证某一相机帧恰好包含 50 Hz 载荷瞬时峰值。

465–479 对应相机 93.0–95.8 s、估计加载 {b['paired_machine_time_s'][0]:.2f}–{b['paired_machine_time_s'][1]:.2f} s，位于 DIC 最后一次由升载转入缓慢变化之后，避开转折与尾部约第 530 帧的位移突降。载荷文件却在 {b['peak_machine_time_s']:.2f} s 结束，所以全部 15 帧的配对载荷留空。不外推、不以末条力值填充，也不根据单点位移平缓就认定载荷恒定。此窗口是导出候选，不是已通过力配对的正式识别平台。

## 第 3 次：完整最大载荷平台

原先同步结论不变，加载时间约为 stage/5 −3.44 s，拟合窗口敏感性 −3.66 至 −3.23 s。235–249 对应加载 43.56–46.36 s，原始载荷范围 {c['force_range_N'][0]:.3f}–{c['force_range_N'][1]:.3f} N，15 个配对时刻均值 {c['force_mean_N']:.3f} N；在上述偏移范围内载荷记录均完整覆盖。

## 核验、边界与复现

相机 5 Hz、首帧 0 已由用户确认。前两份 CSV 的时间列实际逐行加 1，因此按行号/5 转为真实秒数；第三次时间列与此换算一致到约 0.003 s。CSV 没有显式图像编号，因此需在全场导出软件中确认图像连续对应。每份 CSV 含两个 DY 测点，不以单点位移代替应变。

已核对六个输入文件哈希、两测点的时间／长度一致性、帧范围、原始峰值的独立列表 max、全部 45 个候选帧的载荷覆盖状态；明确区分第 1 次 15 帧可配对但非峰值、第 2 次 15 帧无载荷覆盖、第 3 次 15 帧完整覆盖。图中左右轴分别为载荷和单点位移，不比较两轴斜率大小。原始文件不改写，未进行反演，也未检查尚未导出的全场相关质量。

各次参考图的载荷需单独核实。第 3 次尤其明显：参考图早于载荷记录起点，不能默认零载荷，也不能自动把第一条载荷当作参考载荷扣除。

复现命令：`python post_tensile_test/1026/60sample/analysis_results/per_run_max_load_frames_20261026/analyze_per_run.py`。依赖原分析目录中的读取与转折拟合函数、numpy、pyxlsb、matplotlib。已有输出拒绝覆盖；复跑需使用同级新目录。转折区间、偏移、逐帧覆盖状态、原始峰值行号及 SHA256 均保存在 selection.json 和 export_requests.csv。PNG 与 SVG 来自同一最终图状态。
'''
    (OUT / 'analysis_report.md').write_text(report, encoding='utf-8')
    print(json.dumps({i: {k: v[k] for k in ('approximate_peak_stage', 'stage_first', 'stage_last',
                       'machine_time_offset_s', 'force_record_coverage_count', 'force_range_N')}
                      for i, v in records.items()}, indent=2))


if __name__ == '__main__':
    main()
