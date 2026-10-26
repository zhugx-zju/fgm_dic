"""Redraw the three frozen frame selections in the project's two-panel style.

Read-only inputs; retain frozen runs 2/3, refresh run 1 after its DIC was extended.
Run from any directory. Existing outputs are protected from overwriting.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import platform
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

OUT = Path(__file__).resolve().parent
SAMPLE = OUT.parents[1]
REPO = SAMPLE.parents[2]
SELECTION = OUT.parent / 'per_run_max_load_frames_20261026/selection.json'
REQUESTS = SELECTION.with_name('export_requests.csv')
PARSER = OUT.parent / 'max_load_frames_20261026/analyze_frames.py'
TEMPLATE = 'results/dic/loading_figure_20261021/plot_loading_selection.py'
COLORS = {1: '#D55E00', 2: '#D55E00', 3: '#0072B2'}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def draw(number: int, data: dict, rec: dict, language: str) -> dict:
    cn = language == 'zh'
    def tr(en, zh):
        return zh if cn else en
    plt.rcParams.update({
        'font.family': 'Microsoft YaHei' if cn else 'Arial', 'font.size': 12,
        'axes.labelsize': 13, 'axes.titlesize': 14, 'axes.linewidth': .9,
        'xtick.direction': 'out', 'ytick.direction': 'out',
        'xtick.major.size': 4, 'ytick.major.size': 4,
        'svg.fonttype': 'path', 'axes.unicode_minus': False,
        'figure.facecolor': 'white', 'savefig.facecolor': 'white',
    })
    f = data['force']
    t = data['time'] + rec['machine_time_offset_s']
    mean = data['mean_dy']
    lo, hi = rec['stage_first'], rec['stage_last']
    center = (lo+hi)//2
    color = COLORS[number]
    end = max(float(t[-1]), float(f[-1, 0]))
    x_end = end*1.025
    fig, (force_ax, dic_ax) = plt.subplots(2, 1, figsize=(12, 6.75), sharex=True)
    fig.subplots_adjust(left=.10, right=.975, top=.855, bottom=.235, hspace=.32)
    fig.suptitle(tr(f'Uniform 60% specimen · Run {number}', f'60% 填充率均匀试样 · 第 {number} 次加载'),
                 x=.10, ha='left', y=.97, fontsize=16, fontweight='bold')
    force_ax.plot(f[:, 0], f[:, 1], color='#292929', lw=1.7)
    dic_ax.plot(t, mean, color='#4D4D4D', lw=1.35)
    force_ax.set_ylim(20, 425)
    force_ax.set_yticks([80, 180, 280, 380])
    force_ax.set_ylabel(tr('Machine force reading (N)', '试验机力示值 (N)'))
    dic_ax.set_ylabel(tr('Mean axial displacement (mm)', '两点平均轴向位移 (mm)'))
    dic_ax.set_xlabel(tr('Estimated machine time (s)', '估计试验机时间 (s)'))
    span = float(mean.max()-mean.min())
    dic_ax.set_ylim(min(-.015, float(mean.min())-.025*span), float(mean.max())+.37*span+.025)
    dic_ax.set_xlim(0, x_end)
    step = 20 if x_end > 100 else 10
    dic_ax.set_xticks(np.arange(0, x_end+.1, step))
    for ax, title in ((force_ax, tr('(a)  Loading history', '(a)  加载过程')),
                      (dic_ax, tr('(b)  DIC frame selection', '(b)  DIC 时刻选取'))):
        ax.set_title(title, loc='left', pad=11, fontweight='bold')
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(labelsize=11)
        ax.axvspan(t[lo], t[hi], color=color, alpha=.14, lw=0, zorder=1)
        for edge in (lo, hi):
            ax.axvline(t[edge], color=color, lw=.75, alpha=.65, ls=(0, (3, 3)))
    force_ax.tick_params(labelbottom=False)
    within = (t[lo:hi+1] >= f[0, 0]) & (t[lo:hi+1] <= f[-1, 0])
    if within.any():
        selected_t = t[lo:hi+1][within]
        force_ax.plot(selected_t, np.interp(selected_t, f[:, 0], f[:, 1]), color=color, lw=3)
    dic_ax.plot(t[lo:hi+1], mean[lo:hi+1], color=color, lw=2.2)
    dic_ax.scatter(t[center], mean[center], s=38, marker='D', color=color,
                   edgecolors='white', linewidths=.7, zorder=5)
    labels = {
        1: tr('End-hold candidate; force not recorded', '末端停留候选；同期载荷缺失'),
        2: tr('End-hold candidate; force not recorded', '末端停留候选；同期载荷缺失'),
        3: tr('Maximum-load hold; force fully recorded', '最大载荷平台；载荷记录完整'),
    }
    force_ax.legend(handles=[Line2D([0], [0], color=color, lw=3, label=labels[number])],
                    loc='upper left', frameon=False, fontsize=11, handlelength=1.8)
    peak_t = rec['peak_machine_time_s']
    peak_f = rec['raw_peak_N']
    force_ax.scatter(peak_t, peak_f, s=20, color='#292929', zorder=5)
    force_ax.annotate(tr(f'Peak {peak_f:.3f} N', f'峰值 {peak_f:.3f} N'),
                      xy=(peak_t, peak_f), xytext=(max(end*.45, peak_t-end*.20), 345 if number == 2 else peak_f-90),
                      fontsize=10.5, ha='center', color='#303030',
                      bbox={'facecolor': 'white', 'edgecolor': 'none', 'pad': 1.5},
                      arrowprops={'arrowstyle': '-', 'color': '.45', 'lw': .8})
    if number in (1, 2, 3):
        # Keep the actual DIC tail visible; show missing force instead of extending it.
        for ax in (force_ax, dic_ax):
            ax.axvspan(f[-1, 0], end, facecolor='.96', hatch='///', edgecolor='.80', lw=0, zorder=0)
        if number in (1, 2):
            force_ax.text((f[-1, 0]+end)/2, 145,
                          tr('No recorded\nforce', '同期载荷\n未记录'),
                          ha='center', va='center', fontsize=10.5, color='.4')
            label = tr(f'Frames {lo}–{hi}: 15-frame candidate\nMidpoint frame: {center}; force unverified',
                       f'第 {lo}–{hi} 帧：15 帧候选\n中间帧：{center}；载荷尚未核实')
            if number == 1:
                foot = tr('Updated DIC now ends at stage 999. Peak vicinity: stages 918–921; export candidate 945–959.',
                          '更新后的 DIC 已到 999 帧；峰值附近约 918–921 帧，末端停留候选为 945–959 帧。')
                note = tr('The force record ends at 179.72 s; the candidate has no recorded force. Verify before quantitative use.',
                          '载荷记录止于 179.72 s；候选窗口缺少同期载荷，补齐后再作定量计算。')
            else:
                foot = tr('Peak vicinity: stages 444–447. The candidate hold is beyond the recorded force time range.',
                          '峰值附近可另导出 444–447 帧；候选停留窗口超出载荷记录时段。')
                note = tr('The force record ends at 84.36 s; do not fill the missing tail with the last force value.',
                          '载荷记录止于 84.36 s；465–479 帧可先导出，补齐同期载荷后再作定量计算。')
        else:
            label = tr(f'Frames {lo}–{hi}: 15-frame window\nSingle-frame option: {center}',
                       f'第 {lo}–{hi} 帧：15 帧窗口\n可选单帧：{center}')
            foot = tr('Selected force: 379.557–379.730 N; mean 379.650 N. Instantaneous peak is near stage 223.',
                      '窗口载荷 379.557–379.730 N，均值 379.650 N；瞬时峰值约在 223 帧。')
            note = tr('Hatching denotes the unrecorded force tail. Reference-frame load must be checked separately.',
                      '斜线区表示载荷记录结束后的 DIC 尾段；参考图载荷须单独核实。')
    dic_ax.annotate(label, xy=(t[center], mean[center]),
                    xytext=(t[center]-end*.12, mean[center]+span*.20+.015),
                    ha='center', va='bottom', color=color, fontsize=11,
                    arrowprops={'arrowstyle': '-', 'color': color, 'lw': .9, 'shrinkA': 4, 'shrinkB': 6})
    fig.text(.10, .115, foot, fontsize=10.5, color='#303030')
    fig.text(.10, .070, note, fontsize=10.5, color='#303030')
    fig.text(.10, .025, tr(f'5 Hz; zero-based stages. Estimated mapping: t_machine = stage / 5 {rec["machine_time_offset_s"]:+.2f} s.',
                         f'采样率 5 Hz；帧号从 0 开始。估计同步：试验机时间 = 帧号 / 5 {rec["machine_time_offset_s"]:+.2f} s。'),
             fontsize=10, color='#505050')
    stem = OUT / f'run{number}_loading_dic_selection_{language}'
    fig.savefig(stem.with_suffix('.png'), dpi=400)
    fig.savefig(stem.with_suffix('.svg'))
    svg = stem.with_suffix('.svg')
    svg.write_bytes(('\n'.join(line.rstrip() for line in svg.read_text(encoding='utf-8').splitlines())+'\n').encode('utf-8'))
    plt.close(fig)
    return {'run': number, 'language': language, 'stage_first': lo, 'stage_last': hi,
            'midpoint_stage': center, 'selected_force_coverage_count': int(within.sum()),
            'plotted_time_range_s': [0, x_end], 'png_pixels': [4800, 2700]}


def main() -> None:
    if any(OUT.glob('run*_loading_dic_selection_*')) or (OUT / 'figure_metadata.json').exists():
        raise FileExistsError('Use a new output directory; existing figures are protected.')
    spec = importlib.util.spec_from_file_location('prior_parser', PARSER)
    parser = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(parser)
    selection = json.loads(SELECTION.read_text(encoding='utf-8'))
    updates = []
    for source in selection['sources']:
        current_hash = sha(SAMPLE / source['relative_path'])
        if current_hash != source['sha256']:
            assert source['relative_path'] == '1/disp.csv', 'Unreviewed input change'
            updates.append({'relative_path': source['relative_path'], 'previous_sha256': source['sha256'],
                            'current_sha256': current_hash, 'reason': 'DIC extended from 800 to 1000 stages; block order reversed.'})
            source['sha256'] = current_hash
    requests = list(csv.DictReader(REQUESTS.open(encoding='utf-8-sig')))
    assert len(updates) == 1
    checks, current_requests = [], []
    for number in (1, 2, 3):
        data = parser.read_run(number)
        rec = selection['runs'][str(number)]
        stages = np.arange(len(data['time']))
        wanted = [int(r['stage']) for r in requests if int(r['run']) == number]
        assert wanted == list(range(rec['stage_first'], rec['stage_last']+1))
        assert len(wanted) == 15
        if number == 1:
            assert len(stages) == 1000
            # Recheck the previously selected rows by mean: the two point blocks changed order.
            old = [r for r in requests if int(r['run']) == 1]
            for row in old:
                stage = int(row['stage'])
                old_mean = (float(row['first_point_DY_mm'])+float(row['second_point_DY_mm']))/2
                np.testing.assert_allclose(data['mean_dy'][stage], old_mean, atol=1e-12, rtol=0)
            event_offsets = []
            for event in rec['events']:
                mk = parser.breakpoint(data['force'][:, 0], data['force'][:, 2], event['machine_window'], event['machine_search'])
                for y in (data['dic'][0][:, 2], data['dic'][1][:, 2], data['mean_dy']):
                    dk = parser.breakpoint(data['time'], y, event['dic_window'], event['dic_search'])
                    event_offsets.append(mk['time_s']-dk['time_s'])
            assert round(float(np.median(event_offsets)), 2) == rec['machine_time_offset_s']
            rec.update({'stage_first': 945, 'stage_last': 959, 'dic_last_stage': 999,
                        'selection_role': 'late_DIC_hold_candidate_force_not_recorded',
                        'force_record_coverage_count': 0, 'force_mean_N': None, 'force_range_N': None,
                        'paired_machine_time_s': [945/5+rec['machine_time_offset_s'], 959/5+rec['machine_time_offset_s']],
                        'updated_DIC_hold_entry': parser.breakpoint(data['time'], data['mean_dy'], (175, 195), (180, 190))})
            wanted = list(range(945, 960))
        t = data['time']+rec['machine_time_offset_s']
        force = data['force']
        covered = (t >= force[0, 0]) & (t <= force[-1, 0])
        paired = np.full(len(t), np.nan)
        paired[covered] = np.interp(t[covered], force[:, 0], force[:, 1])
        selected = np.isin(stages, wanted)
        assert int(covered[selected].sum()) == rec['force_record_coverage_count']
        if rec['force_mean_N'] is not None:
            np.testing.assert_allclose(paired[selected].mean(), rec['force_mean_N'], atol=1e-10, rtol=0)
        for stage in wanted:
            current_requests.append([number, stage, f'stage_{stage:05d}.comdat_comdata.csv',
                                     rec['selection_role'], data['time'][stage], t[stage], bool(covered[stage]),
                                     paired[stage] if covered[stage] else ''])
        for language in ('zh', 'en'):
            checks.append(draw(number, data, rec, language))
        with (OUT / f'run{number}_timing_curve_data.csv').open('x', encoding='utf-8-sig', newline='') as handle:
            writer = csv.writer(handle, lineterminator='\n')
            writer.writerow(['stage_0based', 'camera_time_s', 'estimated_machine_time_s',
                             'first_exported_DY_mm', 'second_exported_DY_mm', 'mean_DY_mm',
                             'raw_force_N_if_covered', 'force_record_coverage', 'selected_window'])
            for i in stages:
                writer.writerow([i, data['time'][i], t[i], data['dic'][0][i, 2], data['dic'][1][i, 2],
                                 data['mean_dy'][i], paired[i] if covered[i] else '', bool(covered[i]), bool(selected[i])])
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
    template_bytes = subprocess.check_output(['git', 'show', f'{head}:{TEMPLATE}'], cwd=REPO)
    metadata = {
        'scope': 'Use historical plot style; update run 1 window for newly extended DIC. Retain runs 2/3.',
        'style_reference': {'git_commit': head, 'path': TEMPLATE,
                            'sha256': hashlib.sha256(template_bytes).hexdigest()},
        'source_sha256': {str(p.relative_to(REPO)): sha(p) for p in [SELECTION, REQUESTS, PARSER]},
        'raw_sources_verified': selection['sources'], 'verification': checks,
        'input_updates': updates,
        'environment': {'python': platform.python_version(), 'numpy': np.__version__, 'matplotlib': matplotlib.__version__},
        'script_sha256': sha(Path(__file__)),
        'limitations': ['No full-field quality check; fields not exported yet.',
                        'Synchronization inferred from events; no hardware timestamp verification.',
                        'Runs 1/2 candidate holds have no recorded contemporaneous force.',
                        'Two-point displacement is a timing signal, not specimen strain.'],
    }
    (OUT / 'figure_metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    (OUT / 'selection_current.json').write_text(json.dumps(selection, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    with (OUT / 'export_requests_current.csv').open('x', encoding='utf-8-sig', newline='') as handle:
        writer = csv.writer(handle, lineterminator='\n')
        writer.writerow(['run', 'stage', 'expected_filename', 'selection_role', 'camera_time_s',
                         'estimated_machine_time_s', 'force_record_covers_frame', 'paired_raw_force_N'])
        writer.writerows(current_requests)
    readme = '''# 三次加载过程与 DIC 选帧图

沿用项目 `results/dic/loading_figure_20261021` 的图件版式：上下两幅共享估计试验机时间轴；上幅为力示值，下幅为 DIC 两点平均 DY。浅色窗口、虚线边界、加粗曲线和中间帧菱形标记对应冻结的逐次选帧记录。中英文分别绘制，每张 PNG 为 4800×2700、400 dpi，SVG 与 PNG 来自同一最终 figure，文字转路径。

| 次数 | 图示窗口（含两端） | 中间帧 | 状态 |
|---|---|---|---|
| 1 | 945–959 | 952 | 更新后 DIC 已到 999 帧；此为末端停留候选，全部 15 帧缺少同期载荷。峰值附近约 918–921 帧。 |
| 2 | 465–479 | 472 | 末端停留候选，全部 15 帧缺少同期载荷；峰值附近可另导出 444–447 帧。 |
| 3 | 235–249 | 242 | 最大载荷平台，载荷覆盖完整，均值 379.650 N。 |

图中“中间帧”仅标识窗口中心，不表示第 1、2 次已通过恒载平台或定量配对核验。三次图的斜线区均表示载荷记录已结束，曲线不外推。区别于旧图仅展示共同覆盖时段，本图保留不完整尾段，以直接说明本次输入缺口。三图力轴同为 20–425 N；DIC 位移轴按各次量级设置，不能直接比较视觉斜率。

## 文件与复现

- `run1/2/3_loading_dic_selection_zh.png/.svg`：三次的中文图（实际文件名按数字展开）。
- `run1/2/3_loading_dic_selection_en.png/.svg`：对应英文图。
- `run1/2/3_timing_curve_data.csv`：逐帧原始位移、均值、同步时间、覆盖标志及选帧标记。无对应载荷时留空。
- `figure_metadata.json`：原始输入与冻结选帧哈希、历史样式来源、核验信息及绘图环境。
- `selection_current.json` / `export_requests_current.csv`：当前三次的选帧记录与逐帧清单；第 1 次更新为 945–959，第 2、3 次不变。

输入首先核对同级 `per_run_max_load_frames_20261026/selection.json` 与 `export_requests.csv`。发现第 1 次 disp.csv 已从 800 帧扩展至 1000 帧，测点导出顺序对调；旧选帧范围内的两点均值一致，最后一次中间平台的进入、退出事件重新拟合给出的中位同步偏移仍为 −4.25 s。新增 DIC 的末端转折为相机约 183.65 s（约 918 帧），后续缓慢变化段选取 945–959 帧，即入口后约 5.35–8.15 s，避开约 980 帧的局部波动；对应估计加载 184.75–187.55 s，超过载荷记录终点 179.72 s。因此之前“第 1 次未拍到峰值”的结论仅适用于原 800 帧版本，现版本可导出末端候选，但仍不能验证其载荷。

第 2、3 次输入哈希与选帧不变；第 1 次变更前后哈希单独保存在元数据中。已核对全部 45 帧的导出清单与窗口覆盖数（分别 0、0、15），复核第 3 次载荷均值；未继承旧图的力偏置、平台时长或识别／预测用途，未检查全场相关质量。原分析报告和原图保持原状，当前结果以本目录的新清单为准。

从项目根目录运行 `python post_tensile_test/1026/60sample/analysis_results/loading_figures_20261026/plot_loading_selection.py`。现有图件拒绝覆盖；复跑请复制脚本到同级新目录。原历史目录在工作区的删除状态保留，样式从 Git 历史只读取得。
'''
    (OUT / 'README.md').write_text(readme, encoding='utf-8')
    print(json.dumps(checks, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
