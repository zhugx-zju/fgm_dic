"""Select maximum-platform DIC stages; do not infer unrecorded forces.

Run with bundled Python from any directory. User confirms all acquisitions at
5 Hz and DIC starts before the machine. Stage numbers are zero based, not image
filenames. Two exported point signals locate motion events, not axial strain.
"""
from pathlib import Path
import csv
import hashlib
import json
import sys
import subprocess

# The bundled runtime lacks scipy, matplotlib and pyxlsb; reuse installed copies.
sys.path.append(r'C:\Users\zhu_g\.conda\envs\abqinter\Lib\site-packages')
import numpy as np
import scipy
from scipy.optimize import least_squares
from pyxlsb import open_workbook
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent
SOURCE = OUT.parents[1]
ROOT = SOURCE.parents[2]
HZ = 5.0


def fit_events(x, y, guesses, bounds, robust=False):
    """Continuous piecewise linear regression, including free dwell drift."""
    midpoint = x.mean()
    count = len(guesses)
    def design(knots):
        return np.column_stack([np.ones(len(x)), x-midpoint] +
                               [np.maximum(x-k, 0) for k in knots])
    coefficients = np.linalg.lstsq(design(guesses), y, rcond=None)[0]
    def prediction(parameters):
        return design(parameters[-count:]) @ parameters[:-count]
    result = least_squares(lambda p: prediction(p)-y, np.r_[coefficients, guesses],
        bounds=(np.r_[[-np.inf]*len(coefficients), np.asarray(bounds)[:, 0]],
                np.r_[[np.inf]*len(coefficients), np.asarray(bounds)[:, 1]]),
        loss='soft_l1' if robust else 'linear', f_scale=.002,
        x_scale='jac', max_nfev=2000)
    if not result.success:
        raise RuntimeError(result.message)
    return dict(knots=result.x[-count:].tolist(),
                slopes=np.cumsum(result.x[1:len(coefficients)]).tolist(),
                rmse=float(np.sqrt(np.mean((prediction(result.x)-y)**2))))


def read_sources(run):
    folder = SOURCE / str(run)
    blocks, labels = [], []
    for row in csv.reader((folder/'disp.csv').read_text(encoding='gb18030').splitlines()):
        if row and '-DY' in row[0]:
            blocks.append([])
            labels.append(row[0])
        elif len(row) == 3 and not row[0]:
            blocks[-1].append([float(row[1]), float(row[2])])
    arrays = [np.asarray(b) for b in blocks]
    if len(arrays) != 2:
        raise ValueError('Expected two exported point blocks')
    for array in arrays:
        np.testing.assert_array_equal(array[:, 0], np.arange(len(array)))
    np.testing.assert_array_equal(arrays[0][:, 0], arrays[1][:, 0])
    points = np.column_stack([b[:, 1] for b in arrays])
    with open_workbook(str(folder/'force.xlsb')) as book:
        with book.get_sheet(1) as sheet:
            rows = [[c.v for c in row] for row in sheet.rows()]
    machine = np.asarray(rows[1:], float)
    if not np.all(np.isfinite(machine)) or not np.all(np.diff(machine[:, 0]) > 0):
        raise ValueError('Invalid machine records')
    return arrays[0][:, 0], points, machine, labels


def main():
    # Only the two well resolved intermediate platforms are used for timing;
    # their displacement amplitudes are never matched to force amplitudes.
    machine_configs = {
        2: [(180, 35, 76, (53, 63)), (280, 75, 110, (89, 99))],
        3: [(180, 25, 64, (41.5, 51.5)), (280, 60, 98, (76, 86))]}
    dic_configs = {
        2: [(180, 185, 385, (288, 338), ((278, 300), (328, 350))),
            (280, 385, 555, (469, 517), ((458, 480), (507, 530)))],
        3: [(180, 200, 440, (322, 395), ((310, 335), (384, 409))),
            (280, 450, 685, (567, 640), ((555, 580), (630, 654)))]}
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
              for run in (1, 2, 3) for p in (SOURCE/str(run)/'disp.csv', SOURCE/str(run)/'force.xlsb')}
    records, arrays = {}, {}
    for run in (1, 2, 3):
        stage, points, machine, labels = read_sources(run)
        mean = points.mean(axis=1)
        arrays[run] = stage, points, machine
        peak = int(np.argmax(machine[:, 1]))
        record = dict(stages=len(stage), point_labels=labels, confirmed_Hz=HZ,
            force_rows=len(machine), machine_end_s=float(machine[-1, 0]),
            maximum_recorded_force_N=float(machine[peak, 1]),
            maximum_recorded_force_time_s=float(machine[peak, 0]),
            initial_machine_force_median_N=float(np.median(machine[machine[:, 0] <= .08, 1])),
            initial_force_is_confirmed_fixture_offset=False,
            maximum_platform_measured_force_N=None)
        records[str(run)] = record
        if run == 1:
            record['status'] = 'no identifiable final maximum-load dwell; no selection'
            continue
        machine_events, dic_events, event_stages, event_times, offsets = {}, {}, [], [], []
        for level, lo, hi, guess in machine_configs[run]:
            mask = (machine[:, 0] >= lo) & (machine[:, 0] <= hi)
            machine_events[str(level)] = fit_events(machine[mask, 0], machine[mask, 1],
                guess, [(g-3, g+3) for g in guess])
        for level, lo, hi, guess, bounds in dic_configs[run]:
            mask = (stage >= lo) & (stage <= hi)
            event = fit_events(stage[mask], mean[mask], guess, bounds, True)
            dic_events[str(level)] = event
            knots = np.asarray(event['knots'])
            times = np.asarray(machine_events[str(level)]['knots'])
            event_stages.extend(knots)
            event_times.extend(times)
            offsets.extend(knots/HZ-times)
        frequency, intercept = np.polyfit(event_times, event_stages, 1)
        median_offset = float(np.median(offsets))
        record.update(machine_events=machine_events, dic_events=dic_events,
            offset_candidates_s=np.asarray(offsets).tolist(), offset_median_s=median_offset,
            offset_range_s=float(np.ptp(offsets)),
            event_regression_effective_Hz=float(frequency),
            effective_Hz_is_diagnostic_not_acquisition_setting=True,
            event_regression_residual_stages=(np.asarray(event_stages)-
                (frequency*np.asarray(event_times)+intercept)).tolist(),
            timing_status='approximately consistent at 5 Hz' if run == 2 else 'inconsistent at confirmed 5 Hz')
        lo, guess, bounds = ((550, (658,), ((643, 675),)) if run == 2 else
                              (720, (823,), ((810, 837),)))
        mask = stage >= lo
        final_event = fit_events(stage[mask], mean[mask], guess, bounds, True)
        entry = final_event['knots'][0]
        first, last = int(np.ceil(entry+6*HZ)), int(np.floor(entry+9*HZ))
        chosen = (stage >= first) & (stage <= last)
        if chosen.sum() != last-first+1 or last >= len(stage):
            raise ValueError('Selected stages are not present')
        center = (first+last)//2
        local_time = stage[chosen]/HZ
        values = mean[chosen]
        slope, intercept_u = np.polyfit(local_time, values, 1)
        # Independently fit both individual points and vary the fitting interval.
        checks = []
        for start in (lo, lo+15, lo+30):
            subset = stage >= start
            for signal in (mean, points[:, 0], points[:, 1]):
                checks.append(fit_events(stage[subset], signal[subset], guess, bounds, True)['knots'][0])
        window = dict(first=first, last=last, count=int(chosen.sum()), single_stage=center,
            selection='6 to 9 seconds after fitted final displacement-platform entry',
            mean_signal_std_um=float(np.std(values, ddof=1)*1000),
            fitted_total_drift_um=float(slope*(local_time[-1]-local_time[0])*1000),
            detrended_rms_um=float(np.sqrt(np.mean((values-slope*local_time-intercept_u)**2))*1000),
            point_total_drift_um=[float(np.polyfit(local_time, points[chosen,j], 1)[0]*
                (local_time[-1]-local_time[0])*1000) for j in range(2)],
            force_N=None, load_level='candidate final 380 N setting; not verified by synchronous force')
        if run == 2:
            estimated = stage[chosen]/HZ-median_offset
            window['estimated_machine_time_s'] = [float(estimated[0]),float(estimated[-1])]
            window['all_offset_candidates_exceed_machine_end'] = bool(
                first/HZ-max(offsets) > machine[-1, 0])
            assert window['all_offset_candidates_exceed_machine_end']
        record.update(final_displacement_event=final_event,
            entry_sensitivity_range_stage=[float(min(checks)),float(max(checks))],
            selected_window=window)

    with plt.rc_context({'font.family':'Microsoft YaHei', 'font.size':10,
                         'axes.unicode_minus':False, 'svg.fonttype':'path'}):
        fig, axes = plt.subplots(3, 2, figsize=(13, 9), layout='constrained')
        for run in (1, 2, 3):
            stage, points, machine = arrays[run]
            left, right = axes[run-1]
            left.plot(machine[:,0],machine[:,1],color='#333333',lw=1.2)
            left.scatter(machine[-1,0],machine[-1,1],s=24,color='#D55E00')
            left.set(xlabel='试验机时间 (s)',ylabel='力示值 (N)',title=f'记录 {run}：实测力在升载末端结束')
            for j,color in enumerate(('#0072B2','#D55E00')):
                right.plot(stage, points[:,j], color=color,lw=.85,label=f'点 {j+1}')
            right.set(xlabel='CSV 阶段号（从 0 开始）',ylabel='DY (mm)',title=f'记录 {run}：两个导出点的位移')
            if run > 1:
                w=records[str(run)]['selected_window']
                right.axvspan(w['first'],w['last'],color='#009E73',alpha=.22,
                              label=f"候选 {w['first']}—{w['last']}")
            else:
                right.text(.03,.90,'未确认覆盖最高载荷平台',transform=right.transAxes)
            right.legend(frameon=False,fontsize=9)
            for ax in (left,right):ax.spines[['top','right']].set_visible(False)
        fig.savefig(OUT/'loading_and_dic.png',dpi=260)
        fig.savefig(OUT/'loading_and_dic.svg')
        plt.close(fig)
        fig, axes = plt.subplots(2,1,figsize=(11,6.5),layout='constrained')
        for ax,run,start in zip(axes,(2,3),(620,795)):
            stage,points,_=arrays[run]
            for j,color in enumerate(('#0072B2','#D55E00')):
                ax.plot(stage,points[:,j]*1000,color=color,lw=1,label=f'点 {j+1}')
            w=records[str(run)]['selected_window'];entry=records[str(run)]['final_displacement_event']['knots'][0]
            ax.axvline(entry,color='.5',ls='--',lw=1,label=f'拟合平台入口 {entry:.1f}')
            ax.axvspan(w['first'],w['last'],color='#009E73',alpha=.20,
                       label=f"导出 {w['first']}—{w['last']}；单帧 {w['single_stage']}")
            ax.set(xlim=(start,stage[-1]+2),ylim=(points[start:].min()*1000-3,points[start:].max()*1000+3),
                xlabel='CSV 阶段号（5 Hz；从 0 开始）',ylabel='DY (μm)',
                title=f'记录 {run}：最高位移平台候选窗' + ('；力—帧同步未通过' if run==3 else '；候选窗无实测力覆盖'))
            ax.legend(frameon=False,fontsize=9,ncol=2)
            ax.spines[['top','right']].set_visible(False)
        fig.savefig(OUT/'maximum_platform_windows.png',dpi=260)
        fig.savefig(OUT/'maximum_platform_windows.svg')
        plt.close(fig)

    # Matplotlib leaves trailing spaces in SVG paths; whitespace is nonsemantic.
    for name in ('loading_and_dic.svg', 'maximum_platform_windows.svg'):
        path = OUT/name
        path.write_text('\n'.join(line.rstrip() for line in
                        path.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8')
    for relative,digest in hashes.items():
        assert hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()==digest
    result=dict(user_confirmed=dict(frequency_Hz=5, dic_starts_before_machine=True),
        method=__doc__, records=records, source_sha256=hashes,
        environment=dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,
                         matplotlib=matplotlib.__version__, executable=sys.executable),
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    (OUT/'analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    with (OUT/'selected_frames.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.writer(stream, lineterminator='\n')
        writer.writerow(['run','stage_first','stage_last','single_stage','confirmed_Hz','synchronous_force_N','status'])
        for run in (2,3):
            w=records[str(run)]['selected_window']
            writer.writerow([run,w['first'],w['last'],w['single_stage'],5,'','candidate only; force unverified'])
    r2,r3=records['2'],records['3'];w2,w3=r2['selected_window'],r3['selected_window']
    report=f'''# sample_1 最高载荷全场导出建议

建议优先导出第2次的 **{w2['first']}—{w2['last']}阶段（15帧），单帧取{w2['single_stage']}**。第3次的 **{w3['first']}—{w3['last']}阶段（15帧），单帧取{w3['single_stage']}**可作最高平台全场检查，但力—DIC同步未通过。第1次不选最高载荷帧。这里只选择末段最高平台，不导出较低载荷工况。

以上均为CSV阶段号，从0开始，窗口首尾包含；不保证等于原照片文件编号。两份CSV块分别是两个点，各次有效阶段数为379、736、939，并非两个数据块行数之和。

## 载荷与参考态限制

用户确认三次均为5 Hz，DIC开始早于试验机；本分析保持这一设定。最高位移平台按加载顺序暂对应380 N示值工况，不能只凭位移变平证明实际载荷恒定或精确等于380 N。

| 目录 | 力记录末时刻 / s | 最大已记录示值 / N | DIC建议 |
|---|---:|---:|---|
| 1 | {records['1']['machine_end_s']:.2f} | {records['1']['maximum_recorded_force_N']:.5f} | 未识别到完整末段最高平台；不选 |
| 2 | {r2['machine_end_s']:.2f} | {r2['maximum_recorded_force_N']:.5f} | {w2['first']}—{w2['last']}，优先 |
| 3 | {r3['machine_end_s']:.2f} | {r3['maximum_recorded_force_N']:.5f} | {w3['first']}—{w3['last']}，仅候选 |

三次最大已记录示值均在最后一行，结束时仍在升载，没有记录380 N保载段。不能把这些末行数值配给后续DIC平均场。第2次候选窗估计对应试验机{w2['estimated_machine_time_s'][0]:.2f}—{w2['estimated_machine_time_s'][1]:.2f} s，晚于126.02 s终点；四个事件给出的全部启动偏移均得到同一“无覆盖”结论，不做记录外恒力外推。

最初0.08 s五个力读数中位数分别为53.4280、46.3272、54.6128 N。这些只是记录开始时的示值，尚未确认全为夹具偏置，也未确认早于试验机启动的DIC参考图与该力状态相同。因此不套用sample_2的60.1 N，不给出本组正式净力。若仅作“380 N减初始示值”的算术，会得到不同值，但现有证据不足以把它们解释为参考态一致的试样净拉力。

## 选帧依据与稳定性

先用连续分段直线拟合“升载—保载—再升载”，允许保载段持续漂移；DIC采用soft-L1稳健拟合。再对末段拟合“升载—平台”单转折，选平台入口后6—9 s。该延迟沿用项目既有选时口径，不按模量、标定或反演误差挑帧。两个点平均信号仅用于事件定位，不能代替全场伸长或应变。

| 目录 | 最高位移平台拟合入口 | 入口敏感性范围（阶段） | 候选窗标准差 / μm | 窗内线性总漂移 / μm | 去趋势RMS / μm |
|---|---:|---:|---:|---:|---:|
| 2 | {r2['final_displacement_event']['knots'][0]:.2f} | {r2['entry_sensitivity_range_stage'][0]:.2f}—{r2['entry_sensitivity_range_stage'][1]:.2f} | {w2['mean_signal_std_um']:.3f} | {w2['fitted_total_drift_um']:.3f} | {w2['detrended_rms_um']:.3f} |
| 3 | {r3['final_displacement_event']['knots'][0]:.2f} | {r3['entry_sensitivity_range_stage'][0]:.2f}—{r3['entry_sensitivity_range_stage'][1]:.2f} | {w3['mean_signal_std_um']:.3f} | {w3['fitted_total_drift_um']:.3f} | {w3['detrended_rms_um']:.3f} |

入口敏感性检查独立拟合两个点及其均值，并将拟合起点后移15、30阶段，共九次；这些范围不是统计置信区间。上表波动包含实际运动和相关误差，不是全场DIC精度，不除以√15。选择的15帧覆盖14个时间间隔，即2.8 s。

## 同步核验

定义`t_machine = stage/5 - offset`，DIC先启动时offset应为正。只用180/280 N两段的四个转折核验，不用位移与载荷的幅值拟合，也不以最终最高平台反过来调时间。

第2次四个offset为{', '.join(f'{v:.3f}' for v in r2['offset_candidates_s'])} s，中位数{r2['offset_median_s']:.3f} s，跨度{r2['offset_range_s']:.3f} s。两段保载各约10.09、9.92 s，与力记录约9.97、9.95 s基本一致；事件回归得到5.055阶段/s，仅作一致性诊断，实际设定保持5 Hz。启动偏差仍有约半秒的事件离散，不能称精确硬件同步。

第3次四个offset为{', '.join(f'{v:.3f}' for v in r3['offset_candidates_s'])} s，跨度{r3['offset_range_s']:.3f} s，不能由一个启动差解释。两个DIC平台长14.31、13.98 s，力记录相应平台约9.99、10.00 s。自由事件回归约7.066阶段/s，仅说明不匹配，不推翻用户确认的5 Hz、不将其改设为7 Hz。应核对force.xlsb与disp.csv是否来自同一次采集、阶段号与原图映射、导出阶段定义及实际时间戳；不能仅凭总时长或末段位移强行对齐。

第1次没有可确认的末段最高平台，且中间突变使事件对应不充分；不根据最终力值把末帧378标成380 N。

## 全场导出内容

1. 优先导出目录2的680—694全部15帧；如只导出一帧，取687。
2. 目录3可导出853—867全部15帧检查空间相关质量；如只导出一帧，取860，明确标注“载荷配对未核验”。
3. 各次保留自身原始参考图/参考坐标、阶段号—照片名对应，以及相同ROI的X、Y、DX、DY和相关质量/有效点标记。参考图用于坐标与位移基准，不是额外低载荷识别工况。
4. 全场平均前核对共同参考图、坐标、有效点集合及逐区轴向伸长随帧变化；单点平稳不能保证全场应变平稳。不要先只导出平均场而丢弃逐帧数据。

目前结论仅为选帧建议，未进行模量反演。若无补充最高平台实测力，后续使用380 N示值或任何净力均须明确假定；第3次还需先解决配对矛盾。

![加载与DIC全程](loading_and_dic.png)

![最高平台候选窗](maximum_platform_windows.png)

## 复现与来源

原始输入为../../1、../../2、../../3中的disp.csv和force.xlsb，均未修改。analysis.json保存六个输入SHA256、拟合结果、环境、源代码哈希和运行时Git版本；selected_frames.csv保存候选清单。PNG与SVG来自同一最终figure状态。

从项目根目录执行：

```powershell
& 'C:/Users/zhu_g/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' -X utf8 -B post_tensile_test/1016/sample_1/analysis_results/max_load_20261026/analyze_max_load.py
```

脚本复用既有abqinter中的缺失解析/科学计算库，不安装或修改环境。复跑只更新本分析目录，不改变原始数据或已有反演结果。
'''
    (OUT/'analysis_report.md').write_text(report,encoding='utf-8')
    print(json.dumps(records,ensure_ascii=False,indent=2))
    print(OUT)


if __name__ == '__main__':
    main()
