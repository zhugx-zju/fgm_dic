"""From project root: python -B -m post_tensile_test.1026.60sample.analysis_results.frame_timing_check_20261027.check_frame_timing

Read-only audit. Existing audit outputs are never overwritten.
"""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

from lib.experiment_data import file_sha256, load_dic_frame_series


def main():
    output = Path(__file__).resolve().parent
    if any((output/name).exists() for name in ('report.md', 'summary.csv', 'per_frame.csv', 'metadata.json')):
        raise FileExistsError('Choose a new audit directory; existing evidence is protected')
    repo = output.parents[4]
    root = repo/'post_tensile_test/1026/60sample'
    parser_path = root/'analysis_results/max_load_frames_20261026/analyze_frames.py'
    spec = importlib.util.spec_from_file_location('frame_parser', parser_path)
    parser = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(parser)
    selection_path = root/'analysis_results/loading_figures_20261026/selection_current.json'
    selection = json.loads(selection_path.read_text(encoding='utf-8'))
    summaries, frames, sources = [], [], {}
    for path in (parser_path, selection_path, Path(__file__), repo/'lib/experiment_data.py', repo/'lib/io_utils.py'):
        sources[str(path.relative_to(repo))] = file_sha256(path)
    for run in (1, 2, 3):
        record = selection['runs'][str(run)]
        data = parser.read_run(run)
        lo, hi = record['stage_first'], record['stage_last']
        peak_stage = (record['peak_machine_time_s']-record['machine_time_offset_s'])*5
        peak_lo, peak_hi = round(peak_stage)-2, round(peak_stage)+2
        peak_dy = float(data['mean_dy'][peak_lo:peak_hi+1].mean())
        selected_dy = float(data['mean_dy'][lo:hi+1].mean())
        folder = root/str(run)/('280N' if run == 2 else '380N')
        series = load_dic_frame_series(folder, lo, hi)
        design = np.column_stack([np.ones(len(series.coordinates)), series.coordinates])
        coefficients = np.einsum('ip,fpc->fic', np.linalg.pinv(design), series.frame_displacements)
        axial_strain = coefficients[:, 2, 1]
        prepared = repo/f'results/dic_full/60sample_1026_run{run}_h1_20261026/01_prepared/fields.npz'
        with np.load(prepared) as saved:
            np.testing.assert_array_equal(saved[f'run{run}_grid_ids'], series.grid_ids)
            # Exact equality verifies no displacement attenuation during import.
            np.testing.assert_allclose(saved[f'run{run}_observed'],
                                       series.frame_displacements.mean(axis=0)[:, ::-1], rtol=0, atol=0)
        for name in ('force.xlsb', 'disp.csv'):
            path = root/str(run)/name
            sources[str(path.relative_to(repo))] = file_sha256(path)
        sources[str(prepared.relative_to(repo))] = file_sha256(prepared)
        for source in series.metadata['sources']:
            sources[str(Path(source['path']).relative_to(repo))] = source['sha256']
        machine_time = np.arange(lo, hi+1)/5+record['machine_time_offset_s']
        force = data['force']
        covered = (machine_time >= force[0, 0]) & (machine_time <= force[-1, 0])
        paired_force = np.full(len(machine_time), np.nan)
        paired_force[covered] = np.interp(machine_time[covered], force[:, 0], force[:, 1])
        summaries.append(dict(run=run,stage_first=lo,stage_last=hi,estimated_peak_stage=peak_stage,
            peak_point_window_first=peak_lo,peak_point_window_last=peak_hi,
            peak_vicinity_point_mean_DY_mm=peak_dy,selected_point_mean_DY_mm=selected_dy,
            selected_minus_peak_point_mean_DY_mm=selected_dy-peak_dy,
            field_axial_strain_mean=float(axial_strain.mean()),
            field_axial_strain_min=float(axial_strain.min()),field_axial_strain_max=float(axial_strain.max()),
            strain_range_over_mean_percent=float(100*np.ptp(axial_strain)/abs(axial_strain.mean())),
            mean_strain_below_window_max_percent=float(100*(1-axial_strain.mean()/axial_strain.max())),
            force_coverage_count=int(covered.sum()),
            selected_actual_mean_force_N=float(paired_force[covered].mean()) if covered.any() else None,
            raw_peak_force_N=float(force[:, 1].max()),exact_raw_mean_input_match=True))
        for index, stage in enumerate(range(lo, hi+1)):
            frames.append(dict(run=run,stage=stage,estimated_machine_time_s=machine_time[index],
                paired_force_N=paired_force[index],point_mean_DY_mm=data['mean_dy'][stage],
                fullfield_mean_DY_mm=series.frame_displacements[index, :, 1].mean(),
                affine_axial_strain=axial_strain[index]))
    table = pd.DataFrame(summaries)
    table.to_csv(output/'summary.csv', index=False, lineterminator='\n')
    pd.DataFrame(frames).to_csv(output/'per_frame.csv', index=False, lineterminator='\n')
    lines = ['# 选帧是否落在位移已减小的时段', '',
        '现有证据未显示导入过程缩小位移，亦未显示所选窗口的两点平均DY相对峰值附近回落。但第1/2次没有所选窗口的同期载荷，因此不能证明它们确实处于假定的380/280 N。第3次载荷完整，窗口位于接近峰值的平台。', '',
        '| 次数 | 峰值附近两点平均DY mm | 选帧窗口两点平均DY mm | 差值 mm | 同期载荷覆盖 |',
        '|---|---:|---:|---:|---|']
    for row in table.itertuples():
        lines.append(f'| {row.run} | {row.peak_vicinity_point_mean_DY_mm:.6f} | {row.selected_point_mean_DY_mm:.6f} | {row.selected_minus_peak_point_mean_DY_mm:+.6f} | {row.force_coverage_count}/15 |')
    lines += ['', '峰值附近使用既有同步估计的峰值帧四舍五入后±2帧共5帧；选帧仍使用原945—959、465—479、235—249窗口。这只是时序诊断，不重新选择识别数据。两点平均DY含刚体平移，不能等同于轴向伸长；其增加不能单独证明应变没有回落。', '',
        '对现有15帧全场，逐帧拟合DX/DY相对参考X/Y的全局仿射梯度，以dDY/dY作为轴向变形诊断，消除平移贡献；不将它替代原自由二维反演。', '',
        '| 次数 | 轴向仿射应变均值 | 窗口内最小—最大 | 极差/均值 % | 15帧均值低于窗口最大值 % |',
        '|---|---:|---:|---:|---:|']
    for row in table.itertuples():
        lines.append(f'| {row.run} | {row.field_axial_strain_mean:.8f} | {row.field_axial_strain_min:.8f}—{row.field_axial_strain_max:.8f} | {row.strain_range_over_mean_percent:.2f} | {row.mean_strain_below_window_max_percent:.2f} |')
    lines += ['', '窗口内平均处理相对最大仿射应变仅降低约0.88%、0.32%、1.75%，没有20%以上的衰减；这一检查限于现有窗口，不能排除尚未导出的其他时段具有不同全场应变。', '',
        '第3次原始峰值380.008 N，所选窗口同期均值379.649597 N，二者相差0.358403 N，即峰值的约0.094%；扣除本次采用的250.72479 N起载后，差异也仅约0.28%。这不足以解释该次相对默认标定约25%的模量偏高。', '',
        '三次冻结反演观测与原始15帧逐点均值（共用换轴）均完全相等，未乘位移缩小系数。它核验导入实现，不能独立证明DIC软件导出文件名与原图帧号对应无误。峰值附近的完整全场目前未提供，无法直接比较峰值与选帧的全场伸长；第1/2次假定末载的有效性仍未建立。', '',
        '因此目前不应把“峰值之后选帧”直接解释为“位移已经回落导致模量高估”。需在同一参考图下补导出第1次918—922、第2次444—448、第3次221—225附近全场，并补齐第1/2次停留段载荷，才可直接检验。上述是诊断候选，不是已冻结的新反演平台。', '',
        '只读核验，未修改原始数据、γ、原选帧或既有反演场。复现脚本check_frame_timing.py；全部输入哈希见metadata.json。']
    (output/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    import sys
    (output/'metadata.json').write_text(json.dumps(dict(input_sha256=sources,command=sys.argv,
        scope='input attenuation, selected-window temporal stability, and force coverage; not new inversion'),indent=2),encoding='utf-8')
    print(table.to_string(index=False))


if __name__ == '__main__':
    main()
