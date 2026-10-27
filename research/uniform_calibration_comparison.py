"""Compare saved uniform-sample inversions with explicit calibration references.

This post-processing never changes a load, modulus field, nu, or selected gamma.
Run with --summary, --profiles, --audit, --calibration, --secondary-calibration,
and --output NEW_DIRECTORY. Paths refer to the saved CSV/JSON files.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, stdev

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lib.experiment_data import file_sha256


def calibration_record(path: Path, method: str) -> dict:
    table = pd.read_csv(path)
    selected = table[(table.infill_percent == 60) & (table.fit_method == method)]
    if len(selected) != 1:
        raise ValueError(f'{path}: expected exactly one 60% calibration row for {method}')
    record = selected.iloc[0].to_dict()
    if not np.isfinite([record['mean_E_MPa'], record['sd_E_MPa']]).all() or record['mean_E_MPa'] <= 0:
        raise ValueError('Calibration mean must be positive and SD finite')
    # Independently reproduce reference moments from saved individual specimens.
    fits = path.parent/('all_fit_results.csv' if method == 'common_stress_window_OLS' else 'fit_results.csv')
    specimens = pd.read_csv(fits)
    specimens = specimens[(specimens.infill_percent == 60) & (specimens.fit_method == method)]
    values = specimens.E_MPa.dropna().tolist()
    if len(values) != record['n']:
        raise ValueError('Calibration specimen count does not match group table')
    np.testing.assert_allclose([mean(values), stdev(values)],
                               [record['mean_E_MPa'], record['sd_E_MPa']], rtol=1e-12)
    return record


def comparison_table(summary: pd.DataFrame, reference: dict) -> pd.DataFrame:
    """Separate mean bias, within-field spread, and total error against constant E.

    RMS_area(E-Ecal)^2 = (mean_area(E)-Ecal)^2 + var_area(E).
    The supplied moments are the saved Q4 area-integrated values.
    """
    table = summary.copy()
    reference_E = reference['mean_E_MPa']
    table['calibration_method'] = reference['fit_method']
    table['calibration_mean_E_MPa'] = reference_E
    table['calibration_specimen_SD_MPa'] = reference['sd_E_MPa']
    table['calibration_n'] = reference['n']
    table['mean_bias_MPa'] = table.mean_E_MPa-reference_E
    table['mean_bias_percent'] = 100*table.mean_bias_MPa/reference_E
    table['field_relative_RMS_difference_percent'] = (
        100*np.hypot(table.mean_bias_MPa, table.spatial_sd_MPa)/reference_E)
    return table


def compare(summary_path: Path, profiles_path: Path, audit_path: Path,
            calibration_path: Path, secondary_path: Path, output: Path):
    primary = calibration_record(calibration_path, 'common_stress_window_OLS')
    secondary = calibration_record(secondary_path, 'individual_window_joint_3pct_monotone')
    summary = pd.read_csv(summary_path)
    profiles = pd.read_csv(profiles_path)
    audit = json.loads(audit_path.read_text(encoding='utf-8'))
    paths = (summary_path, profiles_path, audit_path, calibration_path, secondary_path)
    original_hashes = {str(p.resolve()): file_sha256(p) for p in paths}
    output.mkdir(parents=True, exist_ok=False)
    tables = [comparison_table(summary, record) for record in (primary, secondary)]
    for name, table in zip(('independent_calibration_comparison', 'constrained_calibration_comparison'), tables):
        table.to_csv(output/f'{name}.csv', index=False, lineterminator='\n')
    primary_rows = tables[0][(tables[0].model == 'H1') & (tables[0].nu == .26)]
    secondary_rows = tables[1][(tables[1].model == 'H1') & (tables[1].nu == .26)]
    colors = {1:'#0072B2', 2:'#D55E00', 3:'#009E73'}
    fig, ax = plt.subplots(figsize=(9, 4.5), layout='constrained')
    mu, sd = primary['mean_E_MPa'], primary['sd_E_MPa']
    ax.axhspan(mu-sd, mu+sd, color='#666666', alpha=.16,
               label='Independent calibration: specimen mean +/- SD (n=4)')
    ax.axhline(mu, color='#333333', linestyle='--', linewidth=1.3,
               label=f'Independent calibration: {mu:.2f} MPa')
    ax.axhline(secondary['mean_E_MPa'], color='#CC79A7', linestyle=':', linewidth=1.4,
               label=f"Constrained window reference: {secondary['mean_E_MPa']:.2f} MPa")
    for run, group in profiles.groupby('run'):
        ax.plot(group.x_mm, group.mean_E_MPa, color=colors[run], label=f'Run {run}: width mean H1 E')
    ax.set(xlabel='Axial x (mm)', ylabel='E (MPa)', title='Saved H1 fields versus 60% calibration')
    ax.legend(fontsize=8, ncol=2, loc='upper right')
    for extension in ('png', 'svg'):
        fig.savefig(output/f'calibration_profiles.{extension}', dpi=300)
    plt.close(fig)
    lines = ['# 60%均匀试样：H1反演与标定模量比较', '',
             '**第3次空间上最接近均匀，但其模量均值仍高于默认独立标定24.96%；三次尚不能认为同时恢复了均匀性与标定模量水平。**', '',
             f"主参照为当前配置采用的1023当前试样60%独立标定：{primary['mean_E_MPa']:.2f}±{primary['sd_E_MPa']:.2f} MPa，n={int(primary['n'])}，方法`{primary['fit_method']}`。±表示不同标定试样的样本SD，不是均值置信区间或允许误差阈值。", '',
             '均值偏差=100×(面积均值E/Ecal−1)；整场相对RMS差异=100×sqrt(面积均值((E−Ecal)²))/Ecal。前者衡量整体偏高/偏低，后者同时计入空间非均匀性。CV仅衡量场内离散，不能代替标定一致性。', '',
             '| 加载 | H1面积均值 MPa | 空间CV % | 相对独立标定均值偏差 % | 整场相对RMS差异 % |',
             '|---|---:|---:|---:|---:|']
    for row in primary_rows.itertuples():
        lines.append(f'| {row.run} | {row.mean_E_MPa:.2f} | {row.spatial_CV_percent:.2f} | {row.mean_bias_percent:+.2f} | {row.field_relative_RMS_difference_percent:.2f} |')
    lines += ['', f"另列后来按组内±3%及填充率单调性联合选段的参照：{secondary['mean_E_MPa']:.2f}±{secondary['sd_E_MPa']:.2f} MPa，n={int(secondary['n'])}，方法`{secondary['fit_method']}`。这些约束参与了标定窗口选择，不能与无约束独立标定混称，也不应因为更接近反演就替换主参照。", '',
              '| 加载 | 相对约束选段参照均值偏差 % | 整场相对RMS差异 % |', '|---|---:|---:|']
    for row in secondary_rows.itertuples():
        lines.append(f'| {row.run} | {row.mean_bias_percent:+.2f} | {row.field_relative_RMS_difference_percent:.2f} |')
    lines += ['', '![沿长度模量与标定参照](calibration_profiles.png)', '',
              '均匀单参数DIC拟合的均值分别为2084.05、1693.03、1473.10 MPa，与自由二维反演的均值很接近。因此整体偏高并非仅由空间非均匀性造成；它已经存在于均匀DIC拟合中。此判断不确定具体误差来源。', '',
              '主ν=0.26，其他ν=0.24/0.28/0.33的完整比较及均匀基线均保存在两张comparison CSV中。相同γ的ν敏感性结果不用于换选主ν或选参。', '',
              '比较范围：标定与DIC并非同一根试样；标定是在原始工程应力—应变曲线上选定线性段的斜率，DIC按用户指定的记录始末力差及参考图位移反演，相应加载区间与参考对应尚未证明等价。独立标定60%共同应力窗口约5.986—8.779 MPa；下表仅由记录起载、采用末载和DIC宽度×3.2 mm计算，不能视为已核实的DIC参考应力。', '',
              '| 次数 | 记录起始应力 MPa | 采用末载应力 MPa | 增量应力 MPa |', '|---|---:|---:|---:|']
    for run, item in audit.items():
        area = item['width_mm']*item['thickness_mm']
        lines.append(f"| {run} | {item['start_force_N']/area:.3f} | {item['end_force_N']/area:.3f} | {item['delta_force_N']/area:.3f} |")
    lines += ['', '以上差异作为标定参照一致性结果报告，不能据此单独归因于反演算法错误、DIC错误或材料真实变化。需进一步核实力—参考帧对应、DIC覆盖宽度与实际承载宽度、实际厚度、标定应力窗口转移以及标量各向同性边界模型。第1/2次末载仍为用户假定380/280 N，第3次起始记录为250.72479 N；不按标定倒调力或γ。', '',
              '本次只对保存结果追加比较，未重新反演、未修改原始数据或既有数值场。标定均值与样本SD已用Python statistics从保存的逐根标定表独立重算核对。输入哈希与命令见metadata.json；源标定CSV的副本保存在本目录。', '',
              '复现：`python -B -m lib.uniform_calibration_comparison --summary <原uniformity_summary.csv> --profiles <原axial_profiles.csv> --audit <原input_audit.json> --calibration <独立group_summary.csv> --secondary-calibration <约束选段group_summary.csv> --output <新目录>`。']
    (output/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    (output/'independent_calibration_source.csv').write_bytes(calibration_path.read_bytes())
    (output/'constrained_calibration_source.csv').write_bytes(secondary_path.read_bytes())
    if original_hashes != {str(p.resolve()): file_sha256(p) for p in paths}:
        raise RuntimeError('An input changed during post-processing')
    import sys
    (output/'metadata.json').write_text(json.dumps(dict(input_sha256=original_hashes,
        command=sys.argv, code_sha256=file_sha256(Path(__file__)),
        primary_reference=primary, secondary_reference=secondary), ensure_ascii=False, indent=2), encoding='utf-8')
    print(primary_rows[['run', 'mean_E_MPa', 'mean_bias_percent', 'field_relative_RMS_difference_percent']].to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ('summary', 'profiles', 'audit', 'calibration', 'secondary-calibration', 'output'):
        parser.add_argument('--'+option, type=Path, required=True)
    args = parser.parse_args()
    compare(args.summary, args.profiles, args.audit, args.calibration, args.secondary_calibration, args.output)
