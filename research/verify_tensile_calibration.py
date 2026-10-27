"""Independently check a saved tensile run and write its reproducibility report.

Run: python -m lib.verify_tensile_calibration --run-id <saved run>
The independent regression uses raw force on extensometer extension and then
E = (L0 / A0) * dF/d(extension), with scipy rather than the fitting routine.
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy.stats import linregress

from .tensile_calibration import ROOT, Settings, analyze, file_sha256


def verify(root: Path, run_id: str) -> None:
    directory = root / "analysis_results" / run_id
    targets = [directory / name for name in (
        "all_fit_results.csv", "independent_verification.csv",
        "verification.json", "analysis_report.md")]
    if any(path.exists() for path in targets):
        raise FileExistsError("Verification outputs already exist; retain the saved report")
    metadata = json.loads((directory / "analysis_metadata.json").read_text(encoding="utf-8"))
    summary = pd.read_csv(directory / "group_summary.csv")
    rows = pd.concat([
        pd.read_csv(root / str(int(infill)) / "analysis_results" / run_id / "fit_results.csv")
        for infill in summary.infill_percent
    ], ignore_index=True)
    source_checks = {}
    for source, expected in metadata["source_sha256"].items():
        path = Path(source)
        if not path.is_absolute():
            path = root / path
        source_checks[source] = file_sha256(path) == expected
    if not all(source_checks.values()):
        raise AssertionError("Source data changed after fitting")
    if file_sha256(Path(__file__).with_name("tensile_calibration.py")) != metadata["script_sha256"]:
        raise AssertionError("Fitting implementation changed after this run")

    comparisons = []
    drawdowns = {}
    for note in metadata["source_blocks"]:
        raw = pd.read_excel(root / note["source"], engine="pyxlsb", header=None)
        start = 4 * (note["selected_block"] - 1)
        values = raw.iloc[1:, start:start + 4].apply(pd.to_numeric, errors="coerce")
        values = values.dropna(subset=values.columns[[0, 1, 3]]).to_numpy(float)
        row = rows.loc[rows.specimen_id == note["specimen"]].iloc[0]
        baseline = float(np.median(values[:10, 1]))
        assert math.isclose(baseline, row.baseline_force_N, abs_tol=1e-12)
        peak = int(np.argmax(values[:, 1] - baseline))
        extension = values[:peak + 1, 3]
        drawdowns[row.specimen_id] = float(np.max(np.maximum.accumulate(extension) - extension))
        if not math.isfinite(row.E_MPa):
            continue
        force = values[:peak + 1, 1]
        area = row.width_mm * row.thickness_mm
        stress = (force - baseline) / area
        selection = (stress >= row.lower_stress_MPa) & (stress <= row.upper_stress_MPa)
        fit = linregress(extension[selection], force[selection])
        independent_E = float(fit.slope * row.gauge_mm / area)
        relative_difference = abs(independent_E / row.E_MPa - 1)
        assert relative_difference < 1e-10, (row.specimen_id, relative_difference)
        assert abs(fit.rvalue**2 - row.R2) < 1e-10
        comparisons.append({
            "specimen_id": row.specimen_id,
            "saved_E_MPa": row.E_MPa,
            "raw_force_extension_E_MPa": independent_E,
            "relative_difference": relative_difference,
            "point_count": int(selection.sum()),
        })
    group_checks = []
    for group in summary.itertuples(index=False):
        values = rows.loc[(rows.infill_percent == group.infill_percent) & rows.E_MPa.notna(), "E_MPa"].tolist()
        assert len(values) == group.n
        assert math.isclose(statistics.mean(values), group.mean_E_MPa, rel_tol=1e-12)
        assert math.isclose(statistics.stdev(values), group.sd_E_MPa, rel_tol=1e-12)
        group_checks.append(int(group.infill_percent))
    # Verify overwrite refusal before any read or write of a completed run.
    try:
        analyze(root, run_id, Settings())
    except FileExistsError:
        overwrite_refused = True
    else:
        raise AssertionError("Completed run was not protected")
    # Retain all accepted specimens and explicitly list exclusions.
    excluded = rows.loc[rows.E_MPa.isna(), "specimen_id"].tolist()
    result = {
        "source_hash_checks": source_checks,
        "accepted_specimens_independently_verified": len(comparisons),
        "maximum_relative_modulus_difference": max(c["relative_difference"] for c in comparisons),
        "group_mean_and_sample_sd_checks": group_checks,
        "overwrite_refused": overwrite_refused,
        "excluded_specimens": excluded,
        "prepeak_extension_maximum_drawdown_mm": drawdowns,
        "verification_script_sha256": file_sha256(Path(__file__)),
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "scipy_version": scipy.__version__,
        "limitations": [
            "Numerical regression and data integrity checks do not establish instrument accuracy.",
            "N and mm instrument units follow the existing project convention; raw headers contain channel names only.",
            "Apparent modulus uses the measured outer area, not the solid polymer area.",
            "Each infill has its own accepted stress window; cross-infill differences are not a matched-window material comparison.",
            "Uniaxial force and axial extension alone do not identify Poisson ratio.",
        ],
    }
    lines = [
        "# 当前 28 根打印试样的拉伸模量标定", "",
        f"运行标识：`{run_id}`。批次标签：`{metadata['batch']}`，来自用户指定的当前试样目录；标签本身不证明试验日期。", "",
        f"共读取 {len(rows)} 根曲线，{len(comparisons)} 根通过默认线性条件，{len(excluded)} 根保留为不可用。原始工作簿与旧分析结果均未覆盖。", "",
        "以逐根实测宽度×厚度作为外截面积 A0，以变形通道除以记录标距 L0=50 mm 得到工程应变。",
        "计算 σ=(F−F0)/A0，ε=ΔL/L0，以带截距 OLS 拟合 σ=Eε+c。F0 为各曲线前 10 点力的中位数。",
        "所得 E 是该打印结构和选定应力区间内的等效／表观杨氏模量，单位 MPa；不是实体聚合物模量。",
        "力和变形通道分别按项目既有 N、mm 口径解释；原始曲线表头只有通道名称，没有单位字段。", "",
        "## 分组结果", "",
        "下表 ± 为有效试样之间的样本标准差（ddof=1），不是均值置信区间。", "",
        "| 填充率 | 有效/输入 n | E 均值 ± SD (MPa) | CV (%) | 共同应力窗口 (MPa) |",
        "|---:|---:|---:|---:|---:|",
    ]
    for group in summary.itertuples(index=False):
        audit = next(a for a in metadata["groups"] if a["infill_percent"] == group.infill_percent)
        lower, upper = audit["common_stress_interval_mpa"]
        lines.append(f"| {group.infill_percent}% | {group.n}/{group.n_input} | {group.mean_E_MPa:.2f} ± {group.sd_E_MPa:.2f} | {group.cv_percent:.2f} | {lower:.3f}–{upper:.3f} |")
    lines += ["", "## 拟合条件与边界", "",
        "搜索区间为组内最小峰值应力的 5%–80%，步长为该峰值的 2.5%，窗口宽度至少为其 15%。每个区间至少 60 点，R²≥0.995，前后半段斜率差/E≤5%，应变非递减步比例≥90%。先最大化通过根数，再选择最宽共同窗口；不合格试样仅在相同规则下寻找邻近可接受窗口。未用旧批次结果或期望的填充率趋势选参数。", "",
        f"不可用试样：{', '.join(excluded) if excluded else '无'}。不可用值保留为空，不以零补齐，也不进入均值。",
    ]
    if "50-4" in excluded:
        lines += [
            f"50-4 的峰前变形最大回退为 {drawdowns['50-4']:.5f} mm，默认条件下没有可接受窗口；该记录不足以单独判定回退来自引伸计滑移、装夹或其他原因。",
            "50% 组若把半段斜率差阈值放宽至 10%，会切换到更高应力窗口并纳入 50-4，因此该对照同时改变了窗口和样本集合。默认结果仍按 5% 阈值与 3 根有效试样报告。",
        ]
    lines += ["", "| 填充率 | 窗口平移/裁剪最大单根变化 (%) | 放宽至10%后的组均值变化 (%) |", "|---:|---:|---:|"]
    for audit in metadata["groups"]:
        changes = [abs(value) for variant in audit["window_sensitivity_percent"].values() for value in variant.values()]
        lines.append(f"| {audit['infill_percent']}% | {max(changes):.3f} | {audit['relaxed_10pct_vs_primary_mean_change_percent']:.3f} |")
    lines += ["",
        "窗口敏感性分别将共同窗口左右平移其宽度的 10%，以及两端各裁掉 15%；这里只衡量替代区间斜率变化，不宣称替代区间全部通过原线性阈值。10% 阈值对照未用于替换默认结果。",
        "60% 默认均值低于 50%，但各组采用的应力窗口不同，且50%组对线性阈值更敏感；不能据此直接认定材料本构随填充率非单调。保留分组参数与原始曲线，不强制拟合单调趋势线。",
        "只有轴向力和变形，尚不能识别泊松比；没有将既有 ν 假设包装为本次测量结果。", "",
        "## 独立核验与复现", "",
        f"以 scipy.stats.linregress 对原始 F–ΔL 独立拟合，再用 E=(L0/A0)·dF/dΔL 换算。{len(comparisons)} 根有效结果全部与保存的 E、R² 一致，最大 E 相对差 {result['maximum_relative_modulus_difference']:.3g}。7 组均值和样本标准差以 Python statistics 独立重算一致。",
        f"{len(source_checks)} 个输入文件 SHA-256 与拟合元数据一致；重复运行同一标识被拒绝。数值核验不能代替仪器精度、截面和变形测量的独立物理校验。", "",
        "环境：`abqinter`；精确 Python 路径和版本见 verification.json，其余依赖版本、输入与拟合脚本哈希见 analysis_metadata.json。", "",
        "在项目根目录运行（重新拟合须换 run-id）：", "", "```powershell",
        "& 'C:/Users/zhu_g/.conda/envs/abqinter/python.exe' -m lib.tensile_calibration --root post_tensile_test/1023 --geometry-file '28个狗棒试样实验记录表.xlsx' --batch 1023 --run-id <新运行名>",
        "& 'C:/Users/zhu_g/.conda/envs/abqinter/python.exe' -m lib.verify_tensile_calibration --run-id <新运行名>",
        "```", "",
        "输出：本目录 group_summary.csv、all_fit_results.csv、independent_verification.csv；逐组拟合图及原始力–机器位移图位于各填充率目录 analysis_results/同运行名；总览图位于 comparison_results/同运行名。PNG 与 SVG 来自同一图状态。", "",
        "本次只进行独立拉伸标定，未启动 DIC 正演或反演，也未自动改动尚未配对的 DIC 输入配置。",
    ]
    rows.to_csv(targets[0], index=False, encoding="utf-8-sig")
    pd.DataFrame(comparisons).to_csv(targets[1], index=False, encoding="utf-8-sig")
    targets[2].write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    targets[3].write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in (
        "accepted_specimens_independently_verified", "maximum_relative_modulus_difference",
        "overwrite_refused", "excluded_specimens")}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--run-id", required=True)
    arguments = parser.parse_args()
    verify(arguments.root.resolve(), arguments.run_id)
