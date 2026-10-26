"""Audit individual windows and jointly select windows under explicit priors.

The user requested every specimen within 3% of its group mean and nondecreasing
group means. These are selection constraints, not independent validation data.
Source curves, dimensions and earlier calibration outputs are never overwritten.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix
from scipy.stats import linregress

from .tensile_calibration import (
    GEOMETRY_FILE, INFILLS, ROOT, Fit, Settings, Specimen, candidate_windows,
    evaluate_fit, file_sha256, load_specimens, plt, read_geometry,
)


def fit_key(fit: Fit, rule: str) -> tuple[float, ...]:
    width = fit.upper_stress_mpa - fit.lower_stress_mpa
    if rule == "highest_R2":
        return fit.r2, -fit.half_slope_difference, width, -fit.lower_stress_mpa
    if rule == "widest":
        return width, fit.r2, -fit.half_slope_difference, -fit.lower_stress_mpa
    raise ValueError(f"Unknown rule: {rule}")


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    result = []
    for infill in INFILLS:
        group = rows.loc[rows.infill_percent == infill]
        values = group.E_MPa.dropna().to_numpy()
        mean = float(values.mean()) if len(values) else float("nan")
        result.append({
            "infill_percent": infill, "batch": "1023",
            "fit_method": str(group.fit_method.iloc[0]),
            "n_input": len(group), "n": len(values),
            "mean_E_MPa": mean,
            "sd_E_MPa": float(values.std(ddof=1)) if len(values) > 1 else float("nan"),
            "max_absolute_specimen_deviation_percent": float(np.max(np.abs(values / mean - 1)) * 100) if len(values) else float("nan"),
        })
    return pd.DataFrame(result)


def make_rows(specimens: list[Specimen], selected: dict[str, Fit | None], method: str) -> pd.DataFrame:
    rows = []
    for specimen in specimens:
        fit = selected[specimen.specimen_id]
        rows.append({
            "infill_percent": specimen.infill, "specimen_id": specimen.specimen_id,
            "fit_method": method, "E_MPa": fit.modulus_mpa if fit else np.nan,
            "R2": fit.r2 if fit else np.nan,
            "lower_stress_MPa": fit.lower_stress_mpa if fit else np.nan,
            "upper_stress_MPa": fit.upper_stress_mpa if fit else np.nan,
            "half_slope_difference_percent": 100 * fit.half_slope_difference if fit else np.nan,
            "nondecreasing_strain_step_percent": 100 * fit.positive_strain_steps if fit else np.nan,
            "point_count": fit.point_count if fit else 0,
            "width_mm": specimen.width_mm, "thickness_mm": specimen.thickness_mm,
            "gauge_mm": specimen.gauge_mm, "baseline_force_N": specimen.baseline_force_n,
        })
    frame = pd.DataFrame(rows)
    frame["group_mean_E_MPa"] = frame.groupby("infill_percent").E_MPa.transform("mean")
    frame["relative_to_group_mean_percent"] = 100 * (frame.E_MPa / frame.group_mean_E_MPa - 1)
    return frame


def joint_select(specimens: list[Specimen], pools: dict[str, list[Fit]],
                 deviation: float = .03, time_limit: float = 40.) -> tuple[dict[str, Fit], dict]:
    """One binary per candidate and seven continuous group means (MPa).

    Minimize sum(1-R²) with equal weight per specimen. Every specimen selects
    exactly one accepted window. Means equal the arithmetic means of selected
    slopes, deviations are bounded by +/-3%, and means are nondecreasing.
    """
    if any(not pools[s.specimen_id] for s in specimens):
        raise ValueError("At least one specimen has no acceptable candidate")
    candidates = [(s, fit) for s in specimens for fit in pools[s.specimen_id]]
    count = len(candidates)
    mean_index = {infill: count + i for i, infill in enumerate(INFILLS)}
    per_specimen = {s.specimen_id: [] for s in specimens}
    for i, (specimen, fit) in enumerate(candidates):
        per_specimen[specimen.specimen_id].append((i, fit.modulus_mpa))
    row_indices, col_indices, coefficients, lower, upper = [], [], [], [], []

    def add(items: list[tuple[int, float]], lo: float, hi: float) -> None:
        row = len(lower)
        for col, coefficient in items:
            row_indices.append(row)
            col_indices.append(col)
            coefficients.append(coefficient)
        lower.append(lo)
        upper.append(hi)

    for specimen in specimens:
        entries = per_specimen[specimen.specimen_id]
        add([(i, 1.) for i, _ in entries], 1., 1.)
        mean_col = mean_index[specimen.infill]
        add(entries + [(mean_col, -(1 - deviation))], 0., np.inf)
        add(entries + [(mean_col, -(1 + deviation))], -np.inf, 0.)
    for infill in INFILLS:
        group = [s for s in specimens if s.infill == infill]
        add([(i, modulus / len(group)) for s in group
             for i, modulus in per_specimen[s.specimen_id]] + [(mean_index[infill], -1.)], 0., 0.)
    for first, second in zip(INFILLS[:-1], INFILLS[1:]):
        add([(mean_index[first], -1.), (mean_index[second], 1.)], 0., np.inf)
    matrix = coo_matrix((coefficients, (row_indices, col_indices)),
                        shape=(len(lower), count + len(INFILLS))).tocsc()
    # Scale the dimensionless objective only, to avoid very small cost values.
    objective = np.array([1e6 * (1 - fit.r2) for _, fit in candidates] + [0.] * len(INFILLS))
    integrality = np.array([1] * count + [0] * len(INFILLS))
    result = milp(objective, integrality=integrality,
                  bounds=Bounds(np.zeros(count + len(INFILLS)),
                                np.array([1.] * count + [np.inf] * len(INFILLS))),
                  constraints=LinearConstraint(matrix, lower, upper),
                  options={"time_limit": time_limit, "mip_rel_gap": 1e-6})
    audit = {"status": int(result.status), "message": result.message,
             "candidate_count": count, "mip_gap": float(result.mip_gap) if result.x is not None else None,
             "objective_scaled": float(result.fun) if result.x is not None else None,
             "time_limit_seconds": time_limit,
             "objective": "minimize sum(1-R2) over all 28 selected windows; internally multiplied by 1e6"}
    if result.x is None:
        raise RuntimeError(json.dumps(audit))
    chosen = {}
    for specimen in specimens:
        indices = [i for i, _ in per_specimen[specimen.specimen_id] if result.x[i] > .5]
        if len(indices) != 1:
            raise AssertionError("Solver did not return one selected window per specimen")
        chosen[specimen.specimen_id] = candidates[indices[0]][1]
    return chosen, audit


def verify_selected(specimens: list[Specimen], selected: dict[str, Fit]) -> list[dict]:
    checks = []
    for specimen in specimens:
        fit = selected[specimen.specimen_id]
        force = specimen.force_n[:specimen.peak_index + 1]
        extension = specimen.extension_mm[:specimen.peak_index + 1]
        stress = (force - specimen.baseline_force_n) / (specimen.width_mm * specimen.thickness_mm)
        mask = (stress >= fit.lower_stress_mpa) & (stress <= fit.upper_stress_mpa)
        regression = linregress(extension[mask], force[mask])
        independent_E = regression.slope * specimen.gauge_mm / (specimen.width_mm * specimen.thickness_mm)
        assert abs(independent_E / fit.modulus_mpa - 1) < 1e-10
        assert abs(regression.rvalue**2 - fit.r2) < 1e-10
        midpoint = (fit.lower_stress_mpa + fit.upper_stress_mpa) / 2
        halves = [mask & (stress <= midpoint), mask & (stress > midpoint)]
        half_slopes = [linregress(extension[m], force[m]).slope for m in halves]
        half_difference = abs(half_slopes[0] - half_slopes[1]) / regression.slope
        assert abs(half_difference - fit.half_slope_difference) < 1e-10
        # Regression residual is a different quantity from specimen consistency.
        predicted = regression.intercept + regression.slope * extension[mask]
        normalized_rmse = np.sqrt(np.mean((force[mask] - predicted)**2)) / np.ptp(force[mask]) * 100
        checks.append({"specimen_id": specimen.specimen_id,
                       "independent_E_MPa": float(independent_E),
                       "relative_E_difference": float(abs(independent_E / fit.modulus_mpa - 1)),
                       "curve_RMSE_over_force_span_percent": float(normalized_rmse)})
    return checks


def plot_results(directory: Path, specimens: list[Specimen], selected: dict[str, Fit],
                 unconstrained: dict[str, Fit | None], rows: pd.DataFrame, summary: pd.DataFrame) -> None:
    for infill in INFILLS:
        fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
        for ax, specimen in zip(axes.flat, [s for s in specimens if s.infill == infill]):
            strain = specimen.raw_strain[:specimen.peak_index + 1]
            x = (strain - strain[0]) * 100
            y = specimen.stress_mpa[:specimen.peak_index + 1]
            ax.plot(x, y, color=".72", lw=1, label="Pre-peak data")
            endpoints = []
            for fit, color, label, style in (
                (selected[specimen.specimen_id], "#0072B2", "Constrained", "-"),
                (unconstrained[specimen.specimen_id], "#D55E00", "Individual max R2", "--")):
                if fit is None:
                    continue
                mask = (y >= fit.lower_stress_mpa) & (y <= fit.upper_stress_mpa)
                ax.plot(x[mask], y[mask], color=color, lw=2, ls=style,
                        label=f"{label}: {fit.modulus_mpa:.0f} MPa")
                ax.plot(x[mask], fit.modulus_mpa * strain[mask] + fit.intercept_mpa,
                        color=color, lw=.8, ls=":")
                endpoints.extend(x[mask].tolist())
            if endpoints:
                margin = max(.06, np.ptp(endpoints) * .1)
                ax.set_xlim(min(endpoints) - margin, max(endpoints) + margin)
                hi = max(selected[specimen.specimen_id].upper_stress_mpa,
                         unconstrained[specimen.specimen_id].upper_stress_mpa if unconstrained[specimen.specimen_id] else 0)
                ax.set_ylim(0, hi * 1.15)
            ax.set_title(specimen.specimen_id)
            ax.set_xlabel("Axial strain from first sample (%)")
            ax.set_ylabel("Apparent stress (MPa)")
            ax.legend(fontsize=8)
        fig.savefig(directory / f"fit_{infill}.png", dpi=300)
        fig.savefig(directory / f"fit_{infill}.svg")
        plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    axes[0].plot(summary.infill_percent, summary.mean_E_MPa, color="#0072B2", marker="o")
    for row in rows.itertuples():
        axes[0].scatter(row.infill_percent, row.E_MPa, color="#0072B2", s=18, alpha=.6)
    axes[0].set(xlabel="Infill (%)", ylabel="Apparent modulus (MPa)", title="Selected slopes and group means")
    for i, infill in enumerate(INFILLS):
        group = rows.loc[rows.infill_percent == infill]
        axes[1].scatter([i] * len(group), group.relative_to_group_mean_percent, color="#0072B2", s=25)
    axes[1].axhline(3, color="#D55E00", ls="--", lw=1)
    axes[1].axhline(-3, color="#D55E00", ls="--", lw=1)
    axes[1].axhline(0, color=".7", lw=.5)
    axes[1].set_xticks(range(len(INFILLS)), INFILLS)
    axes[1].set(xlabel="Infill (%)", ylabel="Deviation from group mean (%)",
                title="Consistency imposed during selection", ylim=(-3.8, 3.8))
    fig.savefig(directory / "constraint_summary.png", dpi=300)
    fig.savefig(directory / "constraint_summary.svg")
    plt.close(fig)


def run(root: Path, run_id: str, time_limit: float = 40.) -> None:
    if not run_id or run_id in (".", "..") or any(c in run_id for c in '\\/:*?"<>|'):
        raise ValueError("run-id must be a single new directory name")
    output = root / "analysis_results" / run_id
    if output.exists():
        raise FileExistsError(output)
    # Preserve every existing source and numerical output, including formal calibration.
    before = {str(p.relative_to(root)): file_sha256(p) for p in root.rglob("*") if p.is_file()}
    specimens, blocks = load_specimens(root, read_geometry(root / GEOMETRY_FILE))
    assert len(specimens) == 28 and all(sum(s.infill == i for s in specimens) == 4 for i in INFILLS)
    permissive = replace(Settings(), minimum_width_fraction=.05, maximum_half_slope_difference=.10)
    pools, candidate_records = {}, []
    for specimen in specimens:
        fits = [fit for lo, hi in candidate_windows([specimen], permissive)
                if (fit := evaluate_fit(specimen, lo, hi, permissive)) is not None and fit.accepted]
        pools[specimen.specimen_id] = fits
        for fit in fits:
            candidate_records.append({"infill_percent": specimen.infill, **asdict(fit)})

    def eligible(specimen: Specimen, width: float, half: float) -> list[Fit]:
        peak = float(specimen.stress_mpa[:specimen.peak_index + 1].max())
        return [f for f in pools[specimen.specimen_id]
                if f.upper_stress_mpa - f.lower_stress_mpa >= width * peak - 1e-10
                and f.half_slope_difference <= half]

    standards, standard_groups = [], []
    baseline_selected = None
    for width in (.05, .10, .15, .20):
        for half in (.05, .10):
            for rule in ("highest_R2", "widest"):
                name = f"{rule}_width{width:g}_half{half:g}"
                selected = {s.specimen_id: max(available, key=lambda f: fit_key(f, rule)) if (available := eligible(s, width, half)) else None
                            for s in specimens}
                frame = make_rows(specimens, selected, name)
                groups = summarize(frame)
                complete = bool(groups.n.eq(4).all())
                consistent = bool(groups.max_absolute_specimen_deviation_percent.le(3 + 1e-10).all())
                monotone = bool(np.all(np.diff(groups.mean_E_MPa) >= -1e-10))
                standards.append({"standard": name, "minimum_peak_stress_span_fraction": width,
                                  "maximum_half_slope_difference": half, "selection_rule": rule,
                                  "all_28_fitted": complete, "all_groups_within_3_percent": consistent,
                                  "group_means_nondecreasing": monotone,
                                  "meets_all_requirements": complete and consistent and monotone,
                                  "maximum_specimen_deviation_percent": float(groups.max_absolute_specimen_deviation_percent.max())})
                standard_groups.extend(groups.assign(standard=name).to_dict("records"))
                if width == .15 and half == .05 and rule == "highest_R2":
                    baseline_selected = selected
    joint_settings = replace(Settings(), minimum_width_fraction=.10)
    joint_pools = {s.specimen_id: eligible(s, .10, .05) for s in specimens}
    chosen, solver = joint_select(specimens, joint_pools, time_limit=time_limit)
    rows = make_rows(specimens, chosen, "individual_window_joint_3pct_monotone")
    groups = summarize(rows)
    # Check actual rounded-to-binary selections, not solver's continuous means.
    assert len(rows) == 28 and rows.E_MPa.notna().all()
    assert groups.n.eq(4).all()
    assert groups.max_absolute_specimen_deviation_percent.le(3 + 1e-6).all()
    assert np.all(np.diff(groups.mean_E_MPa) >= -1e-6)
    checks = verify_selected(specimens, chosen)
    assert baseline_selected is not None
    original_rows = make_rows(specimens, baseline_selected, "individual_highest_R2_width15pct_half5pct")
    original_groups = summarize(original_rows)
    # Match candidate pools exactly so this comparison isolates the joint priors.
    matched_selected = {s.specimen_id: max(joint_pools[s.specimen_id], key=lambda f: fit_key(f, "highest_R2"))
                        for s in specimens}
    baseline_rows = make_rows(specimens, matched_selected, "individual_highest_R2_width10pct_half5pct")
    baseline_groups = summarize(baseline_rows)
    rows["unconstrained_individual_E_MPa"] = baseline_rows.E_MPa
    rows["change_from_unconstrained_percent"] = 100 * (rows.E_MPa / baseline_rows.E_MPa - 1)
    for source, expected in before.items():
        assert file_sha256(root / source) == expected, source
    metadata = {
        "status": "constraint_selected_estimates; trend and 3pct consistency are imposed, not independently validated",
        "batch": "1023", "settings": asdict(joint_settings),
        "candidate_grid": "Each specimen's own peak: 5%-80%, step 2.5%; minimum stress span 10% of own peak",
        "requirements": {"each_specimen_deviation_from_group_arithmetic_mean": .03,
                         "group_means_nondecreasing": True, "all_28_specimens_included": True},
        "solver": solver, "unconstrained_standards_tested": len(standards),
        "unconstrained_standards_meeting_all_requirements": sum(s["meets_all_requirements"] for s in standards),
        "independent_checks": checks, "source_blocks": blocks,
        "existing_files_sha256_verified_unchanged": before,
        "code_sha256": {p.name: file_sha256(p) for p in (Path(__file__), Path(__file__).with_name("tensile_calibration.py"))},
        "python_executable": sys.executable,
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "scipy": scipy.__version__},
    }
    output.mkdir(parents=True)
    for name, table in (("fit_results", rows), ("group_summary", groups),
                        ("unconstrained_individual_fits", baseline_rows),
                        ("unconstrained_group_summary", baseline_groups),
                        ("original_width15_individual_fits", original_rows),
                        ("original_width15_group_summary", original_groups),
                        ("standard_search", pd.DataFrame(standards)),
                        ("standard_search_groups", pd.DataFrame(standard_groups)),
                        ("accepted_candidate_windows", pd.DataFrame(candidate_records)),
                        ("independent_verification", pd.DataFrame(checks))):
        table.to_csv(output / f"{name}.csv", index=False, encoding="utf-8-sig")
    (output / "analysis_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    plot_results(output, specimens, chosen, matched_selected, rows, groups)
    lines = ["# 逐根线性段与组内一致性约束", "",
        "本运行找到满足每根相对本组均值偏差≤3%、七组均值不减的选段组合，全部28根均纳入。",
        "**这是按用户条件联合选择窗口后的等效模量估计。3%一致性和单调性是选择约束，不是数据独立验证出的性质；不能据此声称原始重复试样的真实离散已降到3%。**", "",
        "## 拟合标准", "",
        "每根以自身峰值应力为基准，在峰前5%–80%范围以2.5%步长枚举窗口，窗口应力跨度至少为该峰值的10%，点数≥60。按实测外截面计算应力、以变形/50 mm标距计算应变，带截距OLS。R²≥0.995、前后半段斜率差/E≤5%、应变非递减步比例≥90%。各曲线起始力中位数仅作常数零点修正。",
        "在所有合格窗口中联合选择：每根恰取一个窗口，各组均值是4根模量的算术平均，每根E位于本组均值的97%–103%，且七组均值依次不减。目标为最小化全部28根的Σ(1−R²)，各根等权。不指定目标模量、不按组使用不同阈值、不删除试样。", "",
        "## 约束结果", "",
        "±为4根之间的样本标准差，最后一列为本次用户定义的一致性误差。", "",
        "| 填充率 | n | E均值 ± SD (MPa) | 最大单根偏差 (%) |", "|---:|---:|---:|---:|"]
    for g in groups.itertuples():
        lines.append(f"| {g.infill_percent}% | {g.n} | {g.mean_E_MPa:.2f} ± {g.sd_E_MPa:.2f} | {g.max_absolute_specimen_deviation_percent:.6f} |")
    lines += ["", "## 无约束对照及解释", "",
        f"在同一批当前数据上审查了{len(standards)}套统一、逐根独立的规则：窗口最小宽度取自身峰值的5%/10%/15%/20%，半段斜率差上限5%/10%，区间按最高R²或最宽合格段排序。满足全部28根、各组±3%和单调趋势的规则数为{metadata['unconstrained_standards_meeting_all_requirements']}。该搜索只覆盖这16套规则，不能证明所有可能的标准都不可行。所有设置及分组结果均保存，不只保留有利情形。",
        "下表使用与约束运行完全相同的10%最小窗口宽度、5%半段斜率差和同一候选池，唯一变化是各根独立取最高R²、不施加3%与单调约束。全部28根均能拟合，组均值已经递增，但组内单根一致性仍不满足3%：", "",
        "| 填充率 | 有效n | E均值 ± SD (MPa) | 最大单根偏差 (%) |", "|---:|---:|---:|---:|"]
    for g in baseline_groups.itertuples():
        lines.append(f"| {g.infill_percent}% | {g.n} | {g.mean_E_MPa:.2f} ± {g.sd_E_MPa:.2f} | {g.max_absolute_specimen_deviation_percent:.2f} |")
    lines += ["", "原15%最小宽度下的独立最高R²结果另存original_width15_group_summary.csv和original_width15_individual_fits.csv，保留最初问题的直接对照。"]
    lines += ["",
        "50-4在15%最小宽度与5%半段斜率差下无合格窗口，约束运行将最小宽度统一改为10%，因此允许其较短的局部线性段。这个改动对全部28根同等适用，但本身也是用本数据筛选的建模选择，需要新数据验证。",
        "同一根曲线可在不同应力阶段表现出不同局部斜率。联合一致性约束会让一些试样选择较晚、较低斜率的区间，另一些选择较早的区间；这些局部表观模量不能自动视为同一材料初始弹性常数。",
        "最终散布较小、曲线残差较小、仪器准确以及实际物理有效性是不同结论。若研究要求不施加组内一致性和制造单调先验，应报告独立逐根结果与实际散布，并复核异常变形记录或补测。", "",
        "## 核验和复现", "",
        f"MILP状态：{solver['status']}，{solver['message']}；相对最优间隙{solver['mip_gap']:.3g}。重新由选中28个斜率计算均值，验证所有偏差和相邻均值差。以scipy.stats.linregress对原始力–变形独立拟合，检查E、R²和半段斜率差；最大E相对差{max(c['relative_E_difference'] for c in checks):.3g}。",
        f"原有{len(before)}个文件SHA-256未变。原标定及日常DIC配置保持原样。本结果独立保存，不覆盖原参数。仪器通道按项目既有N、mm口径解释，本次不识别泊松比。", "",
        "```powershell",
        "& 'C:/Users/zhu_g/.conda/envs/abqinter/python.exe' -m lib.tensile_fit_standard --run-id <新的运行名>",
        "```", "",
        "本工具需要支持scipy.optimize.milp的SciPy（≥1.9）；本次环境见analysis_metadata.json。全部候选段、16套规则审查、逐根窗口、分组汇总和独立核验CSV均位于同目录，fit_40至fit_100为逐根窗口对照图，constraint_summary为约束结果总览。PNG与SVG由同一图状态输出。",
    ]
    (output / "analysis_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(groups.to_string(index=False))
    print(json.dumps(solver))
    print("Output:", output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--time-limit", type=float, default=40.)
    args = parser.parse_args()
    if args.time_limit <= 0:
        parser.error("time-limit must be positive")
    run(args.root.resolve(), args.run_id, args.time_limit)
