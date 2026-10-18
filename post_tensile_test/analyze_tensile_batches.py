"""Process 1016 using the 1013 fitter and evaluate 60/90% batch bridges.

Run with the abqinter Python environment. Every run has isolated outputs.
The legacy fitter remains unchanged; both independent and matched-window
results are reported so a fitting-window change cannot masquerade as a batch
effect. The 1013 70/80% groups are diagnostic only, never pooled by this script.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import platform
import sys
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parent
LEVELS = (60, 70, 80, 90)
BRIDGES = (60, 90)
COLORS = {"1013": "#0072B2", "1016": "#D55E00"}


def load_core(root: Path):
    path = root / "1013" / "analyze_youngs_modulus.py"
    spec = importlib.util.spec_from_file_location("legacy_tensile", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load legacy fitter: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, value: object) -> None:
    def finite_json(item):
        if isinstance(item, dict):
            return {str(k): finite_json(v) for k, v in item.items()}
        if isinstance(item, (list, tuple)):
            return [finite_json(v) for v in item]
        if isinstance(item, np.generic):
            return finite_json(item.item())
        if isinstance(item, float) and not math.isfinite(item):
            return None
        return item

    path.write_text(json.dumps(finite_json(value), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def save_figure(fig, destination: Path) -> None:
    fig.savefig(destination.with_suffix(".png"), dpi=240, bbox_inches="tight")
    fig.savefig(destination.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def fit_row(specimen, fit, batch: str, method: str) -> dict:
    return {
        "batch": batch,
        "infill_percent": specimen.infill,
        "specimen_id": specimen.specimen_id.split("/")[-1],
        "E_MPa": fit.modulus_mpa if fit is not None else np.nan,
        "R2": fit.r2 if fit is not None else np.nan,
        "strain_min_pct": fit.strain_min * 100 if fit is not None else np.nan,
        "strain_max_pct": fit.strain_max * 100 if fit is not None else np.nan,
        "force_min_N": fit.force_min_n if fit is not None else np.nan,
        "force_max_N": fit.force_max_n if fit is not None else np.nan,
        "fit_method": method,
    }


def summary(values) -> dict:
    a = np.asarray(list(values), dtype=float)
    a = a[np.isfinite(a)]
    return {
        "n": len(a),
        "mean_E_MPa": float(a.mean()) if len(a) else np.nan,
        "sd_E_MPa": float(a.std(ddof=1)) if len(a) > 1 else np.nan,
        "cv_percent": float(100 * a.std(ddof=1) / a.mean()) if len(a) > 1 else np.nan,
    }


def mean_relative_deviation(rows: list[dict]) -> pd.DataFrame:
    """Error of representing specimen E_i by its group mean, denominator E_i.

    This is within-sample dispersion, not measurement error against true E.
    Batch and selection method remain separate populations throughout.
    """
    frame = pd.DataFrame(rows).copy()
    keys = ["batch", "infill_percent", "fit_method"]
    mean = frame.groupby(keys).E_MPa.transform("mean")
    frame["group_mean_E_MPa"] = mean
    frame["mean_minus_specimen_relative_percent"] = 100 * (mean / frame.E_MPa - 1)
    frame["absolute_relative_percent"] = frame.mean_minus_specimen_relative_percent.abs()
    frame["specimen_minus_mean_over_mean_percent"] = 100 * (frame.E_MPa / mean - 1)
    return frame


def compare_batches(rows: list[dict]) -> list[dict]:
    comparisons = []
    for level in LEVELS:
        a = summary(r["E_MPa"] for r in rows if r["batch"] == "1013" and r["infill_percent"] == level)
        b = summary(r["E_MPa"] for r in rows if r["batch"] == "1016" and r["infill_percent"] == level)
        comparisons.append({"infill_percent": level, **{"old_"+k:v for k,v in a.items()},
                            **{"new_"+k:v for k,v in b.items()},
                            "arithmetic_mean_difference_percent": 100 * (b["mean_E_MPa"] / a["mean_E_MPa"] - 1)})
    return comparisons


def plot_individual_optimal(group, fits, destination: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for color, specimen in zip(plt.cm.tab10(np.linspace(0, 1, len(group))), group):
        fit = fits[specimen.specimen_id]
        if fit is None:
            continue
        n = specimen.peak_index + 1
        stress = specimen.stress_mpa[:n]
        strain = (specimen.raw_strain[:n] - fit.strain_zero) * 100
        mask = (stress >= fit.lower_stress_mpa) & (stress <= fit.upper_stress_mpa)
        ax.plot(strain, stress, color=color, alpha=0.18, lw=0.8)
        ax.plot(strain[mask], stress[mask], color=color, lw=2,
                label=f"{specimen.specimen_id}: {fit.modulus_mpa:.0f} MPa")
    ax.set(xlabel="Engineering strain (%)", ylabel="Apparent engineering stress (MPa)",
           title=f"{group[0].infill}% — each specimen selects its own interval")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=9)
    save_figure(fig, destination / "individual_optimal_fit_curves")


def log_ratio_interval(old, new, confidence: float = 0.95) -> tuple[float, float, float]:
    """Welch t interval for the ratio of geometric means of positive E.

    This interval describes sampling uncertainty, not instrument/geometry error.
    It is not an ANOVA or a test of differences between infill levels.
    """
    old, new = np.log(np.asarray(old, dtype=float)), np.log(np.asarray(new, dtype=float))
    if min(len(old), len(new)) < 2:
        return np.nan, np.nan, np.nan
    va, vb = old.var(ddof=1) / len(old), new.var(ddof=1) / len(new)
    se = math.sqrt(va + vb)
    df = (va + vb) ** 2 / (va**2 / (len(old) - 1) + vb**2 / (len(new) - 1)) if se else np.inf
    delta = new.mean() - old.mean()
    radius = float(stats.t.ppf((1 + confidence) / 2, df)) * se
    return tuple(float(math.exp(x)) for x in (delta, delta - radius, delta + radius))


def common_window_study(core, combined, settings) -> tuple[dict, dict, list]:
    """Use the original coverage/width rule, with batch-unique specimen IDs.

    Individual fallback fits may be useful diagnostically but are excluded
    from the matched-window comparison. The candidate audit retains every
    interval passing *all* specimens, rather than selecting for agreement.
    """
    _, audit = core.select_group_fits(combined, settings)
    lower, upper = audit["common_stress_interval_mpa"]
    fits = {}
    for specimen in combined:
        f = core.evaluate_fit(specimen, lower, upper, settings)
        fits[specimen.specimen_id] = f if f is not None and f.accepted else None
    candidates = []
    for lo, hi in core.candidate_windows(combined, settings):
        trial = [core.evaluate_fit(s, lo, hi, settings) for s in combined]
        if not all(f is not None and f.accepted for f in trial):
            continue
        by_batch = {
            b: [f.modulus_mpa for s, f in zip(combined, trial) if s.specimen_id.startswith(b + "/")]
            for b in ("1013", "1016")
        }
        candidates.append({
            "infill_percent": combined[0].infill,
            "stress_min_MPa": lo,
            "stress_max_MPa": hi,
            "mean_1013_MPa": float(np.mean(by_batch["1013"])),
            "mean_1016_MPa": float(np.mean(by_batch["1016"])),
            "difference_percent": float(100 * (np.mean(by_batch["1016"]) / np.mean(by_batch["1013"]) - 1)),
        })
    audit["all_specimen_accepted_window_count"] = len(candidates)
    audit["all_accepted_window_difference_percent_min"] = min((x["difference_percent"] for x in candidates), default=np.nan)
    audit["all_accepted_window_difference_percent_max"] = max((x["difference_percent"] for x in candidates), default=np.nan)
    audit["comparison_includes_only_exact_common_window"] = True
    return fits, audit, candidates


def plot_batch_group(core, group, fits, audit, destination: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.7), constrained_layout=True)
    for batch in ("1013", "1016"):
        members = [s for s in group if s.specimen_id.startswith(batch + "/")]
        for index, specimen in enumerate(members):
            fit = fits[specimen.specimen_id]
            label = f"{batch}, n={len(members)}" if index == 0 else None
            axes[0].plot(specimen.displacement_mm, specimen.force_n, color=COLORS[batch], alpha=0.7, lw=1, label=label)
            axes[1].plot(specimen.extension_mm - specimen.extension_mm[0], specimen.force_n,
                         color=COLORS[batch], alpha=0.7, lw=1)
            if fit is None:
                continue
            end = specimen.peak_index + 1
            stress = specimen.stress_mpa[:end]
            strain = (specimen.raw_strain[:end] - fit.strain_zero) * 100
            mask = (stress >= fit.lower_stress_mpa) & (stress <= fit.upper_stress_mpa)
            axes[2].plot(strain[mask], stress[mask], color=COLORS[batch], lw=1.3, alpha=0.8)
    lo, hi = audit["common_stress_interval_mpa"]
    axes[0].set(xlabel="Machine displacement (mm)", ylabel="Raw force (N)", title="Original force vs machine displacement")
    axes[1].set(xlabel="Extensometer extension from first point (mm)", ylabel="Raw force (N)", title="Original force vs extensometer extension")
    axes[2].set(xlabel="Engineering strain (%)", ylabel="Apparent engineering stress (MPa)", title=f"Matched fit: {lo:.2f}–{hi:.2f} MPa")
    axes[0].legend(fontsize=9)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", lw=0.4, alpha=0.2)
    infill = group[0].infill
    purpose = "bridge validation" if infill in BRIDGES else "old batch diagnostic only"
    fig.suptitle(f"{infill}% infill — {purpose}", fontsize=13)
    save_figure(fig, destination / "batch_comparison")


def plot_modulus_comparison(rows: list[dict], destination: Path) -> None:
    frame = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), constrained_layout=True, sharey=True)
    for axis, method in zip(axes, ("independent", "matched")):
        for batch, offset, marker in (("1013", -1.3, "o"), ("1016", 1.3, "s")):
            for level in LEVELS:
                a = frame[(frame.batch == batch) & (frame.infill_percent == level) & (frame.fit_method == method)].E_MPa.dropna().to_numpy()
                if not len(a):
                    continue
                axis.scatter(level + offset + np.linspace(-0.45, 0.45, len(a)), a, s=23, marker=marker, color=COLORS[batch], alpha=0.65)
                axis.errorbar(level + offset, a.mean(), yerr=a.std(ddof=1) if len(a) > 1 else 0,
                              marker=marker, color=COLORS[batch], capsize=4, markersize=7, lw=1.4,
                              label=f"{batch}: mean ± SD" if level == 60 else None)
        axis.set(xticks=LEVELS, xlabel="Infill (%)", title="Original independent window selection" if method == "independent" else "Same stress window within each infill")
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", lw=0.5, alpha=0.2)
        axis.legend(fontsize=8)
    axes[0].set_ylabel("Apparent fitted modulus (MPa)")
    fig.suptitle("Batch comparison — 60/90% are bridges; old 70/80% are diagnostic", fontsize=12)
    save_figure(fig, destination / "batch_modulus_comparison")


def plot_bridge_difference(comparisons: list[dict], destination: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.5), constrained_layout=True)
    for bound, color in ((15, "#F6F6F6"), (10, "#E8F0E9"), (5, "#D5E5D7")):
        ax.axvspan(-bound, bound, color=color, zorder=0)
    for y, row in enumerate([r for r in comparisons if r["role"] == "bridge"]):
        mid = 100 * (row["geometric_ratio"] - 1)
        lo = 100 * (row["geometric_ratio_ci95_low"] - 1)
        hi = 100 * (row["geometric_ratio_ci95_high"] - 1)
        ax.errorbar(mid, y, xerr=[[mid - lo], [hi - mid]], fmt="o", color="#333333", capsize=5, markersize=7)
        ax.annotate(f"mean shift {row['arithmetic_mean_difference_percent']:+.1f}%", (mid, y), xytext=(0, 12), textcoords="offset points", ha="center", fontsize=9)
    ax.axvline(0, color="black", lw=0.8)
    ax.set(yticks=[0, 1], yticklabels=["60% bridge", "90% bridge"], xlabel="1016 / 1013 geometric mean modulus − 1 (%)", ylim=(-0.55, 1.55))
    ax.set_title("Matched-window batch differences and 95% intervals", fontsize=12)
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(0.5, -0.03, "Bands: ±5%, ±10%, ±15% reference tolerances; no application tolerance was supplied.", ha="center", fontsize=8)
    save_figure(fig, destination / "bridge_difference")


def plot_primary_calibration(old_rows: list[dict], new_rows: list[dict], destination: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.8), constrained_layout=True)
    # A review view only: no unlabelled pooled dataset is produced.
    for batch, rows, levels, marker in (("1013", old_rows, (40, 50, 60, 90, 100), "o"),
                                        ("1016", new_rows, LEVELS, "s")):
        for level in levels:
            values = [r["E_MPa"] for r in rows if r["infill_percent"] == level and math.isfinite(r["E_MPa"])]
            ax.errorbar(level + (-0.8 if batch == "1013" else 0.8), np.mean(values), yerr=np.std(values, ddof=1),
                        marker=marker, color=COLORS[batch], capsize=4, label=batch if level == levels[0] else None)
    ax.set(xticks=[40, 50, 60, 70, 80, 90, 100], xlabel="Infill (%)", ylabel="Apparent fitted modulus (MPa)", title="Review view: 1016 fills the 70/80% calibration gap")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", lw=0.5, alpha=0.2)
    ax.legend(title="Batch; mean ± SD")
    save_figure(fig, destination / "calibration_review")


def plot_new_summary(rows: list[dict], destination: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    for level in LEVELS:
        values = np.array([r["E_MPa"] for r in rows if r["infill_percent"] == level and math.isfinite(r["E_MPa"])])
        ax.scatter(level + np.linspace(-0.7, 0.7, len(values)), values, s=25, color=COLORS["1016"], alpha=0.6)
        ax.errorbar(level, values.mean(), yerr=values.std(ddof=1), fmt="s", color=COLORS["1016"], capsize=4, markersize=7)
    ax.set(xticks=LEVELS, xlabel="Infill (%)", ylabel="Apparent fitted modulus (MPa)", title="1016 batch — specimen values and mean ± SD")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", lw=0.5, alpha=0.2)
    save_figure(fig, destination / "modulus_by_infill")


def write_report(output: Path, figures: Path, new_summary, new_audits, comparisons, independent_comparisons, diagnostics, metadata, individual_comparisons) -> None:
    """Write the interpretation beside the measurements, with batch roles explicit."""
    independent = {row["infill_percent"]: row for row in independent_comparisons}
    lines = [
        "# 1013 与 1016 拉伸模量及桥接验证", "",
        "1013 脚本移入新目录后运行正常，33 根试样的原结果精确复现。1016 的 16 根均匀填充试样已按同一算法处理。",
        "60% 与 90% 是桥接验证点；1016 的 70% 与 80% 是主要补测数据；1013 的 70%/80% 仅作疑似异常组的诊断对照，不参与平均合并。", "",
        "## 计算条件", "",
        "1016 使用用户提供的统一尺寸：宽 9.89 mm、厚 3.26 mm、标距 50 mm。1013 使用原尺寸表中每根试样的实测尺寸。1016 未提供逐根尺寸，因此新批次小幅几何差异未纳入不确定度。",
        "工程应变为引伸计变形 / 标距，应力为扣除初始力偏置后的力 / 外轮廓初始截面积。含截距 OLS 拟合得到局部表观模量；拟合后的横向零点平移不改变斜率。横梁位移不用于计算模量。",
        "沿用原筛选设置：峰值前搜索、每段至少 60 点、R²≥0.995、前后半段斜率差≤5%、非递减应变步比例≥90%。同填充率优先共同窗口，按通过数、窗口宽度等排序；未以两批接近程度作为选段准则。",
        "共同窗口比较只保留在完全相同应力范围内通过筛选的结果；个别试样的替代窗口不会混入该比较。高 R² 只能确认局部线性，不能独立证明初始弹性、应变传感器无偏或卸载可恢复。", "",
        "## 1016 原方法结果", "",
        "下表均为均值 ± 组内样本标准差，单位 MPa。", "",
        "| 填充率 | n | E (MPa) | CV | 共同拟合应力 (MPa) |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    audit_by_level = {a["infill_percent"]: a for a in new_audits}
    for row in new_summary.to_dict("records"):
        lo, hi = audit_by_level[row["infill_percent"]]["common_stress_interval_mpa"]
        lines.append(f"| {row['infill_percent']}% | {row['n']} | {row['mean_E_MPa']:.1f} ± {row['sd_E_MPa']:.1f} | {row['cv_percent']:.2f}% | {lo:.2f}–{hi:.2f} |")
    lines.extend(["", "## 跨批次桥接比较", "",
        "各自按原算法选段的均值不能单独证明批次一致，因为两批可能落在不同应力阶段。下表给出独立选段表观差异与相同应力窗口下的差异。", "",
        "| 填充率及用途 | 独立选段均值差 | 共同应力窗口 (MPa) | 1013 E (MPa) | 1016 E (MPa) | 同窗差异 | 所有全员通过窗口的差异范围 |",
        "| --- | ---: | --- | ---: | ---: | ---: | --- |"])
    for row in comparisons:
        level = row["infill_percent"]
        role = "桥接" if level in BRIDGES else "诊断"
        lines.append(f"| {level}% {role} | {independent[level]['arithmetic_mean_difference_percent']:+.2f}% | {row['stress_min_MPa']:.2f}–{row['stress_max_MPa']:.2f} | {row['old_mean_E_MPa']:.1f} ± {row['old_sd_E_MPa']:.1f} | {row['new_mean_E_MPa']:.1f} ± {row['new_sd_E_MPa']:.1f} | {row['arithmetic_mean_difference_percent']:+.2f}% | {row['all_candidate_difference_percent_min']:+.2f}% 至 {row['all_candidate_difference_percent_max']:+.2f}%（{row['all_candidate_count']} 个窗口） |")
    lines.extend(["", "### 每根试样各自选最优段的补充对照", "",
        "上面的独立选段是每批同填充率优先共用窗口，不是每根试样独立。此处对每根试样分别调用同一筛选器：R²、点数、曲率与应变单调性阈值不变，通过后优先最宽窗口，R²等用于打破并列。搜索范围为各自峰值应力的5%–80%。不按两批接近程度、最大斜率或单独最大R²选段。",
        "该比较允许所有试样处于不同应力阶段，回答所定义选段算法下的局部表观模量是否接近，不能独立证明制造批次等效。", "",
        "| 填充率 | 1013 E (MPa) | 1016 E (MPa) | 新批相对变化 |",
        "| --- | ---: | ---: | ---: |"])
    for row in individual_comparisons:
        lines.append(f"| {row['infill_percent']}% | {row['old_mean_E_MPa']:.1f} ± {row['old_sd_E_MPa']:.1f} | {row['new_mean_E_MPa']:.1f} ± {row['new_sd_E_MPa']:.1f} | {row['arithmetic_mean_difference_percent']:+.2f}% |")
    lines.extend(["", "这两种非跨批同窗的结果均已保留：independent_batch_comparison.csv 为原同组优先共窗方法，individual_optimal_batch_comparison.csv 为逐根独立方法。原同组方法仍保留为主结果，未因新比较更接近就替换主方法。",
        "60% 原同组方法独立选段时两批均值仅差约 0.74%，但共同窗口下差约 12.85%；在所有全员通过的候选窗口内，该差异仍为约 11.18%–14.69%。60% 的接近依赖于选段，不能作为两批一致的充分依据。",
        "90% 共同窗口差约 −2.89%，所有全员通过窗口范围约 −3.12% 至 −2.18%，就局部模量而言接近。", "",
        "采样不确定度以 log(E) 上的 Welch t 区间估计，再还原为几何均值比；它与上表算术均值百分差不是同一个量。未做填充率之间的显著性检验。95% 区间仅反映这些试样的采样变化，不含传感器、统一尺寸或拟合方法的系统误差。", ""])
    for row in comparisons:
        if row["infill_percent"] in BRIDGES:
            lines.append(f"- {row['infill_percent']}%：1016/1013 几何均值比变化的 95% 区间为 {100*(row['geometric_ratio_ci95_low']-1):+.2f}% 至 {100*(row['geometric_ratio_ci95_high']-1):+.2f}%。")
    lines.extend(["", "±5%、±10%、±15% 是报告敏感性用的参考带，用户尚未给定实际允许误差，不能把参考带当成已确定的工程等效标准。样本量分别只有旧批 4–5 根、新批 3 根，区间仍存在较大不确定性。", "",
        "## 不依赖引伸计的对照", "",
        "同取原始 150–400 N 加载区间计算横梁力–位移刚度，作为辅助诊断；其中包含试样标距外、机器与夹具的变形，不是杨氏模量。峰值力是承载强度指标，也不能替代模量。", "",
        "| 桥接填充率 | 1013 横梁刚度 (N/mm) | 1016 横梁刚度 (N/mm) | 变化 | 峰值力均值 1013→1016 (N) | 变化 |",
        "| --- | ---: | ---: | ---: | --- | ---: |"])
    diagframe = pd.DataFrame(diagnostics)
    for level in BRIDGES:
        a = diagframe[(diagframe.batch == "1013") & (diagframe.infill_percent == level)]
        b = diagframe[(diagframe.batch == "1016") & (diagframe.infill_percent == level)]
        ka, kb = a.crosshead_stiffness_150_400N_N_per_mm.mean(), b.crosshead_stiffness_150_400N_N_per_mm.mean()
        fa, fb = a.peak_raw_force_n.mean(), b.peak_raw_force_n.mean()
        lines.append(f"| {level}% | {ka:.2f} | {kb:.2f} | {100*(kb/ka-1):+.2f}% | {fa:.2f} → {fb:.2f} | {100*(fb/fa-1):+.2f}% |")
    lines.extend(["", "两批桥接组在主要加载区间的横梁速度均约为 2 mm/min。60% 的独立横梁刚度对照也偏高，说明差别并非仅由引伸计选段产生；仍不能单凭这些通道把原因确定为打印质量、装夹或其他单一因素。",
        "1016 个别试样早期也有引伸计回退，原始图保留了这些信号。16 根在所选拟合段通过筛选，不等于全程应变记录均可直接使用。", "",
        "## 数据使用建议", "",
        "本次桥接为部分吻合：90% 接近，60% 存在稳定偏移。因此当前证据不支持把两个制造/测试批次直接当作同一批，并忽略批次身份。也不适合用一个统一倍率校正：两个桥接点的偏移方向和幅度不同。",
        "建议以 1016 的 60%–90% 作为一套内部一致的主要标定数据；1013 的 40%、50%、100% 可作为扩展填充率范围的独立来源，使用时保留批次字段，并把跨批拼接视为暂定标定。1013 的 60%/90% 继续作为桥接外部检查，1013 的 70%/80% 保留为诊断数据。不要把旧批疑似异常 70%/80% 与新批平均。",
        "若研究允许约 10%–15% 的模量级误差，可以将上述数据作粗略联合探索，但应报告60%桥接差异，并比较保留/替换旧60%时模型结果的变化；精细反演或正式材料参数标定不应直接宣称两批等效。几何尺寸仅支持约不到1%的两批截面积差异，不能独自解释60%的约13%同窗偏移。",
        "当前输出没有生成匿名平均后的单批数据。calibration_review_with_batch.csv 保留了每根试样的批次和用途字段，供审查。", "",
        "## 验证与位置", "",
        f"1013 代码哈希与存档一致：{metadata['verification']['legacy_script_matches_archived_sha256']}。原33根结果复现，原始文件哈希一致。",
        f"共同窗口及逐根独立选段的 {metadata['verification']['independent_scipy_and_units_slope_checks_count']} 个斜率均用 SciPy 独立回归及 E=(dF/d伸长)×L/A 的单位换算核对。两批无内容完全相同的原始工作簿。",
        "1016 单批数据、原始曲线、拟合、逐根相对偏差及其汇总仅位于 1016/ 内。涉及两批的逐组对照放在根目录 analysis_results/<run_id>/<填充率>/，跨填充率桥接图放在根目录 comparison_results/<run_id>/。PNG/SVG 均由同一最终图状态保存。sample_2/3 未进入本次处理。", "",
        f"跨批图目录：{figures}", f"数值与审查报告目录：{output}", ""])
    study_root = output.parent.parent
    run_id = metadata["run_id"]
    for level in LEVELS:
        group_dir = study_root / "1016" / str(level) / "analysis_results" / run_id
        lines.append(f"- {level}%：[原始力–位移图]({(group_dir/'raw_force_displacement.png').as_posix()})、[拟合图]({(group_dir/'fit_curves.png').as_posix()})、[逐根结果]({(group_dir/'fit_results.csv').as_posix()})、[批次对照]({(output/str(level)/'batch_comparison.png').as_posix()})。")
    lines.extend(["", "运行命令：", "", "```powershell", metadata["command"], "```", "",
        "重新运行必须使用新的 --run-id，避免覆盖已有输出。", ""])
    (output / "batch_comparison_report.md").write_text("\n".join(lines), encoding="utf-8")


def write_new_batch_report(output: Path, group_summary, deviations, individual_rows) -> None:
    """Keep the 1016-only interpretation entirely inside its own batch folder."""
    lines = ["# 1016 均匀填充试样模量分析", "",
             "16 根试样；宽9.89 mm、厚3.26 mm、标距50 mm。sample_2/3不参与。",
             "主结果沿用原方法：每个填充率优先共用线性窗口，不要求与其他批次共用窗口。",
             "应变=引伸计伸长/50 mm，应力=扣除初始偏置的力/(9.89×3.26 mm²)。E为所选段含截距OLS斜率。", "",
             "## 主结果及均值代表每根试样的偏差", "",
             "绝对相对偏差=|组均值−该试样E|/该试样E×100%。分母是该试样E；并非与真实模量比较的测量误差。",
             "CSV还保留带符号偏差，以及以组均值为分母的另一种口径，避免混淆。", "",
             "| 填充率 | n | E均值±标准差(MPa) | 平均绝对相对偏差 | 最大绝对相对偏差 |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for row in group_summary.to_dict("records"):
        values = deviations[deviations.infill_percent == row["infill_percent"]].absolute_relative_percent
        lines.append(f"| {row['infill_percent']}% | {row['n']} | {row['mean_E_MPa']:.1f} ± {row['sd_E_MPa']:.1f} | {values.mean():.2f}% | {values.max():.2f}% |")
    lines.extend(["", "| 试样 | E (MPa) | 组均值 (MPa) | 绝对相对偏差 |",
                  "| --- | ---: | ---: | ---: |"])
    for row in deviations.to_dict("records"):
        lines.append(f"| {row['specimen_id']} | {row['E_MPa']:.1f} | {row['group_mean_E_MPa']:.1f} | {row['absolute_relative_percent']:.2f}% |")
    lines.extend(["", "## 每根试样独立最优窗口：方法敏感性对照", "",
                  "R²≥0.995、至少60点、前后半段斜率差≤5%、应变非递减步比例≥90%；每根独立搜索峰值应力5%–80%，通过后优先选最宽段，随后按R²及曲率排序。",
                  "最优只是这一明确准则下的最优，不代表已经验证为初始可恢复弹性段。", "",
                  "| 填充率 | E均值±标准差 (MPa) |", "| --- | ---: |"])
    for level in LEVELS:
        row = summary(r["E_MPa"] for r in individual_rows if r["infill_percent"] == level)
        lines.append(f"| {level}% | {row['mean_E_MPa']:.1f} ± {row['sd_E_MPa']:.1f} |")
    lines.extend(["", "## 图表入口", ""])
    batch_root = output.parent.parent
    for level in LEVELS:
        folder = batch_root / str(level) / "analysis_results" / output.name
        lines.append(f"- {level}%：[原始力–位移]({(folder/'raw_force_displacement.png').as_posix()})、[原方法拟合]({(folder/'fit_curves.png').as_posix()})、[逐根独立拟合]({(folder/'individual_optimal_fit_curves.png').as_posix()})、[逐根结果]({(folder/'fit_results.csv').as_posix()})。")
    lines.extend(["", "逐根独立方法属于补充对照，未覆盖原方法主结果。局部高R²不能排除传感器系统偏差，也不能证明初始弹性。新批统一尺寸未反映逐根几何差异。", ""])
    (output / "analysis_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--run-id", default="batch_study_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    args = parser.parse_args()
    root, run_id = args.root.resolve(), args.run_id
    if not run_id or any(c in run_id for c in '\\/:*?"<>|'):
        raise ValueError(f"Invalid run ID: {run_id!r}")
    core = load_core(root)
    settings = core.Settings()
    geometry_path = root / "1016" / "geometry.json"
    geometry_config = json.loads(geometry_path.read_text(encoding="utf-8"))
    dims = tuple(float(geometry_config[k]) for k in ("width_mm", "thickness_mm", "gauge_mm"))
    if not all(math.isfinite(v) and v > 0 for v in dims):
        raise ValueError(f"Invalid new-batch geometry: {dims}")
    old_geometry_path = root / "1013" / core.GEOMETRY_FILE.name
    old, old_blocks = core.load_specimens(root / "1013", core.read_geometry(old_geometry_path))
    new_geometry = {p.stem.replace("_", "-"): dims for level in LEVELS for p in (root / "1016" / str(level)).glob("*.xlsb")}
    new, new_blocks = core.load_specimens(root / "1016", new_geometry)
    if not new:
        raise ValueError("No new-batch dogbone data found")
    expected_counts = {60: 3, 70: 5, 80: 5, 90: 3}
    counts = {level: sum(s.infill == level for s in new) for level in LEVELS}
    if counts != expected_counts:
        raise ValueError(f"1016 input inventory changed: {counts}; review expected sample counts first")
    source_hashes = {f"{batch}/{s.source}": core.file_sha256(root / batch / s.source)
                     for batch, specimens in (("1013", old), ("1016", new)) for s in specimens}
    source_hashes["1013/" + old_geometry_path.name] = core.file_sha256(old_geometry_path)
    source_hashes["1016/geometry.json"] = core.file_sha256(geometry_path)
    old_hashes = {v for k, v in source_hashes.items() if k.startswith("1013/") and k.endswith(".xlsb")}
    if any(v in old_hashes for k, v in source_hashes.items() if k.startswith("1016/") and k.endswith(".xlsb")):
        raise ValueError("An identical raw workbook exists in both batches; do not double-count it")

    output = root / "analysis_results" / run_id
    figures = root / "comparison_results" / run_id
    new_output = root / "1016" / "analysis_results" / run_id
    new_figures = root / "1016" / "comparison_results" / run_id
    groups_out = {level: root / "1016" / str(level) / "analysis_results" / run_id for level in LEVELS}
    bridge_groups_out = {level: output / str(level) for level in LEVELS}
    destinations = [output, figures, new_output, new_figures, *groups_out.values(), *bridge_groups_out.values()]
    if any(p.exists() for p in destinations):
        raise FileExistsError("One or more output directories already exist; choose a new --run-id")

    independent_rows, new_rows, all_diagnostics, new_audits = [], [], [], []
    individual_rows, individual_audits = [], []
    independent_fits, individual_fits = {}, {}
    for batch, specimens in (("1013", old), ("1016", new)):
        for level in sorted({s.infill for s in specimens}):
            group = [s for s in specimens if s.infill == level]
            fits, audit = core.select_group_fits(group, settings)
            independent_fits[(batch, level)] = fits
            independent_rows.extend(fit_row(s, fits[s.specimen_id], batch, "independent") for s in group)
            for specimen in group:
                single_fits, single_audit = core.select_group_fits([specimen], settings)
                single = single_fits[specimen.specimen_id]
                individual_fits[(batch, specimen.specimen_id)] = single
                individual_rows.append(fit_row(specimen, single, batch, "individual_optimal"))
                individual_audits.append({"batch": batch, "specimen_id": specimen.specimen_id, **single_audit})
                diag = core.specimen_diagnostics(specimen, fits[specimen.specimen_id])
                diag.update(batch=batch, infill_percent=level)
                end = specimen.peak_index + 1
                lo, hi = np.percentile(specimen.force_n[:end], [20, 60])
                mask = (specimen.force_n[:end] >= lo) & (specimen.force_n[:end] <= hi)
                diag["loading_machine_speed_mm_per_min"] = core.linear_fit(specimen.time_s[:end][mask], specimen.displacement_mm[:end][mask])[0] * 60
                all_diagnostics.append(diag)
            if batch == "1016":
                new_rows.extend(fit_row(s, fits[s.specimen_id], batch, "independent") for s in group)
                audit["window_sensitivity_percent"] = core.window_sensitivity(group, fits, audit, settings)
                new_audits.append(audit)

    # Reproduce the archived results after relocation, including file identity.
    historical_path = root / "1013" / "analysis_results" / "modulus_20261013_165808" / "group_summary.csv"
    historical = pd.read_csv(historical_path)
    old_summary = pd.DataFrame([{"infill_percent": level, **summary(r["E_MPa"] for r in independent_rows if r["batch"] == "1013" and r["infill_percent"] == level)} for level in core.INFILLS])
    np.testing.assert_allclose(old_summary[["mean_E_MPa", "sd_E_MPa"]], historical[["mean_E_MPa", "sd_E_MPa"]], rtol=1e-11, atol=1e-9)
    historical_meta_path = historical_path.with_name("analysis_metadata.json")
    historical_meta = json.loads(historical_meta_path.read_text(encoding="utf-8"))
    for name, digest in historical_meta["source_sha256"].items():
        if source_hashes["1013/" + name] != digest:
            raise AssertionError(f"Historical source changed: {name}")

    matched_rows, comparisons, window_rows, matching_audits, matching_fits, combined_groups = [], [], [], [], {}, {}
    for level in LEVELS:
        combined = [replace(s, specimen_id=batch + "/" + s.specimen_id)
                    for batch, specimens in (("1013", old), ("1016", new)) for s in specimens if s.infill == level]
        fits, audit, candidates = common_window_study(core, combined, settings)
        combined_groups[level], matching_fits[level] = combined, fits
        matching_audits.append(audit)
        window_rows.extend(candidates)
        for s in combined:
            matched_rows.append(fit_row(s, fits[s.specimen_id], s.specimen_id.split("/")[0], "matched"))
        values = {batch: [fits[s.specimen_id].modulus_mpa for s in combined if s.specimen_id.startswith(batch + "/") and fits[s.specimen_id] is not None] for batch in ("1013", "1016")}
        a, b = summary(values["1013"]), summary(values["1016"])
        ratio, ci_lo, ci_hi = log_ratio_interval(values["1013"], values["1016"])
        lo, hi = audit["common_stress_interval_mpa"]
        within_var = ((a["n"] - 1) * a["sd_E_MPa"]**2 + (b["n"] - 1) * b["sd_E_MPa"]**2) / (a["n"] + b["n"] - 2)
        comparisons.append({
            "infill_percent": level, "role": "bridge" if level in BRIDGES else "old_batch_diagnostic_only",
            "stress_min_MPa": lo, "stress_max_MPa": hi,
            **{"old_" + k: v for k, v in a.items()}, **{"new_" + k: v for k, v in b.items()},
            "arithmetic_mean_difference_percent": 100 * (b["mean_E_MPa"] / a["mean_E_MPa"] - 1),
            "geometric_ratio": ratio, "geometric_ratio_ci95_low": ci_lo, "geometric_ratio_ci95_high": ci_hi,
            "difference_in_within_batch_SD": (b["mean_E_MPa"] - a["mean_E_MPa"]) / math.sqrt(within_var),
            "pooled_cv_percent_for_diagnostic_only": summary(values["1013"] + values["1016"])["cv_percent"],
            "all_candidate_difference_percent_min": audit["all_accepted_window_difference_percent_min"],
            "all_candidate_difference_percent_max": audit["all_accepted_window_difference_percent_max"],
            "all_candidate_count": len(candidates),
            **{f"CI95_within_{tol}pct_reference": bool(ci_lo >= 1 - tol / 100 and ci_hi <= 1 + tol / 100) for tol in (5, 10, 15)},
        })

    # Verify numerical slopes by a separate regression implementation and an
    # explicit force-extension to modulus conversion on every accepted fit.
    slope_checks = []
    check_sets = [(s, matching_fits[level][s.specimen_id]) for level, group in combined_groups.items() for s in group]
    check_sets.extend((s, individual_fits[(batch, s.specimen_id)])
                      for batch, group in (("1013", old), ("1016", new)) for s in group)
    for s, f in check_sets:
            if f is None:
                continue
            mask = (s.stress_mpa[:s.peak_index + 1] >= f.lower_stress_mpa) & (s.stress_mpa[:s.peak_index + 1] <= f.upper_stress_mpa)
            direct = stats.linregress(s.raw_strain[:s.peak_index + 1][mask], s.stress_mpa[:s.peak_index + 1][mask])
            force_slope = stats.linregress(s.extension_mm[:s.peak_index + 1][mask], s.force_n[:s.peak_index + 1][mask]).slope
            converted = force_slope * s.gauge_mm / s.area_mm2
            np.testing.assert_allclose([direct.slope, converted], f.modulus_mpa, rtol=1e-10, atol=1e-8)
            slope_checks.append(s.specimen_id)

    for directory in destinations:
        directory.mkdir(parents=True)
    user_columns = ["infill_percent", "specimen_id", "E_MPa", "R2", "strain_min_pct", "strain_max_pct", "force_min_N", "force_max_N"]
    for audit in new_audits:
        level = audit["infill_percent"]
        group = [s for s in new if s.infill == level]
        pd.DataFrame([r for r in new_rows if r["infill_percent"] == level])[user_columns].to_csv(groups_out[level] / "fit_results.csv", index=False, encoding="utf-8-sig")
        core.plot_group(group, independent_fits[("1016", level)], audit, groups_out[level])
        core.plot_raw_force_displacement(group, root / "1016", groups_out[level])
        group_matched = pd.DataFrame([r for r in matched_rows if r["infill_percent"] == level])
        group_matched.to_csv(bridge_groups_out[level] / "matched_batch_fit_results.csv", index=False, encoding="utf-8-sig")
        plot_batch_group(core, combined_groups[level], matching_fits[level], next(a for a in matching_audits if a["infill_percent"] == level), bridge_groups_out[level])
        singles = [r for r in individual_rows if r["batch"] == "1016" and r["infill_percent"] == level]
        pd.DataFrame(singles)[user_columns].to_csv(groups_out[level] / "individual_optimal_fit_results.csv", index=False, encoding="utf-8-sig")
        plot_individual_optimal(group, {s.specimen_id: individual_fits[("1016", s.specimen_id)] for s in group}, groups_out[level])
    new_summary = pd.DataFrame([{"infill_percent": level, **summary(r["E_MPa"] for r in new_rows if r["infill_percent"] == level)} for level in LEVELS])
    new_summary.to_csv(new_output / "group_summary.csv", index=False, encoding="utf-8-sig")
    deviations = mean_relative_deviation(new_rows)
    deviations.to_csv(new_output / "mean_relative_deviation.csv", index=False, encoding="utf-8-sig")
    for level in LEVELS:
        deviations[deviations.infill_percent == level].to_csv(groups_out[level] / "mean_relative_deviation.csv", index=False, encoding="utf-8-sig")
    new_individual_rows = [r for r in individual_rows if r["batch"] == "1016"]
    mean_relative_deviation(new_individual_rows).to_csv(new_output / "individual_optimal_mean_relative_deviation.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([{"infill_percent": level, **summary(r["E_MPa"] for r in new_individual_rows if r["infill_percent"] == level)} for level in LEVELS]).to_csv(new_output / "individual_optimal_group_summary.csv", index=False, encoding="utf-8-sig")
    plot_new_summary(new_rows, new_figures)
    independent_comparisons = compare_batches(independent_rows)
    individual_comparisons = compare_batches(individual_rows)
    pd.DataFrame(individual_comparisons).to_csv(output / "individual_optimal_batch_comparison.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(individual_rows).to_csv(output / "individual_optimal_fit_results_with_batch.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(independent_comparisons).to_csv(output / "independent_batch_comparison.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(independent_rows + matched_rows).to_csv(output / "all_fit_results_with_batch.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(comparisons).to_csv(output / "matched_batch_comparison.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(window_rows).to_csv(output / "matched_window_sensitivity.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(all_diagnostics).to_csv(output / "specimen_diagnostics.csv", index=False, encoding="utf-8-sig")
    # Candidate review data retain source batch and role. Not a pooled calibration.
    review = [{**r, "role": ("suspect_old_70_80_diagnostic" if r["batch"] == "1013" and r["infill_percent"] in (70, 80)
                             else "bridge" if r["infill_percent"] in BRIDGES else "primary_calibration")}
              for r in independent_rows]
    pd.DataFrame(review).to_csv(output / "calibration_review_with_batch.csv", index=False, encoding="utf-8-sig")
    plot_modulus_comparison(independent_rows + matched_rows, figures)
    plot_bridge_difference(comparisons, figures)
    plot_primary_calibration([r for r in independent_rows if r["batch"] == "1013"], new_rows, figures)

    metadata = {
        "run_id": run_id, "settings": asdict(settings), "new_geometry": geometry_config,
        "design": {"bridges": list(BRIDGES), "new_primary_infill_levels": [70, 80],
                   "suspect_old_levels": [70, 80], "excluded_nonuniform_folders": ["sample_2", "sample_3"],
                   "pooling": "No anonymous pooled dataset created. Retain batch identity. Bridge agreement alone cannot prove identical manufacturing populations."},
        "independent_fit_audits_1016": new_audits, "matched_fit_audits": matching_audits,
        "individual_optimal_fit_audits": individual_audits,
        "individual_optimal_definition": "Run unchanged select_group_fits on each singleton specimen. Among accepted candidates maximize width, then R2 and curvature. No between-batch agreement criterion.",
        "relative_deviation_definition": "100 * (group_mean_E - specimen_E) / specimen_E; absolute value reported as error magnitude; group means computed within batch and fit method.",
        "source_blocks": {"1013": old_blocks, "1016": new_blocks}, "source_sha256": source_hashes,
        "script_sha256": {"driver": core.file_sha256(Path(__file__)), "legacy_core": core.file_sha256(root / "1013" / "analyze_youngs_modulus.py")},
        "verification": {"archived_means_and_SD_reproduced": True, "archived_source_hashes_unchanged": True,
                         "legacy_script_matches_archived_sha256": core.file_sha256(root / "1013" / "analyze_youngs_modulus.py") == historical_meta["script_sha256"],
                         "independent_scipy_and_units_slope_checks_count": len(slope_checks), "no_cross_batch_identical_raw_files": True},
        "uncertainty": "95% Welch t interval on log(E) for ratio of geometric means; sampling uncertainty only; no application-specific pooling tolerance supplied. 5/10/15% are sensitivity reference bands, not established equivalence margins.",
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "scipy": __import__("scipy").__version__, "matplotlib": matplotlib.__version__},
        "command": f'& "{sys.executable}" -X utf8 -B "{Path(__file__).resolve()}" --root "{root}" --run-id "{run_id}"',
    }
    write_json(output / "analysis_metadata.json", metadata)
    write_json(new_output / "analysis_metadata.json", {"geometry": geometry_config, "groups": new_audits,
        "individual_optimal_audits": [a for a in individual_audits if a["batch"] == "1016"],
        "settings": asdict(settings), "versions": metadata["versions"], "script_sha256": metadata["script_sha256"],
        "relative_deviation_definition": metadata["relative_deviation_definition"],
        "source_sha256": {k:v for k,v in source_hashes.items() if k.startswith("1016/")}, "full_study_metadata": str(output / "analysis_metadata.json")})
    write_report(output, figures, new_summary, new_audits, comparisons, independent_comparisons, all_diagnostics, metadata, individual_comparisons)
    write_new_batch_report(new_output, new_summary, deviations, new_individual_rows)
    for name, digest in source_hashes.items():
        if core.file_sha256(root / name) != digest:
            raise AssertionError(f"Raw source changed during analysis: {name}")
    print("1013 archived results reproduced; original inputs unchanged.")
    print("1016 independent-window results:")
    print(new_summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("Matched-window batch comparison (not pooled):")
    cols = ["infill_percent", "role", "old_mean_E_MPa", "new_mean_E_MPa", "arithmetic_mean_difference_percent", "all_candidate_difference_percent_min", "all_candidate_difference_percent_max"]
    print(pd.DataFrame(comparisons)[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("Each-specimen optimal-window comparison:")
    print(pd.DataFrame(individual_comparisons).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("1016 mean vs specimen relative deviations:")
    print(deviations[["specimen_id", "E_MPa", "group_mean_E_MPa", "absolute_relative_percent"]].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("Tables/metadata:", output)
    print("Comparison figures:", figures)


if __name__ == "__main__":
    main()
