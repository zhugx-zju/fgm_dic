"""Explore E(infill) for old 40/50/60/90/100% plus new 70/80%.

Read existing verified specimen moduli; never refit raw curves or overwrite
batch analyses. Equal weight per infill level prevents sample counts or small
estimated within-group variances from setting the scientific target.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parent
LEVELS = (40, 50, 60, 70, 80, 90, 100)
BATCH_AT_LEVEL = {level: "1016" if level in (70, 80) else "1013" for level in LEVELS}
EXPECTED_COUNTS = {level: 4 if level == 90 else 5 for level in LEVELS}
COLORS = {"1013": "#0072B2", "1016": "#D55E00"}
METHOD_NAMES = {"independent": "Group-priority window (primary)",
                "individual_optimal": "Individual optimal windows (sensitivity)"}


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def design(x: np.ndarray, degree: int) -> np.ndarray:
    return np.vander(np.asarray(x, dtype=float), degree + 1, increasing=True)


def fit_model(x: np.ndarray, y: np.ndarray, degree: int) -> np.ndarray:
    coefficients, _, rank, _ = np.linalg.lstsq(design(x, degree), y, rcond=None)
    if rank != degree + 1:
        raise ValueError("Infill design is rank deficient")
    return coefficients


def select_mixed(source: Path, method: str) -> pd.DataFrame:
    frame = pd.read_csv(source, dtype={"batch": str, "specimen_id": str})
    frame = frame[frame.fit_method == method].copy()
    frame = frame[frame.batch == frame.infill_percent.map(BATCH_AT_LEVEL)].copy()
    frame = frame.sort_values(["infill_percent", "specimen_id"]).reset_index(drop=True)
    if frame.duplicated(["batch", "specimen_id"]).any():
        raise ValueError("Duplicate specimen in mixed dataset")
    counts = frame.groupby("infill_percent").size().to_dict()
    if counts != EXPECTED_COUNTS:
        raise ValueError(f"Unexpected mixed specimen inventory: {counts}")
    if not np.isfinite(frame.E_MPa).all() or not frame.E_MPa.gt(0).all():
        raise ValueError("Missing/nonpositive modulus in mixed dataset")
    if not frame.R2.ge(0.995).all():
        raise ValueError("A supplied curve fit does not pass the original R2 threshold")
    return frame


def analyse(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[dict], list[dict], dict]:
    method = frame.fit_method.iloc[0]
    group = frame.groupby(["infill_percent", "batch"], as_index=False).agg(
        n=("E_MPa", "size"), mean_E_MPa=("E_MPa", "mean"), sd_E_MPa=("E_MPa", "std"))
    group["fit_method"] = method
    group["infill_fraction"] = group.infill_percent / 100
    x, y = group.infill_fraction.to_numpy(), group.mean_E_MPa.to_numpy()
    metrics, loo_records, models = [], [], {}
    for degree, name in ((1, "linear"), (2, "quadratic")):
        coefficients = fit_model(x, y, degree)
        predicted = design(x, degree) @ coefficients
        residual = y - predicted
        loo = np.empty_like(y)
        for index, level in enumerate(LEVELS):
            keep = np.arange(len(x)) != index
            coef_without_level = fit_model(x[keep], y[keep], degree)
            loo[index] = (design(x[index:index + 1], degree) @ coef_without_level)[0]
            loo_records.append({"fit_method": method, "model": name,
                                "held_out_infill_percent": level, "observed_mean_E_MPa": y[index],
                                "predicted_mean_E_MPa": loo[index], "residual_MPa": y[index] - loo[index],
                                "absolute_relative_error_percent": 100 * abs(y[index] - loo[index]) / y[index]})
        individual_prediction = design(frame.infill_percent.to_numpy() / 100, degree) @ coefficients
        individual_residual = frame.E_MPa.to_numpy() - individual_prediction
        metrics.append({
            "fit_method": method, "model": name, "degree": degree,
            "n_infill_levels": len(group), "n_specimens": len(frame),
            "intercept_MPa": coefficients[0], "slope_per_fraction_MPa": coefficients[1],
            "quadratic_per_fraction_squared_MPa": coefficients[2] if degree == 2 else 0.,
            "R2_group_means": 1 - np.sum(residual**2) / np.sum((y - y.mean())**2),
            "RMSE_group_means_MPa": np.sqrt(np.mean(residual**2)),
            "MAPE_group_means_percent": 100 * np.mean(abs(residual) / y),
            "max_group_mean_relative_error_percent": 100 * max(abs(residual) / y),
            "LOO_infill_RMSE_MPa": np.sqrt(np.mean((y - loo)**2)),
            "LOO_infill_MAPE_percent": 100 * np.mean(abs(y - loo) / y),
            "LOO_infill_max_relative_error_percent": 100 * max(abs(y - loo) / y),
            "R2_specimens_for_group_mean_model": 1 - np.sum(individual_residual**2) / np.sum((frame.E_MPa - frame.E_MPa.mean())**2),
            "RMSE_specimens_MPa": np.sqrt(np.mean(individual_residual**2)),
        })
        group[f"{name}_prediction_MPa"] = predicted
        group[f"{name}_residual_MPa"] = residual
        group[f"{name}_residual_over_mean_percent"] = 100 * residual / y
        group[f"{name}_residual_over_SD"] = residual / group.sd_E_MPa
        group[f"{name}_LOO_prediction_MPa"] = loo
        models[name] = coefficients
    # Independent regression implementation on the same seven group means.
    check = stats.linregress(x, y)
    np.testing.assert_allclose(models["linear"], [check.intercept, check.slope], rtol=1e-11)
    # A separate baseline weights each specimen equally rather than each infill.
    individual_baseline = fit_model(frame.infill_percent.to_numpy() / 100, frame.E_MPa.to_numpy(), 1)
    models["all_specimen_linear_baseline"] = individual_baseline
    return group, metrics, loo_records, models


def save_figure(fig, path: Path) -> None:
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def draw_data(ax, frame, group, models, title: str, quadratic: bool = False) -> None:
    for batch, marker in (("1013", "o"), ("1016", "s")):
        subset = frame[frame.batch == batch]
        for level, specimens in subset.groupby("infill_percent"):
            # Horizontal jitter is display-only; fits always use exact infill.
            jitter = np.linspace(-0.85, 0.85, len(specimens))
            ax.scatter(level + jitter, specimens.E_MPa, color=COLORS[batch], alpha=0.35, s=20,
                       marker=marker, linewidths=0)
        g = group[group.batch == batch]
        ax.errorbar(g.infill_percent, g.mean_E_MPa, yerr=g.sd_E_MPa, fmt=marker,
                    color=COLORS[batch], markersize=6, capsize=4, lw=1.2,
                    label=f"{batch}: mean ± SD")
    grid = np.linspace(0.4, 1., 250)
    ax.plot(grid * 100, design(grid, 1) @ models["linear"], color="#222222", lw=1.5,
            label="Linear: equal weight per infill")
    if quadratic:
        ax.plot(grid * 100, design(grid, 2) @ models["quadratic"], color="#666666", lw=1.2,
                ls="--", label="Quadratic: exploratory check")
    ax.set(xlim=(37, 103), xticks=LEVELS, ylabel="Apparent fitted modulus (MPa)", title=title)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=8, loc="upper left", frameon=False)


def plot_primary(frame, group, models, metrics, output: Path) -> None:
    fig, (ax, residual_ax) = plt.subplots(2, 1, figsize=(8, 7.8), sharex=True,
                                        height_ratios=[2.4, 1], constrained_layout=True)
    draw_data(ax, frame, group, models, "Mixed batches: 1013 except 70/80%; 1016 at 70/80%")
    linear = next(row for row in metrics if row["model"] == "linear")
    equation = (f"E = {linear['intercept_MPa']:.2f} + {linear['slope_per_fraction_MPa']:.2f} ρ MPa\n"
                f"ρ = infill / 100; means R² = {linear['R2_group_means']:.4f}")
    ax.text(0.98, 0.04, equation, transform=ax.transAxes, ha="right", va="bottom", fontsize=9)
    residual_ax.axhline(0, color="#444444", lw=1.)
    for batch, marker in (("1013", "o"), ("1016", "s")):
        g = group[group.batch == batch]
        residual_ax.errorbar(g.infill_percent, g.linear_residual_MPa, yerr=g.sd_E_MPa,
                             fmt=marker, color=COLORS[batch], markersize=5, capsize=4, lw=1.1)
    residual_ax.set(xlabel="Nominal infill (%)", ylabel="Mean − line (MPa)",
                    title="Residuals; error bars show within-group SD, not confidence intervals")
    residual_ax.set_title(residual_ax.get_title(), fontsize=9)
    residual_ax.spines[["top", "right"]].set_visible(False)
    save_figure(fig, output / "modulus_vs_infill")


def plot_sensitivity(results: dict, figures: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True, sharey=True, constrained_layout=True)
    for axis, method in zip(axes, METHOD_NAMES):
        frame, group, metrics, _, models = results[method]
        draw_data(axis, frame, group, models, METHOD_NAMES[method], quadratic=True)
        axis.set_xlabel("Nominal infill (%)")
        line = next(m for m in metrics if m["model"] == "linear")
        axis.text(.98, .03, f"Linear means R² = {line['R2_group_means']:.4f}\n"
                  f"Max mean error = {line['max_group_mean_relative_error_percent']:.2f}%",
                  transform=axis.transAxes, ha="right", va="bottom", fontsize=8)
    save_figure(fig, figures / "selection_method_sensitivity")


def write_report(output: Path, figures: Path, results: dict, metadata: dict) -> None:
    _, group, metrics, _, models = results["independent"]
    linear, quadratic = metrics
    a, b = models["linear"]
    lines = ["# 混合批次的模量—填充率关系", "",
        "## 结论", "",
        f"在本次40%–100%名义填充率范围，线性函数可作为简洁的近似描述，但不能据此认定严格线性。7个组均值的R²={linear['R2_group_means']:.5f}，最大均值相对偏差={linear['max_group_mean_relative_error_percent']:.2f}%。",
        "70%和80%的均值位于直线下方，最高填充率回到直线上方，存在系统残差。由于这两个中间点恰好来自另一批次，曲率、批次偏移与模量选段效应不能由这个拼接数据集单独区分。", "",
        "## 数据与计算口径", "",
        "仅选1013的40%、50%、60%、90%、100%，及1016的70%、80%；共34根、7个填充率。没有加入1016的60%/90%，没有使用1013的70%/80%，没有进行批次倍率校正。",
        "主分析沿用原同填充率优先共同窗口结果。敏感性分析对所有填充率统一使用逐根独立选段结果，两个口径不交叉混用。本分析只读取已验证的试样模量，不改变原始曲线或重新挑选拟合段。",
        "以每组均值等权拟合E=a+bρ，ρ为0.4–1.0的填充率比例。不是把均值按试样数或估计方差加权；目标是七个填充率同等重要。另保存全部试样等权OLS系数用于核对。",
        "图中淡点为单根试样，实心符号和误差棒为均值±组内标准差（不是置信区间），颜色与符号保留批次身份。", "",
        "| 填充率 | 来源批次 | n | E均值±标准差 (MPa) | 线性预测 (MPa) | 均值−预测/均值 |",
        "| --- | --- | ---: | ---: | ---: | ---: |"]
    for row in group.to_dict("records"):
        lines.append(f"| {row['infill_percent']}% | {row['batch']} | {row['n']} | {row['mean_E_MPa']:.1f} ± {row['sd_E_MPa']:.1f} | {row['linear_prediction_MPa']:.1f} | {row['linear_residual_over_mean_percent']:+.2f}% |")
    lines.extend(["", "## 线性经验式", "", f"E(MPa) = {a:.4f} + {b:.4f} ρ，ρ∈[0.4,1.0]。",
        f"若填充率p按40、50、…、100输入：E(MPa) = {a:.4f} + {b/100:.4f} p。每提高10个百分点，对应平均增加约{b*.1:.1f} MPa。",
        f"组均值训练RMSE={linear['RMSE_group_means_MPa']:.2f} MPa，组均值平均绝对相对偏差={linear['MAPE_group_means_percent']:.2f}%。34根个体相对于该均值模型的R²={linear['R2_specimens_for_group_mean_model']:.5f}，不能把组均值R²误读为每根试样精度。",
        "80%残差为负，直线高估其组均值。不同填充率局部偏差应保留，不应只报告整体R²。", "",
        "## 是否需要二次项", "",
        "只比较事先选定的一次和二次模型，不搜索更高阶。交叉检验每次整组留出一个填充率，用其余6个均值拟合，再预测被留出的组均值。每次剔除该组全部试样，避免同一填充率同时出现在拟合和检验中。",
        "这是同一小数据集内的探索性交叉检验，不是独立外部测试；40%/100%留出折属于边界外推，已包括在误差中。", "",
        "| 模量选段口径 | E–填充率模型 | 训练R²（7个均值） | 训练RMSE (MPa) | 留一填充率RMSE (MPa) | 留一平均相对误差 |",
        "| --- | --- | ---: | ---: | ---: | ---: |"])
    for method, (_, _, rows, _, _) in results.items():
        for row in rows:
            lines.append(f"| {'同组优先共同窗口' if method == 'independent' else '逐根独立窗口'} | {'线性' if row['model'] == 'linear' else '二次'} | {row['R2_group_means']:.5f} | {row['RMSE_group_means_MPa']:.2f} | {row['LOO_infill_RMSE_MPa']:.2f} | {row['LOO_infill_MAPE_percent']:.2f}% |")
    lines.extend(["", f"二次项提高训练R²至{quadratic['R2_group_means']:.5f}，但主分析留一误差从{linear['LOO_infill_RMSE_MPa']:.2f}增加至{quadratic['LOO_infill_RMSE_MPa']:.2f} MPa；在本次比较中，尚未显示二次模型的预测优势。不能把这一结果反过来当作严格线性的证明。", "",
        "## 选段敏感性", ""])
    alt = results["individual_optimal"][2][0]
    lines.extend([f"将全部34根统一换为逐根独立选段后，直线为E={alt['intercept_MPa']:.4f}+{alt['slope_per_fraction_MPa']:.4f}ρ MPa；R²={alt['R2_group_means']:.5f}，最大组均值偏差={alt['max_group_mean_relative_error_percent']:.2f}%，留一RMSE={alt['LOO_infill_RMSE_MPa']:.2f} MPa。",
        "整体近似随填充率增加的趋势对选段口径保持，但局部误差受选段影响，尤其是80%。主结果没有因为另一口径更接近直线而被替换。", "",
        "## 使用边界", "",
        "该关系针对名义填充率、外轮廓截面积定义的表观试样模量，不是PLA实体材料本征模量，也不是相对密度的直接定律。打印壁层、内部构型和实测相对密度没有由本次回归识别。",
        "只建议作为40%–100%范围内的初步经验标定。正截距不应解释为0%填充率的实际模量，也不应外推到此范围以外。原始曲线局部高R²不能单独保证选中的都是初始可恢复弹性段。",
        "混合数据集没有在同一填充率下保留两批重复，因而本图不能证明两批可互换。此前桥接差异仍是使用该经验式时的系统不确定性，未被拟合误差或标准差完整覆盖。", "",
        "## 文件与复现", "",
        f"[主图]({(figures/'modulus_vs_infill.png').as_posix()})、[主图SVG]({(figures/'modulus_vs_infill.svg').as_posix()})、[选段与曲率对照图]({(figures/'selection_method_sensitivity.png').as_posix()})。",
        "数值：selected_specimens.csv、group_summary_and_residuals.csv、model_comparison.csv、leave_one_infill_out.csv。元数据保存来源哈希、数据选择、系数、环境与脚本哈希。", "",
        "```powershell", metadata["command"], "```", "", "复现时需使用新的--run-id，避免覆盖当前结果。", ""])
    (output / "linearity_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--source-run", default="analysis_20261017")
    parser.add_argument("--run-id", default="mixed_infill_20261017")
    args = parser.parse_args()
    for value in (args.run_id, args.source_run):
        if not value or value in (".", "..") or any(c in value for c in '\\/:*?"<>|'):
            raise ValueError(f"Invalid run name: {value!r}")
    root = args.root.resolve()
    source = root / "analysis_results" / args.source_run
    output, figures = root / "analysis_results" / args.run_id, root / "comparison_results" / args.run_id
    if output.exists() or figures.exists():
        raise FileExistsError("Choose a new --run-id; prior results are never overwritten")
    inputs = {"independent": source / "all_fit_results_with_batch.csv",
              "individual_optimal": source / "individual_optimal_fit_results_with_batch.csv"}
    upstream_metadata_path = source / "analysis_metadata.json"
    upstream = json.loads(upstream_metadata_path.read_text(encoding="utf-8"))
    input_hashes = {str(p.relative_to(root)): sha256(p) for p in [*inputs.values(), upstream_metadata_path]}
    for name, digest in upstream["source_sha256"].items():
        if sha256(root / name) != digest:
            raise AssertionError(f"Upstream raw source changed: {name}")
    results = {}
    for method, path in inputs.items():
        frame = select_mixed(path, method)
        group, metrics, loo_records, models = analyse(frame)
        results[method] = frame, group, metrics, loo_records, models
    # Same physical specimen inventory in the primary and sensitivity analyses.
    pd.testing.assert_frame_equal(results["independent"][0][["batch", "infill_percent", "specimen_id"]],
                                  results["individual_optimal"][0][["batch", "infill_percent", "specimen_id"]])
    for directory in (output, figures):
        directory.mkdir(parents=True)
    pd.concat([v[0] for v in results.values()]).to_csv(output / "selected_specimens.csv", index=False, encoding="utf-8-sig")
    pd.concat([v[1] for v in results.values()]).to_csv(output / "group_summary_and_residuals.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([row for v in results.values() for row in v[2]]).to_csv(output / "model_comparison.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([row for v in results.values() for row in v[3]]).to_csv(output / "leave_one_infill_out.csv", index=False, encoding="utf-8-sig")
    metadata = {
        "source_run": args.source_run, "run_id": args.run_id,
        "selection": {str(k): v for k, v in BATCH_AT_LEVEL.items()}, "counts": EXPECTED_COUNTS,
        "primary_method": "independent", "sensitivity_method": "individual_optimal",
        "regression_target": "Seven group means, each infill equally weighted; E(MPa)=a+b*infill_fraction",
        "validation": "Exploratory leave-one-entire-infill-out, including boundary extrapolation folds; no external validation dataset",
        "error_bars": "Within-group sample SD, not confidence intervals or complete systematic uncertainty",
        "input_sha256": input_hashes, "raw_source_sha256": upstream["source_sha256"],
        "script_sha256": sha256(Path(__file__)),
        "models": {method: {name: coef.tolist() for name, coef in value[4].items()} for method, value in results.items()},
        "verification": {"same_34_specimen_inventory_both_methods": True,
                         "linear_coefficients_checked_with_scipy_linregress": True,
                         "upstream_raw_source_hash_count": len(upstream["source_sha256"])},
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                     "scipy": __import__("scipy").__version__, "matplotlib": matplotlib.__version__},
        "command": f'& "{sys.executable}" -X utf8 -B "{Path(__file__).resolve()}" --root "{root}" --source-run "{args.source_run}" --run-id "{args.run_id}"',
    }
    (output / "analysis_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    frame, group, metrics, _, models = results["independent"]
    plot_primary(frame, group, models, metrics, figures)
    plot_sensitivity(results, figures)
    write_report(output, figures, results, metadata)
    for name, digest in {**input_hashes, **upstream["source_sha256"]}.items():
        if sha256(root / name) != digest:
            raise AssertionError(f"Input changed during processing: {name}")
    print(group.to_string(index=False))
    print(pd.DataFrame([row for v in results.values() for row in v[2]]).to_string(index=False))
    print("Report:", output / "linearity_report.md")
    print("Figures:", figures)


if __name__ == "__main__":
    main()
