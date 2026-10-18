"""Inspect DIC timing without modifying the supplied CSV/XLSB files.

CSV integer 'time' is treated as exported stage index, not seconds. The user
specified run 3 at 5 Hz and run 4 at 7 Hz; 5 Hz is a competing run-4 hypothesis.
The two points share a cross-section, so their mean locates motion events;
their difference is a consistency diagnostic, not specimen extension.
"""
from pathlib import Path
import csv
import hashlib
import json
import sys

sys.path.append(r"C:\Users\zhu_g\.conda\envs\abqinter\Lib\site-packages")
import numpy as np
from scipy.optimize import least_squares
from pyxlsb import open_workbook
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent
SOURCE = OUT.parents[1]


def read_dic(run):
    path = SOURCE / f"DY_{run}.csv"
    blocks, labels = [], []
    for row in csv.reader(path.read_text(encoding="gb18030").splitlines()):
        if row and "-DY" in row[0]:
            labels.append(row[0])
            blocks.append([])
        elif len(row) == 3 and row[0] == "":
            blocks[-1].append([float(row[1]), float(row[2])])
    first, second = map(np.asarray, blocks)
    assert np.array_equal(first[:, 0], second[:, 0])
    assert np.array_equal(first[:, 0], np.arange(len(first)))
    points = np.column_stack([first[:, 1], second[:, 1]])
    return first[:, 0], points, labels


def read_machine(run):
    with open_workbook(str(SOURCE / f"sample_2_{run}.xlsb")) as book:
        with book.get_sheet(1) as sheet:
            rows = [[cell.v for cell in row] for row in sheet.rows()]
    return np.asarray(rows[1:], dtype=float)[:, :2]


def fit_plateau(x, y, knots, bounds, robust=False):
    """Continuous ramp/hold/ramp spline; plateau slope remains free."""
    midpoint = np.mean(x)
    def model(p):
        b, s1, ds1, ds2, k1, k2 = p
        return b + s1 * (x - midpoint) + ds1 * np.maximum(x-k1, 0) + ds2 * np.maximum(x-k2, 0)
    # Solve linear coefficients at initial knots before nonlinear refinement.
    k1, k2 = knots
    design = np.column_stack([np.ones(len(x)), x-midpoint,
                              np.maximum(x-k1, 0), np.maximum(x-k2, 0)])
    coeff = np.linalg.lstsq(design, y, rcond=None)[0]
    p0 = np.r_[coeff, knots]
    result = least_squares(lambda p: model(p)-y, p0,
        bounds=(np.r_[[-np.inf]*4, [b[0] for b in bounds]],
                np.r_[[np.inf]*4, [b[1] for b in bounds]]),
        loss="soft_l1" if robust else "linear", f_scale=0.002,
        max_nfev=2000, x_scale="jac")
    p = result.x
    return {"knots": p[-2:].tolist(), "slopes": [float(p[1]), float(p[1]+p[2]), float(sum(p[1:4]))],
            "rmse": float(np.sqrt(np.mean((model(p)-y)**2)))}, model(p)


records = {}
for run, hz in ((3, 5), (4, 7)):
    k, points, labels = read_dic(run)
    mean = points.mean(axis=1)
    machine = read_machine(run)
    t, force = machine.T
    baseline = float(np.median(force[t <= 0.08]))
    record = {"n": len(k), "nominal_hz": hz, "labels": labels,
              "baseline_force_N": baseline, "machine_plateaus": {}, "dic_plateaus": {}}
    for level, lo, hi, guess in ((80, 0, 23, (5, 15)), (180, 20, 62, (40, 50)), (280, 55, 98, (76, 86))):
        if run == 4:
            guess = tuple(g-2.5 for g in guess)
        keep = (t >= lo) & (t <= hi)
        fit, _ = fit_plateau(t[keep], force[keep], guess,
                             [(guess[0]-4, guess[0]+4), (guess[1]-4, guess[1]+4)])
        record["machine_plateaus"][str(level)] = fit
    configs = {3: [(80, 35, 125, (55, 105)), (180, 165, 315, (230, 280)),
                    (280, 340, 495, (410, 458))],
               4: [(180, 180, 408, (322, 388))]}
    for level, lo, hi, guess in configs[run]:
        keep = (k >= lo) & (k <= hi)
        fit, _ = fit_plateau(k[keep], mean[keep], guess,
                             [(guess[0]-13, guess[0]+13), (guess[1]-13, guess[1]+13)], True)
        record["dic_plateaus"][str(level)] = fit
    offsets = []
    for level, plateau in record["dic_plateaus"].items():
        offsets.extend(np.array(plateau["knots"])/hz - np.array(record["machine_plateaus"][level]["knots"]))
    record["offset_candidates_s"] = [float(v) for v in offsets]
    record["offset_median_s"] = float(np.median(offsets))
    if run == 4:
        record["5Hz_offset_candidates_s"] = (np.array(record["dic_plateaus"]["180"]["knots"])/5 -
                                               np.array(record["machine_plateaus"]["180"]["knots"])).tolist()
        keep = (k >= 130) & (k <= 380)
        independent, _ = fit_plateau(k[keep], mean[keep], (148, 320), [(140, 160), (307, 332)], True)
        exit_stage = independent["knots"][0]
        record["independent_first_plateau_exit_check"] = {
            "stage": exit_stage,
            "offset_at_7Hz_s": exit_stage/7-record["machine_plateaus"]["80"]["knots"][1],
            "offset_at_5Hz_s": exit_stage/5-record["machine_plateaus"]["80"]["knots"][1],
            "effective_hz_from_intact_plateau": float(np.diff(record["dic_plateaus"]["180"]["knots"])[0] /
                  np.diff(record["machine_plateaus"]["180"]["knots"])[0])}
    changes = np.diff(mean)
    record["largest_adjacent_mean_changes_mm"] = [
        {"from_stage": int(i), "to_stage": int(i+1), "change_mm": float(changes[i])}
        for i in np.argsort(abs(changes))[-6:][::-1]]
    # After inspecting the entry transients, apply the SAME 6--9 s dwell
    # window to all retained levels. This is a timing/quality selection only;
    # two same-section points cannot establish full-field elastic stability.
    selected = []
    for level, plateau in record["dic_plateaus"].items():
        entry = plateau["knots"][0]
        selected.append((int(level), int(np.ceil(entry+6*hz)), int(np.floor(entry+9*hz))))
    windows = {run: selected + ([(380, 615, 635)] if run == 3 else [(80, 130, 142)])}
    record["window_checks"] = []
    for level, first, last in windows[run]:
        keep = (k >= first) & (k <= last)
        x = k[keep]/hz
        yy = mean[keep]
        slope, intercept = np.polyfit(x, yy, 1)
        residual = yy-(slope*x+intercept)
        mapped = x-record["offset_median_s"]
        covered = (mapped >= t[0]) & (mapped <= t[-1])
        loads = np.interp(mapped, t, force)-baseline
        center_stage = int(round((first+last)/2))
        center_machine_time = center_stage/hz-record["offset_median_s"]
        record["window_checks"].append({"nominal_level_N": level, "stage_inclusive": [first, last],
            "single_stage": center_stage, "single_stage_time_s": center_stage/hz,
            "single_stage_machine_time_s": center_machine_time,
            "single_stage_force_N": float(np.interp(center_machine_time, t, force)-baseline) if t[0] <= center_machine_time <= t[-1] else None,
            "n": int(keep.sum()), "dic_time_s": [float(x[0]), float(x[-1])],
            "machine_time_s": [float(mapped[0]), float(mapped[-1])],
            "mean_displacement_mm": float(yy.mean()), "std_mean_displacement_mm": float(yy.std(ddof=1)),
            "mean_drift_mm_s": float(slope), "linear_drift_across_window_mm": float(slope*(x[-1]-x[0])),
            "detrended_rms_mm": float(np.sqrt(np.mean(residual**2))),
            "point_slopes_mm_s": [float(np.polyfit(x, points[keep, j], 1)[0]) for j in range(2)],
            "std_point_difference_mm": float(np.std(points[keep, 1]-points[keep, 0], ddof=1)),
            "machine_coverage": bool(covered.all()),
            "matched_mean_force_N": float(loads.mean()) if covered.all() else None})
    reference = next(item for item in record["window_checks"] if item["nominal_level_N"] == 80)
    record["incremental_reference_stage"] = reference["single_stage"]
    for item in record["window_checks"]:
        if item["machine_coverage"]:
            item["force_increment_from_reference_N"] = item["matched_mean_force_N"]-reference["matched_mean_force_N"]
            item["mean_displacement_increment_from_reference_mm"] = item["mean_displacement_mm"]-reference["mean_displacement_mm"]
    records[str(run)] = record

    fig, axes = plt.subplots(3, 1, figsize=(10.5, 8), layout="constrained")
    axes[0].plot(k, points[:, 0], color="#0072B2", lw=0.9, label=labels[0].replace("阶段点 ", "Point "))
    axes[0].plot(k, points[:, 1], color="#D55E00", lw=0.9, label=labels[1].replace("阶段点 ", "Point "))
    axes[0].set(xlabel="CSV stage index (starts at 0)", ylabel="Axial displacement (mm)",
                title=f"Run {run}: observed DIC displacement")
    axes[0].legend(frameon=False)
    td = k/hz-record["offset_median_s"]
    axes[1].plot(td, mean, color="#0072B2", lw=1, label="Mean DIC displacement")
    axes[1].set(xlabel="Estimated machine time (s)", ylabel="Mean displacement (mm)")
    force_ax = axes[1].twinx()
    force_ax.plot(t, force-baseline, color="0.25", lw=1, ls="--", label="Corrected measured force")
    force_ax.set_ylabel("Corrected force (N)")
    for item in record["window_checks"]:
        lo, hi = item["stage_inclusive"]
        axes[0].axvspan(lo, hi, color="#009E73" if item["machine_coverage"] else "0.5", alpha=0.14)
        axes[0].text((lo+hi)/2, mean[lo]+0.025, f"{item['nominal_level_N']} N" + (" (no force record)" if not item["machine_coverage"] else ""), ha="center", fontsize=8)
    for level, color in ((180, "#0072B2"), (280, "#D55E00")):
        if str(level) not in record["dic_plateaus"]:
            continue
        first, last = record["dic_plateaus"][str(level)]["knots"]
        keep = (k >= first-6) & (k <= last+6)
        local_time = (k[keep]-first)/hz
        yy = mean[keep]
        center = yy[(local_time >= 2) & (local_time <= 6)].mean()
        axes[2].plot(local_time, (yy-center)*1000, color=color, lw=1,
                     label=f"{level} N setting (centered)")
    axes[2].set(xlabel="Seconds after fitted plateau entry", ylabel="Centered DIC mean (micrometers)",
                title="Plateau stability; shared cross-section motion, not specimen extension")
    axes[2].legend(frameon=False)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(OUT / f"dic_timing_run{run}.png", dpi=180)
    fig.savefig(OUT / f"dic_timing_run{run}.svg")
    plt.close(fig)

source_paths = [SOURCE/f"{prefix}{run}{suffix}" for run in (3, 4)
                for prefix, suffix in (("DY_", ".csv"), ("sample_2_", ".xlsb"))]
result = {"conventions": __doc__, "python": sys.version, "numpy": np.__version__,
          "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths},
          "records": records}
(OUT / "timing_checks.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
with (OUT / "selected_dic_windows.csv").open("w", newline="", encoding="utf-8-sig") as stream:
    writer = csv.writer(stream)
    writer.writerow(["run", "setting_N", "stage_first", "stage_last", "single_stage", "dic_time_s", "machine_time_s_estimated", "mean_corrected_force_N", "force_increment_from_80N_reference_N", "mean_signal_std_um", "linear_drift_um", "full_force_coverage"])
    for run, record in records.items():
        for item in record["window_checks"]:
            writer.writerow([run, item["nominal_level_N"], *item["stage_inclusive"], item["single_stage"],
                             item["single_stage_time_s"], item["single_stage_machine_time_s"],
                             item["matched_mean_force_N"], item.get("force_increment_from_reference_N"), item["std_mean_displacement_mm"]*1000,
                             item["linear_drift_across_window_mm"]*1000, item["machine_coverage"]])
print(json.dumps(records, ensure_ascii=False, indent=2))
