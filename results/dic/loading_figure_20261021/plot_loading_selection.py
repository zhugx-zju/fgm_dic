"""Plot the loading and DIC windows actually frozen in results/dic.

Run from any directory. No fitting, inverse solving, or input modification.
The two same-section DIC points are timing signals, not specimen extension.
"""
from pathlib import Path
import csv
import hashlib
import json
import platform
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from pyxlsb import open_workbook


OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
SAMPLE = ROOT / "post_tensile_test/1016/sample_2"
TIMING = SAMPLE / "analysis_results/dic_timing_20261018/timing_checks.json"
FROZEN = OUT.parent / "01_prepared/run_metadata.json"
COLORS = {"180N": "#0072B2", "280N": "#D55E00"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_inputs():
    frozen = json.loads(FROZEN.read_text(encoding="utf-8"))
    timing = json.loads(TIMING.read_text(encoding="utf-8"))
    record = timing["records"]["3"]
    dic_path = SAMPLE / "DY_3.csv"
    machine_path = SAMPLE / "sample_2_3.xlsb"
    # Check raw identities against the pre-existing timing analysis.
    for path in (dic_path, machine_path):
        expected = next(v for k, v in timing["source_sha256"].items()
                        if Path(k).name == path.name)
        assert sha256(path) == expected, f"Timing source changed: {path}"
    blocks = []
    for row in csv.reader(dic_path.read_text(encoding="gb18030").splitlines()):
        if row and "-DY" in row[0]:
            blocks.append([])
        elif len(row) == 3 and row[0] == "":
            blocks[-1].append([float(row[1]), float(row[2])])
    points = [np.asarray(block) for block in blocks]
    assert len(points) == 2
    stages = points[0][:, 0]
    assert np.array_equal(stages, np.arange(record["n"]))
    assert np.array_equal(stages, points[1][:, 0])
    displacement = np.column_stack([p[:, 1] for p in points])
    with open_workbook(str(machine_path)) as book:
        with book.get_sheet(1) as sheet:
            rows = [[c.v for c in row] for row in sheet.rows()]
    machine = np.asarray(rows[1:], dtype=float)[:, :2]
    assert np.all(np.diff(machine[:, 0]) > 0)
    hz = record["nominal_hz"]
    offset = record["offset_median_s"]
    time = stages / hz - offset
    baseline = float(np.median(machine[machine[:, 0] <= 0.08, 1]))
    assert baseline == record["baseline_force_N"] == 60.1
    checks = []
    # Validate the actual full-field files, not merely the current config.
    for case, pair in frozen["force_pairing"].items():
        assert pair["run"] == 3
        sources = frozen["platforms"][case]["sources"]
        actual_stages = [s["stage"] for s in sources]
        wanted = list(range(pair["stage_first"], pair["stage_last"] + 1))
        assert actual_stages == wanted and len(sources) == 15
        for source in sources:
            path = Path(source["path"])
            assert sha256(path) == source["sha256"], f"Frozen DIC source changed: {path}"
        lo, hi = wanted[0], wanted[-1]
        assert np.all((time[wanted] >= machine[0, 0]) &
                      (time[wanted] <= machine[-1, 0]))
        loads = np.interp(time[wanted], machine[:, 0], machine[:, 1]) - baseline
        np.testing.assert_allclose(loads.mean(), pair["mean_corrected_force_N"], atol=1e-10, rtol=0)
        knots = record["dic_plateaus"][str(pair["setting_N"])]["knots"]
        assert lo == int(np.ceil(knots[0] + 6 * hz))
        assert hi == int(np.floor(knots[0] + 9 * hz))
        checks.append({"case": case, "stage_inclusive": [lo, hi], "frames": len(wanted),
                       "representative_stage": pair["single_stage"],
                       "machine_time_s_estimated": time[wanted][[0, -1]].tolist(),
                       "mean_corrected_force_N_recomputed": float(loads.mean()),
                       "full_field_source_hashes_verified": len(sources)})
    return frozen, record, stages, displacement, time, machine, checks


def draw(frozen, record, stages, displacement, time, machine, language):
    cn = language == "zh"
    def tr(en, zh):
        return zh if cn else en
    plt.rcParams.update({
        "font.family": "Microsoft YaHei" if cn else "Arial",
        "font.size": 12, "axes.labelsize": 13, "axes.titlesize": 14,
        "axes.linewidth": 0.9, "xtick.direction": "out", "ytick.direction": "out",
        "xtick.major.size": 4, "ytick.major.size": 4,
        "svg.fonttype": "path", "axes.unicode_minus": False,
        "savefig.facecolor": "white", "figure.facecolor": "white",
    })
    fig, (force_ax, dic_ax) = plt.subplots(2, 1, figsize=(12, 6.75), sharex=True)
    fig.subplots_adjust(left=0.095, right=0.975, top=0.915, bottom=0.17, hspace=0.27)
    force_ax.plot(machine[:, 0], machine[:, 1], color="#292929", lw=1.7)
    mean = displacement.mean(axis=1)
    covered = (time >= machine[0, 0]) & (time <= machine[-1, 0])
    dic_ax.plot(time[covered], mean[covered], color="#4D4D4D", lw=1.35)
    force_ax.set_ylim(40, 425)
    force_ax.set_yticks([80, 180, 280, 380])
    force_ax.set_ylabel(tr("Machine force reading (N)", "试验机力示值 (N)"))
    dic_ax.set_ylabel(tr("Mean axial displacement (mm)", "两点平均轴向位移 (mm)"))
    dic_ax.set_xlabel(tr("Machine time (s)", "试验机时间 (s)"))
    dic_ax.set_ylim(-0.03, float(mean[covered].max()) + 0.10)
    dic_ax.set_xlim(0, machine[-1, 0])
    dic_ax.set_xticks(np.arange(0, machine[-1, 0] + 0.1, 20))
    for ax, label in ((force_ax, tr("(a)  Loading history", "(a)  加载过程")),
                      (dic_ax, tr("(b)  DIC frame selection", "(b)  DIC 时刻选取"))):
        ax.set_title(label, loc="left", pad=13, fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=11)
    force_ax.tick_params(labelbottom=False)
    # Show actual machine plateaus; their boundaries are inherited, not refitted.
    for level in (80, 180, 280):
        k1, k2 = record["machine_plateaus"][str(level)]["knots"]
        force_ax.annotate("", xy=(k1, level + 47), xytext=(k2, level + 47),
                          arrowprops={"arrowstyle": "|-|", "color": "0.45", "lw": 0.8})
        force_ax.text((k1+k2)/2, level + 58, tr("10 s hold", "保载 10 s"),
                      ha="center", va="bottom", fontsize=10, color="0.35")
        label_x = (k1 + k2) / 2 if level == 80 else k1 - 1.3
        force_ax.text(label_x, level + 8 if level == 80 else level - 19, f"{level} N",
                      ha="center" if level == 80 else "right",
                      va="bottom" if level == 80 else "top", fontsize=11)
    force_ax.text(0.99, 0.96, tr("Ramp rate ≈ 4 N/s", "加载速率约 4 N/s"),
                  transform=force_ax.transAxes, ha="right", va="top", fontsize=11, color="0.3")
    legend = []
    for case, pair in frozen["force_pairing"].items():
        color = COLORS[case]
        lo, hi, center = pair["stage_first"], pair["stage_last"], pair["single_stage"]
        for ax in (force_ax, dic_ax):
            ax.axvspan(time[lo], time[hi], color=color, alpha=0.14, lw=0)
            for edge in (lo, hi):
                ax.axvline(time[edge], color=color, lw=0.75, alpha=0.65, ls=(0, (3, 3)))
        force_ax.plot(time[lo:hi+1], np.interp(time[lo:hi+1], machine[:, 0], machine[:, 1]),
                      color=color, lw=3)
        dic_ax.plot(time[lo:hi+1], mean[lo:hi+1], color=color, lw=2.1)
        dic_ax.scatter(time[center], mean[center], s=37, marker="D", color=color,
                       edgecolors="white", linewidths=0.7, zorder=5)
        role = tr("Identification", "识别") if case == "180N" else tr("Prediction", "预测")
        legend.append(Line2D([0], [0], color=color, lw=3, label=f"{case[:-1]} N  ·  {role}"))
        # Labels sit above the signal, with leaders identifying the actual frames.
        text_y = mean[center] + 0.17
        text_x = time[center] - (10 if case == "180N" else 7)
        text = tr(f"Frames {lo}–{hi} (15-frame mean)\nSingle-frame option: {center}",
                  f"第 {lo}–{hi} 帧（15 帧平均）\n可选单帧：{center}")
        dic_ax.annotate(text, xy=(time[center], mean[center]), xytext=(text_x, text_y),
                        color=color, fontsize=11, ha="center", va="bottom",
                        arrowprops={"arrowstyle": "-", "color": color, "lw": 0.9,
                                    "shrinkA": 4, "shrinkB": 6})
    force_ax.legend(handles=legend, frameon=False, loc="upper left",
                    bbox_to_anchor=(0.015, 1.015), ncol=2, fontsize=11,
                    handlelength=1.6, columnspacing=2)
    # Figure notes give the selection rule and force definition, not diagnostic claims.
    fig.text(0.095, 0.068,
             tr("Selection: 6–9 s after fitted DIC plateau entry; 5 Hz; 15 frames per window.",
                "选取规则：DIC 拟合平台入口后 6–9 s；采样率 5 Hz；每个平台取 15 帧。"),
             fontsize=11, color="#303030")
    fig.text(0.095, 0.025,
             tr("Specimen force = machine reading − 60.1 N: 119.78 N (identification), 219.74 N (prediction).",
                "试样拉力 = 试验机示值 − 60.1 N：识别平台 119.78 N，预测平台 219.74 N。"),
             fontsize=10.5, color="#303030")
    stem = OUT / f"loading_dic_selection_{language}"
    fig.savefig(stem.with_suffix(".png"), dpi=400)
    fig.savefig(stem.with_suffix(".svg"))
    svg_path = stem.with_suffix(".svg")
    svg_path.write_text("\n".join(line.rstrip() for line in
                                 svg_path.read_text(encoding="utf-8").splitlines()) + "\n",
                        encoding="utf-8")
    plt.close(fig)


def main():
    frozen, record, stages, displacement, time, machine, checks = load_inputs()
    for language in ("en", "zh"):
        draw(frozen, record, stages, displacement, time, machine, language)
    paired = np.full(len(stages), np.nan)
    keep = (time >= machine[0, 0]) & (time <= machine[-1, 0])
    paired[keep] = np.interp(time[keep], machine[:, 0], machine[:, 1])
    selected = np.zeros(len(stages), dtype=int)
    for pair in frozen["force_pairing"].values():
        selected[pair["stage_first"]:pair["stage_last"]+1] = pair["setting_N"]
    np.savetxt(OUT / "timing_curve_data.csv",
               np.column_stack([stages, time, displacement, displacement.mean(axis=1), paired,
                                paired - record["baseline_force_N"], selected]), delimiter=",",
               header="stage_0based,machine_time_s_estimated,point0_DY_mm,point1_DY_mm,mean_DY_mm,machine_reading_N,corrected_force_N,selected_setting_N",
               comments="", fmt="%.12g")
    sources = [FROZEN, TIMING, SAMPLE / "DY_3.csv", SAMPLE / "sample_2_3.xlsb",
               Path(frozen["configuration"]["FORCE_PAIRING_CSV"])]
    metadata = {"case_id": frozen["case_id"], "target_result": "results/dic/结果总览.html",
                "run": 3, "frequency_Hz": record["nominal_hz"],
                "time_mapping": "t_machine = stage / 5 - 5.618778938656367 s (estimated)",
                "force_baseline_N": record["baseline_force_N"],
                "plotted_time_range_s": [0, float(machine[-1, 0])],
                "verification": checks,
                "source_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in sources},
                "script_sha256": sha256(Path(__file__)),
                "environment": {"python": platform.python_version(), "numpy": np.__version__,
                                "matplotlib": matplotlib.__version__},
                "git_head_before_figure_commit": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "limitations": ["Synchronization is estimated from plateau events; no photo timestamp verification.",
                                "Two same-section points indicate motion timing, not specimen extension or full-field stability.",
                                "Only the common force/DIC time range is plotted; no force extrapolation."]}
    (OUT / "figure_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(checks, ensure_ascii=False, indent=2))
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
