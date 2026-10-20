"""Plot the 220 mm dogbone outline and seven lengthwise infill regions."""

import json
import importlib.util
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle

from gen_long_gradient_dogbone_specimen import (
    OUTPUT_DIR,
    REGION_DENSITIES_PERCENT,
    REGION_PARALLEL_LENGTHS_MM,
    SPECIMEN_CFG,
)
from stl_builders import build_smooth_dogbone_polygon


OUTPUT_PATH = OUTPUT_DIR / "dogbone_220mm_region_schematic.png"
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    outline = build_smooth_dogbone_polygon(SPECIMEN_CFG)
    parallel_left = -SPECIMEN_CFG.parallel_length / 2
    spec = importlib.util.spec_from_file_location("active_dic_plot_config", PROJECT_ROOT / "dic_case_config.py")
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    prepared = Path(config.CASE_OUTPUT) / "01_prepared" / "run_metadata.json"
    if not prepared.exists():
        raise FileNotFoundError("Run python -m script.step1_prepare_dic before plotting the observed ROI")
    record = json.loads(prepared.read_text(encoding="utf-8"))
    placement = record["material_registration"]
    marks = sorted(placement["manufacturing_mark_x_mm"])
    low_distance = record["registration"]["low_infill_mark_end_distance_mm"]
    high_distance = record["registration"]["high_infill_mark_end_distance_mm"]
    roi_left, roi_right = sorted(placement["manufacturing_roi_x_mm"])
    material_sign = placement.get("manufacturing_x_fem_coefficient", 1)
    if abs(SPECIMEN_CFG.total_length - record["registration"]["specimen_length_mm"]) > 1e-8:
        raise ValueError("Printed specimen and prepared DIC must use the same nominal length")

    fig, ax = plt.subplots(figsize=(12, 3.5), dpi=250)
    specimen_patch = Polygon(outline, closed=True, facecolor="#f1f1f1", edgecolor="none")
    ax.add_patch(specimen_patch)
    colors = plt.get_cmap("viridis")
    for index, density in enumerate(REGION_DENSITIES_PERCENT):
        parallel_right = parallel_left + REGION_PARALLEL_LENGTHS_MM[index]
        region_left = -SPECIMEN_CFG.total_length / 2 if index == 0 else parallel_left
        region_right = (
            SPECIMEN_CFG.total_length / 2
            if index == len(REGION_DENSITIES_PERCENT) - 1
            else parallel_right
        )
        width = (
            SPECIMEN_CFG.grip_width
            if index in (0, len(REGION_DENSITIES_PERCENT) - 1)
            else SPECIMEN_CFG.gauge_width
        )
        patch = Rectangle(
            (region_left, -width / 2),
            region_right - region_left,
            width,
            facecolor=colors(0.18 + 0.68 * index / (len(REGION_DENSITIES_PERCENT) - 1)),
            edgecolor="white",
            linewidth=0.6,
            alpha=0.8,
        )
        patch.set_clip_path(specimen_patch)
        ax.add_patch(patch)
        label_x = (parallel_left + parallel_right) / 2
        ax.text(
            label_x,
            0,
            f"X{index + 1}\n{density}%",
            ha="center",
            va="center",
            fontsize=9,
            color="white" if index <= 3 else "black",
        )
        parallel_left = parallel_right

    ax.text(-100, 0, "40%\ngrip", ha="center", va="center", fontsize=8, color="white")
    ax.text(100, 0, "100%\ngrip", ha="center", va="center", fontsize=8, color="black")

    ax.add_patch(Polygon(outline, closed=True, fill=False, edgecolor="#222222", linewidth=1.3))
    for x in (-75, 75):
        ax.axvline(x, color="#444444", linestyle="--", linewidth=0.8)
    for x in marks:
        ax.axvline(x, color="#222222", linestyle=":", linewidth=1.0)
    ax.annotate(f"{low_distance:g} mm to low-infill end; mark {marks[0]:+g}",
                xy=(marks[0], 10), xytext=(-64, 23), fontsize=8, ha="center",
                arrowprops={"arrowstyle": "->", "linewidth": 0.7})
    ax.annotate(f"{high_distance:g} mm to high-infill end; mark {marks[1]:+g}",
                xy=(marks[1], 10), xytext=(76, 23), fontsize=8, ha="center",
                arrowprops={"arrowstyle": "->", "linewidth": 0.7})
    ax.plot([roi_left, roi_right], [-14, -14], color="#c33a2b", linewidth=2)
    ax.text((roi_left+roi_right)/2, -18,
            f"DIC span {roi_right-roi_left:.6f} mm (independent mm scale)",
            ha="center", fontsize=8, color="#9a2920")
    direction = "opposite" if material_sign == -1 else "same"
    ax.set(xlim=(-114, 114), ylim=(-29, 29),
           xlabel=f"Manufacturing xi (mm); FEM x is {direction} to xi",
           ylabel="Width (mm)")
    ax.set_aspect("equal")
    ax.grid(False)
    ax.set_title("220 mm dogbone: longer edge regions and seven full-width infill levels")
    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=300)
    fig.savefig(OUTPUT_PATH.with_suffix(".pdf"))
    fig.savefig(OUTPUT_PATH.with_suffix(".svg"))
    plt.close(fig)


if __name__ == "__main__":
    main()
