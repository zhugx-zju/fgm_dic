"""Plot the 220 mm dogbone outline and seven lengthwise infill regions."""

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
# Actual experiment markers, measured from each outer end (not the shoulder).
MARKER_DISTANCE_FROM_OUTER_END_MM = 75.0


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    outline = build_smooth_dogbone_polygon(SPECIMEN_CFG)
    parallel_left = -SPECIMEN_CFG.parallel_length / 2
    roi_half_length = SPECIMEN_CFG.total_length / 2 - MARKER_DISTANCE_FROM_OUTER_END_MM

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
    for x in (-roi_half_length, roi_half_length):
        ax.axvline(x, color="#222222", linestyle=":", linewidth=1.0)
    ax.text(0, 23, f"{2 * roi_half_length:g} mm inverse ROI", ha="center", va="center", fontsize=9)
    ax.set(xlim=(-114, 114), ylim=(-27, 27), xlabel="Tensile axis x (mm)", ylabel="Width y (mm)")
    ax.set_aspect("equal")
    ax.grid(False)
    ax.set_title("220 mm dogbone: longer edge regions and seven full-width infill levels")
    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=300)
    fig.savefig(OUTPUT_PATH.with_suffix(".pdf"))
    plt.close(fig)


if __name__ == "__main__":
    main()
