"""Fit apparent tensile modulus for PLA dogbones at each infill level.

Adapted from the retained 1023/analyze_youngs_modulus.py window selection.
Run with the ``abqinter`` Python environment. Source workbooks are read only;
every run writes to a new result directory. The nominal (outer) cross-section
defines apparent stress, so the reported slope is an effective specimen modulus.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from openpyxl import load_workbook


ROOT = Path(__file__).resolve().parents[1] / "post_tensile_test" / "1023"
GEOMETRY_FILE = Path("28个狗棒试样实验记录表.xlsx")
INFILLS = (40, 50, 60, 70, 80, 90, 100)


@dataclass(frozen=True)
class Settings:
    # Fractions of the smallest peak engineering stress in a group. The upper
    # bound keeps the search away from peak load; curvature checks still decide
    # whether a candidate segment is locally linear.
    search_start_fraction: float = 0.05
    search_end_fraction: float = 0.80
    grid_step_fraction: float = 0.025
    minimum_width_fraction: float = 0.15
    minimum_points: int = 60
    minimum_r2: float = 0.995
    maximum_half_slope_difference: float = 0.05
    minimum_positive_strain_steps: float = 0.90


@dataclass
class Specimen:
    specimen_id: str
    infill: int
    width_mm: float
    thickness_mm: float
    gauge_mm: float
    source: str
    selected_block: int
    time_s: np.ndarray
    force_n: np.ndarray
    displacement_mm: np.ndarray
    extension_mm: np.ndarray
    baseline_force_n: float
    peak_index: int

    @property
    def area_mm2(self) -> float:
        return self.width_mm * self.thickness_mm

    @property
    def stress_mpa(self) -> np.ndarray:
        return (self.force_n - self.baseline_force_n) / self.area_mm2

    @property
    def raw_strain(self) -> np.ndarray:
        return self.extension_mm / self.gauge_mm


@dataclass(frozen=True)
class Fit:
    specimen_id: str
    lower_stress_mpa: float
    upper_stress_mpa: float
    modulus_mpa: float
    intercept_mpa: float
    r2: float
    half_slope_difference: float
    positive_strain_steps: float
    point_count: int
    strain_zero: float
    strain_min: float
    strain_max: float
    force_min_n: float
    force_max_n: float
    accepted: bool


def read_geometry(path: Path) -> dict[str, tuple[float, float, float]]:
    workbook = load_workbook(path, read_only=True, data_only=False)
    try:
        sheet = workbook["试验记录"]
        dimensions = {}
        for row in sheet.iter_rows(min_row=6, values_only=True):
            infill, index, specimen_id, width, thickness, _, gauge = row[:7]
            if specimen_id is None:
                continue
            expected_id = f"{int(infill)}-{int(index)}"
            if str(specimen_id) != expected_id:
                raise ValueError(f"Geometry specimen ID mismatch: {specimen_id!r}")
            if expected_id in dimensions or int(infill) not in INFILLS:
                raise ValueError(f"Duplicate or unsupported specimen: {expected_id}")
            values = (float(width), float(thickness), float(gauge))
            if not all(math.isfinite(v) and v > 0 for v in values):
                raise ValueError(f"Invalid dimensions for {expected_id}: {values}")
            dimensions[expected_id] = values
        if not dimensions:
            raise ValueError("No specimen dimensions recorded")
        return dimensions
    finally:
        workbook.close()


def read_blocks(path: Path) -> list[np.ndarray]:
    frame = pd.read_excel(path, engine="pyxlsb", header=None)
    if frame.shape[1] % 4:
        raise ValueError(f"{path}: expected columns in time/force/displacement/extension blocks")
    blocks = []
    for start in range(0, frame.shape[1], 4):
        headers = tuple(str(value).split("_")[0] for value in frame.iloc[0, start : start + 4])
        if headers != ("时间", "力", "位移", "变形"):
            raise ValueError(f"{path}: unexpected channel order {headers}")
        block = frame.iloc[1:, start : start + 4].apply(pd.to_numeric, errors="coerce")
        block = block.dropna(subset=block.columns[[0, 1, 3]])
        values = block.to_numpy(dtype=float)
        if len(values) < 3:
            blocks.append(np.empty((0, 4)))
            continue
        if np.any(np.diff(values[:, 0]) < 0):
            raise ValueError(f"{path}: time decreases in block {start // 4 + 1}")
        blocks.append(values)
    return blocks


def load_specimens(root: Path, geometry: dict[str, tuple[float, float, float]]) -> tuple[list[Specimen], list[dict]]:
    specimens = []
    block_notes = []
    for infill in INFILLS:
        for path in sorted((root / str(infill)).glob("*.xlsb")):
            try:
                file_infill, index = map(int, path.stem.split("_"))
            except ValueError as exc:
                raise ValueError(f"Unexpected specimen filename: {path.name}") from exc
            if file_infill != infill:
                raise ValueError(f"Filename and folder infill disagree: {path}")
            specimen_id = f"{infill}-{index}"
            if specimen_id not in geometry:
                raise ValueError(f"No dimensions recorded for {specimen_id}")
            blocks = read_blocks(path)
            if not blocks:
                raise ValueError(f"No data blocks: {path}")
            # A valid tensile run has the largest sustained load excursion.
            # Keep every block's peak in the audit note to expose this choice.
            excursions = []
            for block in blocks:
                if len(block) < 10:
                    excursions.append(float("-inf"))
                else:
                    baseline = float(np.median(block[:10, 1]))
                    excursions.append(float(np.max(block[:, 1] - baseline)))
            selected = int(np.argmax(excursions))
            if not math.isfinite(excursions[selected]) or excursions[selected] <= 0:
                raise ValueError(f"No positive loading block: {path}")
            block = blocks[selected]
            baseline = float(np.median(block[:10, 1]))
            peak_index = int(np.argmax(block[:, 1] - baseline))
            if peak_index < 100:
                raise ValueError(f"Peak too early to fit: {path}")
            width, thickness, gauge = geometry[specimen_id]
            specimens.append(
                Specimen(
                    specimen_id=specimen_id,
                    infill=infill,
                    width_mm=width,
                    thickness_mm=thickness,
                    gauge_mm=gauge,
                    source=str(path.relative_to(root)),
                    selected_block=selected + 1,
                    time_s=block[:, 0],
                    force_n=block[:, 1],
                    displacement_mm=block[:, 2],
                    extension_mm=block[:, 3],
                    baseline_force_n=baseline,
                    peak_index=peak_index,
                )
            )
            block_notes.append(
                {
                    "specimen": specimen_id,
                    "source": str(path.relative_to(root)),
                    "block_peak_load_excursions_n": excursions,
                    "selected_block": selected + 1,
                }
            )
    if len({s.specimen_id for s in specimens}) != len(specimens):
        raise ValueError("Duplicate specimen IDs")
    return specimens, block_notes


def linear_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    x_mean = float(np.mean(x))
    y_mean = float(np.mean(y))
    sxx = float(np.dot(x - x_mean, x - x_mean))
    if sxx <= 0:
        return float("nan"), float("nan"), float("nan")
    slope = float(np.dot(x - x_mean, y - y_mean) / sxx)
    intercept = y_mean - slope * x_mean
    residual = y - (slope * x + intercept)
    syy = float(np.dot(y - y_mean, y - y_mean))
    r2 = 1 - float(np.dot(residual, residual)) / syy if syy > 0 else float("nan")
    return slope, intercept, r2


def evaluate_fit(specimen: Specimen, lower: float, upper: float, settings: Settings) -> Fit | None:
    stress = specimen.stress_mpa[: specimen.peak_index + 1]
    strain = specimen.raw_strain[: specimen.peak_index + 1]
    force = specimen.force_n[: specimen.peak_index + 1]
    indices = np.flatnonzero((stress >= lower) & (stress <= upper))
    if len(indices) < settings.minimum_points:
        return None
    x, y, f = strain[indices], stress[indices], force[indices]
    slope, intercept, r2 = linear_fit(x, y)
    if not math.isfinite(slope) or slope <= 0:
        return None
    midpoint = (lower + upper) / 2
    left = y <= midpoint
    right = y > midpoint
    if min(int(np.sum(left)), int(np.sum(right))) < settings.minimum_points // 4:
        return None
    first_slope, _, _ = linear_fit(x[left], y[left])
    last_slope, _, _ = linear_fit(x[right], y[right])
    half_difference = abs(first_slope - last_slope) / slope
    # Small negative steps are common from sensor quantization. Require at
    # least 90% forward steps; R² and slope stability catch larger reversals.
    positive_steps = float(np.mean(np.diff(x) >= 0))
    strain_zero = -intercept / slope
    corrected_strain = x - strain_zero
    accepted = (
        r2 >= settings.minimum_r2
        and half_difference <= settings.maximum_half_slope_difference
        and positive_steps >= settings.minimum_positive_strain_steps
    )
    return Fit(
        specimen_id=specimen.specimen_id,
        lower_stress_mpa=lower,
        upper_stress_mpa=upper,
        modulus_mpa=slope,
        intercept_mpa=intercept,
        r2=r2,
        half_slope_difference=half_difference,
        positive_strain_steps=positive_steps,
        point_count=len(indices),
        strain_zero=strain_zero,
        strain_min=float(np.min(corrected_strain)),
        strain_max=float(np.max(corrected_strain)),
        force_min_n=float(np.min(f)),
        force_max_n=float(np.max(f)),
        accepted=accepted,
    )


def candidate_windows(group: list[Specimen], settings: Settings) -> list[tuple[float, float]]:
    minimum_peak = min(float(np.max(s.stress_mpa[: s.peak_index + 1])) for s in group)
    grid = np.arange(
        settings.search_start_fraction,
        settings.search_end_fraction + settings.grid_step_fraction / 2,
        settings.grid_step_fraction,
    )
    windows = [
        (float(lo * minimum_peak), float(hi * minimum_peak))
        for lo in grid
        for hi in grid
        if hi - lo >= settings.minimum_width_fraction - 1e-12
    ]
    return sorted(windows, key=lambda item: (-(item[1] - item[0]), item[0]))


def select_group_fits(group: list[Specimen], settings: Settings) -> tuple[dict[str, Fit | None], dict]:
    windows = candidate_windows(group, settings)
    if not windows:
        raise ValueError("No candidate intervals")
    best_score = None
    best_fits = None
    best_window = None
    for lower, upper in windows:
        fits = {s.specimen_id: evaluate_fit(s, lower, upper, settings) for s in group}
        passing = [f for f in fits.values() if f is not None and f.accepted]
        # Coverage comes first. For equal coverage, select the widest common
        # interval; the minimum R² and curvature are tie breakers.
        score = (
            len(passing),
            upper - lower,
            min((f.r2 for f in passing), default=-math.inf),
            -max((f.half_slope_difference for f in passing), default=math.inf),
        )
        if best_score is None or score > best_score:
            best_score, best_fits, best_window = score, fits, (lower, upper)
    assert best_fits is not None and best_window is not None
    final: dict[str, Fit | None] = {}
    exceptions = []
    for specimen in group:
        common_fit = best_fits[specimen.specimen_id]
        if common_fit is not None and common_fit.accepted:
            final[specimen.specimen_id] = common_fit
            continue
        # If no common interval works, choose the nearest acceptable local
        # interval rather than silently forcing an invalid common fit.
        options = []
        for lower, upper in windows:
            fit = evaluate_fit(specimen, lower, upper, settings)
            if fit is not None and fit.accepted:
                distance = abs(lower - best_window[0]) + abs(upper - best_window[1])
                options.append((distance, -(upper - lower), -fit.r2, fit))
        if options:
            options.sort(key=lambda item: item[:3])
            final[specimen.specimen_id] = options[0][3]
            exceptions.append({"specimen": specimen.specimen_id, "reason": "individual interval"})
        else:
            final[specimen.specimen_id] = None
            exceptions.append({"specimen": specimen.specimen_id, "reason": "no acceptable linear interval"})
    audit = {
        "infill_percent": group[0].infill,
        "common_stress_interval_mpa": list(best_window),
        "common_interval_accepted": best_score[0],
        "specimen_count": len(group),
        "exceptions": exceptions,
    }
    return final, audit


def plot_group(group: list[Specimen], fits: dict[str, Fit | None], audit: dict, directory: Path) -> None:
    fig, (ax_full, ax_fit) = plt.subplots(1, 2, figsize=(12.5, 5), constrained_layout=True)
    colors = plt.cm.tab10(np.linspace(0, 1, len(group)))
    fit_strains = []
    for color, specimen in zip(colors, group):
        fit = fits[specimen.specimen_id]
        if fit is None:
            strain_zero = float(specimen.raw_strain[0])
        else:
            strain_zero = fit.strain_zero
        x = (specimen.raw_strain - strain_zero) * 100
        y = specimen.stress_mpa
        label = specimen.specimen_id
        ax_full.plot(x, y, color=color, linewidth=1, label=label)
        ax_fit.plot(x[: specimen.peak_index + 1], y[: specimen.peak_index + 1], color=color, alpha=0.28, linewidth=1)
        if fit is not None:
            fit_strains.extend((fit.strain_min * 100, fit.strain_max * 100))
            in_fit = (y >= fit.lower_stress_mpa) & (y <= fit.upper_stress_mpa)
            in_fit[specimen.peak_index + 1 :] = False
            ax_fit.plot(x[in_fit], y[in_fit], color=color, linewidth=2.5, label=f"{label}: {fit.modulus_mpa:.0f} MPa")
            line_x = np.array([fit.strain_min, fit.strain_max])
            ax_fit.plot(line_x * 100, fit.modulus_mpa * line_x, color=color, linestyle="--", linewidth=1.2)
        else:
            ax_fit.plot([], [], color=color, label=f"{label}: fit unavailable")
    lower, upper = audit["common_stress_interval_mpa"]
    ax_fit.axhspan(lower, upper, color="grey", alpha=0.07, zorder=0)
    if fit_strains:
        x_span = max(fit_strains) - min(fit_strains)
        x_margin = max(x_span * 0.12, 0.03)
        ax_fit.set_xlim(min(fit_strains) - x_margin, max(fit_strains) + x_margin)
        y_margin = (upper - lower) * 0.18
        ax_fit.set_ylim(lower - y_margin, upper + y_margin)
    ax_full.set_title(f"{group[0].infill}% infill — complete curves")
    ax_fit.set_title(f"Linear fits — common stress {lower:.1f}–{upper:.1f} MPa")
    for axis in (ax_full, ax_fit):
        axis.set_xlabel("Engineering strain (%)")
        axis.set_ylabel("Apparent engineering stress (MPa)")
        axis.grid(alpha=0.2, linewidth=0.5)
    ax_fit.legend(fontsize=8, loc="best")
    fig.savefig(directory / "fit_curves.png", dpi=220)
    fig.savefig(directory / "fit_curves.svg")
    plt.close(fig)


def plot_raw_force_displacement(group: list[Specimen], root: Path, directory: Path) -> None:
    """Plot uncorrected machine force against uncorrected crosshead displacement."""
    fig, ax = plt.subplots(figsize=(8, 5.5), constrained_layout=True)
    colors = plt.cm.tab10(np.linspace(0, 1, len(group)))
    for color, specimen in zip(colors, group):
        blocks = read_blocks(root / specimen.source)
        for block_index, block in enumerate(blocks, start=1):
            if len(block) == 0:
                continue
            selected = block_index == specimen.selected_block
            label = specimen.specimen_id if len(blocks) == 1 else f"{specimen.specimen_id} block {block_index}"
            if not selected:
                label += " (excluded run)"
            ax.plot(
                block[:, 2],
                block[:, 1],
                color=color,
                linewidth=1.2 if selected else 1,
                linestyle="-" if selected else ":",
                alpha=1 if selected else 0.8,
                label=label,
            )
    ax.set_title(f"{group[0].infill}% infill — raw force–displacement")
    ax.set_xlabel("Machine displacement (mm)")
    ax.set_ylabel("Raw force (N)")
    ax.legend(title="Specimen", fontsize=9, loc="upper left")
    ax.grid(alpha=0.2, linewidth=0.5)
    fig.savefig(directory / "raw_force_displacement.png", dpi=220)
    fig.savefig(directory / "raw_force_displacement.svg")
    plt.close(fig)


def plot_summary(rows: list[dict], directory: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    for infill in INFILLS:
        values = [row["E_MPa"] for row in rows if row["infill_percent"] == infill and math.isfinite(row["E_MPa"])]
        if not values:
            continue
        positions = np.linspace(infill - 1.5, infill + 1.5, len(values))
        ax.scatter(positions, values, color="#1764a3", s=30, alpha=0.8)
        ax.scatter([infill], [float(np.mean(values))], color="#c44e2b", marker="D", s=55, zorder=3)
    ax.set_xticks(INFILLS)
    ax.set_xlabel("Infill (%)")
    ax.set_ylabel("Apparent Young's modulus (MPa)")
    ax.grid(axis="y", alpha=0.2, linewidth=0.5)
    fig.savefig(directory / "modulus_by_infill.png", dpi=220)
    fig.savefig(directory / "modulus_by_infill.svg")
    plt.close(fig)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def window_sensitivity(
    group: list[Specimen], fits: dict[str, Fit | None], audit: dict, settings: Settings
) -> dict:
    lower, upper = audit["common_stress_interval_mpa"]
    width = upper - lower
    variants = {
        "left_10pct_width": (lower - width * 0.10, upper - width * 0.10),
        "right_10pct_width": (lower + width * 0.10, upper + width * 0.10),
        "trim_15pct_each_end": (lower + width * 0.15, upper - width * 0.15),
    }
    changes = {}
    for label, (new_lower, new_upper) in variants.items():
        specimen_changes = {}
        for specimen in group:
            base = fits[specimen.specimen_id]
            alternative = evaluate_fit(specimen, new_lower, new_upper, settings)
            if base is not None and alternative is not None:
                specimen_changes[specimen.specimen_id] = round(
                    100 * (alternative.modulus_mpa / base.modulus_mpa - 1), 3
                )
        changes[label] = specimen_changes
    return changes


def specimen_diagnostics(specimen: Specimen, fit: Fit | None) -> dict:
    if fit is None:
        return {"specimen": specimen.specimen_id, "fit_available": False}
    stress = specimen.stress_mpa[: specimen.peak_index + 1]
    first_fit_index = int(np.flatnonzero(stress >= fit.lower_stress_mpa)[0])
    initial_extension = specimen.extension_mm[: first_fit_index + 1]
    initial_displacement = specimen.displacement_mm[: first_fit_index + 1]
    within_fit = (stress >= fit.lower_stress_mpa) & (stress <= fit.upper_stress_mpa)
    _, _, crosshead_r2 = linear_fit(
        specimen.displacement_mm[: specimen.peak_index + 1][within_fit],
        specimen.force_n[: specimen.peak_index + 1][within_fit],
    )
    loading_force = specimen.force_n[: specimen.peak_index + 1]
    diagnostic_force_range = (loading_force >= 150) & (loading_force <= 400)
    if int(np.sum(diagnostic_force_range)) >= 20:
        crosshead_slope, _, crosshead_range_r2 = linear_fit(
            specimen.displacement_mm[: specimen.peak_index + 1][diagnostic_force_range],
            loading_force[diagnostic_force_range],
        )
    else:
        crosshead_slope, crosshead_range_r2 = float("nan"), float("nan")
    return {
        "specimen": specimen.specimen_id,
        "fit_available": True,
        "fit_start_fraction_of_peak_raw_force": fit.force_min_n / float(np.max(specimen.force_n[: specimen.peak_index + 1])),
        "pre_fit_extension_drawdown_mm": float(np.max(np.maximum.accumulate(initial_extension) - initial_extension)),
        "pre_fit_displacement_drawdown_mm": float(
            np.max(np.maximum.accumulate(initial_displacement) - initial_displacement)
        ),
        "crosshead_force_displacement_R2_within_fit": crosshead_r2,
        "crosshead_stiffness_150_400N_N_per_mm": crosshead_slope,
        "crosshead_stiffness_150_400N_R2": crosshead_range_r2,
        "peak_raw_force_n": float(np.max(loading_force)),
        "fit_half_slope_difference": fit.half_slope_difference,
        "fit_point_count": fit.point_count,
    }


def analyze(root: Path, run_id: str, settings: Settings, *,
            geometry_file: Path = GEOMETRY_FILE, batch: str = "1023") -> None:
    if not run_id or any(char in run_id for char in "\\/:*?\"<>|"):
        raise ValueError(f"Invalid run ID: {run_id!r}")
    metadata_dir = root / "analysis_results" / run_id
    comparison_dir = root / "comparison_results" / run_id
    group_dirs = {infill: root / str(infill) / "analysis_results" / run_id for infill in INFILLS}
    destinations = [metadata_dir, comparison_dir, *group_dirs.values()]
    existing = [path for path in destinations if path.exists()]
    if existing:
        raise FileExistsError(f"Run output already exists: {existing}")
    geometry_path = geometry_file if geometry_file.is_absolute() else root / geometry_file
    geometry = read_geometry(geometry_path)
    specimens, block_notes = load_specimens(root, geometry)
    if not specimens:
        raise ValueError("No tensile data files found")
    missing = set(geometry) - {s.specimen_id for s in specimens}
    if missing:
        raise ValueError(f"Recorded specimens have no tensile workbook: {sorted(missing)}")
    by_infill = {infill: [s for s in specimens if s.infill == infill] for infill in INFILLS}
    all_fits = {}
    audits = []
    for infill, group in by_infill.items():
        if not group:
            continue
        group_fits, audit = select_group_fits(group, settings)
        all_fits.update(group_fits)
        audit["window_sensitivity_percent"] = window_sensitivity(group, group_fits, audit, settings)
        relaxed_fits, _ = select_group_fits(group, replace(settings, maximum_half_slope_difference=0.10))
        primary_values = [fit.modulus_mpa for fit in group_fits.values() if fit is not None]
        relaxed_values = [fit.modulus_mpa for fit in relaxed_fits.values() if fit is not None]
        audit["relaxed_10pct_half_slope_mean_E_MPa"] = float(np.mean(relaxed_values))
        audit["relaxed_10pct_vs_primary_mean_change_percent"] = float(
            100 * (np.mean(relaxed_values) / np.mean(primary_values) - 1)
        )
        audits.append(audit)
    rows = []
    for specimen in specimens:
        fit = all_fits[specimen.specimen_id]
        rows.append(
            {
                "infill_percent": specimen.infill,
                "specimen_id": specimen.specimen_id,
                "E_MPa": fit.modulus_mpa if fit is not None else float("nan"),
                "R2": fit.r2 if fit is not None else float("nan"),
                "strain_min_pct": fit.strain_min * 100 if fit is not None else float("nan"),
                "strain_max_pct": fit.strain_max * 100 if fit is not None else float("nan"),
                "force_min_N": fit.force_min_n if fit is not None else float("nan"),
                "force_max_N": fit.force_max_n if fit is not None else float("nan"),
                "width_mm": specimen.width_mm,
                "thickness_mm": specimen.thickness_mm,
                "area_mm2": specimen.area_mm2,
                "gauge_mm": specimen.gauge_mm,
                "baseline_force_N": specimen.baseline_force_n,
                "lower_stress_MPa": fit.lower_stress_mpa if fit else float("nan"),
                "upper_stress_MPa": fit.upper_stress_mpa if fit else float("nan"),
                "fit_method": ("common_stress_window_OLS" if fit and
                    [fit.lower_stress_mpa, fit.upper_stress_mpa] == next(
                        a["common_stress_interval_mpa"] for a in audits
                        if a["infill_percent"] == specimen.infill)
                    else "individual_exception_OLS" if fit else "unavailable"),
            }
        )
    for destination in destinations:
        destination.mkdir(parents=True)
    for infill, group in by_infill.items():
        if not group:
            continue
        pd.DataFrame([row for row in rows if row["infill_percent"] == infill]).to_csv(
            group_dirs[infill] / "fit_results.csv", index=False, encoding="utf-8-sig"
        )
    summaries = []
    for infill in INFILLS:
        values = np.array([row["E_MPa"] for row in rows if row["infill_percent"] == infill and math.isfinite(row["E_MPa"])])
        summaries.append(
            {
                "infill_percent": infill,
                "batch": batch,
                "fit_method": ("common_stress_window_OLS" if all(
                    row["fit_method"] == "common_stress_window_OLS" for row in rows
                    if row["infill_percent"] == infill and math.isfinite(row["E_MPa"]))
                    else "common_window_with_individual_exceptions_OLS"),
                "n_input": len(by_infill[infill]),
                "n_excluded": len(by_infill[infill]) - len(values),
                "n": len(values),
                "mean_E_MPa": float(np.mean(values)) if len(values) else float("nan"),
                "sd_E_MPa": float(np.std(values, ddof=1)) if len(values) > 1 else float("nan"),
                "cv_percent": float(np.std(values, ddof=1) / np.mean(values) * 100) if len(values) > 1 else float("nan"),
            }
        )
    pd.DataFrame(summaries).to_csv(metadata_dir / "group_summary.csv", index=False, encoding="utf-8-sig")
    for audit in audits:
        infill = audit["infill_percent"]
        plot_group(by_infill[infill], all_fits, audit, group_dirs[infill])
        plot_raw_force_displacement(by_infill[infill], root, group_dirs[infill])
    plot_summary(rows, comparison_dir)
    metadata = {
        "method": "OLS stress on extensometer strain within a selected pre-peak stress interval",
        "meaning": "effective/apparent modulus using outer cross-sectional area",
        "strain_zero": "zero-stress intercept extrapolated from each selected linear fit",
        "force_range": "raw instrument force, including initial offset",
        "crosshead_diagnostic": "150–400 N raw force versus machine displacement stiffness; not a Young's modulus",
        "geometry_source": str(geometry_path),
        "batch": batch,
        "batch_note": "User identified these as the current printed specimens; batch is the supplied directory label, not a verified calendar date.",
        "units": {"force": "N", "extension": "mm", "dimensions": "mm", "modulus": "MPa"},
        "unit_note": "Dimensions and 50 mm gauge are recorded in the geometry workbook; instrument force and deformation channels are interpreted in the existing project units N and mm.",
        "baseline_rule": "Median of first 10 force samples per selected run, not inherited from a previous batch. A constant force offset does not affect the OLS slope with intercept, but can affect window selection.",
        "settings": asdict(settings),
        "groups": audits,
        "source_blocks": block_notes,
        "specimen_diagnostics": [specimen_diagnostics(s, all_fits[s.specimen_id]) for s in specimens],
        "missing_specimens": sorted(set(geometry) - {s.specimen_id for s in specimens}),
        "output_layout": {
            "group_results": {str(infill): str(group_dirs[infill].relative_to(root)) for infill in INFILLS},
            "comparison": str(comparison_dir.relative_to(root)),
            "summary_and_metadata": str(metadata_dir.relative_to(root)),
        },
        "source_sha256": {
            str(geometry_path): file_sha256(geometry_path),
            **{s.source: file_sha256(root / s.source) for s in specimens},
        },
        "script_sha256": file_sha256(Path(__file__).resolve()),
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "matplotlib": matplotlib.__version__,
            "openpyxl": importlib.metadata.version("openpyxl"),
            "pyxlsb": importlib.metadata.version("pyxlsb"),
        },
    }
    (metadata_dir / "analysis_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(pd.DataFrame(summaries).to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    for audit in audits:
        print(
            f"{audit['infill_percent']}%: common {audit['common_stress_interval_mpa'][0]:.2f}–"
            f"{audit['common_stress_interval_mpa'][1]:.2f} MPa; "
            f"{audit['common_interval_accepted']}/{audit['specimen_count']} accepted; "
            f"exceptions={audit['exceptions']}"
        )
    print(f"Run ID: {run_id}")
    print(f"Group results: <infill>/analysis_results/{run_id}")
    print(f"Comparison: {comparison_dir}")
    print(f"Metadata and summary: {metadata_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT, help="Directory with infill folders and geometry workbook")
    parser.add_argument("--run-id", help="Unique result subfolder name; existing runs are refused")
    parser.add_argument("--geometry-file", type=Path, default=GEOMETRY_FILE)
    parser.add_argument("--batch", default="1023", help="Explicit source batch label")
    args = parser.parse_args()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = args.run_id or f"modulus_{stamp}"
    analyze(args.root.resolve(), run_id, Settings(), geometry_file=args.geometry_file, batch=args.batch)


if __name__ == "__main__":
    main()
