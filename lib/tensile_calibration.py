"""Shared tensile regression, candidate windows and channel reader (mm/N/MPa)."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import math
import numpy as np
import pandas as pd


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
