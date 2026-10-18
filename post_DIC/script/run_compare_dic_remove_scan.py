from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config as cfg
from lib.dic_grid_utils import reconstruct_structured_grid
from lib.dic_preprocessing import plot_grid_fields
from lib.fixed_point_reference import apply_fixed_point_reference_to_grid
from lib.processing_utils import load_raw_processed
from lib.rigid_body_reference import remove_rigid_body_motion_from_grid


def _corrcoef(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    a_valid = np.asarray(a[mask], dtype=float).ravel()
    b_valid = np.asarray(b[mask], dtype=float).ravel()
    return float(np.corrcoef(a_valid, b_valid)[0, 1])


def _demean(field: np.ndarray) -> np.ndarray:
    return np.asarray(field, dtype=float) - float(np.nanmean(field))


case_base = cfg.DIC_EXPORT_BASENAME
analysis_dir = PROJECT_ROOT / "data_analysis"
raw_csv = analysis_dir / f"{case_base}_grid.csv"
remove_csv = analysis_dir / f"{case_base}_grid_remove.csv"

raw_df = load_raw_processed(raw_csv, required_fields=["X", "Y", "DX", "DY", "grid_row", "grid_col"])
remove_df = load_raw_processed(remove_csv, required_fields=["X", "Y", "DX", "DY", "grid_row", "grid_col"])
raw_grid = reconstruct_structured_grid(raw_df)
remove_grid = reconstruct_structured_grid(remove_df)

x_grid = np.asarray(raw_grid["X_ref"], dtype=float)
y_grid = np.asarray(raw_grid["Y_ref"], dtype=float)
ux_remove = np.asarray(remove_grid["DX"], dtype=float)
uy_remove = np.asarray(remove_grid["DY"], dtype=float)
ux_remove_demean = _demean(ux_remove)
uy_remove_demean = _demean(uy_remove)

ratios = [0.05, 0.08, 0.10, 0.12, 0.15, 0.20]
summary_lines = [
    f"fixed_end_side={cfg.PREPROCESS_FIXED_END_SIDE}",
    f"strip_y_margin_ratio={cfg.PREPROCESS_STRIP_Y_MARGIN_RATIO}",
    f"fixed_point={cfg.PREPROCESS_FIXED_POINT}",
    "",
    "ratio,strip_points,strip_residual_rms,ux_corr,uy_corr,ux_demeaned_rms,uy_demeaned_rms,score",
]

best = None
for ratio in ratios:
    rigid_only = remove_rigid_body_motion_from_grid(
        x_grid=x_grid,
        y_grid=y_grid,
        ux_grid=raw_grid["DX"],
        uy_grid=raw_grid["DY"],
        strip_side=cfg.PREPROCESS_FIXED_END_SIDE,
        strip_width_ratio=ratio,
        y_margin_ratio=cfg.PREPROCESS_STRIP_Y_MARGIN_RATIO,
    )
    code_grid = apply_fixed_point_reference_to_grid(
        x_grid=x_grid,
        y_grid=y_grid,
        ux_grid=np.asarray(rigid_only["UX_grid"], dtype=float),
        uy_grid=np.asarray(rigid_only["UY_grid"], dtype=float),
        point=cfg.PREPROCESS_FIXED_POINT,
    )

    ux_code = np.asarray(code_grid["UX_grid"], dtype=float)
    uy_code = np.asarray(code_grid["UY_grid"], dtype=float)
    ux_code_demean = _demean(ux_code)
    uy_code_demean = _demean(uy_code)
    ux_diff_demean = ux_code_demean - ux_remove_demean
    uy_diff_demean = uy_code_demean - uy_remove_demean

    ux_corr = _corrcoef(ux_code, ux_remove)
    uy_corr = _corrcoef(uy_code, uy_remove)
    ux_rms = float(np.sqrt(np.nanmean(ux_diff_demean ** 2)))
    uy_rms = float(np.sqrt(np.nanmean(uy_diff_demean ** 2)))
    score = ux_rms + uy_rms
    summary_lines.append(
        f"{ratio:.2f},{rigid_only['metadata']['strip_point_count']},"
        f"{rigid_only['metadata']['strip_residual_rms']:.12e},{ux_corr:.12f},{uy_corr:.12f},{ux_rms:.12e},{uy_rms:.12e},{score:.12e}"
    )

    if best is None or score < best["score"]:
        best = {
            "ratio": ratio,
            "ux_corr": ux_corr,
            "uy_corr": uy_corr,
            "ux_rms": ux_rms,
            "uy_rms": uy_rms,
            "score": score,
            "ux_diff_demean": ux_diff_demean,
            "uy_diff_demean": uy_diff_demean,
            "ux_code_demean": ux_code_demean,
            "uy_code_demean": uy_code_demean,
        }

if best is None:
    raise RuntimeError("No ratio comparison results were produced.")

summary_path = analysis_dir / f"{cfg.get_case_prefix()}_grid_remove_ratio_scan.txt"
summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

ratio_tag = f"ratio_{best['ratio']:.2f}".replace(".", "p")
plot_grid_fields(
    output_dir=analysis_dir,
    case_name=f"{cfg.get_case_prefix()}_grid_remove_best_{ratio_tag}_code_demeaned",
    x_grid=x_grid,
    y_grid=y_grid,
    fields=[
        ("ux", best["ux_code_demean"], f"Best ratio code $u_x$ (demeaned)"),
        ("uy", best["uy_code_demean"], f"Best ratio code $u_y$ (demeaned)"),
    ],
    cmap=cfg.PREPROCESS_CMAP,
)
plot_grid_fields(
    output_dir=analysis_dir,
    case_name=f"{cfg.get_case_prefix()}_grid_remove_best_{ratio_tag}_demeaned_diff",
    x_grid=x_grid,
    y_grid=y_grid,
    fields=[
        ("ux", best["ux_diff_demean"], f"Best ratio demeaned diff $u_x$"),
        ("uy", best["uy_diff_demean"], f"Best ratio demeaned diff $u_y$"),
    ],
    cmap="coolwarm",
)

print(f"saved summary: {summary_path}")
print(f"best ratio: {best['ratio']:.2f}")
print(f"ux corr: {best['ux_corr']:.6f}")
print(f"uy corr: {best['uy_corr']:.6f}")
print(f"ux demeaned rms: {best['ux_rms']:.6e}")
print(f"uy demeaned rms: {best['uy_rms']:.6e}")
