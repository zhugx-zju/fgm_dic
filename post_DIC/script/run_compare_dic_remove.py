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

rigid_only = remove_rigid_body_motion_from_grid(
    x_grid=raw_grid["X_ref"],
    y_grid=raw_grid["Y_ref"],
    ux_grid=raw_grid["DX"],
    uy_grid=raw_grid["DY"],
    strip_side=cfg.PREPROCESS_FIXED_END_SIDE,
    strip_width_ratio=cfg.PREPROCESS_STRIP_WIDTH_RATIO,
    y_margin_ratio=cfg.PREPROCESS_STRIP_Y_MARGIN_RATIO,
)
code_grid = apply_fixed_point_reference_to_grid(
    x_grid=raw_grid["X_ref"],
    y_grid=raw_grid["Y_ref"],
    ux_grid=np.asarray(rigid_only["UX_grid"], dtype=float),
    uy_grid=np.asarray(rigid_only["UY_grid"], dtype=float),
    point=cfg.PREPROCESS_FIXED_POINT,
)

x_grid = np.asarray(raw_grid["X_ref"], dtype=float)
y_grid = np.asarray(raw_grid["Y_ref"], dtype=float)
ux_code = np.asarray(code_grid["UX_grid"], dtype=float)
uy_code = np.asarray(code_grid["UY_grid"], dtype=float)
ux_remove = np.asarray(remove_grid["DX"], dtype=float)
uy_remove = np.asarray(remove_grid["DY"], dtype=float)

ux_code_demean = _demean(ux_code)
uy_code_demean = _demean(uy_code)
ux_remove_demean = _demean(ux_remove)
uy_remove_demean = _demean(uy_remove)
ux_diff_demean = ux_code_demean - ux_remove_demean
uy_diff_demean = uy_code_demean - uy_remove_demean

metrics_lines = [
    f"raw_csv={raw_csv}",
    f"remove_csv={remove_csv}",
    f"fixed_end_side={cfg.PREPROCESS_FIXED_END_SIDE}",
    f"strip_width_ratio={cfg.PREPROCESS_STRIP_WIDTH_RATIO}",
    f"strip_y_margin_ratio={cfg.PREPROCESS_STRIP_Y_MARGIN_RATIO}",
    f"fixed_point={cfg.PREPROCESS_FIXED_POINT}",
    f"strip_point_count={rigid_only['metadata']['strip_point_count']}",
    f"strip_residual_rms={rigid_only['metadata']['strip_residual_rms']:.12e}",
    f"strip_residual_max={rigid_only['metadata']['strip_residual_max']:.12e}",
    f"ux_corr={_corrcoef(ux_code, ux_remove):.12f}",
    f"uy_corr={_corrcoef(uy_code, uy_remove):.12f}",
    f"ux_demeaned_rms={np.sqrt(np.nanmean(ux_diff_demean ** 2)):.12e}",
    f"uy_demeaned_rms={np.sqrt(np.nanmean(uy_diff_demean ** 2)):.12e}",
]

compare_prefix = f"{cfg.get_case_prefix()}_grid_remove_compare"
metrics_path = analysis_dir / f"{compare_prefix}_metrics.txt"
metrics_path.write_text("\n".join(metrics_lines) + "\n", encoding="utf-8")

plot_grid_fields(
    output_dir=analysis_dir,
    case_name=f"{compare_prefix}_software_demeaned",
    x_grid=x_grid,
    y_grid=y_grid,
    fields=[
        ("ux", ux_remove_demean, "Software removed $u_x$ (demeaned)"),
        ("uy", uy_remove_demean, "Software removed $u_y$ (demeaned)"),
    ],
    cmap=cfg.PREPROCESS_CMAP,
)
plot_grid_fields(
    output_dir=analysis_dir,
    case_name=f"{compare_prefix}_code_demeaned",
    x_grid=x_grid,
    y_grid=y_grid,
    fields=[
        ("ux", ux_code_demean, "Code removed+fixed-point $u_x$ (demeaned)"),
        ("uy", uy_code_demean, "Code removed+fixed-point $u_y$ (demeaned)"),
    ],
    cmap=cfg.PREPROCESS_CMAP,
)
plot_grid_fields(
    output_dir=analysis_dir,
    case_name=f"{compare_prefix}_demeaned_diff",
    x_grid=x_grid,
    y_grid=y_grid,
    fields=[
        ("ux", ux_diff_demean, "Demeaned difference $u_x$"),
        ("uy", uy_diff_demean, "Demeaned difference $u_y$"),
    ],
    cmap="coolwarm",
)

print(f"saved metrics: {metrics_path}")
print(f"ux corr: {_corrcoef(ux_code, ux_remove):.6f}")
print(f"uy corr: {_corrcoef(uy_code, uy_remove):.6f}")
print(f"ux demeaned rms: {np.sqrt(np.nanmean(ux_diff_demean ** 2)):.6e}")
print(f"uy demeaned rms: {np.sqrt(np.nanmean(uy_diff_demean ** 2)):.6e}")
