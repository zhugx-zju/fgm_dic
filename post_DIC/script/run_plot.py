from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config as cfg

from lib.config_utils import ensure_output_dirs, get_project_paths
from lib.dic_preprocessing import plot_preprocessed_displacement_fields, preprocess_structured_dic


paths = get_project_paths()
ensure_output_dirs(paths)
output_dir = paths.data
jobs = [
    ("grid", cfg.get_grid_dic_csv_path(), cfg.get_grid_case_name()),
    ("pixel", cfg.get_pixel_dic_csv_path(), cfg.get_pixel_case_name()),
]

for label, csv_path, prefix in jobs:
    arrays = preprocess_structured_dic(
        csv_path=csv_path,
        fixed_point=cfg.PREPROCESS_FIXED_POINT,
        fixed_end_side=cfg.PREPROCESS_FIXED_END_SIDE,
        strip_width_ratio=cfg.PREPROCESS_STRIP_WIDTH_RATIO,
        strip_y_margin_ratio=cfg.PREPROCESS_STRIP_Y_MARGIN_RATIO,
        displacement_signs=(cfg.PREPROCESS_UX_SIGN, cfg.PREPROCESS_UY_SIGN),
        orientation_operations=cfg.PREPROCESS_ORIENTATION_OPERATIONS,
    )
    plot_preprocessed_displacement_fields(
        output_dir=output_dir,
        case_name=prefix,
        arrays=arrays,
        cmap=cfg.PREPROCESS_CMAP,
    )
    print(f"[{label}] plotted preprocessed displacement fields from {csv_path}")
