from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config as cfg
from fgm_asm import run_real_data_inverse_lcurve


run_real_data_inverse_lcurve(cfg)
