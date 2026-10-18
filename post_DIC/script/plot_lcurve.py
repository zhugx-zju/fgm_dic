from pathlib import Path
import pickle
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt

import config as cfg
from fgm_asm import plot_lcurve_results

results_dir = cfg.get_case_results_dir()
lcurve_path = results_dir / "lcurve_analysis.pkl"

if not lcurve_path.exists():
    raise FileNotFoundError(
        f"L-curve analysis not found: {lcurve_path}\n"
        "Run script/inverse_l_curve.py first."
    )

with open(lcurve_path, "rb") as file:
    loaded = pickle.load(file)
lcurve_results = loaded["lcurve_results"] if isinstance(loaded, dict) and "lcurve_results" in loaded else loaded

print("=" * 70)
print("L-curve Results Plotter")
print("=" * 70)
print(f"Loaded L-curve analysis from {lcurve_path}")

plot_lcurve_results(lcurve_results, save_path=results_dir)
plt.show()
