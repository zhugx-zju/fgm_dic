from pathlib import Path
import pickle
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt

import config as cfg
from fgm_asm.visualization import (
    plot_gradient_field,
    plot_iteration_history,
    plot_reconstruction_comparison,
    plot_reconstruction_results,
)

results_dir = cfg.get_case_results_dir()
inverse_results_path = results_dir / "inverse_results.pkl"

if not inverse_results_path.exists():
    raise FileNotFoundError(
        f"Inverse results not found: {inverse_results_path}\n"
        "Run script/inverse_l_curve.py first."
    )

with open(inverse_results_path, "rb") as file:
    inverse_results = pickle.load(file)

mesh_info = inverse_results["mesh_info"]
results = inverse_results["results"]
scan_optimal = inverse_results.get("scan_optimal")

print("=" * 70)
print("Inverse Results Plotter")
print("=" * 70)
print(f"Loaded inverse results from {inverse_results_path}")
print(f"Final cost: {results['final_cost']:.6e}")
print(f"Iterations: {results['n_iterations']}")

plot_reconstruction_results(
    mesh_info=mesh_info,
    E_true=None,
    E_reconstructed=inverse_results["E_reconstructed"],
    errors=None,
    noise_level=0.0,
    save_path=results_dir,
    filename_stem="reconstruction_results",
)
plot_iteration_history(
    results,
    save_path=results_dir,
    noise_level=0.0,
    filename_stem="iteration_history",
)
plot_gradient_field(
    mesh_info,
    results,
    noise_level=0.0,
    save_path=results_dir,
    filename_stem="gradient_field",
)

if scan_optimal is not None:
    plot_reconstruction_comparison(
        mesh_info=mesh_info,
        E_true=None,
        scan_E_reconstructed=scan_optimal["E_reconstructed"],
        scan_errors=None,
        rerun_E_reconstructed=inverse_results["E_reconstructed"],
        rerun_errors=None,
        noise_level=0.0,
        save_path=results_dir,
    )

plt.show()
