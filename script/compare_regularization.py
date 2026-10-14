"""Compare two saved inverse fields with matched color scales.

Example:
    python -m script.compare_regularization OLD/inverse_results.pkl NEW/inverse_results.pkl
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from fgm_asm.results_io import load_inverse_data
from fgm_asm.visualization import reshape_nodal_values_for_plot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path, help="Baseline inverse_results.pkl")
    parser.add_argument("candidate", type=Path, help="Candidate inverse_results.pkl")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    old_forward, old = load_inverse_data(args.baseline)
    new_forward, new = load_inverse_data(args.candidate)
    if args.baseline.parent.parent.resolve() != args.candidate.parent.parent.resolve():
        raise ValueError("Both inverse runs must use the same saved forward case")
    old_mesh = old_forward["mesh_info"]
    new_mesh = new_forward["mesh_info"]
    if not np.array_equal(old_mesh.coord, new_mesh.coord):
        raise ValueError("Both results must use the same mesh coordinates")
    if not np.array_equal(old["E_true"], new["E_true"]):
        raise ValueError("Both results must use the same synthetic truth field")
    if old["noise_level"] != new["noise_level"]:
        raise ValueError("Both results must use the same noise level")
    if old["noise_level"] != 0:
        if "U_measured" not in old or "U_measured" not in new:
            raise ValueError("Noisy comparisons require saved measured displacements in both runs")
        if not np.array_equal(old["U_measured"], new["U_measured"]):
            raise ValueError("Both noisy runs must use identical measured displacements")

    truth = np.asarray(old["E_true"])
    methods = ("Gradient penalty", "Affine-detrended penalty")
    runs = (old, new)
    estimates = [reshape_nodal_values_for_plot(old_mesh, run["E_reconstructed"])
                 for run in runs]
    errors = [100 * np.abs(estimate - truth) / truth for estimate in estimates]
    modulus_min = min(float(truth.min()), *(float(a.min()) for a in estimates))
    modulus_max = max(float(truth.max()), *(float(a.max()) for a in estimates))
    error_max = max(float(a.max()) for a in errors)

    fig, axes = plt.subplots(2, 2, figsize=(12, 5.7), constrained_layout=True)
    for column, (name, estimate, error, run) in enumerate(zip(methods, estimates, errors, runs)):
        modulus_artist = axes[0, column].pcolormesh(
            old_mesh.plot_x, old_mesh.plot_y, estimate, shading="auto",
            cmap="viridis", vmin=modulus_min, vmax=modulus_max,
        )
        error_artist = axes[1, column].pcolormesh(
            old_mesh.plot_x, old_mesh.plot_y, error, shading="auto",
            cmap="magma", vmin=0, vmax=error_max,
        )
        axes[0, column].set_title(name)
        axes[1, column].set_title(f"Absolute relative error, MAE {run['errors']['mae']:.4g}%")
        for row in range(2):
            axes[row, column].set_aspect("equal")
            axes[row, column].set_xlabel("x (mm)")
            axes[row, column].set_ylabel("y (mm)")

    fig.colorbar(modulus_artist, ax=axes[0, :], label="E (MPa)", shrink=0.82)
    fig.colorbar(error_artist, ax=axes[1, :], label="Absolute relative error (%)", shrink=0.82)
    output_dir = args.output_dir or args.candidate.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        output_path = output_dir / f"regularization_field_comparison.{extension}"
        fig.savefig(output_path, dpi=300)
        print(output_path)
    plt.close(fig)


if __name__ == "__main__":
    main()
