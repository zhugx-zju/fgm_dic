"""Transfer saved local tensile windows to the 20 x 3.2 mm layered specimen.

This estimates a loading range for testing, not a yield load or a guarantee of
recoverable elasticity. Original calibration windows and moduli remain fixed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import linregress

from analyze_tensile_batches import load_core


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    core = load_core(root)
    settings = core.Settings()
    mixed = root / "analysis_results/mixed_infill_20261017"
    selected = pd.read_csv(mixed / "selected_specimens.csv")
    selected = selected[selected.fit_method == "independent"]
    assert len(selected) == 34
    geometry_path = root / "1013" / core.GEOMETRY_FILE.name
    old_dims = core.read_geometry(geometry_path)
    new_dims_path = root / "1016/geometry.json"
    new_dims = json.loads(new_dims_path.read_text(encoding="utf-8"))
    meta_paths = {
        1013: root / "1013/analysis_results/modulus_20261013_165808/analysis_metadata.json",
        1016: root / "1016/analysis_results/analysis_20261017/analysis_metadata.json",
    }
    audits = {batch: {g["infill_percent"]: g for g in
                     json.loads(path.read_text(encoding="utf-8"))["groups"]}
              for batch, path in meta_paths.items()}
    hashes = {}
    def record(path: Path) -> None:
        hashes[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    for path in [geometry_path, new_dims_path, *meta_paths.values(),
                 mixed / "selected_specimens.csv", Path(__file__),
                 root / "1013/analyze_youngs_modulus.py"]:
        record(path)
    rows, extension_rows, working_rows, curves = [], [], [], []
    max_slope_error = 0.0
    for row in selected.itertuples():
        batch, level, sid = int(row.batch), int(row.infill_percent), row.specimen_id
        source = root / str(batch) / str(level) / (sid.replace("-", "_") + ".xlsb")
        record(source)
        blocks = core.read_blocks(source)
        excursions = [np.max(b[:, 1] - np.median(b[:10, 1])) if len(b) >= 10
                      else -np.inf for b in blocks]
        index = int(np.argmax(excursions))
        block = blocks[index]
        baseline = float(np.median(block[:10, 1]))
        dims = old_dims[sid] if batch == 1013 else tuple(
            float(new_dims[k]) for k in ("width_mm", "thickness_mm", "gauge_mm"))
        specimen = core.Specimen(sid, level, *dims, str(source), index + 1,
                                 block[:, 0], block[:, 1], block[:, 2], block[:, 3],
                                 baseline, int(np.argmax(block[:, 1] - baseline)))
        lower, upper = audits[batch][level]["common_stress_interval_mpa"]
        fit = core.evaluate_fit(specimen, lower, upper, settings)
        assert fit is not None and fit.accepted, (batch, sid)
        assert np.isclose(fit.modulus_mpa, row.E_MPa, rtol=1e-10), (batch, sid)
        stress = specimen.stress_mpa[:specimen.peak_index + 1]
        strain = specimen.raw_strain[:specimen.peak_index + 1]
        mask = (stress >= lower) & (stress <= upper)
        independent = linregress(strain[mask], stress[mask]).slope
        force_slope = linregress(block[:len(stress), 3][mask], block[:len(stress), 1][mask]).slope
        direct_e = force_slope * specimen.gauge_mm / specimen.area_mm2
        max_slope_error = max(max_slope_error, abs(independent / fit.modulus_mpa - 1),
                              abs(direct_e / fit.modulus_mpa - 1))
        rows.append(dict(batch=batch, infill_percent=level, specimen_id=sid,
                         E_MPa=fit.modulus_mpa, baseline_force_N=baseline,
                         original_area_mm2=specimen.area_mm2,
                         lower_stress_MPa=lower, upper_stress_MPa=upper,
                         upper_equivalent_net_force_N=upper * 64,
                         actual_sample_upper_net_force_N=(fit.force_max_n - baseline) * 64 / specimen.area_mm2,
                         R2=fit.r2, half_slope_difference=fit.half_slope_difference,
                         selected_block=index + 1))
        # Common working-stress diagnostic, kept separate from saved calibration.
        for net_load in [219.7354506515086, 270., 319.9]:
            low_fit = core.evaluate_fit(specimen, 2.141816795948816, net_load / 64, settings)
            working_rows.append(dict(batch=batch, infill_percent=level, specimen_id=sid,
                                     net_load_N=net_load,
                                     lower_stress_MPa=2.141816795948816,
                                     upper_stress_MPa=net_load / 64,
                                     evaluable=low_fit is not None,
                                     accepted=low_fit.accepted if low_fit else False,
                                     local_E_MPa=low_fit.modulus_mpa if low_fit else np.nan,
                                     change_from_saved_E_percent=100 * (low_fit.modulus_mpa / fit.modulus_mpa - 1) if low_fit else np.nan))
        # Fixed lower endpoint, extended upper endpoint: no new modulus selection.
        # This diagnoses sensitivity to the old local window, not initial elasticity.
        if level == 40:
            curves.append((sid, 100 * (strain - fit.strain_zero), stress, mask, fit))
            for high in sorted(set([*np.arange(3., 9.01, .1), upper,
                                    219.7354506515086 / 64, 270 / 64, 319.9 / 64])):
                test = core.evaluate_fit(specimen, lower, float(high), settings)
                if test is not None:
                    extension_rows.append(dict(specimen_id=sid, upper_stress_MPa=high,
                                               upper_net_force_N=high * 64,
                                               E_MPa=test.modulus_mpa, R2=test.r2,
                                               half_slope_difference=test.half_slope_difference,
                                               positive_steps=test.positive_strain_steps,
                                               accepted=test.accepted))
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "specimen_audit.csv", index=False)
    pd.DataFrame(extension_rows).to_csv(out / "fixed_lower_extension_40.csv", index=False)
    pd.DataFrame(working_rows).to_csv(out / "common_working_stress_diagnostic.csv", index=False)
    summary = frame.groupby(["infill_percent", "batch"]).agg(
        n=("E_MPa", "size"), E_MPa=("E_MPa", "mean"), sd_E_MPa=("E_MPa", "std"),
        lower_stress_MPa=("lower_stress_MPa", "min"),
        upper_stress_MPa=("upper_stress_MPa", "min"),
        upper_equivalent_net_force_N=("upper_equivalent_net_force_N", "min"),
        actual_sample_upper_min_N=("actual_sample_upper_net_force_N", "min")).reset_index()
    summary["upper_linear_model_strain_percent"] = 100 * summary.upper_stress_MPa / summary.E_MPa
    summary.to_csv(out / "group_limits.csv", index=False)
    loads = np.array([119.77617870327234, 219.7354506515086, 250., 270., 319.9, 342.6906873518106])
    lengths = np.array([35., 16., 16., 16., 16., 16., 35.])
    moduli = summary.E_MPa.to_numpy()
    equivalent_e = 150 / np.sum(lengths / moduli)
    predicted = []
    for load in loads:
        predicted.append(dict(net_force_N=load, machine_force_with_60p1_offset_N=load + 60.1,
                              stress_MPa=load / 64, strain_40_percent=100 * load / 64 / moduli[0],
                              parallel_150mm_extension_mm=load / 64 * np.sum(lengths / moduli)))
    pd.DataFrame(predicted).to_csv(out / "load_predictions.csv", index=False)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.2), layout="constrained")
    for ax, (sid, strain, stress, mask, fit) in zip(axes.flat, curves):
        visible = (strain >= -.1) & (strain <= 1.2) & (stress <= 12)
        ax.plot(strain[visible], stress[visible], color=".55", lw=1, label="Raw pre-peak")
        ax.plot(strain[mask], stress[mask], color="#0072B2", lw=2, label="Saved local fit window")
        x = np.linspace(0, .8, 100)
        ax.plot(x, fit.modulus_mpa * x / 100, color="#0072B2", ls="--", lw=.8)
        ax.axhline(270 / 64, color="#009E73", ls=":", lw=1.3, label="270 N net / 64")
        ax.axhline(319.9 / 64, color="#D55E00", ls="--", lw=1, label="319.9 N net / 64")
        ax.set(xlim=(0, 1.0), ylim=(0, 10), title=f"1013 / {sid}",
               xlabel="Strain from saved fit intercept (%)", ylabel="Nominal stress (MPa)")
    axes[0, 0].legend(fontsize=7, frameon=False)
    ax = axes.flat[-1]
    ax.bar(summary.infill_percent.astype(str), summary.upper_equivalent_net_force_N, color="#0072B2")
    ax.axhline(270, color="#009E73", ls=":", label="Provisional operating cap")
    ax.set(xlabel="Infill (%)", ylabel="Equivalent net force (N)", title="Saved local window upper endpoints")
    ax.legend(fontsize=8, frameon=False)
    fig.savefig(out / "linear_window_transfer.png", dpi=240)
    fig.savefig(out / "linear_window_transfer.svg")
    plt.close(fig)
    metadata = dict(analysis="local-window-transfer-not-yield-identification",
                    area_mm2=64., parallel_length_mm=150., zone_lengths_mm=lengths.tolist(),
                    force_offset_for_sample2_run3_N=60.1, equivalent_series_E_MPa=equivalent_e,
                    local_window_margin_fraction=.8, proposed_net_cap_N=270.,
                    exact_80_percent_transfer_cap_N=.8 * summary.upper_equivalent_net_force_N.min(),
                    max_independent_slope_relative_error=max_slope_error,
                    calibration_fits_reproduced=len(frame),
                    diagnostic_common_lower_stress_MPa=2.141816795948816,
                    selection_settings=vars(settings),
                    python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                    source_sha256=hashes)
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(summary.to_string(index=False))
    print(pd.DataFrame(predicted).to_string(index=False))
    print(json.dumps({k: v for k, v in metadata.items() if k != "source_sha256"}, indent=2))
    print(f"Output: {out}")


if __name__ == "__main__":
    main()
