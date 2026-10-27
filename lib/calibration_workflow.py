"""Configured material calibration using the existing regression and MILP code."""
from dataclasses import asdict
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import linregress

from .experiment_data import file_sha256
from .results_io import save_run
from .tensile_calibration import Settings, Specimen, candidate_windows, evaluate_fit, read_blocks


def load_specimens(manifest: Path, baseline_points: int) -> list[Specimen]:
    """Portable manifest; each curve uses its measured outer section and gauge."""
    table = pd.read_csv(manifest)
    columns = {'specimen_id', 'infill_percent', 'width_mm', 'thickness_mm', 'gauge_mm', 'file', 'block'}
    if not columns.issubset(table) or table.empty or table.specimen_id.duplicated().any():
        raise ValueError(f'{manifest}: require unique specimens and columns {sorted(columns)}')
    if not isinstance(baseline_points, int) or baseline_points < 1:
        raise ValueError('CALIBRATION_BASELINE_POINTS must be a positive integer')
    specimens = []
    for row in table.itertuples():
        if not re.fullmatch(r'[A-Za-z0-9_-]+', str(row.specimen_id)):
            raise ValueError('specimen_id must contain only letters, digits, hyphens or underscores')
        if not np.isfinite(row.infill_percent) or int(row.infill_percent) != row.infill_percent:
            raise ValueError('infill_percent must be a finite integer')
        path = (manifest.parent / row.file).resolve()
        if path.suffix.lower() == '.xlsb':
            blocks = read_blocks(path)
            block = int(row.block)
            if block != row.block or not 1 <= block <= len(blocks):
                raise ValueError(f'{path}: block must explicitly identify a 1-based data block')
            values = blocks[block-1]
        else:
            frame = pd.read_csv(path)
            values = frame[['time_s', 'force_N', 'displacement_mm', 'extension_mm']].to_numpy(float)
            if row.block != 1:
                raise ValueError(f'{path}: normalized CSV has exactly one block')
            block = 1
        dimensions = np.array([row.width_mm, row.thickness_mm, row.gauge_mm], dtype=float)
        if (values.ndim != 2 or len(values) <= baseline_points or not np.isfinite(values).all()
                or not np.all(np.diff(values[:, 0]) > 0) or not np.all(np.isfinite(dimensions) & (dimensions > 0))):
            raise ValueError(f'{path}: require finite channels, increasing time and positive dimensions')
        baseline = float(np.median(values[:baseline_points, 1]))
        peak = int(np.argmax(values[:, 1]-baseline))
        if peak <= baseline_points:
            raise ValueError(f'{path}: no usable positive loading branch')
        specimens.append(Specimen(str(row.specimen_id), int(row.infill_percent), *dimensions,
                                  str(path), block, *values.T, baseline, peak))
    return specimens


def joint_select(specimens, pools, deviation=.03, time_limit=40.):
    """Original MILP with groups derived from inputs, rather than seven fixed groups."""
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import coo_matrix

    if not 0 < deviation < 1 or not np.isfinite(time_limit) or time_limit <= 0:
        raise ValueError('Require 0 < deviation < 1 and a positive time limit')
    if any(not pools[s.specimen_id] for s in specimens):
        raise ValueError('A specimen has no acceptable calibration window')
    groups = sorted({s.infill for s in specimens})
    candidates = [(s, fit) for s in specimens for fit in pools[s.specimen_id]]
    count = len(candidates)
    means = {infill: count+i for i, infill in enumerate(groups)}
    entries = {s.specimen_id: [] for s in specimens}
    for index, (specimen, fit) in enumerate(candidates):
        entries[specimen.specimen_id].append((index, fit.modulus_mpa))
    rows, cols, values, lower, upper = [], [], [], [], []

    def add(items, lo, hi):
        for index, value in items:
            rows.append(len(lower)); cols.append(index); values.append(value)
        lower.append(lo); upper.append(hi)

    for specimen in specimens:
        selected = entries[specimen.specimen_id]
        add([(i, 1.) for i, _ in selected], 1., 1.)
        add(selected+[(means[specimen.infill], -(1-deviation))], 0., np.inf)
        add(selected+[(means[specimen.infill], -(1+deviation))], -np.inf, 0.)
    for infill in groups:
        group = [s for s in specimens if s.infill == infill]
        add([(i, modulus/len(group)) for s in group for i, modulus in entries[s.specimen_id]]
            + [(means[infill], -1.)], 0., 0.)
    for first, second in zip(groups[:-1], groups[1:]):
        add([(means[first], -1.), (means[second], 1.)], 0., np.inf)
    matrix = coo_matrix((values, (rows, cols)), shape=(len(lower), count+len(groups))).tocsc()
    result = milp(np.r_[[1e6*(1-fit.r2) for _, fit in candidates], np.zeros(len(groups))],
                  integrality=np.r_[np.ones(count), np.zeros(len(groups))],
                  bounds=Bounds(np.zeros(count+len(groups)), np.r_[np.ones(count), np.full(len(groups), np.inf)]),
                  constraints=LinearConstraint(matrix, lower, upper),
                  options={'time_limit': time_limit, 'mip_rel_gap': 1e-6})
    audit = dict(status=int(result.status), message=result.message,
                 mip_gap=float(result.mip_gap) if result.x is not None else None)
    if result.x is None:
        raise RuntimeError(f'No feasible joint calibration: {audit}')
    chosen = {}
    for specimen in specimens:
        indices = [i for i, _ in entries[specimen.specimen_id] if result.x[i] > .5]
        if len(indices) != 1:
            raise ValueError('MILP did not select exactly one window per specimen')
        chosen[specimen.specimen_id] = candidates[indices[0]][1]
    return chosen, audit


def calibrate_material(cfg):
    output = Path(cfg.CALIBRATION_OUTPUT)
    if output.exists():
        raise FileExistsError(f'Choose a new RUN_NAME: {output}')
    specimens = load_specimens(Path(cfg.CALIBRATION_MANIFEST), cfg.CALIBRATION_BASELINE_POINTS)
    settings = Settings(**cfg.CALIBRATION_SETTINGS)
    pools = {}
    for specimen in specimens:
        pools[specimen.specimen_id] = [fit for lo, hi in candidate_windows([specimen], settings)
            if (fit := evaluate_fit(specimen, lo, hi, settings)) is not None and fit.accepted]
        if not pools[specimen.specimen_id]:
            raise ValueError(f'{specimen.specimen_id}: no window meets CALIBRATION_SETTINGS')
    independent = {s.specimen_id: max(pools[s.specimen_id], key=lambda f: (f.r2, -f.half_slope_difference))
                   for s in specimens}
    solver = None
    method = cfg.CALIBRATION_METHOD
    if method == 'individual_window_joint_3pct_monotone':
        if cfg.CALIBRATION_MAX_DEVIATION != .03:
            raise ValueError('3pct method name requires CALIBRATION_MAX_DEVIATION=.03')
        selected, solver = joint_select(specimens, pools, cfg.CALIBRATION_MAX_DEVIATION,
                                        cfg.CALIBRATION_TIME_LIMIT)
    elif method == 'individual_highest_R2':
        selected = independent
    else:
        raise ValueError('Select joint_3pct_monotone or individual_highest_R2 calibration method')

    def tables(fits, label):
        rows = pd.DataFrame([dict(infill_percent=s.infill, **asdict(fits[s.specimen_id])) for s in specimens])
        groups = []
        for infill, frame in rows.groupby('infill_percent', sort=True):
            values = frame.modulus_mpa.to_numpy()
            groups.append(dict(infill_percent=int(infill), batch=cfg.CALIBRATION_BATCH,
                n=len(values), mean_E_MPa=float(values.mean()), sd_E_MPa=float(values.std(ddof=1)) if len(values)>1 else 0.,
                fit_method=label, max_absolute_specimen_deviation_percent=float(100*np.max(abs(values/values.mean()-1)))))
        return rows, pd.DataFrame(groups)

    rows, groups = tables(selected, method)
    baseline_rows, baseline_groups = tables(independent, 'individual_highest_R2')
    if solver:
        if (groups.max_absolute_specimen_deviation_percent.max() > 3+1e-6
                or np.any(np.diff(groups.mean_E_MPa) < -1e-6)):
            raise ValueError('Selected MILP slopes fail the explicit group constraints')
    checks = []
    for specimen in specimens:
        fit = selected[specimen.specimen_id]
        stress = specimen.stress_mpa[:specimen.peak_index+1]
        mask = (stress >= fit.lower_stress_mpa) & (stress <= fit.upper_stress_mpa)
        regression = linregress(specimen.extension_mm[:specimen.peak_index+1][mask],
                               specimen.force_n[:specimen.peak_index+1][mask])
        independent_E = regression.slope*specimen.gauge_mm/specimen.area_mm2
        np.testing.assert_allclose(independent_E, fit.modulus_mpa, rtol=1e-10)
        checks.append(dict(specimen_id=specimen.specimen_id, independent_E_MPa=independent_E))
    sources = {s.source: file_sha256(Path(s.source)) for s in specimens}
    sources[str(cfg.CALIBRATION_MANIFEST)] = file_sha256(Path(cfg.CALIBRATION_MANIFEST))
    arrays = {}
    for specimen in specimens:
        arrays[f'{specimen.specimen_id}_strain'] = specimen.raw_strain[:specimen.peak_index+1]
        arrays[f'{specimen.specimen_id}_stress'] = specimen.stress_mpa[:specimen.peak_index+1]
    save_run(output, cfg, dict(workflow='material_calibration', status='computed', solver=solver,
        source_hashes=sources, independent_regression_checks=checks,
        imposed_constraints=bool(solver),
        interpretation='Joint consistency and monotonicity are selection priors, not independent validation.'), arrays)
    for name, table in [('fit_results', rows), ('group_summary', groups),
                        ('unconstrained_fits', baseline_rows), ('unconstrained_group_summary', baseline_groups)]:
        table.to_csv(output/f'{name}.csv', index=False)
    plot_calibration_results(output)
    return groups


def plot_calibration_results(output):
    """Replot frozen curves and selected slopes, without raw input or fitting."""
    output = Path(output)
    rows = pd.read_csv(output/'fit_results.csv')
    with np.load(output/'fields.npz') as saved:
        for fit in rows.itertuples():
            stress, strain = saved[f'{fit.specimen_id}_stress'], saved[f'{fit.specimen_id}_strain']
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(strain, stress, color='.6', lw=1)
            mask = (stress >= fit.lower_stress_mpa) & (stress <= fit.upper_stress_mpa)
            ax.plot(strain[mask], fit.modulus_mpa*strain[mask]+fit.intercept_mpa, color='#0072B2', lw=1.5)
            ax.set(xlabel='Strain', ylabel='Stress (MPa)', title=f'{fit.specimen_id}: E={fit.modulus_mpa:.2f} MPa')
            for suffix in ('.png', '.svg'):
                fig.savefig(output/f'fit_{fit.specimen_id}{suffix}', dpi=300, bbox_inches='tight')
            plt.close(fig)
