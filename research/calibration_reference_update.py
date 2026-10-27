"""Re-reference an unchanged saved inverse after an external calibration update.

No optimization is executed. New preparation, observed arrays, FE geometry,
load, objective and solver settings must match the original identification.
Only external calibration and output identity may differ. Original trial files
remain verbatim copies with their original provenance.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from lib.dic_case_workflow import load_prepared, _stage_summary, _regional_comparison, plot_saved_stage
from lib.dic_case_reporting import publish_stage
from lib.experiment_data import file_sha256
from lib.results_io import save_run


def re_reference(source: Path, prepared: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError('Choose a new inverse-reference output; existing results are protected.')
    old_prepared = source.parent/'01_prepared'
    old_cfg, old_mesh, old_cases, old_meta, _ = load_prepared(old_prepared)
    cfg, mesh, cases, metadata, provenance = load_prepared(prepared)
    allowed = {'RUN_NAME', 'CASE_ID', 'CASE_OUTPUT', 'CALIBRATION_CSV', 'CALIBRATION_METHOD'}
    old_values, new_values = old_meta['configuration'], metadata['configuration']
    changed = {key for key in old_values.keys() | new_values.keys() if old_values.get(key) != new_values.get(key)}
    if not changed <= allowed:
        raise ValueError(f'Identification configuration changed: {sorted(changed-allowed)}')
    with np.load(old_prepared/'fields.npz') as a, np.load(prepared/'fields.npz') as b:
        if set(a.files) != set(b.files):
            raise ValueError('Prepared array names changed.')
        for key in a.files:
            np.testing.assert_array_equal(a[key], b[key], err_msg=f'Prepared field changed: {key}')
    np.testing.assert_array_equal(old_mesh.coord, mesh.coord)
    for a, b in zip(old_cases, cases, strict=True):
        if a.name != b.name or a.bc_info['tensile_end_force'] != b.bc_info['tensile_end_force']:
            raise ValueError('Observation name or applied load changed.')
        for key in ('data', 'indices', 'indptr'):
            np.testing.assert_array_equal(getattr(a.observation.H, key), getattr(b.observation.H, key))
        np.testing.assert_array_equal(a.observation.weight_vector, b.observation.weight_vector)
    source_meta = json.loads((source/'run_metadata.json').read_text(encoding='utf-8'))
    if source_meta['status'] not in ('computed', 'inverse_not_converged'):
        raise ValueError('Source inverse is not complete.')
    if (source_meta['prepared_metadata_sha256'] != file_sha256(old_prepared/'run_metadata.json') or
            source_meta['prepared_fields_sha256'] != file_sha256(old_prepared/'fields.npz')):
        raise ValueError('Source inverse does not match its original frozen preparation.')
    if source_meta.get('initialization', {}).get('uses_calibration') is not False:
        raise ValueError('Reuse requires an explicitly calibration-independent initialization.')
    summary = {**source_meta, **_stage_summary(metadata, provenance, 'dic_case_inverse')}
    summary['status'] = source_meta['status']
    for key in ('configuration', 'environment', 'source_sha256'):
        summary.pop(key, None)
    summary['inverse_reuse'] = {
        'operation': 'external_calibration_reference_update_only; no optimization or gamma selection rerun',
        'source_inverse': str(source.resolve()),
        'source_metadata_sha256': file_sha256(source/'run_metadata.json'),
        'source_prepared_metadata_sha256': file_sha256(old_prepared/'run_metadata.json'),
        'changed_configuration_keys': sorted(changed),
        'all_prepared_arrays_exactly_equal': True, 'observation_operator_exactly_equal': True,
        'weights_and_applied_load_exactly_equal': True,
        'source_solver_environment': source_meta['environment'],
        'source_numerical_source_sha256': source_meta['source_sha256'],
        'trial_provenance_note': 'l_curve trials copied verbatim; their metadata describes the original optimization.',
    }
    with np.load(source/'fields.npz') as saved:
        save_run(output, cfg, summary, {k: saved[k].copy() for k in saved.files})
    shutil.copytree(source/'l_curve', output/'l_curve')
    shutil.copy2(source/'selection.csv', output/'selection.csv')
    shutil.copy2(source/'comparison.csv', output/'comparison.csv')
    calibration = pd.DataFrame(metadata['calibration'])
    regions = []
    field_checks = []
    for nu in cfg.NU_VALUES:
        old = source/f'nu_{nu:.2f}'
        with np.load(old/'fields.npz') as saved:
            fields = {k: saved[k].copy() for k in saved.files}
        solver = next(r for r in source_meta['solvers'] if r['nu'] == nu)
        save_run(output/f'nu_{nu:.2f}', cfg, dict(summary, nu=nu, solver=solver,
                 inverse_parameter_count=mesh.n_nod), fields)
        with np.load(output/f'nu_{nu:.2f}'/'fields.npz') as replay:
            for key, value in fields.items():
                np.testing.assert_array_equal(value, replay[key])
        region = _regional_comparison(mesh, fields['E_inverse'], calibration, cfg.REGION_BOUNDARIES_X)
        region.insert(0, 'nu', nu)
        region.to_csv(output/f'nu_{nu:.2f}'/'regional_modulus_comparison.csv', index=False)
        regions.append(region)
        field_checks.append({'nu': nu, 'all_inverse_arrays_exactly_equal': True,
                             'source_fields_sha256': file_sha256(old/'fields.npz'),
                             'updated_fields_sha256': file_sha256(output/f'nu_{nu:.2f}'/'fields.npz')})
    pd.concat(regions, ignore_index=True).to_csv(output/'regional_comparison.csv', index=False)
    assert (source/'selection.csv').read_bytes() == (output/'selection.csv').read_bytes()
    assert (source/'comparison.csv').read_bytes() == (output/'comparison.csv').read_bytes()
    (output/'inverse_reuse_checks.json').write_text(json.dumps(field_checks, indent=2), encoding='utf-8')
    plot_saved_stage(output, prepared)
    publish_stage(output.parent, '03_inverse', output, summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    summary = re_reference(args.source, args.prepared, args.output)
    print('Inverse reused without optimization:', summary['selected_gamma'], args.output.resolve())


if __name__ == '__main__':
    main()
