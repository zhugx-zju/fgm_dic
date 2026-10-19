"""Audit DIC platform scatter, drift and correlations before physical registration."""

import argparse
from datetime import datetime
import importlib
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import REPO_ROOT
from lib.dic_noise import analyze_dic_series
from lib.experiment_data import load_dic_frame_series
from lib.research_reporting import save_run


def _plot_diagnostics(output: Path, name: str, arrays: dict, std_limits: np.ndarray) -> None:
    stages = arrays['stages']
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), constrained_layout=True)
    for component, label in enumerate(('DX', 'DY')):
        axes[0].plot(stages, arrays['translations'][:, component], '.-', label=label)
    axes[0].set(xlabel='DIC stage', ylabel='Translation about platform mean\n(export units)')
    axes[0].legend(frameon=False)
    axes[1].plot(stages, arrays['frame_rms_to_mean'], '.-', label='Rigid removed')
    axes[1].plot(stages, arrays['detrended_frame_rms'], '.-', label='Rigid + linear trend removed')
    selected = stages[np.argmin(arrays['frame_rms_to_mean'])]
    axes[1].axvline(selected, color='0.5', lw=.8, linestyle='--', label=f'Closest stage {selected}')
    axes[1].set(xlabel='DIC stage', ylabel='Equal-point RMS (export units)')
    axes[1].legend(frameon=False)
    fig.savefig(output / f'{name}_temporal.png', dpi=300)
    fig.savefig(output / f'{name}_temporal.svg')
    plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(9, 7), constrained_layout=True)
    xy = arrays['coordinates']
    for component, label in enumerate(('DX', 'DY')):
        for column, (field, title) in enumerate((('raw_sd', 'Raw temporal SD'),
                                                 ('cleaned_sd', 'Rigid removed SD'),
                                                 ('detrended_sd', 'Detrended SD'))):
            ax = axes[component, column]
            artist = ax.scatter(*xy.T, c=arrays[field][:, component], s=2, cmap='viridis',
                                vmin=0, vmax=std_limits[component], rasterized=True)
            ax.set(title=f'{label}: {title}', xlabel='X_ref (export units)',
                   ylabel='Y_ref (export units)', aspect='equal')
            fig.colorbar(artist, ax=ax, label='Export units')
    fig.savefig(output / f'{name}_scatter.png', dpi=300)
    fig.savefig(output / f'{name}_scatter.svg')
    plt.close(fig)


def run_noise_audit(config, output: Path, frame_rate_hz: float) -> dict:
    output = Path(output)
    if output.exists():
        raise FileExistsError(f'Choose a new output directory: {output}')
    records, fields, saved = {}, {}, {}
    for name, spec in config.PLATFORMS.items():
        print(f'Reading {name} frames {spec["stages"]}...', flush=True)
        series = load_dic_frame_series(spec['folder'], *spec['stages'],
                                       coordinate_state=config.DIC_COORDINATE_STATE)
        record, arrays = analyze_dic_series(series, frame_rate_hz)
        records[name], fields[name] = record, arrays
        saved.update({f'{name}_{key}': value for key, value in arrays.items()})
        print(f'{name}: residual SD medians {record["detrended_temporal_sd"]["median"]}; '
              f'closest stage {record["representative_stage_after_rigid_removal"]}', flush=True)
    summary = {'workflow': 'dic_noise_audit', 'status': 'diagnostic_export_units',
               'frame_rate_hz': frame_rate_hz, 'platforms': records,
               'threshold_status': 'No physical-unit acceptance threshold established; '
                                   'physical registration and uncertainty budget remain required.'}
    save_run(output, config, summary, saved)
    std_limits = np.max([np.max(a['raw_sd'], axis=0) for a in fields.values()], axis=0)
    std_limits = np.maximum(std_limits, np.finfo(float).eps)
    for name, arrays in fields.items():
        _plot_diagnostics(output, name, arrays, std_limits)
        pd.DataFrame({'stage': arrays['stages'], 'relative_time_s': arrays['time_s'],
                      'translation_DX': arrays['translations'][:, 0],
                      'translation_DY': arrays['translations'][:, 1],
                      'rotation_rad': arrays['rotations'],
                      'rigid_removed_rms': arrays['frame_rms_to_mean'],
                      'detrended_rms': arrays['detrended_frame_rms']}).to_csv(
                          output / f'{name}_frame_diagnostics.csv', index=False)
        point_table = {'grid_row': arrays['grid_ids'][:, 0],
                       'grid_col': arrays['grid_ids'][:, 1],
                       'X_ref': arrays['coordinates'][:, 0],
                       'Y_ref': arrays['coordinates'][:, 1]}
        for key in ('raw_sd', 'cleaned_sd', 'detrended_sd', 'robust_mad', 'slope', 'drift'):
            for component, label in enumerate(('DX', 'DY')):
                point_table[f'{key}_{label}'] = arrays[key][:, component]
        pd.DataFrame(point_table).to_csv(output / f'{name}_point_diagnostics.csv', index=False)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='experiment_config')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--frame-rate', type=float, default=5., help='Run 3 acquisition rate (Hz)')
    args = parser.parse_args()
    output = args.output or REPO_ROOT / 'results' / (
        'dic_noise_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
    summary = run_noise_audit(importlib.import_module(args.config), output, args.frame_rate)
    print(f'{summary["status"]}: {output.resolve()}', flush=True)


if __name__ == '__main__':
    main()
