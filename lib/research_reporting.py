"""Save mainline results, provenance, modulus maps and width profiles."""

import json
from pathlib import Path
import platform
import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy

from .experiment_data import file_sha256


def _json_value(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(v) for v in value]
    return value

def save_run(output, config, summary, arrays=None):
    """Write into a new case directory; never overwrite an established run."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    try:
        revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root,
                                           text=True, stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=root,
                                             text=True, stderr=subprocess.DEVNULL).strip())
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = None, None
    metadata = dict(summary)
    metadata['configuration'] = {k: v for k, v in vars(config).items() if k.isupper()}
    metadata['environment'] = {'python': sys.version, 'executable': sys.executable,
                               'numpy': np.__version__, 'scipy': scipy.__version__,
                               'platform': platform.platform(), 'git_revision': revision,
                               'git_dirty': dirty, 'command': sys.argv}
    sources = [*root.glob('*config.py'), *root.glob('fgm_asm/*.py'),
               *root.glob('lib/*.py'), *root.glob('script/*.py')]
    metadata['source_sha256'] = {str(p.relative_to(root)): file_sha256(p) for p in sources}
    config_file = getattr(config, '__file__', None)
    if config_file is not None:
        (output / 'configuration_snapshot.py').write_bytes(Path(config_file).read_bytes())
    (output / 'run_metadata.json').write_text(
        json.dumps(_json_value(metadata), ensure_ascii=False, indent=2, allow_nan=False),
        encoding='utf-8',
    )
    if arrays:
        np.savez_compressed(output / 'fields.npz', **arrays)


def plot_l_curve(output, selection, regularization_mode='gradient'):
    """Replot saved scan norms and signed log-log curvature; never solve again."""
    output = Path(output)
    data = pd.DataFrame(selection).sort_values('gamma')
    valid = (data.converged.astype(bool) & np.isfinite(data.residual_norm) &
             (data.residual_norm > 0) & np.isfinite(data.regularization_norm) &
             (data.regularization_norm > 0))
    selected = data.get('selected', pd.Series(False, index=data.index)).astype(bool)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(data.residual_norm.where(valid), data.regularization_norm.where(valid),
              'o-', color='#1764a0', markersize=4, label='Converged scan')
    failed = ~data.converged.astype(bool) & (data.residual_norm > 0) & (data.regularization_norm > 0)
    if failed.any():
        ax.loglog(data.residual_norm[failed], data.regularization_norm[failed],
                  'x', color='gray', label='Unconverged (excluded)')
    if selected.any():
        row = data[selected].iloc[0]
        ax.loglog(row.residual_norm, row.regularization_norm, '*', color='#c0392b',
                  markersize=13, label=f'Selected gamma={row.gamma:.3g}')
    regularization_label = (r'Huber-TV norm $\sqrt{R}$ ($(\mathrm{MPa}\,\mathrm{mm})^{1/2}$)'
                            if regularization_mode == 'huber_tv' else r'Tikhonov gradient norm $\eta$ (MPa)')
    ax.set(xlabel=r'Area-weighted residual norm $\rho$ (mm$^2$)',
           ylabel=regularization_label,
           title='L-curve' if selected.any() else 'L-curve: selection unresolved')
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.canvas.draw()
    # Near-coincident tail points must not produce overlapping gamma labels.
    labelled = [ax.transData.transform((row.residual_norm, row.regularization_norm))
                for row in data[selected].itertuples()]
    for index in np.unique(np.linspace(0, len(data)-1, min(6, len(data)), dtype=int)):
        row = data.iloc[index]
        point = ax.transData.transform((row.residual_norm, row.regularization_norm)) if valid.iloc[index] else None
        if point is not None and all(np.linalg.norm(point-previous) > 50 for previous in labelled):
            ax.annotate(f'{row.gamma:.1e}', (row.residual_norm, row.regularization_norm),
                        xytext=(4, 5), textcoords='offset points', fontsize=8)
            labelled.append(point)
    for extension in ('png', 'svg'):
        fig.savefig(output/f'l_curve.{extension}', dpi=300, bbox_inches='tight')
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    curvature = data.get('curvature', pd.Series(np.nan, index=data.index))
    ax.semilogx(data.gamma, curvature, 'o-', color='#1764a0', markersize=4)
    ax.axhline(0, color='gray', linewidth=.7)
    if selected.any():
        row = data[selected].iloc[0]
        ax.plot(row.gamma, row.curvature, '*', color='#c0392b', markersize=13)
    ax.set(xlabel=r'Regularization parameter $\gamma$', ylabel='Signed log-log curvature',
           title='Discrete L-curve curvature')
    fig.tight_layout()
    for extension in ('png', 'svg'):
        fig.savefig(output/f'l_curve_curvature.{extension}', dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_modulus_comparison(output, mesh, reference, reconstructed):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), constrained_layout=True)
    lo, hi = min(reference.min(), reconstructed.min()), max(reference.max(), reconstructed.max())
    for index, (values, title) in enumerate(((reference, 'Calibrated E (MPa)'),
                                             (reconstructed, 'Inverse E (MPa)'),
                                             (100 * (reconstructed / reference - 1), 'Relative difference (%)'))):
        kwargs = {'vmin': lo, 'vmax': hi} if index < 2 else {}
        image = axes[index].pcolormesh(mesh.plot_x, mesh.plot_y,
                                      values.reshape(mesh.nods_y, mesh.nods_x),
                                      shading='auto', cmap='viridis' if index < 2 else 'coolwarm',
                                      **kwargs)
        axes[index].set(title=title, xlabel='x (mm)', ylabel='y (mm)', aspect='equal')
        axes[index].set_xlim(0., mesh.geo_l)
        axes[index].set_ylim(0., mesh.geo_h)
        fig.colorbar(image, ax=axes[index], shrink=0.8)
    fig.savefig(Path(output) / 'modulus_comparison.png', dpi=300)
    fig.savefig(Path(output) / 'modulus_comparison.svg')
    plt.close(fig)

def save_x_profile(output, mesh, reconstructed, reference):
    """Summarize a free 2D inverse without imposing equality between y rows.

    Width statistics use trapezoidal weights for the Q4 nodal field. Spatial
    spread is not an uncertainty/confidence interval for the material modulus.
    """
    field = np.asarray(reconstructed).reshape(mesh.nods_y, mesh.nods_x)
    weights = np.ones(mesh.nods_y)
    weights[[0, -1]] = .5
    mean = np.average(field, axis=0, weights=weights)
    spread = np.sqrt(np.average((field - mean)**2, axis=0, weights=weights))
    x = mesh.plot_x[0]
    expected = reference(x, np.zeros_like(x))
    table = pd.DataFrame({'x_mm': x, 'reference_E_MPa': expected,
                          'inverse_width_mean_E_MPa': mean,
                          'inverse_transverse_sd_MPa': spread,
                          'inverse_width_min_E_MPa': field.min(axis=0),
                          'inverse_width_max_E_MPa': field.max(axis=0)})
    table.to_csv(Path(output) / 'x_profile.csv', index=False, encoding='utf-8-sig')
    fig, axes = plt.subplots(2, 1, figsize=(7., 5.), sharex=True, constrained_layout=True)
    dense_x = np.linspace(0., mesh.geo_l, 1401)
    axes[0].plot(dense_x, reference(dense_x, np.zeros_like(dense_x)), color='black',
                 linestyle='--', label='Independent calibration')
    axes[0].plot(x, mean, color='#0072B2', label='2D inverse: width mean')
    axes[0].fill_between(x, mean-spread, mean+spread, color='#0072B2', alpha=.18,
                         label='Transverse spatial SD')
    axes[0].set_ylabel('E (MPa)')
    axes[0].legend(frameon=False)
    axes[1].plot(x, spread, color='#D55E00')
    axes[1].set(xlabel='x (mm)', ylabel='Transverse SD (MPa)', xlim=(0., mesh.geo_l))
    fig.savefig(Path(output) / 'x_profile.png', dpi=300)
    fig.savefig(Path(output) / 'x_profile.svg')
    plt.close(fig)
