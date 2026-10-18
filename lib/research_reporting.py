"""Save reproducible arrays and comparable field plots without rerunning FEM."""

import json
from pathlib import Path
import platform
import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np
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


def plot_displacement_comparison(output, name, xy, measured, state):
    """Compare each component at identical physical points and shared limits."""
    prediction = state['prediction'].reshape(-1, 2)
    residual = prediction - measured
    fig, axes = plt.subplots(2, 3, figsize=(12, 5.5), constrained_layout=True)
    for component, label in enumerate(('ux', 'uy')):
        lo = min(measured[:, component].min(), prediction[:, component].min())
        hi = max(measured[:, component].max(), prediction[:, component].max())
        error_limit = max(np.max(np.abs(residual[:, component])), 1e-12)
        for column, (field, title) in enumerate(((measured, 'Observed'),
                                                (prediction, 'Predicted + rigid fit'),
                                                (residual, 'Prediction - observation'))):
            ax = axes[component, column]
            artist = ax.scatter(xy[:, 0], xy[:, 1], c=field[:, component], s=6,
                                cmap='coolwarm' if column == 2 else 'viridis',
                                vmin=-error_limit if column == 2 else lo,
                                vmax=error_limit if column == 2 else hi,
                                rasterized=True)
            ax.set(title=f'{title}: {label} (mm)', xlabel='x (mm)', ylabel='y (mm)', aspect='equal')
            fig.colorbar(artist, ax=ax, shrink=0.75)
    fig.savefig(Path(output) / f'{name}_displacement.png', dpi=300)
    fig.savefig(Path(output) / f'{name}_displacement.svg')
    plt.close(fig)


def plot_modulus_comparison(output, mesh, reference, reconstructed):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), constrained_layout=True)
    lo, hi = min(reference.min(), reconstructed.min()), max(reference.max(), reconstructed.max())
    for index, (values, title) in enumerate(((reference, 'Reference E (MPa)'),
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
