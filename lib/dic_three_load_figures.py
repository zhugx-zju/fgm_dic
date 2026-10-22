"""Replot the three coarse H1 cases directly from their frozen point fields.

No mechanical solver is loaded: registered predictions and physical coordinates
are retained exactly, even when the active FE implementation has since changed.
"""

from pathlib import Path
import hashlib
import json

import matplotlib.pyplot as plt
import numpy as np

from .dic_preprocessing import grid_fields_from_points


def plot_three_load_displacements(project_root: Path, output: Path) -> dict:
    """Compare all loads with common component scales and no amplitude scaling."""
    project_root, output = Path(project_root).resolve(), Path(output).resolve()
    cases = {
        '180N': 'coarse',
        '280N': 'h1_280N_independent_coarse_20261022',
        '380N': 'load380_20261022/h1',
    }
    records, hashes = [], {}
    for name, relative in cases.items():
        root = project_root / 'results/dic_full' / relative
        paths = [root / '01_prepared/run_metadata.json',
                 root / '01_prepared/fields.npz',
                 root / '02_forward/nu_0.26/fields.npz',
                 root / '03_inverse/nu_0.26/fields.npz']
        hashes.update({str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
        metadata = json.loads(paths[0].read_text(encoding='utf-8'))
        mesh, config = metadata['shared_mesh'], metadata['configuration']
        if (mesh['nel_x'], mesh['nel_y'], mesh['nodes']) != (121, 19, 2440):
            raise ValueError(f'{name}: expected the 121 x 19 coarse mesh')
        if config['FIT_CASES'] != [name] or config['PRIMARY_NU'] != .26:
            raise ValueError(f'{name}: expected own-platform identification at nu=0.26')
        if config.get('DISPLACEMENT_COMPONENT_WEIGHTS', [1., 1.]) != [1., 1.]:
            raise ValueError(f'{name}: expected equal displacement weights')
        with np.load(paths[1]) as data:
            xy = data[f'{name}_coordinates'].copy()
            observed = data[f'{name}_observed'].reshape(-1, 2).copy()
            ids = data[f'{name}_grid_ids'].copy()
        predictions = []
        for path in paths[2:]:
            with np.load(path) as data:
                predictions.append(data[f'{name}_prediction'].reshape(-1, 2).copy())
                if path == paths[3]:
                    np.testing.assert_allclose(data[f'{name}_coordinates'], xy, atol=0, rtol=0)
                    np.testing.assert_array_equal(data[f'{name}_observed'].reshape(-1, 2), observed)
        if observed.shape != (23920, 2) or any(a.shape != observed.shape for a in predictions):
            raise ValueError(f'{name}: point fields must have shape (23920, 2)')
        if records:
            np.testing.assert_allclose(xy, records[0]['xy'], atol=2e-8, rtol=0)
            if metadata['calibration_source_sha256'] != records[0]['calibration_sha256']:
                raise ValueError('All loads must use the same independent calibration')
        # The same translation is removed from DIC and both predictions. This
        # changes only the display origin, preserving all pointwise residuals.
        fields = np.stack([observed, *predictions]) - observed.mean(axis=0)
        records.append(dict(name=name, xy=xy, ids=ids, fields=fields,
                            force_N=metadata['force_pairing'][name]['mean_corrected_force_N'],
                            force_assumed=bool(config.get('FORCE_ASSUMED', False)),
                            calibration_sha256=metadata['calibration_source_sha256']))

    limits = [max(abs(r['fields'][:, :, c]).max() for r in records) for c in range(2)]
    labels = ['DIC Data', 'Displacement from\nCalibrated E', 'Displacement from\nIdentified E']
    with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 11}):
        fig = plt.figure(figsize=(17, 9))
        blocks = fig.add_gridspec(2, 1, left=.18, right=.91, bottom=.105,
                                 top=.89, hspace=.5)
        axes = np.empty((6, 3), dtype=object)
        for component in range(2):
            grid = blocks[component].subgridspec(3, 3, hspace=.55, wspace=.14)
            for model in range(3):
                row = 3 * component + model
                for column, record in enumerate(records):
                    ax = axes[row, column] = fig.add_subplot(grid[model, column])
                    x, y, fields = grid_fields_from_points(
                        record['xy'], {'u': record['fields'][model, :, component]}, record['ids'])
                    artist = ax.pcolormesh(x, y, fields['u'], cmap='jet', shading='nearest',
                                          vmin=-limits[component], vmax=limits[component], rasterized=True)
                    ax.set_aspect('equal')
                    ax.set_xticks([0, 25, 50, 75, 100])
                    ax.set_yticks([0, 10, 18])
                    ax.tick_params(labelsize=10, length=3)
                    if model == 0:
                        ax.set_title(record['name'], fontsize=14, pad=9)
                    if model == 2:
                        ax.set_xlabel('x (mm)', labelpad=1)
                    else:
                        ax.tick_params(labelbottom=False)
                    if column == 0:
                        ax.set_ylabel('y (mm)', labelpad=2)
                    else:
                        ax.tick_params(labelleft=False)
                position = axes[row, 0].get_position()
                fig.text(.018, (position.y0 + position.y1) / 2, labels[model],
                         fontsize=12, va='center')
            top = axes[3 * component, 2].get_position().y1
            bottom = axes[3 * component + 2, 2].get_position().y0
            bar = fig.colorbar(artist, cax=fig.add_axes([.93, bottom, .012, top-bottom]))
            bar.set_label(r'$u_x$ (mm)' if component == 0 else r'$u_y$ (mm)', fontsize=13)
        fig.suptitle('Three-load displacement comparison | coarse mesh 121 x 19 | H1 | '
                     r'$\nu=0.26$', fontsize=17, y=.98)
        fig.text(.18, .935, 'Axial displacement', fontsize=14, weight='bold')
        fig.text(.18, axes[3, 0].get_position().y1 + .065,
                 'Transverse displacement', fontsize=14, weight='bold')
        fig.text(.18, .03, 'Common color scale across all loads within each component; '
                 'display centered by each load\'s DIC mean.\n'
                 'Each identified E uses its own load data. 380N: assumed net force 319.9 N.',
                 fontsize=11)
        output.mkdir(parents=True, exist_ok=True)
        stem = output / 'three_load_displacement_comparison'
        fig.savefig(stem.with_suffix('.png'), dpi=240)
        fig.savefig(stem.with_suffix('.svg'))
        plt.close(fig)
    # Verify the postprocessing has not changed its frozen numerical inputs.
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == h for p, h in hashes.items())
    result = dict(protocol='three_load_coarse_displacement_comparison', source_sha256=hashes,
                  mesh=[121, 19], nu=.26, weights=[1., 1.],
                  shared_symmetric_color_limits_mm=[[-a, a] for a in limits],
                  display='Subtract each load DIC mean from all three fields; no amplitude scaling',
                  loads=[{k: r[k] for k in ('name', 'force_N', 'force_assumed')} for r in records])
    (output / 'three_load_displacement_metadata.json').write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return result
