"""Four independent stages for the unified material model on 380N DIC.

Reuse the frozen simulation's solver and scan implementation. Experimental
execution is exploratory; the failed simulation gate is preserved separately.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np

from fgm_asm.fem_forward import fem_assemble, forward_solver
from fgm_asm.load_case import evaluate_observation
from fgm_asm.material import MaterialInfo
from fgm_asm.observation import compute_axis_cell_widths
from .config_utils import configuration_fingerprint, load_python_configuration
from .dic_preprocessing import grid_fields_from_points, plot_grid_fields
from .research_reporting import plot_l_curve, save_run
from . import unified_coarse_lcurve as shared
from .unified_gradient_layered_study import read_json, write_json, write_csv

ROOT = Path(__file__).resolve().parents[1]


def sources():
    return dict(shared.hashes(), **{'lib/unified_380n_workflow.py': shared.digest(Path(__file__))})


def prepare(cfg):
    root = Path(cfg.CASE_OUTPUT)
    if root.exists():
        raise FileExistsError('Run exists; use another RUN_NAME instead of overwriting')
    requested_hash = configuration_fingerprint(cfg)
    source = Path(cfg.REFERENCE_PREPARED)
    meta = read_json(source/'run_metadata.json')
    cfg.GEO_L, cfg.GEO_H, cfg.THICKNESS = meta['physical_model_mm']
    cfg.BOUNDARIES = meta['configuration']['REGION_BOUNDARIES_X']
    cfg.BEARING_WIDTH = meta['domain_geometry']['load_section_width_mm']
    cfg.FORCE = meta['force_pairing'][cfg.PLATFORM]['mean_corrected_force_N']
    cfg.CALIBRATION = meta['calibration']
    with np.load(source/'fields.npz') as saved:
        arrays = dict(xy=saved[f'{cfg.PLATFORM}_coordinates'],
                      data=saved[f'{cfg.PLATFORM}_observed'].ravel(),
                      areas=saved[f'{cfg.PLATFORM}_point_area_mm2'])
    if arrays['xy'].shape != (23920, 2) or arrays['data'].shape != (47840,):
        raise ValueError('Expected all 23920 DIC points and two interleaved components')
    if not all(np.isfinite(a).all() for a in arrays.values()) or np.any(arrays['areas'] <= 0):
        raise ValueError('DIC arrays must be finite, with positive point areas')
    summary = dict(status='prepared', scope='380N exploratory inverse, fixed known interfaces',
                   requested_configuration_sha256=requested_hash, kernel_sha256=sources(),
                   geometry_source_sha256=shared.digest(source/'run_metadata.json'),
                   data_source_sha256=shared.digest(source/'fields.npz'),
                   force_pairing=meta['force_pairing'][cfg.PLATFORM],
                   coordinates_verified=meta['coordinates_verified'],
                   material_registration=meta['material_registration'],
                   calibration_provenance=meta['calibration_source_sha256'],
                   observation_sources=meta['platforms'][cfg.PLATFORM],
                   simulation_gate_overridden_for_exploration=True,
                   calibration_used_in_initialization=False, calibration_used_in_selection=False)
    save_run(root/'01_prepared', cfg, summary, arrays)
    write_json(root/'01_prepared/inputs.json', dict(fields_sha256=shared.digest(root/'01_prepared/fields.npz')))
    publish(root)


def load(config):
    requested = load_python_configuration(config)
    root = Path(requested.CASE_OUTPUT)
    meta = read_json(root/'01_prepared/run_metadata.json')
    if configuration_fingerprint(requested) != meta['requested_configuration_sha256'] or sources() != meta['kernel_sha256']:
        raise ValueError('Configuration/kernel differs from frozen preparation')
    if shared.digest(root/'01_prepared/fields.npz') != read_json(root/'01_prepared/inputs.json')['fields_sha256']:
        raise ValueError('Frozen DIC input hash mismatch')
    return root, SimpleNamespace(**meta['configuration'])


def observations(root):
    with np.load(root/'01_prepared/fields.npz') as saved:
        return {key: saved[key] for key in saved.files}


def calibrated_function(cfg):
    values = np.array([row['mean_E_MPa'] for row in cfg.CALIBRATION])
    return lambda x, y: values[np.searchsorted(cfg.BOUNDARIES, x, side='right')]


def displacement_plots(folder, cfg, data, predictions):
    xy, d, area = data['xy'], data['data'].reshape(-1, 2), data['areas']
    center = np.sum(area[:, None]*d, axis=0)/area.sum()
    values = {'dic': d-center, **{name: v.reshape(-1, 2)-center for name, v in predictions.items()}}
    for c, component in enumerate(('ux', 'uy')):
        fields = {name: value[:, c] for name, value in values.items()}
        X, Y, grids = grid_fields_from_points(xy, fields)
        lo, hi = min(a.min() for a in fields.values()), max(a.max() for a in fields.values())
        plot_grid_fields(folder, component, X, Y, [(name, value, f'380N {name}: {component}') for name, value in grids.items()],
                         'jet', limits={name: (lo, hi) for name in grids}, unit='mm', title_prefix='',
                         bounds=(0, cfg.GEO_L, 0, cfg.GEO_H))
        errors = {name: (pred.reshape(-1, 2)[:, c]-d[:, c])*1000 for name, pred in predictions.items()}
        X, Y, grids = grid_fields_from_points(xy, errors)
        limit = max(float(np.max(abs(v))) for v in errors.values())
        plot_grid_fields(folder, component+'_residual', X, Y,
                         [(name, value, f'380N {name} - DIC: {component}') for name, value in grids.items()],
                         'coolwarm', limits={name: (-limit, limit) for name in grids}, unit='um', title_prefix='',
                         bounds=(0, cfg.GEO_L, 0, cfg.GEO_H))


def forward(root, cfg):
    out = root/'02_forward'/cfg.PLATFORM
    out.mkdir(parents=True, exist_ok=False)
    data = observations(root)
    m = shared.mesh(cfg, truth=True, kind='layered')
    case = shared.case_from_arrays(m, cfg, data['xy'], data['data'], data['areas'], cfg.PLATFORM)
    material = MaterialInfo(cfg.NU, modulus_function=calibrated_function(cfg))
    u = forward_solver(fem_assemble(m, material, case.bc_info))
    evaluated = evaluate_observation(case, u)
    np.savez_compressed(out/'observations.npz', **data)
    np.savez_compressed(out/'calibrated.npz', U=u, prediction=evaluated['prediction'],
                        residual=evaluated['residual'], rigid_coefficients=evaluated['rigid_coefficients'])
    write_json(out/'summary.json', dict(source_sha256=shared.hashes(),
               observation_sha256=shared.digest(out/'observations.npz'),
               calibrated_sha256=shared.digest(out/'calibrated.npz'),
               forward_evaluations=1, sharp_interface_subcell_quadrature=True,
               calibration_used_in_inverse=False))
    displacement_plots(out, cfg, data, {'calibrated': evaluated['prediction']})
    publish(root)


def compare(root, cfg):
    folder = root/'03_inverse'/cfg.PLATFORM
    selection = read_json(folder/'selection.json')
    selected = selection['selected_gamma'] is not None
    indices = (selection['optimal_index'],) if selected else (0, 24, 49)
    out = root/'04_comparison'
    out.mkdir(exist_ok=True)
    data = observations(root)
    forward_folder = root/'02_forward'/cfg.PLATFORM
    forward_summary = read_json(forward_folder/'summary.json')
    if shared.digest(forward_folder/'calibrated.npz') != forward_summary['calibrated_sha256']:
        raise ValueError('Independent calibrated forward identity mismatch')
    with np.load(forward_folder/'calibrated.npz') as saved:
        calibrated_prediction = saved['prediction']
    ny, nx = cfg.EVALUATION_SHAPE
    X, Y = np.meshgrid(np.linspace(0, cfg.GEO_L, nx), np.linspace(0, cfg.GEO_H, ny))
    xy = np.column_stack((X.ravel(), Y.ravel()))
    wy = compute_axis_cell_widths(Y[:, 0])
    p = shared.model(shared.mesh(cfg), cfg)
    reference = calibrated_function(cfg)(X, Y)
    states, records = [], []
    for index in indices:
        point = folder/f'gamma_{index:02d}'
        row = read_json(point/'summary.json')
        with np.load(point/'fields.npz') as saved:
            E = p.sample(saved['theta'], xy).reshape(X.shape)
            prediction, residual = saved['prediction'], saved['residual'].reshape(-1, 2)
        states.append(E)
        row.update(index=index, selected=selected,
                   component_rmse_um=(1000*np.sqrt(np.sum(data['areas'][:, None]*residual**2, axis=0)/data['areas'].sum())).tolist())
        records.append(row)
        label = 'identified' if selected else f'unselected_{index:02d}'
        displacement_plots(out/label, cfg, data, {'calibrated': calibrated_prediction, label: prediction})
        if not (out/f'{label}.npz').exists():
            np.savez_compressed(out/f'{label}.npz', X=X, Y=Y, E=E)
    lo, hi = min(a.min() for a in [reference, *states]), max(a.max() for a in [reference, *states])
    plot_grid_fields(out, '380N', X, Y, [('calibration', reference, 'Independent calibration')], 'jet',
                     limits={'calibration': (lo, hi)}, unit='MPa', title_prefix='', bounds=(0, cfg.GEO_L, 0, cfg.GEO_H))
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.plot(X[0], wy@reference/wy.sum(), 'k--', label='Independent calibration')
    for E, row, color in zip(states, records, ('#0072B2', '#D55E00', '#009E73')):
        label = 'Identified' if selected else f'Unselected #{row["index"]:02d}'
        title = f'{label}, gamma={row["gamma"]:.2e}, first-order={row["first_order_passed"]}'
        suffix = f'gamma_{row["index"]:02d}'
        plot_grid_fields(out, '380N', X, Y, [(suffix, E, title)], 'jet', limits={suffix: (lo, hi)},
                         unit='MPa', title_prefix='', bounds=(0, cfg.GEO_L, 0, cfg.GEO_H))
        mean = wy@E/wy.sum()
        spread = np.sqrt(wy@(E-mean[None, :])**2/wy.sum())
        ax.plot(X[0], mean, color=color, label=label)
        if not (out/f'{suffix}_profile.csv').exists():
            write_csv(out/f'{suffix}_profile.csv', [dict(x_mm=x, mean_E_MPa=v, spatial_sd_MPa=s)
                      for x, v, s in zip(X[0], mean, spread)])
    ax.set(xlabel='x (mm)', ylabel='Width-averaged E (MPa)', title='380N, assumed net force 319.9 N')
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out/'width_profile.png', dpi=300, bbox_inches='tight')
    fig.savefig(out/'width_profile.svg', bbox_inches='tight')
    plt.close(fig)
    write_json(out/'summary.json', dict(status='selected_exploratory' if selected else 'unselected_diagnostics',
               selection=selection, records=records, force_is_assumed=True,
               physical_validation=False, source_sha256=sources()))
    publish(root)


def publish(root):
    style = '<style>body{max-width:1100px;margin:25px auto;padding:20px;font:16px/1.7 "Microsoft YaHei",sans-serif}img{max-width:100%}a{color:#075fa2}</style>'
    parts = ['<!doctype html><meta charset="utf-8">'+style+'<h1>380N：统一材料模型／区内正则</h1>',
             '<p>121×19单元、23920个DIC测点、七个二维Q4分支；已知界面，ν=0.26。'
             '沿用既有380N载荷假设：净力319.9 N，非同步实测力。仿真进入条件未通过，本组为直接实验诊断。</p>']
    selection = root/'03_inverse'/'380N'/'selection.json'
    if selection.exists():
        record = read_json(selection)
        parts.append(f'<p>一阶通过 {record["valid_points"]}/50；选参状态：{record["status"]}；γ：{record["selected_gamma"]}。</p>')
        if record['selected_gamma'] is None:
            parts.append('<p>未取得有效γ。编号00、24、49仅作预定诊断，不作为正式识别场。</p>')
    else:
        parts.append('<p>扫描尚未完成；不使用标定误差或其他载荷选择γ。</p>')
    for path in sorted(root.rglob('*.png')):
        rel = path.relative_to(root).as_posix()
        parts.append(f'<p>{rel}</p><a href="{Path(rel).with_suffix(".svg").as_posix()}"><img src="{rel}" loading="lazy"></a>')
    for path in sorted(root.rglob('selection.json'))+sorted(root.rglob('04_comparison/summary.json')):
        rel = path.relative_to(root).as_posix()
        parts.append(f'<p><a href="{rel}">{rel}</a></p>')
    (root/'结果总览.html').write_text('\n'.join(parts), encoding='utf-8')
    shared_root = root.parent
    (shared_root/'结果总览.html').write_text('<meta charset="utf-8">'+style+'<h1>统一材料算法</h1>'+''.join(
        f'<p><a href="{p.parent.name}/结果总览.html">{p.parent.name}</a></p>'
        for p in sorted(shared_root.glob('*/结果总览.html'))), encoding='utf-8')
    print('保存位置：', root/'结果总览.html', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare', 'forward', 'inverse', 'compare'))
    parser.add_argument('--config', default='unified_380n_config')
    parser.add_argument('--plot-only', action='store_true')
    args = parser.parse_args()
    if args.stage == 'prepare' and not args.plot_only:
        prepare(load_python_configuration(args.config)); return
    root, cfg = load(args.config)
    if args.plot_only:
        if args.stage == 'inverse':
            import pandas as pd
            folder = root/'03_inverse'/cfg.PLATFORM
            plot_l_curve(folder, pd.read_csv(folder/'scan.csv').to_dict('records'))
        elif args.stage == 'compare':
            compare(root, cfg)
        elif args.stage == 'forward':
            with np.load(root/'02_forward'/cfg.PLATFORM/'calibrated.npz') as saved:
                displacement_plots(root/'02_forward'/cfg.PLATFORM, cfg, observations(root), {'calibrated': saved['prediction']})
        publish(root)
    elif args.stage == 'forward':
        forward(root, cfg)
    elif args.stage == 'inverse':
        shared.inverse(root, cfg, cfg.PLATFORM)
        publish(root)
    else:
        compare(root, cfg)


if __name__ == '__main__':
    main()
