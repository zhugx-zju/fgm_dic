"""Four independent stages for the unified material P0/P1 research branch.

Run as a module from the repository root. New solves never overwrite outputs;
--plot-only regenerates figures from saved arrays without calling a solver.
The production DIC entry points and their configuration remain unchanged.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from fgm_asm.fem_forward import fem_assemble, forward_solver
from fgm_asm.integration import rectangular_quadrature
from fgm_asm.load_case import ObservationCase, evaluate_observation
from fgm_asm.mesh import MeshInfo, setup_boundary_conditions
from fgm_asm.observation import build_observation_package, compute_axis_cell_widths
from fgm_asm.unified_inverse import fit_uniform_initial_modulus, solve_unified
from fgm_asm.unified_material import MappedMaterial, RegionMaterialMap
from .config_utils import configuration_fingerprint, load_python_configuration
from .dic_preprocessing import plot_grid_fields
from .research_reporting import _json_value, save_run

ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILES = ('fgm_asm/fem_forward.py', 'fgm_asm/inverse_solver.py', 'fgm_asm/mesh.py',
                'fgm_asm/integration.py', 'fgm_asm/material.py', 'fgm_asm/observation.py',
                'fgm_asm/load_case.py', 'fgm_asm/unified_material.py',
                'fgm_asm/unified_inverse.py', 'lib/unified_gradient_layered_study.py')
NOTICE = ('P0/P1 小规模无噪声同表示诊断。有限平滑界面、单拉伸观测；'
          '不代表尖锐移动界面、含噪稳定性、正式三方法对照或真实 DIC 验证。')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, data):
    Path(path).write_text(json.dumps(_json_value(data), ensure_ascii=False,
                                   indent=2, allow_nan=False)+'\n', encoding='utf-8')


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def source_hashes():
    return {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in SOURCE_FILES}


def validate_config(cfg):
    if (not cfg.CASES or set(cfg.CASES)-{'continuous_2d', 'layered', 'mixed_2d'} or
            not cfg.METHODS or set(cfg.METHODS)-{'continuous_q4', 'layered_constant', 'unified_q4'}):
        raise ValueError('Unsupported or empty diagnostic cases/methods')
    if any(not np.isfinite(v) or v <= 0 for v in
           (cfg.GEO_L, cfg.GEO_H, cfg.THICKNESS, cfg.BEARING_WIDTH, cfg.FORCE, cfg.EPSILON_MM)):
        raise ValueError('Geometry, thickness, force and epsilon must be positive finite values')
    if (not 0.24 <= cfg.NU <= 0.33 or not np.isfinite(cfg.GAMMA) or cfg.GAMMA < 0 or
            any(not isinstance(v, int) or v < 1 for v in
                (cfg.NEL_X, cfg.NEL_Y, cfg.REFERENCE_REFINEMENT, cfg.MAX_ITERATIONS, cfg.MAX_EVALUATIONS))):
        raise ValueError('Invalid nu/gamma or integer mesh/budget settings')
    low, high = cfg.UNKNOWN_INTERFACE_BOUNDS
    if not 0 < low < high < 1 or any(not low <= s <= high for s in cfg.UNKNOWN_INTERFACE_STARTS):
        raise ValueError('Unknown interface starts must lie in the declared domain bounds')
    if not 0 < cfg.TRUTH_INTERFACE_FRACTION < 1 or not 0 < cfg.KNOWN_INTERFACE_FRACTION < 1:
        raise ValueError('Interfaces must lie inside the synthetic domain')
    if min(cfg.EVALUATION_NX, cfg.EVALUATION_NY) < 2:
        raise ValueError('Evaluation grid must contain at least two points per axis')


def mesh_from_config(cfg, refinement=1):
    mesh = MeshInfo(cfg.GEO_L, cfg.GEO_H, cfg.NEL_X*refinement, cfg.NEL_Y*refinement)
    return rectangular_quadrature(mesh, cfg.QUADRATURE_ORDER, cfg.QUADRATURE_SUBDIVISIONS)


def boundary_conditions(mesh, cfg):
    return setup_boundary_conditions(mesh, cfg.GEO_L, cfg.GEO_H,
                                     cfg.FORCE*cfg.GEO_H/cfg.BEARING_WIDTH, cfg.THICKNESS)


def truth_parameters(model, kind, cfg):
    """Separate synthetic generator; inverse code never calls this function."""
    x, y = model.mesh.X/cfg.GEO_L, model.mesh.Y/cfg.GEO_H
    if kind == 'continuous_2d':
        a = 1400 + 650*x + 200*y + 100*x*y
        b = a.copy()
    elif kind == 'layered':
        a, b = np.full_like(x, 1300.), np.full_like(x, 2300.)
    elif kind == 'mixed_2d':
        a, b = 1100+450*x+180*y+80*x*y, 2100+250*x-140*y+60*x*y
    else:
        raise ValueError(kind)
    return np.log(np.r_[a, b])


def truth_map(mesh, cfg):
    return RegionMaterialMap(mesh, interfaces=[[cfg.GEO_L*cfg.TRUTH_INTERFACE_FRACTION]],
                             epsilon=cfg.EPSILON_MM)


def prepare(cfg):
    validate_config(cfg)
    root = Path(cfg.CASE_OUTPUT).resolve()
    output = root/'01_prepared'
    if root.exists():
        raise FileExistsError(f'运行目录已存在，另设 RUN_NAME 或使用 --plot-only：{root}')
    mesh = mesh_from_config(cfg)
    areas = np.outer(compute_axis_cell_widths(np.linspace(0, cfg.GEO_H, mesh.nods_y)),
                     compute_axis_cell_widths(np.linspace(0, cfg.GEO_L, mesh.nods_x))).ravel()
    summary = dict(workflow='unified_gradient_layered_p0_p1', status='prepared',
                   scope=NOTICE, requested_configuration_sha256=configuration_fingerprint(cfg),
                   kernel_source_sha256=source_hashes(), synthetic_observations=True,
                   units=dict(length='mm', force='N', modulus='MPa'),
                   truth_and_inverse_same_representation=True,
                   truth_initialization_used=False, coordinates_verified=False)
    save_run(output, cfg, summary, dict(xy=mesh.coord, areas=areas))
    folder = output/'truth'
    folder.mkdir()
    model = truth_map(mesh, cfg)
    for kind in cfg.CASES:
        np.savez_compressed(folder/f'{kind}.npz', theta=truth_parameters(model, kind, cfg),
                            interfaces=model.interfaces)
    write_json(output/'input_sha256.json',
               {str(p.relative_to(output)): sha(p) for p in [output/'fields.npz', *folder.glob('*.npz')]})
    plot_preparation(root, cfg)
    publish(root)


def frozen_config(config_name, explicit_root=None):
    requested = load_python_configuration(config_name)
    root = Path(explicit_root or requested.CASE_OUTPUT).resolve()
    metadata = read_json(root/'01_prepared'/'run_metadata.json')
    if explicit_root is None and configuration_fingerprint(requested) != metadata['requested_configuration_sha256']:
        raise ValueError('当前配置与准备阶段不同；更换 RUN_NAME，或用 --input 指向原冻结运行')
    if source_hashes() != metadata['kernel_source_sha256']:
        raise ValueError('算法源码与冻结准备阶段不一致；新建 RUN_NAME 重新准备，不混用输入')
    for name, expected in read_json(root/'01_prepared'/'input_sha256.json').items():
        if sha(root/'01_prepared'/name) != expected:
            raise ValueError(f'Prepared input identity mismatch: {name}')
    cfg = SimpleNamespace(**metadata['configuration'])
    validate_config(cfg)
    return root, cfg


def plot_preparation(root, cfg):
    mesh = mesh_from_config(cfg)
    model = truth_map(mesh, cfg)
    fields = []
    for kind in cfg.CASES:
        with np.load(root/'01_prepared'/'truth'/f'{kind}.npz') as saved:
            field = model.sample(saved['theta'], mesh.coord).reshape(mesh.nods_y, mesh.nods_x)
        fields.append((kind, field, f'{kind}: synthetic truth'))
    plot_grid_fields(root/'01_prepared', 'truth', mesh.plot_x, mesh.plot_y, fields,
                     cmap='jet', unit='MPa', title_prefix='',
                     limits={name: (1000, 2500) for name, _, _ in fields},
                     bounds=(0, cfg.GEO_L, 0, cfg.GEO_H))
    gallery(root/'01_prepared', '① 小规模合成输入')


def make_case(mesh, cfg, xy, data, areas, name):
    obs = build_observation_package(mesh, xy[:, 0], xy[:, 1], data[0::2], data[1::2],
                                    areas, sampling_mode='synthetic_physical_points')
    return ObservationCase(name, boundary_conditions(mesh, cfg), obs,
                           metadata=dict(reference='unloaded', synthetic=True, force_N=cfg.FORCE))


def forward(root, cfg):
    output = root/'02_forward'
    if output.exists():
        raise FileExistsError(f'正演结果已存在；使用 --plot-only：{output}')
    output.mkdir()
    mesh, fine = mesh_from_config(cfg), mesh_from_config(cfg, cfg.REFERENCE_REFINEMENT)
    with np.load(root/'01_prepared'/'fields.npz') as saved:
        xy, areas = saved['xy'], saved['areas']
    rows = []
    for kind in cfg.CASES:
        model = truth_map(mesh, cfg)
        with np.load(root/'01_prepared'/'truth'/f'{kind}.npz') as saved:
            theta = saved['theta']
        u = forward_solver(fem_assemble(mesh, MappedMaterial(cfg.NU, model, theta),
                                        boundary_conditions(mesh, cfg)))
        reference_model = truth_map(fine, cfg)
        theta_fine = truth_parameters(reference_model, kind, cfg)
        uf = forward_solver(fem_assemble(fine, MappedMaterial(cfg.NU, reference_model, theta_fine),
                                         boundary_conditions(fine, cfg)))
        case = make_case(fine, cfg, xy, u, areas, kind)
        residual = evaluate_observation(case, uf)['residual'].reshape(-1, 2)
        row = dict(case=kind, forward_evaluations=2,
                   axial_refinement_rmse_mm=float(np.sqrt(areas @ residual[:, 0]**2/areas.sum())),
                   transverse_refinement_rmse_mm=float(np.sqrt(areas @ residual[:, 1]**2/areas.sum())))
        rows.append(row)
        np.savez_compressed(output/f'{kind}.npz', xy=xy, areas=areas, displacement=u)
    write_csv(output/'mesh_diagnostic.csv', rows)
    write_json(output/'summary.json', dict(scope=NOTICE, observations='coarse same-representation truth',
               refinement_is_diagnostic_only=True, refinement=cfg.REFERENCE_REFINEMENT,
               records=rows, kernel_source_sha256=source_hashes(),
               observations_sha256={kind: sha(output/f'{kind}.npz') for kind in cfg.CASES},
               prepared_metadata_sha256=sha(root/'01_prepared'/'run_metadata.json')))
    plot_forward(root, cfg)
    publish(root)


def plot_forward(root, cfg):
    mesh = mesh_from_config(cfg)
    for kind in cfg.CASES:
        with np.load(root/'02_forward'/f'{kind}.npz') as saved:
            u = saved['displacement'].reshape(mesh.nods_y, mesh.nods_x, 2)
        plot_grid_fields(root/'02_forward', kind, mesh.plot_x, mesh.plot_y,
                         [(c, u[..., i], f'{kind}: synthetic {c}') for i, c in enumerate(('ux', 'uy'))],
                         cmap='jet', unit='mm', title_prefix='', bounds=(0, cfg.GEO_L, 0, cfg.GEO_H))
    gallery(root/'02_forward', '② 共用内核正演与加密诊断')


def inverse_map(mesh, cfg, method, geometry, start):
    if method not in cfg.METHODS:
        raise ValueError('Method must belong to the frozen METHODS')
    if method == 'continuous_q4':
        if geometry != 'fixed':
            raise ValueError('continuous_q4 has no interface; use --geometry fixed')
        return RegionMaterialMap(mesh, n_regions=1, epsilon=cfg.EPSILON_MM)
    position = (cfg.KNOWN_INTERFACE_FRACTION if geometry == 'fixed' else start)*cfg.GEO_L
    if geometry == 'unknown' and start not in cfg.UNKNOWN_INTERFACE_STARTS:
        raise ValueError('Unknown start must be declared in frozen UNKNOWN_INTERFACE_STARTS')
    return RegionMaterialMap(mesh, branch_kind='constant' if method == 'layered_constant' else 'q4',
                             interfaces=[[position]], epsilon=cfg.EPSILON_MM,
                             optimize_geometry=geometry == 'unknown')


def inverse_folder(root, kind, method, geometry, start):
    suffix = geometry if geometry == 'fixed' else f'unknown_start_{start:.6g}'
    return root/'03_inverse'/kind/method/suffix


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inverse(root, cfg, kind, method, geometry, start):
    if kind not in cfg.CASES:
        raise ValueError('Case must belong to frozen CASES')
    reference = read_json(root/'02_forward'/'summary.json')
    if (reference['kernel_source_sha256'] != source_hashes() or
            reference['prepared_metadata_sha256'] != sha(root/'01_prepared'/'run_metadata.json')):
        raise ValueError('Forward and prepared sources do not match')
    output = inverse_folder(root, kind, method, geometry, start)
    if output.exists():
        raise FileExistsError(f'识别结果已存在：{output}')
    mesh = mesh_from_config(cfg)
    model = inverse_map(mesh, cfg, method, geometry, start)
    data_file = root/'02_forward'/f'{kind}.npz'
    if reference['observations_sha256'][kind] != sha(data_file):
        raise ValueError('Frozen forward observation identity mismatch')
    with np.load(data_file) as saved:
        case = make_case(mesh, cfg, saved['xy'], saved['displacement'], saved['areas'], kind)
    uniform = fit_uniform_initial_modulus(model, case, cfg.NU)
    bounds = [tuple(cfg.GEO_L*np.array(cfg.UNKNOWN_INTERFACE_BOUNDS))]
    output.mkdir(parents=True)
    write_json(output/'started.json', dict(case=kind, method=method, geometry=geometry,
               start_fraction=start if geometry == 'unknown' else None,
               initial_modulus_MPa=uniform, initialization='weighted nonrigid uniform compliance fit',
               observation_sha256=sha(data_file), kernel_source_sha256=source_hashes()))
    state, evidence, history = solve_unified(model, case, uniform, nu=cfg.NU, gamma=cfg.GAMMA,
                modulus_bounds=cfg.MODULUS_BOUNDS_MPA, geometry_bounds=bounds,
                max_iterations=cfg.MAX_ITERATIONS, max_evaluations=cfg.MAX_EVALUATIONS,
                ftol=cfg.FTOL, gtol=cfg.GTOL)
    np.savez_compressed(output/'fields.npz', **{key: state[key] for key in
                        ('theta', 'U', 'prediction', 'residual', 'gradient',
                         'data_gradient', 'regularization_gradient', 'rigid_coefficients')})
    write_csv(output/'history.csv', history)
    write_json(output/'summary.json', dict(**read_json(output/'started.json'), **evidence,
               initialization_forward_evaluations=1, total_forward_evaluations=evidence['forward_evaluations']+1,
               n_parameters=model.n_parameters, n_material_parameters=model.n_material,
               gamma=cfg.GAMMA, data_cost=state['data_cost'], regularization=state['regularization'],
               scope=NOTICE, truth_used_in_objective=False, truth_used_in_initialization=False))
    print(f'{kind}/{method}/{output.name}: {evidence["message"]}; FE={evidence["forward_evaluations"]}', flush=True)
    plot_inverse(output, root, cfg)
    publish(root)


def evaluation_grid(cfg):
    x, y = np.linspace(0, cfg.GEO_L, cfg.EVALUATION_NX), np.linspace(0, cfg.GEO_H, cfg.EVALUATION_NY)
    X, Y = np.meshgrid(x, y)
    weights = np.outer(compute_axis_cell_widths(y), compute_axis_cell_widths(x))
    return X, Y, np.column_stack([X.ravel(), Y.ravel()]), weights


def saved_model(output, cfg):
    summary = read_json(output/'summary.json')
    model = inverse_map(mesh_from_config(cfg), cfg, summary['method'], summary['geometry'],
                        summary['start_fraction'])
    return summary, model


def plot_inverse(output, root, cfg):
    summary, model = saved_model(output, cfg)
    X, Y, xy, _ = evaluation_grid(cfg)
    with np.load(output/'fields.npz') as saved:
        E = model.sample(saved['theta'], xy).reshape(X.shape)
        residual = saved['residual'].reshape(model.mesh.nods_y, model.mesh.nods_x, 2)
    plot_grid_fields(output, 'inverse', X, Y, [('E', E, f'{summary["case"]}: {summary["method"]}')],
                     cmap='jet', unit='MPa', title_prefix='', limits={'E': (1000, 2500)},
                     bounds=(0, cfg.GEO_L, 0, cfg.GEO_H))
    plot_grid_fields(output, 'residual', model.mesh.plot_x, model.mesh.plot_y,
                     [(c, residual[..., i], f'Aligned data - fitted {c}')
                      for i, c in enumerate(('ux', 'uy'))], cmap='coolwarm', unit='mm', title_prefix='',
                     bounds=(0, cfg.GEO_L, 0, cfg.GEO_H))
    gallery(output, '③ 无噪声反演诊断')


def compare(root, cfg):
    """Read saved fields; evaluate truth only here, never choose a method/gamma."""
    output = root/'04_comparison'
    output.mkdir(exist_ok=True)
    X, Y, xy, weights = evaluation_grid(cfg)
    rows = []
    for path in sorted((root/'03_inverse').rglob('summary.json')):
        folder = path.parent
        summary, model = saved_model(folder, cfg)
        if (summary['observation_sha256'] != sha(root/'02_forward'/f'{summary["case"]}.npz') or
                summary['kernel_source_sha256'] != source_hashes()):
            raise ValueError(f'Inverse result source identity mismatch: {folder}')
        with np.load(folder/'fields.npz') as saved:
            E = model.sample(saved['theta'], xy).reshape(X.shape)
            _, geometry = model.unpack(saved['theta'])
            residual = saved['residual'].reshape(-1, 2)
        truth = truth_map(model.mesh, cfg)
        with np.load(root/'01_prepared'/'truth'/f'{summary["case"]}.npz') as saved:
            reference = truth.sample(saved['theta'], xy).reshape(X.shape)
        with np.load(root/'02_forward'/f'{summary["case"]}.npz') as saved:
            areas = saved['areas']
        relative_rms = float(np.sqrt(np.sum(weights*(E-reference)**2)/np.sum(weights*reference**2)))
        geometry_error = (float(np.mean(abs(geometry-cfg.TRUTH_INTERFACE_FRACTION*cfg.GEO_L)))
                          if geometry.size and summary['case'] != 'continuous_2d' else None)
        rows.append(dict(case=summary['case'], method=summary['method'], protocol=folder.name,
                    modulus_relative_rms=relative_rms, interface_error_mm=geometry_error,
                    ux_rmse_mm=float(np.sqrt(areas @ residual[:, 0]**2/areas.sum())),
                    uy_rmse_mm=float(np.sqrt(areas @ residual[:, 1]**2/areas.sum())),
                    solver_success=summary['solver_success'], first_order_passed=summary['first_order_passed'],
                    projected_gradient_inf=summary['projected_gradient_inf'],
                    total_forward_evaluations=summary['total_forward_evaluations'],
                    n_parameters=summary['n_parameters'], active_bounds=summary['active_bounds'],
                    elapsed_seconds=summary['elapsed_seconds']))
        key = '_'.join([summary['case'], summary['method'], folder.name])
        np.savez_compressed(output/f'{key}.npz', X=X, Y=Y, weights=weights,
                            E=E, reference=reference, geometry=geometry)
    if not rows:
        raise ValueError('No saved inverse results to compare')
    write_csv(output/'comparison.csv', rows)
    write_json(output/'summary.json', dict(scope=NOTICE, records=rows,
               metric='common physical trapezoidal integration; relative RMS = sqrt(int error^2/int truth^2)',
               selection='No truth-based parameter or method selection; all saved successes/failures included',
               kernel_source_sha256=source_hashes()))
    plot_comparison(root, cfg)
    publish(root)


def plot_comparison(root, cfg):
    import matplotlib.pyplot as plt
    output = root/'04_comparison'
    rows = read_json(output/'summary.json')['records']
    for kind in cfg.CASES:
        records = [row for row in rows if row['case'] == kind]
        if not records:
            continue
        fields = []
        for row in records:
            key = '_'.join([kind, row['method'], row['protocol']])
            with np.load(output/f'{key}.npz') as saved:
                fields.append((key, row, {name: saved[name] for name in ('X', 'Y', 'E', 'reference')}))
        low = min(min(f['E'].min(), f['reference'].min()) for _, _, f in fields)
        high = max(max(f['E'].max(), f['reference'].max()) for _, _, f in fields)
        error_max = max(float(abs(f['E']-f['reference']).max()) for _, _, f in fields)
        for key, row, f in fields:
            plot_grid_fields(output, key, f['X'], f['Y'],
                             [('E', f['E'], f'{row["method"]}: {row["protocol"]}')],
                             cmap='jet', unit='MPa', limits={'E': (low, high)}, title_prefix='',
                             bounds=(0, cfg.GEO_L, 0, cfg.GEO_H))
            plot_grid_fields(output, key, f['X'], f['Y'],
                             [('error', f['E']-f['reference'], 'Identified - truth')],
                             cmap='coolwarm', unit='MPa', limits={'error': (-error_max, error_max)},
                             title_prefix='', bounds=(0, cfg.GEO_L, 0, cfg.GEO_H))
        f = fields[0][2]
        plot_grid_fields(output, kind, f['X'], f['Y'], [('truth', f['reference'], f'{kind}: truth')],
                         cmap='jet', unit='MPa', limits={'truth': (low, high)}, title_prefix='',
                         bounds=(0, cfg.GEO_L, 0, cfg.GEO_H))
        fig, ax = plt.subplots(figsize=(8, 4))
        y = f['Y'][:, 0]
        ax.plot(f['X'][0], np.trapezoid(f['reference'], y, axis=0)/cfg.GEO_H, 'k-', lw=2, label='Truth')
        styles = {'continuous_q4': ('#0072B2', '-'), 'layered_constant': ('#D55E00', '--'),
                  'unified_q4': ('#009E73', '-.')}
        for _, row, f in fields:
            color, style = styles[row['method']]
            mean = np.trapezoid(f['E'], y, axis=0)/cfg.GEO_H
            std = np.sqrt(np.trapezoid((f['E']-mean)**2, y, axis=0)/cfg.GEO_H)
            label = f'{row["method"]} / {row["protocol"]}'
            ax.plot(f['X'][0], mean, color=color, linestyle=style, lw=1.4, label=label)
            ax.fill_between(f['X'][0], mean-std, mean+std, color=color, alpha=.08)
        ax.set(xlabel='x (mm)', ylabel='Width-averaged E (MPa)')
        ax.legend(fontsize=8)
        fig.tight_layout()
        for ext in ('png', 'svg'):
            fig.savefig(output/f'{kind}_profile.{ext}', dpi=300, bbox_inches='tight')
        plt.close(fig)
    gallery(output, '④ 共同评价：二维模量、宽度均值及误差', rows)


PAGE_STYLE = ('<style>body{max-width:1150px;margin:35px auto;padding:0 20px;font:16px/1.7 '
              '"Microsoft YaHei",sans-serif;color:#17212b}a{color:#075fa2}img{max-width:100%}'
              'table{border-collapse:collapse}th,td{border:1px solid #ccd4dc;padding:6px 10px;'
              'text-align:right}code{font-size:14px}figure{margin:30px 0}</style>')


def gallery(output, title, rows=None):
    parts = ['<!doctype html><html lang="zh-CN"><meta charset="utf-8">', PAGE_STYLE,
             f'<title>{html.escape(title)}</title><h1>{html.escape(title)}</h1>',
             f'<p>{NOTICE}</p>']
    if rows:
        parts.append('<table><tr><th>真值</th><th>方法／初始界面</th><th>E 相对 RMS</th>'
                     '<th>轴向 RMSE (μm)</th><th>横向 RMSE (μm)</th><th>FE 总数</th><th>一阶证据</th></tr>')
        for r in rows:
            parts.append(f'<tr><td>{r["case"]}</td><td>{r["method"]}/{r["protocol"]}</td>'
                         f'<td>{100*r["modulus_relative_rms"]:.4f}%</td>'
                         f'<td>{1000*r["ux_rmse_mm"]:.5g}</td><td>{1000*r["uy_rmse_mm"]:.5g}</td>'
                         f'<td>{r["total_forward_evaluations"]}</td><td>{r["first_order_passed"]}</td></tr>')
        parts.append('</table><p>阴影为沿宽度的空间标准差，不是参数置信区间。'
                     '连续真值的候选界面无物理定位目标；位移拟合小不代表材料／几何唯一。</p>')
    for path in sorted(output.glob('*.png')):
        parts.append(f'<figure><figcaption>{html.escape(path.stem)}</figcaption>'
                     f'<a href="{path.with_suffix(".svg").name}"><img src="{path.name}" loading="lazy"></a></figure>')
    for path in sorted(output.glob('*.json')) + sorted(output.glob('*.csv')):
        parts.append(f'<p><a href="{path.name}">{html.escape(path.name)}</a></p>')
    (output/'云图.html').write_text('\n'.join(parts), encoding='utf-8')


def publish(root):
    parts = ['<!doctype html><html lang="zh-CN"><meta charset="utf-8">', PAGE_STYLE,
             '<title>统一梯度／分层材料算法诊断</title><h1>统一材料 P0/P1 结果总览</h1>',
             f'<p>{NOTICE}</p>', f'<p>保存位置：{html.escape(str(root))}</p>']
    for title, stage in [('① 合成输入', '01_prepared'), ('② 共用内核正演', '02_forward'),
                         ('④ 共同评价', '04_comparison')]:
        if (root/stage/'云图.html').exists():
            parts.append(f'<p><a href="{stage}/云图.html">{title}</a></p>')
    verification = root/'02_forward'/'verification.json'
    if verification.exists():
        parts.append('<p><a href="02_forward/verification.json">P0 独立核验记录</a></p>')
    parts.append('<h2>③ 每次识别与失败记录</h2><ul>')
    for page in sorted((root/'03_inverse').rglob('云图.html')):
        label = page.parent.relative_to(root).as_posix()
        parts.append(f'<li><a href="{label}/云图.html">{label}</a></li>')
    parts.append('</ul>')
    (root/'结果总览.html').write_text('\n'.join(parts), encoding='utf-8')
    (root/'index.html').write_text('<meta charset="utf-8"><meta http-equiv="refresh" '
                                  'content="0;url=结果总览.html">', encoding='utf-8')
    research = root.parent
    (research/'结果总览.html').write_text('\n'.join([
        '<!doctype html><meta charset="utf-8">'+PAGE_STYLE+'<h1>统一材料算法测试</h1>',
        *[f'<p><a href="{p.parent.name}/结果总览.html">{p.parent.name}</a></p>'
          for p in sorted(research.glob('*/结果总览.html'))]]), encoding='utf-8')
    print(f'保存位置：{root}\n结果总览：{root / "结果总览.html"}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'forward', 'inverse', 'compare'])
    parser.add_argument('--config', default='unified_gradient_layered_config')
    parser.add_argument('--input', type=Path, help='已有冻结运行根目录，显式使用原配置')
    parser.add_argument('--plot-only', action='store_true')
    parser.add_argument('--case', choices=['continuous_2d', 'layered', 'mixed_2d'])
    parser.add_argument('--method', choices=['continuous_q4', 'layered_constant', 'unified_q4'], default='unified_q4')
    parser.add_argument('--geometry', choices=['fixed', 'unknown'], default='fixed')
    parser.add_argument('--start', type=float, default=0.35, help='未知界面初始位置占域长比例')
    args = parser.parse_args()
    if args.stage == 'prepare' and not args.plot_only:
        if args.input:
            parser.error('prepare creates a new RUN_NAME; --input is for saved runs')
        prepare(load_python_configuration(args.config))
        return
    root, cfg = frozen_config(args.config, args.input)
    if args.stage == 'inverse' and args.case is None:
        parser.error('inverse requires --case')
    if args.plot_only:
        if args.stage == 'prepare':
            plot_preparation(root, cfg)
        elif args.stage == 'forward':
            plot_forward(root, cfg)
        elif args.stage == 'inverse':
            plot_inverse(inverse_folder(root, args.case, args.method, args.geometry, args.start), root, cfg)
        else:
            plot_comparison(root, cfg)
        publish(root)
    elif args.stage == 'forward':
        forward(root, cfg)
    elif args.stage == 'inverse':
        inverse(root, cfg, args.case, args.method, args.geometry, args.start)
    else:
        compare(root, cfg)


if __name__ == '__main__':
    main()
