"""Compare independently identified load platforms using saved numerical fields."""

from pathlib import Path
import html
import json
import os

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd

from .dic_case_workflow import load_prepared
from .experiment_data import file_sha256
from .research_reporting import modulus_width_profile


def plot_identification_workflow(stem: Path) -> None:
    """Document separate parameter selection and modulus recovery for each load."""
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(13.5, 5.2))
    ax.set(xlim=(0, 13.5), ylim=(0, 5.2))
    ax.axis('off')

    def box(x, y, width, label, color):
        patch = FancyBboxPatch((x, y), width, .75, boxstyle='round,pad=0.08',
                               edgecolor=color, facecolor='white', linewidth=1.4)
        ax.add_patch(patch)
        ax.text(x+width/2, y+.375, label, ha='center', va='center', fontsize=12)

    def arrow(start, end, color='#4b5563'):
        ax.annotate('', xy=end, xytext=start,
                    arrowprops=dict(arrowstyle='->', color=color, lw=1.4))

    ax.text(6.75, 4.92, 'Independent inversion of 180N and 280N DIC data',
            ha='center', va='center', fontsize=17)
    box(.15, 3.72, 4.9, 'Independent material calibration\nE for each region', '#555555')
    box(5.7, 3.72, 7.15, 'Displacement from Calibrated E\ncompare with 180N and 280N DIC', '#555555')
    arrow((5.13, 4.095), (5.62, 4.095))
    for y, platform, color in ((2.38, '180N', '#0072b2'), (.97, '280N', '#d55e00')):
        box(.15, y, 1.75, f'{platform} DIC Data', color)
        box(2.55, y, 2.5, 'Own L-curve\nselect gamma', color)
        box(5.7, y, 2.4, f'Identify E(x,y)\nusing {platform} DIC', color)
        box(8.75, y, 4.1, f'Displacement from Identified E\ncompare with {platform} DIC', color)
        for start, end in ((1.98, 2.47), (5.13, 5.62), (8.18, 8.67)):
            arrow((start, y+.375), (end, y+.375), color)
    ax.text(6.75, .25,
            'Repeat independently on both meshes: coarse 121 x 19; fine 367 x 64.\n'
            'Same 23,920 DIC points. Calibration is used for forward comparison and evaluation.',
            ha='center', va='center', fontsize=11)
    fig.tight_layout()
    for extension in ('png', 'svg', 'pdf'):
        fig.savefig(stem.with_suffix('.'+extension), dpi=300, bbox_inches='tight')
    plt.close(fig)
    svg = stem.with_suffix('.svg')
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text(encoding='utf-8').splitlines())+'\n',
                   encoding='utf-8')


def _save_figure(fig, output: Path, stem: str) -> None:
    for extension in ('png', 'svg'):
        fig.savefig(output/f'{stem}.{extension}', dpi=300, bbox_inches='tight')
    plt.close(fig)


def _load_identification(root: Path, platform: str):
    root = Path(root).resolve()
    inverse = json.loads((root/'03_inverse/run_metadata.json').read_text(encoding='utf-8'))
    if inverse['status'] not in ('computed', 'inverse_not_converged'):
        raise ValueError(f'Incomplete independent inversion: {root}')
    cfg, mesh, cases, metadata, _ = load_prepared(root/'01_prepared', saved_fields_only=True)
    if list(cfg.FIT_CASES) != [platform] or inverse['initialization']['fit_cases'] != [platform]:
        raise ValueError(f'{root} did not independently identify E using {platform}')
    case = next(c for c in cases if c.name == platform)
    nu_directory = f'nu_{cfg.PRIMARY_NU:.2f}'
    with np.load(root/'03_inverse'/nu_directory/'fields.npz') as values:
        field = values['E_inverse'].copy()
        displacement = values[f'{platform}_prediction'].reshape(-1, 2).copy()
    with np.load(root/'02_forward'/nu_directory/'fields.npz') as values:
        forward = values[f'{platform}_prediction'].reshape(-1, 2).copy()
    if field.shape != (mesh.n_nod,) or not np.isfinite(field).all() or np.any(field <= 0):
        raise ValueError(f'Invalid identified nodal E: {root}')
    return dict(root=root, cfg=cfg, mesh=mesh, case=case, metadata=metadata,
                inverse=inverse, E=field, displacement=displacement, forward=forward)


def _plot_mesh_comparison(output: Path, records: list[dict], *, modulus_limits=None) -> None:
    """Use a shared E range, and a common displacement scale per load/component."""
    output.mkdir(exist_ok=True, parents=True)
    mesh = records[0]['mesh']
    calibration = pd.DataFrame(records[0]['metadata']['calibration'])
    from fgm_asm.research import partition_modulus
    reference = partition_modulus(calibration.mean_E_MPa, records[0]['cfg'].REGION_BOUNDARIES_X)
    reference_values = reference(mesh.X, mesh.Y)
    values = [reference_values, *(r['E'] for r in records)]
    lo, hi = min(v.min() for v in values), max(v.max() for v in values)
    if modulus_limits is not None:
        lo, hi = modulus_limits
    fig, axes = plt.subplots(3, 1, figsize=(12, 6), constrained_layout=True)
    for ax, field, title in zip(axes, values,
            ('Independent calibrated E', 'E identified using 180N DIC', 'E identified using 280N DIC')):
        im = ax.pcolormesh(mesh.plot_x, mesh.plot_y, field.reshape(mesh.nods_y, mesh.nods_x),
                           shading='auto', cmap='viridis', vmin=lo, vmax=hi, rasterized=True)
        ax.set(title=title, xlabel='x (mm)', ylabel='y (mm)', aspect='equal',
               xlim=(0, mesh.geo_l), ylim=(0, mesh.geo_h))
    fig.colorbar(im, ax=axes, label='E (MPa)', shrink=.85)
    _save_figure(fig, output, 'independent_modulus_fields')

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True, constrained_layout=True)
    profiles = []
    for record, color in zip(records, ('#0072b2', '#d55e00')):
        platform = record['cfg'].FIT_CASES[0]
        profile = modulus_width_profile(mesh, record['E'], reference)
        x = profile.x_mm.to_numpy()
        mean = profile.inverse_width_mean_E_MPa.to_numpy()
        sd = profile.inverse_transverse_sd_MPa.to_numpy()
        axes[0].plot(x, mean, color=color, label=f'E identified using {platform} DIC')
        axes[0].fill_between(x, mean-sd, mean+sd, color=color, alpha=.12)
        axes[1].plot(x, sd, color=color, label=platform)
        profiles.extend(dict(platform=platform, x_mm=float(a), mean_E_MPa=float(b), sd_E_MPa=float(c))
                        for a, b, c in zip(x, mean, sd))
    dense_x = np.linspace(0, mesh.geo_l, 2400)
    axes[0].plot(dense_x, reference(dense_x, np.zeros_like(dense_x)), 'k--', label='Independent calibration')
    axes[0].set(ylabel='Width-mean E (MPa)')
    axes[1].set(xlabel='x (mm)', ylabel='Transverse spatial SD (MPa)')
    for ax in axes:
        ax.set_xlim(0, mesh.geo_l)
        ax.legend(fontsize=10)
    _save_figure(fig, output, 'independent_modulus_profiles')
    pd.DataFrame(profiles).to_csv(output/'width_profiles.csv', index=False)

    from .dic_preprocessing import grid_fields_from_points
    fig, axes = plt.subplots(3, 4, figsize=(17, 7), constrained_layout=True)
    for record_index, record in enumerate(records):
        case = record['case']
        xy = np.column_stack([case.observation.x_meas, case.observation.y_meas])
        observed = case.observation.u_measured.reshape(-1, 2)
        with np.load(record['root']/'01_prepared/fields.npz') as fields:
            ids = fields[f'{case.name}_grid_ids']
        maps = {label: array-observed.mean(axis=0) for label, array in
                (('dic', observed), ('forward', record['forward']), ('inverse', record['displacement']))}
        for component in range(2):
            column = record_index*2+component
            low = min(array[:, component].min() for array in maps.values())
            high = max(array[:, component].max() for array in maps.values())
            for row, (key, title) in enumerate((('dic', 'DIC Data'),
                    ('forward', 'Displacement from\nCalibrated E'), ('inverse', 'Displacement from\nIdentified E'))):
                x, y, grid = grid_fields_from_points(xy, {'u': maps[key][:, component]}, ids)
                im = axes[row, column].pcolormesh(x, y, grid['u'], cmap='jet',
                    shading='nearest', vmin=low, vmax=high, rasterized=True)
                axes[row, column].set(title=f'{case.name}: $u_{"xy"[component]}$\n{title}',
                                      xlabel='x (mm)', ylabel='y (mm)', aspect='equal')
                axes[row, column].title.set_fontsize(11)
            fig.colorbar(im, ax=axes[:, column], label=f'u{"xy"[component]} (mm)', shrink=.8)
    _save_figure(fig, output, 'independent_displacement_comparison')


def write_independent_comparison(runs: dict, output: Path, *, allow_incomplete: bool = False) -> dict:
    """Render saved coarse/fine 180N/280N identifications; never launch solves."""
    mesh_labels = {'coarse': '粗网格', 'fine': '细网格', 'both': '粗、细网格'}
    output = Path(output).resolve()
    output.mkdir(exist_ok=True, parents=True)
    plot_identification_workflow(output/'independent_platform_workflow')
    rows, sources, completed, mesh_records = [], {}, [], {}
    for label, platforms in runs.items():
        ready = all(all((Path(root)/'04_comparison'/name).exists() for name in
                        ('run_metadata.json', 'comparison.csv', '打开查看结果.html'))
                    for root in platforms.values())
        if not ready:
            if allow_incomplete:
                continue
            raise ValueError(f'{label}: finish both independent four-step runs first')
        records = [_load_identification(Path(platforms[name]), name) for name in ('180N', '280N')]
        left, right = records
        for key in ('GEO_L', 'GEO_H', 'THICKNESS', 'PRIMARY_NU', 'REGULARIZATION_MODE', 'FIT_CASES'):
            if key != 'FIT_CASES' and getattr(left['cfg'], key) != getattr(right['cfg'], key):
                raise ValueError(f'{label}: inconsistent {key}')
        pd.testing.assert_frame_equal(pd.DataFrame(left['metadata']['calibration']),
                                      pd.DataFrame(right['metadata']['calibration']), check_exact=True)
        np.testing.assert_array_equal(getattr(left['cfg'], 'DISPLACEMENT_COMPONENT_WEIGHTS', (1., 1.)),
                                      getattr(right['cfg'], 'DISPLACEMENT_COMPONENT_WEIGHTS', (1., 1.)))
        for platform in ('180N', '280N'):
            for key in ('mean_corrected_force_N', 'model_end_force_N', 'load_section_width_mm'):
                if left['metadata']['force_pairing'][platform][key] != right['metadata']['force_pairing'][platform][key]:
                    raise ValueError(f'{label}: inconsistent {platform} {key}')
        np.testing.assert_array_equal(left['mesh'].coord, right['mesh'].coord)
        nu_directory = f'nu_{left["cfg"].PRIMARY_NU:.2f}'
        with np.load(left['root']/'02_forward'/nu_directory/'fields.npz') as calibrated:
            for record in records:
                platform = record['case'].name
                np.testing.assert_allclose(record['forward'], calibrated[f'{platform}_prediction'].reshape(-1, 2),
                                           rtol=1e-8, atol=1e-10)
        with np.load(left['root']/'01_prepared/fields.npz') as a, np.load(right['root']/'01_prepared/fields.npz') as b:
            if set(a.files) != set(b.files):
                raise ValueError('Independent runs must preserve the same prepared arrays')
            for key in a.files:
                np.testing.assert_array_equal(a[key], b[key])
        for record in records:
            platform = record['cfg'].FIT_CASES[0]
            metrics = pd.read_csv(record['root']/'04_comparison/comparison.csv')
            for metric in metrics[metrics.nu == record['cfg'].PRIMARY_NU].to_dict(orient='records'):
                # Calibration is one fixed reference, independent of either identification.
                inverse_model = metric['model'] == 'inverse'
                if not inverse_model and platform != '180N':
                    continue
                metric.update(mesh=label,
                              identified_using=platform if inverse_model else 'independent_calibration',
                              evaluation=('same_load' if metric['case'] == platform else 'cross_load')
                                         if inverse_model else 'calibration_forward',
                              selected_gamma=record['inverse']['selected_gamma'] if inverse_model else None)
                if not inverse_model:
                    metric['role'] = 'calibration_reference'
                rows.append(metric)
            source = record['root']/'03_inverse/run_metadata.json'
            sources[f'{label}/{platform}'] = dict(root=str(record['root']), sha256=file_sha256(source),
                prepared_fields_sha256=file_sha256(record['root']/'01_prepared/fields.npz'),
                identified_fields_sha256=file_sha256(record['root']/'03_inverse'/f'nu_{record["cfg"].PRIMARY_NU:.2f}'/'fields.npz'),
                forward_fields_sha256=file_sha256(record['root']/'02_forward'/f'nu_{record["cfg"].PRIMARY_NU:.2f}'/'fields.npz'),
                nu=record['cfg'].PRIMARY_NU,
                gamma=record['inverse']['selected_gamma'], status=record['inverse']['status'])
        mesh_records[label] = records
        completed.append(label)
    color_limits = None
    if mesh_records:
        all_records = [record for records in mesh_records.values() for record in records]
        color_limits = (
            min(min(float(record['E'].min()), min(r['mean_E_MPa'] for r in record['metadata']['calibration']))
                for record in all_records),
            max(max(float(record['E'].max()), max(r['mean_E_MPa'] for r in record['metadata']['calibration']))
                for record in all_records))
        for label, records in mesh_records.items():
            _plot_mesh_comparison(output/label, records, modulus_limits=color_limits)
    summary = dict(status='computed' if len(completed) == len(runs) else 'running',
                   protocol='independent_platform_identification', completed_meshes=completed,
                   modulus_color_limits_MPa=color_limits,
                   renderer_sha256=file_sha256(Path(__file__)),
                   sources=sources, runs={k: {p: str(Path(v).resolve()) for p, v in platforms.items()}
                                          for k, platforms in runs.items()})
    (output/'run_metadata.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    metrics = pd.DataFrame(rows)
    metrics.to_csv(output/'comparison.csv', index=False)
    if len(completed) == len(runs) and len(completed) > 1:
        fig, axes = plt.subplots(2, 1, figsize=(10, 7), constrained_layout=True)
        for label, linestyle in zip(completed, ('--', '-')):
            profile = pd.read_csv(output/label/'width_profiles.csv')
            for ax, platform, color in zip(axes, ('180N', '280N'), ('#0072b2', '#d55e00')):
                group = profile[profile.platform == platform]
                ax.plot(group.x_mm, group.mean_E_MPa, color=color, linestyle=linestyle,
                        label=f'{platform} DIC: {label}')
        for ax in axes:
            ax.set(xlabel='x (mm)', ylabel='Width-mean E (MPa)')
            ax.legend(fontsize=10)
        _save_figure(fig, output, 'mesh_comparison_profiles')
    page = ['<!doctype html><meta charset="utf-8"><title>两平台独立反演：粗细网格</title>',
            '<style>body{max-width:1450px;margin:30px auto;font:16px/1.7 Microsoft YaHei}img{max-width:100%}td,th{padding:6px;border-bottom:1px solid #ddd}table{border-collapse:collapse}</style>',
            '<h1>180N、280N 分别独立反演：粗细网格</h1>',
            '<p>每个平台只用自己的DIC位移初始化、扫描50个γ并识别自由二维E。主ν=0.26；四个ν为固定本平台γ后的敏感性比较。独立标定不进入目标或初值。</p>',
            '<p>同平台位移为回代；将识别模量用于另一载荷称为跨载荷对照。两平台均已参与各自识别，本页不将280N称为未使用的独立验证数据。</p>',
            f'<p>汇总状态：{"已生成" if summary["status"] == "computed" else "计算进行中"}。完成网格：{html.escape("、".join(mesh_labels.get(label, label) for label in completed))}。</p>',
            '<img src="independent_platform_workflow.png">']
    for label, platforms in runs.items():
        page.append(f'<h2>{html.escape(mesh_labels.get(label, label))}</h2>')
        for platform, root in platforms.items():
            target = Path(root)/'04_comparison/打开查看结果.html'
            link = Path(os.path.relpath(target, output)).as_posix()
            page.append(f'<p><a href="{link}">{platform} 独立反演完整结果</a></p>' if target.exists()
                        else f'<p>{platform} 独立反演：计算进行中。</p>')
        if label not in completed:
            page.append('<p>计算进行中，尚未生成本网格的双平台比较结论。</p>')
            continue
        for stem in ('independent_modulus_fields', 'independent_modulus_profiles', 'independent_displacement_comparison'):
            page.append(f'<p><a href="{label}/{stem}.svg">SVG</a></p><img src="{label}/{stem}.png">')
    if len(completed) == len(runs) and len(completed) > 1:
        page.append('<h2>同平台的粗细网格宽度平均模量对照</h2>'
                    '<p><a href="mesh_comparison_profiles.svg">SVG</a></p><img src="mesh_comparison_profiles.png">')
    if not metrics.empty:
        display = metrics.replace({'mesh': mesh_labels,
                                   'role': {'fit': '本平台识别/回代', 'prediction': '跨载荷对照',
                                            'calibration_reference': '独立标定参照'},
                                   'identified_using': {'independent_calibration': '独立标定'},
                                   'model': {'forward': '标定模量', 'inverse': '识别模量'},
                                   'evaluation': {'same_load': '本平台回代', 'cross_load': '跨载荷对照',
                                                  'calibration_forward': '独立标定正演'}})
        display = display.rename(columns={'mesh': '网格', 'case': '位移载荷', 'model': '正演模量',
                                          'identified_using': 'E来源', 'role': '用途',
                                          'evaluation': '位移比较类型', 'selected_gamma': 'γ',
                                          'rmse_ux_mm': '轴向RMSE(mm)', 'rmse_uy_mm': '横向RMSE(mm)'})
        page.append('<h2>主ν下的误差与各自所选γ</h2>'+display.to_html(index=False, border=0, na_rep='—'))
    page.append('<p>优化停止、材料一致性与物理验证分别评价。几何配准、截断边界和跨批标定转移限制沿用原记录。</p>')
    (output/'index.html').write_text('\n'.join(page), encoding='utf-8')
    manifest = [dict(mesh='both', content='两平台独立识别流程', path=str(output/'independent_platform_workflow.png'))]
    for label in completed:
        for stem, content in (('independent_modulus_fields', '独立标定与两平台二维模量'),
                              ('independent_modulus_profiles', '两平台E(x)与独立标定'),
                              ('independent_displacement_comparison', '两平台DIC、独立标定正演与本平台回代')):
            manifest.append(dict(mesh=label, content=content, path=str(output/label/f'{stem}.png')))
        for platform, root in runs[label].items():
            root = Path(root).resolve()
            nu = sources[f'{label}/{platform}']['nu']
            for relative, content in (
                    ('03_inverse/l_curve.png', '本平台L-curve'),
                    (f'03_inverse/nu_{nu:.2f}/inverse_modulus_E.png', '本平台独立识别二维E'),
                    (f'04_comparison/nu_{nu:.2f}/{platform}/{platform}_forward_dic_components.png', 'DIC与独立标定正演位移'),
                    (f'04_comparison/nu_{nu:.2f}/{platform}/{platform}_inverse_dic_components.png', 'DIC与本平台识别模量计算的位移'),
                    (f'04_comparison/nu_{nu:.2f}/modulus_comparison.png', '本平台E与独立标定差异'),
                    (f'04_comparison/nu_{nu:.2f}/x_profile.png', '本平台宽度平均E与横向离散')):
                path = root/relative
                if path.exists():
                    manifest.append(dict(mesh=label, platform=platform, content=content, path=str(path)))
    if (output/'mesh_comparison_profiles.png').exists() and summary['status'] == 'computed':
        manifest.append(dict(mesh='both', content='同平台粗细网格E(x)对照', path=str(output/'mesh_comparison_profiles.png')))
    pd.DataFrame(manifest).to_csv(output/'ppt_image_manifest.csv', index=False, encoding='utf-8-sig')
    guide = ['# PPT图片清单', '', '每张PNG同目录下均有同名SVG。模量来源与绘图载荷分别标注。', '',
             '建议正文依次使用：独立识别流程图 → 粗、细网格的二维E和E(x) → DIC、独立标定正演与本平台回代位移 → 同平台粗细网格E(x)对照。L-curve及单个平台分图用于附录或放大展示。', '',
             '清单默认主ν=0.26；其他ν及跨载荷对照见各组完整结果入口。', '',
             '| 网格 | 识别平台 | 内容 | 图片 |', '| --- | --- | --- | --- |']
    for item in manifest:
        path = Path(item['path'])
        guide.append(f'| {mesh_labels.get(item["mesh"], item["mesh"])} | {item.get("platform", "两平台")} | {item["content"]} | [{path.name}]({path.as_posix()}) |')
    (output/'PPT图片清单.md').write_text('\n'.join(guide)+'\n', encoding='utf-8')
    return summary
