"""Plot saved experimental inversions against layered and linear calibration.

Run: python -X utf8 -B -m lib.calibration_trend_comparison --refresh-existing
This renderer never calls a forward solver, an optimizer, or gamma selection.
"""

import argparse
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fgm_asm.mesh import MeshInfo
from fgm_asm.research import partition_modulus
from .calibration_trend import TREND_NOTE, add_calibration_trend
from .experiment_data import file_sha256
from .research_reporting import modulus_width_profile, save_x_profile
# Import existing renderers before setting this module's font context: the
# legacy preprocessing import configures a serif font globally.
from .dic_exact_tv_report import _profile
from .dic_380n_report import report as report_load380


ROOT = Path(__file__).resolve().parents[1]
FULL = ROOT/'results/dic_full'
OUTPUT = FULL/'calibration_trend_comparison'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(fig, output, stem):
    for extension in ('png', 'svg'):
        fig.savefig(output/f'{stem}.{extension}', dpi=300, bbox_inches='tight')
    plt.close(fig)


def references(ax, metadata, output=None):
    cfg = metadata['configuration']
    table = pd.DataFrame(metadata['calibration'])
    edges = np.r_[0., cfg['REGION_BOUNDARIES_X'], cfg['GEO_L']]
    ax.stairs(table.mean_E_MPa, edges, baseline=None, color='#333333', linestyle='--',
              linewidth=1.3, label='分层独立标定均值')
    ax.fill_between(np.repeat(edges, 2)[1:-1],
                    np.repeat(table.mean_E_MPa-table.sd_E_MPa, 2),
                    np.repeat(table.mean_E_MPa+table.sd_E_MPa, 2),
                    color='gray', alpha=.14, label='标定试样间 SD')
    return add_calibration_trend(ax, table, cfg['REGION_BOUNDARIES_X'], cfg['GEO_L'], output=output)


def method_label(cfg):
    mode = cfg['REGULARIZATION_MODE']
    name = {'gradient': 'H1', 'huber_tv': f'Huber-TV δ={cfg.get("TV_DELTA", 1):g}',
            'exact_tv': '精确 TV'}[mode]
    if list(cfg.get('DISPLACEMENT_COMPONENT_WEIGHTS', [1., 1.])) != [1., 1.]:
        name += '（横向平方残差权重10）'
    return f'{name} · {cfg["NEL_X"]}×{cfg["NEL_Y"]}'


def profile_records(*, refresh=False):
    records, statuses = [], []
    for path in sorted(FULL.rglob('01_prepared/run_metadata.json')):
        if 'synthetic' in path.parts:
            continue
        metadata = read(path)
        cfg = metadata['configuration']
        if cfg.get('DATA_MODE', 'dic') != 'dic' or len(metadata.get('calibration', [])) != 7:
            continue
        root = path.parent.parent
        inverse_path = root/'03_inverse/run_metadata.json'
        if not inverse_path.exists():
            continue
        inverse = read(inverse_path)
        label = method_label(cfg)
        platform = ', '.join(cfg['FIT_CASES'])
        statuses.append(dict(run=str(root.relative_to(ROOT)), platform=platform,
                             method=label, status=inverse['status'],
                             selected_gamma=inverse.get('selected_gamma')))
        if inverse['status'] not in ('computed', 'inverse_not_converged'):
            continue
        for field_path in sorted((root/'03_inverse').glob('nu_*/fields.npz')):
            nu = float(field_path.parent.name[3:])
            with np.load(field_path) as values:
                field, coordinates = values['E_inverse'].copy(), values['coordinates'].copy()
            mesh = MeshInfo(cfg['GEO_L'], cfg['GEO_H'], cfg['NEL_X'], cfg['NEL_Y'],
                            nodal_coordinates=coordinates)
            if field.shape != (mesh.n_nod,) or not np.isfinite(field).all() or np.any(field <= 0):
                raise ValueError(f'Invalid saved nodal E: {field_path}')
            reference = partition_modulus(pd.DataFrame(metadata['calibration']).mean_E_MPa,
                                          cfg['REGION_BOUNDARIES_X'])
            profile = modulus_width_profile(mesh, field, reference)
            if refresh:
                for stage in ('03_inverse', '04_comparison'):
                    destination = root/stage/field_path.parent.name
                    if not (destination/'x_profile.png').exists():
                        continue
                    save_x_profile(destination, mesh, field, reference, identified_using=platform,
                                   calibration=metadata['calibration'],
                                   region_boundaries=cfg['REGION_BOUNDARIES_X'])
            records.append(dict(root=root, metadata=metadata, inverse=inverse, cfg=cfg,
                                platform=platform, label=label, nu=nu, mesh=mesh,
                                field=field, profile=profile, source=field_path))
    return records, statuses


def add_page_note(path):
    """Idempotently explain the added reference beside existing result images."""
    if not path.exists():
        return
    text = path.read_text(encoding='utf-8')
    marker = '<p data-calibration-trend="linear">'
    if marker in text:
        return
    note = marker+html.escape(TREND_NOTE)+'</p>'
    position = text.find('<img ')
    if position >= 0:
        path.write_text(text[:position]+note+text[position:], encoding='utf-8')


def refresh_tv_panels(records):
    """Re-render existing TV summary panels from saved fields/tables only."""
    lookup = {r['root']: r for r in records if np.isclose(r['nu'], .26)}
    coarse = FULL/'exact_tv_coarse_pd_20261021'
    refined = FULL/'exact_tv_refine_242x38_20261021'
    specifications = [
        (coarse/'method_comparison', 'method_comparison', [FULL/'coarse', coarse],
         ['H1 · 121×19', '精确 TV · 121×19'], 'method', 'regional_comparison.csv',
         '180N：各方法独立选 γ；ν=0.26'),
        (refined/'mesh_diagnostic', 'mesh_comparison', [coarse, refined],
         ['精确 TV · 121×19', '精确 TV · 242×38（固定粗网格 γ）'], 'mesh',
         'common_regional_comparison.csv', '180N：固定粗网格 γ 的加密诊断；ν=0.26'),
        (refined/'mesh_reselected', 'mesh_reselected', [coarse, refined, refined],
         ['精确 TV · 121×19', '精确 TV · 242×38（重选 γ）', '精确 TV · 242×38（固定 γ 诊断）'],
         'mesh', 'common_regional_comparison.csv', '180N：独立重选 γ 与固定 γ 诊断；ν=0.26')]
    for output, stem, roots, labels, key, regional_csv, title in specifications:
        if not (output/f'{stem}.png').exists() or any(root not in lookup for root in roots):
            continue
        selected = [dict(lookup[root]) for root in roots]
        if stem == 'mesh_comparison':
            selected[1]['source'] = refined/'mesh_diagnostic/solution/fields.npz'
        if stem == 'mesh_reselected':
            selected[2]['source'] = refined/'mesh_diagnostic/solution/fields.npz'
        for record in selected:
            with np.load(record['source']) as values:
                record['field'] = values['E_inverse'].copy()
        region = pd.read_csv(output/regional_csv)
        if 'nu' in region:
            region = region[np.isclose(region.nu, .26)]
        keys = list(region[key].unique())
        fig, axes = plt.subplots(2, 2, figsize=(14, 8.2), layout='constrained')
        lo = min(r['field'].min() for r in selected)
        hi = max(r['field'].max() for r in selected)
        for ax, record, label in zip(axes[0], selected[:2], labels[:2]):
            mesh = record['mesh']
            image = ax.pcolormesh(mesh.plot_x, mesh.plot_y,
                                 record['field'].reshape(mesh.nods_y, mesh.nods_x),
                                 shading='gouraud', cmap='viridis', vmin=lo, vmax=hi)
            ax.set(title=label, xlabel='x (mm)', ylabel='y (mm)', aspect='equal')
        fig.colorbar(image, ax=list(axes[0]), label='E (MPa)', shrink=.7)
        references(axes[1, 0], selected[0]['metadata'], output)
        colors = ('#0072B2', '#D55E00', '#009E73')
        # Use the same physical-width moment as the existing TV reports.
        for i, (record, label, color) in enumerate(zip(selected, labels, colors)):
            mean, sd = _profile(record['mesh'], record['field'])
            x = record['mesh'].plot_x[0]
            axes[1, 0].plot(x, mean, color=color, linestyle='--' if i == 2 else '-', label=label)
            if i < 2:
                axes[1, 0].fill_between(x, mean-sd, mean+sd, color=color, alpha=.12)
            subset = region[region[key] == keys[i]]
            axes[1, 1].bar(np.arange(7)+(i-(len(selected)-1)/2)*.25,
                           subset.difference_percent, width=.25, color=color, label=label)
        axes[1, 0].set(xlabel='x (mm)', ylabel='宽度平均 E (MPa)',
                       title='分层标定、标定线性趋势与识别场')
        axes[1, 0].legend(fontsize=7.5)
        axes[1, 1].axhline(0, color='black', linewidth=.7)
        axes[1, 1].set(xticks=np.arange(7), xticklabels=[f'{p:g}%' for p in
                       pd.DataFrame(selected[0]['metadata']['calibration']).infill_percent],
                       xlabel='制造填充率（仅事后统计）', ylabel='区域均值相对分层标定差异 (%)')
        axes[1, 1].legend(fontsize=7.5)
        fig.suptitle(title+'；色带为横向空间 SD，灰带为标定试样间 SD', fontsize=11)
        save(fig, output, stem)
        add_page_note(output/'index.html')


def refresh_platform_panels(records):
    output = FULL/'independent_platforms_20261022'
    groups = dict(coarse=[('180N', 121), ('280N', 121)],
                  fine=[('180N', 367), ('280N', 367)],
                  coarse_three_platforms=[('180N', 121), ('280N', 121), ('380N', 121)],
                  **{'380N': [('380N', 121)]})
    for name, choices in groups.items():
        destination = output/name
        if not (destination/'independent_modulus_profiles.png').exists():
            continue
        fig, axes = plt.subplots(2, 1, figsize=(10, 6.7), sharex=True, layout='constrained')
        metadata = None
        for platform, nx in choices:
            record = next(r for r in records if r['platform'] == platform and
                          r['cfg']['NEL_X'] == nx and r['cfg']['REGULARIZATION_MODE'] == 'gradient'
                          and list(r['cfg'].get('DISPLACEMENT_COMPONENT_WEIGHTS', [1., 1.])) == [1., 1.]
                          and np.isclose(r['nu'], .26))
            metadata = record['metadata']
            table = record['profile']
            color = {'180N': '#0072B2', '280N': '#D55E00', '380N': '#009E73'}[platform]
            x, mean, sd = table.x_mm, table.inverse_width_mean_E_MPa, table.inverse_transverse_sd_MPa
            axes[0].plot(x, mean, color=color, label=f'{platform} DIC 独立识别')
            axes[0].fill_between(x, mean-sd, mean+sd, color=color, alpha=.12)
            axes[1].plot(x, sd, color=color, label=platform)
        references(axes[0], metadata, destination)
        axes[0].set(ylabel='宽度平均 E (MPa)')
        axes[1].set(xlabel='x (mm)', ylabel='横向空间 SD (MPa)', xlim=(0, metadata['configuration']['GEO_L']))
        for ax in axes:
            ax.legend(fontsize=9)
        if any(p == '380N' for p, _ in choices):
            fig.suptitle('各平台独立识别；380N 暂取净力 319.9 N，非同步实测力', fontsize=11)
        save(fig, destination, 'independent_modulus_profiles')
    if (output/'mesh_comparison_profiles.png').exists():
        fig, axes = plt.subplots(2, 1, figsize=(10, 7), layout='constrained')
        for ax, platform in zip(axes, ('180N', '280N')):
            for nx, style in ((121, '--'), (367, '-')):
                record = next(r for r in records if r['platform'] == platform and
                              r['cfg']['NEL_X'] == nx and r['cfg']['REGULARIZATION_MODE'] == 'gradient'
                              and list(r['cfg'].get('DISPLACEMENT_COMPONENT_WEIGHTS', [1., 1.])) == [1., 1.]
                              and np.isclose(r['nu'], .26))
                table = record['profile']
                ax.plot(table.x_mm, table.inverse_width_mean_E_MPa, linestyle=style,
                        color='#0072B2' if nx == 121 else '#D55E00', label=f'{platform} · {nx}×{record["cfg"]["NEL_Y"]}')
            references(ax, record['metadata'])
            ax.set(xlabel='x (mm)', ylabel='宽度平均 E (MPa)')
            ax.legend(fontsize=9)
        save(fig, output, 'mesh_comparison_profiles')
    add_page_note(output/'index.html')


def build(*, refresh_existing=False):
    OUTPUT.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family': 'Microsoft YaHei', 'font.size': 10,
                         'axes.unicode_minus': False, 'svg.fonttype': 'none'})
    # Protect original numerical evidence independently of plot regeneration.
    protected = [p for p in FULL.rglob('*') if p.is_file() and
                 (p.suffix == '.npz' or p.name in ('run_metadata.json', 'selection.csv',
                  'selection.json', 'comparison.csv', 'regional_comparison.csv', 'summary.json'))]
    before = {str(p.relative_to(ROOT)): file_sha256(p) for p in protected}
    print(f'Protected {len(before)} numerical files; rendering saved profiles.', flush=True)
    records, statuses = profile_records(refresh=refresh_existing)
    print(f'Loaded {len(records)} final fields across {len(statuses)} experimental runs.', flush=True)
    page = ['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>反演与分层标定趋势对比</title>',
            '<style>body{max-width:1300px;margin:30px auto;padding:20px;font:16px/1.7 Microsoft YaHei}img{max-width:100%}table{border-collapse:collapse}td,th{padding:6px;border-bottom:1px solid #ddd}</style>',
            '<h1>各算法反演与分层标定趋势对比</h1><p>'+TREND_NOTE+'</p>',
            '<p>趋势沿用既有混合批次 independent 口径：七组均值等权拟合 E=a+b p。'
            'p(x) 由原六个材料界面与相邻填充率中值确定；不按 ROI 两端重新等分，不在 40%—100%以外外推。'
            '该连续坐标只是趋势比较约定，不代表试样实际连续制造。跨批桥接、物理配准和边界近似限制仍保留。</p>',
            '<p>按识别载荷分别展示，ν=0.26；γ 均保留原选参结果。色带为横向空间离散，不能作为置信区间。'
            '380N 使用假定净力319.9 N；不同方法的网格、权重和预算见图例及原结果，不能据此作严格同条件排名。</p>']
    profile_rows = []
    for platform in ('180N', '280N', '380N'):
        current = [r for r in records if r['platform'] == platform and np.isclose(r['nu'], .26)]
        if not current:
            continue
        rows = (len(current)+1)//2
        fig, axes = plt.subplots(rows, 2, figsize=(14, rows*3.55), sharex=True, sharey=True,
                                  layout='constrained', squeeze=False)
        for ax, record in zip(axes.flat, current):
            references(ax, record['metadata'], OUTPUT)
            table = record['profile']
            x, mean, sd = table.x_mm, table.inverse_width_mean_E_MPa, table.inverse_transverse_sd_MPa
            ax.plot(x, mean, color='#0072B2', label='反演宽度均值')
            ax.fill_between(x, mean-sd, mean+sd, color='#0072B2', alpha=.15)
            ax.set(title=f'{record["label"]}；γ={record["inverse"]["selected_gamma"]:.3g}',
                   xlabel='x (mm)', ylabel='E (MPa)', xlim=(0, record['mesh'].geo_l))
            ax.legend(fontsize=7.5)
            for row in table.to_dict('records'):
                profile_rows.append(dict(platform=platform, method=record['label'], **row))
        for ax in list(axes.flat)[len(current):]:
            ax.axis('off')
        fig.suptitle(f'{platform} DIC 独立识别与分层标定趋势；ν=0.26', fontsize=13)
        save(fig, OUTPUT, platform+'_method_trends')
        page += [f'<h2>{platform}</h2><img src="{platform}_method_trends.png">',
                 f'<p><a href="{platform}_method_trends.svg">矢量图 SVG</a></p>']
    page += ['<h2>全部实验方案的保存状态</h2>', pd.DataFrame(statuses).to_html(index=False),
             '<p>无有效 γ 的方案未补指定正式识别曲线。合成真值测试与历史不同域结果未混入本实验趋势比较。</p>']
    if refresh_existing:
        print('Refreshing saved TV, platform and load-comparison panels.', flush=True)
        refresh_tv_panels(records)
        refresh_platform_panels(records)
        report_load380(plot_only=True)
        for record in records:
            for stage, name in [('03_inverse', '云图.html'), ('04_comparison', '打开查看结果.html')]:
                add_page_note(record['root']/stage/name)
        add_page_note(FULL/'load380_20261022/index.html')
    spaces = FULL/'load380_20261022/comparison/space_profiles.png'
    if spaces.exists():
        page += ['<h2>节点／单元模量与测试函数的五方法对照</h2>',
                 '<p>42×8位移单元；180N和380N分别扫描选参。失败方案保留原注记，'
                 'P0曲线保持单元常值，不作平滑替换。</p>',
                 '<img src="../load380_20261022/comparison/space_profiles.png">',
                 '<p><a href="../load380_20261022/comparison/space_profiles.svg">矢量图 SVG</a> · '
                 '<a href="../load380_20261022/index.html">完整380N方法、状态与载荷对照</a></p>']
    # Known-interface unified prototype has no selected gamma; keep its status explicit.
    diagnostic = ROOT/'results/unified_gradient_layered/380n_coarse_lcurve_20261023/04_comparison'
    paths = [diagnostic/f'gamma_{index:02d}_profile.csv' for index in (0, 24, 49)]
    if all(p.exists() for p in paths):
        metadata = next(r['metadata'] for r in records if r['platform'] == '380N')
        fig, ax = plt.subplots(figsize=(10, 5.5), layout='constrained')
        references(ax, metadata, diagnostic if refresh_existing else None)
        for index, path, color in zip((0, 24, 49), paths, ('#0072B2', '#D55E00', '#009E73')):
            table = pd.read_csv(path)
            ax.plot(table.x_mm, table.mean_E_MPa, color=color, label=f'预定诊断 #{index:02d}（未选 γ）')
        ax.set(xlabel='x (mm)', ylabel='宽度平均 E (MPa)',
               title='统一分区 Q4 原型：0/50 通过一阶条件；曲线为失败诊断')
        ax.legend(fontsize=9)
        save(fig, OUTPUT, 'unified_unselected_diagnostics')
        if refresh_existing:
            # Replace only the existing width-mean figure; preserve fields and failure evidence.
            import shutil
            for extension in ('png', 'svg'):
                shutil.copyfile(OUTPUT/f'unified_unselected_diagnostics.{extension}',
                                diagnostic/f'width_profile.{extension}')
            add_page_note(diagnostic.parent/'index.html')
        page += ['<h2>统一分区算法：未选参诊断</h2><p>以下三条均为预先指定的失败诊断，'
                 '不与已选参结果作精度排名；不改变仿真进入条件未通过的结论。</p>',
                 '<img src="unified_unselected_diagnostics.png">']
    pd.DataFrame(profile_rows).to_csv(OUTPUT/'method_width_profiles.csv', index=False)
    pd.DataFrame(statuses).to_csv(OUTPUT/'method_status.csv', index=False, encoding='utf-8-sig')
    after = {str(p.relative_to(ROOT)): file_sha256(p) for p in protected}
    if before != after:
        raise AssertionError('A protected numerical result changed during rendering')
    verification = dict(protected_files=len(before), numerical_evidence_unchanged=True,
                        refreshed_profile_figures=2*len(records) if refresh_existing else 0,
                        runs=len(statuses), final_fields=len(records), source_sha256=before,
                        command='python -X utf8 -B -m lib.calibration_trend_comparison --refresh-existing')
    (OUTPUT/'verification.json').write_text(json.dumps(verification, ensure_ascii=False, indent=2), encoding='utf-8')
    page += ['<p><a href="verification.json">数值文件身份核验</a> · '
             '<a href="calibration_trend.json">趋势定义与系数</a> · '
             '<a href="calibration_trend_anchors.csv">标定来源、样本数、SD 与残差</a></p></html>']
    (OUTPUT/'index.html').write_text('\n'.join(page), encoding='utf-8')
    project_page = ROOT/'results/index.html'
    text = project_page.read_text(encoding='utf-8')
    if 'id="calibration-trend-comparison"' not in text:
        link = '<section id="calibration-trend-comparison"><h2><a href="dic_full/calibration_trend_comparison/index.html">各算法反演与分层标定趋势对比 →</a></h2></section>'
        text = text.replace('<h1>项目计算结果</h1>', '<h1>项目计算结果</h1>\n'+link)
        project_page.write_text(text, encoding='utf-8')
    print(json.dumps({k: v for k, v in verification.items() if k != 'source_sha256'}, ensure_ascii=False))
    print((OUTPUT/'index.html').resolve())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--refresh-existing', action='store_true', help='Refresh saved modulus/profile figures only')
    build(refresh_existing=parser.parse_args().refresh_existing)
