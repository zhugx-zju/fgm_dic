"""Reader-facing comparison of one prepared DIC case and its saved FE results."""

import html
import json
import os
from pathlib import Path
from urllib.parse import quote

import pandas as pd


PAGE_STYLE = '''<style>
body{max-width:1400px;margin:28px auto;padding:0 24px;font:16px/1.65 "Microsoft YaHei",sans-serif;color:#17212b;background:#fafbfc}
a{color:#075fa2}h1{font-size:28px}h2{margin-top:32px}.cards,.fields{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}
.card,figure{margin:0;background:white;padding:16px;border:1px solid #dce1e6;border-radius:8px}.card strong{display:block}.path,code{overflow-wrap:anywhere;font:13px/1.6 Consolas,monospace}
table{border-collapse:collapse;font-size:14px;display:block;overflow:auto}th,td{padding:6px 10px;border-bottom:1px solid #ddd}img{width:100%;height:auto}pre{white-space:pre-wrap}
@media(max-width:800px){.cards,.fields{grid-template-columns:1fr}}
</style>'''


def _link(path, base):
    return quote(Path(os.path.relpath(Path(path).resolve(), Path(base).resolve())).as_posix(), safe='/.:')


def publish_stage(case_root, stage, output, summary, project_root=None):
    """Update generated navigation only; scientific results remain immutable.

    Four independent scripts call this after saving their own output. The
    directory is for browsing, never an implicit input to a later solve.
    """
    project_root = Path(project_root or Path(__file__).resolve().parents[1]).resolve()
    case_root, output = Path(case_root).resolve(), Path(output).resolve()
    case_root.mkdir(parents=True, exist_ok=True)
    catalog = case_root/'结果目录.json'
    record = json.loads(catalog.read_text(encoding='utf-8')) if catalog.exists() else {'stages': {}}
    record.update(case_id=summary['case_id'], coordinates_verified=summary['coordinates_verified'])
    record['stages'][stage] = {'path': str(output), 'status': summary['status']}
    catalog.write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    titles = {'01_prepared': ('① DIC 输入与核查', 'input_audit.csv', 'fields.npz'),
              '02_forward': ('② 独立标定正演', 'comparison.csv', None),
              '03_inverse': ('③ DIC 模量反演', 'regional_comparison.csv', 'selection.csv'),
              '04_comparison': ('④ 位移、模量与误差对照', 'comparison.csv', 'regional_comparison.csv')}
    parts = ['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>实验结果总览</title>', PAGE_STYLE,
             f'<h1>{html.escape(record["case_id"])}</h1>',
             f'<p class="path">保存位置：{html.escape(str(case_root))}</p>']
    if not record['coordinates_verified']:
        parts.append('<p>当前为条件性比较：点云位置与材料分区的物理配准尚未独立核验。</p>')
    comparison = record['stages'].get('04_comparison')
    if comparison and (Path(comparison['path'])/'打开查看结果.html').exists():
        parts.append(f'<h2><a href="{_link(Path(comparison["path"])/"打开查看结果.html", case_root)}">查看云图、模量曲线和定量对照 →</a></h2>')
    parts.append('<div class="cards">')
    for key, (title, first, second) in titles.items():
        item = record['stages'].get(key)
        parts.append(f'<section class="card"><strong>{title}</strong>')
        if item:
            directory = Path(item['path'])
            parts.append(f'<p>状态：{html.escape(item["status"])}</p><p class="path">{html.escape(str(directory))}</p>')
            for name in (first, second, 'run_metadata.json'):
                if name and (directory/name).exists():
                    parts.append(f'<p><a href="{_link(directory/name, case_root)}">{html.escape(name)}</a></p>')
            for fields in sorted(directory.glob('nu_*/fields.npz')):
                parts.append(f'<p><a href="{_link(fields, case_root)}">{fields.parent.name} / 完整数值场</a></p>')
        else:
            parts.append('<p>尚未运行。</p>')
        parts.append('</section>')
    parts.append('</div></html>')
    overview = case_root/'结果总览.html'
    overview.write_text('\n'.join(parts), encoding='utf-8')
    results = project_root/'results'
    results.mkdir(exist_ok=True)
    parts = ['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>项目计算结果</title>', PAGE_STYLE,
             '<h1>项目计算结果</h1>',
             f'<h2><a href="{_link(overview, results)}">当前实验：{html.escape(record["case_id"])} →</a></h2>',
             f'<p class="path">{html.escape(str(case_root))}</p>']
    if comparison and (Path(comparison['path'])/'打开查看结果.html').exists():
        parts.append(f'<h2><a href="{_link(Path(comparison["path"])/"打开查看结果.html", results)}">直接查看 DIC、正反演云图与误差 →</a></h2>')
    parts.append('<p>实验总览内按四个步骤列出保存位置、表格、数值场和最终云图。</p><h2>其他实验</h2><ul>')
    for page in sorted(results.glob('*/结果总览.html')):
        if page.resolve() != overview:
            parts.append(f'<li><a href="{_link(page, results)}">{html.escape(page.parent.name)}</a></li>')
    parts.append('</ul><details><summary>历史结果目录</summary><ul>')
    for directory in sorted(results.iterdir()):
        if directory.is_dir() and not (directory/'结果总览.html').exists():
            parts.append(f'<li><a href="{_link(directory, results)}/">{html.escape(directory.name)}</a></li>')
    parts.append('</ul></details></html>')
    (results/'index.html').write_text('\n'.join(parts), encoding='utf-8')
    print(f'结果总览: {overview}\n固定入口: {project_root / "打开结果.html"}')


def write_case_report(output, summary, metrics, regions, selection):
    output = Path(output)
    mesh = summary['shared_mesh']
    verified = summary['coordinates_verified']
    status = summary['status']
    cover = pd.DataFrame(summary['coverage'])
    metric_table, region_table = pd.DataFrame(metrics), pd.DataFrame(regions)
    title = f"DIC 正反演对照：{summary['case_id']}"
    overview = Path(summary.get('prepared_directory', output.parent/'01_prepared')).parent/'结果总览.html'
    lines = [f'# {title}', '', f'运行状态：`{status}`。物理坐标配准已核验：{verified}。', '',
             '## 几何、观测与统一网格', '',
             f"物理模型尺寸为 {summary['physical_model_mm']} mm；DIC 点云只定义有效观测覆盖，不定义试样边界。未用点云跨度缩放坐标或修改模型尺寸。", '',
             f"独立标定正演与自由二维反演共用 {mesh['nel_x']}×{mesh['nel_y']} 单元，单元尺寸 {mesh['hx_mm']}×{mesh['hy_mm']} mm。DIC 在原始测点参与比较，不先重采样到有限元节点。", '',
             '有限元三点约束只固定数值刚体规范；DIC 与模型在观测空间通过同一两平移、一微小转动子空间配准。未删除仿射伸长或横向收缩。', '',
             '有效观测与采样间距见 `input_audit.csv`。', '',
             '## 独立标定正演、DIC 与反演回代', '']
    if not metric_table.empty:
        lines += ['| ν | 平台 | 用途 | 模型 | 双分量 RMS (mm) | 轴向 RMSE (mm) | 横向 RMSE (mm) |',
                  '| --- | --- | --- | --- | ---: | ---: | ---: |']
        for r in metrics:
            lines.append(f"| {r['nu']:.2f} | {r['case']} | {r['role']} | {r['model']} | {r['rms_mm']:.7f} | {r['rmse_ux_mm']:.7f} | {r['rmse_uy_mm']:.7f} |")
        lines += ['', '## 自由二维模量与独立标定', '',
                  '| ν | 填充率 | 标定均值±组内 SD (MPa) | 反演区域均值 (MPa) | 相对差异 |',
                  '| --- | --- | ---: | ---: | ---: |']
        for r in regions:
            lines.append(f"| {r['nu']:.2f} | {r['infill_percent']}% | {r['mean_E_MPa']:.1f}±{r['sd_E_MPa']:.1f} | {r['inverse_mean_E_MPa']:.1f} | {r['difference_percent']:+.2f}% |")
        lines += ['', '组内 SD 和反演场空间 SD 的含义不同，均不作为参数置信区间；分区统计包含未观测边带的模型延拓。', '',
                  f"主泊松比预先指定为 {summary['primary_nu_prespecified']}；其他 ν 为同条件敏感性比较，不根据留出平台误差更换主值。", '',
                  f"gamma={summary['selected_gamma']:g}，只按识别平台的带缓冲空间验证选择；是否处于扫描边界：{summary['gamma_at_scan_boundary']}。扫描边界最小值不代表全局最优。"]
    else:
        lines += ['本阶段仅完成输入核查；未生成正反演结论。']
    lines += ['', '## 结论范围与复现', '',
              '优化停止状态、物理配准状态和实验物理一致性分别记录。完整狗骨与截取矩形域的边界等效、绝对测量误差与材料标定转移仍需独立证据，本运行不自动宣告实验验证通过。', '',
              '每次输出目录独立保存输入哈希、配置快照、软件环境、识别与预测分工、选择记录、二维数组、误差与区域统计。不得覆盖既有结果。', '',
              '汇总文件：`comparison.csv`、`regional_comparison.csv`、`selection.csv`；每个 ν 子目录保存完整场及 PNG/SVG。']
    (output / 'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    parts = ['<!doctype html><meta charset="utf-8">', f'<title>{html.escape(title)}</title>',
             PAGE_STYLE,
             f'<h1>{html.escape(title)}</h1>',
             f'<p>运行状态：{html.escape(status)}；物理配准已核验：{verified}。</p>',
             f'<p><a href="{_link(overview, output)}">返回实验结果总览</a> · <a href="comparison.csv">位移误差 CSV</a> · <a href="regional_comparison.csv">五区模量 CSV</a> · <a href="selection.csv">选参记录 CSV</a></p>',
             f'<p class="path">本页与图表保存于：{html.escape(str(output.resolve()))}</p>',
             '<p>DIC 云图沿用 post_DIC：规则网格、jet 色图、等比例坐标及独立单图。三个位移场采用相同色标；两幅残差采用相同对称色标。显示时仅共同减去 DIC 分量均值，CSV 保留原参考态。</p>']
    if not metric_table.empty:
        parts += ['<h2>位移定量对照</h2>', metric_table.to_html(index=False, border=0, float_format=lambda x: f'{x:.7g}'),
                  '<h2>模量与独立标定对照</h2>', region_table.to_html(index=False, border=0, float_format=lambda x: f'{x:.6g}')]
        for solver in sorted(summary['solvers'], key=lambda s: (s['nu'] != summary['primary_nu_prespecified'], s['nu'])):
            directory = f"nu_{solver['nu']:.2f}"
            parts += [f"<h2>ν={solver['nu']:.2f}，收敛状态：{solver['converged']}</h2>"]
            for case in summary['force_pairing']:
                parts += [f'<h3>{html.escape(case)} 位移场与残差</h3>',
                          f'<p><a href="{directory}/{case}/point_fields.csv">逐测点位移与残差 CSV</a>（mm）</p>'] if (output/directory/case/'point_fields.csv').exists() else []
                for component, label in (('ux', '轴向 ux'), ('uy', '横向 uy')):
                    parts.append(f'<h4>{label}</h4><div class="fields">')
                    for key, title in (('dic', 'DIC 实测'), ('forward', '独立标定正演'),
                                       ('inverse', '反演回代'), ('forward_error', '独立标定正演 − DIC'),
                                       ('inverse_error', '反演回代 − DIC')):
                        stem = f'{directory}/{case}/{case}_{key}_{component}'
                        if (output/f'{stem}.png').exists():
                            parts.append(f'<figure><figcaption>{title}</figcaption><a href="{stem}.png"><img src="{stem}.png" alt="{title} {label}"></a><a href="{stem}.svg">SVG</a></figure>')
                    parts.append('</div>')
            for name in ('modulus_comparison.png', 'x_profile.png'):
                if (output/directory/name).exists():
                    parts.append(f'<img src="{directory}/{name}">')
    parts += ['<details><summary>物理域内的有效观测与 DIC 采样</summary>', cover.to_html(index=False, border=0), '</details>',
              '<details><summary>完整报告与适用范围</summary><pre>'+html.escape('\n'.join(lines))+'</pre></details>']
    (output / '打开查看结果.html').write_text('\n'.join(parts), encoding='utf-8')
