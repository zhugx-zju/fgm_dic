"""Reader-facing comparison of one prepared DIC case and its saved FE results."""

import html
import json
import os
from pathlib import Path
from urllib.parse import quote

import pandas as pd

from .dic_preprocessing import RELATIVE_ERROR_DESCRIPTION


PAGE_STYLE = '''<style>
body{max-width:1400px;margin:28px auto;padding:0 24px;font:16px/1.65 "Microsoft YaHei",sans-serif;color:#17212b;background:#fafbfc}
a{color:#075fa2}h1{font-size:28px}h2{margin-top:32px}.cards,.fields{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}
.card,figure{margin:0;background:white;padding:16px;border:1px solid #dce1e6;border-radius:8px}.card strong{display:block}.path,code{overflow-wrap:anywhere;font:13px/1.6 Consolas,monospace}
.clouds{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}
table{border-collapse:collapse;font-size:14px;display:block;overflow:auto}th,td{padding:6px 10px;border-bottom:1px solid #ddd}img{width:100%;height:auto}pre{white-space:pre-wrap}
@media(max-width:800px){.cards,.fields,.clouds{grid-template-columns:1fr}}
</style>'''

COMPONENT_REFERENCE_TEXT = (
    '四格位移图：上排为处理后的 DIC，下排为配准后的有限元；左列 ux（拉伸方向），'
    '右列 uy（宽度方向），各列共用色标，单位 mm。DIC 显示 d−mean(d)，'
    '有限元显示 HU+Rc−mean(d)，其中 Rc 为已有残差配准的两平移和一微小转动。'
    '两者使用相同物理坐标和真实 DIC 测点，不直接将三个约束下的原始节点 U 与 DIC 对比。'
    '共同均值只改变绘图零点，不改变残差或伸长。坐标和位移采用同一个正交变换，位置平移不加到位移上。'
)


def _geometry_description(summary):
    """Keep computation bounds distinct from the section carrying measured F."""
    geometry = summary.get('domain_geometry', {})
    if geometry.get('source') == 'dic_reference_bounds':
        marks = summary.get('material_registration')
        position = ('材料位置仍按点云与试样居中假设；' if marks is None else
                    f"材料按横线中点定位，制造坐标从左至右为 {marks['manufacturing_roi_x_mm']} mm；"
                    f"制造坐标相对FEM x的系数为 {marks.get('manufacturing_x_fem_coefficient', 1):+d}；"
                    f"标线端距推算间隔 {marks['nominal_mark_gap_mm']:.6f} mm，"
                    f"与导出跨度之差 {marks['dic_minus_nominal_mark_gap_mm']:+.6f} mm，保留独立毫米尺度。"
                    '当前方向 ux=DY、uy=−DX；')
        return (f"计算域由识别平台 DIC 初始坐标范围确定，尺寸为 {summary['physical_model_mm']} mm；"
                f"真实承载宽度为 {geometry['load_section_width_mm']} mm。"
                '牵引按真实截面计算，截取域端力按宽度比例分配；坐标和材料分区均未缩放。'
                '内部横向截面暂按自由边界近似；'+position+'不代表子域严格等效。')
    return (f"物理模型尺寸为 {summary['physical_model_mm']} mm；"
            'DIC 点云定义有效观测覆盖，未用点云跨度缩放坐标。')


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
    parts.append('<p>'+html.escape(_geometry_description(summary))+'</p>')
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
            gallery = directory/('打开查看结果.html' if key == '04_comparison' else '云图.html')
            if gallery.exists():
                count = len(list(directory.rglob('*.png')))
                parts.append(f'<h3><a href="{_link(gallery, case_root)}">查看本步云图（{count} 幅）→</a></h3>')
            for name in (first, second, 'run_metadata.json'):
                if name and (directory/name).exists():
                    parts.append(f'<p><a href="{_link(directory/name, case_root)}">{html.escape(name)}</a></p>')
            for fields in sorted(directory.glob('nu_*/fields.npz')):
                parts.append(f'<p><a href="{_link(fields, case_root)}">{fields.parent.name} / 完整数值场</a></p>')
        else:
            parts.append('<p>尚未运行。</p>')
        parts.append('</section>')
    parts.append('</div>')
    history = case_root/'history'/'结果总览.html'
    if history.exists():
        parts.append(f'<p><a href="{_link(history, case_root)}">历史版本结果 →</a>（按原计算条件保留）</p>')
    for archive in sorted((case_root/'history').glob('*/结果总览.html')):
        parts.append(f'<p><a href="{_link(archive, case_root)}">历史版本：{html.escape(archive.parent.name)} →</a></p>')
    diagnostics = sorted(case_root.rglob('诊断报告.html'))
    if diagnostics:
        parts.append('<h2>辅助诊断</h2><ul>')
        for page in diagnostics:
            label = page.parent.relative_to(case_root).as_posix()
            parts.append(f'<li><a href="{_link(page, case_root)}">{html.escape(label)}</a></li>')
        parts.append('</ul>')
    parts.append('</html>')
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
    parts.append('<p>实验总览内按四个步骤列出保存位置、表格、数值场和最终云图。</p>')
    others = [p for p in sorted(results.glob('*/结果总览.html')) if p.resolve() != overview]
    if others:
        parts.append('<h2>其他实验</h2><ul>')
        for page in others:
            parts.append(f'<li><a href="{_link(page, results)}">{html.escape(page.parent.name)}</a></li>')
        parts.append('</ul>')
    history = [p for p in sorted(results.iterdir()) if p.is_dir() and not (p/'结果总览.html').exists()]
    if history:
        parts.append('<details><summary>其他结果目录</summary><ul>')
        for directory in history:
            parts.append(f'<li><a href="{_link(directory, results)}/">{html.escape(directory.name)}</a></li>')
        parts.append('</ul></details>')
    parts.append('</html>')
    (results/'index.html').write_text('\n'.join(parts), encoding='utf-8')
    print(f'结果总览: {overview}\n固定入口: {project_root / "打开结果.html"}')


def write_stage_gallery(output, summary, title):
    """Browse all generated stage plots from one visible entry."""
    output = Path(output)
    overview = Path(summary.get('prepared_directory', output.parent/'01_prepared')).parent/'结果总览.html'
    reference_text = ('DIC 单图仅在显示时减去对应分量均值，位置与位移均已转换到 FEM 轴向。'
                      if summary['workflow'] == 'dic_case_preparation' else COMPONENT_REFERENCE_TEXT)
    parts = ['<!doctype html><html lang="zh-CN"><meta charset="utf-8">',
             f'<title>{html.escape(title)}</title>', PAGE_STYLE,
             f'<h1>{html.escape(title)}</h1>',
             f'<p><a href="{_link(overview, output)}">返回四步结果总览</a></p>',
             f'<p class="path">保存位置：{html.escape(str(output.resolve()))}</p>',
             f'<p>{reference_text}</p>',
             '<p>实际拉力：'+html.escape('；'.join(f"{name}: {record['mean_corrected_force_N']:.6f} N"
                                                 for name, record in summary['force_pairing'].items()))+'</p>']
    if not summary['coordinates_verified']:
        parts.append('<p>物理配准尚未独立核验；当前图表对应显式位置假设。</p>')
    parts.append('<p>'+html.escape(_geometry_description(summary))+'</p>')
    if summary['workflow'] != 'dic_case_preparation':
        parts.append(f'<p>{RELATIVE_ERROR_DESCRIPTION} 位移图单位为 mm，误差图单位为 %。</p>')
    if summary['workflow'] == 'dic_case_inverse':
        config = summary['configuration']
        gamma = summary.get('selected_gamma')
        selected = f'{gamma:g}' if gamma is not None else '尚未选定'
        parts.append(f'<p>本次保存计算：FTOL={config["FTOL"]:g}，GTOL={config["GTOL"]:g}。'
                     f'γ={selected}，由识别平台的 L-curve 正曲率拐点选择。'
                     '修改源配置或重新出图均不会更新已保存的反演。</p>')
        parts.append('<p>目标函数：0.5 rᵀW_A r + 0.5 γ ∫|∇E|² dA。'
                     'W_A 仅为测点面积权重；不设置参考模量，不做模量或噪声尺度归一化。</p>')
        parts.append(f'<p>L-curve 状态：{html.escape(summary.get("l_curve_status", "running"))}；'
                     f'{html.escape(summary.get("l_curve_message", ""))}。'
                     '<a href="selection.csv">扫描范数、曲率、收敛与选参记录</a></p>')
    primary = f"nu_{summary['primary_nu_prespecified']:.2f}"
    pictures = sorted(output.rglob('*.png'), key=lambda p: (
        0 if p.parent == output else 1 if primary in p.parts else 2,
        p.relative_to(output).parts[0], len(p.relative_to(output).parts), p.as_posix()))
    parents = dict.fromkeys(p.parent for p in pictures)
    for parent in parents:
        label = parent.relative_to(output).as_posix()
        parts.append(f'<h2>{html.escape(label if label != "." else "材料场")}</h2>')
        grouped = set()
        case = parent.name
        for model, model_title in (('forward', '独立标定正演'), ('inverse', '反演回代')):
            if not (parent/f'{case}_dic_ux.png').exists() or not (parent/f'{case}_{model}_ux.png').exists():
                continue
            parts.append(f'<p><a href="{_link(parent/"point_fields.csv", output)}">逐测点位移、残差（mm）与相对误差（%）</a>'
                         f' · <a href="{_link(parent.parent/"relative_error_summary.csv", output)}">相对误差汇总 CSV</a></p>')
            panel = parent/f'{case}_{model}_dic_components.png'
            if panel.exists():
                grouped.add(panel)
                label = ('反演模量预测' if model == 'inverse' and
                         case not in summary['configuration']['FIT_CASES'] else model_title)
                parts.append(f'<h3>DIC—{label}：ux、uy 四格对照</h3>')
                parts.append(_field_figure(panel, '上排 DIC，下排有限元；左 ux，右 uy；每列共用色标（mm）', output))
            for component, component_title in (('ux', '轴向 ux'), ('uy', '横向 uy')):
                parts.append(f'<h3>{component_title}：DIC—{model_title}对照</h3><div class="fields">')
                for key, caption in (('dic', 'DIC（mm）'), (model, f'{model_title}（mm）'),
                                     (f'{model}_error', f'{model_title}相对误差（%）')):
                    picture = parent/f'{case}_{key}_{component}.png'
                    grouped.add(picture)
                    parts.append(_field_figure(picture, caption, output))
                parts.append('</div>')
        parts.append('<div class="clouds">')
        for picture in (p for p in pictures if p.parent == parent and p not in grouped):
            link = _link(picture, output)
            parts.append(f'<figure><a href="{link}"><img src="{link}" alt="{html.escape(picture.stem)}"></a>')
            vector = picture.with_suffix('.svg')
            if vector.exists():
                parts.append(f'<a href="{_link(vector, output)}">下载 SVG</a>')
            parts.append('</figure>')
        parts.append('</div>')
    if not pictures:
        parts.append('<p>当前仅有输入核查记录，尚无可绘制的已配准场。</p>')
    parts.append('</html>')
    (output/'云图.html').write_text('\n'.join(parts), encoding='utf-8')


def _field_figure(picture, caption, output):
    link = _link(picture, output)
    vector = _link(picture.with_suffix('.svg'), output)
    return (f'<figure><figcaption>{html.escape(caption)}</figcaption>'
            f'<a href="{link}"><img src="{link}" alt="{html.escape(caption)}"></a>'
            f'<a href="{vector}">SVG</a></figure>')


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
             _geometry_description(summary), '',
             f"独立标定正演与自由二维反演共用 {mesh['nel_x']}×{mesh['nel_y']} 单元，单元尺寸 {mesh['hx_mm']}×{mesh['hy_mm']} mm。DIC 在原始测点参与比较，不先重采样到有限元节点。", '',
             '有限元三点约束只固定数值刚体规范；DIC 与模型在观测空间通过同一两平移、一微小转动子空间配准。未删除仿射伸长或横向收缩。', '',
             '有效观测与采样间距见 `input_audit.csv`。', '',
             '## 独立标定正演、DIC 与反演回代', '', RELATIVE_ERROR_DESCRIPTION, '',
             '位移及原始残差 CSV 保留 mm；相对误差图与逐测点新增列使用 %，各 ν 的 `relative_error_summary.csv` 保存分母和汇总指标。', '']
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
                  f"gamma={summary['selected_gamma']:g}，只按识别平台的 L-curve 最大正曲率拐点选择；状态：{summary['l_curve_status']}。不按预测平台误差或独立材料标定调参。"]
        settings = summary['inverse_settings']
        lines += ['', '## 反演选参与实际停止设置', '',
                  '目标函数为 `0.5*r.T@W_area@r + 0.5*gamma*integral(|grad(E)|^2)`。W_area 仅保留真实测点面积权重，不设参考模量，不作模量或噪声尺度归一化。', '',
                  f"已启用 L-curve。gamma 候选值为 {settings['GAMMA_CANDIDATES']}；在预设主 ν 下，仅使用识别平台，计算原始残差范数 ρ 与正则范数 η（不含 gamma）。", '',
                  '在 log(ρ)、log(η) 上用有符号三点曲率识别拐点，排除未收敛、重合、非单调及端部缺少支撑的候选；不能识别拐点时保留诊断并停止，不回退至空间验证或固定 gamma。', '',
                  f"扫描初始化策略：{summary.get('scan_initialization', '见上游反演元数据')}。全部ν最后从同一个所选场出发重新求解，独立标定不进入反演目标或初值。", '',
                  f"已保存计算实际使用 FTOL={settings['FTOL']:g}、GTOL={settings['GTOL']:g}，最多 {settings['MAX_ITER']} 次迭代、{settings['MAX_FUN']} 次函数计算。配置取自上游反演记录，修改源配置和重新出图不会更新已有计算。", '',
                  '函数容差和梯度容差是不同的停止条件；任一满足即可停止，不要求二者同时满足。', '',
                  '| ν | 迭代数 | 最终梯度 ∞ 范数 | 满足 GTOL | 停止信息 |', '| --- | ---: | ---: | --- | --- |']
        for solver in summary['solvers']:
            lines.append(f"| {solver['nu']:.2f} | {solver['n_iterations']} | {solver['final_gradient_inf']:.6g} | {solver['gradient_tolerance_satisfied']} | {solver['message']} |")
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
             '<p>DIC 云图沿用 post_DIC：规则网格、jet 色图、等比例坐标及独立单图。三个位移场采用相同色标；两幅相对误差采用相同对称百分比色标。显示时仅共同减去 DIC 分量均值，CSV 保留原参考态。</p>',
             f'<p>{COMPONENT_REFERENCE_TEXT}</p>',
             f'<p>{RELATIVE_ERROR_DESCRIPTION}</p>']
    if not metric_table.empty:
        settings = summary['inverse_settings']
        parts.append(f'<p>本次反演实际使用 FTOL={settings["FTOL"]:g}、GTOL={settings["GTOL"]:g}；'
                     f'γ={summary["selected_gamma"]:g}，由识别平台的 L-curve 选定。'
                     '完整停止信息见页末报告。</p>')
        inverse = Path(summary['inverse_directory'])
        parts.append(f'<p><a href="{_link(inverse/"云图.html", output)}">查看 L-curve、曲率图与反演场</a></p>')
        parts += ['<h2>位移定量对照</h2>', metric_table.to_html(index=False, border=0, float_format=lambda x: f'{x:.7g}'),
                  '<h2>模量与独立标定对照</h2>', region_table.to_html(index=False, border=0, float_format=lambda x: f'{x:.6g}')]
        for solver in sorted(summary['solvers'], key=lambda s: (s['nu'] != summary['primary_nu_prespecified'], s['nu'])):
            directory = f"nu_{solver['nu']:.2f}"
            parts += [f"<h2>ν={solver['nu']:.2f}，收敛状态：{solver['converged']}</h2>"]
            relative_path = output/directory/'relative_error_summary.csv'
            if relative_path.exists():
                parts.append(f'<p><a href="{directory}/relative_error_summary.csv">相对误差汇总 CSV（含各分量的 DIC 分母）</a></p>')
                relative_table = pd.read_csv(relative_path).rename(columns={
                    'case': '平台', 'model': '模型', 'component': '分量',
                    'dic_deformation_rms_mm': 'DIC 变形 RMS (mm)',
                    'relative_weighted_rms_percent': '相对 RMS (%)',
                    'absolute_relative_p95_percent': '绝对相对误差 P95 (%)'})
                parts.append(relative_table.drop(columns=['role', 'relative_defined']).to_html(
                    index=False, border=0, float_format=lambda x: f'{x:.6g}', na_rep='未定义'))
            for case in summary['force_pairing']:
                parts += [f'<h3>{html.escape(case)} 位移场与残差</h3>',
                          f'<p><a href="{directory}/{case}/point_fields.csv">逐测点位移、残差（mm）与相对误差（%）CSV</a></p>'] if (output/directory/case/'point_fields.csv').exists() else []
                for model, label in (('forward', '独立标定正演'), ('inverse', '反演模量回代或预测')):
                    panel = output/directory/case/f'{case}_{model}_dic_components.png'
                    if panel.exists():
                        parts.append(_field_figure(panel, f'DIC—{label}四格图：上排 DIC，下排有限元；左 ux，右 uy；单位 mm', output))
                for component, label in (('ux', '轴向 ux'), ('uy', '横向 uy')):
                    parts.append(f'<h4>{label}</h4><div class="fields">')
                    for key, title in (('dic', 'DIC 实测（mm）'), ('forward', '独立标定正演（mm）'),
                                       ('inverse', '反演回代（mm）'), ('forward_error', '独立标定正演相对误差（%）'),
                                       ('inverse_error', '反演回代相对误差（%）')):
                        if key == 'forward_error':
                            parts.append('</div><div class="clouds">')
                        stem = f'{directory}/{case}/{case}_{key}_{component}'
                        if (output/f'{stem}.png').exists():
                            parts.append(_field_figure(output/f'{stem}.png', title, output))
                    parts.append('</div>')
            for name in ('modulus_comparison.png', 'x_profile.png'):
                if (output/directory/name).exists():
                    parts.append(f'<img src="{directory}/{name}">')
    parts += ['<details><summary>物理域内的有效观测与 DIC 采样</summary>', cover.to_html(index=False, border=0), '</details>',
              '<details><summary>完整报告与适用范围</summary><pre>'+html.escape('\n'.join(lines))+'</pre></details>']
    (output / '打开查看结果.html').write_text('\n'.join(parts), encoding='utf-8')
