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
    '四格位移图：上排 DIC Data 为实测位移；下排 Displacement from Identified E 为识别模量计算的位移，'
    'Displacement from Calibrated E 为独立标定模量计算的位移。左列 ux（拉伸方向），'
    '右列 uy（宽度方向），各列共用色标，单位 mm。DIC 显示 d−mean(d)，'
    '有限元显示 HU+Rc−mean(d)，其中 Rc 为已有残差配准的两平移和一微小转动。'
    '两者使用相同物理坐标和真实 DIC 测点，不直接将三个约束下的原始节点 U 与 DIC 对比。'
    '共同均值只改变绘图零点，不改变残差或伸长。坐标和位移采用同一个正交变换，位置平移不加到位移上。'
)


def _inverse_displacement_label(summary, case):
    """Name the data used to identify E separately from the plotted load."""
    fit = summary.get('configuration', {}).get('FIT_CASES')
    if fit is None:
        fit = (summary.get('initialization') or {}).get('fit_cases', [])
    source = '、'.join(fit) if fit else '识别平台'
    role = '本平台回代' if case in fit else '跨载荷对照'
    return f'由{source}识别模量计算的位移（{role}）'


def _synthetic_notice(summary):
    source = summary.get('synthetic_observations')
    if not source:
        return ''
    return ('本页为无噪声合成恢复试验：位移全部由有限元生成，不使用实验DIC位移；'
            f"已知ν={source['known_nu']}，材料表示={source['representation']}。"
            '沿用DIC测点和坐标，图表中的DIC字段名仅表示观测位置。'
            '节点表示与突变分层表示的差异须单独评价；本试验不验证实验测量精度或物理模型。')


def _objective_description(summary):
    weights = summary.get('displacement_component_weights', {'ux': 1., 'uy': 1.})
    config = summary.get('configuration', summary.get('inverse_settings', {}))
    regularizer = ('正则为Huber-TV，阈值δ='+str(config.get('TV_DELTA', '见冻结配置'))+' MPa/mm，作用于E的梯度幅值。'
                   if config.get('REGULARIZATION_MODE') == 'huber_tv' else
                   '正则为模量梯度平方积分。')
    if config.get('REGULARIZATION_MODE') == 'exact_tv':
        regularizer = ('正则项为 γ∫Ω√[(∂E/∂x)²+(∂E/∂y)²]dA，无Huber阈值或平滑ε；'
                       'L-curve纵坐标直接取该积分（不含γ、不再开平方，单位MPa·mm）。'
                       '模量为连续Q4插值 E=ΣNᵢEᵢ，每个节点Eᵢ独立，各y行不绑定；'
                       '位移试函数和测试函数仍取原Q4的Nᵢ，不使用SUPG，也不输入分区界面或层数。')
    return (f"位移失配使用面积权重 × 分量系数：ux={weights['ux']:g}、uy={weights['uy']:g}。"
            '系数乘在残差平方上；同一权重用于刚体配准、初值、伴随梯度和 L-curve，不缩放原始位移。'
            '不设参考模量，不作模量或噪声尺度归一化。双分量 RMS 随权重定义改变，跨权重比较应分别看两个分量的 mm 误差。'
            + regularizer)


def _solver_settings_description(settings):
    if settings.get('REGULARIZATION_MODE') == 'exact_tv':
        opts = settings['EXACT_TV_OPTIONS']
        if settings.get('EXACT_TV_SOLVER', 'admm') == 'primal_dual':
            return (f"精确TV使用非线性原始—对偶迭代，最多{opts['max_outer']}步、"
                    f"每个γ/ν最多{settings['MAX_FUN']}次FE函数计算，"
                    f"初始模量步长{settings['EXACT_TV_PRIMAL_STEP']:g}，耦合系数{settings['EXACT_TV_COUPLING']:g}。"
                    f"驻点残差阈值={opts['stationarity_atol']:g}+{opts['stationarity_rtol']:g}×数据/正则导数∞范数的较大值；"
                    f"间隙阈值={opts['gap_atol']:g}+{opts['gap_rtol']:g}×γTV(E)。"
                    '对偶场逐点投影到|p|≤γ，允许的数值超界为1e-12×γ；驻点残差与Fenchel间隙须同时通过。'
                    'z=GE直接计算，因此约束残差恒为零，不另作ADMM迭代残差解释。'
                    '模量更新直接在正E上进行，记录的驻点残差仍为对log(E)的导数，便于与原证据对照。'
                    '原FTOL/GTOL不决定成功；局部一阶条件通过不证明全局最优或实验物理有效。')
        return (f"精确TV使用非线性ADMM，最多{opts['max_outer']}次外迭代、"
                f"每次材料子问题最多{opts['max_inner']}次迭代，每个γ/ν总计最多{settings['MAX_FUN']}次FE函数计算。"
                f"实际核验阈值为{opts}。驻点残差、GE=z约束、Fenchel间隙及|p|≤γ须同时通过；"
                '内层函数下降或原GTOL不作为精确TV收敛证明。强正则均匀场允许直接核验同一组条件。'
                '这些条件仅证明近似局部一阶驻点，物理模型有效性另行评价。')
    return (f"FTOL={settings['FTOL']:g}、GTOL={settings['GTOL']:g}，"
            f"最多{settings['MAX_ITER']}次迭代、{settings['MAX_FUN']}次函数计算。"
            '函数容差和梯度容差任一满足即可停止。')


def _tv_recovery_description(summary):
    recovery = summary.get('scan_recovery')
    if not recovery or not recovery.get('additional_evaluation_budget_per_attempt'):
        return ''
    attempts = '；'.join(
        f"γ={a['gamma']:.9g}：追加{a['extra_evaluations']}次，累计{a['total_evaluations']}次FE"
        for a in recovery['attempts'])
    return (f"基础扫描之外，按180N曲率峰缺少的邻点另行追加计算；每个相关γ最多追加"
            f"{recovery['additional_evaluation_budget_per_attempt']}次，实际总追加"
            f"{recovery['actual_extra_FE_evaluations']}次。{attempts}。"
            '同一γ从保存的失败E和对偶场重启，目标和停止容差不变；标定与280N不参与补算对象选择。'
            '原50点状态与原失败场分别保留；此处不能称为两网格相同总预算的效率比较。')


def _scan_completion_description(summary):
    record = summary.get('scan_completion')
    if not record:
        return ''
    return (f"γ={record['interrupted_gamma']:g}的原尝试因正模量步长下溢中断，最后状态未保存；"
            f"日志确认至少{record['interrupted_FE_lower_bound']}次FE，配置上限为"
            f"{record['interrupted_FE_upper_bound']}次，确数不可恢复。随后用较小数值步长"
            f"{record['retry_primal_step']:g}补算{record['retry_actual_evaluations']}次，"
            f"通过一阶条件：{record['retry_converged']}。目标与容差不变，未通过的场仍排除。"
            '表中求解次数统计保存的尝试，中断尝试的上述计算量需另计。')


def _geometry_description(summary):
    """Keep computation bounds distinct from the section carrying measured F."""
    geometry = summary.get('domain_geometry', {})
    if geometry.get('source') == 'dic_reference_bounds':
        marks = summary.get('material_registration')
        position = ('材料位置仍按点云与试样居中假设；' if marks is None else
                    f"制造坐标从左至右为 {marks['manufacturing_roi_x_mm']} mm；"
                    f"制造坐标相对FEM x的系数为 {marks.get('manufacturing_x_fem_coefficient', 1):+d}；"
                    f"沿FEM x的填充率顺序为 {marks.get('infill_order_along_fem_x', '见冻结配置')}%；"
                    '当前方向 ux=DY、uy=−DX；')
        if marks and marks['method'] == 'shared_reference_material_coordinates_unit_scale':
            position += '材料定位沿用已确认共用参考图的原始坐标；连续截取目标范围与实测数据的交集，缺失端不补点。'
        elif marks and 'nominal_mark_gap_mm' in marks:
            position += (f"按横线中点定位，标线端距推算间隔 {marks['nominal_mark_gap_mm']:.6f} mm，"
                         f"与导出跨度之差 {marks['dic_minus_nominal_mark_gap_mm']:+.6f} mm，保留独立毫米尺度。")
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
    if record.get('case_id') not in (None, summary['case_id']):
        record = {'stages': {}}
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
    if _synthetic_notice(summary):
        parts.append('<p><strong>'+html.escape(_synthetic_notice(summary))+'</strong></p>')
        recovery_page = case_root.parent/'分层位移反演核验.html'
        if recovery_page.exists():
            parts.append(f'<h2><a href="{_link(recovery_page, case_root)}">分层位移恢复与真实DIC反演汇总 →</a></h2>')
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
             f'<p>{_objective_description(summary)}</p>',
             '<p>实际拉力：'+html.escape('；'.join(f"{name}: {record['mean_corrected_force_N']:.6f} N"
                                                 for name, record in summary['force_pairing'].items()))+'</p>']
    if _synthetic_notice(summary):
        parts.insert(5, '<p><strong>'+html.escape(_synthetic_notice(summary))+'</strong></p>')
    if not summary['coordinates_verified']:
        parts.append('<p>物理配准尚未独立核验；当前图表对应显式位置假设。</p>')
    parts.append('<p>'+html.escape(_geometry_description(summary))+'</p>')
    if summary['workflow'] != 'dic_case_preparation':
        parts.append(f'<p>{RELATIVE_ERROR_DESCRIPTION} 位移图单位为 mm，误差图单位为 %。</p>')
    if summary['workflow'] == 'dic_case_inverse':
        config = summary['configuration']
        gamma = summary.get('selected_gamma')
        selected = f'{gamma:g}' if gamma is not None else '尚未选定'
        label = '基础扫描与最终ν设置（追加预算另列）' if _tv_recovery_description(summary) else '本次保存计算'
        parts.append(f'<p>{label}：{html.escape(_solver_settings_description(config))}'
                     f'γ={selected}，由识别平台的 L-curve 正曲率拐点选择。'
                     '修改源配置或重新出图均不会更新已保存的反演。</p>')
        parts.append('<p>目标函数：'+html.escape(summary['objective_definition'])+'</p>')
        parts.append(f'<p>L-curve 状态：{html.escape(summary.get("l_curve_status", "running"))}；'
                     f'{html.escape(summary.get("l_curve_message", ""))}。'
                     '<a href="selection.csv">扫描范数、曲率、收敛与选参记录</a></p>')
        if _tv_recovery_description(summary):
            parts.append('<p><strong>'+html.escape(_tv_recovery_description(summary))+'</strong> '
                         '<a href="tv_recovery_protocol.json">追加预算协议</a> · '
                         '<a href="selection_before_tv_recovery.csv">原50点记录</a></p>')
        if _scan_completion_description(summary):
            parts.append('<p>'+html.escape(_scan_completion_description(summary))+
                         ' <a href="interrupted_scan_protocol.json">中断与补算记录</a></p>')
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
                label = _inverse_displacement_label(summary, case) if model == 'inverse' else model_title
                parts.append(f'<h3>DIC—{label}：ux、uy 四格对照</h3>')
                parts.append(_field_figure(panel, f'上排 DIC，下排{label}；左 ux，右 uy；每列共用色标（mm）', output))
            for component, component_title in (('ux', '轴向 ux'), ('uy', '横向 uy')):
                label = _inverse_displacement_label(summary, case) if model == 'inverse' else model_title
                parts.append(f'<h3>{component_title}：DIC—{label}对照</h3><div class="fields">')
                for key, caption in (('dic', 'DIC（mm）'), (model, f'{label}（mm）'),
                                     (f'{model}_error', f'{label}相对误差（%）')):
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
    role_labels = {'fit': '本平台识别/回代', 'prediction': '跨载荷对照'}
    metric_table = metric_table.replace({'role': role_labels,
                                         'model': {'forward': '独立标定正演', 'inverse': '识别模量正演'}})
    observation_description = (
        'DIC参考点直接作为有限元节点，H仅将节点位移置换到测点顺序，无测点插值。'
        if mesh.get('source') == 'dic_grid' else
        'DIC在原始测点参与比较，H在实际位置计算Q4插值。'
    )
    title = f"DIC 正反演对照：{summary['case_id']}"
    fit_names = (summary.get('initialization') or {}).get('fit_cases', [])
    identification = ('、'.join(fit_names) or '见冻结配置')
    overview = Path(summary.get('prepared_directory', output.parent/'01_prepared')).parent/'结果总览.html'
    lines = [f'# {title}', '', f'运行状态：`{status}`。物理坐标配准已核验：{verified}。', '',
             '## 几何、观测与统一网格', '',
             _geometry_description(summary), '',
             f"独立标定正演与自由二维反演共用 {mesh['nel_x']}×{mesh['nel_y']} 单元，单元尺寸 {mesh['hx_mm']}×{mesh['hy_mm']} mm。{observation_description}", '',
             '有限元三点约束只固定数值刚体规范；DIC 与模型在观测空间通过同一两平移、一微小转动子空间配准。未删除仿射伸长或横向收缩。', '',
             '有效观测与采样间距见 `input_audit.csv`。', '',
             '## 独立标定正演、DIC 与反演回代', '',
             f'本组模量只用{identification}的DIC位移独立选γ并识别。其他载荷下的图使用本组固定模量正演；模量来源与当前绘图载荷分别标注。', '',
             RELATIVE_ERROR_DESCRIPTION, '',
             '位移及原始残差 CSV 保留 mm；相对误差图与逐测点新增列使用 %，各 ν 的 `relative_error_summary.csv` 保存分母和汇总指标。', '']
    if not metric_table.empty:
        lines += ['| ν | 平台 | 用途 | 模型 | 双分量 RMS (mm) | 轴向 RMSE (mm) | 横向 RMSE (mm) |',
                  '| --- | --- | --- | --- | ---: | ---: | ---: |']
        for r in metrics:
            lines.append(f"| {r['nu']:.2f} | {r['case']} | {role_labels.get(r['role'], r['role'])} | {r['model']} | {r['rms_mm']:.7f} | {r['rmse_ux_mm']:.7f} | {r['rmse_uy_mm']:.7f} |")
        lines += ['', '## 自由二维模量与独立标定', '',
                  '| ν | 填充率 | 标定均值±组内 SD (MPa) | 反演区域均值 (MPa) | 相对差异 |',
                  '| --- | --- | ---: | ---: | ---: |']
        for r in regions:
            lines.append(f"| {r['nu']:.2f} | {r['infill_percent']}% | {r['mean_E_MPa']:.1f}±{r['sd_E_MPa']:.1f} | {r['inverse_mean_E_MPa']:.1f} | {r['difference_percent']:+.2f}% |")
        lines += ['', '组内 SD 和反演场空间 SD 的含义不同，均不作为参数置信区间；分区统计限于当前计算域，并非完整制造区统计。', '',
                  f"主泊松比预先指定为 {summary['primary_nu_prespecified']}；其他 ν 为同条件敏感性比较，不根据跨载荷误差更换主值。", '',
                  f"gamma={summary['selected_gamma']:g}，只按识别平台的 L-curve 最大正曲率拐点选择；状态：{summary['l_curve_status']}。不按另一平台误差或独立材料标定调参。"]
        settings = summary['inverse_settings']
        lines += ['', '## 反演选参与实际停止设置', '',
                  f"目标函数：`{summary['objective_definition']}`。", '', _objective_description(summary), '',
                  f"已启用 L-curve。gamma 候选值为 {settings['GAMMA_CANDIDATES']}；在预设主 ν 下，仅使用识别平台，计算原始残差范数 ρ 与正则范数 η（不含 gamma）。", '',
                  '在 log(ρ)、log(η) 上用有符号三点曲率识别拐点，排除未收敛、重合、非单调及端部缺少支撑的候选；不能识别拐点时保留诊断并停止，不回退至空间验证或固定 gamma。', '',
                  f"扫描初始化策略：{summary.get('scan_initialization', '见上游反演元数据')}。全部ν最后从同一个所选场出发重新求解，独立标定不进入反演目标或初值。", '',
                  _solver_settings_description(settings), '']
        if _tv_recovery_description(summary):
            lines += ['基础预算的例外：'+_tv_recovery_description(summary), '']
        if _scan_completion_description(summary):
            lines += [_scan_completion_description(summary), '']
        if settings.get('REGULARIZATION_MODE') == 'exact_tv':
            lines += ['TV实际证据如下：三个残差比值须≤1，对偶超界/γ须≤1e-12。', '',
                      '| ν | 更新步数 | 驻点残差/阈值 | 约束残差/阈值 | 间隙绝对值/阈值 | 对偶超界/γ | 通过一阶证据 |',
                      '| --- | ---: | ---: | ---: | ---: | ---: | --- |']
            for solver in summary['solvers']:
                c = solver['tv_certificate']
                lines.append(f"| {solver['nu']:.2f} | {solver['n_iterations']} | {c['stationarity']/c['stationarity_tol']:.6g} | "
                             f"{c['primal_rms']/c['primal_tol']:.6g} | {abs(c['fenchel_gap'])/c['gap_tol']:.6g} | "
                             f"{c['dual_violation']/summary['selected_gamma']:.6g} | {solver['converged']} |")
        else:
            lines += ['| ν | 迭代数 | 最终梯度 ∞ 范数 | 满足 GTOL | 停止信息 |', '| --- | ---: | ---: | --- | --- |']
            for solver in summary['solvers']:
                lines.append(f"| {solver['nu']:.2f} | {solver['n_iterations']} | {solver['final_gradient_inf']:.6g} | {solver['gradient_tolerance_satisfied']} | {solver['message']} |")
    else:
        lines += ['本阶段仅完成输入核查；未生成正反演结论。']
    lines += ['', '## 结论范围与复现', '',
              '优化停止状态、物理配准状态和实验物理一致性分别记录。完整狗骨与截取矩形域的边界等效、绝对测量误差与材料标定转移仍需独立证据，本运行不自动宣告实验验证通过。', '',
              '每次输出目录独立保存输入哈希、配置快照、软件环境、识别平台与跨载荷分工、选择记录、二维数组、误差与区域统计。不得覆盖既有结果。', '',
              '汇总文件：`comparison.csv`、`regional_comparison.csv`、`selection.csv`；每个 ν 子目录保存完整场及 PNG/SVG。']
    if _synthetic_notice(summary):
        lines.insert(2, _synthetic_notice(summary))
    (output / 'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    parts = ['<!doctype html><meta charset="utf-8">', f'<title>{html.escape(title)}</title>',
             PAGE_STYLE,
             f'<h1>{html.escape(title)}</h1>',
             f'<p>运行状态：{html.escape(status)}；物理配准已核验：{verified}。</p>',
             f'<p><a href="{_link(overview, output)}">返回实验结果总览</a> · <a href="comparison.csv">位移误差 CSV</a> · <a href="regional_comparison.csv">区域模量 CSV</a> · <a href="selection.csv">选参记录 CSV</a></p>',
             f'<p class="path">本页与图表保存于：{html.escape(str(output.resolve()))}</p>',
             '<p>DIC 云图沿用 post_DIC：规则网格、jet 色图、等比例坐标及独立单图。三个位移场采用相同色标；两幅相对误差采用相同对称百分比色标。显示时仅共同减去 DIC 分量均值，CSV 保留原参考态。</p>',
             f'<p>本组模量识别数据：{html.escape(identification)} DIC；其他载荷仅使用本组固定模量进行跨载荷对照。</p>',
             f'<p>{COMPONENT_REFERENCE_TEXT}</p>',
             f'<p>{RELATIVE_ERROR_DESCRIPTION}</p>',
             f'<p>{_objective_description(summary)}</p>']
    if _synthetic_notice(summary):
        parts.insert(4, '<p><strong>'+html.escape(_synthetic_notice(summary))+'</strong></p>')
    if not metric_table.empty:
        settings = summary['inverse_settings']
        label = '基础扫描与最终ν设置（追加预算另列）' if _tv_recovery_description(summary) else '本次反演实际使用'
        parts.append(f'<p>{label}：{html.escape(_solver_settings_description(settings))}'
                     f'γ={summary["selected_gamma"]:g}，由识别平台的 L-curve 选定。'
                     '完整停止信息见页末报告。</p>')
        inverse = Path(summary['inverse_directory'])
        if _tv_recovery_description(summary):
            parts.append('<p><strong>'+html.escape(_tv_recovery_description(summary))+'</strong> '
                         f'<a href="{_link(inverse/"tv_recovery_protocol.json", output)}">追加预算协议</a></p>')
        if _scan_completion_description(summary):
            parts.append('<p>'+html.escape(_scan_completion_description(summary))+'</p>')
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
                    if model == 'inverse':
                        label = _inverse_displacement_label(summary, case)
                    panel = output/directory/case/f'{case}_{model}_dic_components.png'
                    if panel.exists():
                        parts.append(_field_figure(panel, f'DIC—{label}四格图：上排 DIC，下排有限元；左 ux，右 uy；单位 mm', output))
                for component, label in (('ux', '轴向 ux'), ('uy', '横向 uy')):
                    parts.append(f'<h4>{label}</h4><div class="fields">')
                    inverse_label = _inverse_displacement_label(summary, case)
                    for key, title in (('dic', 'DIC 实测（mm）'), ('forward', '独立标定正演（mm）'),
                                       ('inverse', f'{inverse_label}（mm）'), ('forward_error', '独立标定正演相对误差（%）'),
                                       ('inverse_error', f'{inverse_label}相对误差（%）')):
                        if key == 'forward_error':
                            parts.append('</div><div class="clouds">')
                        stem = f'{directory}/{case}/{case}_{key}_{component}'
                        if (output/f'{stem}.png').exists():
                            parts.append(_field_figure(output/f'{stem}.png', title, output))
                    parts.append('</div>')
            for name in ('modulus_comparison.png', 'x_profile.png'):
                if (output/directory/name).exists():
                    if name == 'x_profile.png' and (output/directory/'calibration_trend.json').exists():
                        from .calibration_trend import TREND_NOTE
                        parts.append('<p>'+html.escape(TREND_NOTE)+'</p>')
                    parts.append(f'<img src="{directory}/{name}">')
    parts += ['<details><summary>物理域内的有效观测与 DIC 采样</summary>', cover.to_html(index=False, border=0), '</details>',
              '<details><summary>完整报告与适用范围</summary><pre>'+html.escape('\n'.join(lines))+'</pre></details>']
    (output / '打开查看结果.html').write_text('\n'.join(parts), encoding='utf-8')
