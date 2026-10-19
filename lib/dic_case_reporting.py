"""Reader-facing comparison of one prepared DIC case and its saved FE results."""

import html
from pathlib import Path

import pandas as pd


def write_case_report(output, summary, metrics, regions, selection):
    output = Path(output)
    mesh = summary['shared_mesh']
    verified = summary['coordinates_verified']
    status = summary['status']
    cover = pd.DataFrame(summary['coverage'])
    metric_table, region_table = pd.DataFrame(metrics), pd.DataFrame(regions)
    title = f"DIC 正反演对照：{summary['case_id']}"
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
             '<style>body{max-width:1500px;margin:25px auto;padding:0 20px;font:16px/1.6 "Microsoft YaHei",sans-serif;color:#17212b}table{border-collapse:collapse;font-size:14px}th,td{padding:6px 10px;border-bottom:1px solid #ddd}img{width:100%;height:auto}pre{white-space:pre-wrap}</style>',
             f'<h1>{html.escape(title)}</h1>',
             f'<p>运行状态：{html.escape(status)}；物理配准已核验：{verified}。</p>',
             '<h2>物理域内的有效观测与 DIC 采样</h2>', cover.to_html(index=False, border=0)]
    if not metric_table.empty:
        parts += ['<h2>位移定量对照</h2>', metric_table.to_html(index=False, border=0, float_format=lambda x: f'{x:.7g}'),
                  '<h2>模量与独立标定对照</h2>', region_table.to_html(index=False, border=0, float_format=lambda x: f'{x:.6g}')]
        for solver in summary['solvers']:
            directory = f"nu_{solver['nu']:.2f}"
            parts += [f"<h2>ν={solver['nu']:.2f}，收敛状态：{solver['converged']}</h2>"]
            for case in summary['force_pairing']:
                image = f'{directory}/{case}_forward_inverse.png'
                if (output/image).exists():
                    parts.append(f'<img src="{image}" alt="DIC、独立标定正演及反演回代">')
            for name in ('modulus_comparison.png', 'x_profile.png'):
                if (output/directory/name).exists():
                    parts.append(f'<img src="{directory}/{name}">')
    parts += ['<h2>报告</h2><pre>'+html.escape('\n'.join(lines))+'</pre>']
    (output / '打开查看结果.html').write_text('\n'.join(parts), encoding='utf-8')
