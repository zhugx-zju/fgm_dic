"""Publish the saved 380N audit and figures without importing a FE solver."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re


def publish(root: Path) -> None:
    def read(relative):
        return json.loads((root/relative).read_text(encoding='utf-8'))

    selection = read('03_inverse/380N/selection.json')
    comparison = read('04_comparison/summary.json')
    audit = read('verification/scan_audit.json')
    gamma_text = '未选定' if selection['selected_gamma'] is None else str(selection['selected_gamma'])
    with_style = '<style>body{max-width:1100px;margin:25px auto;padding:20px;font:16px/1.7 "Microsoft YaHei",sans-serif}img{max-width:100%}table{border-collapse:collapse}td,th{padding:8px;border:1px solid #ccc}a{color:#075fa2}</style>'
    parts = ['<!doctype html><meta charset="utf-8">'+with_style,
             '<h1>380N实验DIC：统一材料模型／区内正则</h1>',
             '<p>121×19单元，23920个观测点，七个自由二维Q4分支；六个已知固定界面，ν=0.26。'
             '本组沿用假设净力319.9 N，不是同步实测平台力。</p>',
             f'<p>50点扫描已完成，正演{selection["forward_evaluations"]:,}次；'
             f'一阶条件通过{selection["valid_points"]}/50，所选γ：{gamma_text}。</p>']
    if selection['selected_gamma'] is None:
        parts.append('<p><strong>有效求解点不足，未完成选参。</strong>当前不能判断380N的L-curve是否有可靠拐点。'
                     '下面编号00/24/49是预定诊断点，均未选定γ；不根据残差或材料标定补选。</p>')
    parts.append(f'<p>记录一致性复核通过：{audit["passed"]}；这不代表优化收敛。'
                 f'逐点停止原因保存在<a href="verification/scan_audit.json">复核文件</a>中。</p>')
    parts.append('<table><tr><th>诊断编号</th><th>γ</th><th>轴向RMSE（μm）</th>'
                 '<th>横向RMSE（μm）</th><th>投影梯度∞范数</th><th>触界分支系数</th></tr>')
    for row in comparison['records']:
        x, y = row['component_rmse_um']
        parts.append(f'<tr><td>{row["index"]:02d}</td><td>{row["gamma"]:.6g}</td>'
                     f'<td>{x:.3f}</td><td>{y:.3f}</td><td>{row["projected_gradient_inf"]:.3g}</td>'
                     f'<td>{row["active_bounds"]}</td></tr>')
    parts += ['</table>', '<p>最弱γ诊断场的残差更小，同时出现强空间振荡和大量参数触界。'
              '残差下降不能单独代表材料恢复可靠；触界计数指分支系数，不是物理节点数。'
              '横向空间离散也不是置信区间。</p>',
              '<p>独立标定正演的轴向／横向RMSE约7.442／3.793 μm。标定仅用于正演和事后比较，'
              '没有用于初值、目标或γ选择。此前仿真进入条件未通过的记录保持不变。</p>',
              '<p><a href="03_inverse/380N/scan.csv">全部扫描点</a> · '
              '<a href="03_inverse/380N/selection.json">选参状态</a> · '
              '<a href="verification/scan_audit.json">50点独立复核</a> · '
              '<a href="verification/independent_stopped_state_checks.json">完整映射及差分检查</a> · '
              '<a href="04_comparison/summary.json">诊断数据</a></p>']
    for title, folder in [('模量与宽度平均', root/'04_comparison'),
                          ('L-curve与曲率', root/'03_inverse/380N'),
                          ('独立标定正演与DIC', root/'02_forward/380N')]:
        parts.append(f'<h2>{title}</h2>')
        for path in sorted(folder.glob('*.png')):
            relative = path.relative_to(root).as_posix()
            parts.append(f'<a href="{Path(relative).with_suffix(".svg").as_posix()}"><img src="{relative}" loading="lazy"></a>')
    for folder in sorted((root/'04_comparison').iterdir()):
        if not folder.is_dir():
            continue
        parts.append(f'<details><summary>{folder.name}：DIC、独立正演与诊断场回代（同色标）</summary>')
        for path in sorted(folder.glob('*.png')):
            relative = path.relative_to(root).as_posix()
            parts.append(f'<p>{path.stem}</p><a href="{Path(relative).with_suffix(".svg").as_posix()}"><img src="{relative}" loading="lazy"></a>')
        parts.append('</details>')
    html = '\n'.join(parts)
    for target in re.findall(r'(?:href|src)="([^"]+)"', html):
        if not (root/target).is_file():
            raise FileNotFoundError(root/target)
    page = root/'结果总览.html'
    page.write_text(html, encoding='utf-8')
    print('保存位置：', page)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1]/
                        'results/unified_gradient_layered/380n_coarse_lcurve_20261023')
    publish(parser.parse_args().root)


if __name__ == '__main__':
    main()
