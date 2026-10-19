"""Render a concise, reproducible report from saved revised-DIC results."""

import argparse
import html
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lib.experiment_data import file_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--sensitivity', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    inverse = args.source / 'inverse_primary'
    meta = json.loads((inverse / 'run_metadata.json').read_text(encoding='utf-8'))
    sensitivity_meta = json.loads((args.sensitivity / 'run_metadata.json').read_text(encoding='utf-8'))
    metrics = pd.read_csv(inverse / 'forward_inverse_comparison.csv')
    regions = pd.read_csv(inverse / 'regional_modulus_comparison.csv')
    reversed_regions = pd.read_csv(args.sensitivity / 'reverse_direction' / 'regional_modulus_comparison.csv')
    profile = pd.read_csv(inverse / 'x_profile.csv')
    reverse_profile = pd.read_csv(args.sensitivity / 'reverse_direction' / 'x_profile.csv')
    validation = pd.read_csv(args.source / 'gamma_validation.csv')
    sensitivity = pd.read_csv(args.sensitivity / 'inverse_sensitivity.csv')
    audit = pd.read_csv(args.source / 'input_audit.csv')
    with np.load(inverse / 'fields.npz') as fields:
        e = fields['E_inverse'].reshape(11, 36)
        coords = fields['coordinates'].reshape(11, 36, 2)
        observation_xy = fields['180N_coordinates'].copy()
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    artist = axes[0, 0].pcolormesh(coords[:, :, 0], coords[:, :, 1], e, shading='auto', cmap='viridis')
    lo, hi = observation_xy.min(axis=0), observation_xy.max(axis=0)
    axes[0, 0].plot([lo[0], hi[0], hi[0], lo[0], lo[0]],
                    [lo[1], lo[1], hi[1], hi[1], lo[1]], '--', color='white', lw=1)
    axes[0, 0].set(title='Free 2D inverse; dashed box: observed coverage', xlabel='Assumed x (mm)',
                   ylabel='Assumed y (mm)', aspect='equal', xlim=(0, 70), ylim=(0, 20))
    fig.colorbar(artist, ax=axes[0, 0], label='E (MPa)', shrink=.7)
    x = profile.x_mm.to_numpy()
    mean, sd = profile.inverse_width_mean_E_MPa.to_numpy(), profile.inverse_transverse_sd_MPa.to_numpy()
    edges = [0., *meta['configuration']['REGION_BOUNDARIES_X'], meta['configuration']['GEO_L']]
    axes[0, 1].stairs(regions.mean_E_MPa, edges, baseline=None, label='Independent calibration', color='black')
    axes[0, 1].plot(x, mean, label='Assumed +Y: 50% to 90%', color='#0072B2')
    axes[0, 1].fill_between(x, mean-sd, mean+sd, color='#0072B2', alpha=.18,
                           label='Transverse spatial SD')
    axes[0, 1].plot(reverse_profile.x_mm, reverse_profile.inverse_width_mean_E_MPa,
                    '--', label='Opposite direction interpretation', color='#D55E00')
    axes[0, 1].set(title='Direction is not established by inverse fitting', xlabel='Assumed x (mm)', ylabel='E (MPa)')
    axes[0, 1].legend(fontsize=8, frameon=False)
    for offset, model, color in [(-.17, 'forward', '#666666'), (.17, 'inverse', '#0072B2')]:
        part = metrics[metrics.model == model]
        axes[1, 0].bar(np.arange(2)+offset, 1000*part.weighted_rms_mm, width=.32,
                       color=color, label='Independent calibration' if model == 'forward' else 'Inverse')
    axes[1, 0].set(xticks=[0, 1], xticklabels=['180N: fit', '280N: held out'],
                   ylabel='Two-component RMS (micrometers)', title='Same DIC points and three rigid modes')
    axes[1, 0].legend(frameon=False, fontsize=9)
    axes[1, 1].semilogx(validation.gamma, 1000*validation.weighted_rms_mm, 'o-', color='#0072B2')
    axes[1, 1].set(title='Buffered spatial validation on 180N only', xlabel='Regularization gamma',
                   ylabel='Validation RMS (micrometers)')
    fig.suptitle('Revised DIC: exploratory registration assumptions; physical validation pending', fontsize=12)
    for suffix in ('png', 'svg'):
        fig.savefig(args.output / f'overview.{suffix}', dpi=250)
    plt.close(fig)
    metric_lines = '\n'.join(
        f"| {r['case']} | {'独立标定正演' if r['model']=='forward' else '二维反演回代'} | "
        f"{r['weighted_rms_mm']:.7f} | {r['rmse_ux_mm']:.7f} | {r['rmse_uy_mm']:.7f} |"
        for r in metrics.to_dict(orient='records'))
    region_lines = '\n'.join(
        f"| {r.infill_percent:.0f}% | {r.mean_E_MPa:.1f} | {r.inverse_mean_E_MPa:.1f} | "
        f"{reversed_regions.iloc[i].inverse_mean_E_MPa:.1f} |"
        for i, r in enumerate(regions.itertuples()))
    linearity = sensitivity_meta['load_proportionality']
    initial = sensitivity[(sensitivity.variant == 'initial1200') & (sensitivity.case == '280N')].iloc[0]
    coarse = sensitivity[(sensitivity.variant == 'coarse_mesh') & (sensitivity.case == '280N')].iloc[0]
    nu = sensitivity[(sensitivity.variant == 'poisson045') & (sensitivity.case == '280N')].iloc[0]
    revision2 = sensitivity[(sensitivity.variant == 'revision2') & (sensitivity.case == '280N')].iloc[0]
    selected = meta['selected_gamma']
    markdown = f'''# 新版 DIC 的条件性正反演与独立标定正演对照

日期：2026-10-19。新版 `_3` 数据的有效宽度为 18.5257 mm、长度为 61.3664 mm，已完成真实 DIC 的自由二维反演及留出载荷预测。优化收敛且位移失配下降，但材料方向与物理标线位置尚未独立核验，跨载荷和横向收缩也存在系统不一致，因此尚不能认定已恢复真实五区模量或通过实验物理验证。

## 数据、参考态与空间范围

主数据为 `post_tensile_test/1016/sample_2/3/180N_3` 和 `280N_3`，分别采用阶段 259—273、436—450，各 15 帧、13,845 个共同测点。180N_3 额外导入的阶段 249—258 不混入已配对窗口。X/Y/DX/DY 均按 mm 处理；初始参考坐标为 X−DX、Y−DY，平台内及两平台共享参考坐标的数值一致性已检查。参考图定义沿用现有初始未加载图记录，未独立读取原始参考照片。

180N、280N 为机台平台示值，窗口实际拉力分别为 119.77618、219.73545 N，已扣除约 60.1 N 夹具偏置；不是把 180/280 N 直接作为牵引合力。时间同步沿用既有事件配对，原照片时间戳尚未独立核验。

`_2` 数据宽度为 19.4355 mm、长度为 61.4974 mm，各 14,484 个共同点。相同网格 ID 下，`_3` 的参考坐标及位移均约为原版的 0.499768864 倍，参考坐标缩放拟合最大分量残差约 2.83e−8 mm。新旧空间梯度基本保持，因此修正尺寸不意味着材料参数必然随之正确。该比例仅描述新旧导出关系，本研究没有再次缩放输入。

## 配准假设与力学模型

保持真实数据单位尺度为 1。探索性假设点云中心对应物理 70×20 mm 域中心，且原始 +Y 从 50% 指向 90%。基准变换为 A=[[0,1],[1,0]]，b={np.round(meta['offset_mm'], 8).tolist()} mm；坐标和位移使用同一 A，b 仅作用于坐标。所有平台及 `_2` 使用由 180N_3 定义的同一变换，不逐平台重新居中。

这项居中是**显式位置假设**，不是已核验的标线对应。数据覆盖约 x=4.317—65.683、y=0.737—19.263 mm，未观测边带由模型和正则延拓。另检查轴向偏移 ±4 mm 和相反材料方向。程序始终保存 `coordinates_verified=false`，原 `experiment_config.py` 的正式配准开关未改变。

采用二维平面应力、70×20×3.2 mm、双端均布牵引、三个刚体约束。独立正演用 70×20 单元、五区独立标定模量在 Gauss 点赋值，界面为 x=11/27/43/59 mm。反演用 35×10 单元、396 个独立二维节点模量，未绑定不同 y 行。主计算 ν=0.30、初值 E=1700 MPa；残差只消去两平移和一微小转动。

## 参数选择及误差定义

采用仿射去趋势梯度正则，仅在 180N 上预设 gamma=100/1000/10000 三个候选。6 mm 空间块划分验证点并从训练集剔除距验证点不超过 1 mm 的缓冲区，训练/验证分别为 8,198/3,520 点，其余 2,127 点为缓冲点。每个候选均从同一均匀场初始化，取已收敛候选中验证 RMS 最小者 gamma={selected:g}，再用完整 180N 场重算；280N 不参与选参。

最优候选位于扫描上边界，不能声称找到了全局最优正则；当前结果是预设候选内的探索性选择。空间分块和缓冲减轻近邻泄漏，不能证明误差已独立。权重为实际测点单元面积除以固定工作尺度 (0.0003 mm)²；没有把时间 SD 除以 √15，也未使用完整空间协方差。

下表 RMS 为双分量等面积 RMS；RMSE 分别对应 x 拉伸、y 横向分量。固定面积及固定分量权重下，它与点等权双分量 RMS 相同。所有模型在相同真实测点比较，允许各自拟合三个刚体模式。指标用于描述失配，不附加工程合格阈值。

| 平台 | 模型 | 双分量 RMS (mm) | 轴向 RMSE (mm) | 横向 RMSE (mm) |
| --- | --- | ---: | ---: | ---: |
{metric_lines}

主反演 {meta['solver']['n_iterations']} 次迭代收敛。这里的收敛指优化器停止条件满足，不构成模型正确或模量唯一的证明。

## 模量与方向歧义

| 假设分区 | 独立标定 E (MPa) | 假设 +Y 指向 90% 的区域均值 | 相反方向解释的区域均值 |
| --- | ---: | ---: | ---: |
{region_lines}

两个方向对应自由二维场的精确镜像，预测残差相同。因而不能用反演失配确定填充率方向；相反方向虽改变与标定的排序对应，也不能替代物理标记。表内数值是条件性识别值；与独立标定的差异不是已知真值误差。横向及区域内空间 SD 表示场的空间变化，不是统计置信区间。

## 独立一致性检查及敏感性

实际力比为 {linearity['force_ratio']:.8f}。用 280N DIC 减去该比例乘以 180N DIC，再去除三个刚体模式，得到双分量 RMS={linearity['rms_both_components_mm']:.8f} mm，轴向/横向分别为 {linearity['rmse_components_mm'][0]:.8f}/{linearity['rmse_components_mm'][1]:.8f} mm。这个检验不依赖有限元模量或材料区顺序。比例载荷下同一线弹性模量场无法同时精确重现这两个观测场；尚不能仅由该差异区分 DIC 系统误差、参考态误差、载荷历史或真实非线性。

新版两平台去刚体和趋势后的逐点时间 SD 空间中位数分别约 DX/DY=0.000155/0.000192、0.000178/0.000198 mm。它们描述短时波动，不能估计共享参考误差或给出绝对误差上限。上述跨平台失配明显高于该时间波动量级，但没有据此构造未验证的显著性或合格判据。

全场仿射描述给出的轴向/横向应变比对应表观收缩比约 0.528、0.628；这不是独立泊松比测量。ν=0.30 的各向同性模型可能不充分，也可能存在光学、参考态或投影影响。

- 初值从 1700 改为 1200 MPa，保持模量归一化尺度与 gamma 不变，280N RMS 为 {initial.weighted_rms_mm:.8f} mm，与主结果基本一致。
- 反演网格改为 28×8 单元、261 个自由参数，280N RMS 为 {coarse.weighted_rms_mm:.8f} mm；这是单次网格敏感性，不是网格收敛证明。
- 条件性轴向位置平移 ±4 mm 后均收敛，位移失配基本不变，但物理分区均值与对应位置会改变，不能据此定位标线。
- 用 `_2` 处理版本重算，280N RMS 为 {revision2.weighted_rms_mm:.8f} mm。它属于同次图像数据的处理敏感性，不是独立重复实验。
- 固定 gamma，将 ν 改为 0.45 后，280N RMS 为 {nu.weighted_rms_mm:.8f} mm；该检查用于揭示模型敏感性，未将其选为新主模型，也未用留出平台重新调参。

## 结论与适用范围

真实 DIC→自由二维模量反演→留出载荷预测链路已执行，独立五区标定正演与 DIC 的定量对照也已生成。当前条件下，反演降低了识别和留出平台的位移失配，但残差超出短时波动量级、横向响应依赖泊松比假设，且方向和标线配准未定；证据不足以认定恢复了真实五区模量。

下一项决定性工作是用初始参考图的标线、自由边缘和材料方向固定物理变换，再核验横向位移、参考态及跨载荷不成比例的来源。完整狗骨与矩形截取域的边界等效、材料各向异性及跨批标定转移仍未验证；本轮未开展完整空间协方差估计、多次独立实验或完整正则区间搜索。

## 输出与复现

- 主结果：`{args.source.as_posix()}`，包括 `input_audit.csv`、空间分块选参表、输入/代码哈希及噪声诊断。
- 正反演对照：`{inverse.as_posix()}`，包括同色标 PNG/SVG、分量误差/P95、二维模量与宽度平均曲线。
- 敏感性和独立比例检查：`{args.sensitivity.as_posix()}`。
- 总览：`{(args.output / 'overview.png').as_posix()}`，另存同源 SVG。结果目录不随 Git 分发，原始数据不改写。

```powershell
$env:TMP = (Resolve-Path results/test_temp).Path
$env:TEMP = $env:TMP
& C:/Users/zhu_g/.conda/envs/abqinter/python.exe -m unittest discover -s tests
& C:/Users/zhu_g/.conda/envs/abqinter/python.exe -m script.analyze_revised_dic --output results/dic_revision_replication_01
& C:/Users/zhu_g/.conda/envs/abqinter/python.exe -m script.check_revised_dic_sensitivity --source results/dic_revision_replication_01 --output results/dic_revision_sensitivity_replication_01
& C:/Users/zhu_g/.conda/envs/abqinter/python.exe -m script.report_revised_dic --source results/dic_revision_replication_01 --sensitivity results/dic_revision_sensitivity_replication_01 --output results/dic_revision_report_replication_01
```

Python 可替换为安装项目依赖的解释器；临时目录应事先创建。各计算输出必须是新目录。所有反演共用现有有限元与伴随内核，没有另写力学或梯度实现。
'''
    (args.output / 'report.md').write_text(markdown, encoding='utf-8')
    content = '<!doctype html><meta charset="utf-8"><title>新版 DIC 正反演结果</title>'
    content += '<style>body{max-width:1250px;margin:30px auto;padding:0 20px;font:16px/1.7 "Microsoft YaHei",sans-serif;color:#17212b}img{width:100%}pre{white-space:pre-wrap;background:#f5f7f9;padding:20px}a{color:#075b95}</style>'
    content += '<h1>新版 DIC 正反演结果</h1><p><b>配准含明确位置与方向假设，物理验证待完成。</b></p>'
    content += '<img src="overview.png" alt="总览">'
    for name in ('180N', '280N'):
        relative = Path('..') / args.source.name / 'inverse_primary' / f'{name}_forward_inverse.png'
        content += f'<h2>{name} 位移与残差对照</h2><img src="{relative.as_posix()}">'
    content += '<h2>完整报告</h2><pre>' + html.escape(markdown) + '</pre>'
    (args.output / '打开查看结果.html').write_text(content, encoding='utf-8')
    manifest = {'date': '2026-10-19', 'primary_revision': '_3', 'raw_files_modified': False,
                'registration_verified': False, 'source_metadata_sha256': file_sha256(inverse / 'run_metadata.json'),
                'calibration_source': {'path': meta['configuration']['CALIBRATION_CSV'],
                                       'sha256': file_sha256(Path(meta['configuration']['CALIBRATION_CSV']))},
                'force_pairing_source': {'path': meta['configuration']['FORCE_PAIRING_CSV'],
                                         'sha256': file_sha256(Path(meta['configuration']['FORCE_PAIRING_CSV']))},
                'calibration_groups': regions[['infill_percent', 'batch', 'n', 'mean_E_MPa',
                                                'sd_E_MPa', 'fit_method']].to_dict(orient='records'),
                'inputs': {revision: {name: fields['sources'] for name, fields in platforms.items()}
                           for revision, platforms in meta['platforms'].items()}}
    (args.output / 'dic_revised_input_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(args.output.resolve())


if __name__ == '__main__':
    main()
