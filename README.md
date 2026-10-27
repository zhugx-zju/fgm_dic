# FGM DIC：配置驱动的正反问题流程

只修改根目录 **`config.py`**，分别执行材料标定、DIC选帧和四步正反问题分析。采用ASM-Python的“配置—入口—共用内核”组织方式；默认自由二维H1，保留Huber-TV及精确TV选项。

```powershell
python -m pip install -r requirements.txt
python -B -m script.calibrate_material
python -B -m script.select_dic_frames
# 按导出清单在DIC软件导出全场；自带合成示例已含所需文件。
python -B -m script.step1_prepare_dic
python -B -m script.step2_forward_calibrated
python -B -m script.step3_inverse_dic
python -B -m script.step4_compare_results
```

在项目根目录执行，Python≥3.10。自带 `data/example/` 全部为合成输入，不读取私人实验或历史结果。新计算修改 `RUN_NAME`，已有数值目录拒绝覆盖。结果入口为 `results/<RUN_NAME>/结果总览.html`。

| 位置 | 用途 |
|---|---|
| `config.py` | 输入/输出路径、标定、选帧、载荷、几何、坐标、网格、正则和求解参数 |
| `script/` | 上述六个独立入口，无总batch |
| `fgm_asm/` | 共用Q4有限元、观测、伴随、H1/Huber-TV/精确TV内核 |
| `lib/` | 输入读取与核对、标定、选帧、主流程和结果保存 |
| `data/example/`、`data/local/` | 合成示例、Git忽略的私人输入清单 |
| `results/<RUN_NAME>/` | 本次配置快照、输入哈希、数值场、扫描、图表与报告 |
| `research/` | 顺序多工况、一维/分层研究索引、裁剪、γ敏感性和旧报告模块 |

[输入格式与选帧依据](docs/public/inputs.md) · [运行命令与结果规范](docs/public/running.md) · [架构与算法定位](docs/public/architecture.md) · [本次核验](docs/public/verification.md)

**每步另有对应的结果代码**：`plot_calibration_results.py`、`plot_frame_selection_results.py`、`plot_prepared_dic.py`、`plot_calibrated_forward.py`、`plot_inverse_results.py`、`plot_comparison_results.py`。例如 `python -B -m script.plot_inverse_results` 只读取保存的反演场并出图。完整计算/结果对应表见运行文档。

真实实验可以选择 `CASE_PROFILE='experiment'`，逐次配置1026 sample_2全段；其历史平台缺同期力，预设明确保留用户已接受的载荷假设。更换试样必须重新填写本次数据、事件区间、参考力、截面与坐标，不套用历史帧号。输入格式与各项条件见上述文档。已有数据、结果、打印文件保留原路径。

独立标定不进入反演初值或γ选择；L-curve没有有效拐点则保存失败，不强行补γ。位移相对误差以DIC变形量为分母，同平台回代属于拟合检查，物理验证另行评价。

纯源码分享：运行 `python -B tools/export_source.py` 生成带合成示例的 `dist/fgm_dic_source.zip`。该包按白名单排除私人实验、历史结果、专题研究与Git历史；不直接把当前实验仓库历史作为公开包。重建示例可运行 `tools/generate_example.py --output <新目录>`，种子和生成器哈希保存在示例中。
