# 加载过程与 DIC 时刻选取图

本图对应上一级 `结果总览.html` 的 `sample2_run3_marks_73_85_minusY`，为第 **3** 次试验，不是原参考图片中的第4次试验。仅重新绘制加载及选时信息，不重新计算或改变正反问题结果。

## 数据来源与核验

- 结果身份、计算输入及载荷：`../01_prepared/run_metadata.json`。
- 试验机原始记录：项目根目录下 `post_tensile_test/1016/sample_2/sample_2_3.xlsb`。
- DIC 两点位移时间序列：项目根目录下 `post_tensile_test/1016/sample_2/DY_3.csv`。
- 已有同步记录：`post_tensile_test/1016/sample_2/analysis_results/dic_timing_20261018/timing_checks.json`；载荷配对表为同目录 `selected_dic_windows.csv`。
- 全场输入：`post_tensile_test/1016/sample_2/3/180N_2/region/stage_00259.comdat_comdata.csv` 至 `stage_00273.comdat_comdata.csv`，以及 `3/280N_2/region/stage_00436.comdat_comdata.csv` 至 `stage_00450.comdat_comdata.csv`。

重绘时核对了原始力/两点DIC记录的历史SHA-256，以及上述30份全场CSV与结果冻结记录的SHA-256。重新插值得到的两个窗口平均实际拉力与冻结结果一致，绝对容差为 `1e-10 N`。未重新拟合平台或调整选帧。

| 用途 | 试验机示值平台 | DIC CSV 阶段号（从0开始） | 实际采用 | 可选单帧 | 估计试验机时间窗口 (s) | 平均试样拉力 (N) |
| --- | --- | --- | --- | --- | --- | --- |
| 识别 | 180 N | 259–273 | 15帧平均 | 266 | 46.1812–48.9812 | 119.776179 |
| 留出预测 | 280 N | 436–450 | 15帧平均 | 443 | 81.5812–84.3812 | 219.735451 |

当前结果采用15帧平均场，不是只采用图上标出的可选单帧。阶段号是导出CSV的索引，不保证等于原照片编号。

## 图件与复现

- `loading_dic_selection_zh.png` / `.svg`：中文 PPT 版本。
- `loading_dic_selection_en.png` / `.svg`：英文论文版本。
- PNG为4800×2700像素、400 dpi；SVG为同一最终figure状态输出的矢量图，文字转为路径，便于跨电脑插入PPT。
- `timing_curve_data.csv`：原始两点位移、均值、估计机器时间、插值力及选帧标记。机器记录以外的力为NaN，不外推。
- `figure_metadata.json`：来源哈希、环境、时间映射、核验结果及限制。

从项目根目录运行：

```powershell
python results/dic/loading_figure_20261021/plot_loading_selection.py
```

绘图程序依赖numpy、matplotlib与pyxlsb；全部路径由文件位置确定。原始记录、计算输入与结果均保持原状。

## 建议图注

**中文：** 加载过程与DIC时刻选取。（a）第3次试验的实测试验机力示值；各平台约保载10 s，再加载段速率约4 N/s。（b）同一横截面两点平均轴向位移随估计试验机时间的变化。蓝色与橙色阴影分别表示识别和留出预测窗口，按DIC拟合平台入口后6–9 s选取；5 Hz下分别对应259–273和436–450阶段，各15帧。结果采用窗口平均位移场；菱形表示可选单帧266和443。扣除60.1 N夹具偏置后，窗口平均试样拉力分别为119.78和219.74 N。

**English:** Loading history and DIC frame selection. (a) Measured machine force readings in run 3, with approximately 10 s holds and 4 N/s loading ramps. (b) Mean axial displacement of two points on the same cross-section, plotted against estimated machine time. Blue and orange bands denote the identification and held-out prediction windows, selected 6–9 s after the fitted DIC plateau entry. At 5 Hz, these windows comprise stages 259–273 and 436–450, respectively (15 frames each). Window-averaged displacement fields are used in the reported results; diamonds indicate optional individual stages 266 and 443. Subtracting the 60.1 N fixture offset gives mean specimen forces of 119.78 and 219.74 N.

## 解释范围

同步沿用既有事件拟合：`t_machine = stage / 5 − 5.618778938656367 s`，尚无照片时间戳的独立核验。图中DIC只展示与机器记录共同覆盖的非负机器时间范围，未展示其余尾部，也没有延长载荷记录。两个同截面点的共同位移用于选时，包含刚体运动，不是试样伸长，不能单独证明全场应变稳定或DIC绝对精度。
