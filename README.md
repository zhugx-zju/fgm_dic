# DIC 位移与二维模量反演

**材料方向已统一更新：** 主流程采用原始+Y方向模量递减，见[更新记录](docs/material_direction_update_20261020.md)。主正演和分区统计已更新，已有自由反演场经输入等同性核对后复用；原方向结果完整归档。

**查看当前结果：双击根目录 [打开结果.html](打开结果.html)。** 页面直接进入当前实验，列出四步保存位置、云图、位移误差、模量对照及数值文件。每个步骤结束时也会打印绝对路径。

本项目的主线是：**独立材料标定 → 分区正演与 DIC 对照；DIC → 自由二维 E(x,y) 反演 → 与独立标定对照；识别场回代及留出载荷预测。** 反演不绑定不同 y 行，也不以五区模量作为强先验。默认 180N 识别、280N 留出预测，不用留出平台调参。

主线比较的参考约定见 [DIC—有限元坐标、位移基准与边界一致性](docs/dic_reference_consistency.md)：区分物理坐标配准、有限元三个刚体约束、残差刚体拟合与绘图居中，说明真实伸长为何保留、DIC 与 FEM 两套网格怎样连接，并列出当前代码入口和核对数值。

## 四个独立步骤

安装 `requirements.txt`。当前配置采用73/85 mm标线中点、原始−Y递增方向和独立毫米尺度，输出到 `results/dic/`。50个对数等距γ覆盖10⁻¹²—10⁻⁴，180N选参、280N固定预测；结果和实际停止条件见[几何修正记录](docs/material_direction_update_20261020.md)。新实验在项目根目录依次执行：

```powershell
python -m script.step1_prepare_dic
python -m script.step2_forward_calibrated
python -m script.step3_inverse_dic
python -m script.step4_compare_results
```

换一批数据时，修改 [dic_case_config.py](dic_case_config.py) 中的输入、载荷配对、坐标与物理定义，将 `CASE_OUTPUT` 改为新目录，然后去掉 `--plot-only` 依次运行四个脚本。每一步计算后自动生成对应云图；后续计算读取第一步冻结的数据。已有数值输出拒绝覆盖。参数及逐项文件说明见 [分步骤流程](docs/dic_stepwise_workflow.md)。

| 步骤 | 保存目录（均在 CASE_OUTPUT 下） | 主要文件 |
| --- | --- | --- |
| DIC 准备 | `01_prepared/` | `云图.html`、DIC 位移云图、冻结输入与核查记录 |
| 独立正演 | `02_forward/` | `云图.html`、标定模量、各 ν 的 DIC／正演／相对误差对照、数值场 |
| DIC 反演 | `03_inverse/` | `云图.html`、二维模量、E(x)、DIC／回代／相对误差对照、识别与预测记录 |
| 比较出图 | `04_comparison/` | `打开查看结果.html`、CSV、各 ν/平台的 PNG/SVG |

浏览入口指向当前递减方向的四步结果，含L-curve、自由二维模量及正演/DIC/反演回代对照。ν=0.33补充图在`results/dic/direction_update/nu_0.33/`。旧方向、旧网格及探索性诊断集中在`results/dic/history/previous_runs.zip`，版本条件与索引见[历史结果](docs/results_history.md)。每步更新同组`结果总览.html`；`--plot-only`只更新图表，不重算正反问题。

误差云图统一采用百分比：位移误差以对应平台 DIC 去除三个刚体分量后的分量 RMS 归一化，避免逐点除以过零位移；原始残差仍以 mm 保存。第 2 步直接展示正演与 DIC 的并排对照，无需等到第 4 步。

反演目标为 `0.5*r.T@W_area@r + 0.5*gamma*integral(|grad E|^2)`，不设置参考模量，不作模量或人为噪声尺度归一化。gamma 仅用识别平台的 L-curve 拐点选择，扫描范数、曲率、停止信息和 PNG/SVG 保存于第 3 步；无可辨识拐点时停止并保留诊断。求解容差来自第 1 步冻结配置，源配置修改后，需重新准备输入才能生效。

## 模型与当前结论范围

**2026-10-20 几何与方向已修正：** 73 mm为高填充率端、85 mm为低填充率端，region两端对应横线，填充率沿原始−Y递增。标线制造坐标−25/+37 mm、中点+6 mm；独立导出跨度61.497354 mm与约62 mm相差0.502646 mm，采用中点配准并保留差异。详见[当前材料坐标](docs/dic_roi_geometry.md)。

当前读取`180N_2/region`及`280N_2/region`，恢复初始坐标后使用`x=+Y₀+bₓ、y=−X₀+bᵧ、ux=DY、uy=−DX`。计算域61.497354×19.435484 mm，DIC为213×68点、点距约0.290081857 mm；真实承载宽度20 mm、厚3.2 mm。五区界面在FEM坐标中为12.748677/28.748677/44.748677/60.748677 mm，制造位置不缩放。独立标定仍采用`independent`口径，各区模量、批次和样本数不变。

正反问题共用70×20 Q4单元、1491个独立模量节点，单元约0.878534×0.971774 mm。平面应力、两端牵引F/(20t)、内部横向自由近似及三个刚体规范保留；随观测宽度更新截取域端力，DIC在真实测点比较。60.1 N偏置及未加载参考保持既定配对。泊松比允许范围0.24—0.33，主值0.26不是已独立测得的材料值；旧E—ν诊断按其原几何解释。

`all_region`已取得约117.84 mm范围，当前仅核对数据覆盖及邻近采样，不替代已确认横线的region。完整80 mm五区及150 mm七区选域需进一步定位，不能把本轮region称为完整五区或七区。

**现有结果属于条件性比较。** 优化收敛与位移拟合改善不代表物理验证完成；材料区定位、完整狗骨与矩形域边界等效、跨批标定转移及误差预算仍需独立核验。数值比较见 [当前结果记录](docs/material_direction_update_20261020.md)。

## 资料与代码位置

- [实验方案](plan/dogbone_parallel_section_adjoint_plan.tex)、[验证方案](plan/validation_plan.md)、[拉伸数据与标定](post_tensile_test/README.md)。
- `fgm_asm/`：共用有限元、观测与伴随内核；`lib/`：数据处理、四步流程及绘图；`script/`：四个入口。
- [架构与接口](docs/architecture.md)、[坐标说明](docs/dic_coordinate_registration.md)。
- `3D_print/`：制造几何；`post_tensile_test/`：原始实验与独立标定；`results/`：各组计算结果。

实现核验采用临时解析算例、梯度差分和四步试跑，不保留独立测试源码。通过实现核验不等于实验物理模型成立。
