# DIC 位移与二维模量反演

**查看当前结果：双击根目录 [打开结果.html](打开结果.html)。** 页面直接进入当前实验，列出四步保存位置、云图、位移误差、模量对照及数值文件。每个步骤结束时也会打印绝对路径。

本项目的主线是：**独立材料标定 → 分区正演与 DIC 对照；DIC → 自由二维 E(x,y) 反演 → 与独立标定对照；识别场回代及留出载荷预测。** 反演不绑定不同 y 行，也不以五区模量作为强先验。默认 180N 识别、280N 留出预测，不用留出平台调参。

## 四个独立步骤

安装 `requirements.txt`，在项目根目录依次执行：

```powershell
python -m script.step1_prepare_dic
python -m script.step2_forward_calibrated
python -m script.step3_inverse_dic
python -m script.step4_compare_results
```

换一批数据时，修改 [dic_case_config.py](dic_case_config.py) 中的输入、载荷配对、坐标与物理定义，并将 `CASE_OUTPUT` 改为新目录。四个脚本各自执行一步，不自动串行调度；后续计算读取第一步冻结的数据。已有输出拒绝覆盖。参数、输入约定与逐项文件说明见 [分步骤流程](docs/dic_stepwise_workflow.md)。

| 步骤 | 保存目录（均在 CASE_OUTPUT 下） | 主要文件 |
| --- | --- | --- |
| DIC 准备 | `01_prepared/` | `input_audit.csv`、`fields.npz`、配置与输入记录 |
| 独立正演 | `02_forward/` | `comparison.csv`、各 ν 的 `fields.npz` |
| DIC 反演 | `03_inverse/` | `selection.csv`、`regional_comparison.csv`、各 ν 的模量及位移 |
| 比较出图 | `04_comparison/` | `打开查看结果.html`、CSV、各 ν/平台的 PNG/SVG |

当前实验目录为 `results/dic_case_commonmesh_20261019_01/`；按原 post_DIC 风格重出的图表位于其中 `04_comparison_postdic/`。实验根目录的 `结果总览.html` 指向当前图表版本。重新出图只运行第 4 步并指定 `--output <新目录>`，不需要重算正反问题。

## 模型与当前结论范围

制造试样总长 220 mm、平行段 150×20 mm、厚 3.2 mm，制造七区。当前截取域为中央 70×20 mm，覆盖 50/60/70/80/90% 五区，长度 11/16/16/16/11 mm；物理域不由 DIC 有效点云跨度定义。

当前新版 DIC 点距约 0.2894643 mm，有效覆盖约 61.3664×18.5257 mm；坐标和位移均为 mm，不缩放跨度。坐标及位移共用 rot90 变换，点云居中和材料区位置仍是显式假设。独立标定沿用 `independent` 口径，保留跨批来源、样本数和标准差。

正反问题共用 70×20 个 1×1 mm Q4 单元、1491 个独立节点模量。单位 mm/N/MPa，平面应力；两端均布牵引，三个约束仅固定有限元刚体规范。模型与 DIC 在测点空间统一处理两平移和一微小转动。实际力约 119.77618/219.73545 N，已扣夹具初始偏置，位移参考初始未加载图。主 ν 预设 0.26，并比较 0.24/0.28。

**现有结果属于条件性比较。** 优化收敛与位移拟合改善不代表物理验证完成；材料区定位、完整狗骨与矩形域边界等效、跨批标定转移及误差预算仍需独立核验。数值比较见 [当前结果记录](docs/dic_shared_mesh_nu_comparison_20261019.md)。

## 资料与代码位置

- [实验方案](plan/dogbone_parallel_section_adjoint_plan.tex)、[验证方案](plan/validation_plan.md)、[拉伸数据与标定](post_tensile_test/README.md)。
- `fgm_asm/`：共用有限元、观测与伴随内核；`lib/`：数据处理、四步流程及绘图；`script/`：四个入口；`tests/`：解析、梯度和端到端验证。
- [架构与接口](docs/architecture.md)、[坐标说明](docs/dic_coordinate_registration.md)、[验证记录](docs/validation.md)。
- [历史研究与辅助验证](archive/legacy_workflows/README.md)：原合成、正则比较、早期 DIC 试算入口及报告。历史结果和原始试验数据仍保留在各自目录。

验证代码：`python -m unittest discover -s tests`。通过代码测试不等于实验物理模型成立。
