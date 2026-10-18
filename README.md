# 非均匀试样的 DIC–伴随模量反演

项目目标见 [实验与反演方案](plan/dogbone_parallel_section_adjoint_plan.tex)。合成数据用于检查算法，独立拉伸标定用于正向预测，真实 DIC 用于检验反演的实验可行性。三者共用一套双端均布牵引的二维有限元与伴随梯度。

## 目录

沿用原项目的分类与 Python 配置方式：

```text
fgm_asm/             网格、材料、正演、观测算子、伴随反演与正则
lib/                 DIC/标定数据处理、研究流程、结果保存和比较图
script/              运行入口及原有合成研究脚本
tests/               解析解、梯度、数据导入及端到端测试
config.py            原合成算例的配置，保留已有研究脚本
synthetic_config.py  统一流程的合成验证配置
experiment_config.py 第 3 次实验及独立标定的配置
results/             新计算的独立输出目录，不覆盖旧结果
docs/                算法说明与迁移记录
plan/                实验方案、整体修改方案
post_tensile_test/   原始试验、分区模量标定和时间配对
3D_print/            试样几何、制造设置
1D_method_compare/   独立的一维方法对照
ASM-Python/          原合成计算结果位置，不再包含活动算法包
post_DIC/            原 DIC 数据、结果和历史说明位置
```

## 环境与运行

在项目根目录安装 `requirements.txt`，并运行：

```powershell
python -m unittest discover -s tests
python -m script.run_synthetic
python -m script.run_experiment --mode inspect
python -m script.run_experiment --mode model
```

- `run_synthetic`：在较细正演网格生成位移，经真实测点采样、噪声与刚体扰动后，以同一反演入口识别二维节点模量，并预测另一载荷级。`TRUTH` 可选均匀场、二维平滑场或七区场。
- `inspect`：读取真实 180N/280N 平台各 15 帧，恢复参考坐标，核对重复点及共同有效测点，保存平台均值、时间标准差、文件哈希、力配对和独立标定信息。
- `model`：采用独立测得的七组模量及实际拉力，在 110 × 20 × 3.2 mm 平行段模型中正演。它不依赖 DIC 注册，也不声称已经与 DIC 定量吻合。

核验 DIC 坐标尺度、轴方向及试样七区位置后，在 `experiment_config.py` 填写 `DIC_TO_FEM_MATRIX`、`DIC_TO_FEM_OFFSET`，设置 `DIC_COORDINATES_VERIFIED=True`，再运行：

```powershell
python -m script.run_experiment --mode forward
python -m script.run_experiment --mode inverse
```

`forward` 用独立标定模量预测真实 DIC 测点位移，并生成两个分量的预测、观测与残差图。`inverse` 默认只使用 180N 平台识别二维模量，以 280N 平台检验位移预测，同时输出七区内的面积加权模量统计与独立标定比较。七区统计不是七参数反演，也不是新的先验约束。

可用 `--config <Python模块名>` 选择另一组配置，`--output <新目录>` 指定输出位置；输出目录已存在时拒绝覆盖。每次保存 `run_metadata.json`、配置源文件快照和 `fields.npz`，比较流程另存 PNG/SVG。元数据含输入哈希、代码哈希、版本、随机种子、Git 状态和运行命令；配置快照用于追溯，复跑时以元数据中的完整参数与原项目位置为准。重复绘图不需要重新反演。

优化达到迭代上限或返回失败时仍保存数组和诊断信息；命令返回退出码 2，不将“得到结果文件”视为收敛成功。

## 力学与实验约定

- 单位：mm、N、MPa。两端等大反向均布牵引，侧边自由，只去除三个刚体自由度；刚体配准只允许平移和微小转动，不消除真实拉伸应变。
- 同一 `ObservationCase` 支持仿真与实验，目标为真实测点的面积/噪声加权残差。原完整节点质量矩阵目标仍保留，供已有正则研究复现。
- `m=log(E)` 保证模量为正；支持原有梯度、仿射去趋势及 Huber TGV 正则。`modulus_scale=E_REF` 明确正则尺度。实验 `GAMMA` 和噪声下限目前是占位配置，未完成独立选参。
- 试验机初始读数是夹具自重偏置。180N/280N 目录的实际试样拉力分别约 119.77618 N、219.73545 N，从既有时间配对表读取。DIC 相对最初未加载图，不能直接配约 100/200 N 的平台间增量。
- 平台帧的时间标准差保留为诊断量，不假设 15 帧独立，不自动除以平方根帧数。当前噪声权重是保守的对角近似，未建模空间相关协方差。
- 独立标定使用 `mixed_infill_20261017` 的 `independent` 方法，保留批次、样本数和标准差，不作为精确空间真值，也不强加到主反演中。

当前导出坐标的数值范围约为 37.07 × 122.79，与已确认的 20 mm 宽平行段尚未建立可核验的关系。代码不通过自动缩放或默认居中来掩盖这一差异。因此真实数据目前可读取质检、可独立正演，正式 DIC 比较与反演等待坐标核验。

## 验证范围与后续研究

代码测试包括均匀材料解析解、端部合力/厚度、三种正则的梯度、带刚体配准的多载荷伴随梯度、两种 CSV 格式、重复点与缺帧、独立标定正演和 DIC 反演的受控数据流程。

主流程保留自由二维节点场。七参数低维反演、完整狗骨到矩形域的边界偏差、空间相关噪声、泊松比与跨试样敏感性，仍需作为独立研究开展；不能由当前烟雾测试或优化器收敛推出实验模量可信。

## Git 与历史代码

整个目录现在由根 `.git` 管理，作者沿用 `zhugx <zhugx@zju.edu.cn>`。原 ASM 历史与分支保留，原远端更名为 `asm-history`，没有推送。大体积 DIC 导出、缓存和计算输出留在原磁盘位置，不放入普通 Git。

本次选用的 30 个 DIC 文件的路径、哈希和平台质检信息记录在 `post_tensile_test/1016/sample_2/3/dic_input_manifest.json`，便于在其他磁盘恢复输入后核对。

代码迁移前的整项目快照为 `696959c`，其中包含原 post_DIC 位移控制/CASE 算法、旧入口，以及当时尚未提交的 TGV 工作。迁移映射与验证记录见 [项目迁移记录](docs/project_unification.md)，后续交接见 [完整修改方案](plan/project_unification_plan.md)。
