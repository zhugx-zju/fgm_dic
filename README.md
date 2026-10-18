# 非均匀试样的 DIC–伴随模量反演

本项目用**二维自由节点模量场 E(x,y)** 识别实际沿拉伸方向变化的 **一维 E(x)**。合成数据检验算法，独立分区模量用于正演预测，真实 DIC 检验实验可行性。三条流程共用双端均布牵引的二维有限元、测点观测算子与伴随反演。科学方案见 [实验与反演方案](plan/dogbone_parallel_section_adjoint_plan.tex)。

## 实际几何与反演维度

打印试样总长 220 mm，平行段 150 × 20 mm，厚 3.2 mm，制造分区仍是七区。实际两条标线各距试样最外端 75 mm，因此 DIC 覆盖中央 **70 × 20 mm**：

| 物理坐标 x (mm) | FEM x (mm) | 填充率 | 观测长度 (mm) |
| --- | --- | --- | --- |
| −35 至 −24 | 0 至 11 | 50% | 11 |
| −24 至 −8 | 11 至 27 | 60% | 16 |
| −8 至 8 | 27 至 43 | 70% | 16 |
| 8 至 24 | 43 至 59 | 80% | 16 |
| 24 至 35 | 59 至 70 | 90% | 11 |

五区定义合成真值、独立标定正演和后处理统计。默认 35 × 10 单元反演包含 **396 个独立节点模量**，不绑定横向节点。输出完整二维场、宽度平均 E(x)、横向空间标准差和五区统计；空间标准差不是统计置信区间。

## 目录

沿用原项目的分类和 Python 配置方式：

```text
fgm_asm/             网格、材料、正演、观测算子、伴随反演与正则
lib/                 DIC/标定数据处理、研究流程、结果保存和比较图
script/              统一运行入口及原有算法验证脚本
tests/               解析解、梯度、数据导入及端到端测试
config.py            原 110 mm 合成研究配置，供历史研究复现
synthetic_config.py  当前 70 mm 域的一维真值、二维反演验证
experiment_config.py 第 3 次实验、五区独立标定及 DIC 配置
results/             每次计算的独立输出，运行时生成，不入 Git
docs/                算法说明、迁移和验证记录
plan/                实验方案与整体架构说明
post_tensile_test/   原始试验、分区模量、时间配对及选定 DIC 场
3D_print/            试样 STL、生成脚本与切片设置
```

## 环境与运行

在项目根目录安装 `requirements.txt` 后运行：

```powershell
python -m unittest discover -s tests
python -m script.run_synthetic
python -m script.run_experiment --mode inspect
python -m script.run_experiment --mode model
```

- `run_synthetic`：在独立细网格生成一维分区真值的位移，加入测点采样、噪声和刚体扰动，识别二维模量，预测留出载荷。`TRUTH` 还支持 `uniform`、`linear_1d`、`exponential_1d`。
- `inspect`：读取 180N/280N 各 15 帧，恢复参考坐标，检查重复点和共同有效测点，保存平台均值、时间标准差、哈希、力配对及独立标定。
- `model`：采用实测 50%—90% 五组模量和实际拉力，在 70 × 20 × 3.2 mm 模型上独立正演。

核验 DIC 坐标尺度、轴方向和标线定位后，在 `experiment_config.py` 填写 `DIC_TO_FEM_MATRIX`、`DIC_TO_FEM_OFFSET`，设置 `DIC_COORDINATES_VERIFIED=True`，运行：

```powershell
python -m script.run_experiment --mode forward
python -m script.run_experiment --mode inverse
```

`forward` 在真实 DIC 测点比较独立标定正演位移，输出两个分量的预测、观测与残差图。`inverse` 默认用 180N 识别二维场，以 280N 检验预测，再与独立标定比较。独立标定不作为主反演强先验。

用 `--config <Python模块名>` 选择配置，`--output <新目录>` 指定输出；已有输出拒绝覆盖。每次保存配置快照、`run_metadata.json`、`fields.npz`，记录输入/代码哈希、版本、种子、Git 状态与命令。图保存为 PNG/SVG；模量比较另存 `x_profile.csv`。优化未收敛仍保留诊断，CLI 返回退出码 2。

## 力学与数据约定

- 单位 mm、N、MPa。两端等大反向均布牵引 ±F/(w·t)，侧边自由，只约束三个刚体自由度。配准仅允许平移和微小转动。
- 合成与实验共用 `ObservationCase` 及面积/噪声加权测点残差。历史完整节点质量矩阵目标仍供原有研究使用。
- 节点未知量 `m=log(E)`；保留 H1、仿射去趋势和 Huber TGV 正则。实验 `GAMMA` 与噪声下限尚未独立选定。
- 机台初始约 60.1 N 为夹具自重偏置。DIC 相对最初未加载图，180N/280N 实际拉力约为 119.77618/219.73545 N，来自既有时间配对表。机台位移和变形通道不参与分析。
- 平台帧标准差作为诊断，不自动除以 √15；当前权重未包含空间相关协方差。
- 分区标定来自 `mixed_infill_20261017` 的 `independent` 方法，保留批次、样本数和标准差。七组原始标定保留，当前 ROI 选用其中五组。

实际 ROI 长度已确认，但导出坐标跨度约 37.07 × 122.79 与 20 × 70 mm 尚未建立可核验的坐标关系。不能独立缩放两轴强行匹配。当前可做输入质检和独立正演，正式 DIC 比较及反演仍需核验导出尺度。

同一线弹性工况下成比例载荷可检验预测和重复性，不会自动增加空间可辨识性。完整狗骨与矩形域边界近似、泊松比及跨批标定转移仍需独立验证。

## Git 与历史

整个目录由根 `.git` 管理，提交身份为 `zhugx <zhugx@zju.edu.cn>`，目标仓库为 [fgm_dic](https://github.com/zhugx-zju/fgm_dic)。原 ASM 历史保留，旧远端名为 `asm-history`。

本次使用的 30 个 DIC CSV、原始试验及 3D 打印文件纳入项目；DIC 哈希见 `post_tensile_test/1016/sample_2/3/dic_input_manifest.json`。缓存和新计算输出不入 Git。

重复/无关的一维方法目录、旧 ASM-Python/post_DIC 目录及旧计算输出已移出工作目录，迁移前代码可从 `696959c` 恢复。清理清单、外部归档与验证记录见 [迁移记录](docs/project_unification.md)，架构交接见 [整体方案](plan/project_unification_plan.md)。
