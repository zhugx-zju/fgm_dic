# DIC 实验的分步骤计算流程

四步共用的物理坐标、位移基准、牵引边界和测点观测约定，见 [DIC—有限元参考一致性](dic_reference_consistency.md)。该说明分别定义原始位移、三个刚体分量拟合、绘图居中及真实端差，附当前代码对应与核对结果。

## 脚本及数据流

每组实验使用一个 Python 配置文件和一个独立结果目录。四个脚本各自完成一步，后续脚本读取已保存的中间文件，不用一个总入口自动串行运行所有步骤。

| 步骤 | 脚本 | 输入 | 保存结果 |
| --- | --- | --- | --- |
| 1. 准备数据 | `script/step1_prepare_dic.py` | 实验配置、DIC CSV、载荷配对、独立标定 | `01_prepared/`：冻结输入、核查表、DIC 两分量云图和 `云图.html` |
| 2. 独立正演 | `script/step2_forward_calibrated.py` | `01_prepared/` | `02_forward/`：分区标定模量、各 ν 的 DIC／正演／相对误差对照、数组与 `云图.html` |
| 3. DIC 反演 | `script/step3_inverse_dic.py` | `01_prepared/` | `03_inverse/`：L-curve 与曲率、各 ν 的二维模量、DIC／回代／相对误差对照、E(x)、数组与 `云图.html` |
| 4. 比较出图 | `script/step4_compare_results.py` | 同一 `01_prepared/` 对应的正反演结果 | `04_comparison/`：DIC/独立正演/反演回代对照、二维模量、E(x)、五区统计、误差表和 HTML 报告 |

第 2、3 步分别求解正问题和反问题；第 3 步不需要第 2 步的位移来识别模量，第 4 步才汇合两条证据链。第 4 步不重新运行有限元或优化器。

## 当前实验的运行命令

在项目根目录、安装 `requirements.txt` 的 Python 环境中，依次运行：

默认配置使用 `180N_2/region`、`280N_2/region`，按73/85 mm标线中点及材料沿−Y递增定位，保留独立毫米尺度，输出到 `results/dic/`。50点扫描及四步计算按新几何执行，结果见[当前修正记录](material_direction_update_20261020.md)。旧居中50点和17点分别在 `history/pre_marks/`、`history/lcurve17/`。新实验四步命令为：

```powershell
python -m script.step1_prepare_dic
python -m script.step2_forward_calibrated
python -m script.step3_inverse_dic
python -m script.step4_compare_results
```

本组四步均已完成；本次材料方向更新复用了输入完全相同的既有反演和50点扫描，只重算独立正演并更新分区统计与全部图表；不要重复执行数值计算覆盖冻结结果。查看既有结果或仅更新显示时使用`--plot-only`。

**查看结果：双击项目根目录的 [打开结果.html](../打开结果.html)。** 每一步结束时都会打印本步绝对保存路径和实验总览路径，并更新这个固定入口。第 2—4 步的默认 `--config dic_case_config` 仅用于定位 `CASE_OUTPUT/01_prepared`；求解参数和数据仍来自第一步冻结文件。也可用 `--input <01_prepared目录>` 明确指定输入。

全部结果集中在配置的 `CASE_OUTPUT` 下：

```text
结果总览.html                 四步保存位置、文件链接及最终图表入口
01_prepared/                 云图.html；180N/、280N/ 的 DIC 位移 PNG/SVG
  reference_audit/           已执行的坐标、刚体及显示一致性核验记录
02_forward/                  云图.html、标定模量；nu_*/平台/ DIC、正演和相对误差 PNG/SVG
03_inverse/                  云图.html；selection.csv、l_curve.*、l_curve_curvature.*
  l_curve/gamma_*/           每个 gamma 的模量、目标分解、梯度及停止信息
  nu_*/                     模量、E(x)、平台 DIC、回代和相对误差 PNG/SVG
04_comparison/               打开查看结果.html、误差与模量 CSV
  nu_0.26/                   主泊松比结果（另有 0.24、0.28）
    180N/、280N/             DIC、正演、反演及相对误差各分量 PNG/SVG，逐测点 CSV
    relative_error_summary.csv  各平台/模型/分量的归一化分母、相对 RMS 与 P95
    modulus_comparison.*     二维模量对照
    x_profile.*              宽度平均 E(x) 与标定对照
history/                     原版本四步结果及核验记录
```

当前实验统一保存在 `results/dic/`。L-curve 扫描与识别模量共用 `03_inverse/`，不再因选参方式另建并列实验目录。历史版本位于 `history/`，保留各自冻结数据和计算条件；所有浏览入口均从统一总览访问。移动整套结果时保留原元数据中的计算时路径与哈希，后续出图以实际输入目录定位，并核验冻结文件身份。

四步默认在完成数值计算后自动出图。已有与当前数值内核一致的结果时，给相应命令加 `--plot-only`，仅刷新已保存场、L-curve、展示页及派生剖面，不重读原始 DIC、不运行正演或优化器，也不覆盖冻结数组、配置或求解记录。图文件按相同文件名刷新，比较结果统一位于 `04_comparison/`。旧 `results/dic/history/` 浏览文件仍保留，但其目标函数与当前方案不同，不能绕过内核核对直接混用或继续计算。

DIC 图复用原 `post_DIC` 的 `lib/dic_preprocessing.py::plot_grid_fields`：7×6 英寸、规则网格 `pcolormesh`、`jet`、等比例坐标、独立单图。第 2、3 步直接按分量并排显示 DIC、模型及相对误差；第 4 步中 DIC/独立正演/反演共用位移色标，两幅相对误差共用对称百分比色标。新输入冻结网格 ID，缺失点留空，不插值补测量。旧版冻结结果仅在可核验的轴对齐规则坐标上恢复网格顺序。绘图不改变配准、观测、载荷或反演结果。

第 2—4 步还自动生成同一张图内的 ux/uy 四格对照（`*_forward_dic_components.*`、`*_inverse_dic_components.*`，PNG/SVG）：上排处理后的 DIC、下排配准后的有限元，左列 ux、右列 uy，每列共用色标。两排分别为 `d−mean(d)` 与 `HU+Rc−mean(d)`，位置均为真实 DIC 测点的 FEM 坐标；图下方标注该位移参考。各步骤 HTML 直接展示四格图，原分量单图、误差图及未居中的逐点 CSV 继续保留。详见[共同参考及实际核验](dic_reference_consistency.md#正演与反演的-uxuy-四格对照)。

### 相对误差的定义

先用与反演相同的观测权重，将 DIC 位移投影到两平移、一微小转动子空间，令去除这些刚体分量后的位移为 `d_def`。各平台、各分量 `c` 的固定分母为：

`s_c = sqrt(sum(w_c * d_def,c²) / sum(w_c))`，单位 mm。

误差图显示有符号 RMS 归一化相对误差 `e_c = 100 * (u_model,c - u_DIC,c) / s_c`（%）。模型已通过原有三个刚体模式配准。分母只取决于对应平台的 DIC，全部 ν、独立正演和反演共用，刚体基准变化不改变该比例；不逐点除以会过零的位移。无可分辨变形的分量标为未定义，不设置人为分母下限或裁剪误差。

图文件沿用 `*_forward_error_ux.*` 等名称，色标和标题明确改为相对误差；模量差异图原有相对百分比定义保持不变。`point_fields.csv` 保留原位移与残差 `*_mm` 列，新增 `*_relative_error_*_percent` 和分母列。各 ν 的 `relative_error_summary.csv` 保存加权相对 RMS 与逐点绝对相对误差 P95；绝对误差表仍保留 mm，供量纲核验与误差预算使用。百分比不等同于 DIC 测量不确定度。

## 换一组 DIC 数据

修改 `dic_case_config.py` 中的新数据目录、物理定义和新的 `CASE_OUTPUT`，然后运行不带 `--plot-only` 的四条命令，完成新一组计算并自动出图：

```powershell
python -m script.step1_prepare_dic
python -m script.step2_forward_calibrated
python -m script.step3_inverse_dic
python -m script.step4_compare_results
```

已有数值结果仍拒绝覆盖；确需重算时使用新的 `CASE_OUTPUT`。若需要保留多组配置，可复制为 `dic_case_sample3.py`，四步均指定它：

```powershell
python -m script.step1_prepare_dic --config dic_case_sample3
python -m script.step2_forward_calibrated --config dic_case_sample3
python -m script.step3_inverse_dic --config dic_case_sample3
python -m script.step4_compare_results --config dic_case_sample3
```

也可用 `--config <配置文件.py>` 指定路径。后续三个脚本读取第一步冻结的配置和数组，不再重新读取可能已改变的原始 CSV。

必须逐组明确的参数：

- `CASE_ID`、`CASE_OUTPUT`：实验身份与新结果目录。
- `PLATFORMS`：各载荷平台的数据目录、首末阶段、识别或预测用途。载荷可通过 `FORCE_PAIRING_CSV` 的 `run/setting_N` 配对，或显式提供 `force_N` 和 `force_note`；平台名称不直接作为实际拉力。
- `DIC_UNIT`、`DIC_COORDINATE_STATE`、位移与力的参考态：当前入口采用 mm、相对初始未加载图位移及相应实际拉力。坐标和位移必须共用长度单位；不自动识别单位或推断预载参考。
- `DOMAIN_SOURCE`：默认 `dic_reference_bounds`，第一步从识别平台初始坐标解析 `GEO_L/GEO_H` 并冻结；预测平台不重新决定域。`THICKNESS/LOAD_SECTION_WIDTH` 来自试样定义，不能用内部 DIC 宽度替代真实承载宽度。`REGION_INFILL_PERCENT`按FEM x排列为90/80/70/60/50%，`REGISTRATION.material_x_sign=-1`控制材料方向，`REGION_BOUNDARIES_MANUFACTURING_X`保留制造界面位置，`specimen_marks`按已确认标线中点映射到FEM原点。旧配置缺少 `DOMAIN_SOURCE` 时仍按显式 `GEO_L/GEO_H` 处理。
- `REGISTRATION`：同一坐标变换用于全部平台，坐标和位移同步变换；平移只作用于坐标，不独立缩放坐标轴。
- `NEL_X/NEL_Y`：同一有限元网格用于独立正演、反演和回代；DIC 保持自己的真实测点和采样间距。
- `NU_VALUES/PRIMARY_NU`：本组允许范围为 0.24—0.33，具体ν尚未独立确定；当前四步冻结的 0.24/0.26/0.28 是预设比较点，0.26 是历史参考值，不是实测中点或已测定材料值。历史 [E—ν联合诊断](joint_modulus_poisson_diagnosis_20261019.md) 保留旧输入。本轮另按新输入、固定主γ计算ν=0.33敏感性，见 [标线修正报告](material_direction_update_20261020.md)，仅用180N识别，不以280N调参。

## 物理域与坐标配准

当前计算域由识别平台 DIC 初始坐标定义，约 61.497354×19.435484 mm，厚 3.2 mm，点距约 0.290081857 mm。不拉伸点云，不补外推位移。真实承载宽度仍为 20 mm，牵引取 F/(20t)；传入现有有限元内核的截取域端力为 F×GEO_H/20。`mean_corrected_force_N` 始终保存实际测得力，`model_end_force_N` 单独保存截取域端力，不能互换。

`INTERNAL_SIDE_BOUNDARY='traction_free_approximation'` 显式采用内部横向截面自由近似；不配置该近似时拒绝将较窄区域直接作为自由侧边模型。严格子域等效尚未验证。几何、分区和载荷的完整定义见 [DIC 计算域及载荷截面](dic_roi_geometry.md)。

当前配置采用`specimen_marks`：73 mm为高填充率端、85 mm为低填充率端，region边界对应横线，方向Q=((0,1),(-1,0))。独立DIC跨度61.497354 mm与名义标线推算62 mm的差异保留；材料制造坐标为36.748676795−FEM x mm。因近似标线间隔、名义总长和横向位置仍有未闭合项，状态保持条件性，但不再将材料位置放在试样中心。

配准支持以下显式模式：

- `pending`：未定位；第 1 步保存输入审查结果，后续求解拒绝启动。
- `specimen_marks`：从已确认的高/低填充率端距及region两端标线对应计算中点，只使用单位尺度；冻结制造覆盖、材料界面及与名义标线间隔的差异。
- `assumed_center`：显式提供正交矩阵及假定点云中心位置；可探索计算，但配准状态保持未核验。
- `explicit_verified`：给出已独立核验的 `matrix/offset_mm` 和核验依据 `note`。
- `landmarks`：给出至少三个不共线的 `dic_points_mm/fem_points_mm`、独立定位依据和允许点位误差 `tolerance_mm`；仅拟合单位尺度的旋转/镜像和平移，不能拟合缩放。

`landmarks` 或 `explicit_verified` 仍可在显式尺寸域中使用。独立控制点必须按所选计算原点给出，不能把点云定域与材料位置核验混为一谈。DIC范围模式支持当前标线中点模式及历史显式居中假设；独立地标定域可选择 `configured_geometry`。

域外点默认报错；只有显式设置 `OUTSIDE_POINT_POLICY='crop_to_physical_roi'` 才裁剪，并记录域内外数量。裁剪不改变物理分区或坐标尺度。

## 刚体规范、网格与参数比较

纯牵引有限元仅用三个约束去除刚体零空间：左下角 ux/uy 和右下角 uy。它们是求解的数值规范，不要求真实 DIC 在这些点位移为零，也不将一端当作物理夹紧边界。比较时对模型与 DIC 残差拟合相同的两平移和一微小转动，保留真实伸长及横向收缩。改变三个规范约束点后，配准残差与应变应不变；该性质已纳入检验。

当前共用 70×20 个 Q4 单元，约 0.878534×0.971774 mm，正反问题均为 1491 个节点；反演所有节点模量独立，不绑定不同 y 行。新域的制造界面不再全部与单元边界对齐，旧 70×20 mm 域的收敛结果不能直接视为新域已收敛。DIC 的点距约 0.2895 mm，不需要等于有限元单元尺寸；Q4 观测算子在真实 DIC 位置求值。

### 原始目标与 L-curve 选参

目标为 `J(E)=0.5*r.T@W_area@r + 0.5*gamma*integral(|grad E|^2)`。只保留真实 DIC 测点面积权重；不设参考模量，不除以模量或人为噪声尺度，不采用仿射去趋势正则。每个节点的 `m=ln(E)` 独立优化，正则作用于 E，不作用于 log(E)。五区标定只用于独立正演和后处理，不进入反演目标、初值或选参。

仅在 `FIT_CASES`、预设主 ν=0.26 下计算 L-curve：横轴 `rho=sqrt(r.T@W_area@r)`（mm²），纵轴 `eta=sqrt(integral(|grad E|^2))`（MPa），两轴取对数；eta 不乘 gamma。默认 gamma 从 10⁻¹² 到 10⁻⁴，共 50 个对数等距点，含两端点；log10 间距为 8/49，相邻值倍率约 1.45635。取消空间验证分割，不按拟合/留出误差或材料标定择优。

扫描由大 gamma 到小 gamma 延续：首点使用识别平台位移拟合的均匀模量初值，后续只继承此前的收敛解。先用均匀1 MPa材料计算单位响应，再将公共位移尺度与每个平台的三个刚体分量一起作面积加权最小二乘，恢复初始E；本组仅用180N得到约2147.9082 MPa。此值只用于初始化，后续1491个节点全部独立；不读取独立材料标定或280N，不作为E_REF，不改变原目标及单位。初始数组、来源和逐γ延续来源均保存。原0.501 MPa冷启动的6个未收敛点已随同组初始化核验归档，不与新扫描混合。

根据递增 gamma 顺序计算有符号三点 Menger 曲率，取最大正曲率内部点；要求至少五个有效点，峰值两侧各有两个有效扫描点。未收敛点、非正范数、近重合点、非单调局部分支及近乎直线的三点不用于选参。数值阈值为 log 范数单调容许波动 10⁻⁵、最小三点边长 10⁻⁴、最小可分辨正曲率 10⁻⁶；近重合点的微小优化波动可能产生虚假大曲率，故先检查边长。这些阈值用于排除数值退化，不是实验一致性判据。

若L-BFGS线搜索以ABNORMAL结束、梯度∞范数已不超过10×GTOL，且该点仍有剩余迭代和函数预算，可从同一γ、同一场清空L-BFGS历史重新求解，最多两次；累计预算不增加，FTOL/GTOL不改变。每次原停止信息、原梯度、重启迭代数和最终停止信息保存于`line_search_restarts`。这不是把10×GTOL作为收敛阈值，也不直接将失败标记改为成功；重新调用原求解器后仍按其实际状态记录，最终是否达到GTOL另列。未满足条件或重启后仍失败的点继续排除。

`selection.csv` 保存原始范数、目标分解、曲率、有效/选中标记、最终梯度 ∞ 范数及停止信息；`l_curve/gamma_*/` 保存逐点数组和配置。无拐点、端部支撑不足或有效收敛点不足时，第 3 步保存诊断图与状态并返回退出码 2，不回退至固定 gamma，也不生成最终反演结论。检查扫描范围或收敛性后，应另存新配置计算。

有可用拐点后，冻结 gamma，所有 ν 从同一个所选场出发，以相同网格和预算重新求解；`PREDICT_CASES` 不参与选参。第 3 步的云图页直接显示 L-curve 和曲率，第 4 步链接该证据。L-curve 是选参启发式，不能单凭曲率峰值认定模量正确或唯一。定义与限制参考 [Hansen 的 L-curve 说明](https://www.imm.dtu.dk/documents/ftp/tr99/tr15_99.pdf)。

### 容差的生效范围

优化器使用 L-BFGS-B，在 `m=ln(E)` 上优化；`FTOL` 是目标函数下降停止条件，`GTOL` 是优化变量梯度停止条件，满足任一条件即可停止，还可能因预算或线搜索而结束。减小 `FTOL` 不代表已满足 `GTOL`，应查看保存的停止信息。

新配置保留 `FTOL=1e-30、GTOL=1e-12`，50点L-curve与四步计算已经完成。50个扫描点中49个按求解器规则成功，最弱γ点达到迭代上限并排除；所选γ与最后各ν的求解按函数下降条件停止，未满足GTOL，不能把`converged=true`写成严格梯度条件已满足。旧居中50点与17点分别位于 `history/pre_marks/`、`history/lcurve17/`；`results/dic/history/` 原四步目录仍保存旧目标、`FTOL=1e-11、GTOL=1e-7` 及空间验证所选 γ=10⁴，后者不是L-curve结果。

要让修改后的容差生效，应先选用新的 `CASE_OUTPUT` 并重新运行第 1 步以冻结新配置，再运行后续步骤。只修改源配置，或给第 3 步指定新的 `--output` 但继续读取旧 `01_prepared`，均不会更新求解设置。结果页显示的是实际保存计算的容差与停止原因。

## 可复核性与完成范围

阶段间核对冻结输入的元数据/数组哈希，拒绝混用不同实验的正反演结果。数值兼容性核对平衡、材料、观测、正则、反演及 L-curve 所依赖的十一个算法文件哈希，缺少记录或数值内核变化均需重新准备；包导出及历史绘图、文件包装不参与力学兼容性判定。所有步骤调用同一正演、观测和伴随内核。

本流程交付独立正演与 DIC、二维模量与独立标定、回代及预测三条比较证据。物理配准未完成、完整狗骨与矩形截取域边界等效未验证、误差预算不完整时，不自动宣告实验物理验证通过。

实现核验使用临时解析牵引算例、原始 H1 正则的积分/矩阵等价性及常数零空间、伴随方向导数差分；最后一项的最小相对偏差约 3.0×10⁻⁹。独立的线性 Tikhonov 闭式解用于检查 L-curve 拐点、顺序不变性和退化曲线拒绝。

临时 18 节点二维算例贯通合成 CSV 导入、独立正演、21 个 gamma 的实际非线性求解与 L-curve、结果比较及重新出图，同时检查选参不使用预测平台、目标分解与文件链接。该小算例使用 FTOL=10⁻¹⁵、GTOL=10⁻¹¹、最多 500 次迭代，仅验证实现；默认严格容差及扫描范围在真实 DIC 上的收敛性、拐点存在性尚未计算。真实数据出图前后仍须核对位移误差、分区模量和逐测点导出一致性。上述检查不增加实验物理验证证据。
