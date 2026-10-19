# DIC 实验的分步骤计算流程

四步共用的物理坐标、位移基准、牵引边界和测点观测约定，见 [DIC—有限元参考一致性](dic_reference_consistency.md)。该说明分别定义原始位移、三个刚体分量拟合、绘图居中及真实端差，附当前代码对应与核对结果。

## 脚本及数据流

每组实验使用一个 Python 配置文件和一个独立结果目录。四个脚本各自完成一步，后续脚本读取已保存的中间文件，不用一个总入口自动串行运行所有步骤。

| 步骤 | 脚本 | 输入 | 保存结果 |
| --- | --- | --- | --- |
| 1. 准备数据 | `script/step1_prepare_dic.py` | 实验配置、DIC CSV、载荷配对、独立标定 | `01_prepared/`：参考坐标、平均位移、测点面积、采样间距、覆盖范围、载荷与几何、配置和输入哈希 |
| 2. 独立正演 | `script/step2_forward_calibrated.py` | `01_prepared/` | `02_forward/`：各 ν 的分区标定正演位移、测点预测、误差表 |
| 3. DIC 反演 | `script/step3_inverse_dic.py` | `01_prepared/` | `03_inverse/`：训练内选参、各 ν 的自由二维模量、回代和留出预测、收敛记录 |
| 4. 比较出图 | `script/step4_compare_results.py` | 同一 `01_prepared/` 对应的正反演结果 | `04_comparison/`：DIC/独立正演/反演回代对照、二维模量、E(x)、五区统计、误差表和 HTML 报告 |

第 2、3 步分别求解正问题和反问题；第 3 步不需要第 2 步的位移来识别模量，第 4 步才汇合两条证据链。第 4 步不重新运行有限元或优化器。

## 当前实验的运行命令

在项目根目录、安装 `requirements.txt` 的 Python 环境中，依次运行：

当前实验已保存结果。重新计算前只需将配置的 `CASE_OUTPUT` 改为新目录，四条命令不需要填写长结果路径。

```powershell
python -m script.step1_prepare_dic --config dic_case_config
python -m script.step2_forward_calibrated
python -m script.step3_inverse_dic
python -m script.step4_compare_results
```

**查看结果：双击项目根目录的 [打开结果.html](../打开结果.html)。** 每一步结束时都会打印本步绝对保存路径和实验总览路径，并更新这个固定入口。第 2—4 步的默认 `--config dic_case_config` 仅用于定位 `CASE_OUTPUT/01_prepared`；求解参数和数据仍来自第一步冻结文件。也可用 `--input <01_prepared目录>` 明确指定输入。

全部结果集中在配置的 `CASE_OUTPUT` 下：

```text
结果总览.html                 四步保存位置、文件链接及最终图表入口
01_prepared/                 DIC 坐标、位移、网格 ID、载荷、核查和配置
02_forward/                  comparison.csv；nu_*/fields.npz 正演数值场
03_inverse/                  selection.csv、regional_comparison.csv；nu_*/fields.npz
04_comparison/               打开查看结果.html、误差与模量 CSV
  nu_0.26/                   主泊松比结果（另有 0.24、0.28）
    180N/、280N/             DIC、正演、反演及残差各分量 PNG/SVG，逐测点 CSV
    modulus_comparison.*     二维模量对照
    x_profile.*              宽度平均 E(x) 与标定对照
```

需要重新出图时，第 4 步用 `--output <新目录>`；总览会指向新的图表目录，旧图仍保留。本次按 post_DIC 风格重新出图的目录是 `04_comparison_postdic/`。第 4 步不重跑前面求解，每一步均拒绝覆盖已有数值结果。

DIC 图复用原 `post_DIC` 的 `lib/dic_preprocessing.py::plot_grid_fields`：7×6 英寸、规则网格 `pcolormesh`、`jet`、等比例坐标、独立单图。DIC/独立正演/反演采用共用色标，两幅残差采用共用对称色标；HTML 并排排列原图。新输入冻结网格 ID，缺失点留空，不插值补测量。旧版冻结结果仅在可核验的轴对齐规则坐标上恢复网格顺序。绘图不改变配准、观测、载荷或反演结果。

## 换一组 DIC 数据

直接修改 `dic_case_config.py` 中的新数据目录、物理定义和新的 `CASE_OUTPUT`，然后运行上述四条命令。若需要保留多组配置，可复制为 `dic_case_sample3.py`，四步均指定它：

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
- `GEO_L/GEO_H/THICKNESS`、分区位置及独立标定 CSV：来自试样与制造定义，不从有效测点跨度反推试样尺寸。标定保留既定方法、批次、样本数及组内 SD。
- `REGISTRATION`：同一坐标变换用于全部平台，坐标和位移同步变换；平移只作用于坐标，不独立缩放坐标轴。
- `NEL_X/NEL_Y`：同一有限元网格用于独立正演、反演和回代；DIC 保持自己的真实测点和采样间距。
- `NU_VALUES/PRIMARY_NU`：当前按实测范围比较 0.24/0.26/0.28，主值预先定为中点 0.26，不能看过留出误差后更换主值。

## 物理域与坐标配准

当前物理计算域为中央 70×20 mm，厚 3.2 mm，五区长度 11/16/16/16/11 mm。新版 DIC 有效覆盖约 61.3664×18.5257 mm，点距约 0.289464301 mm。观测范围小于物理域时，只在真实有效测点计算残差，不补外推位移，不将该范围强行拉伸至 70×20 mm。未观测区域的识别值由平衡和正则延拓，不能与实测覆盖区等同解释。

当前配置的 `assumed_center` 保留 rot90 工作约定和点云居中假设，因此输出仍标为条件性结果。旋转平移不改变平面应力方程和合力定义，但不能单独证明材料区对应正确，也不能证明截取矩形的均布牵引近似成立。

配准支持四种显式模式：

- `pending`：未定位；第 1 步保存输入审查结果，后续求解拒绝启动。
- `assumed_center`：显式提供正交矩阵及假定点云中心位置；可探索计算，但配准状态保持未核验。
- `explicit_verified`：给出已独立核验的 `matrix/offset_mm` 和核验依据 `note`。
- `landmarks`：给出至少三个不共线的 `dic_points_mm/fem_points_mm`、独立定位依据和允许点位误差 `tolerance_mm`；仅拟合单位尺度的旋转/镜像和平移，不能拟合缩放。

例如物理左标线中点、右标线中点和左标线处一侧边缘，可分别对应 FEM 的 (0,10)、(70,10)、(0,20) mm；原始参考图中必须独立定位这些点，不能拿有效点云的极值代替。控制点不必位于有效 DIC 测点内。

域外点默认报错；只有显式设置 `OUTSIDE_POINT_POLICY='crop_to_physical_roi'` 才裁剪，并记录域内外数量。裁剪不改变物理分区或坐标尺度。

## 刚体规范、网格与参数比较

纯牵引有限元仅用三个约束去除刚体零空间：左下角 ux/uy 和右下角 uy。它们是求解的数值规范，不要求真实 DIC 在这些点位移为零，也不将一端当作物理夹紧边界。比较时对模型与 DIC 残差拟合相同的两平移和一微小转动，保留真实伸长及横向收缩。改变三个规范约束点后，配准残差与应变应不变；该性质已纳入检验。

当前共用 70×20 个 Q4 单元、1×1 mm，正反问题均为 1491 个节点；反演所有节点模量独立，不绑定不同 y 行。该尺寸与五区界面对齐，但不等同于已经证明网格收敛。DIC 的点距约 0.2895 mm，不需要等于有限元单元尺寸；Q4 观测算子在真实 DIC 位置求值。

gamma 仅在 `FIT_CASES` 的带缓冲空间验证点上选择。当前预设 ν=0.26 完成选参后，冻结 gamma，以相同初值、网格、预算分别运行三个 ν；`PREDICT_CASES` 不参与选择。扫描边界最小值、优化成功状态、物理配准状态和实验验证结论分别报告。

## 可复核性与完成范围

阶段间核对冻结输入的元数据/数组哈希，拒绝混用不同实验的正反演结果。数值兼容性核对平衡、材料、观测、正则和反演所依赖的十个算法文件哈希，缺少记录或数值内核变化均需重新准备；包导出及历史绘图、文件包装不参与力学兼容性判定。所有步骤调用同一正演、观测和伴随内核。

本流程交付独立正演与 DIC、二维模量与独立标定、回代及预测三条比较证据。物理配准未完成、完整狗骨与矩形截取域边界等效未验证、误差预算不完整时，不自动宣告实验物理验证通过。

实现与流程核验采用临时解析算例、梯度差分及四步 CLI 试跑。真实数据出图前后须核对位移误差、分区模量和逐测点导出一致性。这些检查验证实现与呈现，不增加实验物理验证证据。
