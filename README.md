# DIC 位移与二维模量反演

**当前主计算为全域DIC连续七区：** 从`all_region`按既定材料坐标截取七个连续16 mm目标区，取实际数据交集后为105.995085×18.484156 mm。368×65个DIC参考点直接作为有限元节点，形成367×64个Q4单元、23,920个独立模量参数。定义、数值状态及结果见[全域七区记录](docs/dic_full_workflow_20261020.md)。

**日常计算只修改 [dic_case_config.py](dic_case_config.py)。** 文件顶部统一选择实验DIC／分层合成／节点合成、DIC／指定网格、H1／Huber-TV、分量权重、泊松比和γ范围。然后依次运行下面四个默认脚本，无需切换配置文件。新结果按`RUN_NAME`保存到`results/dic_full/<RUN_NAME>/`；已有主线结果保留原位置。[选项与示例](docs/unified_configuration.md)。

**查看当前结果：双击根目录 [打开结果.html](打开结果.html)。** 页面直接进入当前实验，列出四步保存位置、云图、位移误差、模量对照及数值文件。每个步骤结束时也会打印绝对路径。

本项目的主线是：**独立材料标定 → 分区正演与 DIC 对照；180N、280N 各自的 DIC → 分别独立识别自由二维 E(x,y) → 与独立标定对照；本平台回代及跨载荷对照。** 两平台在121×19粗网格和367×64细网格上分别选γ、分别保存模量场，不做两平台联合拟合，不绑定不同 y 行，也不以分区模量作为强先验。当前协议见[两平台独立反演](docs/dic_independent_platforms_20261022.md)。

主线比较的参考约定见 [DIC—有限元坐标、位移基准与边界一致性](docs/dic_reference_consistency.md)：区分物理坐标配准、有限元三个刚体约束、残差刚体拟合与绘图居中，说明真实伸长为何保留、DIC 与 FEM 两套网格怎样连接，并列出当前代码入口和核对数值。

## 计划任务：梯度／分层材料统一高通量表征

新研究方案见 [统一反演方案（TeX）](plan/unified_gradient_layered_inverse_plan.tex) 与 [可浏览PDF](plan/unified_gradient_layered_inverse_plan.pdf)。目标是以“区域归属函数＋区内自由二维模量场”兼容连续梯度、分层常值和两者共存；模量系数与界面几何可联合识别，所有y行仍独立。数值部分比较原连续Q4方法、分层常值方法和统一方法，实验部分针对可制造分层结构，以独立标定及DIC检验新方法，并与原方法对照。

**当前为方案与先导核验阶段，尚未接入默认四步流程。** 文档给出完整表示及数据项伴随梯度、未知量与结构先验的区分、固定／未知界面的公平协议、文献先例和JMPS目标的证据边界。后续先做P0表示／积分／导数核验，再做P1小规模无噪声反演诊断；确需稳定化时，优先检验仅含一个强度参数γ的区内梯度正则，允许跨区跳变，不预先叠加界面与材料参考场等多套权重。该候选尚未实现或验证，几何稳定性及正则项几何导数仍需检查；数据项诊断不代表含噪反演已稳定，正式含噪比较前须明确稳定化与选参协议。构造示意图与小核验记录保存在 `plan/unified_gradient_layered_assets/`，不是正式反演结果。

本任务采用独立研究输出（计划根目录 `results/unified_gradient_layered/`），不替换既有H1主线、不修改已有实验场与验证状态。高通量评价针对规定精度下可可靠获取的材料状态及实验／计算成本，不以节点或测点数量代替。后续测试入口、阶段状态和完成标准以该方案为准。

## 四个独立步骤

安装 `requirements.txt`。统一配置沿用73/85 mm标线中点、原始−Y递增方向和独立毫米尺度。`IDENTIFICATION_PLATFORM`选择本次独立识别平台；`GEOMETRY_REFERENCE_PLATFORM='180N'`固定共同几何，粗细网格分别给出`RUN_NAME`。H1默认50个对数等距γ覆盖10⁻¹²—10⁻⁴，每个平台只用自身DIC选参。已有180N结果保留，新补算280N独立反演。新计算在项目根目录依次执行：

```powershell
python -m script.step1_prepare_dic
python -m script.step2_forward_calibrated
python -m script.step3_inverse_dic
python -m script.step4_compare_results
```

换数据或计算选项时，修改同一配置文件并更换`RUN_NAME`，然后依次运行四个脚本。每一步自动生成云图；后续步骤核对用户配置并读取第1步冻结数据。已有数值输出拒绝覆盖。参数及逐项文件说明见 [分步骤流程](docs/dic_stepwise_workflow.md)。

| 步骤 | 保存目录（均在 CASE_OUTPUT 下） | 主要文件 |
| --- | --- | --- |
| DIC 准备 | `01_prepared/` | `云图.html`、DIC 位移云图、冻结输入与核查记录 |
| 独立正演 | `02_forward/` | `云图.html`、标定模量、各 ν 的 DIC／正演／相对误差对照、数值场 |
| DIC 反演 | `03_inverse/` | `云图.html`、二维模量、E(x)、DIC／回代／相对误差对照、识别与预测记录 |
| 比较出图 | `04_comparison/` | `打开查看结果.html`、CSV、各 ν/平台的 PNG/SVG |

浏览入口指向`results/dic_full/`。取数及实现核验位于`verification/`，同点方向对照位于`direction_check/`；旧region节点网格保留在`results/dic_mesh/`，旧粗网格保留在`results/dic/`。

日常粗网格选择`MESH_SOURCE='configured_grid'`与`CONFIGURED_ELEMENTS=(121,19)`。已有对照在`results/dic_full/coarse/`，`dic_coarse_config.py`保留其复现入口；几何、数据和载荷与细网格相同。

日常横向十倍权重直接在统一文件设置`DISPLACEMENT_COMPONENT_WEIGHTS=(1.,10.)`。已有`dic_uy10_config.py`及`results/dic_full/uy10/`保留其复现定义。系数表示面积加权的残差平方权重；协议及状态见[十倍横向权重](docs/dic_uy10_workflow_20261021.md)。

正则对照使用`dic_tv_config.py`及`results/dic_full/tv/`；δ=10阈值对照为`dic_tv10_config.py`及其`delta10/`。四步指定相应`--config`，同一粗网格、等权重，Huber-TV与原H1分别选γ。δ=1未得到有效拐点，δ=10已完成但横向误差没有实质改善。定义、核验和状态见[Huber-TV对照](docs/dic_tv_workflow_20261021.md)。

分层正问题位移的无噪声恢复采用`dic_synthetic_config.py`与节点表示对照`dic_synthetic_nodal_config.py`，仍用原四步入口，结果另存`results/dic_full/synthetic/`。固定已知ν、同粗网格与观测，区分正则偏差、突变界面表示和实验额外偏差；合成恢复不能唯一归因DIC误差。协议及结果见[合成恢复](docs/synthetic_recovery_20261021.md)。

误差云图统一采用百分比：位移误差以对应平台 DIC 去除三个刚体分量后的分量 RMS 归一化，避免逐点除以过零位移；原始残差仍以 mm 保存。第 2 步直接展示正演与 DIC 的并排对照，无需等到第 4 步。

等权重主线反演目标为 `0.5*r.T@W_area@r + 0.5*gamma*integral(|grad E|^2)`，不设置参考模量，不作模量或人为噪声尺度归一化。gamma 仅用识别平台的 L-curve 拐点选择，扫描范数、曲率、停止信息和 PNG/SVG 保存于第 3 步；无可辨识拐点时停止并保留诊断。求解容差来自第 1 步冻结配置，源配置修改后，需重新准备输入才能生效。

## 模型与当前结论范围

all_region与region已确认共用参考图及毫米坐标系，沿用既定横线中点定位与原始−Y材料递增方向。目标制造范围[−56,56] mm不重新居中；可用数据在40%端不足5.752190 mm，100%端边界另舍去不足一个采样间距的0.252725 mm。实际沿FEM x为100/90/80/70/60/50/40%，长度15.747275/16/16/16/16/16/10.247810 mm。

坐标和位移采用同一正交变换，独立毫米尺度保留；FEM x沿原始+Y。DIC网格约0.288815×0.288815 mm，H直接取对应节点位移。真实承载宽度20 mm、厚度3.2 mm；均布牵引F/(20t)，截取域端力随实际观测宽度换算。平面应力、三个刚体规范和内部横向自由近似保持既定定义。

180N采用259—273帧、实际力119.776179 N；280N采用436—450帧、219.735451 N，60.1 N偏置与未加载图参考已确认一致。两平台分别独立选γ与识别E，再比较本平台位移回代及跨载荷响应。比较ν=0.24/0.26/0.28/0.33，主值0.26为预设基线；两平台都参与各自识别后，当前比较不再称为未使用数据的独立验证。七区正演模量来自`independent`标定均值，包含40%与100%，不通过调模量减少独立正演误差。

**现有结果属于条件性比较。** 优化收敛与位移拟合改善不代表物理验证完成；材料区定位、完整狗骨与矩形域边界等效、跨批标定转移及误差预算仍需独立核验。数值比较见[当前全域记录](docs/dic_full_workflow_20261020.md)。

## 资料与代码位置

- [实验方案](plan/dogbone_parallel_section_adjoint_plan.tex)、[验证方案](plan/validation_plan.md)、[拉伸数据与标定](post_tensile_test/README.md)。
- `fgm_asm/`：共用有限元、观测与伴随内核；`lib/`：数据处理、四步流程及绘图；`script/`：四个入口。
- [架构与接口](docs/architecture.md)、[坐标说明](docs/dic_coordinate_registration.md)。
- `3D_print/`：制造几何；`post_tensile_test/`：原始实验与独立标定；`results/`：各组计算结果。

实现核验采用临时解析算例、梯度差分和四步试跑，不保留独立测试源码。通过实现核验不等于实验物理模型成立。
