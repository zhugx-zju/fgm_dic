# 单文件配置与四步运行

日常计算统一修改根目录`dic_case_config.py`。数据模式、网格、正则、位移分量权重、泊松比、γ扫描范围与求解容差集中在顶部；数据目录、帧范围、材料分区和几何定义在同一文件下半部分。四个流程脚本默认读取该文件，无需更换`--config`。

## 选择方式

| 参数 | 可选值 | 含义 |
| --- | --- | --- |
| `RUN_NAME` | 如`run01` | 本次结果目录`results/dic_full/run01/`，重算使用新名称 |
| `DATA_MODE` | `'dic'` | 读取实验DIC平均位移 |
| `DATA_MODE` | `'synthetic_layered'` | 用同配置的几何、测点及载荷生成突变分层正演位移 |
| `DATA_MODE` | `'synthetic_nodal'` | 用可由反演节点表示的Q4模量生成正演位移 |
| `MESH_SOURCE` | `'dic_grid'` | DIC参考点直接作为有限元节点 |
| `MESH_SOURCE` | `'configured_grid'` | 用`CONFIGURED_ELEMENTS`指定单元数，保留全部DIC观测位置 |
| `REGULARIZATION_MODE` | `'gradient'` | 原一阶Tikhonov/H1，作用于E的梯度平方 |
| `REGULARIZATION_MODE` | `'huber_tv'` | Huber-TV，阈值由`TV_DELTA`给出，单位MPa/mm |
| `REGULARIZATION_MODE` | `'exact_tv'` | 不平滑的各向同性TV，目标中为`gamma*integral(sqrt(E_x**2+E_y**2))`，无1/2或Huber阈值 |
| `EXACT_TV_OPTIONS` | 外/内迭代预算及四类停止证据阈值 | 仅精确TV使用；`MAX_FUN`为每个γ/ν的总FE函数计算预算 |
| `EXACT_TV_SOLVER` | `'primal_dual'`或`'admm'` | 同一精确TV目标的两种数值求解器；本轮实验使用前者 |
| `EXACT_TV_PRIMAL_STEP` / `EXACT_TV_COUPLING` | 2e8 / 0.9 | 原始—对偶法的数值步长与谱耦合系数，不进入目标函数 |
| `DISPLACEMENT_COMPONENT_WEIGHTS` | 如`(1.,10.)` | 轴向ux、横向uy的残差平方系数，不缩放位移 |
| `PRIMARY_NU` / `NU_VALUES` | 当前允许0.24—0.33 | 实验模式的主泊松比与固定γ敏感性比较点 |
| `SYNTHETIC_NU` | 如`.26` | 合成生成及反演共用的已知ν，自动冻结为单个比较值 |
| `GAMMA_COUNT` | 默认50 | L-curve候选数量，含范围两端 |
| `GAMMA_LOG10_RANGE` | `None`或如`(-12.,-4.)` | `None`时H1用1e-12—1e-4，TV用1e-10—1e-2；不同正则的γ单位不同 |

支持三种数据模式、两种网格及三种正则。TGV尚未接入当前四步流程，不作为可选值展示。选择网格只改变FE离散，不改变原始毫米尺度、真实承载截面或DIC测点数。精确TV接入与本轮实验配置见[实验精确TV对照](dic_exact_tv_20261021.md)。

例如用分层位移、原粗网格和TV进行新计算：

```python
RUN_NAME = 'layer_tv01'
DATA_MODE = 'synthetic_layered'
MESH_SOURCE = 'configured_grid'
CONFIGURED_ELEMENTS = (121, 19)
REGULARIZATION_MODE = 'huber_tv'
TV_DELTA = 10.0
```

实验DIC的横向十倍权重对照只需把`DATA_MODE`改为`'dic'`，设置`DISPLACEMENT_COMPONENT_WEIGHTS=(1.,10.)`并另给`RUN_NAME`。旧七区合成恢复结果、TV对照与粗细网格结果均保留各自冻结定义；新选择不会复用旧γ或覆盖旧场。

## 运行与冻结

修改并保存配置后，在项目根目录分别运行：

```powershell
python -m script.step1_prepare_dic
python -m script.step2_forward_calibrated
python -m script.step3_inverse_dic
python -m script.step4_compare_results
```

第1步将实际配置、DIC来源、力、坐标、观测与源码身份冻结到`01_prepared/`。合成模式同样读取配置中的DIC CSV以取得参考坐标和测点，然后由FE生成位移；不以实验DIC位移作为反演观测，不再要求预先存在某次TV或其他试验的准备目录。真值另存`01_prepared/truth/fields.npz`。合成与实验走相同的正演、观测、刚体配准和反演内核。

第2—4步核对当前用户配置与第1步的配置摘要，再使用冻结值求解或出图。若中途改动配置，它们会明确报出不一致；应更换`RUN_NAME`从第1步重新开始。不能把源文件刚改成TV、但仍读取旧H1准备文件的计算称为TV。

加载配置时重新读取已保存源文件，不复用Python模块或字节码缓存，因此同一IDE会话中再次运行也能取得新设置。

已有结果按原冻结配置读取，可显式给出原准备目录，例如：

```powershell
python -m script.step4_compare_results --input results/dic_full/synthetic/layered/01_prepared --plot-only
```

显式`--input`表示有意使用该冻结运行，不与当前编辑中的配置混合；内核源码和上游准备身份仍核对。`--plot-only`不重新求解。原各专用`*_config.py`保留为既有结果的复现入口，日常选择不需要编辑它们。

## 当前层次

`dic_case_config.py`负责用户选择；`script/step1_*.py`至`step4_*.py`负责分步执行；`lib/dic_case_workflow.py`负责准备、正反演和比较，`lib/synthetic_recovery.py`只生成合成观测及其核验；`fgm_asm/`保留共用有限元与伴随内核。H1与Huber-TV使用原L-BFGS求解器；精确TV使用`fgm_asm/exact_tv.py`的原始—对偶法或非线性ADMM，复用相同正演、观测和伴随数据梯度。

精确TV的L-curve纵坐标为`TV(E)`本身（MPa·mm），不是`√TV(E)`；代价为`gamma*TV(E)`。满足局部KKT条件的均匀场保存为有效求解，但零TV或梯度未超过数值约束分辨阈值的场不参与对数曲率。每个γ和最终ν保存`tv_history.csv`、对偶场及分裂梯度；须同时满足驻点残差、Fenchel间隙和对偶可行性。原始—对偶法直接计算`z=GE`，约束残差恒为零；ADMM还需独立检查其约束残差。原`FTOL`、`GTOL`不决定精确TV的成功状态，`--resume`的旧L-BFGS异常线搜索恢复不用于TV。

第2步始终保留独立分层标定正演。选择`synthetic_nodal`时，它与节点真值存在材料表示差异，准确合成误差应对照`truth/fields.npz`，不能将第2步的分层场误当成节点真值。合成模式不自动宣告实验物理验证完成；无有效L-curve拐点时仍保留诊断并停止。

## 本次核验范围

12种数据×网格×正则组合均完成选项解析检查。三种数据分别搭配H1/TV的6组准备文件，与已有对应运行的数组逐值相同；两组合成真值也逐值相同，说明统一入口没有改变数据或力学定义。另检验了合成TV的横向十倍权重、非法选项拒绝、配置改动检测及显式冻结输入读取。

另执行一次23920节点DIC细网格的分层合成准备，两平台生成的U与原细网格独立正演逐值相同。配置源文件同秒保存后，在同一Python会话中按模块名和文件路径重新加载均取得新值。旧TV及合成准备仍可读取；旧H1粗网格和十倍权重准备保留原有内核哈希拦截，不绕过其在TV内核升级前形成的不兼容状态。

临时75节点算例执行了准备、正演和选参路径：初次13点H1扫描未形成有效拐点；50点、原严格容差下H1与TV也均无有效拐点，原样保留停止状态，未生成虚构最终反演或执行比较步骤。这验证了配置传递及选参失败分支，不能称为两种正则的新成功恢复结果。记录在`results/configuration_checks_20261021/`与对应日志中。已有正式长计算未重跑，有限元、观测和伴随内核未修改。
