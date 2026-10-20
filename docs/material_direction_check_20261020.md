# 独立标定正演的材料方向对照

## 结论

制造方向目前存在争议，原“填充率沿原始+Y递增”改为待复核的计算假设。固定DIC坐标、载荷、几何、标定模量和泊松比，将整套材料分布镜像后，正演轴向位移误差减小，但伸长总量偏差仍在。不能仅凭拟合优劣确定真实方向，也不能仅凭原方向下反演E与标定趋势相反判定反演或材料实验失败。

主ν=0.26的实际结果：

| 平台 | 假设 | 轴向RMSE（μm） | 横向RMSE（μm） |
| --- | --- | ---: | ---: |
| 180N | 沿+Y递增（原输入） | 3.960 | 1.197 |
| 280N | 沿+Y递增（原输入） | 6.418 | 3.052 |
| 180N | 沿−Y递增（镜像） | 3.474 | 1.187 |
| 280N | 沿−Y递增（镜像） | 5.761 | 3.047 |

180N轴向RMSE下降约12.27%。两方向预测位移之间的轴向RMS差为1.446 μm。

280N轴向RMSE下降约10.24%。两方向预测位移之间的轴向RMS差为2.654 μm。

## 与DIC和已有反演回代位移的直接比较

保持既有反演结果不变，仅加载保存的位移数组。改变独立正演材料趋势后，两平台的正演位移均更接近DIC，也更接近既有反演模量生成的位移；轴向改善较明显，横向与DIC的差异仅小幅改善。以下都是完整二维测点场的分量RMSE，单位μm，未用宽度平均曲线代替全场误差。

| 平台 | 对比对象 | 原方向轴向 | 反方向轴向 | 原方向横向 | 反方向横向 |
| --- | --- | ---: | ---: | ---: | ---: |
| 180N | DIC | 3.960 | 3.474 | 1.197 | 1.187 |
| 180N | 已有反演回代/预测 | 3.714 | 3.256 | 0.431 | 0.280 |
| 280N | DIC | 6.418 | 5.761 | 3.052 | 3.047 |
| 280N | 已有反演回代/预测 | 6.814 | 5.973 | 0.791 | 0.514 |

四曲线图保存在`results/dic/direction_check/forward_inverse_dic_profiles.png/.svg`，包含DIC、已有反演模量位移及两方向独立正演位移；逐点全场对比见`forward_to_dic_inverse.csv`。这项结果支持重新核实正演材料方向与DIC物理位置的对应关系，尚不能仅凭拟合优劣独立确定真实制造方向。

## 控制条件与镜像定义

保留原始观测到FEM的转换x=Y₀+bₓ、y=−X₀+bᵧ、ux=DY、uy=−DX；两假设使用同一位置、位移和刚体拟合方式，不翻转DIC图，不改变毫米尺度。变化仅为材料在观测域中的方向。73 mm仍对应高填充率端、85 mm仍对应低填充率端，region仍对应横线；反向假设改变这两端与原始Y方向的对应。

令当前正演分布为E₊(x)，反向取E₋(x)=E₊(L−x)。界面及各段长度随材料一起镜像，不能只将五个E值倒序而保持原界面不动。

| 沿固定FEM x由左至右 | 原方向 | 反向假设 |
| --- | --- | --- |
| 填充率（%） | 50/60/70/80/90 | 90/80/70/60/50 |
| 各段长（mm） | 0.748677/16/16/16/12.748677 | 12.748677/16/16/16/0.748677 |
| 界面x（mm） | 0.748677/16.748677/32.748677/48.748677 | 12.748677/28.748677/44.748677/60.748677 |

两方向均使用独立标定1396.4108/1638.6619/1708.3381/1842.0790/2120.0569 MPa，不拟合任何模量或位移尺度。厚度3.2 mm、真实承载宽20 mm、DIC范围61.497354×19.435484 mm和70×20 Q4网格保持不变。

ν=0.24/0.26/0.28/0.33均完成两方向、两平台对照；各ν下反向均降低轴向误差，横向只小幅改善。未以280N选择方向或参数，两个方向作为待核验的候选保留。

## 为什么会影响位移场而不改变总伸长

在一维均布拉伸近似下，duₓ/dx=σ/E(x)，因此局部软硬顺序决定位移曲线斜率的空间顺序；总伸长Δu=σ∫dx/E(x)。将整套E(x)镜像，积分不变。二维有限元也应满足轴向反射对称性（允许刚体参考不同）；本轮已独立核对。

| 平台 | DIC端差（mm） | 原方向正演（mm） | 镜像正演（mm） | 仍偏大 |
| --- | ---: | ---: | ---: | ---: |
| 180N | 0.054727 | 0.064348 | 0.064348 | 17.58% |
| 280N | 0.103792 | 0.118050 | 0.118050 | 13.74% |

反向可以改善局部位移场形状，却不能消除本组17.58%/13.74%的总伸长偏差。原反演E沿固定原始+Y总体下降，若真实填充率沿−Y递增，则二者的趋势不再矛盾；但模量数值是否符合标定仍需按新方向重新统计。

自由二维反演目标不读取五区方向或标定E；在保持观测坐标、载荷与几何不变时，单纯修正填充率标签不会改变其E(x,y)求解，只会改变制造位置、分区统计及与标定比较的解释。本轮未重跑或改写主反演。

## 核验、输出与复现

16次正演已完成。原方向在四个ν、两个平台的8组结果与已存主流程/ν=0.33结果完全一致；两方向原始节点位移按x镜像、uₓ反号后仅差刚体项，最大剩余误差7.02e-14 mm。核对记录在`validation.json`。

输出统一位于`results/dic/direction_check/`：comparison.csv、end_differences.csv、direction_sensitivity.csv；plusY/minusY下各ν保存完整位移、预测和模量数组；主ν两平台各保存同色标六格图、残差图、宽度平均位移曲线及逐点CSV，图件均为PNG/SVG。主四步结果未覆盖。

冻结准备、配置快照、输入及源代码哈希、Python/NumPy/SciPy环境和Git版本见run_metadata.json。无随机步骤。物理方向须由带标识的试样图或独立几何定位确定，本对照不作为方向已确认的证据。

从项目根目录执行以下数值复现，使用相同conda解释器；不写文件、不重跑反演：

```python
from pathlib import Path
import numpy as np
import pandas as pd
from lib.dic_case_workflow import load_prepared
from fgm_asm.research import partition_modulus, solve_forward
from fgm_asm.load_case import evaluate_observation
cfg, mesh, cases, metadata, provenance = load_prepared(Path("results/dic/01_prepared"))
values = pd.DataFrame(metadata["calibration"]).mean_E_MPa
plus = partition_modulus(values, cfg.REGION_BOUNDARIES_X)
minus = lambda x, y: plus(mesh.geo_l - np.asarray(x), y)
for nu in (.24, .26, .28, .33):
    for direction, field in (("plusY", plus), ("minusY", minus)):
        for case in cases:
            u, _ = solve_forward(mesh, case.bc_info["tensile_end_force"], cfg.THICKNESS, nu, field)
            state = evaluate_observation(case, u)
            rmse = 1000 * np.sqrt(np.mean(state["residual"].reshape(-1, 2)**2, axis=0))
            print(nu, direction, case.name, rmse)
```
