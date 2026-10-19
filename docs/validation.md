# 验证记录

协议见 [正演一致性与模量反演验证方案](../plan/validation_plan.md)。下列数值来自已保存的运行记录；合成结果不构成实验可行性结论。

## 70 mm 五区模型（2026-10-18）

| 检查 | 已有证据 | 范围与限制 |
| --- | --- | --- |
| 实现检查 | 35 项单元及集成测试通过，最近代码检查对应 `13b4fd9` | 包括均匀解析解、端部合力和厚度、正则及多观测伴随有限差分、CSV/参考坐标、受控正反演和未收敛状态保存；不验证实验模型假设 |
| 五区合成反演 | `results/synthetic_70mm_20261018/run_metadata.json`：59 次迭代收敛；模量全场 MAE 2.79920%；识别/留出载荷加权 RMS 为 0.000482242/0.000514948 mm | 细网格 70×20、反演网格 35×10、400 点、seed=42、噪声标准差 0.0005 mm、仿射去趋势正则、gamma=0.001；单次噪声实现，界面存在平滑偏差 |
| 二维伪变化 | 同一合成运行横向 RMS 离散 7.43952 MPa | 为宽度方向空间变化，不是估计置信区间；仍需多初值、多噪声和网格检查 |
| 独立标定正演 | `results/model_70mm_20261018`：两载荷位移比例关系最大绝对偏差约 1.67e-16 mm | 只验证线性比例及独立正演流程，尚未与 DIC 作定量物理比较 |
| DIC 数据审查 | `results/dic_registration_20261018_final`：每平台 15 帧、13,845 个共同点；逐帧去除 213 条一致重复记录；跨平台恢复参考坐标最大分量差约 9.81e-8 导出单位 | 支持共享参考构形，不确定绝对单位、原始轴向与材料分区对应 |

原始 DIC 文件及哈希清单位于 `post_tensile_test/1016/sample_2/3/`。完整运行元数据记录代码哈希及运行时 Git 状态；不能仅凭记录中的 HEAD 推断运行源码，因为部分运行在工作区有修改时完成。`results/` 为本地独立计算输出，不随 Git 分发；复现需使用相同配置和输入。

## 尚待完成的实验验证

- 核实导出单位、轴向正负、标线和边缘坐标，建立可追溯的统一配准。
- 区分保载漂移、相关波动与刚体运动，评估噪声后提出并冻结误差判据；目前尚无工程允许误差标准。
- 采用独立标定模量比较两个平台的正演与 DIC，检验网格、载荷、尺寸、泊松比、材料变异及截断边界影响。
- 在相同条件下开展自由二维实验反演、留出载荷预测与稳定性检查；成比例载荷不单独证明唯一性。

当前没有正式实验正演一致性或实验模量恢复的通过结论。小网格均匀场可恢复不代表分区场普遍可恢复；较弱正则的部分试算达到迭代上限，正则条件性仍需受控检查。

## 复现入口

在安装项目依赖的环境中，从项目根目录运行：

```powershell
python -m unittest discover -s tests
python -m script.run_synthetic --output results/synthetic_replication_01
python -m script.run_experiment --mode inspect --output results/dic_inspect_01
python -m script.run_experiment --mode model --output results/calibrated_model_01
```

输出目录必须未被使用。正式 `forward` 和 `inverse` 还需完成坐标配准；测试通过与优化收敛分别记录。

## 110 mm 历史数值基准

以下为较早的算法比较记录，几何、模量尺度、噪声定义和正则系数与当前实验不同，不用于当前实验误差判定。原记录中的局部路径仅标识历史运行，当前复现应核对相应配置及版本。

- Status: two-end traction boundary condition and existing nodal-modulus inverse solver connected
- Last checked: 2026-10-14
- Checks: edge resultants, homogeneous analytic displacement, gauge reactions, adjoint gradient finite difference
- Executed: `python -m unittest discover -s tests -v` (5 passed); default `55 x 10` forward solve (finite displacement, maximum gauge reaction `3.1e-16`); small-mesh inverse smoke run reduced objective from `3.41e-5` to `5.58e-10` in 20 iterations (iteration cap reached)
- Open: paired full-specimen boundary-error study, DIC gauge registration, seven-zone parameterization and identifiability
- Synthetic `E(x)` case: `EY=1.0`, `Ex=2.0`, `55 x 10` mesh, `F=0.01 N`, `gamma=1e-6`, zero added noise; the original 616-parameter `E(x,y)` inverse converged in 342 iterations. Full-field MAE `0.2123%`, y-mean x-profile MAE `0.1779%`, reconstructed transverse RMS spread `0.1709%` relative to mean true modulus. Largest local relative error `2.7182%`, concentrated near the loaded ends. The fixed-gamma workflow ran end to end; the 50-point L-curve workflow was not rerun.
- Boundary-error diagnosis, same synthetic displacement, mesh, initial field, and `gamma=1e-6`: the original `gradient` penalty had mean absolute relative error `0.6493%` at `x=0..10 mm`, `0.02570%` at `x=20..90 mm`, and `0.9571%` at `x=100..110 mm`. Its gradient penalty was about 44 times its data term at the final iterate. Lowering `gamma` to `1e-8` reduced the corresponding end errors to `0.1706%` and `0.1805%`, but setting `gamma=0` did not converge within 2000 iterations and produced large 2D field errors. These checks support regularization bias as a major cause, while not proving unique recovery from displacements.
- Affine-detrended gradient penalty: `R(E)=integral[(E_x-mean(E_x))^2+(E_y-mean(E_y))^2] dA`. It retains all 616 nodal inverse unknowns and allows nonzero constant gradients in both directions. Unit checks cover its affine nullspace and finite-difference gradient; a separate finite-difference check covers the full adjoint objective in log modulus. The fixed-gamma zero-noise run converged in 306 iterations: full-field MAE `8.53e-6%`, left-end MAE `1.67e-5%`, right-end MAE `7.67e-6%`, maximum error `1.53e-4%`. The result is especially favorable because the synthetic truth is exactly in the penalty's nullspace. A single exploratory 1% noise realization (seed 42) reduced end MAE from `0.6588%`/`3.0070%` to `0.2550%`/`2.3648%` and full-field MAE from `1.2050%` to `1.0975%`; this is not a multi-seed robustness result. Seven-zone modulus profiles, DIC noise/correlation, and full L-curve selection remain untested with this penalty. The original result was preserved in `noise_0.00pct/`; the new run is in `noise_0.00pct_affine_detrended_fixed/`.
- A five-point small-mesh L-curve smoke scan (`4 x 2` elements, zero noise) gave identical residual and penalty norms at all gamma values after the first solve, hence zero discrete curvature. This scan contains no usable corner. The L-curve routine now raises an actionable error on flat/invalid curvature instead of reporting an arbitrary middle gamma. The full `55 x 10`, 50-point L-curve was not run; `gamma=1e-6` is the fixed value used for the comparison above.
- Exponential-profile check (`python -m script.benchmark_exponential_regularization`): same two-end force (`0.01 N` per end), thickness (`3.2 mm`), `55 x 10` mesh, independent nodal `E(x,y)` unknowns, common initial field, and fixed `gamma=1e-6` for both penalties. For true `E(x)=exp[ln(2)x/110] MPa` with zero noise, original-gradient vs affine-detrended full-field MAE was `0.2256%` vs `0.07086%`; left-end (`0..10 mm`) `0.4539%` vs `0.1846%`; right-end (`100..110 mm`) `1.2699%` vs `0.3426%`. Both optimizers converged. With one 1% displacement-noise draw (seed 42), full MAE was `1.2886%` vs `1.2123%` and right-end MAE `3.3571%` vs `2.7145%`. Results, identical measured displacement per comparison, logs, fields, and common-scale PNG/PDF figures are saved in ignored `Geo_*/benchmark_exponential_55x10_zero/` and `Geo_*/benchmark_exponential_55x10_noise1_seed42/` folders.
- Stronger exponential-profile boundary case: `E(x)=exp[ln(8)x/110] MPa`, same zero-noise protocol and `gamma=1e-6`, gave full MAE `4.4167%` vs `2.7853%`, with right-end MAE `23.7202%` vs `14.3102%`; the left-end MAE worsened from `1.4291%` to `2.1444%`. Thus the affine-detrended penalty improves the whole field in this controlled comparison but does not adequately recover the high-curvature end at the current gamma, nor improve every region. An exploratory `gamma=1e-8` rerun reduced full MAE to `0.6841%` vs `0.4828%`, with right-end MAE `4.9719%` vs `2.8976%`; this gamma was inspected using known synthetic truth and is **not** an independently selected parameter. The two strong-contrast runs are saved under ignored `Geo_*/benchmark_exponential_55x10_8mpa_zero/` and `Geo_*/benchmark_exponential_55x10_8mpa_gamma1e-8_zero/`.
