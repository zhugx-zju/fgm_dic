# 自主运行与结果位置

Python 3.10及以上；在项目根目录执行 `python -m pip install -r requirements.txt`。所有日常参数在 `config.py`，不需要编辑入口或求解器。根配置默认合成示例，可先独立跑通，再设置真实路径和本次实验假设。

下表每个文件都在 `script/`，均以 `python -B -m script.<不含.py的文件名>` 执行。左列计算一次，右列只读取已保存的对应结果。

| 步骤 | 主流程计算代码 | 对应结果代码 | `results/<RUN_NAME>/`下的输出 |
|---|---|---|---|
| A 标定 | `calibrate_material.py` | `plot_calibration_results.py` | `00_calibration/` |
| B 选帧与力配对 | `select_dic_frames.py` | `plot_frame_selection_results.py` | `00_frame_selection/` |
| 1 准备输入 | `step1_prepare_dic.py` | `plot_prepared_dic.py` | `01_prepared/` |
| 2 标定正演 | `step2_forward_calibrated.py` | `plot_calibrated_forward.py` | `02_forward/` |
| 3 反演 | `step3_inverse_dic.py` | `plot_inverse_results.py` | `03_inverse/` |
| 4 对比与回代结果 | `step4_compare_results.py` | `plot_comparison_results.py` | `04_comparison/` |

A与B可以分别完成；B之后须在DIC软件按导出清单导出全场，才能运行第1步。自带示例已包含该窗口文件。没有总batch，不会因运行一个入口自动开始大扫描或其他专题研究。

H1、Huber-TV和精确TV共用以上入口；选择方法后使用新的 `RUN_NAME`，并分别设置γ范围、Huber阈值或精确TV参数。各正则γ的单位不同，不能机械复用同一数值。当前示例预算用于快速演示，不是所有数据上的收敛保证。

第1步冻结原始文件哈希、实际载荷、配准、观测、面积权重、标定及配置。后续三步默认读取同一冻结输入，并检查根配置未改变。已经准备好后修改任何配置参数，应使用新 `RUN_NAME` 重新准备；查看旧运行可以显式指定 `--input results/<old>/01_prepared`，不受当前配置选择影响。

第3步先扫描γ，再从有效、有足够邻域支撑的L-curve拐点选γ；标定不参与选参。无有效拐点时完整保存失败和扫描场、返回非零状态，不手动补选γ，也不能继续把诊断场当正式反演结果。可用 `--resume` 在原剩余预算内恢复未解决的线搜索；不自动追加预算。

每次新数值计算换 `RUN_NAME`，已有步骤目录拒绝覆盖。仅重绘时：

```powershell
python -B -m script.step2_forward_calibrated --input results/<old>/01_prepared --plot-only
python -B -m script.step3_inverse_dic --input results/<old>/01_prepared --plot-only
python -B -m script.step4_compare_results --input results/<old>/01_prepared --plot-only
```

也可使用表中的独立结果脚本，例如 `python -B -m script.plot_inverse_results`。它与 `step3_inverse_dic --plot-only` 完全等价。标定和选帧的结果脚本读取各自冻结曲线/时序，不重读原始工作簿、不重新拟合或选帧；可用 `--input` 指定旧的00目录。准备输入的结果入口用 `--output` 指定旧01目录；其他结果入口用 `--input` 指定旧01目录。

重绘读取保存场，不重跑优化；第4步比较也不重跑FE。`--output` 可显式改变某一步输出，但后续应相应传入 `--input/--forward/--inverse`，日常优先只在根配置指定路径。

## 输出规范

运行目录 `results/<RUN_NAME>/结果总览.html` 是每次结果入口。`results/index.html` 只是浏览索引，不参与后续输入选择；`OUTPUT_ROOT` 可更换。A/B输出标定结果、选帧记录和导出请求；四步输出按既有规范保留：

- `run_metadata.json`：配置、输入/代码哈希、环境版本、命令、停止状态与假设。
- `configuration_snapshot.py`：本次配置源码快照。
- `fields.npz`：准备观测或各次保存的数值场。
- `selection.csv`、`l_curve/gamma_*/`：全部扫描候选、停止记录和场，失败也保留。
- `nu_0.26/` 等：各泊松比的最终场、回代位移与局部结果。
- `comparison.csv`、`regional_comparison.csv`：位移失配、分区模量与标定参照差异。
- `04_comparison/打开查看结果.html`、`report.md`：可浏览的最终比较；图片同时保留PNG及SVG。

位移百分比统一采用残差的面积加权范数除以DIC去刚体运动后的变形范数，不逐点除过零位移。求解器函数下降停止、严格梯度条件、位移拟合、标定一致性与物理模型验证分别报告。参与识别的同平台回代属于拟合检查，不能称独立物理验证。

## 纯源码分享

```powershell
python -B tools/export_source.py
```

生成 `dist/fgm_dic_source.zip`，每个文件的SHA256在包内 `source_manifest.json`。该包包含合成输入，可在另一目录直接执行以上六个命令；排除实验数据、历史结果、专题研究与Git历史。重新导出须换 `--output`，不会覆盖已有包。
