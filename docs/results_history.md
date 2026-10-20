# 历史结果与复现来源

当前DIC节点网格结果为`results/dic_mesh/01_prepared/`至`04_comparison/`，采用沿原始+Y模量递减的材料方向。旧70×20单元粗网格结果原位保留在`results/dic/`；其`direction_update/`记录当时的定位、复用及ν=0.33补充条件。

粗网格完整备份为[coarse_mesh.zip](../results/dic/history/coarse_mesh.zip)，成员与SHA-256见[coarse_mesh_manifest.json](../results/dic/history/coarse_mesh_manifest.json)，共687个成员，包含冻结结果和当时的配置、算法、流程源码。原位结果与备份均保留，不将粗网格数值当成当前DIC网格结果。

历史包：[previous_runs.zip](../results/dic/history/previous_runs.zip)。逐文件身份见[archive_manifest.json](../results/dic/history/archive_manifest.json)。共2529个成员，已逐一核对SHA-256；每个成员名保留原项目相对路径。

| 历史版本 | 包内原目录 | 条件 |
| --- | --- | --- |
| 原材料方向 | `results/dic/history/plusY/` | region、+Y递增、50点扫描 |
| 旧居中版本 | `results/dic/history/pre_marks/` | _3、居中位置、50点扫描 |
| 17点基线 | `results/dic/history/lcurve17/` | 原17点扫描及其冻结输入 |
| 旧70×20 mm | `results/dic/history/01_prepared/`至`04_comparison/` | 原物理域与选参协议 |
| 标线、网格及方向对照 | `results/dic/geometry_fix/`、`direction_check/`等 | 各自冻结条件，不作为当前结果 |
| 其他参数诊断 | 包内各`*_diagnosis_*`及`*_diagnostic_*`原目录 | 原目标、数据版本和参数协议 |

旧粗网格反演记录的`reused_numerics.source_directory`、原执行环境和代码哈希保留原身份；对应原文件可按成员路径在包内找到。归档包可以在另一个目录解压核查，不覆盖当前四步结果。数值数据和历史图件按原条件保留，不重新解释为当前方向。

历史报告成员：

- `docs/calibrated_forward_mismatch_20261019.md`
- `docs/dic_reference_code_review_20261019.md`
- `docs/dic_shared_mesh_nu_comparison_20261019.md`
- `docs/joint_modulus_poisson_diagnosis_20261019.md`
- `docs/l_curve_50_points_20261020.md`
- `docs/l_curve_dic_inversion_20261020.md`
- `docs/load_geometry_mesh_audit_20261019.md`
- `docs/mark_geometry_correction_20261020.md`
- `docs/mark_geometry_forward_diagnostic_20261020.md`
- `docs/material_direction_check_20261020.md`
- `docs/partition_forward_diagnostic_20261020.md`
- `docs/partition_modulus_diagnosis_20261019.md`

独立材料标定、原始DIC、制造几何和当前算法源码仍保留在项目原目录，与历史计算包分开。当前研究方法、结果和限制以[DIC网格记录](dic_mesh_workflow_20261020.md)、[验证方案](../plan/validation_plan.md)和[分步流程](dic_stepwise_workflow.md)为准。
