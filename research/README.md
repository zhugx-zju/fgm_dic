# 辅助研究与历史记录

日常运行请从根目录README和config.py开始。本目录不被主流程导入，不要求用户运行所有专题。

| 类别 | 模块 |
|---|---|
| 顺序多工况与联合参考诊断 | `sequential_lcurve_scan`、`joint_reference_diagnostic`、`sequential_case_diagnostic` |
| γ敏感性、稳定性、区间核验 | `dic_gamma_sensitivity`、`dic_gamma_stability`、`sequential_gamma_band`、`material_window_gamma_check` |
| 裁剪及同区比较 | `material_window_analysis`、`material_window_range_analysis` |
| 均匀试样与标定差异 | `uniform_sample_analysis`、`uniform_calibration_comparison` |
| 光滑/分层参照及专题报告 | `smooth_calibration_reference`、`dic_calibration_curve_comparison`、`sequential_lcurve_report`等 |
| 旧标定标准搜索与验证 | `tensile_fit_standard`、`tensile_calibration`、`verify_tensile_calibration` |

这些模块保留原研究限制和参数口径，可能需要明确的历史结果输入。迁移后的调用前缀为 `python -m research.<模块> --help`。原一维、已知界面等更早研究仍在既有 `archive/legacy_20261026.zip` 中；本次未恢复或重复新增求解器。

`history/docs/`、`previous_dic_case_config.py.txt` 是整理前的说明与完整根配置快照，不代表新主线默认设置。专题原输出、原始输入及打印资料保留原路径。代码迁移清单和原始哈希在 `docs/reorganization_manifest.json`，主线不靠归档文件启动。

研究测试集中于 `research/tests/`，可单独运行：`python -B -m unittest discover -s research/tests -v`。
