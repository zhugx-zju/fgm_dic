# PLA 单轴拉伸数据

当前有效结果：1013 保留 `modulus_20261013_165808`；1016 和跨批次研究使用 `analysis_20261017`。

- [1016 单批分析报告](1016/analysis_results/analysis_20261017/analysis_report.md)：组均值、逐根相对偏差、各填充率原始曲线和拟合入口。
- [1013–1016 桥接报告](analysis_results/analysis_20261017/batch_comparison_report.md)：60%/90%桥接、旧70%/80%诊断，以及三种选段口径的区别。
- [混合批次模量—填充率分析](analysis_results/mixed_infill_20261017/linearity_report.md)：1013的40%、50%、60%、90%、100%与1016的70%、80%，比较线性近似、残差与逐根选段敏感性。
- [混合批次模量—填充率主图](comparison_results/mixed_infill_20261017/modulus_vs_infill.png)：34根试样、7个组均值及标准差，保留批次标记。
- `1016/<填充率>/analysis_results/analysis_20261017/`：该组新批试样的原始曲线、拟合图和结果。
- `1016/comparison_results/analysis_20261017/`：仅1016各填充率的总对比图。
- `analysis_results/analysis_20261017/<填充率>/`：两批在该填充率下的对照。
- `comparison_results/analysis_20261017/`：跨填充率的两批总对比图。

`fit_results.csv` / `group_summary.csv` 沿用原同填充率优先共同窗口的方法，不强求两批共窗。`individual_optimal_*` 为每根试样独立选段的补充对照，不覆盖主结果。`matched_*` 为跨批同应力段的诊断比较。各方法不能混作同一套标定参数。

相对偏差定义：`100 × |组均值 − 该试样模量| / 该试样模量`，反映组均值代表这些试样的偏差，并非相对于真实材料模量的测量误差。

原始工作簿、尺寸记录和sample_2/3保持不变。旧试运行结果与重复复现结果从活动目录清理，先存入 `analysis_archive/obsolete_analysis_20261017.zip`；清理清单和逐文件校验值位于同目录。

复现使用abqinter环境，必须指定未使用过的运行编号：

```powershell
& 'C:\Users\zhu_g\.conda\envs\abqinter\python.exe' -X utf8 -B 'D:\MyProject\post_tensile_test\analyze_tensile_batches.py' --root 'D:\MyProject\post_tensile_test' --run-id '新的运行编号'
```
