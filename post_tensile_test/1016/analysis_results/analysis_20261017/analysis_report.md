# 1016 均匀填充试样模量分析

16 根试样；宽9.89 mm、厚3.26 mm、标距50 mm。sample_2/3不参与。
主结果沿用原方法：每个填充率优先共用线性窗口，不要求与其他批次共用窗口。
应变=引伸计伸长/50 mm，应力=扣除初始偏置的力/(9.89×3.26 mm²)。E为所选段含截距OLS斜率。

## 主结果及均值代表每根试样的偏差

绝对相对偏差=|组均值−该试样E|/该试样E×100%。分母是该试样E；并非与真实模量比较的测量误差。
CSV还保留带符号偏差，以及以组均值为分母的另一种口径，避免混淆。

| 填充率 | n | E均值±标准差(MPa) | 平均绝对相对偏差 | 最大绝对相对偏差 |
| --- | ---: | ---: | ---: | ---: |
| 60% | 3 | 1650.8 ± 22.3 | 1.00% | 1.48% |
| 70% | 5 | 1708.3 ± 26.7 | 1.22% | 1.92% |
| 80% | 5 | 1842.1 ± 46.5 | 1.80% | 3.51% |
| 90% | 3 | 2077.6 ± 11.3 | 0.42% | 0.62% |

| 试样 | E (MPa) | 组均值 (MPa) | 绝对相对偏差 |
| --- | ---: | ---: | ---: |
| 60-1 | 1644.7 | 1650.8 | 0.37% |
| 60-2 | 1675.5 | 1650.8 | 1.48% |
| 60-3 | 1632.1 | 1650.8 | 1.15% |
| 70-1 | 1688.9 | 1708.3 | 1.15% |
| 70-2 | 1707.9 | 1708.3 | 0.03% |
| 70-3 | 1676.2 | 1708.3 | 1.92% |
| 70-4 | 1741.3 | 1708.3 | 1.89% |
| 70-5 | 1727.4 | 1708.3 | 1.10% |
| 80-1 | 1858.2 | 1842.1 | 0.87% |
| 80-2 | 1909.1 | 1842.1 | 3.51% |
| 80-3 | 1829.3 | 1842.1 | 0.70% |
| 80-4 | 1832.1 | 1842.1 | 0.54% |
| 80-5 | 1781.7 | 1842.1 | 3.39% |
| 90-1 | 2071.9 | 2077.6 | 0.27% |
| 90-2 | 2090.6 | 2077.6 | 0.62% |
| 90-3 | 2070.2 | 2077.6 | 0.36% |

## 每根试样独立最优窗口：方法敏感性对照

R²≥0.995、至少60点、前后半段斜率差≤5%、应变非递减步比例≥90%；每根独立搜索峰值应力5%–80%，通过后优先选最宽段，随后按R²及曲率排序。
最优只是这一明确准则下的最优，不代表已经验证为初始可恢复弹性段。

| 填充率 | E均值±标准差 (MPa) |
| --- | ---: |
| 60% | 1731.9 ± 89.0 |
| 70% | 1752.0 ± 51.3 |
| 80% | 1814.3 ± 90.9 |
| 90% | 2252.2 ± 120.1 |

## 图表入口

- 60%：[原始力–位移](D:/MyProject/post_tensile_test/1016/60/analysis_results/analysis_20261017/raw_force_displacement.png)、[原方法拟合](D:/MyProject/post_tensile_test/1016/60/analysis_results/analysis_20261017/fit_curves.png)、[逐根独立拟合](D:/MyProject/post_tensile_test/1016/60/analysis_results/analysis_20261017/individual_optimal_fit_curves.png)、[逐根结果](D:/MyProject/post_tensile_test/1016/60/analysis_results/analysis_20261017/fit_results.csv)。
- 70%：[原始力–位移](D:/MyProject/post_tensile_test/1016/70/analysis_results/analysis_20261017/raw_force_displacement.png)、[原方法拟合](D:/MyProject/post_tensile_test/1016/70/analysis_results/analysis_20261017/fit_curves.png)、[逐根独立拟合](D:/MyProject/post_tensile_test/1016/70/analysis_results/analysis_20261017/individual_optimal_fit_curves.png)、[逐根结果](D:/MyProject/post_tensile_test/1016/70/analysis_results/analysis_20261017/fit_results.csv)。
- 80%：[原始力–位移](D:/MyProject/post_tensile_test/1016/80/analysis_results/analysis_20261017/raw_force_displacement.png)、[原方法拟合](D:/MyProject/post_tensile_test/1016/80/analysis_results/analysis_20261017/fit_curves.png)、[逐根独立拟合](D:/MyProject/post_tensile_test/1016/80/analysis_results/analysis_20261017/individual_optimal_fit_curves.png)、[逐根结果](D:/MyProject/post_tensile_test/1016/80/analysis_results/analysis_20261017/fit_results.csv)。
- 90%：[原始力–位移](D:/MyProject/post_tensile_test/1016/90/analysis_results/analysis_20261017/raw_force_displacement.png)、[原方法拟合](D:/MyProject/post_tensile_test/1016/90/analysis_results/analysis_20261017/fit_curves.png)、[逐根独立拟合](D:/MyProject/post_tensile_test/1016/90/analysis_results/analysis_20261017/individual_optimal_fit_curves.png)、[逐根结果](D:/MyProject/post_tensile_test/1016/90/analysis_results/analysis_20261017/fit_results.csv)。

逐根独立方法属于补充对照，未覆盖原方法主结果。局部高R²不能排除传感器系统偏差，也不能证明初始弹性。新批统一尺寸未反映逐根几何差异。
