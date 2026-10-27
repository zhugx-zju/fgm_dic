# 梯度材料模量反演：精简工程

主线为：独立材料标定 → 分层正演与DIC比较；DIC → 自由二维H1模量反演 → 标定比较；识别模量正演回代。分层试样提供可制造的验证载体，研究目标仍是连续空间模量表征。

[仅40%–70%区段反演](post_tensile_test/1026/sample_2/analysis_results/infill40_70_h1_20261027/report.md)已单独计算1026 sample_2三次记录。第2/3次主ν的四区平均绝对参照差为5.83%/6.05%，与全段场在同区的6.20%/6.17%相比变化很小；第1次主ν线搜索异常，仅作诊断。当前根配置为第3次裁剪运行，新计算仍须换RUN_NAME；原全段结果与原始数据保留。

1026均匀60%试样的三次独立H1反演已完成，见[均匀性报告](post_tensile_test/1026/60sample/analysis_results/h1_uniformity_20261026_final/report.md)：主ν=0.26时空间CV分别20.58%、8.54%、4.18%。第3次最接近均匀，但三次均保留横向位移拟合与力参考对应的限制；这不是独立物理验证。当前根配置停留在第3次，全部原始数值结果和逐次配置在results/dic_full/60sample_1026_run*_h1_20261026。

[标定模量对照](post_tensile_test/1026/60sample/analysis_results/h1_calibration_comparison_20261026/report.md)补充绝对水平核验：相对60%默认独立标定1173.52 MPa，三次反演均值分别偏高79.56%、44.31%、24.96%。第3次空间较均匀仍不代表与标定一致；约束选段1276.05 MPa另作明确标注的对照，不据此调γ或载荷。

当前材料参数见[1023拉伸标定报告](post_tensile_test/1023/analysis_results/modulus_20261026_28_final/analysis_report.md)及[分组参数表](post_tensile_test/1023/analysis_results/modulus_20261026_28_final/group_summary.csv)。只使用本次试样的独立标定，不进行不同批次的填充率结果比较。

按逐根线性段重新分析，并检查组内单根偏差≤3%、组均值随填充率不减的要求，见[逐根与约束选段报告](post_tensile_test/1023/analysis_results/individual_3pct_monotone_20261026_final/analysis_report.md)。其中3%一致性和单调性作为显式选段约束，不是独立验证结论；原独立标定和默认配置保留。

## 日常入口

只修改[dic_case_config.py](dic_case_config.py)。默认H1、121×19单元；Huber-TV与精确TV在同一文件选择，不再切换多份旧配置。安装requirements.txt后，分别执行四步：

```powershell
python -B -m script.step1_prepare_dic
python -B -m script.step2_forward_calibrated
python -B -m script.step3_inverse_dic
python -B -m script.step4_compare_results
```

**1026均匀试样已配置并保存结果；再次计算须换RUN_NAME，已有目录拒绝覆盖。1023分层输入仍未就绪。** 保留的工作簿不能替代逐帧DIC场，试样之间不能套用帧号、力偏置与坐标。[输入准备](docs/input_setup.md) · [四步用法](docs/workflow.md)

## 项目内容

- `fgm_asm/`：共用有限元、观测、伴随、H1／Huber-TV／精确TV内核。
- `lib/`：四步流程、数据读取、冻结输入、合成恢复及绘图；`script/`只保留四个入口。
- `post_tensile_test/1023/`：当前原始试验资料、基础标定及本次逐根／约束选段结果；旧日期与重复首轮拉伸输出已移除。
- `3D_print/`：原样保留的制造资料。
- `docs/`、`plan/validation_plan.md`：当前方法定位、输入和验证要求。
- `archive/legacy_20261026.zip`：旧配置、探索代码、历史说明和被替换文件的完整原件；路径与哈希见[整理清单](docs/slimming_manifest.json)。

代码依赖保留完整，未改力学或正则定义；TGV／随机场模块作为现有内核依赖保留，不作为本轮主方法。已知界面分层与完全一维试验仅作对照，不代替自由二维识别。
