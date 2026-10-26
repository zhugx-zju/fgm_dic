# 梯度材料模量反演：精简工程

主线为：独立材料标定 → 分层正演与DIC比较；DIC → 自由二维H1模量反演 → 标定比较；识别模量正演回代。分层试样提供可制造的验证载体，研究目标仍是连续空间模量表征。

当前材料参数见[1023拉伸标定报告](post_tensile_test/1023/analysis_results/modulus_20261026_28_final/analysis_report.md)及[分组参数表](post_tensile_test/1023/analysis_results/modulus_20261026_28_final/group_summary.csv)。只使用本次试样的独立标定，不进行不同批次的填充率结果比较。

## 日常入口

只修改[dic_case_config.py](dic_case_config.py)。默认H1、121×19单元；Huber-TV与精确TV在同一文件选择，不再切换多份旧配置。安装requirements.txt后，分别执行四步：

```powershell
python -B -m script.step1_prepare_dic
python -B -m script.step2_forward_calibrated
python -B -m script.step3_inverse_dic
python -B -m script.step4_compare_results
```

**当前1023输入未就绪，默认不会启动反演。** 保留的工作簿不是逐帧DIC场，旧试样帧号、力偏置和坐标不能直接套用。[输入准备](docs/input_setup.md) · [四步用法](docs/workflow.md)

## 项目内容

- `fgm_asm/`：共用有限元、观测、伴随、H1／Huber-TV／精确TV内核。
- `lib/`：四步流程、数据读取、冻结输入、合成恢复及绘图；`script/`只保留四个入口。
- `post_tensile_test/1023/`：当前原始试验资料及`modulus_20261026_28_final`标定结果；旧日期与重复首轮拉伸输出已移除。
- `3D_print/`：原样保留的制造资料。
- `docs/`、`plan/validation_plan.md`：当前方法定位、输入和验证要求。
- `archive/legacy_20261026.zip`：旧配置、探索代码、历史说明和被替换文件的完整原件；路径与哈希见[整理清单](docs/slimming_manifest.json)。

代码依赖保留完整，未改力学或正则定义；TGV／随机场模块作为现有内核依赖保留，不作为本轮主方法。已知界面分层与完全一维试验仅作对照，不代替自由二维识别。
