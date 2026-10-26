# 四步流程

日常只修改根目录dic_case_config.py：DATA_MODE选择实验或两类合成，REGULARIZATION_MODE选择gradient／huber_tv／exact_tv，CONFIGURED_ELEMENTS默认121×19。每次换数据、方法、γ范围或分量权重时换RUN_NAME。

1. `python -B -m script.step1_prepare_dic`：读取DIC及载荷，冻结坐标、观测、配置和输入身份。
2. `python -B -m script.step2_forward_calibrated`：独立标定分层正演，与同一DIC比较。
3. `python -B -m script.step3_inverse_dic`：扫描本平台L-curve，再执行所选γ反演；没有有效拐点就保留诊断并停止。
4. `python -B -m script.step4_compare_results`：对照DIC、独立标定正演和识别回代，保存模量、位移和相对误差图。

四个目录依次为01_prepared／02_forward／03_inverse／04_comparison，位于results/dic_full/RUN_NAME。各步骤可用--help查看选项；已有结果用--plot-only重绘。后续步骤默认核对当前配置与第1步冻结摘要；显式--input读取指定冻结输入。

H1默认50点γ覆盖1e-12—1e-4；TV默认1e-10—1e-2，不比较不同正则γ的数值大小。四个ν是固定本平台γ的敏感性分析，不是ν、γ联合测量。函数下降停止、GTOL及TV一阶证据分别报告。

1023输入尚未配置时，第1步给出明确提示并停止，不创建空结果。历史参考包只提供查看，不作为四步的冻结输入。
