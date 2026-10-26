# 当前输入状态与配置要求

现有post_tensile_test/1023包含原始拉伸／加载工作簿、制造资料及旧标定输出，没有四步流程要求的stage_*.comdat_comdata.csv逐帧DIC场。目录内部分分析仍标记modulus_20261013_165808；目录名不能单独证明新的试验日期或批次。所有原文件保持不变。

开始计算前在dic_case_config.py逐项填写：

1. PLATFORMS：每个平台DIC文件夹、连续帧号、run与setting_N；FIT_CASES指定本次识别平台，PREDICT_CASES列跨载荷对照。
2. FORCE_PAIRING_CSV：列含run、setting_N、mean_corrected_force_N、full_force_coverage、stage_first、stage_last，完整覆盖DIC窗口。若显式提供平台force_N，必须有force_note并如实标注假设，不能沿用旧319.9 N或60.1 N偏置。
3. CALIBRATION_CSV：包含infill_percent、batch、n、mean_E_MPa、sd_E_MPa、fit_method。每个填充率在所选口径下恰有一行。保留组内离散、批次及共同窗口／逐根最优窗口的区别。现有group_summary.csv缺batch与fit_method，不自动填造这些来源。
4. REGISTRATION：独立毫米尺度、坐标状态、参考图、方向与材料位置。当前pending需改为有依据的配置；材料锚点、界面及窗口由当前试样确定，不照搬旧试样位置或归一化点云跨度。
5. 核对真实承载宽、厚、制造界面、ROI边界近似与ν范围；再设置INPUTS_CONFIGURED=True并使用新的RUN_NAME。

合成模式也需要真实几何／观测位置及明确载荷输入；只替换位移，不从标定误差选择γ。默认提示是输入状态检查，不是要求为每次计算重新申请许可。
