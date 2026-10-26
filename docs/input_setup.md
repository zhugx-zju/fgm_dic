# 当前输入状态与配置要求

1026均匀60%试样现有三组全场DIC（各15帧），分别保留参考图、独立网格和运行目录。用户指定厚度约3.2 mm、长宽取DIC参考跨度、载荷取各自记录始末差值。第1/2次末载分别显式假定380/280 N；第3次末载用记录最后一条379.78479 N。三次起载分别46.0744/68.7656/250.72479 N，增量333.9256/211.2344/129.06 N。特别是第3次，载荷记录起点与DIC第0帧未同步核实，绝对模量需保留此参考对应假设。

该情形使用`DISPLACEMENT_REFERENCE='initial_reference_image'`与`FORCE_REFERENCE='force_increment_from_record_start'`，并强制填写力参考说明，不得将其标成已核实的零载参考。每次只有一个识别平台，`PREDICT_CASES=()`；回代仅为拟合检查。H1选参不使用“均匀”先验或1023标定结果。结果位置与输入哈希见本次均匀性分析报告；以下条目仍说明1023分层主线的输入要求。

现有post_tensile_test/1023包含当前原始拉伸／加载工作簿、制造资料及modulus_20261026_28_final标定结果，没有四步流程要求的stage_*.comdat_comdata.csv逐帧DIC场。旧日期与重复首轮拉伸输出已移除；原始工作簿和最终标定数据未改动。当前项目只使用本次试样，不进行不同批次填充率结果比较。

开始计算前在dic_case_config.py逐项填写：

1. PLATFORMS：每个平台DIC文件夹、连续帧号、run与setting_N；FIT_CASES指定本次识别平台，PREDICT_CASES列跨载荷对照。
2. FORCE_PAIRING_CSV：列含run、setting_N、mean_corrected_force_N、full_force_coverage、stage_first、stage_last，完整覆盖DIC窗口。若显式提供平台force_N，必须有force_note并如实标注假设，不能沿用旧319.9 N或60.1 N偏置。
3. CALIBRATION_CSV：已指向post_tensile_test/1023/analysis_results/modulus_20261026_28_final/group_summary.csv，CALIBRATION_METHOD为common_stress_window_OLS。表中包含infill_percent、batch、n、mean_E_MPa、sd_E_MPa、fit_method，每个填充率恰有一行；batch字段仅记录当前1023来源，保留组内离散与拟合窗口口径。50%组仅3根通过默认线性条件，其他各4根；不同填充率采用不同应力窗口，详见同目录analysis_report.md。其余DIC输入仍需补齐。
4. REGISTRATION：独立毫米尺度、坐标状态、参考图、方向与材料位置。当前pending需改为有依据的配置；材料锚点、界面及窗口由当前试样确定，不照搬旧试样位置或归一化点云跨度。
5. 核对真实承载宽、厚、制造界面、ROI边界近似与ν范围；再设置INPUTS_CONFIGURED=True并使用新的RUN_NAME。

合成模式也需要真实几何／观测位置及明确载荷输入；只替换位移，不从标定误差选择γ。默认提示是输入状态检查，不是要求为每次计算重新申请许可。

逐根／约束选段研究另存于post_tensile_test/1023/analysis_results/individual_3pct_monotone_20261026_final。其fit_method为individual_window_joint_3pct_monotone，组内3%和随填充率不减是显式先验，不能混称为独立逐根最优拟合或独立验证。本次保留默认独立标定路径，约束结果与同条件无约束对照见该目录报告。
