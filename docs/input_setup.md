# 当前输入状态与配置要求

1026 `sample_2`现有三次全场DIC，分别为591–605、625–639、689–703帧。三次独立参考图，分别执行四步，不合并为多载荷共享参考。用户确认以下均作为假设：厚度约3.2 mm，长宽与承载宽度取DIC参考跨度，末载取名义280/380/480 N并减记录首载50.7776/70.6192/82.2848 N，增量为229.2224/309.3808/397.7152 N；记录首载对应DIC第0帧及末端恒载未同步核实。

用户确认导出ROI位于两标线之间，各标线距试样外端53 mm。220 mm制造试样的标线坐标为±57 mm；DIC实际轴向跨度105.9261/105.7749/109.0552 mm，按两标线中点定位，保持软件毫米尺度，不缩放为114 mm。沿位移较大端为高填充率，本次FEM轴向x沿原+Y，七区依次40/50/60/70/80/90/100%，制造界面−40/−24/−8/8/24/40 mm。标线中点定位、内部侧面自由边界和截面等效仍是条件解释；七区只用于独立标定正演与外部比较，不限制自由二维H1反演。当前根配置通过SAMPLE_RUN选1/2/3，后续已有步骤用--input读取各自冻结输入；每次新的计算仍须更换RUN_NAME。审计及本次四步报告见`post_tensile_test/1026/sample_2/analysis_results/four_step_h1_20261026/`。

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
