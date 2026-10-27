# 当前输入状态与配置要求

结果图同时保留两种标定参照：黑色实线为连续光滑E(φ)，灰色虚线为原分层标定阶梯线，彩线为反演结果；两种参照不互相替代。当前叠加图见`post_tensile_test/1026/sample_2/analysis_results/sequential_lcurve_gmax1e-8_20261027/smooth_and_layered_20261027/`。本次仅补回显示线，数值误差与前次一致。

当前用户指定的辅助参照为随填充率变化的完整连续光滑E(φ)，不能用层中心分段直线代替。`lib/smooth_calibration_reference.py`以本次七个单调标定均值构造C²自然三次样条，并解析核验单调不减；在填充率40%–100%上定义，不外推。空间比较暂保留既有层中心对应φ=70+0.625ξ、ξ∈[-48,48] mm，不缩放物理坐标，不把连续化参照视为实测连续真值。修正图表和误差见`post_tensile_test/1026/sample_2/analysis_results/sequential_lcurve_gmax1e-8_20261027/smooth_reference_final/report.md`。历史`trend_*`列仍指旧折线口径，仅用于追溯；新结果使用`smooth_*`列。该修正只改变事后参照，不改标定均值、反演场、目标函数或γ。

根配置当前对应40%–70%区段γ上限降低试验第3次：`RUN_NAME=sample2_1026_run3_h1_infill40_70_gmax1e7_20261027`，范围1e-12至1e-7，仍为50点，FTOL/GTOL及其余输入均保持原值。三次扫描完整结束；第1次选得6.25055e-8，第2/3次分别为insufficient_corner_support/range_insufficient，无正式新选γ。报告见`post_tensile_test/1026/sample_2/analysis_results/infill40_70_gmax1e7_20261027/report.md`；已有结果使用--input查看，新运行仍需更换RUN_NAME。

用户指定的1026 sample_2顺序多工况L-curve研究见`post_tensile_test/1026/sample_2/analysis_results/sequential_lcurve_20261027/report_final/report.md`：每个γ内按1→2→3传递模量初值，三个独立参考图分别去刚体运动，在共同ROI上识别；没有将参考图改称共享参考。50个γ、39条有效完整链。按同一最终场回代三工况的端点曲线选得3.55648e-7，末工况对照曲线为1.67683e-7。该研究保留共同中心定位、厚度和各次载荷参考假设，不是已核实材料点配准，也不修改默认单工况配置；顺序最终场只优化末工况目标，主端点曲线不能称为同时联合目标的标准解路径。标定仅用于选点后比较。

1026 `sample_2`的40%–70%区段计算见`post_tensile_test/1026/sample_2/analysis_results/infill40_70_h1_20261027/report.md`。根配置当前停在第3次裁剪运行；以各次原全段冻结输入为本次同一参考图的材料坐标锚点，提取制造坐标[-57,8] mm内原始DIC点，保留40/50/60/70四区。只继承坐标，不继承原初值、γ或模量；分别重做50点扫描。第2、3次主ν正常函数下降停止，四区平均绝对参照差为5.83%/6.05%；第1次主ν及第3次ν=0.28最终线搜索异常，完整保留失败状态。新端面为均匀牵引近似；121×19单元保持不变使轴向网格变细。与全段场的比较使用同一裁剪数据、权重、分母及分区精确积分，不能将变化单独归因于删除高填充区。后续新运行必须换RUN_NAME，查看本次已有结果用显式--input。

1026 `sample_2`现有三次全场DIC，分别为591–605、625–639、689–703帧。三次独立参考图，分别执行四步，不合并为多载荷共享参考。用户确认以下均作为假设：厚度约3.2 mm，长宽与承载宽度取DIC参考跨度，末载取名义280/380/480 N并减记录首载50.7776/70.6192/82.2848 N，增量为229.2224/309.3808/397.7152 N；记录首载对应DIC第0帧及末端恒载未同步核实。

用户确认导出ROI位于两标线之间，各标线距试样外端53 mm。220 mm制造试样的标线坐标为±57 mm；DIC实际轴向跨度105.9261/105.7749/109.0552 mm，按两标线中点定位，保持软件毫米尺度，不缩放为114 mm。沿位移较大端为高填充率，本次FEM轴向x沿原+Y，七区依次40/50/60/70/80/90/100%，制造界面−40/−24/−8/8/24/40 mm。标线中点定位、内部侧面自由边界和截面等效仍是条件解释；七区只用于标定正演与外部比较，不限制自由二维H1反演。当前根配置通过SAMPLE_RUN选1/2/3，后续已有步骤用--input读取各自冻结输入；每次新的计算仍须更换RUN_NAME。

按用户最新指定，sample_2采用`individual_3pct_monotone_20261026_final/group_summary.csv`的随填充率不减标定分布，方法为`individual_window_joint_3pct_monotone`。本次更换标定参照后新建`sample2_1026_run{1,2,3}_h1_monotone_20261026`，重新准备并重算标定正演，逐元素核验准备数组、观测算子、权重、载荷及原反演数组后复用原识别场/γ，不重跑优化。当前报告为`post_tensile_test/1026/sample_2/analysis_results/four_step_h1_monotone_20261026/`；原无约束参照结果保留。位移统一报告面积加权L₂相对误差：残差范数/DIC去刚体运动后的变形范数，不逐点相除。

1026均匀60%试样现有三组全场DIC（各15帧），分别保留参考图、独立网格和运行目录。用户指定厚度约3.2 mm、长宽取DIC参考跨度、载荷取各自记录始末差值。第1/2次末载分别显式假定380/280 N；第3次末载用记录最后一条379.78479 N。三次起载分别46.0744/68.7656/250.72479 N，增量333.9256/211.2344/129.06 N。特别是第3次，载荷记录起点与DIC第0帧未同步核实，绝对模量需保留此参考对应假设。

该情形使用`DISPLACEMENT_REFERENCE='initial_reference_image'`与`FORCE_REFERENCE='force_increment_from_record_start'`，并强制填写力参考说明，不得将其标成已核实的零载参考。每次只有一个识别平台，`PREDICT_CASES=()`；回代仅为拟合检查。H1选参不使用“均匀”先验或1023标定结果。结果位置与输入哈希见本次均匀性分析报告；以下条目仍说明1023分层主线的输入要求。

现有post_tensile_test/1023包含当前原始拉伸／加载工作簿、制造资料及modulus_20261026_28_final标定结果，没有四步流程要求的stage_*.comdat_comdata.csv逐帧DIC场。旧日期与重复首轮拉伸输出已移除；原始工作簿和最终标定数据未改动。当前项目只使用本次试样，不进行不同批次填充率结果比较。

开始计算前在dic_case_config.py逐项填写：

1. PLATFORMS：每个平台DIC文件夹、连续帧号、run与setting_N；FIT_CASES指定本次识别平台，PREDICT_CASES列跨载荷对照。
2. FORCE_PAIRING_CSV：列含run、setting_N、mean_corrected_force_N、full_force_coverage、stage_first、stage_last，完整覆盖DIC窗口。若显式提供平台force_N，必须有force_note并如实标注假设，不能沿用旧319.9 N或60.1 N偏置。
3. CALIBRATION_CSV：当前指向post_tensile_test/1023/analysis_results/individual_3pct_monotone_20261026_final/group_summary.csv，CALIBRATION_METHOD为individual_window_joint_3pct_monotone。表中包含infill_percent、batch、n、mean_E_MPa、sd_E_MPa、fit_method，每个填充率恰有一行，每组4根；batch字段记录1023来源。必须保留组内3%和组均值不减是显式选窗约束的说明，不将约束分布视为独立验证出的性质。原common_stress_window_OLS标定及其50%组三根样本口径保留在modulus_20261026_28_final，作为历史参照。
4. REGISTRATION：独立毫米尺度、坐标状态、参考图、方向与材料位置。当前pending需改为有依据的配置；材料锚点、界面及窗口由当前试样确定，不照搬旧试样位置或归一化点云跨度。
5. 核对真实承载宽、厚、制造界面、ROI边界近似与ν范围；再设置INPUTS_CONFIGURED=True并使用新的RUN_NAME。

合成模式也需要真实几何／观测位置及明确载荷输入；只替换位移，不从标定误差选择γ。默认提示是输入状态检查，不是要求为每次计算重新申请许可。

逐根／约束选段研究位于post_tensile_test/1023/analysis_results/individual_3pct_monotone_20261026_final。其fit_method为individual_window_joint_3pct_monotone，组内3%和随填充率不减是显式先验，不能混称为独立逐根最优拟合或独立验证。当前sample_2按用户指定使用此约束参照，约束结果与同条件无约束对照见该目录报告。
