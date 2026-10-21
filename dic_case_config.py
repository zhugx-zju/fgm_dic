"""统一用户配置：在本文件选择数据、网格和正则，再分别运行原四步脚本。"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# ==================== 常用选择：通常只修改这一段 ====================
RUN_NAME = 'exact_tv_coarse_pd_20261021'  # 新计算换一个名称；结果保存在 results/dic_full/<RUN_NAME>/。

# 'dic'：实验DIC位移；'synthetic_layered'：分层正演生成位移；
# 'synthetic_nodal'：节点Q4模量正演生成位移。三者使用下方同一组坐标和载荷。
DATA_MODE = 'dic'

# 'dic_grid'：DIC点直接作为FE节点；'configured_grid'：指定FE单元数，
# 仍保留全部DIC观测位置。当前原粗网格密度对应121×19单元。
MESH_SOURCE = 'configured_grid'
CONFIGURED_ELEMENTS = (121, 19)

# 'gradient'：梯度平方；'huber_tv'：Huber平滑TV；'exact_tv'：不平滑的梯度幅值积分。
REGULARIZATION_MODE = 'exact_tv'
TV_DELTA = 10.0  # 仅Huber-TV使用，单位MPa/mm。
# 精确TV停止须同时满足驻点、分裂约束、对偶可行性和Fenchel间隙条件。
# MAX_FUN为每个γ/ν的总FE函数预算；原始—对偶法严格计数。
# FTOL/GTOL只用于原H1/Huber求解器；max_inner/adapt_interval仅用于ADMM。
EXACT_TV_SOLVER = 'primal_dual'  # 'admm'保留既有合成验证入口。
EXACT_TV_PRIMAL_STEP, EXACT_TV_COUPLING = 2e8, .9  # 数值步长；不缩放目标函数。
EXACT_TV_OPTIONS = dict(max_outer=12000, max_inner=150, stationarity_atol=1e-9,
    stationarity_rtol=1e-3, primal_atol=1e-6, primal_rtol=1e-4,
    gap_atol=1e-9, gap_rtol=1e-4, adapt_interval=10)
DISPLACEMENT_COMPONENT_WEIGHTS = (1., 1.)  # (轴向ux, 横向uy)残差平方系数。

PRIMARY_NU = .26
NU_VALUES = (.24, .26, .28, .33)  # 实验模式的固定γ敏感性比较。
SYNTHETIC_NU = .26  # 合成模式生成和反演均用此已知值，不自动混入ν敏感性。

GAMMA_COUNT = 50
# None：H1自动用(-12,-4)，TV自动用(-10,-2)；也可显式写(-12.,-4.)。
# γ单位随正则不同，不能直接照搬相同数值范围比较强度。
GAMMA_LOG10_RANGE = None
MAX_ITER, MAX_FUN = 1500, 12000
FTOL, GTOL = 1e-30, 1e-12

# ==================== 数据目录、帧范围和物理定义 ====================
SAMPLE_ROOT = PROJECT_ROOT / 'post_tensile_test' / '1016' / 'sample_2'
PLATFORMS = {
    '180N': {'folder': SAMPLE_ROOT / '3' / '180N_2' / 'all_region', 'stages': (259, 273), 'run': 3, 'setting_N': 180},
    '280N': {'folder': SAMPLE_ROOT / '3' / '280N_2' / 'all_region', 'stages': (436, 450), 'run': 3, 'setting_N': 280},
}
FORCE_PAIRING_CSV = SAMPLE_ROOT / 'analysis_results' / 'dic_timing_20261018' / 'selected_dic_windows.csv'
FORCE_REFERENCE = 'corrected_force_from_unloaded_state'
FORCE_REFERENCE_NOTE = 'Initial 60.1 N offset and unloaded DIC reference confirmed consistent for this experiment.'
DIC_UNIT = 'mm'
DIC_COORDINATE_STATE = 'deformed'
DISPLACEMENT_REFERENCE = 'initial_unloaded_image'
DOMAIN_SOURCE = 'dic_reference_bounds'
# GEO_L/H are resolved from the fit platform before preparation is frozen.
GEO_L, GEO_H, THICKNESS = None, None, 3.2
LOAD_SECTION_WIDTH = 20.  # Actual specimen width, not the internal DIC ROI width.
INTERNAL_SIDE_BOUNDARY = 'traction_free_approximation'
# Groups are ordered along increasing FEM x = raw +Y.
REGION_INFILL_PERCENT = (100, 90, 80, 70, 60, 50, 40)
# Manufacturing x increases with infill; zero is the specimen center.
REGION_BOUNDARIES_MANUFACTURING_X = (-40., -24., -8., 8., 24., 40.)
# Seven contiguous 16 mm target sections; retain only actually exported points.
MATERIAL_WINDOW_MM = (-56., 56.)
REGION_BOUNDARIES_X = None
CALIBRATION_CSV = (PROJECT_ROOT / 'post_tensile_test' / 'analysis_results' /
                   'mixed_infill_20261017' / 'group_summary_and_residuals.csv')
CALIBRATION_METHOD = 'independent'
REGISTRATION = {
    'mode': 'material_coordinates',
    'matrix': ((0., 1.), (-1., 0.)),  # x=+Y, y=-X; ux=DY, uy=-DX.
    'material_x_sign': -1,  # Manufacturing infill coordinate decreases along FEM x.
    'specimen_length_mm': 220.,  # Manufacturing nominal length.
    'low_infill_mark_end_distance_mm': 85.,
    'high_infill_mark_end_distance_mm': 73.,
    'raw_axial_anchor_mm': -21.50749765093,  # Original region mark midpoint in raw Y0.
    'manufacturing_anchor_mm': 6.,  # Midpoint of manufacturing marks -25/+37 mm.
    'shared_reference_confirmed': True,
    'anchor_prepared_metadata_sha256': '4ea6dd2939ba290b60ee9c6fc89b04758b2d95386e9f45c365138d3d8204dbba',
    'length_scale_source': 'independent_dic_calibration',
    'note': 'all_region and region share the initial image and millimetre coordinates, as confirmed. Inherit the region mark-midpoint raw Y0=-21.50749765093 at manufacturing xi=6 mm. Retain material increase along raw -Y; same-point forward direction comparison saved in results/dic_full/direction_check. Select the continuous requested xi=[-56,56] window intersected with available data, preserving raw nodes and units.',
}
OUTSIDE_POINT_POLICY = 'reject'
FIT_CASES, PREDICT_CASES = ('180N',), ('280N',)

# ==================== 派生设置：通常不需要修改 ====================
if (not RUN_NAME or RUN_NAME in ('.', '..') or
        any(character in RUN_NAME for character in '<>:"/\\|?*')):
    raise ValueError('RUN_NAME应为单个目录名，例如 run01')
if DATA_MODE not in ('dic', 'synthetic_layered', 'synthetic_nodal'):
    raise ValueError('DATA_MODE应为dic、synthetic_layered或synthetic_nodal')
if MESH_SOURCE not in ('dic_grid', 'configured_grid'):
    raise ValueError('MESH_SOURCE应为dic_grid或configured_grid')
if REGULARIZATION_MODE not in ('gradient', 'huber_tv', 'exact_tv'):
    raise ValueError('REGULARIZATION_MODE应为gradient、huber_tv或exact_tv')
if not isinstance(GAMMA_COUNT, int) or GAMMA_COUNT < 5:
    raise ValueError('GAMMA_COUNT必须为不少于5的整数')
CASE_ID = f'sample2_run3_{RUN_NAME}'
CASE_OUTPUT = PROJECT_ROOT / 'results' / 'dic_full' / RUN_NAME
NEL_X, NEL_Y = CONFIGURED_ELEMENTS if MESH_SOURCE == 'configured_grid' else (None, None)
if DATA_MODE != 'dic':
    SYNTHETIC_REPRESENTATION = ('sharp_layers' if DATA_MODE == 'synthetic_layered' else 'nodal_q4')
    PRIMARY_NU, NU_VALUES = SYNTHETIC_NU, (SYNTHETIC_NU,)
if GAMMA_LOG10_RANGE is None:
    GAMMA_LOG10_RANGE = (-12., -4.) if REGULARIZATION_MODE == 'gradient' else (-10., -2.)
if len(GAMMA_LOG10_RANGE) != 2 or not GAMMA_LOG10_RANGE[0] < GAMMA_LOG10_RANGE[1]:
    raise ValueError('GAMMA_LOG10_RANGE应为递增的两个log10端点')
GAMMA_CANDIDATES = tuple(10.**(GAMMA_LOG10_RANGE[0] +
    (GAMMA_LOG10_RANGE[1]-GAMMA_LOG10_RANGE[0])*i/(GAMMA_COUNT-1)) for i in range(GAMMA_COUNT))
