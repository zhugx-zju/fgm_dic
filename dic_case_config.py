"""唯一日常配置：1026均匀60%试样，逐次独立H1反演。"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
RUN_NAME = '60sample_1026_run3_h1_20261026'
INPUTS_CONFIGURED = True
DATA_MODE = 'dic'  # dic / synthetic_layered / synthetic_nodal
MESH_SOURCE = 'configured_grid'  # configured_grid / dic_grid
CONFIGURED_ELEMENTS = (121, 19)
REGULARIZATION_MODE = 'gradient'  # gradient / huber_tv / exact_tv
TV_DELTA = 10.0  # MPa/mm，仅Huber-TV使用。
DISPLACEMENT_COMPONENT_WEIGHTS = (1., 1.)
PRIMARY_NU = .26
NU_VALUES = (.24, .26, .28, .33)
SYNTHETIC_NU = .26
GAMMA_COUNT = 50
GAMMA_LOG10_RANGE = None  # H1默认(-12,-4)，TV默认(-10,-2)。
MAX_ITER, MAX_FUN = 1500, 12000
FTOL, GTOL = 1e-30, 1e-12
EXACT_TV_SOLVER = 'primal_dual'
# 数值步长为旧粗网格起点，不保证新数据收敛；失败保留，不自动增预算。
EXACT_TV_PRIMAL_STEP, EXACT_TV_COUPLING = 2e7, .9
EXACT_TV_OPTIONS = dict(max_outer=12000, max_inner=150, stationarity_atol=1e-9,
    stationarity_rtol=1e-3, primal_atol=1e-6, primal_rtol=1e-4,
    gap_atol=1e-9, gap_rtol=1e-4, adapt_interval=10)

# 三次各自使用DIC参考坐标和用户指定的始末载荷差；不合并参考图。
SAMPLE_ROOT = PROJECT_ROOT / 'post_tensile_test' / '1026' / '60sample'
PLATFORMS = {'run3': {'folder': 'D:\\MyProject\\uniform_pressure_experiment_1023\\post_tensile_test\\1026\\60sample\\3\\380N', 'stages': (235, 249), 'run': 3, 'setting_N': 380, 'force_N': 129.05999999999997, 'force_note': 'User-defined end-minus-start force: end 379.78479000 N minus record-start 250.72479000 N. End force last measured record; plateau mean differs by 0.135 N. DIC reference/force-start coincidence unverified.'}}
FIT_CASES, PREDICT_CASES = ('run3',), ()
FORCE_PAIRING_CSV = SAMPLE_ROOT / 'selected_dic_windows.csv'
FORCE_REFERENCE = 'force_increment_from_record_start'
FORCE_REFERENCE_NOTE = '用户指定始末载荷差值。第3次：379.78479000 - 250.72479000 = 129.06000000 N。末载为载荷文件最后一条记录，窗口内均值为379.6495967 N。载荷记录起点与DIC第0帧未同步核实；绝对模量是此参考对应假设下的条件性结果。'
FORCE_ASSUMED = False
DIC_UNIT = 'mm'
DIC_COORDINATE_STATE = 'deformed'
DISPLACEMENT_REFERENCE = 'initial_reference_image'
DOMAIN_SOURCE = 'dic_reference_bounds'
GEO_L, GEO_H, THICKNESS = None, None, 3.2
LOAD_SECTION_WIDTH = 18.36575062254
INTERNAL_SIDE_BOUNDARY = 'traction_free_approximation'
REGION_INFILL_PERCENT = (60,)
REGION_BOUNDARIES_MANUFACTURING_X = ()
REGION_BOUNDARIES_X = ()
REGION_BOUNDARIES_CENTERED_X = ()
CALIBRATION_CSV = PROJECT_ROOT / 'post_tensile_test' / '1023' / 'analysis_results' / 'modulus_20261026_28_final' / 'group_summary.csv'
CALIBRATION_METHOD = 'common_stress_window_OLS'
REGISTRATION = {'mode': 'assumed_center', 'matrix': [[0.0, 1.0], [1.0, 0.0]], 'note': '均匀试样无材料分区；DIC原Y作为拉伸轴x，原X作为横向y；坐标与位移共用换轴，保持毫米长度。参考点云居中仅确定原点，不缩放。DIC覆盖宽度按用户要求作为承载宽度，厚度近似3.2 mm。'}
OUTSIDE_POINT_POLICY = 'reject'

if not RUN_NAME or RUN_NAME in ('.', '..') or any(c in RUN_NAME for c in '<>:"/\\|?*'):
    raise ValueError('RUN_NAME必须为单个目录名')
if DATA_MODE not in ('dic', 'synthetic_layered', 'synthetic_nodal'):
    raise ValueError('未知DATA_MODE')
if MESH_SOURCE not in ('configured_grid', 'dic_grid'):
    raise ValueError('未知MESH_SOURCE')
if REGULARIZATION_MODE not in ('gradient', 'huber_tv', 'exact_tv'):
    raise ValueError('未知REGULARIZATION_MODE')
CASE_ID = f'project1023_{RUN_NAME}'
CASE_OUTPUT = PROJECT_ROOT / 'results' / 'dic_full' / RUN_NAME
NEL_X, NEL_Y = CONFIGURED_ELEMENTS if MESH_SOURCE == 'configured_grid' else (None, None)
if DATA_MODE != 'dic':
    SYNTHETIC_REPRESENTATION = 'sharp_layers' if DATA_MODE == 'synthetic_layered' else 'nodal_q4'
    PRIMARY_NU, NU_VALUES = SYNTHETIC_NU, (SYNTHETIC_NU,)
if GAMMA_LOG10_RANGE is None:
    GAMMA_LOG10_RANGE = (-12., -4.) if REGULARIZATION_MODE == 'gradient' else (-10., -2.)
if not isinstance(GAMMA_COUNT, int) or GAMMA_COUNT < 5:
    raise ValueError('GAMMA_COUNT至少为5')
if len(GAMMA_LOG10_RANGE) != 2 or not GAMMA_LOG10_RANGE[0] < GAMMA_LOG10_RANGE[1]:
    raise ValueError('γ范围必须递增')
GAMMA_CANDIDATES = tuple(10.**(GAMMA_LOG10_RANGE[0]+(GAMMA_LOG10_RANGE[1]-GAMMA_LOG10_RANGE[0])*i/(GAMMA_COUNT-1)) for i in range(GAMMA_COUNT))
