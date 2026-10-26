"""唯一日常配置：H1主线，Huber-TV／精确TV对照；1023输入尚未配对。"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
RUN_NAME = 'h1_run01'
INPUTS_CONFIGURED = False  # 完成docs/input_setup.md中的真实输入配置后改为True。
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

# 1023目录目前只有工作簿及既有标定资料，没有本流程要求的逐帧DIC CSV。
# 不沿用旧试样的帧号、60.1 N偏置、319.9 N假定力或材料坐标锚点。
SAMPLE_ROOT = PROJECT_ROOT / 'post_tensile_test' / '1023'
PLATFORMS = {}  # {平台名: {'folder': Path(...), 'stages': (首帧,末帧), 'run': ..., 'setting_N': ...}}
FIT_CASES, PREDICT_CASES = (), ()
FORCE_PAIRING_CSV = SAMPLE_ROOT / 'selected_dic_windows.csv'
FORCE_REFERENCE = 'corrected_force_from_unloaded_state'
FORCE_REFERENCE_NOTE = '待核对当前试样的参考图、净力与逐帧配对。'
FORCE_ASSUMED = False
DIC_UNIT = 'mm'
DIC_COORDINATE_STATE = 'deformed'
DISPLACEMENT_REFERENCE = 'initial_unloaded_image'
DOMAIN_SOURCE = 'dic_reference_bounds'
GEO_L, GEO_H, THICKNESS = None, None, 3.2
LOAD_SECTION_WIDTH = 20.  # 制造名义尺寸；新试样计算前核对实际尺寸。
INTERNAL_SIDE_BOUNDARY = 'traction_free_approximation'
REGION_INFILL_PERCENT = (100, 90, 80, 70, 60, 50, 40)
REGION_BOUNDARIES_MANUFACTURING_X = (-40., -24., -8., 8., 24., 40.)
MATERIAL_WINDOW_MM = None
REGION_BOUNDARIES_X = None
CALIBRATION_CSV = SAMPLE_ROOT / 'analysis_results' / 'modulus_20261026_28_final' / 'group_summary.csv'
CALIBRATION_METHOD = 'common_stress_window_OLS'
REGISTRATION = {'mode': 'pending', 'note': '当前试样的方向、毫米尺度及材料位置尚未确认。'}
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
