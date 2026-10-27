"""唯一日常配置：1026 sample_2的40%–70%区段，逐次独立自由二维H1反演。"""
from pathlib import Path
import hashlib
import json

PROJECT_ROOT = Path(__file__).resolve().parent
SAMPLE_RUN = 2  # 1/2/3分别执行四个入口；参考图不合并。
RUN_NAME = f'sample2_1026_run{SAMPLE_RUN}_h1_infill40_70_20261027'
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

# 用户确认：厚度、DIC跨度作为几何/承载宽度、名义末载及起载参考均为假设。
# 标线各距端部53 mm；沿原+Y位移较大一端为高填充率。制造界面不缩放。
SAMPLE_ROOT = PROJECT_ROOT / 'post_tensile_test' / '1026' / 'sample_2'
_runs = {
    1: dict(stages=(591, 605), setting_N=280, start_N=50.7776, width_mm=18.43550794901),
    2: dict(stages=(625, 639), setting_N=380, start_N=70.6192, width_mm=18.4636725353),
    3: dict(stages=(689, 703), setting_N=480, start_N=82.2848, width_mm=19.3018040506),
}
_run = _runs[SAMPLE_RUN]
_name = f'run{SAMPLE_RUN}'
_delta_force = round(_run['setting_N']-_run['start_N'], 8)
FORCE_REFERENCE_NOTE = (
    f"用户明确接受假设：第{SAMPLE_RUN}次名义末载{_run['setting_N']} N减去记录首载"
    f"{_run['start_N']:.8f} N，增量{_delta_force:.8f} N。候选窗口无同期载荷；"
    '假定末端保持名义载荷及记录起载对应DIC参考图，二者均未同步验证。'
)
PLATFORMS = {_name: {
    'folder': str(SAMPLE_ROOT/str(SAMPLE_RUN)/f"{_run['setting_N']}N"),
    'stages': _run['stages'], 'run': SAMPLE_RUN, 'setting_N': _run['setting_N'],
    'force_N': _delta_force, 'force_note': FORCE_REFERENCE_NOTE,
}}
FIT_CASES, PREDICT_CASES = (_name,), ()
FORCE_PAIRING_CSV = SAMPLE_ROOT / 'selected_dic_windows.csv'
FORCE_REFERENCE = 'force_increment_from_record_start'
FORCE_ASSUMED = True
DIC_UNIT = 'mm'
DIC_COORDINATE_STATE = 'deformed'
DISPLACEMENT_REFERENCE = 'initial_reference_image'
DOMAIN_SOURCE = 'dic_reference_bounds'
GEO_L, GEO_H, THICKNESS = None, None, 3.2
LOAD_SECTION_WIDTH = _run['width_mm']
INTERNAL_SIDE_BOUNDARY = 'traction_free_approximation'
REGION_INFILL_PERCENT = (40, 50, 60, 70, 80, 90, 100)
REGION_BOUNDARIES_MANUFACTURING_X = (-40., -24., -8., 8., 24., 40.)
REGION_BOUNDARIES_X = ()
REGION_BOUNDARIES_CENTERED_X = ()
CALIBRATION_CSV = PROJECT_ROOT / 'post_tensile_test' / '1023' / 'analysis_results' / 'individual_3pct_monotone_20261026_final' / 'group_summary.csv'
CALIBRATION_METHOD = 'individual_window_joint_3pct_monotone'
REGISTRATION = {
    'mode': 'specimen_marks', 'matrix': [[0., 1.], [1., 0.]],
    'mark_reference': 'roi_axial_extrema', 'length_scale_source': 'independent_dic_calibration',
    'specimen_length_mm': 220., 'low_infill_mark_end_distance_mm': 53.,
    'high_infill_mark_end_distance_mm': 53., 'material_x_sign': 1,
    'note': '用户确认导出ROI在两条标线之间，各标线距端部53 mm；制造文件给出220 mm总长和七区界面。'
            '高填充率在位移较大端，本次原+Y端DY较大，FEM x沿原+Y，填充率沿x为40至100%。'
            '软件毫米尺度作为独立于标线间距的长度来源，尺度本身按用户口径假设有效；'
            '两标线中点对齐，不缩放DIC实际跨度为名义114 mm。DIC覆盖宽度假定为承载宽度，厚度约3.2 mm。',
}
OUTSIDE_POINT_POLICY = 'reject'

# 只继承同一次、同一参考图全段准备结果的材料坐标，绝不读取其模量或γ。
# 裁剪后不能再将新区段中点对齐试样中点。各次参考图之间不共享锚点。
MATERIAL_ANCHOR_PREPARED = (PROJECT_ROOT / 'results' / 'dic_full' /
    f'sample2_1026_run{SAMPLE_RUN}_h1_monotone_20261026' / '01_prepared' / 'run_metadata.json')
_anchor_bytes = MATERIAL_ANCHOR_PREPARED.read_bytes()
_anchor = json.loads(_anchor_bytes)
_source_platform = _anchor['configuration']['PLATFORMS'][_name]
if (Path(_source_platform['folder']).resolve() != Path(PLATFORMS[_name]['folder']).resolve()
        or tuple(_source_platform['stages']) != tuple(PLATFORMS[_name]['stages'])):
    raise ValueError('区段裁剪锚点必须来自本次相同帧号和参考图')
MATERIAL_WINDOW_MM = (-57., 8.)
REGION_INFILL_PERCENT = (40, 50, 60, 70)
REGION_BOUNDARIES_MANUFACTURING_X = (-40., -24., -8.)
REGISTRATION = dict(REGISTRATION, mode='material_coordinates',
    shared_reference_confirmed=True,
    raw_axial_anchor_mm=-_anchor['offset_mm'][0],
    manufacturing_anchor_mm=_anchor['material_registration']['manufacturing_x_offset_mm'],
    anchor_prepared_metadata_sha256=hashlib.sha256(_anchor_bytes).hexdigest(),
    note=REGISTRATION['note'] + '仅沿用本次同一参考图全段坐标锚点，截取40%–70%区段；'
         '保持坐标与位移毫米尺度，截取端均匀牵引为新增边界近似。')

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
