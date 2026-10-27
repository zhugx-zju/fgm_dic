"""唯一日常配置。所有路径相对本文件，导入时不读取数据或历史结果。"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
CASE_PROFILE = 'example'  # example：自带合成示例 / experiment：本地1026 sample_2全段
RUN_NAME = 'example_h1_002'  # 新计算更换名字，所有数值步骤拒绝覆盖。
INPUT_ROOT = PROJECT_ROOT/'data'/'example'
OUTPUT_ROOT = PROJECT_ROOT/'results'
CASE_OUTPUT, CASE_ID = OUTPUT_ROOT/RUN_NAME, RUN_NAME

# 独立标定：逐根路径、数据块和实测尺寸在specimens.csv中声明。
CALIBRATION_MANIFEST = INPUT_ROOT/'calibration'/'specimens.csv'
CALIBRATION_OUTPUT = CASE_OUTPUT/'00_calibration'
CALIBRATION_CSV = CALIBRATION_OUTPUT/'group_summary.csv'
CALIBRATION_BATCH = 'synthetic_example'
CALIBRATION_METHOD = 'individual_window_joint_3pct_monotone'  # 或individual_highest_R2
CALIBRATION_BASELINE_POINTS = 10
CALIBRATION_MAX_DEVIATION, CALIBRATION_TIME_LIMIT = .03, 40.
CALIBRATION_SETTINGS = dict(search_start_fraction=.05, search_end_fraction=.80,
    grid_step_fraction=.025, minimum_width_fraction=.10, minimum_points=60,
    minimum_r2=.995, maximum_half_slope_difference=.05, minimum_positive_strain_steps=.90)
# 组内±3%和模量不减是联合选段约束，不是独立验证结论。

# DIC选帧。事件区间用s，fps和0基连续帧必须按本次采集确认。
SPECIMEN_LABEL = 'Synthetic example'
FRAME_SELECTION_OUTPUT = CASE_OUTPUT/'00_frame_selection'
FORCE_PAIRING_CSV = FRAME_SELECTION_OUTPUT/'selected_dic_windows.csv'
USE_FRAME_SELECTION = True  # 执行第1步时从选帧表读取窗口，配置导入时不读结果。
FRAME_SELECTION_OPTIONS = dict(delay_after_entry_s=6., latest_after_entry_s=9., frame_count=15,
    minimum_settle_s=4., maximum_tail_slope_ratio=.25, sensitivity_shifts_s=(-1.,0.,1.),
    maximum_force_gap_s=.1, maximum_force_range_fraction=.02, maximum_peak_difference_fraction=.02)
FIGURE_LANGUAGES, FORCE_PLOT_LIMITS = ('en','zh'), (0.,110.)
TIMING_RUNS = [dict(run=1, setting_N=90, folder=INPUT_ROOT/'timing'/'run1',
    force_file='force.csv', displacement_file='disp.csv', fps=5.,
    zero_based_uninterrupted_confirmed=True, synchronization='events',
    events=[dict(machine_interval=(1.,8.), machine_search=(3.,7.), dic_interval=(3.,10.), dic_search=(5.,9.)),
            dict(machine_interval=(7.,13.), machine_search=(8.,12.), dic_interval=(9.,15.), dic_search=(10.,14.))],
    tail_interval=(17.,24.), tail_search=(18.,22.), reference_force_N=10.,
    reference_note='Synthetic stage 0 reference has exactly 10 N.')]
PLATFORMS = {'run1': dict(folder=str(INPUT_ROOT/'dic'/'run1'), stages=None, run=1, setting_N=90)}
FIT_CASES, PREDICT_CASES = ('run1',), ()  # 不同参考图分次运行。
FORCE_REFERENCE = 'force_increment_from_record_start'
FORCE_REFERENCE_NOTE = 'Synthetic force increment matches the initial reference image exactly.'
FORCE_ASSUMED = False

# 物理模型。mm、N、MPa，平面应力；坐标和位移用同一保长变换。
INPUTS_CONFIGURED, DATA_MODE, DATA_PROVENANCE = True, 'dic', 'synthetic_demo'
# DATA_MODE可选dic / synthetic_layered / synthetic_nodal。示例CSV本身已是合成数据。
DIC_UNIT, DIC_COORDINATE_STATE = 'mm', 'reference'
DISPLACEMENT_REFERENCE = 'initial_reference_image'
DOMAIN_SOURCE = 'configured_geometry'
GEO_L, GEO_H, THICKNESS, LOAD_SECTION_WIDTH = 48.,8.,3.2,8.
INTERNAL_SIDE_BOUNDARY = 'physical_free_edges'
REGISTRATION = dict(mode='explicit_verified', matrix=[[1.,0.],[0.,1.]], offset_mm=[0.,0.],
    note='Synthetic CSV positions are exact FEM reference coordinates in mm.')
REGION_INFILL_PERCENT, REGION_BOUNDARIES_X = (40,70,100), (16.,32.)
REGION_BOUNDARIES_CENTERED_X, REGION_BOUNDARIES_MANUFACTURING_X = (), ()
OUTSIDE_POINT_POLICY = 'reject'
SHOW_SMOOTH_REFERENCE = True  # 事后辅助显示，分层标定线同时保留。
SMOOTH_REFERENCE_COORDINATES = 'fem'  # fem / manufacturing
SMOOTH_INFILL_ANCHORS = ((8.,40.), (40.,100.))  # 两对(物理x,填充率)，明确线性phi(x)，不外推。

# 三种正则共用流程。各正则gamma单位不同，切换方法须重新设置扫描范围。
REGULARIZATION_MODE = 'gradient'  # gradient（H1） / huber_tv / exact_tv
MESH_SOURCE, CONFIGURED_ELEMENTS = 'configured_grid', (12,4)  # 真实实验默认121×19。
DISPLACEMENT_COMPONENT_WEIGHTS = (1.,1.)  # 残差平方系数，不缩放位移。
PRIMARY_NU, NU_VALUES = .26, (.26,)
GAMMA_COUNT, GAMMA_LOG10_RANGE = 17, (-12.,-4.)
MAX_ITER, MAX_FUN, FTOL, GTOL = 500,5000,1e-12,1e-10
TV_DELTA = 10.  # MPa/mm，仅Huber-TV使用。
EXACT_TV_SOLVER, EXACT_TV_PRIMAL_STEP, EXACT_TV_COUPLING = 'primal_dual',2e7,.9
EXACT_TV_OPTIONS = dict(max_outer=12000, max_inner=150, stationarity_atol=1e-9,
    stationarity_rtol=1e-3, primal_atol=1e-6, primal_rtol=1e-4, gap_atol=1e-9, gap_rtol=1e-4, adapt_interval=10)
SYNTHETIC_NU, SYNTHETIC_REPRESENTATION = .26, 'nodal_q4'

# 本地实验预设：保留原始目录，不依赖历史反演结果。所有假设明确记录。
if CASE_PROFILE == 'experiment':
    SAMPLE_RUN = 3  # 每次独立参考图分别运行1/2/3。
    INPUT_ROOT = PROJECT_ROOT/'post_tensile_test'/'1026'/'sample_2'
    CALIBRATION_BATCH = '1023'
    CALIBRATION_MANIFEST = PROJECT_ROOT/'data'/'local'/'calibration'/'specimens.csv'
    CALIBRATION_OUTPUT = CASE_OUTPUT/'00_calibration'
    CALIBRATION_CSV = CALIBRATION_OUTPUT/'group_summary.csv'
    FRAME_SELECTION_OUTPUT = CASE_OUTPUT/'00_frame_selection'
    FORCE_PAIRING_CSV = FRAME_SELECTION_OUTPUT/'selected_dic_windows.csv'
    _runs = {
        1: dict(stages=(591,605), setting=280, start=50.7776, width=18.43550794901,
            events=[((7,19),(9,15),(10,25),(12,20)), ((17,29),(19,25),(20,34),(24,30)),
                    ((53,66),(57,63),(57,71),(61,67)), ((64,77),(66,73),(67,81),(70,77))], tail=((105,125),(109,119))),
        2: dict(stages=(625,639), setting=380, start=70.6192, width=18.4636725353,
            events=[((34,47),(37,44),(39,53),(42,50)), ((45,58),(48,55),(49,64),(53,60)),
                    ((69,82),(72,79),(75,89),(78,85)), ((80,93),(83,90),(85,100),(88,96))], tail=((112,131),(116,126))),
        3: dict(stages=(689,703), setting=480, start=82.2848, width=19.3018040506,
            events=[((20,33),(23,30),(24,38),(27,34)), ((31,44),(34,41),(35,49),(38,45)),
                    ((83,97),(87,94),(87,102),(91,98)), ((94,108),(97,104),(98,113),(101,109))], tail=((123,144),(128,138))),
    }
    _r = _runs[SAMPLE_RUN]
    SPECIMEN_LABEL, FORCE_PLOT_LIMITS = 'sample_2',(20.,525.)
    TIMING_RUNS = [dict(run=SAMPLE_RUN, setting_N=_r['setting'], folder=INPUT_ROOT/str(SAMPLE_RUN),
        force_file='force.xlsb', displacement_file='disp.csv', fps=5., zero_based_uninterrupted_confirmed=True,
        synchronization='events', events=[dict(zip(('machine_interval','machine_search','dic_interval','dic_search'),e)) for e in _r['events']],
        tail_interval=_r['tail'][0], tail_search=_r['tail'][1], reference_force_N=_r['start'],
        reference_note='User-accepted assumption: first recorded force corresponds to DIC reference; not synchronized.')]
    USE_FRAME_SELECTION, FORCE_ASSUMED = False, True  # 历史窗口缺同期力；保留用户已接受的显式假设。
    FORCE_REFERENCE_NOTE = (f'Assumed nominal end force {_r["setting"]} N minus first recorded {_r["start"]} N; '
        'end-hold force and reference-image force are not synchronized.')
    PLATFORMS = {f'run{SAMPLE_RUN}': dict(folder=str(INPUT_ROOT/str(SAMPLE_RUN)/f'{_r["setting"]}N'),
        stages=_r['stages'], run=SAMPLE_RUN, setting_N=_r['setting'],
        force_N=_r['setting']-_r['start'], force_note=FORCE_REFERENCE_NOTE)}
    FIT_CASES, PREDICT_CASES = (f'run{SAMPLE_RUN}',), ()
    DATA_PROVENANCE, DIC_COORDINATE_STATE, DOMAIN_SOURCE = 'experiment','deformed','dic_reference_bounds'
    GEO_L, GEO_H, LOAD_SECTION_WIDTH = None,None,_r['width']
    REGION_INFILL_PERCENT, REGION_BOUNDARIES_X = (40,50,60,70,80,90,100), ()
    REGION_BOUNDARIES_MANUFACTURING_X = (-40.,-24.,-8.,8.,24.,40.)
    SMOOTH_REFERENCE_COORDINATES = 'manufacturing'
    SMOOTH_INFILL_ANCHORS = ((-48.,40.), (48.,100.))
    REGISTRATION = dict(mode='specimen_marks', matrix=[[0.,1.],[1.,0.]], mark_reference='roi_axial_extrema',
        length_scale_source='independent_dic_calibration', specimen_length_mm=220.,
        low_infill_mark_end_distance_mm=53., high_infill_mark_end_distance_mm=53., material_x_sign=1,
        note='Conditional mark-midpoint placement; +Y toward high infill. Keep DIC mm scale, never stretch to 114 mm. '
             'DIC width assumed load-bearing width; thickness about 3.2 mm.')
    CONFIGURED_ELEMENTS, NU_VALUES = (121,19),(.24,.26,.28,.33)
    GAMMA_COUNT, GAMMA_LOG10_RANGE = 50,(-12.,-4.)
    MAX_ITER, MAX_FUN, FTOL, GTOL = 1500,12000,1e-30,1e-12
elif CASE_PROFILE != 'example':
    raise ValueError('CASE_PROFILE must be example or experiment')

# 派生值。日常只修改上面的参数。
if not RUN_NAME or RUN_NAME in ('.','..') or any(c in RUN_NAME for c in '<>:"/\\|?*'):
    raise ValueError('RUN_NAME must be one directory name')
if GAMMA_COUNT < 5 or GAMMA_LOG10_RANGE[0] >= GAMMA_LOG10_RANGE[1]:
    raise ValueError('Require at least five gamma candidates in an increasing range')
NEL_X, NEL_Y = CONFIGURED_ELEMENTS if MESH_SOURCE == 'configured_grid' else (None,None)
GAMMA_CANDIDATES = tuple(10.**(GAMMA_LOG10_RANGE[0]+i*(GAMMA_LOG10_RANGE[1]-GAMMA_LOG10_RANGE[0])/(GAMMA_COUNT-1)) for i in range(GAMMA_COUNT))
