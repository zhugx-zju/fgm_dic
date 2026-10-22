"""380 N exploratory comparison. Change METHOD, then run the original four steps."""
from dic_coarse_config import *

METHOD = 'h1'
SUPPORTED_METHODS = ('h1', 'h1_uy10', 'huber1', 'huber10', 'exact_tv', 'exact_tv_fine')
if METHOD not in SUPPORTED_METHODS:
    raise ValueError(f'Unknown method: {METHOD}')
RUN_NAME = 'load380_20261022'
DATA_MODE = 'dic'
CASE_ID = f'sample2_run3_{RUN_NAME}_{METHOD}'
CASE_OUTPUT = PROJECT_ROOT / 'results' / 'dic_full' / RUN_NAME / METHOD
PLATFORMS = {
    '380N': {'folder': SAMPLE_ROOT/'3'/'380N_2'/'all_region',
             'stages': (618,632), 'run': 3, 'setting_N': 380,
             'force_N': 319.9,
             'force_note': 'Exploratory force authorized by the user: nominal 380 N minus initial 60.1 N offset. No synchronized measured force covers stages 618-632. This is an assumption, not a measured platform force.'},
    '180N': {'folder': SAMPLE_ROOT/'3'/'180N_2'/'all_region',
             'stages': (259,273), 'run': 3, 'setting_N': 180},
    '280N': {'folder': SAMPLE_ROOT/'3'/'280N_2'/'all_region',
             'stages': (436,450), 'run': 3, 'setting_N': 280},
}
FIT_CASES, PREDICT_CASES = ('380N',), ('180N','280N')
FORCE_REFERENCE_NOTE = ('380N uses assumed 319.9 N with user authorization; 180N/280N retain measured paired forces. All displacement fields refer to the initial unloaded image and use the original 60.1 N offset convention.')
FORCE_ASSUMED = True
DISPLACEMENT_COMPONENT_WEIGHTS = (1.,10.) if METHOD == 'h1_uy10' else (1.,1.)
REGULARIZATION_MODE = ('exact_tv' if METHOD.startswith('exact_tv') else
                       'huber_tv' if METHOD.startswith('huber') else 'gradient')
TV_DELTA = 1. if METHOD == 'huber1' else 10.
MESH_SOURCE = 'configured_grid'
NEL_X,NEL_Y = (242,38) if METHOD == 'exact_tv_fine' else (121,19)
CONFIGURED_ELEMENTS = (NEL_X,NEL_Y)
GAMMA_COUNT = 50
GAMMA_LOG10_RANGE = (-12.,-4.) if REGULARIZATION_MODE == 'gradient' else (-10.,-2.)
GAMMA_CANDIDATES = tuple(10.**(GAMMA_LOG10_RANGE[0]+8*i/49) for i in range(50))
# Fixed numerical steps account for the larger data-term curvature (force squared).
# They do not normalize the objective or change gamma or the KKT tolerances.
EXACT_TV_SOLVER = 'primal_dual'
EXACT_TV_PRIMAL_STEP = 8e7 if METHOD == 'exact_tv_fine' else 2e7
EXACT_TV_COUPLING = .9
EXACT_TV_OPTIONS = dict(max_outer=12000,max_inner=150,stationarity_atol=1e-9,
    stationarity_rtol=1e-3,primal_atol=1e-6,primal_rtol=1e-4,
    gap_atol=1e-9,gap_rtol=1e-4,adapt_interval=10)
NU_VALUES,PRIMARY_NU = (.24,.26,.28,.33),.26
