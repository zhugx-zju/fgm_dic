"""统一材料算法 P0/P1 小规模诊断；与真实 DIC 主配置隔离。

四个阶段分别运行 python -m lib.unified_gradient_layered_study
prepare / forward / inverse / compare。inverse 一次只运行指定的小算例。
本配置不启动正式 P2、噪声选参、真实 DIC 或大网格扫描。
"""
from pathlib import Path

RUN_NAME = 'p0_p1_20261022'
CASE_OUTPUT = Path(__file__).resolve().parent/'results'/'unified_gradient_layered'/RUN_NAME
CASES = ('continuous_2d', 'layered', 'mixed_2d')
GEO_L = 6.0  # mm，小规模合成诊断域，不是实验 ROI
GEO_H = 2.0
NEL_X = 6
NEL_Y = 3
THICKNESS = 3.2  # mm
BEARING_WIDTH = 20.0  # mm
FORCE = 96.0  # N，合成净拉力；ROI 端力为 FORCE*GEO_H/BEARING_WIDTH
NU = 0.26
QUADRATURE_ORDER = 3
QUADRATURE_SUBDIVISIONS = 2
EPSILON_MM = 0.3  # 有限平滑宽度，不能称为尖锐界面
REFERENCE_REFINEMENT = 2  # 仅诊断位移离散误差，不用于调参

# 真值只用于准备／正演及最终比较，不传给反演初值和目标。
TRUTH_INTERFACE_FRACTION = 0.47
# 已知界面协议的显式信息；未知界面协议不读取此值。
KNOWN_INTERFACE_FRACTION = 0.47
UNKNOWN_INTERFACE_STARTS = (0.35, 0.65)
UNKNOWN_INTERFACE_BOUNDS = (0.1, 0.9)  # 域内可行区间，非真值附近窄窗
MODULUS_BOUNDS_MPA = (100.0, 6000.0)
METHODS = ('continuous_q4', 'layered_constant', 'unified_q4')
GAMMA = 0.0  # P1 数据项诊断；不称为含噪稳定算法，不用真值选择 gamma
MAX_ITERATIONS = 500
MAX_EVALUATIONS = 800  # 每次识别的硬正演预算，另外记录初值拟合一次正演
FTOL = 1e-16
GTOL = 1e-11
SEED = 20261022
EVALUATION_NX = 121
EVALUATION_NY = 41
