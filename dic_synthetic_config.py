"""Noiseless sharp-layer recovery on the existing full-region coarse geometry."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
CASE_ID = 'full_coarse_synthetic_layered'
CASE_OUTPUT = PROJECT_ROOT / 'results' / 'dic_full' / 'synthetic' / 'layered'
# This preparation has the current kernel and exactly the original coarse inputs.
SYNTHETIC_SOURCE_PREPARED = PROJECT_ROOT / 'results' / 'dic_full' / 'tv' / 'delta10' / '01_prepared'
SYNTHETIC_REPRESENTATION = 'sharp_layers'
SYNTHETIC_NU = .26
NU_VALUES, PRIMARY_NU = (.26,), .26
REGULARIZATION_MODE = 'gradient'
DISPLACEMENT_COMPONENT_WEIGHTS = (1., 1.)
GAMMA_CANDIDATES = tuple(10.**(-12 + 8*i/49) for i in range(50))
MAX_ITER, MAX_FUN = 1500, 12000
FTOL, GTOL = 1e-30, 1e-12
# Controlled diagnostics, not synthetic L-curve selections or truth-based tuning.
RECOVERY_DIAGNOSTIC_GAMMAS = (3.906939937054621e-9, 1e-12)
