"""Strict platform import and independent tensile-calibration data adapters."""

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re

import numpy as np
import pandas as pd

from .dic_fields import restore_reference_coordinates
from .io_utils import read_dic_csv, remove_duplicate_grid_points


@dataclass(frozen=True)
class DICPlatform:
    """Reference coordinates and time-mean displacements, all in export units."""

    coordinates: np.ndarray
    displacement: np.ndarray
    temporal_std: np.ndarray
    grid_ids: np.ndarray
    metadata: dict


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load_dic_platform(folder: Path, first_stage: int, last_stage: int) -> DICPlatform:
    """Average matched grid IDs; retain temporal SD without sqrt(frame count)."""
    paths = sorted(Path(folder).glob('stage_*.comdat_comdata.csv'))
    selected = {}
    for path in paths:
        stage = int(re.search(r'stage_(\d+)', path.name).group(1))
        if first_stage <= stage <= last_stage:
            if stage in selected:
                raise ValueError(f'Duplicate stage {stage} in {folder}')
            selected[stage] = path
    expected = set(range(first_stage, last_stage + 1))
    if set(selected) != expected:
        raise ValueError(f'{folder}: missing stages {sorted(expected - selected.keys())}')
    frames, sources = [], []
    for stage in sorted(selected):
        path = selected[stage]
        raw = read_dic_csv(path)
        finite = raw.dropna(subset=['grid_row', 'grid_col', 'X', 'Y', 'DX', 'DY'])
        if not np.allclose(finite[['grid_row', 'grid_col']],
                           np.round(finite[['grid_row', 'grid_col']]), rtol=0, atol=0):
            raise ValueError(f'{path}: noninteger DIC grid IDs')
        unique = remove_duplicate_grid_points(finite)
        frame = restore_reference_coordinates(unique).set_index(['grid_row', 'grid_col'])
        frames.append(frame)
        sources.append({'path': str(path), 'stage': stage, 'sha256': file_sha256(path),
                        'raw_rows': len(raw), 'valid_rows': len(finite),
                        'duplicates_removed': len(finite) - len(unique)})
    common = frames[0].index
    for frame in frames[1:]:
        common = common.intersection(frame.index)
    common = common.sort_values()
    if len(common) < 2:
        raise ValueError(f'{folder}: fewer than two common valid points')
    coordinates = frames[0].loc[common, ['X_ref', 'Y_ref']].to_numpy()
    for frame in frames[1:]:
        if not np.allclose(frame.loc[common, ['X_ref', 'Y_ref']], coordinates,
                           rtol=0, atol=1e-5):
            raise ValueError(f'{folder}: reference coordinates change across platform frames')
    displacement = np.stack([f.loc[common, ['DX', 'DY']].to_numpy() for f in frames])
    metadata = {
        'sources': sources, 'frame_count': len(frames), 'common_points': len(common),
        'valid_points_per_frame': [len(f) for f in frames],
        'reference_bounds': np.stack([coordinates.min(axis=0), coordinates.max(axis=0)]).tolist(),
        'reference_span': np.ptp(coordinates, axis=0).tolist(),
        'displacement_reference': 'initial_unloaded_image',
        'noise_note': 'Temporal SD is diagnostic; frames are not assumed independent.',
    }
    return DICPlatform(coordinates, displacement.mean(axis=0),
                       displacement.std(axis=0, ddof=1) if len(frames) > 1
                       else np.zeros_like(coordinates), np.array(common.tolist()), metadata)


def load_calibration(path: Path, method: str = 'independent',
                     infill_percent=(40, 50, 60, 70, 80, 90, 100)) -> pd.DataFrame:
    """Retain group SD, count and batch; do not replace groups by a fitted line."""
    table = pd.read_csv(path)
    table = table.loc[table['fit_method'] == method].sort_values('infill_percent').copy()
    selected = list(infill_percent)
    if not selected or len(set(selected)) != len(selected):
        raise ValueError('Requested infill groups must be nonempty and distinct')
    table = table[table['infill_percent'].isin(selected)]
    if table['infill_percent'].duplicated().any() or set(table['infill_percent']) != set(selected):
        raise ValueError(f'Calibration must contain exactly one group per requested infill: {selected}')
    table = table.set_index('infill_percent').loc[selected].reset_index()
    if not np.all(np.isfinite(table['mean_E_MPa']) & (table['mean_E_MPa'] > 0)):
        raise ValueError('Calibration modulus must be finite and positive (MPa)')
    return table[['infill_percent', 'batch', 'n', 'mean_E_MPa', 'sd_E_MPa', 'fit_method']]


def load_platform_force(path: Path, setting_N: int) -> tuple[float, dict]:
    table = pd.read_csv(path)
    row = table[(table['run'] == 3) & (table['setting_N'] == setting_N)]
    if len(row) != 1:
        raise ValueError(f'Expected one run 3 force pairing for {setting_N} N')
    record = row.iloc[0].to_dict()
    force = float(record['mean_corrected_force_N'])
    if not np.isfinite(force) or force <= 0 or not record['full_force_coverage']:
        raise ValueError('Platform does not have a valid measured force pairing')
    return force, record


def transform_dic_platform(platform, matrix, offset, *, verified=False):
    """Map both vectors and coordinates using an explicitly verified similarity.

    x_FEM = A x_DIC + b and u_FEM = A u_DIC. Registration b locates the
    physical ROI and zone boundaries; it must not be inferred from data extrema.
    """
    if not verified or matrix is None or offset is None:
        raise ValueError('DIC coordinate scale and physical ROI registration are unverified; '
                         'set verified matrix and offset in experiment_config.py first')
    matrix, offset = np.asarray(matrix, dtype=float), np.asarray(offset, dtype=float)
    if matrix.shape != (2, 2) or offset.shape != (2,) or not (
            np.all(np.isfinite(matrix)) and np.all(np.isfinite(offset))):
        raise ValueError('Expected finite DIC matrix (2,2) and offset (2,)')
    metric = matrix @ matrix.T
    if metric[0, 0] <= 0 or not np.allclose(metric, metric[0, 0] * np.eye(2)):
        raise ValueError('DIC transform must be a rotation/reflection with one uniform unit scale')
    xy = platform.coordinates @ matrix.T + offset
    uv = platform.displacement @ matrix.T
    # The admitted matrix can rotate component covariance. Without measured
    # covariance use a conservative scalar SD; never invent independent axes.
    std = np.linalg.norm(platform.temporal_std, axis=1) * np.sqrt(metric[0, 0])
    return xy, uv, std
