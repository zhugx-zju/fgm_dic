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


@dataclass(frozen=True)
class DICFrameSeries:
    """Matched reference points and displacements with shape (frames, points, 2)."""

    coordinates: np.ndarray
    frame_displacements: np.ndarray
    grid_ids: np.ndarray
    metadata: dict


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load_dic_frame_series(folder: Path, first_stage: int, last_stage: int, *,
                          coordinate_state: str = 'deformed') -> DICFrameSeries:
    """Read matched frames once, retaining all temporal samples for diagnostics."""
    if coordinate_state not in {'deformed', 'reference'}:
        raise ValueError("DIC coordinate_state must be 'deformed' or 'reference'")
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
        if coordinate_state == 'deformed':
            # This requires X/Y and DX/DY to share the export length unit.
            frame = restore_reference_coordinates(unique)
        else:
            frame = unique.assign(X_ref=unique['X'], Y_ref=unique['Y'])
        frame = frame.set_index(['grid_row', 'grid_col'])
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
        'csv_coordinate_state': coordinate_state,
        'reference_coordinate_rule': ('[X-DX,Y-DY]' if coordinate_state == 'deformed'
                                       else '[X,Y]'),
        'noise_note': 'Temporal SD is diagnostic; frames are not assumed independent.',
    }
    return DICFrameSeries(coordinates, displacement, np.array(common.tolist()), metadata)


def load_dic_platform(folder: Path, first_stage: int, last_stage: int, *,
                      coordinate_state: str = 'deformed') -> DICPlatform:
    """Average matched grid IDs; retain temporal SD without sqrt(frame count)."""
    series = load_dic_frame_series(folder, first_stage, last_stage,
                                   coordinate_state=coordinate_state)
    displacement = series.frame_displacements
    return DICPlatform(series.coordinates, displacement.mean(axis=0),
                       displacement.std(axis=0, ddof=1) if len(displacement) > 1
                       else np.zeros_like(series.coordinates), series.grid_ids, series.metadata)


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


def check_platform_reference(platforms: dict[str, DICPlatform]) -> dict:
    """Check shared grid IDs against one initial reference, before registration."""
    frames = {name: pd.DataFrame(platform.coordinates,
                                index=pd.MultiIndex.from_arrays(platform.grid_ids.T))
              for name, platform in platforms.items()}
    if not frames:
        return {}
    first_name = next(iter(frames))
    first = frames[first_name]
    report = {}
    for name, frame in frames.items():
        common = first.index.intersection(frame.index)
        difference = (frame.loc[common].to_numpy() - first.loc[common].to_numpy())
        max_error = float(np.max(np.abs(difference))) if len(common) else None
        if max_error is not None and max_error > 1e-5:
            raise ValueError(f'{name}: shared reference grid differs from {first_name}; '
                             'check coordinate_state and initial reference image')
        report[name] = {'reference_platform': first_name, 'common_grid_points': len(common),
                        'max_coordinate_difference_export_units': max_error}
    return report


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
    # Normalize before comparison: absolute tolerances on A A^T would admit
    # anisotropic scaling when converting very small export units to mm.
    scale_squared = float(np.trace(metric) / 2)
    if not np.isfinite(scale_squared) or scale_squared <= 0 or not np.allclose(
            metric / scale_squared, np.eye(2), rtol=1e-7, atol=1e-10):
        raise ValueError('DIC transform must be a rotation/reflection with one uniform unit scale')
    xy = platform.coordinates @ matrix.T + offset
    uv = platform.displacement @ matrix.T
    # The admitted matrix can rotate component covariance. Without measured
    # covariance use a conservative scalar SD; never invent independent axes.
    std = np.linalg.norm(platform.temporal_std, axis=1) * np.sqrt(scale_squared)
    return xy, uv, std


def dic_roi_mask(coordinates: np.ndarray, length: float, width: float) -> np.ndarray:
    """Admit boundary roundoff using the Q4 observation operator's tolerance."""
    tol = 1e-9 * max(length, width, 1.) + 1e-12
    return ((coordinates[:, 0] >= -tol) & (coordinates[:, 0] <= length + tol) &
            (coordinates[:, 1] >= -tol) & (coordinates[:, 1] <= width + tol))


def dic_registration_report(platform: DICPlatform, matrix, offset, length: float,
                            width: float, boundaries, *, verified=False) -> dict:
    """Audit physical placement without inferring orientation from data extrema."""
    report = {'status': 'verified' if verified else 'pending_physical_registration',
              'fem_domain_mm': [[0., 0.], [length, width]],
              'fem_x_region_boundaries_mm': list(boundaries),
              'raw_reference_bounds': platform.metadata['reference_bounds'],
              'raw_reference_span': platform.metadata['reference_span'],
              'coordinate_rule': platform.metadata['reference_coordinate_rule'],
              'unit_contract': 'X,Y,DX,DY must share one checked export length unit',
              'placement_rule': 'One physical registration for all platforms; no min-shift or span fitting'}
    if verified:
        xy, _, _ = transform_dic_platform(platform, matrix, offset, verified=True)
        valid = dic_roi_mask(xy, length, width)
        regions = np.searchsorted(boundaries, xy[valid, 0], side='right')
        report.update(matrix=np.asarray(matrix).tolist(), offset_mm=np.asarray(offset).tolist(),
                      transformed_bounds_mm=[xy.min(axis=0).tolist(), xy.max(axis=0).tolist()],
                      points_inside_roi=int(valid.sum()), points_outside_roi=int((~valid).sum()),
                      points_per_region=np.bincount(regions, minlength=len(boundaries)+1).tolist())
    return report
