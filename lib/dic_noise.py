"""Temporal DIC diagnostics in export units, without independence assumptions."""

import numpy as np

from .experiment_data import DICFrameSeries


def _cosine_correlation(first: np.ndarray, second: np.ndarray) -> float | None:
    """Normalized pooled cross-product; samples must already be time-centered."""
    denominator = np.linalg.norm(first.ravel()) * np.linalg.norm(second.ravel())
    if denominator <= np.finfo(float).tiny:
        return None
    return float(np.sum(first * second) / denominator)


def _component_quantiles(values: np.ndarray) -> dict:
    return {label: np.quantile(values, q, axis=0).tolist()
            for label, q in (('p10', .1), ('median', .5), ('p90', .9), ('p95', .95))}


def temporal_projection_baseline(time_s: np.ndarray) -> dict:
    """IID temporal null moments after the same centering/detrending.

    This is a ratio of expected cross-products, not an exact expected sample
    correlation or a significance test. It prevents attributing projection-
    induced negative lag correlations and fitted slopes directly to physics.
    Arbitrary fixed spatial covariance cancels from these pooled null moments.
    """
    time_s = np.asarray(time_s, dtype=float)
    centered_time = time_s - time_s.mean()
    count = len(time_s)
    if count < 3 or np.sum(centered_time**2) <= 0:
        raise ValueError('At least three distinct sample times are required')
    center = np.eye(count) - np.ones((count, count)) / count
    trend = center - np.outer(centered_time, centered_time) / np.sum(centered_time**2)
    correlations = []
    for label, projection in (('cleaned', center), ('detrended', trend)):
        variance = np.diag(projection)
        for lag in range(1, min(5, count)):
            correlations.append({'processing': label, 'lag_frames': lag,
                                 'normalized_expected_cross_product': float(
                                     np.diag(projection, k=lag).sum() / np.sqrt(
                                         variance[:-lag].sum() * variance[lag:].sum()))})
    return {'linear_trend_expected_energy_fraction': 1. / (count - 1),
            'temporal_correlations': correlations,
            'scope': 'IID temporal null moments only; not proof of independent frames.'}


def _decompose(coordinates: np.ndarray, displacement: np.ndarray,
               time_s: np.ndarray) -> dict:
    """Equal-point least-squares rigid projection, followed by temporal OLS.

    Remove only frame-to-mean translation and infinitesimal in-plane rotation.
    Per-point linear trends are diagnostic and must not be removed from the
    actual displacement mean used in a mechanical inverse problem.
    """
    xy = coordinates - coordinates.mean(axis=0)
    radius_squared = np.sum(xy**2)
    if radius_squared <= 0:
        raise ValueError('Distinct spatial points are required to identify rotation')
    centered = displacement - displacement.mean(axis=0)
    translations = centered.mean(axis=1)
    rotations = np.sum(-xy[None, :, 1] * centered[:, :, 0]
                       + xy[None, :, 0] * centered[:, :, 1], axis=1) / radius_squared
    rigid = translations[:, None, :] + rotations[:, None, None] * np.column_stack(
        [-xy[:, 1], xy[:, 0]])[None, :, :]
    cleaned = centered - rigid
    time_centered = time_s - time_s.mean()
    slope = np.einsum('t,tpc->pc', time_centered, cleaned) / np.sum(time_centered**2)
    detrended = cleaned - time_centered[:, None, None] * slope[None, :, :]
    duration = time_s[-1] - time_s[0]
    return {'centered': centered, 'cleaned': cleaned, 'detrended': detrended,
            'translations': translations, 'rotations': rotations, 'slope': slope,
            'drift': slope * duration,
            'raw_sd': np.std(centered, axis=0, ddof=1),
            'cleaned_sd': np.std(cleaned, axis=0, ddof=1),
            'detrended_sd': np.sqrt(np.sum(detrended**2, axis=0) / (len(time_s) - 2)),
            'robust_mad': 1.4826 * np.median(np.abs(
                detrended - np.median(detrended, axis=0)), axis=0)}


def analyze_dic_series(series: DICFrameSeries, frame_rate_hz: float) -> tuple[dict, dict]:
    """Return diagnostics and arrays; no conversion to mm or noise-of-mean claim."""
    xy = np.asarray(series.coordinates, dtype=float)
    u = np.asarray(series.frame_displacements, dtype=float)
    ids = np.asarray(series.grid_ids)
    if (u.ndim != 3 or u.shape[2] != 2 or u.shape[0] < 5 or
            xy.shape != u.shape[1:] or ids.shape != xy.shape):
        raise ValueError('Expected at least five frames with coordinates/IDs (points,2) '
                         'and displacement (frames,points,2)')
    if not np.isfinite(frame_rate_hz) or frame_rate_hz <= 0 or not (
            np.all(np.isfinite(u)) and np.all(np.isfinite(xy))):
        raise ValueError('Finite coordinates/displacements and a positive frame rate are required')
    if not np.all(np.isfinite(ids)) or not np.array_equal(ids, np.round(ids)):
        raise ValueError('Spatial correlations require integer DIC grid IDs')
    if len(np.unique(ids, axis=0)) != len(ids):
        raise ValueError('Spatial correlations require unique DIC grid IDs')
    stages = np.array([record['stage'] for record in series.metadata['sources']])
    if len(stages) != len(u) or np.any(np.diff(stages) <= 0):
        raise ValueError('Frame stages must be strictly increasing and match the sample count')
    time_s = (stages - stages[0]) / frame_rate_hz
    fields = _decompose(xy, u, time_s)
    rms = np.sqrt(np.mean(fields['cleaned']**2, axis=(1, 2)))
    detrended_rms = np.sqrt(np.mean(fields['detrended']**2, axis=(1, 2)))
    representative = int(np.argmin(rms))
    energy = np.sum(fields['cleaned']**2, axis=(0, 1))
    trend_fraction = np.divide(
        energy - np.sum(fields['detrended']**2, axis=(0, 1)), energy,
        out=np.zeros(2), where=energy > 0)
    correlations = []
    for label in ('cleaned', 'detrended'):
        values = fields[label]
        for lag in range(1, min(5, len(u))):
            correlations.append({'processing': label, 'lag_frames': lag,
                                 'correlation': [_cosine_correlation(
                                     values[:-lag, :, c], values[lag:, :, c]) for c in range(2)]})
    spatial = []
    lookup = {tuple(key): i for i, key in enumerate(ids)}
    for axis in range(2):
        for lag in (1, 2, 4, 8):
            pairs = []
            for i, key in enumerate(ids):
                other = key.copy()
                other[axis] += lag
                j = lookup.get(tuple(other))
                if j is not None:
                    pairs.append((i, j))
            if not pairs:
                continue
            first, second = np.array(pairs).T
            spatial.append({'grid_axis': ('row', 'column')[axis], 'lag_indices': lag,
                            'pair_count': len(pairs), 'median_distance_export_units': float(
                                np.median(np.linalg.norm(xy[first] - xy[second], axis=1))),
                            'correlation': [_cosine_correlation(
                                fields['detrended'][:, first, c],
                                fields['detrended'][:, second, c]) for c in range(2)]})
    windows = []
    half = max(3, len(u) // 2)
    for label, window in (('full', slice(None)), ('first_half', slice(0, half)),
                          ('last_half', slice(-half, None))):
        part = _decompose(xy, u[window], time_s[window])
        part_rms = np.sqrt(np.mean(part['cleaned']**2, axis=(1, 2)))
        windows.append({'window': label, 'stages': stages[window].tolist(),
                        'median_detrended_sd': np.median(part['detrended_sd'], axis=0).tolist(),
                        'median_absolute_drift': np.median(np.abs(part['drift']), axis=0).tolist(),
                        'representative_stage': int(stages[window][np.argmin(part_rms)])})
    summary = {
        'unit': 'unverified_export_length_unit', 'frame_rate_hz': frame_rate_hz,
        'frame_count': len(u), 'point_count': len(xy), 'duration_s': float(time_s[-1]),
        'component_order': ['DX', 'DY'], 'point_weighting': 'equal diagnostic point weights',
        'raw_temporal_sd': _component_quantiles(fields['raw_sd']),
        'rigid_removed_temporal_sd': _component_quantiles(fields['cleaned_sd']),
        'detrended_temporal_sd': _component_quantiles(fields['detrended_sd']),
        'detrended_robust_mad': _component_quantiles(fields['robust_mad']),
        'absolute_linear_drift_over_window': _component_quantiles(np.abs(fields['drift'])),
        'linear_trend_energy_fraction_after_rigid_removal': trend_fraction.tolist(),
        'temporal_correlations': correlations, 'spatial_correlations': spatial,
        'temporal_projection_null_baseline': temporal_projection_baseline(time_s),
        'window_sensitivity': windows,
        'representative_stage_after_rigid_removal': int(stages[representative]),
        'representative_frame_rms_to_mean': float(rms[representative]),
        'interpretation': ('Residual scatter includes measurement variation and unresolved '
                           'physical evolution. Correlations are descriptive pooled cross-products, '
                           'not confidence limits. No independent frames/points or sqrt(N) reduction '
                           'is assumed. A single platform mean does not remove shared reference error.'),
        'sources': series.metadata,
    }
    arrays = dict(coordinates=xy, grid_ids=ids, stages=stages, time_s=time_s,
                  frame_displacements=u, mean_displacement=u.mean(axis=0),
                  frame_rms_to_mean=rms, detrended_frame_rms=detrended_rms, **fields)
    return summary, arrays
