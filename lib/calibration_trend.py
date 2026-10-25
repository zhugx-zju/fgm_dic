"""Independent calibration trend for post-processing only (mm, MPa).

Fit the group means equally in nominal infill, then use the existing material
interfaces to map infill to x. Never infer geometry from the clipped ROI ends.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd


TREND_NOTE = ('紫色点划线：七组独立标定均值等权线性趋势，仅作事后参照；'
              '不是连续材料实测真值，不用于反演、初值、选γ或修改独立标定正演。'
              '阶梯与灰带仍表示实际分层标定均值及试样间SD。')


def calibration_trend(calibration, boundaries, x):
    """Return E=a+b*p(x) on the calibrated infill range; elsewhere NaN.

    p is in percentage points (40,...,100). Each physical interface is assigned
    the midpoint of its two neighboring nominal infills. This auxiliary affine
    trend coordinate does not alter the actual piecewise manufacturing field.
    """
    table = pd.DataFrame(calibration).copy()
    p = table['infill_percent'].to_numpy(dtype=float)
    values = table['mean_E_MPa'].to_numpy(dtype=float)
    boundaries = np.asarray(boundaries, dtype=float)
    x = np.asarray(x, dtype=float)
    if (len(p) < 3 or boundaries.shape != (len(p)-1,) or
            not np.isfinite(np.r_[p, values, boundaries, x.ravel()]).all() or
            np.any(values <= 0) or np.any(np.diff(boundaries) <= 0) or
            not (np.all(np.diff(p) > 0) or np.all(np.diff(p) < 0))):
        raise ValueError('Trend needs ordered calibration groups and matching increasing interfaces')
    if 'fit_method' in table and set(table.fit_method) != {'independent'}:
        raise ValueError('Trend comparison requires the independent calibration convention')
    interface_p = (p[:-1]+p[1:])/2
    mapping = np.polynomial.polynomial.polyfit(boundaries, interface_p, 1)
    if not np.allclose(np.polynomial.polynomial.polyval(boundaries, mapping),
                       interface_p, rtol=0, atol=1e-8):
        raise ValueError('Interfaces do not define an affine nominal-infill trend coordinate')
    coefficients = np.polynomial.polynomial.polyfit(p, values, 1)
    mapped_p = np.polynomial.polynomial.polyval(x, mapping)
    prediction = np.polynomial.polynomial.polyval(mapped_p, coefficients)
    prediction = np.where((mapped_p >= p.min()-1e-10) &
                          (mapped_p <= p.max()+1e-10), prediction, np.nan)
    fitted = np.polynomial.polynomial.polyval(p, coefficients)
    table['trend_anchor_x_mm'] = (p-mapping[0])/mapping[1]
    table['trend_fitted_E_MPa'] = fitted
    table['mean_minus_trend_MPa'] = values-fitted
    residual = float(np.sum((values-fitted)**2))
    total = float(np.sum((values-values.mean())**2))
    info = dict(definition='equal-group OLS E=a+b*p; p(x) from physical interfaces',
                role='postprocessing_only_not_material_truth',
                intercept_MPa=float(coefficients[0]),
                slope_MPa_per_infill_percent=float(coefficients[1]),
                infill_at_x_zero_percent=float(mapping[0]),
                infill_slope_percent_per_mm=float(mapping[1]),
                slope_MPa_per_mm=float(coefficients[1]*mapping[1]),
                calibrated_infill_range_percent=[float(p.min()), float(p.max())],
                trend_support_x_mm=[float(table.trend_anchor_x_mm.min()),
                                    float(table.trend_anchor_x_mm.max())],
                group_fit_r_squared=1-residual/total if total > 0 else None,
                group_fit_rmse_MPa=float(np.sqrt(residual/len(p))),
                extrapolation=False, inverse_or_selection_use=False,
                note=TREND_NOTE)
    return prediction, table, info


def add_calibration_trend(axis, calibration, boundaries, length, *, output=None, english=False):
    """Draw the same purple reference and optionally save its complete provenance."""
    _, anchors, info = calibration_trend(calibration, boundaries, np.array([0., length]))
    low, high = info['trend_support_x_mm']
    x = np.linspace(max(0., low), min(float(length), high), 801)
    values, _, _ = calibration_trend(calibration, boundaries, x)
    axis.plot(x, values, color='#7B3294', linestyle='-.', linewidth=2.,
              label='Calibration linear trend (post hoc)' if english else '独立标定线性趋势（仅事后参照）')
    if output is not None:
        output = Path(output)
        anchors.to_csv(output/'calibration_trend_anchors.csv', index=False, encoding='utf-8-sig')
        pd.DataFrame({'x_mm': x, 'calibration_trend_E_MPa': values}).to_csv(
            output/'calibration_trend.csv', index=False)
        (output/'calibration_trend.json').write_text(
            json.dumps(info, ensure_ascii=False, indent=2), encoding='utf-8')
    return info
