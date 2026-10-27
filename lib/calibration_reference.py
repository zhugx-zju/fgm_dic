"""Post-hoc C2 calibration reference; never used by the inverse objective."""
import numpy as np
from scipy.interpolate import CubicSpline


def configured_smooth_reference(cfg, calibration):
    """Map a declared physical x-to-infill relation without extending its support."""
    if not getattr(cfg, 'SHOW_SMOOTH_REFERENCE', False):
        return None, {'status': 'disabled'}
    table = calibration.sort_values('infill_percent')
    try:
        spline, minimum = make_reference(table.infill_percent, table.mean_E_MPa)
        anchors = np.asarray(cfg.SMOOTH_INFILL_ANCHORS, dtype=float)
        if anchors.shape != (2, 2) or not np.isfinite(anchors).all() or anchors[0, 0] >= anchors[1, 0]:
            raise ValueError('Declare two increasing physical-x/infill anchor pairs')
        coefficient = (anchors[1, 1]-anchors[0, 1])/(anchors[1, 0]-anchors[0, 0])
        coordinate = cfg.SMOOTH_REFERENCE_COORDINATES
        if coordinate not in ('fem', 'manufacturing'):
            raise ValueError('Smooth reference coordinates must be fem or manufacturing')
        sign = cfg.MANUFACTURING_X_SIGN if coordinate == 'manufacturing' else 1.
        offset = cfg.MANUFACTURING_X_OFFSET_MM if coordinate == 'manufacturing' else 0.

        def evaluate(x):
            material_x = sign*np.asarray(x)+offset
            infill = anchors[0, 1]+coefficient*(material_x-anchors[0, 0])
            return spline(infill)

        return evaluate, dict(status='available', minimum_dE_dinfill=minimum,
            support_infill_percent=[float(table.infill_percent.min()), float(table.infill_percent.max())],
            anchors=anchors.tolist(), interpretation='Post-hoc interpolated reference, not measured continuous truth.')
    except ValueError as exc:
        # A display reference must never force monotonic calibration, extrapolate
        # a spline, or invalidate a completed inverse when assumptions fail.
        return None, dict(status='unavailable', reason=str(exc))


def make_reference(infill,modulus):
    """Interpolate the calibration means, checking monotonicity analytically."""
    infill,modulus=np.asarray(infill,dtype=float),np.asarray(modulus,dtype=float)
    if (infill.ndim!=1 or infill.shape!=modulus.shape or len(infill)<3 or
            not np.isfinite(infill).all() or not np.isfinite(modulus).all() or
            np.any(np.diff(infill)<=0) or np.any(np.diff(modulus)<0) or np.any(modulus<=0)):
        raise ValueError('Use increasing infill percentages and positive nondecreasing calibration means.')
    spline=CubicSpline(infill,modulus,bc_type='natural',extrapolate=False)
    minima=[]
    for index,h in enumerate(np.diff(infill)):
        a,b,c,_=spline.c[:,index]
        points=[0.,float(h)]
        if a!=0 and 0 < -b/(3*a) < h:
            points.append(float(-b/(3*a)))
        minima.extend((3*a*np.asarray(points)**2+2*b*np.asarray(points)+c).tolist())
        if index<len(infill)-2:
            np.testing.assert_allclose(3*a*h*h+2*b*h+c,spline.c[2,index+1],rtol=1e-10,atol=1e-10)
            np.testing.assert_allclose(6*a*h+2*b,2*spline.c[1,index+1],rtol=1e-10,atol=1e-10)
    if min(minima)<-1e-10:
        raise ValueError('Natural cubic interpolation is not monotone for these anchors; use a constrained C2 model instead of silently clipping.')
    np.testing.assert_allclose(spline(infill),modulus,rtol=0,atol=1e-10)
    return spline,float(min(minima))
