"""Discrete L-curve corner selection, independent of the FE solver and plots.

Use rho=||W_A**.5 r|| and eta=sqrt(integral(|grad E|**2)), without gamma
inside either norm. Curvature is measured in log(rho), log(eta) coordinates.
See P. C. Hansen, The L-curve and its use in the numerical treatment of
inverse problems (2000), https://www.imm.dtu.dk/documents/ftp/tr99/tr15_99.pdf.
This is a discrete three-point approximation, not the adaptive pruning method.
"""

import numpy as np

MIN_LOG_EDGE = 1e-4
MONOTONIC_LOG_TOL = 1e-5
MIN_POSITIVE_CURVATURE = 1e-6


def select_l_curve_corner(gamma, residual_norm, regularization_norm, converged):
    """Return a resolved interior positive-curvature corner, or an explicit failure.

    Entries and returned arrays have the caller's order. Nonpositive/invalid
    norms and unsuccessful solves are excluded. Three adjacent valid points
    are required; holes are never bridged. Increasing gamma should increase
    rho and decrease eta. At least two scan points must bracket each side of
    the selected corner. Collinear/coincident curves cannot select a gamma.
    """
    gamma, rho, eta = [np.asarray(value, dtype=float) for value in
                       (gamma, residual_norm, regularization_norm)]
    success = np.asarray(converged, dtype=bool)
    if (gamma.ndim != 1 or len(gamma) < 5 or
            any(value.shape != gamma.shape for value in (rho, eta, success)) or
            not np.all(np.isfinite(gamma) & (gamma > 0)) or
            len(np.unique(gamma)) != len(gamma)):
        raise ValueError('L-curve needs at least five distinct positive gamma values and aligned arrays')
    order = np.argsort(gamma)
    good = success & np.isfinite(rho) & (rho > 0) & np.isfinite(eta) & (eta > 0)
    curvature = np.full(len(gamma), np.nan)
    result = {'selected_gamma': None, 'optimal_index': None, 'curvature': curvature,
              'eligible': good, 'status': 'insufficient_valid_points',
              'message': 'At least five converged solutions with positive finite norms are required.'}
    if good.sum() < 5:
        return result
    for position in range(1, len(order)-1):
        triple = order[position-1:position+2]
        if not good[triple].all():
            continue
        x, y = np.log(rho[triple]), np.log(eta[triple])
        # Small optimization fluctuations are tolerated; reversed branches are
        # not silently converted to positive curvature with an absolute value.
        if np.any(np.diff(x) < -MONOTONIC_LOG_TOL) or np.any(np.diff(y) > MONOTONIC_LOG_TOL):
            continue
        points = np.column_stack([x, y])
        first, second = points[1]-points[0], points[2]-points[1]
        lengths = np.array([np.linalg.norm(first), np.linalg.norm(second),
                            np.linalg.norm(points[2]-points[0])])
        # Near-coincident solves can have enormous circle curvature from tiny
        # optimization fluctuations. Require a resolved log-norm edge first.
        if lengths.min() <= MIN_LOG_EDGE:
            continue
        cross = first[0]*second[1]-first[1]*second[0]
        curvature[triple[1]] = 2*cross/np.prod(lengths)
    candidates = np.flatnonzero(np.isfinite(curvature) & (curvature > MIN_POSITIVE_CURVATURE))
    if not len(candidates):
        result.update(status='no_resolved_corner',
                      message='No resolved positive-curvature corner; inspect convergence and scan norms.')
        return result
    best = int(candidates[np.argmax(curvature[candidates])])
    position = int(np.flatnonzero(order == best)[0])
    if position < 2 or position > len(order)-3:
        result.update(status='range_insufficient',
                      message='Maximum curvature is too near a scan end; use a wider gamma range.')
        return result
    if not good[order[position-2:position+3]].all():
        result.update(status='insufficient_corner_support',
                      message='The curvature peak is not bracketed by two valid solves on each side.')
        return result
    result.update(selected_gamma=float(gamma[best]), optimal_index=best,
                  status='selected', message='Selected maximum positive discrete log-log curvature.')
    return result
