"""Composite Q4 integration on a copy of a uniform rectangular mesh.

The original mesh and its standard 2x2 rule remain unchanged. Fixed straight
sharp interfaces split elements in physical x. Smooth moving interfaces use
fixed integration points, so their derivatives contain no moving-rule terms.
"""
from __future__ import annotations

from copy import copy
import numpy as np


def rectangular_quadrature(mesh, order: int = 3, subdivisions: int = 1,
                           x_breaks: tuple[float, ...] = ()):
    """Return a mesh copy with a composite rule; lengths are in mm.

    ``gauss_det_j`` on this copy includes the subcell Gauss weight, with
    ``gauss_w=1``. Thus every existing assembly/contraction uses the same
    physical integration measure. Padded cut subcells have zero weight.
    Only fixed straight cuts are supported; moving sharp cuts need shape
    derivatives and are deliberately excluded from the inverse prototype.
    """
    if (not isinstance(order, int) or order < 2 or
            not isinstance(subdivisions, int) or subdivisions < 1):
        raise ValueError('order >= 2 and subdivisions >= 1 must be integers')
    if getattr(mesh, 'measured_nodal_grid', False):
        raise ValueError('Composite research quadrature requires a configured uniform mesh')
    cuts = np.asarray(x_breaks, dtype=float)
    if (cuts.ndim != 1 or not np.isfinite(cuts).all() or
            np.any(np.diff(cuts) <= 0) or np.any((cuts <= 0) | (cuts >= mesh.geo_l))):
        raise ValueError('Fixed x_breaks must be strictly ordered inside the domain')
    a, w = np.polynomial.legendre.leggauss(order)
    nseg = subdivisions + len(cuts)
    q = nseg * subdivisions * order**2
    s = np.zeros((q, mesh.n_el))
    t = np.zeros_like(s)
    measure = np.zeros_like(s)
    y_edges = np.linspace(-1, 1, subdivisions + 1)
    for e in range(mesh.n_el):
        x0 = mesh.x_ele[e, 0]
        internal = cuts[(cuts > x0) & (cuts < x0 + mesh.el)]
        edges = np.unique(np.r_[np.linspace(-1, 1, subdivisions + 1),
                                  2 * (internal - x0) / mesh.el - 1])
        offset = 0
        for left, right in zip(edges[:-1], edges[1:]):
            for bottom, top in zip(y_edges[:-1], y_edges[1:]):
                ss, tt = np.meshgrid((left+right)/2 + a*(right-left)/2,
                                     (bottom+top)/2 + a*(top-bottom)/2)
                size = order**2
                s[offset:offset+size, e] = ss.ravel()
                t[offset:offset+size, e] = tt.ravel()
                measure[offset:offset+size, e] = (np.outer(w, w).ravel()
                    * (right-left)*(top-bottom)*mesh.el*mesh.eh/16)
                offset += size
    sx, sy = np.array([-1, 1, 1, -1]), np.array([-1, -1, 1, 1])
    N = (1+s[..., None]*sx)*(1+t[..., None]*sy)/4
    dx = sx*(1+t[..., None]*sy)/(2*mesh.el)
    dy = sy*(1+s[..., None]*sx)/(2*mesh.eh)
    B = np.zeros((q, mesh.n_el, 3, 8))
    B[..., 0, 0::2], B[..., 1, 1::2] = dx, dy
    B[..., 2, 0::2], B[..., 2, 1::2] = dy, dx
    Nm = np.zeros((q, mesh.n_el, 2, 8))
    Nm[..., 0, 0::2], Nm[..., 1, 1::2] = N, N
    result = copy(mesh)
    result.gauss_N = N
    result.gauss_dNdx, result.gauss_dNdy = dx, dy
    result.gauss_w = np.ones(q)
    result.gauss_det_j = result.gauss_abs_det_j = measure
    result.B_gauss = B
    result.gauss_mass_local = np.einsum('geki,gekj->geij', Nm, Nm)
    result.M = result.L_reg = result._integrated_gradients = None
    result._unit_stiffness_cache = None
    result.quadrature_metadata = dict(order=order, subdivisions=subdivisions,
                                      fixed_x_breaks=cuts.tolist(), points_per_element=q)
    return result
