"""Compiled fixed-geometry region map, numerically equivalent to full branches.

All Q4 coefficients and noncompact switch tails are retained. Only fixed
weights and quadratic forms are cached; there is no support truncation or
binding of y rows. Moving geometry continues through RegionMaterialMap.
"""
import numpy as np
from scipy.sparse import hstack

from .unified_material import RegionMaterialMap


class FixedRegionMaterialMap(RegionMaterialMap):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.optimize_geometry:
            raise ValueError('Compiled fixed region map cannot optimize geometry')
        _, _, chi, *_ = self._state(self.initial_parameters(1.))
        N, dx, dy = self.basis
        self._value_map = hstack([N.multiply(chi[:, i, None]) for i in range(self.n_regions)],
                                  format='csr')
        self._regional_laplacians = []
        for i in range(self.n_regions):
            weight = self.weights*chi[:, i]
            matrix = dx.T @ dx.multiply(weight[:, None]) + dy.T @ dy.multiply(weight[:, None])
            self._regional_laplacians.append(matrix.tocsr())

    def values(self, theta):
        c, _ = self.unpack(theta)
        return np.asarray(self._value_map @ c.ravel()).reshape(self.gauss_shape)

    def pullback(self, theta, sensitivity):
        c, _ = self.unpack(theta)
        if np.shape(sensitivity) != self.gauss_shape:
            raise ValueError('Sensitivity must have shape (quadrature, elements)')
        return c.ravel()*np.asarray(self._value_map.T @ np.asarray(sensitivity).ravel()).ravel()

    def regularization(self, theta):
        c, _ = self.unpack(theta)
        value = 0.
        gradient = np.empty_like(c)
        for i, matrix in enumerate(self._regional_laplacians):
            # Constants are exactly in the gradient nullspace. Subtracting a
            # branch constant avoids cancellation of ~E^2 when R is near zero.
            delta = c[i]-c[i,0]
            delta -= np.mean(delta)
            product = matrix @ delta
            value += float(delta @ product)
            gradient[i] = 2*c[i]*(product-np.mean(product))
        if value < 0:
            raise FloatingPointError('Negative quadratic intra-region regularization')
        return value, gradient.ravel()
