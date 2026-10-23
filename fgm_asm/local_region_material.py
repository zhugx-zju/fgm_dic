"""Exact support elimination for fixed sharp regional Q4 coefficient fields.

The active coefficients are original background-node coefficients, not values
at newly inserted interface nodes. Their box constraints are unchanged.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import block_diag

from .fixed_region_material import FixedRegionMaterialMap


class LocalRegionMaterialMap:
    def __init__(self, full_map: FixedRegionMaterialMap):
        if not full_map.sharp or full_map.optimize_geometry or full_map.branch_kind != 'q4':
            raise ValueError('Local support requires fixed sharp Q4 regions')
        self.full_map, self.mesh = full_map, full_map.mesh
        self.n_regions, self.optimize_geometry = full_map.n_regions, False
        x = self.mesh.coord[:, 0]
        axis = np.unique(x)
        col = np.searchsorted(axis, x)
        left = axis[np.maximum(col-1, 0)]
        right = axis[np.minimum(col+1, len(axis)-1)]
        boundaries = np.r_[0., full_map.interfaces[:, 0], self.mesh.geo_l]
        self.region_nodes, indices, self.slices = [], [], []
        offset = 0
        for i, (lo, hi) in enumerate(zip(boundaries[:-1], boundaries[1:])):
            nodes = np.flatnonzero((right > lo) & (left < hi))
            self.region_nodes.append(nodes)
            indices.extend(i*full_map.n_branch+nodes)
            self.slices.append(slice(offset, offset+len(nodes)))
            offset += len(nodes)
        self.active_indices = np.asarray(indices, dtype=int)
        self.n_material = self.n_parameters = offset
        self.gauss_shape = full_map.gauss_shape
        self._value_map = full_map._value_map[:, self.active_indices].tocsr()
        self._regional_laplacians = [matrix[nodes][:, nodes].tocsr()
            for matrix, nodes in zip(full_map._regional_laplacians, self.region_nodes)]
        self.coefficient_laplacian = block_diag(self._regional_laplacians, format='csr')
        self.support_mass = np.asarray(self._value_map.T @ full_map.weights).ravel()
        if np.any(self.support_mass <= 0):
            raise ValueError('Every retained coefficient must have positive integrated support')

    def unpack(self, theta):
        theta = np.asarray(theta, dtype=float)
        if theta.shape != (self.n_parameters,) or not np.isfinite(theta).all():
            raise ValueError(f'Expected {self.n_parameters} finite log coefficients')
        c = np.exp(theta)
        if not np.isfinite(c).all() or np.any(c <= 0):
            raise ValueError('Material coefficients must be finite and positive')
        return c, None

    def initial_parameters(self, uniform_modulus):
        if not np.isfinite(uniform_modulus) or uniform_modulus <= 0:
            raise ValueError('Initial modulus must be positive and finite')
        return np.full(self.n_parameters, np.log(uniform_modulus))

    def expand(self, theta, inactive_modulus=1000.):
        self.unpack(theta)
        full = self.full_map.initial_parameters(inactive_modulus)
        full[self.active_indices] = theta
        return full

    def values(self, theta):
        c, _ = self.unpack(theta)
        return np.asarray(self._value_map @ c).reshape(self.gauss_shape)

    def pullback(self, theta, sensitivity):
        c, _ = self.unpack(theta)
        if np.shape(sensitivity) != self.gauss_shape:
            raise ValueError('Sensitivity shape must match integration points/elements')
        return c*np.asarray(self._value_map.T @ np.asarray(sensitivity).ravel()).ravel()

    def regularization(self, theta):
        c, _ = self.unpack(theta)
        gradient, value = np.empty_like(c), 0.
        for segment, matrix in zip(self.slices, self._regional_laplacians):
            delta = c[segment]-c[segment][0]
            delta -= delta.mean()
            product = matrix @ delta
            value += float(delta @ product)
            gradient[segment] = 2*c[segment]*(product-product.mean())
        if value < 0:
            raise FloatingPointError('Negative regional gradient energy')
        return value, gradient

    def sample(self, theta, xy):
        return self.full_map.sample(self.expand(theta), xy)
