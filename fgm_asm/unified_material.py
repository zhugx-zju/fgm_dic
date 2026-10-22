"""Region switches times independent two-dimensional Q4 material branches.

Moduli use MPa, coordinates and interface controls use mm. Coefficients are
exp(m) BEFORE interpolation. All background nodes and y rows remain independent;
small-support coefficients are retained, including in smooth switch tails.
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import BSpline
from scipy.sparse import csr_matrix

from .fem_forward import gauss_shape_values
from .material import MaterialInfo


def q4_basis(mesh, xy):
    """Return value and physical x/y derivative matrices, shape (points,nodes)."""
    xy = np.asarray(xy, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2 or not np.isfinite(xy).all():
        raise ValueError('xy must be finite, shape (points, 2)')
    tol = 1e-10 * max(mesh.geo_l, mesh.geo_h, 1)
    if np.any(xy < -tol) or np.any(xy > np.array([mesh.geo_l, mesh.geo_h]) + tol):
        raise ValueError('Material evaluation points lie outside the background mesh')
    ex = np.clip(np.floor(xy[:, 0]/mesh.el).astype(int), 0, mesh.nel_x-1)
    ey = np.clip(np.floor(xy[:, 1]/mesh.eh).astype(int), 0, mesh.nel_y-1)
    s = np.clip((xy[:, 0]-ex*mesh.el)/mesh.el, 0, 1)
    t = np.clip((xy[:, 1]-ey*mesh.eh)/mesh.eh, 0, 1)
    nodes = mesh.ele_nods_id[ey*mesh.nel_x+ex] - 1
    values = np.column_stack([(1-s)*(1-t), s*(1-t), s*t, (1-s)*t])
    dx = np.column_stack([t-1, 1-t, t, -t])/mesh.el
    dy = np.column_stack([s-1, -s, s, 1-s])/mesh.eh
    rows = np.repeat(np.arange(len(xy)), 4)
    return tuple(csr_matrix((v.ravel(), (rows, nodes.ravel())),
                            shape=(len(xy), mesh.n_nod)) for v in (values, dx, dy))


class RegionMaterialMap:
    """Continuous Q4, layered constants, or region-wise free Q4 branches.

    Interfaces x=s_j(y) use a clamped B-spline of chosen degree and control
    count. Degree zero with one control is a straight interface. Fixed sharp
    straight interfaces require a matching cut integration rule. Sharp moving
    interfaces are rejected: the smooth derivative is not a shape derivative.
    """

    def __init__(self, mesh, *, n_regions=2, branch_kind='q4', interfaces=None,
                 optimize_geometry=False, epsilon=0.3, geometry_degree=0,
                 geometry_controls=1, sharp=False):
        if branch_kind not in {'q4', 'constant'}:
            raise ValueError("branch_kind must be 'q4' or 'constant'")
        if not isinstance(n_regions, int) or n_regions < 1:
            raise ValueError('n_regions must be a positive integer')
        if (not isinstance(geometry_degree, int) or geometry_degree < 0 or
                not isinstance(geometry_controls, int) or geometry_controls <= geometry_degree):
            raise ValueError('geometry_controls must exceed nonnegative geometry_degree')
        if (not np.isfinite(epsilon) or epsilon <= 0 or
                (sharp and (optimize_geometry or geometry_controls != 1))):
            raise ValueError('Use epsilon > 0; sharp interfaces must be fixed and straight')
        self.mesh, self.n_regions = mesh, n_regions
        self.branch_kind, self.sharp = branch_kind, sharp
        self.optimize_geometry = bool(optimize_geometry and n_regions > 1)
        self.epsilon = float(epsilon)
        self.degree, self.n_controls = geometry_degree, geometry_controls
        interior = np.linspace(0, mesh.geo_h, geometry_controls-geometry_degree+1)[1:-1]
        self.knots = np.r_[np.zeros(geometry_degree+1), interior,
                           np.full(geometry_degree+1, mesh.geo_h)]
        expected = (n_regions-1, geometry_controls)
        if interfaces is None:
            interfaces = np.repeat(np.linspace(0, mesh.geo_l, n_regions+1)[1:-1, None],
                                   geometry_controls, axis=1)
        self.interfaces = np.asarray(interfaces, dtype=float).reshape(expected).copy()
        self._validate_geometry(self.interfaces)
        self.n_branch = mesh.n_nod if branch_kind == 'q4' else 1
        self.n_material = n_regions*self.n_branch
        self.n_parameters = self.n_material + (self.interfaces.size if self.optimize_geometry else 0)
        N = gauss_shape_values(mesh)
        self.gauss_shape = mesh.gauss_det_j.shape
        self.xy = np.column_stack([np.einsum('en,gen->ge', mesh.x_ele, N).ravel(),
                                   np.einsum('en,gen->ge', mesh.y_ele, N).ravel()])
        self.basis = self._basis(self.xy)
        self.geometry_basis = self._geometry_basis(self.xy[:, 1])
        self.weights = (mesh.gauss_abs_det_j*mesh.gauss_w[:, None]).ravel()
        if sharp:
            cuts = getattr(mesh, 'quadrature_metadata', {}).get('fixed_x_breaks', [])
            if len(cuts) != len(self.interfaces) or not np.allclose(cuts, self.interfaces[:, 0]):
                raise ValueError('Sharp fixed regions require the same x_breaks in quadrature')

    def _validate_geometry(self, g):
        if (not np.isfinite(g).all() or np.any(g <= 0) or np.any(g >= self.mesh.geo_l)
                or np.any(np.diff(g, axis=0) <= 0)):
            raise ValueError('Interface controls must be ordered, noncrossing and inside the domain')

    def _geometry_basis(self, y):
        return BSpline.design_matrix(np.asarray(y), self.knots, self.degree).toarray()

    def _basis(self, xy):
        if self.branch_kind == 'q4':
            return q4_basis(self.mesh, xy)
        n = len(xy)
        return csr_matrix(np.ones((n, 1))), csr_matrix((n, 1)), csr_matrix((n, 1))

    def initial_parameters(self, uniform_modulus, interfaces=None):
        if not np.isfinite(uniform_modulus) or uniform_modulus <= 0:
            raise ValueError('Uniform starting modulus must be positive and finite')
        m = np.full(self.n_material, np.log(uniform_modulus))
        g = self.interfaces if interfaces is None else np.asarray(interfaces).reshape(self.interfaces.shape)
        self._validate_geometry(g)
        return np.r_[m, g.ravel()] if self.optimize_geometry else m

    def unpack(self, theta):
        theta = np.asarray(theta, dtype=float)
        if theta.shape != (self.n_parameters,) or not np.isfinite(theta).all():
            raise ValueError(f'Expected {self.n_parameters} finite material/geometry parameters')
        c = np.exp(theta[:self.n_material].reshape(self.n_regions, self.n_branch))
        if not np.isfinite(c).all() or np.any(c <= 0):
            raise ValueError('exp(m) must be finite and positive')
        g = (theta[self.n_material:].reshape(self.interfaces.shape)
             if self.optimize_geometry else self.interfaces)
        self._validate_geometry(g)
        return c, g

    def _state(self, theta, basis=None, geometry_basis=None, xy=None):
        c, g = self.unpack(theta)
        N, dx, dy = self.basis if basis is None else basis
        bg = self.geometry_basis if geometry_basis is None else geometry_basis
        xy = self.xy if xy is None else xy
        branches = np.asarray(N @ c.T)
        phi = xy[:, 0, None] - bg @ g.T
        if self.sharp:
            H = (phi >= 0).astype(float)
            hp = np.zeros_like(H)
        else:
            tanh = np.tanh(phi/self.epsilon)
            H, hp = (1+tanh)/2, (1-tanh*tanh)/(2*self.epsilon)
        switches = np.column_stack([np.ones(len(xy)), H, np.zeros(len(xy))])
        chi = switches[:, :-1] - switches[:, 1:]
        return c, branches, chi, hp, bg, N, dx, dy

    def values(self, theta):
        _, branches, chi, *_ = self._state(theta)
        return np.sum(chi*branches, axis=1).reshape(self.gauss_shape)

    def sample(self, theta, xy):
        xy = np.asarray(xy, dtype=float)
        _, branches, chi, *_ = self._state(theta, self._basis(xy),
                                          self._geometry_basis(xy[:, 1]), xy)
        return np.sum(chi*branches, axis=1)

    def pullback(self, theta, sensitivity):
        """Apply material Jacobian transpose to weighted dJ/dE_q; no dense Jacobian."""
        sensitivity = np.asarray(sensitivity, dtype=float)
        if sensitivity.shape != self.gauss_shape or not np.isfinite(sensitivity).all():
            raise ValueError('Sensitivity must match finite (quadrature, elements) values')
        c, branches, chi, hp, bg, N, *_ = self._state(theta)
        v = sensitivity.ravel()
        grad_m = np.asarray(N.T @ (v[:, None]*chi)).T*c
        grad_g = (v[:, None]*(branches[:, :-1]-branches[:, 1:])*hp).T @ bg
        return np.r_[grad_m.ravel(), grad_g.ravel()] if self.optimize_geometry else grad_m.ravel()

    def regularization(self, theta):
        """R=sum_i integral chi_i |grad E_i|² and its FULL parameter derivative.

        At finite epsilon this is a smooth-region approximation to the sharp
        intra-region integral. No derivative of chi is included in grad E_i;
        geometry derivatives of the integration weights ARE included in dR/dg.
        """
        c, _, chi, hp, bg, _, dx, dy = self._state(theta)
        gx, gy = np.asarray(dx @ c.T), np.asarray(dy @ c.T)
        energy = gx*gx + gy*gy
        value = float(self.weights @ np.sum(chi*energy, axis=1))
        wc = self.weights[:, None]*chi
        grad_m = 2*np.asarray(dx.T @ (wc*gx) + dy.T @ (wc*gy)).T*c
        grad_g = (self.weights[:, None]*(energy[:, :-1]-energy[:, 1:])*hp).T @ bg
        gradient = (np.r_[grad_m.ravel(), grad_g.ravel()] if self.optimize_geometry
                    else grad_m.ravel())
        return value, gradient


class MappedMaterial(MaterialInfo):
    """Adapter into the shared FEM core, without constructing global nodal E."""

    def __init__(self, nu, parameter_map, theta):
        super().__init__(nu)
        self.parameter_map, self.theta = parameter_map, np.asarray(theta)

    def gauss_modulus(self, mesh):
        if mesh is not self.parameter_map.mesh:
            raise ValueError('Material map and FEM must use the same integration mesh')
        return self.parameter_map.values(self.theta)
