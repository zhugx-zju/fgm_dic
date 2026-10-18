"""Load cases shared by synthetic and experimental displacement observations."""

from dataclasses import dataclass, field

import numpy as np

from .observation import ObservationPackage


@dataclass(frozen=True)
class ObservationCase:
    """One force-controlled state; displacements mm, force N, modulus MPa.

    ``bc_info`` uses the unit-thickness FEM convention (force/thickness).
    Metadata records the displacement reference and the physical force offset.
    """

    name: str
    bc_info: dict
    observation: ObservationPackage
    remove_rigid_body: bool = True
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.bc_info.get('loading_mode') != 'two_end_traction':
            raise ValueError('ObservationCase requires two_end_traction boundary conditions')
        obs = self.observation
        if obs.u_measured.size == 0:
            raise ValueError('ObservationCase needs at least one valid measurement')
        if not all(np.all(np.isfinite(a)) for a in
                   (obs.x_meas, obs.y_meas, obs.u_measured, obs.weight_vector)):
            raise ValueError('Observation coordinates, displacements and weights must be finite')
        if np.any(obs.weight_vector < 0) or not np.any(obs.weight_vector > 0):
            raise ValueError('Observation weights must be nonnegative with positive total')


def evaluate_observation(case, displacement):
    """Eliminate three rigid modes, returning cost and the adjoint RHS.

    The fitted prediction is Hu + R c. The envelope theorem gives
    dJ/du = -H.T W (d-Hu-Rc); no derivative of c is needed. W contains
    quadratic weights, rather than their square roots.
    """
    obs = case.observation
    prediction = np.asarray(obs.H @ displacement).ravel()
    residual = obs.u_measured - prediction
    coefficients = np.zeros(3)
    if case.remove_rigid_body:
        # Centering improves conditioning without changing the rigid subspace.
        x = obs.x_meas - np.mean(obs.x_meas)
        y = obs.y_meas - np.mean(obs.y_meas)
        rigid = np.zeros((residual.size, 3))
        rigid[0::2, 0] = 1.0
        rigid[1::2, 1] = 1.0
        rigid[0::2, 2] = -y
        rigid[1::2, 2] = x
        root_weight = np.sqrt(obs.weight_vector)
        coefficients, _, rank, _ = np.linalg.lstsq(
            root_weight[:, None] * rigid, root_weight * residual, rcond=None,
        )
        if rank < 3:
            raise ValueError('Valid DIC points cannot determine three rigid-body modes')
        prediction = prediction + rigid @ coefficients
        residual = obs.u_measured - prediction
    weighted = obs.weight_vector * residual
    return {
        'name': case.name,
        'prediction': prediction,
        'residual': residual,
        'rigid_coefficients': coefficients,
        'rigid_origin': np.array([np.mean(obs.x_meas), np.mean(obs.y_meas)]),
        'cost_tar': 0.5 * float(residual @ weighted),
        'adjoint_rhs': np.asarray(obs.H.T @ weighted).ravel(),
        'weighted_rms': float(np.sqrt(residual @ weighted / obs.weight_vector.sum())),
    }
