import unittest

import numpy as np

from fgm_asm.inverse_solver import _raw_to_ehat, _transform_gradient_to_raw


class TestBoundedMeanUnitEhatMapping(unittest.TestCase):
    def test_bounded_mapping_keeps_unit_mean(self):
        raw = np.array([-4.0, -1.2, 0.0, 0.5, 2.0, 5.0])

        ehat = _raw_to_ehat(raw, ehat_min=0.05, ehat_max=5.0)

        self.assertGreaterEqual(ehat.min(), 0.05)
        self.assertLessEqual(ehat.max(), 5.0)
        self.assertAlmostEqual(float(ehat.mean()), 1.0, places=12)

    def test_bounded_gradient_matches_finite_difference(self):
        raw = np.array([-2.0, -0.6, 0.1, 0.9, 2.4])
        grad_ehat = np.array([1.5, -0.3, 0.7, -1.2, 0.4])

        grad_raw = _transform_gradient_to_raw(raw, grad_ehat, ehat_min=0.05, ehat_max=5.0)

        eps = 1e-6
        fd_grad = np.zeros_like(raw)
        for i in range(raw.size):
            step = np.zeros_like(raw)
            step[i] = eps
            plus = _raw_to_ehat(raw + step, ehat_min=0.05, ehat_max=5.0)
            minus = _raw_to_ehat(raw - step, ehat_min=0.05, ehat_max=5.0)
            fd_grad[i] = float(grad_ehat @ (plus - minus) / (2.0 * eps))

        np.testing.assert_allclose(grad_raw, fd_grad, rtol=1e-5, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
