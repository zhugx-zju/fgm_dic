import unittest

import numpy as np

from lib.dic_grid_orientation import orient_grid_field


class DICGridOrientationTests(unittest.TestCase):
    def test_rot180_transforms_displacement_components(self):
        x_grid = np.array([[0.0, 1.0], [0.0, 1.0]])
        y_grid = np.array([[0.0, 0.0], [1.0, 1.0]])
        ux_grid = np.full((2, 2), 2.0)
        uy_grid = np.full((2, 2), -3.0)

        oriented = orient_grid_field(
            x_grid=x_grid,
            y_grid=y_grid,
            ux_grid=ux_grid,
            uy_grid=uy_grid,
            operations=["rot180"],
        )

        np.testing.assert_allclose(oriented["UX_grid"], -np.rot90(ux_grid, 2))
        np.testing.assert_allclose(oriented["UY_grid"], -np.rot90(uy_grid, 2))

    def test_transpose_swaps_displacement_components(self):
        x_grid = np.array([[0.0, 1.0], [0.0, 1.0]])
        y_grid = np.array([[0.0, 0.0], [1.0, 1.0]])
        ux_grid = np.array([[1.0, 2.0], [3.0, 4.0]])
        uy_grid = np.array([[5.0, 6.0], [7.0, 8.0]])

        oriented = orient_grid_field(
            x_grid=x_grid,
            y_grid=y_grid,
            ux_grid=ux_grid,
            uy_grid=uy_grid,
            operations=["transpose"],
        )

        np.testing.assert_allclose(oriented["UX_grid"], uy_grid.T)
        np.testing.assert_allclose(oriented["UY_grid"], ux_grid.T)


if __name__ == "__main__":
    unittest.main()
