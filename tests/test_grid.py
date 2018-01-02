from mixlayer.grid.mapped import SinhGrid
from numpy.testing import assert_equal, assert_allclose

import numpy as np

def test_sinh_grid():
    g = SinhGrid((32, 32), (1, 1), (1, 1), 5)
    x = np.linspace(-0.5, 0.5, 32)

    y = np.sinh(5*x)
    y = y/(y.max()) * 0.5

    assert_allclose(y, g.y[:, 0])

    g = SinhGrid(32, 1, (1, 1), 5)

    assert_allclose(y, g.y[:, 0])

def test_no_stretching():
    g = SinhGrid((32, 32), (1, 1), (1, 1), 1e-5)

    y = np.linspace(-0.5, 0.5, 32)

    assert_allclose(g.y[:, 0], y)

    g = SinhGrid(32, 1, (1, 1), 1e-5)

    assert_allclose(y, g.y[:, 0])
