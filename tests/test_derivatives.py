import numpy as np
from numpy.testing import assert_equal, assert_allclose
from mixlayer.derivatives import dfdx, dfdy, BoundaryType

def test_dfdx():
    x, y = np.meshgrid(np.linspace(0, 2*np.pi),
                       np.linspace(0, 2*np.pi))
    dx = x[0,1] - x[0,0]
    
    z = x**2

    dzdx = dfdx(z, dx, boundary_type=BoundaryType.INNER)
    assert_allclose(dzdx, 2*x)

    x = x[:, :-1]
    z = np.sin(x)

    dzdx = dfdx(z, dx, boundary_type=BoundaryType.PERIODIC)
    assert_allclose(dzdx, np.cos(x))

def test_dfdy():
    x, y = np.meshgrid(np.linspace(0, 2*np.pi),
                       np.linspace(0, 2*np.pi))
    dy = y[1,0] - y[0,0]
    
    z = y**2

    dzdy = dfdy(z, dy, boundary_type=BoundaryType.INNER)
    assert_allclose(dzdy, 2*y)

    y = y[:-1, :]
    z = np.sin(y)

    dzdy = dfdy(z, dy, boundary_type=BoundaryType.PERIODIC)
    assert_allclose(dzdy, np.cos(y))

