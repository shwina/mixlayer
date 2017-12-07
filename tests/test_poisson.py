import numpy as np
from numpy.testing import assert_allclose
from mixlayer.poisson import PoissonSolver

def test_poisson_3by3():
    A = np.array([
            [-1,  0,  0,  0,  0,  0,  0,  0,  0],
            [ 0, -1,  0,  0,  0,  0,  0,  0,  0],
            [ 0,  0, -1,  0,  0,  0,  0,  0,  0],
            [ 1,  0,  0, -4,  1,  1,  1,  0,  0],
            [ 0,  1,  0,  1, -4,  1,  0,  1,  0],
            [ 0,  0,  1,  1,  1, -4,  0,  0,  1],
            [ 0,  0,  0,  0,  0,  0, -1,  0,  0],
            [ 0,  0,  0,  0,  0,  0,  0, -1,  0],
            [ 0,  0,  0,  0,  0,  0,  0,  0, -1]],
            dtype=np.float64 )
    b = np.random.rand(3, 3)
    b[[0, -1], :] = 0
    ps = PoissonSolver(3, 1, 1)
    true = np.linalg.solve(A, b.ravel()).reshape([3, 3])
    print(ps.A.todense())
    got = np.zeros([3, 3])
    ps.solve(b, got)
    print(got)
    print(true)
    assert_allclose(got, true)
