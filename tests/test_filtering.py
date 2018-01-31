from mixlayer.filtering import filter5
from numpy.testing import assert_equal, assert_allclose

import numpy as np

def test_filter_zero_array():
    a = np.zeros([10, 10], dtype=np.float64)
    filter5(a)
    assert_equal(a, 0)
    
def test_filter_ones_array():
    a = np.ones([10, 10], dtype=np.float64)
    filter5(a)
    assert_equal(a, 1)

def test_filter_symmetric_array():
    # symmetric about vertical line
    a = np.random.rand(10, 5)
    a = np.hstack([a, a[:, ::-1]])
    filter5(a)
    assert_equal(a[:, :5], a[:, -1:-6:-1])

    # symmetric about horizontal line
    a = np.random.rand(10, 5)
    a = np.vstack([a, a[::-1, :]])
    filter5(a)
    assert_equal(a[:5, :], a[-1:-6:-1, :])
