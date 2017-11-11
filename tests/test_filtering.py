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
