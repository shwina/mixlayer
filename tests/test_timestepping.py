from mixlayer.timestepping import RK4
from numpy.testing import assert_equal, assert_allclose
import numpy as np

class TestRK4:

    def setup(self):
        fields = np.ones([4, 10, 10], dtype=np.float64)

        def rhs(out=None):
            out[...] = np.ones([4, 10, 10], dtype=np.float64)

        self.fields = fields
        self.stepper = RK4(fields, rhs)

    def test_take_step_zero_length(self):
        self.stepper.step(0)
        assert_allclose(self.fields[0], 1)
        assert_allclose(self.fields[1], 1)

    def test_take_one_step_and_compare_fields(self):
        self.stepper.step(0.5)
        assert_allclose(self.fields[0], self.fields[1])
