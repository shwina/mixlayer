from mixlayer.timestepping import RK4
from mixlayer.equation import Equation
from numpy.testing import assert_equal, assert_allclose
import numpy as np

class TestRK4:

    def setup(self):

        self.f1 = np.ones(10, dtype=np.float64)
        self.f2 = np.ones(10, dtype=np.float64)

        def rhs_1(f, out=None):
            out[...] = np.ones_like(f)
        def rhs_2(f, out=None):
            out[...] = np.ones_like(f)

        equations = [Equation(self.f1), Equation(self.f2)]
        equations[0].set_rhs_func(rhs_1, self.f1)
        equations[1].set_rhs_func(rhs_2, self.f2)

        self.stepper = RK4(equations)

    def test_take_step_zero_length(self):
        self.stepper.step(0)
        assert_allclose(self.f1, 1)
        assert_allclose(self.f2, 1)

    def test_take_one_step_and_compare_fields(self):
        self.stepper.step(0.5)
        assert_allclose(self.f1, self.f2)
