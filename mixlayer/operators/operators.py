from mixlayer.derivatives import derivatives
from mixlayer.boundaries import BoundaryConditionType

class Operators:
    def __init__(self, grid, boundary_conditions):
        self.grid = grid
        self.boundary_conditions = boundary_conditions

    def dfdx(self, f):
        return derivatives.dfdx(
            f, self.grid.dx,
            self.boundary_conditions[0].bc_type,
            self.boundary_conditions[0].bc_val,
            self.boundary_conditions[1].bc_type,
            self.boundary_conditions[1].bc_val)

    def dfdy(self, f):
        return derivatives.dfdy(
            f, self.grid.dn,
            self.boundary_conditions[2].bc_type,
            self.boundary_conditions[2].bc_val,
            self.boundary_conditions[3].bc_type,
            self.boundary_conditions[3].bc_val
            ) * self.grid.dndy

    def divergence(self, u, v):
        return self.dfdx(u) + self.dfdy(v)
    
    def laplacian(self, f):
        return (derivatives.dfdx(self.dfdx(f),
                                 self.grid.dx,
                                 BoundaryConditionType.INNER, 0,
                                 BoundaryConditionType.INNER, 0)
              + derivatives.dfdy(self.dfdy(f),
                                 self.grid.dn,
                                 BoundaryConditionType.INNER, 0,
                                 BoundaryConditionType.INNER, 0) * self.grid.dndy
              )
