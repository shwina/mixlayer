import enum

class BoundaryConditionType(enum.Enum):
    PERIODIC = 1
    INNER = 2
    DIRICHLET = 3
    NEUMANN = 4

class BoundaryCondition:
    def __init__(self, bc_type, bc_val=None):
        if bc_type == 'periodic':
            self.bc_type = BoundaryConditionType.PERIODIC
        if bc_type == 'inner':
            self.bc_type = BoundaryConditionType.INNER
        if bc_type == 'dirichlet':
            self.bc_type = BoundaryConditionType.DIRICHLET
        if bc_type == 'neumann':
            self.bc_type = BoundaryConditionType.NEUMANN
        self.bc_val = bc_val
        if bc_val == None:
            self.bc_val = 0
