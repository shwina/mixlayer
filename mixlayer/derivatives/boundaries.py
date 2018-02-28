import enum
import enum

class BoundaryConditionType(enum.Enum):
    PERIODIC = 1
    INNER = 2
    NEUMANN = 3
