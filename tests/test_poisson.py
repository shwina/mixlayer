import numpy as np

from mixlayer.poisson import PoissonSolver

N = 32
dx = 1./(N-1)
dy = 2./(N-1)
rhs = np.random.rand(N*N)
solution = np.zeros_like(rhs)
solver = PoissonSolver(N, dx, dy)

A = np.zeros([N*N, N*N])
for i in range(N):
    for j in range(N):
        diag = 0.
        ix = i*N + j
        if j > 0:
            left = 1./(dx**2)
            diag -= 1./(dx**2)
            A[ix, ix-1] = left
        if j < N-1:
            right = 1./(dx**2)
            diag -= 1./(dx**2)
            A[ix, ix+1] = right
        if i > 0:
            bottom = 1./(dy**2)
            diag -= 1./(dy**2)
            A[ix, ix-N] = bottom
        if i < N-1:
            top = 1./(dy**2)
            diag -= 1./(dy**2)
            A[ix, ix+N] = top
        A[ix, ix] = diag+1e-5/(dx**2 + dy**2)

true_solution = np.linalg.solve(A, rhs)
solver.solve(rhs, solution)
print(solution - solution.mean())
print(true_solution - true_solution.mean())
