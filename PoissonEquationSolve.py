import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt

def idx(i, j, Nx):
    return i + j * Nx

def solve_poisson_2d(a, b, c, d, Nx, Ny, f_fn, u_exact):
    dx = (b - a) / Nx
    dy = (d - c) / Ny
    x = np.linspace(a, b, Nx + 1)
    y = np.linspace(c, d, Ny + 1)
    nx, ny = Nx + 1, Ny + 1
    X, Y = np.meshgrid(x, y, indexing='ij')

    Dx = sp.diags([-1, 2, -1], [-1, 0, 1], shape=(nx, nx)) / dx**2
    Dy = sp.diags([-1, 2, -1], [-1, 0, 1], shape=(ny, ny)) / dy**2

    I_x = sp.eye(nx)
    I_y = sp.eye(ny)
    L = sp.kron(Dx, I_y) + sp.kron(I_x, Dy)

    F = f_fn(X, Y).flatten()

    for i in range(nx):
        for j in range(ny):
            k = idx(i, j, nx)
            if i == 0 or i == nx - 1 or j == 0 or j == ny - 1:
                L[k,k] = 1.0
                F[k] = u_exact(x[i], y[j])
    phi = spla.spsolve(L, F)
    return phi.reshape((nx, ny))

a, b = 0.0, 5.0
c, d = 0.0, 5.0
u_fn = lambda x, y: np.sin(x) * np.cos(y)
f_fn = lambda x, y: -2.0 * np.sin(x) * np.cos(y)

errInf, errL2 = [], []

NxVals = np.array([100, 150, 200, 250, 300])
NyVals = np.array([100, 150, 200, 250, 300])
for Nx, Ny in zip(NxVals, NyVals):
    phi = solve_poisson_2d(a, b, c, d, Nx, Ny, f_fn, u_fn)
    x = np.linspace(a, b, Nx + 1)
    y = np.linspace(c, d, Ny + 1)
    X, Y = np.meshgrid(x, y, indexing='ij')
    u_exact = u_fn(X, Y)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(projection='3d')
scatter = ax.scatter(X, Y, u_exact, c=u_exact, cmap='viridis', marker='o')
ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
ax.set_zlabel('Z Axis')
ax.set_title('Matplotlib 3D Scatter Plot')
fig.colorbar(scatter, ax=ax, label='Z depth')
