import numpy as np
from typing import Tuple
from config import Config
from boundary import boundary_u

def solve_laplace_fd(cfg: Config) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    nx, ny, nz = cfg.nx, cfg.ny, cfg.nz
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    z = np.linspace(0, 1, nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    u = np.zeros((nx, ny, nz))
    u[0,:,:]   = boundary_u(Y[0],Z[0],Z[0],'x0')
    u[-1,:,:]  = boundary_u(Y[-1],Z[-1],Z[-1],'x1')
    u[:,0,:]   = boundary_u(X[:,0],Z[:,0],Z[:,0],'y0')
    u[:,-1,:]  = boundary_u(X[:,-1],Z[:,-1],Z[:,-1],'y1')
    u[:,:,0]   = boundary_u(X[:,:,0],Y[:,:,0],Z[:,:,0],'z0')
    u[:,:,-1]  = boundary_u(X[:,:,-1],Y[:,:,-1],Z[:,:,-1],'z1')

    u_old = u.copy()

    for it in range(cfg.max_iters):
        u[1:-1,1:-1,1:-1] = (
            u_old[:-2,1:-1,1:-1] + u_old[2:,1:-1,1:-1] +
            u_old[1:-1,:-2,1:-1] + u_old[1:-1,2:,1:-1] +
            u_old[1:-1,1:-1,:-2] + u_old[1:-1,1:-1,2:]
        ) / 6.0

        diff = np.max(np.abs(u - u_old))
        if it % 50 == 0:
            print(f"Jacobi {it:4d} | max Δu = {diff:.3e}")
        if diff < cfg.tol:
            print(f"Converged at {it}, Δu={diff:.3e}")
            break

        u_old, u = u, u_old

    if (it + 1) % 2 == 1:
        u = u_old
    return u, (X, Y, Z)
