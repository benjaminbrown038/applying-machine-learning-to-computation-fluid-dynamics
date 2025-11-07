import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os

# -----------------------------
# 1. PHYSICS: POISSON
# -----------------------------
def solve_physics(cfg):
    nx, ny, nz = cfg["nx"], cfg["ny"], cfg["nz"]
    tol, max_iters = cfg["tol"], cfg["max_iters"]

    x = np.linspace(-1,1,nx)
    y = np.linspace(-1,1,ny)
    z = np.linspace(-1,1,nz)
    X,Y,Z = np.meshgrid(x,y,z,indexing="ij")

    f = np.exp(-(X**2+Y**2+Z**2)*8)

    u = np.zeros((nx,ny,nz))

    for _ in range(max_iters):
        u_old = u.copy()
        u[1:-1,1:-1,1:-1] = (
            u_old[:-2,1:-1,1:-1] +
            u_old[2:,1:-1,1:-1] +
            u_old[1:-1,:-2,1:-1] +
            u_old[1:-1,2:,1:-1] +
            u_old[1:-1,1:-1,:-2] +
            u_old[1:-1,1:-1,2:] -
            f[1:-1,1:-1,1:-1]
        ) / 6.0
        if np.linalg.norm(u - u_old) < tol: break
    return u


# -------------- SHARED SECTIONS (unchanged) ---------------
# dataset, MLP, training, plotting, main (same as previous)
# ----------------------------------------------------------

from laplace_baseline import build_dataset, MLP, train_model, plot_slice

def main():
    cfg={
        "nx":40,"ny":40,"nz":40,
        "tol":1e-6,"max_iters":3000,
        "hidden":256,"depth":4,
        "lr":8e-4,"epochs":800,
        "outdir":"experiments/results/poisson"
    }

    os.makedirs(cfg["outdir"], exist_ok=True)

    u = solve_physics(cfg)
    pts, vals = build_dataset(u)

    model = MLP(cfg["hidden"], cfg["depth"])
    train_model(model, pts, vals, cfg)

    plot_slice(u, cfg["outdir"])

if __name__=="__main__":
    main()
