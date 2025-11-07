import numpy as np
from laplace_baseline import (
    build_dataset, MLP, train_model, plot_slice
)
import os

# -----------------------------
# 1. PHYSICS: TIME-DEPENDENT HEAT EQ
# -----------------------------
def solve_physics(cfg):
    nx, ny, nz = cfg["nx"], cfg["ny"], cfg["nz"]
    dt, steps = cfg["dt"], cfg["steps"]
    alpha = 1.0

    u = np.zeros((nx,ny,nz))
    u[0,:,:]=1.0

    for _ in range(steps):
        u_old = u.copy()
        lap = (
            u_old[:-2,1:-1,1:-1] +
            u_old[2:,1:-1,1:-1] +
            u_old[1:-1,:-2,1:-1] +
            u_old[1:-1,2:,1:-1] +
            u_old[1:-1,1:-1,:-2] +
            u_old[1:-1,1:-1,2:] -
            6*u_old[1:-1,1:-1,1:-1]
        )
        u[1:-1,1:-1,1:-1] += alpha * dt * lap

    return u


# -----------------------------
# MAIN
# -----------------------------
def main():
    cfg={
        "nx":30,"ny":30,"nz":30,
        "steps":500,"dt":1e-3,
        "hidden":128,"depth":3,
        "lr":1e-3,"epochs":600,
        "outdir":"experiments/results/heat_time"
    }

    os.makedirs(cfg["outdir"], exist_ok=True)

    u = solve_physics(cfg)
    pts, vals = build_dataset(u)

    model = MLP(cfg["hidden"], cfg["depth"])
    train_model(model, pts, vals, cfg)

    plot_slice(u, cfg["outdir"])

if __name__=="__main__":
    main()
