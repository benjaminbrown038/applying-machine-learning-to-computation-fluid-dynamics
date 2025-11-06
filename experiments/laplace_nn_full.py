# ========================================================================
# Laplace Equation Solver + Neural Network Surrogate Model
# Combines FD solver, dataset builder, NN training, and visualization
# Trey Brown — 2025
# ========================================================================

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from dataclasses import dataclass
import pandas as pd

# ------------------------------------------------------------------------
# Boundary Condition
# ------------------------------------------------------------------------
def boundary_u(x, y, z, face: str):
    if face == 'x0': return np.zeros_like(y)
    if face == 'x1': return np.ones_like(y)
    if face == 'y0': return np.zeros_like(x)
    if face == 'y1': return np.zeros_like(x)
    if face == 'z0': return np.zeros_like(x)
    if face == 'z1': return np.zeros_like(x)
    raise ValueError(f"Unknown face {face}")

# ------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------
@dataclass
class Config:
    nx: int = 30
    ny: int = 30
    nz: int = 30
    max_iters: int = 2000
    tol: float = 1e-5

    hidden: int = 128
    depth: int = 3
    lr: float = 1e-3
    epochs: int = 3000
    batch_size: int = 8192
    train_frac: float = 0.9
    seed: int = 42

    outdir: str = "outputs"

# ------------------------------------------------------------------------
# Finite Difference Laplace Solver
# ------------------------------------------------------------------------
def solve_laplace_fd(cfg: Config):
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

# ------------------------------------------------------------------------
# Dataset Builder
# ------------------------------------------------------------------------
def build_dataset(u, X, Y, Z, frac, seed):
    coords = np.stack([X.reshape(-1), Y.reshape(-1), Z.reshape(-1)], axis=1).astype(np.float32)
    vals = u.reshape(-1).astype(np.float32)[:, None]
    rng = np.random.default_rng(seed)
    idx = np.arange(coords.shape[0]); rng.shuffle(idx)
    n = int(frac * len(idx))
    ti, vi = idx[:n], idx[n:]
    return (
        torch.from_numpy(coords[ti]),
        torch.from_numpy(vals[ti]),
        torch.from_numpy(coords[vi]),
        torch.from_numpy(vals[vi]),
    )

# ------------------------------------------------------------------------
# Neural Network Model
# ------------------------------------------------------------------------
class SigmoidMLP(nn.Module):
    def __init__(self, in_dim, hidden, depth, out_dim=1):
        super().__init__()
        layers = []
        last = in_dim
        for _ in range(depth):
            layers += [nn.Linear(last, hidden), nn.Sigmoid()]
            last = hidden
        layers.append(nn.Linear(last, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# ------------------------------------------------------------------------
# Training Loop
# ------------------------------------------------------------------------
def train_nn(cfg, Xtr, ytr, Xv, yv):
    torch.manual_seed(cfg.seed)
    model = SigmoidMLP(3, cfg.hidden, cfg.depth)
    opt = optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.MSELoss()
    ds = torch.utils.data.TensorDataset(Xtr, ytr)
    dl = torch.utils.data.DataLoader(ds, batch_size=cfg.batch_size, shuffle=True)

    best = float('inf'); best_state = None
    for epoch in range(1, cfg.epochs+1):
        model.train(); total = 0
        for xb, yb in dl:
            opt.zero_grad(True)
            loss = loss_fn(model(xb), yb)
            loss.backward(); opt.step()
            total += loss.item() * xb.size(0)
        tl = total / len(ds)

        model.eval()
        with torch.no_grad():
            vl = loss_fn(model(Xv), yv).item()

        if vl < best:
            best = vl
            best_state = {k:v.cpu().clone() for k,v in model.state_dict().items()}

        if epoch % 200 == 0 or epoch == 1:
            print(f"Epoch {epoch:4d} | train {tl:.3e} | val {vl:.3e}")

    model.load_state_dict(best_state)
    print(f"Best val MSE {best:.3e}")
    return model

# ------------------------------------------------------------------------
# Global Error Evaluation
# ------------------------------------------------------------------------
def eval_global(cfg, u_fd, model, X, Y, Z):
    coords = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1).astype(np.float32)
    step = max(1, len(coords)//200000)
    c = torch.from_numpy(coords[::step])
    fd = u_fd.ravel()[::step]
    nn_pred = model(c).detach().numpy().ravel()
    mse = float(np.mean((nn_pred - fd)**2))
    mae = float(np.mean(np.abs(nn_pred - fd)))
    print(f"MSE={mse:.3e}, MAE={mae:.3e}")

# ------------------------------------------------------------------------
# Visualization
# ------------------------------------------------------------------------
def ensure_outdir(path): os.makedirs(path, exist_ok=True)

def plot_slice(title, arr, path):
    plt.figure()
    plt.imshow(arr, origin="lower", interpolation="nearest")
    plt.title(title); plt.colorbar(); plt.tight_layout()
    plt.savefig(path, dpi=150); plt.close()

def plot_and_compare(cfg, u_fd, model, X, Y, Z):
    ensure_outdir(cfg.outdir)
    ix = cfg.nx // 2
    x0, y0, z0 = X[ix], Y[ix], Z[ix]
    fd = u_fd[ix]

    coords = np.stack([x0.ravel(), y0.ravel(), z0.ravel()], axis=1).astype(np.float32)
    with torch.no_grad():
        nn_pred = model(torch.from_numpy(coords)).numpy().reshape(x0.shape)
    err = np.abs(nn_pred - fd)

    plot_slice("FD", fd, f"{cfg.outdir}/fd.png")
    plot_slice("NN", nn_pred, f"{cfg.outdir}/nn.png")
    plot_slice("Err", err, f"{cfg.outdir}/err.png")


def plot_training_log(cfg):
    log_path = os.path.join(cfg.outdir, "training_log.csv")
    if not os.path.exists(log_path):
        print("No training log found.")
        return

    data = pd.read_csv(log_path)
    plt.figure()
    plt.plot(data["epoch"], data["train_loss"], label="Train Loss")
    plt.plot(data["epoch"], data["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.outdir, "training_curve.png"), dpi=150)
    plt.close()
    print(f"Saved training curve to {cfg.outdir}/training_curve.png")
# ------------------------------------------------------------------------
# Main Script
# ------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    for k,v in Config().__dict__.items(): 
        p.add_argument(f"--{k}", type=type(v), default=v)
    args = p.parse_args()
    cfg = Config(**vars(args))

    print("=== FD Solve ===")
    u,(X,Y,Z) = solve_laplace_fd(cfg)

    print("=== Dataset ===")
    Xtr,ytr,Xv,yv = build_dataset(u,X,Y,Z,cfg.train_frac,cfg.seed)

    print("=== Train NN ===")
    model = train_nn(cfg, Xtr,ytr,Xv,yv)

    print("=== Plot ===")
    plot_and_compare(cfg, u, model, X,Y,Z)

    print("=== Global Error ===")
    eval_global(cfg, u, model, X,Y,Z)

if __name__ == "__main__":
    main()
