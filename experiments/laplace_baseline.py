import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os

# -----------------------------
# 1. PHYSICS: LAPLACE
# -----------------------------
def solve_physics(cfg):
    nx, ny, nz = cfg["nx"], cfg["ny"], cfg["nz"]
    tol, max_iters = cfg["tol"], cfg["max_iters"]

    u = np.zeros((nx,ny,nz))
    u[0,:,:] = 1.0  # boundary

    for _ in range(max_iters):
        u_old = u.copy()
        u[1:-1,1:-1,1:-1] = (
            u_old[:-2,1:-1,1:-1] +
            u_old[2:,1:-1,1:-1] +
            u_old[1:-1,:-2,1:-1] +
            u_old[1:-1,2:,1:-1] +
            u_old[1:-1,1:-1,:-2] +
            u_old[1:-1,1:-1,2:]
        ) / 6.0
        if np.linalg.norm(u - u_old) < tol: break
    return u


# -----------------------------
# 2. SHARED: DATASET
# -----------------------------
def build_dataset(u):
    nx, ny, nz = u.shape
    X,Y,Z = np.meshgrid(
        np.linspace(0,1,nx),
        np.linspace(0,1,ny),
        np.linspace(0,1,nz),
        indexing="ij"
    )
    pts = torch.tensor(np.stack([X,Y,Z],axis=-1).reshape(-1,3),dtype=torch.float32)
    vals = torch.tensor(u.reshape(-1,1),dtype=torch.float32)
    return pts, vals


# -----------------------------
# 3. SHARED: MLP
# -----------------------------
class MLP(nn.Module):
    def __init__(self, hidden=128, depth=3):
        super().__init__()
        layers=[nn.Linear(3,hidden), nn.Tanh()]
        for _ in range(depth-1):
            layers += [nn.Linear(hidden,hidden), nn.Tanh()]
        layers.append(nn.Linear(hidden,1))
        self.net = nn.Sequential(*layers)
    def forward(self,x): return self.net(x)


# -----------------------------
# 4. SHARED: TRAIN
# -----------------------------
def train_model(model, pts, vals, cfg):
    opt = optim.Adam(model.parameters(), lr=cfg["lr"])
    loss_fn = nn.MSELoss()
    for _ in range(cfg["epochs"]):
        opt.zero_grad()
        loss = loss_fn(model(pts), vals)
        loss.backward()
        opt.step()


# -----------------------------
# 5. SHARED: PLOTTING
# -----------------------------
def plot_slice(u, outdir):
    plt.imshow(u[:,:,u.shape[2]//2], cmap="inferno")
    plt.colorbar()
    plt.savefig(os.path.join(outdir,"slice.png"))
    plt.close()


# -----------------------------
# 6. MAIN
# -----------------------------
def main():
    cfg = {
        "nx":30,"ny":30,"nz":30,
        "tol":1e-5,"max_iters":2000,
        "hidden":128,"depth":3,
        "lr":1e-3,"epochs":600,
        "outdir":"experiments/results/laplace"
    }

    os.makedirs(cfg["outdir"], exist_ok=True)

    u = solve_physics(cfg)
    pts, vals = build_dataset(u)

    model = MLP(cfg["hidden"], cfg["depth"])
    train_model(model, pts, vals, cfg)

    plot_slice(u, cfg["outdir"])

if __name__=="__main__":
    main()
