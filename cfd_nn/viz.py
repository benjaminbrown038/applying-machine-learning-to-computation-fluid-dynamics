import os
import numpy as np
import matplotlib.pyplot as plt

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
        nn = model(torch.from_numpy(coords)).numpy().reshape(x0.shape)

    err = np.abs(nn - fd)

    plot_slice("FD", fd, f"{cfg.outdir}/fd.png")
    plot_slice("NN", nn, f"{cfg.outdir}/nn.png")
    plot_slice("Err", err, f"{cfg.outdir}/err.png")
