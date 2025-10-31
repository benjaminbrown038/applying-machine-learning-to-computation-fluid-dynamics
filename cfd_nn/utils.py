import numpy as np
import torch

def eval_global(cfg, u_fd, model, X, Y, Z):
    coords = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1).astype(np.float32)
    step = max(1, len(coords)//200000)
    c = torch.from_numpy(coords[::step])
    fd = u_fd.ravel()[::step]
    nn = model(c).detach().numpy().ravel()
    mse = float(np.mean((nn - fd)**2))
    mae = float(np.mean(abs(nn - fd)))
    print(f"MSE={mse:.3e}, MAE={mae:.3e}")
