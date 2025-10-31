import numpy as np
import torch

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
