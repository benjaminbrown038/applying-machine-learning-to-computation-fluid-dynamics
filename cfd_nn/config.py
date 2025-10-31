from dataclasses import dataclass

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
