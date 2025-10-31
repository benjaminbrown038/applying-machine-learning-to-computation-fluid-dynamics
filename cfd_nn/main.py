import argparse
from config import Config
from fd_solver import solve_laplace_fd
from data import build_dataset
from train import train_nn
from viz import plot_and_compare
from utils import eval_global

def main():
    p = argparse.ArgumentParser()
    for k,v in Config().__dict__.items(): p.add_argument(f"--{k}", type=type(v), default=v)
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
