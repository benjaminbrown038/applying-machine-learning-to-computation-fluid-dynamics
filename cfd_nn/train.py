import torch
import torch.optim as optim
import torch.nn as nn
from model import SigmoidMLP

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
