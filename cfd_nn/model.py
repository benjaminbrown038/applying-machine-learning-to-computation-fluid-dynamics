import torch.nn as nn

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
