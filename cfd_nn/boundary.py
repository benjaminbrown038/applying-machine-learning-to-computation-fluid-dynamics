import numpy as np

def boundary_u(x, y, z, face: str):
    if face == 'x0': return np.zeros_like(y)
    if face == 'x1': return np.ones_like(y)
    if face == 'y0': return np.zeros_like(x)
    if face == 'y1': return np.zeros_like(x)
    if face == 'z0': return np.zeros_like(x)
    if face == 'z1': return np.zeros_like(x)
    raise ValueError(f"Unknown face {face}")
