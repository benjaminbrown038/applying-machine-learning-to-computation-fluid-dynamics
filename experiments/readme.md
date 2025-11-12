 # Experiments 🧪

This folder contains fully self-contained experiment scripts.
Each experiment trains a neural surrogate model on a different physics solver, while keeping the rest of the pipeline identical.

## Shared Architecture 🔁

✅ Only the physics solver changes

✅ All other logic — dataset builder, MLP model, training loop, and plotting — is shared

✅ Each experiment runs standalone (no src/ imports required)

experiments/
│
├── laplace_baseline.py      # Laplace equation (steady-state heat)
├── poisson_surrogate.py     # Poisson equation (with source term)
└── heat_time.py             # Time-dependent heat diffusion

## Code Structure 🧩 

Each script follows the same modular layout:

1. PHYSICS SOLVER — unique PDE implementation

2. Dataset Builder — shared mesh-to-tensor converter

3. MLP Model — shared neural surrogate

4. Training Loop — shared optimizer & loss logic

5. Visualization — shared plotting utilities

6. Main Entrypoint — runs the experiment end-to-end

### Running an Experiment ▶️ 

Each file is a standalone program that can be executed directly:

```python3 experiments/laplace_baseline.py```

```python3 experiments/poisson_surrogate.py```

```python3 experiments/heat_time.py```


| Experiment            | PDE Type  | Equation                    | Physical Meaning                            |
| --------------------- | --------- | --------------------------- | ------------------------------------------- |
| **Laplace Baseline**  | Laplace   | ( \nabla^2 u = 0 )          | Steady-state heat or potential (no sources) |
| **Poisson Surrogate** | Poisson   | ( \nabla^2 u = f )          | Heat or potential with internal generation  |
| **Heat-Time**         | Diffusion | ( u_t = \alpha \nabla^2 u ) | Transient (time-dependent) heat flow        |
