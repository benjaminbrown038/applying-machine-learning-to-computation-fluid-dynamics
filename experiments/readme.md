✅ Experiments

This folder contains fully self-contained experiment scripts.
Each experiment trains a neural surrogate model on a different physics solver, while keeping the rest of the pipeline identical.

✅ Only the physics portion changes
✅ All other logic (dataset builder, MLP model, training loop, plotting) is shared
✅ Experiments are fully runnable standalone (no src/ imports)

```
experiments/
│
├── laplace_baseline.py      # Laplace equatio(steady heat)
├── poisson_surrogate.py     # Poisson equation (source term)
└── heat_time.py             # Time-dependent heat diffusion
```
Each script follows this exact pattern:
1. PHYSICS SOLVER (unique)
2. Dataset builder (shared)
3. MLP model (shared)
4. Training loop (shared)
5. Visualization (shared)
6. Main entrypoint

▶️ Running an Experiment

Each file is a complete program:
```
python experiments/laplace_baseline.py
```

```
python experiments/poisson_surrogate.py
```
```
python experiments/heat_time.py
```

flowchart TB
    A[Steady-State Heat Block] --> B{No Internal Sources}
    B --> C["nabla^2 u = 0"]
    C --> D[Interior Temperature Field]

    subgraph Boundaries
        E["Hot Face: u = 1"]
        F["Other Faces: u = 0"]
    end

    E --> C
    F --> C





| Experiment        | PDE       | Equation                    | Physical Meaning                           |
| ----------------- | --------- | --------------------------- | ------------------------------------------ |
| Laplace Baseline  | Laplace   | ( \nabla^2 u = 0 )          | Steady-state heat or potential, no sources |
| Poisson Surrogate | Poisson   | ( \nabla^2 u = f )          | Heat or potential with internal generation |
| Heat-Time         | Diffusion | ( u_t = \alpha \nabla^2 u ) | Transient heat flow                        |
