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

.

🔬 Physics Overview

This project evaluates neural network surrogates on several foundational partial differential equations (PDEs).
Each experiment corresponds to a different physics model and numerical solver.

✅ 1. Laplace Equation (Steady-State Heat / Potential Flow)

File: experiments/laplace_baseline.py

This experiment solves the 3D Laplace equation:

∇
2
𝑢
=
0
∇
2
u=0

or equivalently:

𝑢
𝑥
𝑥
+
𝑢
𝑦
𝑦
+
𝑢
𝑧
𝑧
=
0
u
xx
	​

+u
yy
	​

+u
zz
	​

=0

This describes steady-state heat conduction, electrostatic potential, and any system where the solution is harmonic and source-free.

Typical boundary conditions:

𝑢
(
0
,
𝑦
,
𝑧
)
=
1
,
𝑢
=
0
 on other faces
u(0,y,z)=1,u=0 on other faces
✅ 2. Poisson Equation (Heat with Internal Sources)

File: experiments/poisson_surrogate.py

This experiment solves the 3D Poisson equation:

∇
2
𝑢
=
𝑓
(
𝑥
,
𝑦
,
𝑧
)
∇
2
u=f(x,y,z)

Expanded:

𝑢
𝑥
𝑥
+
𝑢
𝑦
𝑦
+
𝑢
𝑧
𝑧
=
𝑓
(
𝑥
,
𝑦
,
𝑧
)
u
xx
	​

+u
yy
	​

+u
zz
	​

=f(x,y,z)

where 
𝑓
f is an internal source term, such as:

a Gaussian heat source

a single hotspot

or any spatially varying heat generation

Physically, this represents systems where heat, charge, or mass is generated internally, such as:

volumetric heating

electrostatics with charge density

diffusion with a source

✅ 3. Time-Dependent Heat Equation (Transient Diffusion)

File: experiments/heat_time.py

This experiment solves the 3D transient heat equation:

∂
𝑢
∂
𝑡
=
𝛼
∇
2
𝑢
∂t
∂u
	​

=α∇
2
u

or:

𝑢
𝑡
=
𝛼
(
𝑢
𝑥
𝑥
+
𝑢
𝑦
𝑦
+
𝑢
𝑧
𝑧
)
u
t
	​

=α(u
xx
	​

+u
yy
	​

+u
zz
	​

)

This describes how temperature evolves over time inside the domain.

Example setup:

interior starts cold

one boundary is heated

heat diffuses inward as time progresses

This is a classical diffusion process.



| Experiment        | PDE       | Equation                    | Physical Meaning                           |
| ----------------- | --------- | --------------------------- | ------------------------------------------ |
| Laplace Baseline  | Laplace   | ( \nabla^2 u = 0 )          | Steady-state heat or potential, no sources |
| Poisson Surrogate | Poisson   | ( \nabla^2 u = f )          | Heat or potential with internal generation |
| Heat-Time         | Diffusion | ( u_t = \alpha \nabla^2 u ) | Transient heat flow                        |
