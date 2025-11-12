# Applying Machine Learning to Computational Fluid Dynamics

Neural Surrogates for Laplace and PDE-Based Physics

This repository contains a fully self-contained Python implementation demonstrating how machine learning can approximate solutions to partial differential equations (PDEs) — specifically the 3D Laplace equation, a fundamental model for steady-state heat conduction and potential flow.

The project walks through the full workflow required to transform a classical numerical simulation into a trainable dataset for a neural network and evaluates how well the learned model reproduces ground-truth physics.


## Project Overview


###  Numerical Physics Solver (Finite Differences)



### Dataset Generation



### Neural Surrogate Model




### Visualization & Evaluation



### Repository Structure

```
applying-machine-learning-to-computation-fluid-dynamics/
│
├── laplace_nn_full.py        # Full pipeline: solver → dataset → NN → plots
├── requirements.txt           # Dependencies
├── README.md                  # Project documentation
└── outputs/                   # Generated plots, logs, models
```
All logic resides in a single, clear script — making it simple to read, modify, and extend.

### How the Script Works (Step-by-Step)

1. Define grid resolution and boundary conditions

2. Solve the Laplace equation numerically (Jacobi iteration)

3. Convert the solution to a supervised dataset

4. Train an MLP to approximate (x, y, z) → u

5. Predict and visualize the learned field

6. Compare neural vs. numerical solutions

### Getting Started

### Create Environment & Install Dependencies

```
python3 -m venv .venv
source .venv/bin/activate
pip3 install -r requirements.txt

python laplace_nn_full.py


python laplace_nn_full.py \
  --nx 30 --ny 30 --nz 30 \
  --max_iters 2000 --tol 1e-5 \
  --hidden 128 --depth 3 \
  --lr 1e-3 --epochs 3000 --batch_size 8192 \
  --train_frac 0.9 --seed 42 \
  --outdir outputs


```