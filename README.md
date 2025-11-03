# applying-machine-learning-to-computation-fluid-dynamics

# 🧩 Laplace Equation Neural Surrogate

A self-contained Python implementation that:
1. Solves the **3D Laplace equation** using **finite differences** (Jacobi iteration).  
2. Generates training data from the numerical solution.  
3. Trains a **feedforward neural network (Sigmoid MLP)** to learn the solution field.  
4. Visualizes the neural prediction vs. finite difference (FD) ground truth.  

This single script demonstrates how **neural networks can approximate PDE solutions** and how data-driven models can act as surrogates for physics-based solvers.

---

## 📁 File

- **`laplace_nn_full.py`** — Contains everything:  
  boundary conditions, configuration, FD solver, dataset builder, neural net, training, evaluation, and visualization.

---

## ⚙️ Requirements

Install dependencies with:

```bash
pip install numpy torch matplotlib

python laplace_nn_full.py


python laplace_nn_full.py \
  --nx 30 --ny 30 --nz 30 \
  --max_iters 2000 --tol 1e-5 \
  --hidden 128 --depth 3 \
  --lr 1e-3 --epochs 3000 --batch_size 8192 \
  --train_frac 0.9 --seed 42 \
  --outdir outputs

```