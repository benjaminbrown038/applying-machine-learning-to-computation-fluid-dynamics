# 🧩 Applying Machine Learning to Computational Fluid Dynamics  
### **Neural Surrogates for Laplace and PDE-Based Physics**

This repository contains a fully self-contained Python implementation demonstrating how **machine learning can approximate solutions to partial differential equations (PDEs)** — specifically the **3D Laplace equation**, a fundamental model for steady-state heat conduction and potential flow.

The project walks through the entire workflow required to transform a classical numerical simulation into a trainable dataset for a neural network, and finally evaluates how well the neural model approximates ground-truth physics.

---

# 🚀 Project Overview

This repository implements a complete end-to-end pipeline:

### ✅ **1. Numerical Physics Solver (Finite Differences)**  
A structured 3D grid is used to solve the Laplace equation:

\[
\nabla^2 u = 0
\]

with user-defined Dirichlet boundary conditions on all cube faces.  
Jacobi iteration is used as the baseline numerical solver, producing the ground-truth solution field.

---

### ✅ **2. Dataset Generation**  
After solving the PDE numerically, the script converts the solution into a dataset:

- Each grid point `(x, y, z)` is paired with its corresponding ground-truth temperature `u(x, y, z)`.  
- The dataset is split into training/validation sets.  
- Normalization and shuffling occur automatically.

This transforms a physics simulation into a machine-learning friendly format.

---

### ✅ **3. Neural Surrogate Model**  
A fully connected **Sigmoid MLP** learns the mapping:

\[
(x, y, z) \longrightarrow u(x, y, z)
\]

Configurable options include:

- Number of hidden layers  
- Hidden dimension width  
- Learning rate  
- Batch size  
- Number of training epochs  

This allows exploration of how network architecture impacts PDE approximation accuracy.

---

### ✅ **4. Visualization & Evaluation**

The script produces:

- Slice-plots comparing predicted vs. finite difference temperature fields  
- Pointwise error maps  
- Training loss curves  
- Optional outputs saved to the `outputs/` directory  

These visualizations help quantify how well the neural network learns underlying physics patterns.

---

# 📁 Repository Structure

applying-machine-learning-to-computation-fluid-dynami/│
├── laplace_nn_full.py # Full pipeline: solver → dataset → NN training → plots
├── requirements.txt # Python dependencies
├── README.md # Project documentation
└── outputs/ # Generated plots, logs, and saved model files


All core functionality is contained in a **single unified script**, making it extremely easy to read, extend, and experiment with.

---

# 🔧 How the Script Works (Step-by-Step)

1. **Define grid resolution and boundary conditions**  
2. **Solve the Laplace equation numerically** using finite differences  
3. **Extract interior nodes** and generate a supervised dataset  
4. **Train the neural model** on `(x, y, z) → u` mappings  
5. **Use the trained model** to predict the solution field everywhere  
6. **Visualize FD vs. NN** to evaluate surrogate performance  

This workflow mirrors modern approaches used in CFD, thermal simulation, and scientific machine learning.

---

# ✅ Getting Started

## ⚙️ Create Environment & Install Dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate

pip3 install -r requirements.txt

▶️ Run the Script (Default Settings)

python laplace_nn_full.py


⚙️ Run with Custom Parameters

python laplace_nn_full.py \
  --nx 30 --ny 30 --nz 30 \
  --max_iters 2000 --tol 1e-5 \
  --hidden 128 --depth 3 \
  --lr 1e-3 --epochs 3000 --batch_size 8192 \
  --train_frac 0.9 --seed 42 \
  --outdir outputs

```
📊 Features & Capabilities
✅ Physics Simulation

3D finite difference method

Configurable grid resolution

Jacobi iteration with tolerance-based stopping

Dirichlet boundary conditions on each face

Vectorized NumPy implementation

✅ Machine Learning

Sigmoid-based MLP

Fully configurable architecture

MSE loss with Adam optimizer

Automatic train/validation split

GPU acceleration when available

✅ Visualization

2D slices at user-defined Z levels

Predicted vs. true fields

Error heatmaps

Training loss curves

Optional saving to the outputs/ directory

directory

📌 Why This Project Matters

This project shows how traditional numerical solvers and modern machine learning techniques can be combined to create fast, lightweight surrogates for physics simulations.

You learn:

How to convert PDE solutions into datasets

How neural networks approximate physical systems

When ML surrogates succeed (and when they fail)

How to accelerate CFD-like workflows

Applications include:

Real-time thermal simulation

Design optimization

Reduced-order modeling

Robotics/embedded thermal estimation

Scientific machine learning research

🔮 Future Extensions

This repository is designed for growth. Planned upgrade paths include:

✅ Poisson Equation Surrogate
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
✅ Time-Dependent Heat Equation
𝑢
𝑡
=
𝛼
∇
2
𝑢
u
t
	​

=α∇
2
u
✅ Spatially Varying Materials

Introduce layered or composite material regions.

✅ Physics-Informed Neural Networks (PINNs)

Add PDE residual to the loss function.

✅ Fourier Neural Operators (FNOs)

State-of-the-art surrogate architecture for PDEs.

✅ Mesh-Based FEM Surrogate Models

Graph neural networks for irregular finite element grids.

If you want any of these implemented, I can generate the code modules.

📬 Author

Benjamin Kahn Brown III — “Trey”
Master’s Student, Mechanical Engineering
University of Georgia