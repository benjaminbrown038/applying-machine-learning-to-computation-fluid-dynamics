## 📁 Project Structure

src/ # Core library code
│
├── physics/ # PDE solvers (Laplace, Poisson, Heat)
├── models/ # Surrogate models (MLP, PINN, FNO)
├── utils/ # Dataset + visualization helpers
└── data/ # Dataset construction

experiments/ # Reproducible experiment folders
│
├── laplace_baseline/
├── poisson_surrogate/
└── fno_comparison/

run_experiment.py # Unified experiment launcher


---

## ▶️ **Run an Experiment**

Direct run:

```bash
python experiments/laplace_baseline/run.py

python run_experiment.py --config experiments/laplace_baseline/config.yaml

flowchart LR
    A[Physics Solver] --> B[Dataset]
    B --> C[ML Model]
    C --> D[Visualization]
    D --> E[Results]
