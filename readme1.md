```mermaid
flowchart LR
    A[📘 Physics Solver\n(Laplace / Poisson / Heat)] --> B[📊 Dataset Builder\n(points, values)]
    B --> C[🧠 ML Model\n(MLP / PINN / FNO)]
    C --> D[📈 Visualization\n(slice plots, errors)]
    D --> E[📁 Results\n(saved in experiments/.../results)]



---

# ✅ **2. Code Architecture Diagram**  
Shows how experiments import from `src/`.

```markdown
```mermaid
flowchart TB
    A[📂 experiments/] --> B[run.py]
    B --> C[📂 src/physics/]
    B --> D[📂 src/models/]
    B --> E[📂 src/utils/]
    
    C --> C1[laplace_fd.py]
    C --> C2[poisson_fd.py]
    C --> C3[heat_equation_time.py]

    D --> D1[mlp.py]
    D --> D2[pinn.py]
    D --> D3[fno.py]

    E --> E1[dataset.py]
    E --> E2[viz.py]
    E --> E3[seed.py]


---

# ✅ **3. End-to-End Research Workflow Diagram**  
Shows how the repo supports multiple experiments.

```markdown
```mermaid
flowchart LR
    subgraph Exp1[Experiment A\nLaplace Baseline]
    E1A[run.py] --> E1B[config.yaml]
    end
    
    subgraph Exp2[Experiment B\nPoisson Surrogate]
    E2A[run.py] --> E2B[config.yaml]
    end

    subgraph Exp3[Experiment C\nFNO Comparison]
    E3A[run.py] --> E3B[config.yaml]
    end

    Exp1 --> S[📂 src/]
    Exp2 --> S
    Exp3 --> S

    S --> P[Physics Solvers]
    S --> M[ML Models]
    S --> U[Utilities]

    M --> R[📁 Results Folder]
    P --> R
    U --> R



---

# ✅ Want it even cleaner?

I can also generate:

✅ An **ASCII-art diagram** (no mermaid required)  
✅ A **PNG diagram** you can embed in your README  
✅ A **GitHub dark-mode optimized SVG**

Just tell me the format you prefer:

- “ASCII diagram”
- “PNG diagram”
- “SVG diagram”
- “nicer mermaid version”

Would you like **all diagrams**, or just one embedded in your README?
