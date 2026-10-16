# HydroMorph-Analytics

Multi-scale sediment transport and rheology suite: micro-physics (benthic boundary layer), meso-scale
morphodynamics (Saint-Venant + Exner) and macro-scale non-Newtonian debris flow (MorphoTensor).
Streamlit frontend, FastAPI compute backend, sized for an Intel Core i3 with 8 GB RAM and no GPU.

## Run
```bash
pip install -r requirements.txt
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload   # terminal 1 (optional: the UI falls back to in-process compute)
python -m streamlit run frontend/app.py                                     # terminal 2, both from the project root
```
Set `HYDROMORPH_API_URL` if the API is not on `http://127.0.0.1:8000`.

## Sub-module pipelines (one class each, in `backend/engines/`)
| Engine | Stages |
|---|---|
| Micro (`micro_engine.py`) | 1.1 Near-Bed Kinematics · 1.2 Viscous Sublayer Diagnostics · 1.3 Particle Settling & Entrainment · 1.4 Turbulent Eddy Diffusivity · 1.5 Suspension ODE Integrator · 1.6 Macroscopic Flux Aggregator |
| Meso (`meso_engine.py`) | 2.1 Hydrodynamic State Solver · 2.2 Boundary Condition Manager · 2.3 Basal Shear & Transport Capacity · 2.4 Bed-Evolution Divergence Solver · 2.5 Morphological Stability Controller · 2.6 State-Space Persister |
| Macro (`macro_engine.py`) | 3.1 Mixture Rheology · 3.2 3D Stress Tensor · 3.3 Non-Newtonian Flux Router · 3.4 Transverse Gravity-Shear Coupler · 3.5 Yield-Surface Tracker · 3.6 3D Morphological Updater |

Each stage declares its own inputs, equations and `run`; `BaseModel` aggregates them (shared inputs such as g
appear once under "Physical constants", the rest are grouped by stage in the UI) and reports per-stage compute time.
Micro and macro run their stages once in dependency order. Macro executes 3.1 → 3.4 → 3.3 → 3.2 → 3.5 → 3.6 because
the topology from 3.4 is needed to build the velocity field that the stress tensor in 3.2 differentiates.
Meso couples 2.1–2.6 inside one explicit time march (the only loop in the engines).

## Layout
`frontend/` (app.py shell, registry.py, views/, components/) · `backend/` (main.py, api_registry.py, engines/) ·
`core/` (base_model.py: `SubModule`, `BaseModel`, tensor serialisation; physics_constants.py: defaults only).

## Rules and where they are enforced
| Rule | Enforcement |
|---|---|
| 0-loop vectorisation | Slicing, broadcasting, `einsum`, batched `eigvalsh`, cumulative sums. One `while` time march in `meso_engine.solve`. |
| No hardcoded constants | g, ρ, ν, κ and empirical coefficients (A, C₁, C₂, θ_c, MPM coefficient, …) are inputs. Only structural model-form constants (regime limits Re* = 5/70, the 3/5 Manning exponent, …) are named module constants. |
| Inputs and validation | `st.number_input` only, SI units in labels; `BaseModel.validate` runs every rerun, errors block the button; the API re-validates (HTTP 422). |
| Explicit confirmation | Nothing computes until **Confirm Inputs & Execute**. |
| Mathematics exposed | `st.latex()` for every stage, above the inputs. |
| Transparent charts | `paper_bgcolor` and `plot_bgcolor` are `rgba(0,0,0,0)` in `plots._style`. |
| Locked animation axes | Meso animation: `xaxis`/`yaxis` with fixed `range` and `fixedrange=True`, computed once from the whole tensor. WebGL traces above 1500 points. |
| Scientific log axes | `plots._sci_log`: decade ticks in 10ⁿ notation. |
| Session Management | Sidebar container (`key="session_mgmt"`) holding state upload, "Download system state (.zip)" and the Parquet export. |
| Volumetric shading | Initial-vs-final bed (meso), yield interface and bed deformation (macro) use `fill='tonexty'`. |

## Persistence
Download packs `meta.json` (inputs, scalars) and `arrays.npz` (compressed float32 tensors, no pickle). Uploading
restores the inputs and renders the results without any solver call. The meso page also offers the decimated
(t, x, z_b, h, u) table as zstd Parquet when `pyarrow` is installed.

## Numerical notes and limits
- Meso: MacCormack with Jameson-type smoothing; Riemann-invariant ghost cells; subcritical flow only (Fr < 0.95); rating-curve outflow.
  Exner is a finite-volume Forward Euler update with upwind face fluxes and MPM capacity; dynamic MorFac caps bed change per step. Sediment mass error is reported.
- Macro: steady uniform flow with a local-depth approximation. The tangential-gravity direction fixes the shear-stress direction, so a
  cross-slope drives lateral flow (wall blockage ignored); vertical velocity comes from continuity. The plug shears laterally, so yielding is
  detected from vertical shear. 3.6 is a scour-only excess-shear entrainment law, not a full topology solver.
- Micro: z₀ = 0.11 ν/u* + k_s/30 holds in all regimes; 1.2 classifies the regime from Re*.

## Adding a model
Subclass `SubModule`/`BaseModel` in `backend/engines/`, register in `backend/api_registry.py`, add a view in `frontend/views/`, and one line in `frontend/registry.py`.
# HydroMorph-Studio

**A Commercial-Grade, Multi-Scale Sediment Transport, Fluvial Morphodynamics, and Non-Newtonian Rheology Suite**

## 1. Executive Overview

**HydroMorph-Studio** (formerly *HydroMorph-Analytics*) is a deterministic hydroinformatics workstation engineered from first-principles continuum mechanics and tensor calculus. Unlike black-box empirical calculators, this suite bridges three distinct physical scales of sediment mechanics within a unified, modular architecture:

1. **Micro-Scale (`Benthic-Boundary Engine`):** 1D vertical boundary-layer kinematics, viscous sublayer diagnostics, turbulent burst entrainment, and analytical/numerical Rouse suspension profiling.

2. **Meso-Scale (`Exner Morphodynamics Engine`):** Coupled 1D Saint-Venant shallow-water hydrodynamics, bedload transport capacity, shock-capturing MacCormack bed evolution, and Courant-Friedrichs-Lewy (CFL) stability management.

3. **Macro-Scale (`MorphoTensor Engine`):** 3D non-Newtonian hyper-concentrated debris flows, Bingham/Herschel-Bulkley viscoplastic stress tensors, yield-surface tracking, and multi-axial finite-volume routing.

## 2. Core Engineering Philosophy & Hardware Optimization

HydroMorph-Studio is explicitly architected to execute high-precision numerical simulations on baseline hardware (**Intel Core i3 CPU with 8 GB RAM**, zero GPU dependency) while preserving machine-epsilon mass conservation:

* **Strict 0-Loop Spatial Vectorization:** Python `for` loops over spatial grids are strictly prohibited. All spatial gradients, finite-volume fluxes, and numerical integrations utilize contiguous C-level NumPy array slicing, broadcasting, and SciPy ODE/quadrature routines. Outer loops are restricted exclusively to temporal integration ($t^n \rightarrow t^{n+1}$).

* **Decoupled Fault-Tolerant Architecture:** A lightweight **Streamlit** frontend communicates with an asynchronous **FastAPI** compute server (`127.0.0.1:8000`). If the FastAPI server is offline, the frontend automatically falls back to an in-process execution engine without interrupting user workflow.

* **Unbreakable Registry Pattern:** The entry point (`frontend/app.py`) is an immutable thin shell that dynamically resolves modules via `frontend/registry.py`. Adding or modifying physical engines never risks crashing the primary application.

* **Columnar State-Space Persistence:** Simulation outputs and decimated space-time tensors are serialized into compressed binary **Apache Parquet** (`.parquet`) tables via `pyarrow` alongside JSON parameter manifests inside downloadable `.zip` archives, enabling zero-copy state restoration.

* **WebGL Kinematic Offloading:** Transient morphological animations offload frame rendering directly to the browser's WebGL context using locked coordinate bounds (`fixedrange=True`), eliminating CPU re-layout bottlenecks during playback.

## 3. Multi-Scale Physics Architecture & Governing Equations

### Scale I: Micro-Scale — Benthic-Boundary Engine

Resolves the vertical equilibrium between downward gravitational settling and upward turbulent diffusion across the flow depth $z \in [a, h]$.

#### Sequential 6-Submodule Pipeline:

* **Sub-module 1.1: Near-Bed Kinematics** — Evaluates basal shear velocity and generates the logarithmic velocity profile.

* **Sub-module 1.2: Viscous Sublayer Diagnostics** — Computes laminar sublayer thickness and classifies hydraulic roughness regimes via the grain Reynolds number.

* **Sub-module 1.3: Particle Settling & Entrainment** — Computes terminal fall velocity and solves the Garcia-Parker near-bed reference concentration $C_a$.

* **Sub-module 1.4: Turbulent Eddy Diffusivity Solver** — Constructs the vertical parabolic momentum/mass diffusivity distribution under the Reynolds analogy.

* **Sub-module 1.5: Suspension ODE Integrator** — Solves the 1D steady-state advection-diffusion ordinary differential equation alongside the exact analytical Rouse profile.

* **Sub-module 1.6: Macroscopic Flux Aggregator** — Performs vectorized numerical quadrature over the product of velocity and concentration fields.

#### Governing Formulations:

* **Shear Velocity & Viscous Sublayer Thickness:**
  

  $$
  u_* = \sqrt{\frac{\tau_b}{\rho_w}}, \qquad \delta_v = 11.6 \frac{\nu}{u_*}, \qquad Re_* = \frac{u_* k_s}{\nu}
  $$

* **Prandtl-von Kármán Law of the Wall:**
  

  $$
  u(z) = \frac{u_*}{\kappa} \ln\left(\frac{z}{z_0}\right)
  $$

* **Garcia-Parker Entrainment & Reference Concentration (**$z = a = 0.05h$**):**
  

  $$
  R_p = \frac{\sqrt{R_g g d_{50}}\, d_{50}}{\nu}, \qquad Z_u = \frac{u_*'}{w_s} R_p^{0.6}, \qquad C_a = E_s = \frac{A Z_u^5}{1 + \frac{A}{0.3} Z_u^5}
  $$

  
  where $A = 1.3 \times 10^{-7}$ and $R_g = (\rho_s - \rho_w)/\rho_w$.

* **Parabolic Turbulent Diffusivity & Advection-Diffusion Balance:**
  

  $$
  \varepsilon_s(z) = \kappa u_* z \left(1 - \frac{z}{h}\right), \qquad w_s C(z) + \varepsilon_s(z) \frac{dC}{dz} = 0
  $$

* **Rouse Analytical Suspension Profile & Depth-Integrated Flux:**
  

  $$
  \frac{C(z)}{C_a} = \left(\frac{h - z}{z} \cdot \frac{a}{h - a}\right)^{Ro}, \qquad Ro = \frac{w_s}{\kappa u_*}, \qquad q_{ss} = \int_{a}^{h} u(z) C(z)\, dz
  $$

### Scale II: Meso-Scale — Exner Morphodynamics Engine

Simulates longitudinal riverbed aggradation, degradation, and bedform migration along a 1D spatial domain $x \in [0, L]$ over morphological time $t$.

#### Sequential 6-Submodule Pipeline:

* **Sub-module 2.1: Hydrodynamic State Solver** — Evaluates quasi-steady 1D Saint-Venant depth, velocity, and free-surface elevation profiles.

* **Sub-module 2.2: Boundary Condition Manager** — Enforces ghost-cell padding, upstream equilibrium sediment supply, and downstream stage conditions.

* **Sub-module 2.3: Basal Shear & Transport Capacity Engine** — Computes local bed shear stress, dimensionless Shields stress, and Meyer-Peter Müller bedload flux.

* **Sub-module 2.4: Bed-Evolution Divergence Solver** — Integrates the 1D Exner conservation law via a shock-capturing MacCormack predictor-corrector finite-volume scheme and audits relative mass-balance error down to $O(10^{-15})$.

* **Sub-module 2.5: Morphological Stability Controller** — Dynamically regulates $\Delta t$ via hydraulic and morphological CFL criteria and applies the Morphological Acceleration Factor ($M_f$).

* **Sub-module 2.6: State-Space Persister** — Decimates temporal frames for WebGL playback and packages `.parquet` state tensors into `.zip` bundles.

#### Governing Formulations:

* **1D Saint-Venant Continuity & Momentum Equations:**
  

  $$
  \frac{\partial h}{\partial t} + \frac{\partial (uh)}{\partial x} = 0, \qquad \frac{\partial (uh)}{\partial t} + \frac{\partial}{\partial x}\left(u^2 h + \frac{1}{2} g h^2\right) = -g h \frac{\partial z_b}{\partial x} - g h S_f
  $$

* **Basal Shear Stress, Shields Parameter & Meyer-Peter Müller Bedload:**
  

  $$
  \tau_b = \frac{\rho_w g n^2 u^2}{h^{1/3}}, \qquad \theta = \frac{\tau_b}{(\rho_s - \rho_w) g d_{50}}, \qquad q_b = \alpha_{\text{mpm}} \sqrt{R_g g d_{50}^3} \max\left(0,\, \theta - \theta_c\right)^{\beta_{\text{mpm}}}
  $$

* **1D Exner Morphodynamic Conservation Equation:**
  

  $$
  (1 - \lambda_p) \frac{\partial z_b}{\partial t} + M_f \frac{\partial q_b}{\partial x} = 0
  $$

* **MacCormack Predictor-Corrector Discretization (with Artificial Viscosity** $D_i$**):**
  

  $$
  z_{b,i}^* = z_{b,i}^n - \frac{M_f \Delta t}{(1 - \lambda_p) \Delta x} \left(q_{b,i+1}^n - q_{b,i}^n\right) + D_i^n
  $$

  $$
  z_{b,i}^{n+1} = \frac{1}{2}\left[z_{b,i}^n + z_{b,i}^* - \frac{M_f \Delta t}{(1 - \lambda_p) \Delta x} \left(q_{b,i}^* - q_{b,i-1}^*\right)\right] + D_i^*
  $$

* **Courant-Friedrichs-Lewy (CFL) Adaptive Time-Step Constraint:**
  

  $$
  \Delta t = \text{CFL} \cdot \frac{\Delta x}{\max_i \left(\vert{}u_i\vert{} + \sqrt{g h_i},\, c_{m,i}\right)}
  $$

### Scale III: Macro-Scale — MorphoTensor Engine

Models hyper-concentrated mudflows and viscoplastic debris flows where water and granular solids behave as a single non-Newtonian continuum.

#### Sequential 6-Submodule Pipeline:

* **Sub-module 3.1: Mixture Rheology Configurator** — Parameterizes bulk mixture density, yield stress $\tau_y$, consistency index $K$, and flow behavior index $m$.

* **Sub-module 3.2: 3D Stress Tensor Generator** — Evaluates the rate-of-strain tensor $S_{ij}$ and partitions yielding vs. unyielded deviatoric stress components.

* **Sub-module 3.3: Non-Newtonian Flux Router** — Computes 3D mass and momentum flux divergence across hexahedral finite-volume cells.

* **Sub-module 3.4: Transverse Gravity-Shear Coupler** — Resolves lateral spreading and topographic slope forcing across $(x, y, z)$.

* **Sub-module 3.5: Yield-Surface Tracker** — Identifies the kinematic interface $h_p$ separating the rigid upper plug-flow region from the basal shear zone.

* **Sub-module 3.6: 3D Morphological Updater** — Executes vectorized multi-dimensional stencil updates for bed and free-surface evolution.

#### Governing Formulations:

* **Cauchy Momentum Conservation in Tensor Form:**
  

  $$
  \rho_m \left(\frac{\partial u_i}{\partial t} + u_j \frac{\partial u_i}{\partial x_j}\right) = -\frac{\partial p}{\partial x_i} + \rho_m g_i + \frac{\partial \tau_{ij}}{\partial x_j}
  $$

* **Herschel-Bulkley / Bingham Constitutive Stress Tensor:**
  

  $$
  \tau_{ij} = 2 \left(K \, \dot{\gamma}^{m-1} + \frac{\tau_y}{\dot{\gamma}}\right) S_{ij} \quad \text{for } \sqrt{\frac{1}{2}\tau_{kl}\tau_{kl}} > \tau_y, \qquad S_{ij} = \frac{1}{2}\left(\frac{\partial u_i}{\partial x_j} + \frac{\partial u_j}{\partial x_i}\right)
  $$

* **3D Finite-Volume Divergence Formulation:**
  

  $$
  (1 - \lambda_p) \frac{\partial \eta}{\partial t} + \frac{\partial q^i}{\partial x_i} = 0 \implies \eta_{i,j,k}^{n+1} = \eta_{i,j,k}^n - \frac{\Delta t}{(1 - \lambda_p)\Delta x \Delta y} \sum_{\text{faces}} \left(\mathbf{q}_{\text{face}} \cdot \mathbf{n}\right)
  $$

## 4. Repository Structure

```
HydroMorph-Studio/
│
├── frontend/
│   ├── app.py                              # Fault-tolerant entry point & global UI override hooks
│   ├── registry.py                         # Central model dictionary & lazy-loading router
│   ├── components/
│   │   ├── api_client.py                   # FastAPI HTTP client with automatic in-process fallback
│   │   ├── theme.py                        # Pink/Blue Glassmorphism CSS injector
│   │   └── validators.py                   # Instant physical bounds & SI unit validation
│   └── views/
│       ├── micro_benthic.py                # Unified Micro-Scale UI (Rouse, Diffusivity, Law of Wall)
│       ├── meso_exner.py                   # Unified Meso-Scale UI (Saint-Venant, Exner, WebGL player)
│       └── macro_tensor.py                 # Unified Macro-Scale UI (Herschel-Bulkley, Plug-flow)
│
├── core/
│   └── schemas/
│       ├── micro_schema.py                 # Explicit data contracts for Micro-Scale
│       ├── meso_schema.py                  # Explicit data contracts for Meso-Scale
│       └── macro_schema.py                 # Explicit data contracts for Macro-Scale
│
├── backend/
│   ├── main.py                             # FastAPI application server
│   ├── routers/
│   │   ├── micro_router.py                 # POST /api/v1/micro/simulate
│   │   ├── meso_router.py                  # POST /api/v1/meso/simulate
│   │   └── macro_router.py                 # POST /api/v1/macro/simulate
│   └── engines/
│       ├── micro/                          # Sub-modules 1.1 to 1.6 (Vectorized 1D Vertical)
│       ├── meso/                           # Sub-modules 2.1 to 2.6 (Vectorized 1D Longitudinal)
│       └── macro/                          # Sub-modules 3.1 to 3.6 (Vectorized 3D Tensorial)
│
├── requirements.txt
└── README.md

```

## 5. UI/UX & Visualization Standards

* **Pink & Blue Glassmorphism:** Frosted translucent containers (`backdrop-filter: blur(12px)`), soft radial gradients, and fully transparent Plotly canvases (`paper_bgcolor='rgba(0,0,0,0)'`, `plot_bgcolor='rgba(0,0,0,0)'`).

* **Zero-Slider Policy & Explicit SI Units:** Sliders (`st.slider`) are strictly banned to prevent accidental continuous re-computation. Every parameter uses `st.number_input` with explicit SI units in the label and zero hardcoded physical constants.

* **Instant Physical Validation & Confirmation Gate:** Out-of-bounds physical values trigger immediate blocking `st.error()` alerts. Simulations execute strictly upon clicking **Confirm Inputs & Execute**.

* **Engineering Graph Formatting:** Log-scale axes enforce power-of-ten scientific notation ($10^{-4}, 10^{-3}$), initial-vs-final bed comparisons feature volumetric `fill='tonexty'` shading, and state export tools are housed inside a dedicated **Session Management** container.

## 6. Installation & Execution

### Prerequisites

* Python $3.10+$

* Git

### 1. Clone the Repository

```
git clone https://github.com/saunakbanerjee484-byte/HydroMorph-Studio.git
cd HydroMorph-Studio

```

### 2. Create & Activate a Virtual Environment

```
# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate

```

### 3. Install Dependencies

```
pip install -r requirements.txt

```

### 4. Launch the Suite

**Option A: Full Decoupled Mode (FastAPI Backend + Streamlit Frontend)**
Open two terminals from the project root:

*Terminal 1 — Start the FastAPI Compute Engine:*

```
uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload

```

*Terminal 2 — Start the Streamlit Workstation:*

```
streamlit run frontend/app.py

```

**Option B: Standalone In-Process Mode**
Simply launch the Streamlit app; it will automatically detect that port `8000` is offline and execute all vectorized NumPy/SciPy pipelines directly in-process:

```
streamlit run frontend/app.py

```

## 7. Author & License

Developed by **Saunak Banerjee** — Repository: [HydroMorph-Studio](https://github.com/saunakbanerjee484-byte/HydroMorph-Studio)