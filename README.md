
## 🏗️ System Architecture & Multi-Scale Pipelines

The workstation is partitioned into three coupled physical scales. Each scale operates as a single unified view in the UI while executing a 6-stage vectorized pipeline in the backend:

### 1. Micro-Scale: `Benthic-Boundary Engine`
Resolves near-bed turbulent boundary-layer kinematics and vertical suspended-sediment concentration profiles:
* **Sub-module 1.1 — Near-Bed Kinematics:** Computes shear velocity (`u*`) and logarithmic velocity profiles via the Law of the Wall.
* **Sub-module 1.2 — Viscous Sublayer Diagnostics:** Classifies hydraulically smooth, transitional, and rough regimes using boundary Reynolds number (`Re*`) and viscous sublayer thickness (`δ_v`).
* **Sub-module 1.3 — Particle Settling & Entrainment:** Evaluates terminal settling velocity (`w_s`) and near-bed reference concentration (`C_a`) via the Garcia–Parker entrainment formulation.
* **Sub-module 1.4 — Turbulent Eddy Diffusivity Solver:** Computes the vertical parabolic turbulent mixing profile (`ε_s(z)`) via the Reynolds analogy.
* **Sub-module 1.5 — Suspension ODE Integrator:** Solves both the analytical Rouse concentration distribution and the numerical 1D advection–diffusion ODE across the water column.
* **Sub-module 1.6 — Macroscopic Flux Aggregator:** Performs zero-loop numerical quadrature of `u(z) · C(z)` to yield total unit suspended load (`q_ss`).

### 2. Meso-Scale: `Exner Morphodynamics Engine`
Simulates 1D unsteady open-channel hydraulics coupled with shock-capturing riverbed evolution:
* **Sub-module 2.1 — Hydrodynamic State Solver:** Solves 1D Saint-Venant continuity and momentum equations for water depth (`h`), velocity (`u`), free-surface elevation (`η_w`), and Froude number (`Fr`).
* **Sub-module 2.2 — Boundary Condition Manager:** Enforces ghost-cell padding, upstream inflow hydrographs (`Q`), and downstream stage boundaries without spatial loops.
* **Sub-module 2.3 — Basal Shear & Transport Capacity Engine:** Computes bed shear stress (`τ_b`), dimensionless Shields parameter (`θ`), and Meyer-Peter & Müller (MPM) volumetric bedload flux (`q_b`).
* **Sub-module 2.4 — Bed-Evolution Divergence Solver:** Advances riverbed elevation (`z_b`) using a second-order MacCormack Predictor–Corrector scheme with artificial dissipation and machine-epsilon mass-balance auditing (`~10⁻¹⁵` relative error).
* **Sub-module 2.5 — Morphological Stability Controller:** Dynamically regulates time-stepping (`Δt`) via the Courant–Friedrichs–Lewy (`CFL`) condition and morphological acceleration factor (`M_f`).
* **Sub-module 2.6 — State-Space Persister:** Decimates temporal frames for 60-FPS WebGL animation and serializes state tensors into compressed PyArrow `.parquet` and `.zip` archives.

### 3. Macro-Scale: `MorphoTensor Engine` (3D Debris & Mudflow Rheology)
Models hyperconcentrated non-Newtonian mudflows and yield-stress continuum mechanics:
* **Sub-module 3.1 — Mixture Rheology Configurator:** Defines Bingham Plastic and Herschel–Bulkley yield stress (`τ_y`), consistency (`K`), and power-law index (`n_r`).
* **Sub-module 3.2 — 3D Stress Tensor Generator:** Partitions isotropic pressure and deviatoric viscous stress tensors (`σ_ij`) using regularized second invariants of the strain-rate tensor (`II_D`).
* **Sub-module 3.3 — Non-Newtonian Flux Router:** Computes vectorized 3D mass and momentum flux divergence across variable topography.
* **Sub-module 3.4 — Transverse Gravity–Shear Coupler:** Resolves bi-directional slope gravity driving forces against basal and internal shear resistance.
* **Sub-module 3.5 — Yield-Surface Tracker:** Identifies the dynamic interface (`h_p`) separating the upper rigid plug-flow zone (`|τ| ≤ τ_y`) from the lower sheared zone (`|τ| > τ_y`).
* **Sub-module 3.6 — 3D Morphological Updater:** Updates volumetric deposition and erosion fields across the macroscopic domain.

---

## 📐 Governing Continuum Equations

### I. Micro-Scale (`Benthic-Boundary`)

| Physical Mechanism | Governing Formulation | Key Variables & SI Units |
| :--- | :--- | :--- |
| **Law of the Wall** | `u(z) = (u* / κ) · ln(z / z₀)` | `u*`: Shear velocity (`m/s`), `κ`: von Kármán constant (`-`), `z₀`: Zero-velocity roughness height (`m`) |
| **Shear Reynolds & Sublayer** | `Re* = (u* · k_s) / ν` &nbsp;\|&nbsp; `δ_v = 11.6 · (ν / u*)` | `k_s`: Nikuradse roughness (`m`), `ν`: Kinematic viscosity (`m²/s`), `δ_v`: Viscous sublayer (`m`) |
| **Garcia–Parker Entrainment** | `E_s = (A · Z_u⁵) / (1 + (A / 0.3) · Z_u⁵)` | `A = 1.3 × 10⁻⁷`, `Z_u`: Similarity variable (`-`), `C_a = E_s`: Reference concentration (`-`) |
| **Parabolic Eddy Diffusivity** | `ε_s(z) = β_s · κ · u* · z · (1 - z / h)` | `ε_s`: Turbulent sediment diffusivity (`m²/s`), `β_s`: Schmidt inverse (`-`), `h`: Flow depth (`m`) |
| **Suspension Advection–Diffusion** | `ε_s(z) · (dC / dz) + w_s · C(z) = 0` | `C(z)`: Volumetric concentration (`-`), `w_s`: Terminal particle settling velocity (`m/s`) |
| **Rouse Analytical Profile** | `C(z) / C_a = [ ((h - z) / z) · (a / (h - a)) ]^Ro` | `Ro = w_s / (β_s · κ · u*)`: Dimensionless Rouse number (`-`), `a`: Reference level (`m`) |
| **Macroscopic Suspended Flux** | `q_ss = ∫ₐʰ u(z) · C(z) dz` | `q_ss`: Depth-integrated volumetric suspended sediment transport rate (`m²/s`) |

---

### II. Meso-Scale (`Exner Morphodynamics`)

| Physical Mechanism | Governing Formulation | Key Variables & SI Units |
| :--- | :--- | :--- |
| **Saint-Venant Continuity** | `∂h/∂t + ∂(h·u)/∂x = 0` | `h`: Water depth (`m`), `u`: Depth-averaged velocity (`m/s`), `t`: Time (`s`), `x`: Streamwise coordinate (`m`) |
| **Saint-Venant Momentum** | `∂(h·u)/∂t + ∂(h·u² + ½·g·h²)/∂x = g·h·(S₀ - S_f)` | `g`: Gravity (`m/s²`), `S₀ = -∂z_b/∂x`: Bed slope (`-`), `S_f = (n²·u·\|u\|) / h^(4/3)`: Friction slope (`-`) |
| **Shields Parameter** | `θ = τ_b / [ (ρ_s - ρ_w) · g · d₅₀ ]` | `τ_b = ρ_w·g·h·S_f`: Basal shear (`Pa`), `ρ_s, ρ_w`: Sediment/water density (`kg/m³`), `d₅₀`: Median grain (`m`) |
| **Meyer-Peter & Müller Flux** | `q_b = α_mpm · √((s - 1)·g·d₅₀³) · max(0, θ - θ_c)^β_mpm` | `q_b`: Volumetric bedload (`m²/s`), `s = ρ_s/ρ_w`, `θ_c`: Critical Shields (`-`), `α_mpm, β_mpm`: MPM constants |
| **Exner Conservation PDE** | `(1 - λ_p) · (∂z_b / ∂t) + M_f · (∂q_b / ∂x) = 0` | `z_b`: Bed elevation (`m`), `λ_p`: Bed porosity (`-`), `M_f`: Morphological acceleration factor (`-`) |
| **MacCormack Predictor** | `z_b,i* = z_b,iⁿ - [ M_f · Δt / ((1 - λ_p)·Δx) ] · (q_b,i+1ⁿ - q_b,iⁿ)` | Forward finite-difference predictor step across spatial grid spacing `Δx` (`m`) |
| **MacCormack Corrector** | `z_b,iⁿ⁺¹ = ½ · [ z_b,iⁿ + z_b,i* - [ M_f · Δt / ((1 - λ_p)·Δx) ] · (q_b,i* - q_b,i-1*) ] + D_art` | Backward finite-difference corrector step with symmetric TVD artificial viscosity `D_art` |
| **CFL Stability Criterion** | `Δt = CFL · Δx / max( \|u\| + √(g·h), c_m )` | `CFL`: Courant number (`≤ 1`), `c_m`: Bedform celerity (`m/s`), `Δt`: Stable time step (`s`) |

---

### III. Macro-Scale (`MorphoTensor` 3D Rheology)

| Physical Mechanism | Governing Formulation | Key Variables & SI Units |
| :--- | :--- | :--- |
| **Herschel–Bulkley Law** | `τ = τ_y + K · (γ̇)^n_r` &nbsp;(for `\|τ\| > τ_y`) | `τ_y`: Yield stress (`Pa`), `K`: Consistency index (`Pa·sⁿ`), `n_r`: Rheological index (`-`), `γ̇`: Shear rate (`s⁻¹`) |
| **3D Cauchy Stress Tensor** | `σ_ij = -p · δ_ij + 2 · η_eff(II_D) · D_ij` | `p`: Isotropic pressure (`Pa`), `δ_ij`: Kronecker delta, `D_ij = ½(∂u_i/∂x_j + ∂u_j/∂x_i)`: Strain-rate tensor |
| **Regularized Viscosity** | `η_eff = K · \|II_D\|^((n_r - 1)/2) + τ_y / (2·√\|II_D\| + ε_reg)` | `II_D = ½(D_ij · D_ij)`: Second invariant of strain rate (`s⁻²`), `ε_reg`: Papanastasiou regularizer (`s⁻¹`) |
| **Yield-Surface Plug Depth** | `h_p = τ_y / (ρ_m · g · sin α)` | `h_p`: Rigid plug-flow thickness (`m`), `ρ_m`: Mixture density (`kg/m³`), `α`: Local terrain inclination (`rad`) |

---

## 📂 Repository File Structure

```text
HydroMorph-Studio/
├── frontend/
│   ├── app.py                        # Never-fail Streamlit shell & Global Plotly/UI override hooks
│   ├── registry.py                   # Dynamic model router (Micro, Meso, Macro)
│   ├── components/
│   │   ├── api_client.py             # FastAPI dispatcher with automatic in-process fallback
│   │   ├── theme.py                  # Pink/Blue Glassmorphism CSS injection
│   │   └── validators.py             # Strict SI unit input boxes & instant physical bounds validation
│   └── views/
│       ├── micro_benthic.py          # Unified Micro-Scale UI
│       ├── meso_exner.py             # Unified Meso-Scale UI
│       └── macro_morphotensor.py     # Unified Macro-Scale UI
├── core/
│   └── schemas/                      # Dataclass/Pydantic contracts (zero hardcoded constants)
├── backend/
│   ├── main.py                       # FastAPI application entry point
│   ├── routers/                      # REST endpoints (/api/v1/micro, /meso, /macro)
│   └── engines/
│       ├── micro/                    # Sub-modules 1.1 to 1.6 (Vectorized NumPy/SciPy)
│       ├── meso/                     # Sub-modules 2.1 to 2.6 (MacCormack & PyArrow persistence)
│       └── macro/                    # Sub-modules 3.1 to 3.6 (3D Non-Newtonian Tensor Solvers)
├── run_hydromorph.py                 # Windows PATH-independent unified process launcher
└── README.md
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

* **Sub-module 2.6: State-Space Persister** — Decimates temporal frames for WebGL playback and packages `.parquet` state tensors into `.zip` bundles
