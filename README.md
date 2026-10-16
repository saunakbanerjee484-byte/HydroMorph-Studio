🌊 HydroMorph-Studio (`HydroMorph-Analytics`)

**Commercial-Grade Multi-Scale Sediment Transport, Exner Morphodynamics & 3D Non-Newtonian Rheology Workstation**

`HydroMorph-Studio` is a decoupled, local-first computational hydroinformatics suite engineered for high-precision river mechanics and debris-flow simulation. Designed to execute seamlessly on constrained hardware (**Intel Core i3 / 8GB RAM**), the engine enforces a strict **0-Loop Spatial Vectorization Policy** using contiguous NumPy/SciPy array broadcasting, a resilient Streamlit Glassmorphic UI (`frontend/app.py`), and an asynchronous FastAPI compute backend (`127.0.0.1:8000`) with automatic in-process fallback.

---

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
