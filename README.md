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
