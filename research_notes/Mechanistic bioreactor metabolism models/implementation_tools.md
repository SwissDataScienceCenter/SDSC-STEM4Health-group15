# Implementing and fitting mechanistic and hybrid fed-batch bioreactor models in code (tools, workflows, pitfalls; as of Nov 2026)

Scope note: this covers code-level implementation for a multi-batch CHO-like fed-batch dataset (event log of measurements, volume stamps and bolus feeds; holdout batches have only t=0 plus feed and volume profiles). Release dates and versions come from a live PyPI JSON query on 2026-11-21 (`https://pypi.org/pypi/<pkg>/json`). API details cited to official documentation pages are standard, documented features. Only some of those pages were re-fetched in this session (see Gaps).

---

## Q1. Which libraries to use for (a) simulation with bolus feeds, (b) gradient-based multi-batch estimation, (c) global/derivative-free fitting, (d) Bayesian/hierarchical estimation, (e) SINDy, (f) hybrid/UDE models, plus bioprocess-specific packages

### Takeaway
For a 1–2 day hackathon, the most robust stack is NumPy/SciPy: `solve_ivp` integrated piecewise between feed events, `least_squares` on log-parameters, and multi-start. Add JAX (diffrax + equinox + optimistix/optax, optionally NumPyro) if you want gradients, `vmap` across batches, hybrid NNs or hierarchical Bayes. PyTorch with torchode/torchdiffeq is the equivalent if the team prefers torch. Heavier systems-biology stacks (pyPESTO/AMICI/PEtab) are well maintained, but many bolus events per batch fit them awkwardly. The bioprocess-specific Python packages (pyFOOMB, KIPET, SBML2HYB) are stale or pinned to old Python versions.

### Cited Findings

**Maintenance status (PyPI latest release, queried 2026-11-21)**
- diffrax 0.7.2 (2026-02-18); equinox 0.13.8 (2026-05-05); optimistix 0.1.0 (2026-02-16); jax 0.12.1 (2026-11-18) — [PyPI diffrax](https://pypi.org/project/diffrax/), [PyPI equinox](https://pypi.org/project/equinox/), [PyPI optimistix](https://pypi.org/project/optimistix/), [PyPI jax](https://pypi.org/project/jax/)
- torchode 1.2.0 (2026-11-17), actively maintained. torchdiffeq 0.2.5 (2024-11-21), slower cadence — [PyPI torchode](https://pypi.org/project/torchode/), [PyPI torchdiffeq](https://pypi.org/project/torchdiffeq/)
- scipy 1.18.1 (2026-08-21); lmfit 1.3.4 (2025-07-19); cma (pycma, CMA-ES) 4.6.0 (2026-11-15); pygmo 2.20.0 (2026-10-06); optuna 5.1.0 (2026-11-16); nevergrad 1.0.12 (2025-04-23) — [PyPI scipy](https://pypi.org/project/scipy/), [PyPI lmfit](https://pypi.org/project/lmfit/), [PyPI cma](https://pypi.org/project/cma/), [PyPI pygmo](https://pypi.org/project/pygmo/), [PyPI nevergrad](https://pypi.org/project/nevergrad/)
- casadi 3.8.2 (2026-10-27); pyomo 6.11.0 (2026-11-19) — [PyPI casadi](https://pypi.org/project/casadi/), [PyPI pyomo](https://pypi.org/project/pyomo/)
- pypesto 0.8.0 (2026-11-04); amici 1.2.0 (2026-10-29); petab 0.10.0 (2026-10-26) — [PyPI pypesto](https://pypi.org/project/pypesto/), [PyPI amici](https://pypi.org/project/amici/), [PyPI petab](https://pypi.org/project/petab/)
- numpyro 0.22.0 (2026-09-18); pymc 6.4.0 (2026-11-01); cmdstanpy 1.3.2 (2026-11-03) — [PyPI numpyro](https://pypi.org/project/numpyro/), [PyPI pymc](https://pypi.org/project/pymc/), [PyPI cmdstanpy](https://pypi.org/project/cmdstanpy/)
- pysindy 2.1.0 (2026-01-08) — [PyPI pysindy](https://pypi.org/project/pysindy/)
- scikit-sundae 1.2.0 (2026-10-16), a current Python SUNDIALS (CVODE/IDA) wrapper. assimulo on PyPI is 3.0 (2018-12-15), though conda-forge may carry newer builds — [PyPI scikit-sundae](https://pypi.org/project/scikit-sundae/), [PyPI assimulo](https://pypi.org/project/assimulo/)
- **Stale or legacy:** KIPET 1.0.7 (2022-03-04); SBML2HYB 1.0.3 (2023-01-03); pyFOOMB not on PyPI at all — [PyPI kipet](https://pypi.org/project/kipet/), [PyPI sbml2hyb](https://pypi.org/project/sbml2hyb/)

**(a) ODE simulation with discontinuous bolus feeds**
- SUNDIALS by itself does not support jump discontinuities in the solution. AMICI handles them by resetting solver state at events, and notes that right-hand-side discontinuities can otherwise be stepped over by the solver — [AMICI docs: Handling of discontinuities](https://amici.readthedocs.io/en/latest/implementation_discontinuities.html)
- In diffrax, `ClipStepSizeController` (wrapping e.g. `PIDController`) takes `step_ts` (times that must be stepped to) and `jump_ts` (times where the vector field has a known discontinuity). The solver then steps to just before the jump and restarts just after it, avoiding many rejected steps. The docs cover vector-field discontinuities only, not jumps in the state — [diffrax stepsize controller docs](https://docs.kidger.site/diffrax/api/stepsize_controller/)
- A diffrax user got wrong gradients for a piecewise ODE with a jump. The fix was to declare the jump time explicitly in the step-size controller — [diffrax issue #363](https://www.github.com/patrick-kidger/diffrax/issues/363)
- diffrax terminating events require fixed output array sizes under JAX, so outputs saved after the event time are padded with `inf` — [diffrax issue #335](https://github.com/patrick-kidger/diffrax/issues/335). The older `DiscreteTerminatingEvent`/`SteadyStateEvent` classes were removed in a refactor — [diffrax PR #387](https://www.github.com/patrick-kidger/diffrax/pull/387)
- In Julia DiffEq, bolus doses were missed when the adaptive solver stepped over them. The fix was a time-preset callback (explicit stop times) — [Julia Discourse: Implementing dosing regimens in DiffEq](https://discourse.julialang.org/t/implementing-dosing-regimens-in-diffeq/43173)
- PEtab.jl (Julia) supports dosing events as `PEtabEvent`s, with condition-specific events via per-condition control parameters — [PEtab.jl events docs (dev)](https://sebapersson.github.io/PEtab.jl/dev/petab_event), per search snippet. The `/stable/Julia_event/` URL returned 404 when fetched.
- The aquakin package separates time-scheduled discontinuities from state-threshold events. It splits integration into segments at event times and applies a reset function between segments. This is the same "segment and reset" pattern recommended below — [aquakin events source](https://aquakin.readthedocs.io/en/stable/_modules/aquakin/integrate/events.html)
- `scipy.integrate.solve_ivp` provides RK45/RK23/DOP853 (non-stiff), Radau/BDF (stiff) and LSODA (auto-switching). Its `events` argument locates state-triggered zero crossings, but there is no built-in impulsive state jump, so bolus feeds have to be applied by restarting integration — [SciPy solve_ivp docs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html)

**(b) Gradient-based estimation across many batches**
- A large benchmark (Villaverde et al., 2019) found multi-start with gradient-based local solvers often succeeds. The best performer was a hybrid: scatter-search metaheuristic plus an interior-point local method with adjoint-based gradients — [Villaverde et al. 2019, Bioinformatics (PMC6394396)](https://pmc.ncbi.nlm.nih.gov/articles/PMC6394396)
- pyPESTO provides parameter estimation on PEtab problems with AMICI as the simulation and sensitivity backend. AMICI returns simulations for all experimental conditions in `result.rdatas` — [AMICI PEtab v2 example](https://amici.readthedocs.io/en/latest/examples/example_petab/petab_v2.html), [PEtab repo](https://github.com/ICB-DCM/PEtab)
- **Important limitation:** AMICI's PEtab v2 importer converts PEtab "experiments" (multi-period protocols) into events. Its source raises `NotImplementedError` because AMICI currently does not support more than two periods. A protocol with ~10+ bolus feeds per batch therefore cannot be expressed as PEtab v2 periods in AMICI. It would need hand-written SBML events instead — [AMICI PEtab importer source](https://amici.readthedocs.io/en/latest/_modules/amici/importers/petab/_petab_importer.html)
- A PEtab-GUI for creating and inspecting PEtab problems was released in 2025 — [arXiv 2511.11515](https://arxiv.org/pdf/2511.11515)

**(c) Global / derivative-free fitting**
- pyFOOMB uses pygmo (the pagmo2 Asynchronous Generalized Islands Model) for parallel global calibration — [pyFOOMB GitHub](https://github.com/MicroPhen/pyFOOMB); [pyFOOMB paper, Eng. Life Sci. 2021 (PMC7923582)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7923582/)
- Multi-start local optimisation cannot guarantee global optimality. Kinetic-model calibration is typically multimodal — [Villaverde et al. protocol, arXiv 2105.12008](https://arxiv.org/abs/2105.12008v2); [PMC8769694](https://pmc.ncbi.nlm.nih.gov/articles/PMC8769694)
- A 2024 comparison of local and global solvers for ODE parameter estimation exists — [arXiv 2405.01989](https://arxiv.org/html/2405.01989v2) (not read in detail here).

**(d) Bayesian / hierarchical**
- sunode wraps SUNDIALS (Adams, BDF) with forward and adjoint sensitivities. RHS functions are written in sympy or numba, and it supplies gradients for PyMC. The README reports about 200x speedup over PyMC's built-in ODE solver on its example, a smaller advantage for large problems. It is installed from conda-forge (not on PyPI) and needs `cores=1` on Windows — [sunode GitHub (pymc-devs)](https://github.com/pymc-devs/sunode); [sunode docs, PyMC quickstart](https://sunode.readthedocs.io/en/latest/quickstart_pymc.html)
- PyMC's own `pymc.ode` module exists, but the PyMC docs point to sunode as the faster alternative — [PyMC ode source](https://docs.pymc.io/en/latest/_modules/pymc/ode/ode.html)

**(e) SINDy**
- PySINDy 2.1.0 was released in January 2026 — [PyPI pysindy](https://pypi.org/project/pysindy/)
- Derivative-based SINDy needs densely sampled data and is sensitive to noise. Neural ODEs are contrasted as more tolerant — [Waterloo thesis (Menard)](https://uwaterloo.ca/computational-mathematics/sites/default/files/uploads/documents/menard_jyler_0.pdf)
- A two-step approach works better on noisy biological data: fit a neural network or hybrid model first to smooth and interpolate, then run SINDy on its outputs to get symbolic terms. This follows the UDE-then-SINDy workflow of Rackauckas et al. — [Data-driven model discovery for noisy biological systems (PMC11753677)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11753677/)
- SINDybrid (2025) applies sparse identification only to the uncertain components of a mechanistic model. It was demonstrated on chemical systems — [arXiv 2506.12498](https://arxiv.org/html/2506.12498v1)
- A 2026 LAPSE preprint proposes symbolic regression for interpretable bioprocess kinetics, citing data scarcity — [LAPSE 2026.0309](https://psecommunity.org/wp-content/plugins/wpor/includes/file/2606/LAPSE-2026.0309-1v1.pdf)

**(f) Hybrid / UDE models**
- **CHO-K1 deep hybrid model (Pinto et al., 2023, 24 fed-batches).** Mass-balance ODEs cover 30 species, including dilution, volume and 5 feed streams with their concentrations. A feed-forward NN maps normalised concentrations (30 inputs) to 30 specific rates.
  - Shallow nets were trained with Levenberg–Marquardt; deep nets with ADAM, minibatch 0.8 and weight dropout 0.2.
  - Gradients came from sensitivity equations, the loss was weighted least squares (inverse measurement variance), and integration was RK4.
  - Everything was in-house MATLAB.
  - Repeated 20-train/4-test batch splits (×10) gave test WMSE 1.88±0.44 (deep, [25,25,25], 2,855 weights) vs 2.46±0.42 (shallow). Deep training took about 3 h CPU on a laptop i5.
  - [Pinto et al. 2023, Front. Bioeng. Biotechnol.](https://www.frontiersin.org/articles/10.3389/fbioe.2023.1237963)
- A physics-informed "gray-box" CHO bioreactor model (2023) combines mass balances and kinetic flux expressions with ML. It covers three cases: black-box evolution equations, fitting unknown parameters, and learning partially unknown kinetics. The overdetermined flux-balance step is a differentiable convex-optimisation layer. No code link appears on the abstract page — [arXiv 2305.03257](https://arxiv.org/abs/2305.03257)
- Pinto's 2024 thesis describes SBML2HYB, a Python tool for SBML-compatible hybrid models. Its last PyPI release was January 2023 — [Pinto 2024 thesis](https://run.unl.pt/bitstream/10362/169227/1/Pinto_2024.pdf); [PyPI sbml2hyb](https://pypi.org/project/sbml2hyb/)
- torchode is a parallel ODE solver for PyTorch that solves a batch of independent IVPs with per-sample step sizes. This suits many batches with different event times — [torchode GitHub](https://github.com/martenlienen/torchode)

**Bioprocess-specific packages**
- pyFOOMB (FZ Jülich) is an object-oriented framework for "simple to medium complex" bioprocess ODE models. It covers calibration, sensitivities and uncertainty, handles replicates as model instances, solves with assimulo (CVode) event handling, and optimises with pygmo. It is MIT-licensed, requires Python 3.7–3.9, and installs assimulo and pygmo via conda-forge. It has 132 commits and 12 open issues, with no releases listed — [pyFOOMB GitHub](https://github.com/MicroPhen/pyFOOMB); [pyFOOMB paper](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7923582/)
- KIPET (Pyomo-based kinetic parameter estimation) last released in March 2022 — [PyPI kipet](https://pypi.org/project/kipet/)

### Inferences
- **Recommended hackathon stack, in order of effort:**
  1. **Baseline (hours):** NumPy plus `scipy.integrate.solve_ivp(method="LSODA" or "BDF")`, integrated piecewise between event times, with `scipy.optimize.least_squares` (TRF, `loss="soft_l1"`, `x_scale="jac"`) on log-parameters. Use 20–50 Latin-hypercube multi-starts parallelised with joblib. pycma (CMA-ES) or `differential_evolution(workers=-1)` can find a starting point when the parameter count is ≤ ~20.
  2. **Fast/differentiable (half day):** JAX. Either diffrax (`Tsit5` or `Kvaerno5`) per segment, or a hand-written fixed-step RK4 in `jax.lax.scan` with feeds applied at grid points. Use `jax.vmap` over batches (padded to equal event counts) and `optimistix.least_squares(..., optx.LevenbergMarquardt(...))` or optax Adam.
  3. **Hierarchical (optional):** the same JAX simulator inside NumPyro, with SVI (`AutoNormal`) for speed or NUTS if time allows.
  4. **Hybrid:** the same JAX simulator with an equinox MLP replacing one or two specific rates.
- **Prefer JAX over PyTorch** for ODE estimation in 2026, given diffrax/equinox/optimistix release cadence and implicit solvers (`Kvaerno5`) for stiffness. torchode is the actively maintained torch option. torchdiffeq is still usable but has seen no release since November 2024.
- **Use pyPESTO/AMICI only** if the team already knows the SBML/PEtab workflow. The 2-period limit for PEtab v2 experiments in AMICI, plus the need for per-batch SBML events for many boluses, makes it a poor fit for a 1–2 day sprint. Its profile-likelihood and multistart utilities are excellent, though, and could be reused on an exported objective via `pypesto.Objective(fun, grad)`.
- **CasADi** (`integrator("F","cvodes",...)` plus an `Opti` stack with IPOPT) is the natural tool for multiple shooting. Each shooting interval ends at a feed time and the continuity constraint includes the jump map, so x_{k+1}^start = J(x_k^end, feed_k). Use it if single-shooting fits keep getting stuck.
- **Julia** (DifferentialEquations.jl with `PresetTimeCallback` or `tstops`, SciMLSensitivity, PEtab.jl, DiffEqFlux UDEs) is the most mature event-plus-adjoint ecosystem. It only pays off if a team member already knows Julia; installation and compile latency would eat hackathon time.
- **Avoid pyFOOMB and KIPET** as primary tools: old Python pin and pygmo/assimulo conda dependencies; no release since 2022. pyFOOMB's design (model class plus replicate instances plus global islands) is still a good conceptual template.

### Gaps
- I did not verify the current diffrax `diffrax.Event` API (introduced after PR #387) or how `diffeqsolve` handles `t0 == t1` for zero-length padded segments. Check against diffrax 0.7.2 docs.
- I did not confirm whether pyFOOMB's examples include a fed-batch pulse-feed notebook, or its latest commit date (the GitHub API was not accessible).
- I did not fetch Stan ODE (`ode_bdf`/`ode_rk45`), Torsten (Stan pharmacometrics with dosing events) or NumPyro+diffrax pages. These are known options, but this session did not verify them.
- I found no maintained Python package specifically for CHO fed-batch modelling with bolus feeds. DataHow-like hybrid platforms are commercial.

---

## Q2. How to implement bolus feeds with volume changes and sampling removal (impulsive jumps, measured volume, unknown feed concentrations)

### Takeaway
Integrate the ODE only between events and apply an exact mass-balance jump map at each event: C⁺ = (C·V + c_feed·F)/(V+F), V⁺ = V+F. Sampling removes volume but leaves concentrations unchanged. Snap V to the measured volume whenever a volume stamp exists, and fit unknown feed concentrations as log-parameters, at least for the substrate feed. Never let an adaptive solver "find" a bolus: stop at it explicitly.

### Cited Findings
- Adaptive solvers can step over dose events unless stop times are declared. Explicit preset-time callbacks or stops are needed — [Julia Discourse dosing thread](https://discourse.julialang.org/t/implementing-dosing-regimens-in-diffeq/43173); [AMICI discontinuities](https://amici.readthedocs.io/en/latest/implementation_discontinuities.html)
- The MCKM CHO study (656 Ambr15 runs, 157 clones) implemented fed-batch as a series of batch ODE solves. Each interval starts from the previous end state plus the feed addition, and the fed-batch error is the sum of interval errors. Glucose and glutamate were bolus-fed five times — [Sietaram et al. 2025, MCKM (PMC12486387)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12486387/)
- **Use actual feed profiles, not averages.** In MCKM, one run's feed schedule was applied to all runs as an approximation. Glucose R² was then negative for one product line, which the authors attribute to the averaged feeding inputs — [MCKM (PMC12486387)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12486387/)
- In MCKM, volume V_L(t) appears only in the glucose balance, and dead cells are not a separate state — [MCKM](https://pmc.ncbi.nlm.nih.gov/articles/PMC12486387/)
- The CHO-K1 hybrid model's mass balances include dilution rate, cultivation volume, five feed streams' rates and their concentrations — [Pinto et al. 2023](https://www.frontiersin.org/articles/10.3389/fbioe.2023.1237963)
- diffrax `jump_ts` handles vector-field discontinuities. A state jump still requires restarting the solve with a new y0 — [diffrax stepsize controller docs](https://docs.kidger.site/diffrax/api/stepsize_controller/); [diffrax issue #363](https://www.github.com/patrick-kidger/diffrax/issues/363)

### Inferences

**Event semantics to settle first (check in the data):**
- Is a measurement at the same timestamp as a feed taken before or after the feed? Usually pre-feed. Order events at equal timestamps consistently: measure → sample removal → feed.
- Is feed amount given as volume (mL) or mass? If volume-only rows bracket feeds, ΔV gives F.
- Do volume-only rows reflect sampling (V drops) or feed/base/antifoam additions (V rises)?
- **Sampling** removes liquid at the current concentration, so concentrations are unchanged and only V changes. Resetting V to the measured value at each volume stamp absorbs sampling, evaporation and unlogged additions. Each feed's dilution then uses the correct V.
- **If V drifts between stamps** (evaporation), either ignore it (small over 14 days) or add a dilution/concentration term −(dV/dt)/V·C, using linear interpolation of V.

**State formulation choices:**
- **(i) Concentration states, with the jump map at feeds.** Simplest. Viable and total cells are diluted by feeds too, so VCD⁺ = VCD·V/(V+F).
- **(ii) Total-amount states** n = C·V, with dn/dt = r(C)·V between events. A feed adds n += c_feed·F and a sample removes n −= C·V_sample. This makes mass conservation exact and is convenient if sample volumes are known.
- **Dead cells:** model TCD = VCD + X_dead, with dX_dead/dt = k_d·X_v − k_lysis·X_dead. Fit TCD in log space.

**Unknown feed concentrations:**
- **Substrate feed:** fit c_S,feed as one shared log-parameter. If pre- and post-feed substrate samples exist, estimate it directly from the jump, c_feed ≈ (C⁺(V+F) − C⁻V)/F, and fix it.
- **Feeds B, C, D (unknown composition):**
  - (a) Include them only as dilution (they always contribute F).
  - (b) Add one lumped "nutrient pool" state N with fitted feed concentrations per feed type, entering growth via a Monod term or delaying death.
  - (c) In a hybrid model, pass cumulative feed B/C/D amounts as NN inputs.
  - Start with (a), then test (b) under leave-batch-out CV.

**SciPy skeleton (piecewise integration; concentration states):**
```python
import numpy as np
from scipy.integrate import solve_ivp

# y = [Xv, Xd, S, L]  (VCD, dead cells, substrate, byproduct) ; theta in natural units
def rhs(t, y, th):
    Xv, Xd, S, L = np.maximum(y, 0.0)              # guard against tiny negatives
    mu  = th["mu_max"] * S / (th["Ks"] + S) * th["KiL"] / (th["KiL"] + L)
    kd  = th["kd0"] + th["kdL"] * L / (th["KdL"] + L)
    qS  = mu / th["Yxs"] + th["mS"]
    qL  = th["YLS"] * qS - th["qLc"] * L / (th["KLc"] + L) * th["KSs"] / (th["KSs"] + S)  # smooth lactate switch
    return [ (mu - kd) * Xv, kd * Xv - th["klys"] * Xd, -qS * Xv, qL * Xv ]

def simulate(th, y0, V0, events, t_obs, c_feed):
    """events: list of (t, kind, value[, feed_name]) sorted, kind in {"feed","vol"};
    returns predictions at t_obs (assumed pre-feed at ties)."""
    t_obs = np.asarray(t_obs); out = np.full((len(t_obs), 4), np.nan)
    t, y, V = 0.0, np.asarray(y0, float), V0
    out[t_obs == 0] = y                                   # t=0 rows
    bounds = sorted({e[0] for e in events} | {t_obs.max()})
    for tb in bounds:
        if tb > t:
            m = (t_obs > t) & (t_obs < tb)                # strictly inside -> no duplicate t_eval
            sol = solve_ivp(rhs, (t, tb), y, args=(th,), method="LSODA",
                            t_eval=np.r_[t_obs[m], tb],
                            rtol=1e-6, atol=1e-6 * np.maximum(np.abs(y), 1e-3))
            if sol.status != 0: return np.full_like(out, 1e6)  # penalise failed solves
            if m.any(): out[m] = sol.y[:, :-1].T
            y = sol.y[:, -1]; t = tb
            out[t_obs == tb] = y                          # obs at event time = PRE-feed state
        for (te, kind, val, *rest) in (e for e in events if e[0] == tb):
            if kind == "vol":  V = val                                   # sampling/evaporation reset
            elif kind == "feed":
                F, cin = val, c_feed[rest[0]]                            # cin: array per state (0 for cells)
                y = (y * V + cin * F) / (V + F); V = V + F
    return out
```
- **Measurements exactly at segment ends:** handle them via `t_eval` including `tb`. Make sure `t_eval` stays within `[t, tb]` and is sorted.
- **JAX/vmap pattern:** convert each batch into a fixed-length list of breakpoints made of the union of feed, volume and observation times. Pad to the maximum count with zero-volume dummy feeds at t_end and a mask. Then `lax.scan` over segments, with each step doing (solve segment → record state → apply jump), and `vmap` over batches. Saving only at segment ends (`SaveAt(t1=True)`) avoids variable-size `SaveAt(ts=...)` arrays.
- **Simpler JAX/torch alternative:** fixed-step RK4 with dt ≈ 0.02–0.05 day inside `lax.scan` or a torch loop, with feeds and volume resets snapped to the nearest grid point. Cell-culture dynamics are slow (hours to days), so accuracy is usually adequate, and gradients and `vmap` are trivial. Verify against LSODA on a few batches.

### Gaps
- No published CHO bolus-feed code in Python was found to cross-check conventions (pre/post-feed sampling).
- The MCKM paper does not describe how sampling volume removal was treated beyond V_L(t) in the glucose balance.

---

## Q3. Best practices for multi-batch fitting (shared vs batch-specific parameters, hierarchical models, weighting, multiple shooting, log-parameters, identifiability, leave-batch-out CV)

### Takeaway
Fit one shared (population) parameter vector to all training batches first. Use log-transformed parameters, log-scale residuals for cell densities and per-species noise-scaled residuals, plus multi-start. Only then add a small number of batch-specific random effects (hierarchical/NLME). Check identifiability with sensitivities, collinearity and profile likelihood, and select models by leave-whole-batch-out CV. Per-batch fits of 13 parameters fit each run well but say nothing about an unseen batch.

### Cited Findings
- **Calibration protocol (Villaverde et al.):** a step-by-step, iterate-until-satisfied workflow with reproducible code. It is presented as a multimodal problem that needs global strategies, typically multi-start — [arXiv 2105.12008](https://arxiv.org/abs/2105.12008v2); [Briefings in Bioinformatics, PMC8769694](https://pmc.ncbi.nlm.nih.gov/articles/PMC8769694)
- **Optimiser choice:** multi-start with gradient-based local methods often succeeds. Scatter search plus interior point with adjoint gradients performed best in the benchmark — [Villaverde et al. 2019 (PMC6394396)](https://pmc.ncbi.nlm.nih.gov/articles/PMC6394396)
- **MCKM objective and identifiability:**
  - The objective was least squares on min-max-normalised data, with 13 run-specific parameters fitted by MATLAB `fmincon` (interior point, max 3,000 evaluations) and ode45.
  - Initial guesses for μmax and K_glc came from `nlinfit` on the first four time points. Bounds came from literature and min/max yields across runs. Global optimisers were considered but not used.
  - Identifiability used normalised sensitivity coefficients and a pairwise collinearity index with threshold CI > 20.
  - Y_lac/glc–Y_X/lac had CI ≈ 5.8×10⁶ and Y_gln/glu–Y_X/gln was infinite, so these pairs are non-identifiable. m_glc was poorly identifiable. If R² < 0.90, μmax or K_glc was fixed, reducing the fit to 12 parameters.
  - [MCKM (PMC12486387)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12486387/)
- **MCKM accuracy:** mean R² was 0.96 (biomass) and 0.97 (titre). NRMSE was 6–7% for biomass but about 33–44% for lactate. Lactate was the hardest species even with run-specific parameters — [MCKM](https://pmc.ncbi.nlm.nih.gov/articles/PMC12486387/)
- **Hybrid CHO model weighting and CV:** weighted least squares with inverse measurement variance; 10 random restarts for shallow nets; whole-batch train/test splits (20/4, repeated 10×) — [Pinto et al. 2023](https://www.frontiersin.org/articles/10.3389/fbioe.2023.1237963)
- **Tooling:** pyPESTO plus AMICI and PEtab provide multistart optimisation, profile likelihood and sampling on top of forward/adjoint sensitivities — [AMICI PEtab example](https://amici.readthedocs.io/en/latest/examples/example_petab/petab_v2.html); [PEtab](https://github.com/ICB-DCM/PEtab)

### Inferences

**Parameterisation**
- Optimise φ = log θ with bounds of ±2–3 decades around literature values (e.g., μmax 0.02–0.06 h⁻¹ ≈ 0.5–1.5 d⁻¹ for CHO). This enforces positivity, equalises scales and usually makes landscapes better conditioned.

**Residuals**
- **VCD and TCD:** r = (log(y_obs+ε) − log(y_pred+ε))/σ_log, with σ_log ≈ 0.1–0.2 for counting error.
- **Substrate and byproduct:** (y_obs − y_pred)/σ_species, using a per-species σ (e.g., pooled measurement SD, or 5–10% of the species range). Optionally add a floor for near-zero substrate.
- Normalise per batch so batches with more samples don't dominate. Use a robust loss (`soft_l1`/Huber) to tolerate outliers.
- Align the loss with the hackathon metric. If scoring is RMSE on raw scale, check that the log-space fit is not biased at high VCD.

**Shared vs batch-specific parameters**
- Step 1: all parameters shared, fitted on all training batches.
- Step 2: inspect per-batch residual patterns. If batches differ mainly in peak VCD or decline timing, free 1–3 parameters as batch random effects. Typical choices are μmax, k_d (death onset) and lactate-switch/yield parameters.
- Step 3: NLME/hierarchical. φ_b = φ_pop + η_b with η_b ~ N(0, Ω). In JAX this is a penalised objective, Σ_b ‖r_b‖² + Σ_b η_bᵀΩ⁻¹η_b (MAP/empirical Bayes), or NumPyro SVI/NUTS.

**Multiple shooting and warm starts**
- To escape local minima, either (i) fit per-segment initial states as extra parameters with continuity penalties (CasADi/IPOPT or plain `least_squares`), or (ii) start from a "teacher-forcing" fit where states are reset to measured values at each sample time.
- Then refine with full single-shooting from t=0, because holdout prediction is open-loop from t=0.

**Identifiability**
- Compute the Jacobian of residuals w.r.t. log-parameters at the optimum (FD or JAX). Examine singular values of J (FIM ≈ JᵀJ) and pairwise collinearity, then fix or merge non-identifiable parameters.
- Typical non-identifiable cases are yields that only appear as ratios, and K_s when substrate is never limiting.
- Run a quick profile likelihood on 3–5 key parameters (re-optimise the others on a grid). pyPESTO's profile module can do this if the objective is wrapped.

**Validation**
- Use `sklearn.model_selection.GroupKFold(groups=batch_id)` or leave-one-batch-out, with the full pipeline (including any covariate regression) inside the fold.
- Report per-species error on the same metric and timestamps as the hackathon.

### Gaps
- Not retrieved: specific recommendations from Raue et al. 2009 (profile likelihood), Raue et al. 2013 and Hass et al. 2019 (benchmark collection). They are well-known references, but their recommendations were not verified in this session.
- No CHO-specific published comparison of shared-only vs NLME vs per-batch fits for open-loop prediction of new batches was found.

---

## Q4. Predicting unseen batches from only t=0 plus feeds (population parameters vs covariate-informed batch parameters)

### Takeaway
With only t=0 and the feed/volume schedule, the defensible prediction is the population model simulated from the batch's own initial state and its actual feeds and volumes. Batch-to-batch differences then enter only through initial conditions and feeding. Add covariate models for random effects (initial VCD, viability, substrate, feed totals or timing, any process metadata) only if leave-batch-out CV shows they help. Purely per-batch fitted parameters (MCKM style) are not usable for a new batch.

### Cited Findings
- MCKM fits all 13 parameters per culture with no sharing or transfer learning. It needs the run's own time-series data, so it describes runs rather than forecasting them — [MCKM (PMC12486387)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12486387/)
- Averaged feed inputs degraded glucose predictions (negative R² for one product). The batch's own feed schedule is a key driver of substrate trajectories — [MCKM](https://pmc.ncbi.nlm.nih.gov/articles/PMC12486387/)
- The CHO-K1 hybrid model predicted held-out whole batches (4 of 24) from process inputs. Test WMSE was about 40% higher than training WMSE (1.88 vs 1.35 for the deep model), showing the expected generalisation gap — [Pinto et al. 2023](https://www.frontiersin.org/articles/10.3389/fbioe.2023.1237963)

### Inferences
- **Prediction recipe:**
  1. Read y0 from the t=0 measurement (VCD, TCD → X_d = TCD − VCD, S, L) and V0.
  2. Replay the batch's feed and volume events.
  3. Simulate with population parameters.
- **Hierarchical model:** report the median of the predictive distribution, integrating over η ~ N(0, Ω). For skewed outputs like VCD, the trajectory at mean parameters is not the mean trajectory. Use the median under MAE-type metrics and the mean under RMSE.
- **Covariate model (two-stage, quick):**
  1. Fit η̂_b per training batch (empirical Bayes) with shared φ_pop.
  2. Regress η̂_b on t=0 and feed-schedule features with a ridge or one-variable linear model, within CV folds.
  3. For a holdout batch, φ = φ_pop + f(covariates).
  With "dozens" of batches, limit this to one or two covariates per random effect to avoid overfitting.
- **One-stage alternative:** put φ_b = φ_pop + β·z_b + η_b directly in the JAX objective or NumPyro model.
- **Hybrid NN option:** feed cumulative feed amounts (B/C/D) and time-since-inoculation as inputs to the rate NN. The NN then learns feed-dependent kinetics from training batches and applies them to the holdout through its feed profile. This is the main way unknown-composition feeds can influence predictions.
- **Sanity baseline:** a non-mechanistic regressor (e.g., gradient boosting on [t, y0, cumulative feeds to t, current V]) under the same GroupKFold. It shows whether the mechanistic model is adding value, and can serve as a residual corrector (hybrid in series).

### Gaps
- Found no published study quantifying how much t=0 covariates explain CHO kinetic-parameter variability across batches of the same clone.

---

## Q5. Example code repos, tutorials and competition solutions

### Takeaway
There are few public, ready-made Python repos for CHO fed-batch Monod models with bolus feeds. The best references are generic frameworks (pyFOOMB examples, the Villaverde protocol code, sunode/PyMC quickstarts, diffrax docs) plus papers whose models can be re-implemented (MCKM, Pinto et al. hybrid). I found no Kaggle-style or hackathon solutions for fed-batch trajectory prediction.

### Cited Findings
- pyFOOMB repo, with an `examples` folder of Jupyter notebooks — [GitHub MicroPhen/pyFOOMB](https://github.com/MicroPhen/pyFOOMB). A second copy at `davincios/pyfoomb` appears to be a fork — [GitHub davincios/pyfoomb](https://github.com/davincios/pyfoomb)
- The Villaverde et al. calibration protocol ships code to reproduce two worked examples — [arXiv 2105.12008](https://arxiv.org/abs/2105.12008v2)
- sunode PyMC quickstart (Lotka–Volterra) — [sunode docs](https://sunode.readthedocs.io/en/latest/quickstart_pymc.html)
- **CHO Monod-model parameter references for priors and bounds:**
  - A 2018 *Bioengineering* table of CHO Monod parameters (μmax, glucose/glutamine Monod constants), from fits and literature (MCMC or differential evolution) — [PMC5874891 table](https://pmc.ncbi.nlm.nih.gov/articles/PMC5874891/table/bioengineering-05-00025-t002)
  - A Monod plus Luedeking–Piret CHO model — [Springer 10.1007/s10616-015-9889-2](https://link.springer.com/content/pdf/10.1007/s10616-015-9889-2.pdf)
  - A variable-μmax Monod model (function of integral VCD) for feed optimisation — [Ben Yahia 2017 thesis](https://jahrbib.sulb.uni-saarland.de/bitstream/20.500.11880/26967/4/Ben%20Yahia_PhD_Thesis_20170804_Final%203.0.pdf)
  - A single dynamic metabolic CHO model — [Robitaille 2015](https://publications.polymtl.ca/3485/1/2015_Robitaille_Single_Dynamic_Metabolic_Model_Can.pdf)
- MCKM has no code release. Its data are confidential (GSK) — [MCKM](https://pmc.ncbi.nlm.nih.gov/articles/PMC12486387/). The Pinto et al. data are not public (industrial IP), and the code is in-house MATLAB — [Pinto et al. 2023](https://www.frontiersin.org/articles/10.3389/fbioe.2023.1237963)
- The ChemRxiv version of the multi-cell-line kinetic modelling of CHO fed-batch runs — [ChemRxiv 67f8e5fe…](https://www.cambridge.org/engage/chemrxiv/article-details/67f8e5fe6dde43c908603c49)

### Inferences
- The MCKM equation set (Monod on glucose, ammonium-driven death, Luedeking–Piret yields, lactate consumption below a glucose threshold, plus the μ constraint) is a good template. Reduce it to the four measured species: VCD, TCD (with dead cells), substrate and byproduct. Death could be driven by byproduct or time instead of ammonium, since ammonium is not measured.
- For a code pattern of "segment-wise solve, then jump, then sum residuals across batches", the snippets in Q2 and Q3 are sufficient. No external repo is needed.

### Gaps
- No GitHub repo with Python code for CHO fed-batch with bolus feeds was found in the searches run. A direct GitHub code search (e.g., "fed-batch bolus solve_ivp", "CHO Monod diffrax") was not possible in this session; the GitHub API was not authenticated.
- No hackathon or Kaggle fed-batch trajectory-prediction solutions were found.

---

## Q6. Typical run-time/cost and pitfalls (stiffness, negative concentrations, non-identifiability, overfitting)

### Takeaway
Small Monod-type fed-batch models (4–8 states, ~10–20 parameters, dozens of batches) are cheap: one full multi-batch simulation should take about 0.1–1 s in SciPy and much less in JAX after JIT. The real risks are statistical and numerical rather than compute:
- non-identifiable yield pairs
- overfitting with batch-specific parameters
- solver issues at feed jumps
- negative states
- wrong event-ordering conventions

Deep hybrid models cost hours, not minutes.

### Cited Findings
- **Hybrid training cost:** the deep CHO-K1 hybrid model (2,855 weights, 30 species, 20 training batches) took about 3 h CPU on a laptop i5 in MATLAB. Deep models took 31.6% more CPU than shallow ones on experimental data — [Pinto et al. 2023](https://www.frontiersin.org/articles/10.3389/fbioe.2023.1237963)
- MCKM does not report runtime — [MCKM](https://pmc.ncbi.nlm.nih.gov/articles/PMC12486387/)
- **Identifiability:** severe collinearity between yield pairs (CI up to ∞), and the maintenance coefficient m_glc was poorly identifiable, with estimates far below literature values — [MCKM](https://pmc.ncbi.nlm.nih.gov/articles/PMC12486387/)
- **Lactate is hard to fit** (NRMSE 33–44% even with per-run parameters) — [MCKM](https://pmc.ncbi.nlm.nih.gov/articles/PMC12486387/)
- **Event handling:**
  - Adaptive solvers step over discontinuities unless they are declared — [AMICI discontinuities](https://amici.readthedocs.io/en/latest/implementation_discontinuities.html); [Julia Discourse](https://discourse.julialang.org/t/implementing-dosing-regimens-in-diffeq/43173)
  - Gradients through piecewise ODEs can be wrong without declared jump times — [diffrax #363](https://www.github.com/patrick-kidger/diffrax/issues/363)
  - Terminating events give `inf`-padded outputs in JAX — [diffrax #335](https://github.com/patrick-kidger/diffrax/issues/335)
- **SINDy** is noise- and sampling-density-sensitive, which is problematic for sparse daily samples — [Waterloo thesis](https://uwaterloo.ca/computational-mathematics/sites/default/files/uploads/documents/menard_jyler_0.pdf)
- **Environment friction:**
  - pyFOOMB requires Python ≤3.9 plus conda-only pygmo/assimulo — [pyFOOMB GitHub](https://github.com/MicroPhen/pyFOOMB)
  - sunode is conda-forge only, with a `cores=1` caveat on Windows — [sunode](https://github.com/pymc-devs/sunode)
  - AMICI's PEtab v2 importer supports at most two periods — [AMICI importer source](https://amici.readthedocs.io/en/latest/_modules/amici/importers/petab/_petab_importer.html)

### Inferences
- **Stiffness:** Monod/death models are usually non-stiff, but steep lactate-switch or inhibition terms and near-zero substrate can make them stiff. Default to LSODA (SciPy) or `Kvaerno5` (diffrax). If stiffness causes slow solves, smooth any `if S < S_switch` logic with Hill or sigmoid functions; hard switches also break gradients.
- **Negative concentrations:**
  - Clip inside the RHS (`np.maximum(y, 0)`).
  - Use saturating uptake (S/(K+S)) so uptake → 0 as S → 0.
  - Optionally integrate log(VCD).
  - Set `atol` relative to species scale (VCD ~1e6–3e7 cells/mL vs substrate ~g/L). Better, rescale states to O(1) units such as 1e6 cells/mL and g/L or mM.
- **Penalise failures:** if `solve_ivp` fails (`sol.status != 0`), return large finite residuals rather than NaN so `least_squares`/CMA-ES keep working.
- **Overfitting:** with dozens of batches, keep shared parameters ≤ ~15 and random effects ≤ 3. Watch for random-effect variances that collapse to 0 or blow up. Always compare against the population-only model under leave-batch-out CV.
- **Unit and time pitfalls:**
  - Convert timestamps to days or hours from inoculation per batch.
  - Make sure feed volumes and V share units.
  - Check whether feed amounts are cumulative or per-event.
  - Check whether substrate measurements are pre- or post-feed.
- **Rough compute budget (SciPy; my estimate, not measured):**
  - One batch: ~15–30 segments × LSODA ≈ 5–30 ms, so 40 batches ≈ 0.2–1 s per objective evaluation.
  - FD Jacobian with 15 parameters ≈ 16 evaluations per iteration, and ~50 iterations per start ≈ 3–15 min per start.
  - 20 multi-starts on 8 cores ≈ 10–40 min. A JAX version with `vmap`, JIT and autodiff Jacobians should be 10–100× faster per iteration.
  - Hierarchical NUTS in NumPyro over dozens of batches could take tens of minutes to hours. SVI is faster.

### Gaps
- No measured benchmark timings were found for SciPy vs diffrax vs torchode on small fed-batch models. The figures above are estimates and should be measured on the actual data early in the hackathon.
- I did not verify that `jaxopt` is deprecated in favour of optax/optimistix. Prefer optimistix or optax, which are actively released per PyPI.
