# Plan: Clean ADMIRE Separating Input

Companion/replacement for `examples/admire/admire_separating_input.py` +
`examples/admire/admire_refined_sequence_optimizer.py`, in a dedicated
folder with three standalone demo scripts (mirroring
`examples/unicycle/`'s `single_step_demo.py` / `multistep_unrefined_demo.py`
/ `multistep_refined_demo.py` split) plus one static 3D/valid-pairs plot
script.

## Why a new folder

`examples/admire/` accreted ~25 files around two clean-ish core modules:
several near-duplicate 3D interval-history plot scripts, seven date-stamped
`staged_interval_plots_*.pdf` snapshots, four notebooks, and a SLURM-oriented
success-rate sweep script. This folder is a from-scratch consolidation of
just the separating-input core (system + 3 algorithmic layers) plus one
plot script -- no notebooks, no SLURM sweep, no CPU/ThreadPoolExecutor
optimizer variants.

## What was ported

- `admire_separating_input.py`'s system/scenario/single-step/multistep-unrefined
  machinery -> this folder's `admire_separating_input.py`.
- `admire_refined_sequence_optimizer.py`'s `propagate_with_refinement_admire`
  (vmap-over-pairs + `jax.checkpoint` refinement loss, the memory-optimised
  strategy that took ADMIRE's compile memory from ~30GB to ~1-3GB) -> this
  folder's `propagate_with_refinement`, unchanged algorithmically except for
  the no-overlap-fallback fix below.
- `examples/admire/multistep_unrefined_demo.py` / `multistep_refined_demo.py`
  (already-clean, already-tuned standalone scripts) -> ported here with
  updated import names; tuned config values (DT, horizon, smart Flap-first
  initial guess) preserved from the originals' hard-won tuning history.
- `examples/admire/plot_refinement_3d.py` -> this folder's
  `plot_refinement_3d.py`, refactored to call the shared refinement helper
  instead of re-deriving the math a third time (see bug fix below); now
  solves its own control sequence instead of requiring a pre-saved
  `admire_u_opt.npz` from a notebook.
- `examples/admire/admire.py` (`AdmireNineDoFLinAct`) is imported directly,
  not re-transcribed -- it is already a small (~150-line), dependency-free,
  sprawl-free physics-constants-and-dynamics file; copying its aerodynamic/
  inertia coefficient literals by hand would risk a transcription error for
  no benefit.

## Dropped (redundant, not used by any demo script here)

- `optimize_multistart` / `optimize_parallel` (sequential-CPU and
  ThreadPoolExecutor variants of the single-step optimizer) -- superseded
  by the GPU-vmap versions everywhere else in this codebase.
- The non-fused `*_rejit` optimizer variants -- only the "_fused"
  (`jax.vmap(jax.value_and_grad(...))`) variants are kept, since they avoid
  compiling the expensive forward pass twice.
- `plot_3d_interval_history` (a general-purpose plotting utility embedded in
  the core module) -- plotting now lives only in `plot_refinement_3d.py`,
  matching this project's separation of core-module vs. plot-script
  concerns elsewhere (unicycle, nonlinear_chain, quadrotor_fault_diagnosis).

## Runtime-optimization changes (explicit user request)

1. **Every `jax.lax.fori_loop` replaced with `jax.lax.scan`**
   (`propagate_scenario`, `_propagate_history`'s inner segment loop, and
   every `optimize_*_gpu`'s outer GD loop, via the `_scan_loop` helper).
   Scan has lower per-iteration dispatch overhead than fori_loop for this
   kind of loop -- an established, already-tested finding in this codebase
   (see `examples/unicycle/car_separating_input.py`'s `_scan_loop`
   docstring and its three fori_loop->scan conversions, verified there via
   the full 26-test suite passing unchanged). `admire_refined_sequence_optimizer.py`'s
   inner per-step `jax.lax.scan` (already scan, vmapped over pairs) was
   left as-is; only its OUTER GD-loop `fori_loop`s were converted.
2. **Only the fused `jax.vmap(jax.value_and_grad(...))` optimizer variants
   are kept** (see "Dropped" above) -- avoids tracing/compiling the
   expensive division-heavy forward pass twice per optimizer.
3. **NaN-safe argmin preserved** (`jnp.where(isnan, inf, losses)` before
   `argmin`) -- ADMIRE's steep dt/horizon sensitivity produces real NaN
   restarts (see `multistep_unrefined_demo.py`'s DT-tuning history:
   dt=0.3 -> 49/50 restarts NaN); plain `argmin` does not reliably skip them.

**Important caveat on what these changes do NOT fix**: ADMIRE's actual
runtime bottleneck is JIT COMPILE time (tens of seconds to several minutes
for a single gradient, confirmed empirically while building this folder --
see below), driven by reverse-mode-differentiating through
`AdmireNineDoFLinAct.f`'s division/tan operations (`alpha_der`, `beta_der`,
`psi_der`, `phi_der`), not by loop mechanics or restart/config batch size.
The fori_loop->scan conversion reduces per-iteration DISPATCH overhead
once compiled, matching the same optimization already validated elsewhere
in this codebase, but it does not touch this compile-time floor -- that is
why every demo script here keeps `num_iters`/`num_restarts` deliberately
small (the controllable, POST-COMPILE knob) rather than trying to shrink
compile time itself.

## Bug found and fixed: refinement no-overlap fallback

`admire_refined_sequence_optimizer.py`'s `step_one_pair` (and
`admire/plot_refinement_3d.py`'s independent reimplementation of the same
math) both fell back, on no-overlap, to a fabricated observation derived
from scenario i's OWN output center alone, then overwrote BOTH scenarios'
output slice with that fabricated value before propagating forward. The
per-step loss is correctly masked to 0 for a no-overlap step (`jnp.where
(has_overlap, raw, 0.0)`), but the CARRIED-FORWARD STATE was not gated the
same way, so a no-overlap step could corrupt the interval seen by later
refinement steps. This is the exact bug already found and fixed in
`examples/unicycle/car_separating_input.py` (its PLAN.md bug #4) and
avoided from the start in `nonlinear_chain_separating_input.py` /
`quadrotor_separating_input.py`. **Fixed here** the same proven way: on
no-overlap, `_step_one_pair` (this folder's one shared per-pair helper,
used by BOTH `propagate_with_refinement`'s jittable loss and
`collect_refinement_history`'s plotting collector -- closing the same
"independently-drifting reimplementations" gap flagged as bug #3 in
unicycle's PLAN.md) now falls back to the UNCHANGED prior `xi_ut`/`xj_ut`
via `jnp.where` gating the whole state, not a derived fallback observation.
Regression test: `test_step_one_pair_no_overlap_falls_back_to_unchanged_state`.

## Verified empirically while building this folder (CPU, JAX_PLATFORMS=cpu)

- Forward-only propagation (no grad) is cheap: a single `euler_step` ~2.6s
  (first-call natif-embedding trace), a 2-step `propagate_scenario` (eager,
  i.e. still traces the `lax.scan` body once) ~3.0s, and its `jax.jit`
  forward-only compile ~2.9s -- confirms the compile-cost problem is
  specifically in the GRADIENT, not the forward pass or the scan/fori_loop
  choice.
- A single 2-scenario, 1-step `SeparatingInputOptimizer.grad_fn` call
  (i.e. one `jax.grad` compile through one `euler_step`) took ~64s on this
  machine -- confirms even a minimal, reduced-scenario-count gradient
  still pays a real compile floor from the dynamics themselves.
- This is consistent with `multistep_unrefined_demo.py`'s documented
  finding (~266-271s for a single unbatched gradient at 10 segments x 11
  scenarios, identical GPU vs CPU) -- compile time scales with horizon
  length and is not meaningfully reducible by num_restarts/num_iters/vmap
  batching, nor materially changed by the fori_loop->scan conversion.

## System (unchanged from the original)

```
State   x = [Vt, alpha, beta, pb, qb, rb, psi, theta, phi]   (9 states)
Control u = [rc, lc, roe, rie, lie, loe, rudder, flap, yaw_tv, pitch_tv]
                                                              (10 inputs, rad)
Params  p = [p0, ..., p9]  actuator effectiveness in [0, 1] per surface
```

## Scenarios (11 total, fixed -- unchanged from the original)

```
Nominal            p = [1]*10
<surface> fault_i  p[i] = fault_effectiveness (default 0.0, complete loss)
  for i in {Right Canard, Left Canard, Right Outer Elev, Right Inner Elev,
            Left Inner Elev, Left Outer Elev, Rudder, Flap, Yaw TV, Pitch TV}
```

Output space for separation: angular rates `[pb, qb, rb]` (state indices 3:6).

## File layout

```
examples/admire_fault_diagnosis/
  PLAN.md                          (this file)
  admire_separating_input.py       core module (system, scenarios, all 3 layers)
  single_step_demo.py              script 1: single-step separating input
  multistep_unrefined_demo.py      script 2: multistep, no refinement
  multistep_refined_demo.py        script 3: multistep, intersection-refinement
  plot_refinement_3d.py            static plot: valid-pairs-over-time, unrefined vs refined
  tests/
    test_admire_separating_input.py
```
