# Plan: Standalone Go2 Separating Input

New, dedicated folder for the Go2 quadruped's immrax-based separating-input
core, extracted from `examples/go2/go2_separating_input_immrax.py` (the
"canonical template" for this project's active-fault-diagnosis modules),
mirroring `examples/unicycle/`'s split off of `examples/faulty_car/`.

## Why a new folder

`examples/go2/` has ~60 files: hardware/Robotarium artifacts, multiple
controller variants (output feedback, yaw-only, CBF), notebooks, LaTeX
reports, datasets. `go2_separating_input_immrax.py` (1224 lines) is itself
already a fairly clean core, but it (a) has no standalone `.py` demo
scripts for its immrax version (only two Jupyter notebooks), (b) depends on
`examples/faulty_car/interval_functions.py` for its overlap-volume helper,
and (c) has a large CBF/obstacle-avoidance section bolted onto the end.
This folder extracts just the separating-input core + demos + plots, with
no CBF and no dependency on `examples/faulty_car`.

## What was ported

- `go2_separating_input.py` = `go2_separating_input_immrax.py`'s Sections
  1-5 (systems, scenarios, propagation, loss, optimizers) verbatim in
  logic, with the changes below. Section 6 (CBF obstacle avoidance) is
  dropped entirely, matching this project's established pattern of keeping
  the separating-input core free of closed-loop/CBF machinery (see
  `unicycle/PLAN.md`).
- `tests/test_go2_separating_input.py` = `tests/test_go2_separating_input_immrax.py`
  verbatim (import path updated only). **27/29 pass** -- the same 2
  failures as the original, not a regression introduced here (see "Known
  pre-existing issue" below).
- `single_step_demo.py`, `multistep_unrefined_demo.py`: new, modeled on
  `unicycle/single_step_demo.py` / `multistep_unrefined_demo.py`'s shape
  (compile-vs-run timing via `time_jit`, pairwise-overlap + volume report).
  **There is no `multistep_refined_demo.py`**: unlike unicycle/admire,
  `go2_separating_input_immrax.py` has no third, intersection-refinement
  algorithmic layer (no `propagate_with_refinement` / `collect_refinement_history`
  / `refined_overlap_loss` equivalent exists in the original module) --
  only single-step and multistep-unrefined. Nothing was cut to reach this;
  the original module simply never had that layer.
- `plot_baseline_intervals.py`: ported from `examples/go2/plot_baseline_intervals.py`
  (itself already a standalone script, not notebook-only) onto this
  folder's core -- bar chart of overlap losses + interval-box grid for a
  few candidate control inputs.
- `plot_position_intervals_3d.py`: new. The 3 PDFs referenced by the
  original notebooks (`go2_separating_input_immrax_sets/history/comparison.pdf`)
  are notebook-inline, not produced by any standalone script (only
  `comparison`/`baseline_intervals` had a script). This new script fills
  that gap with a 3-D (px, time, py) interval-history plot over a multistep
  horizon, styled like `unicycle/plot_unrefined_four_panel.py` /
  `admire/plot_refinement_3d.py`. It reuses `go2_separating_input.py`'s own
  `_propagate_history`/`position_interval` rather than re-deriving
  propagation locally, so it cannot numerically drift from the optimized
  loss (the failure mode documented as unicycle's PLAN.md bug #3).

## Runtime-optimization changes

- Every `jax.lax.fori_loop` in the ported code is replaced by `jax.lax.scan`
  (`_scan_loop` for the outer GD restart loops; direct `scan` for
  `propagate_scenario`'s per-step loop and `_propagate_history`'s
  per-segment inner loop) -- scan has lower per-iteration dispatch overhead
  than fori_loop, the same finding already applied in
  `nonlinear_chain_separating_input.py` / `unicycle/car_separating_input.py`.
  `jax.checkpoint` gradient-memory wrapping is preserved exactly around each
  converted body.
- **Verified these scan conversions are numerically inert**: forward
  propagation (`_propagate_history`) and single-instance gradients
  (`jax.grad(separation_loss_multistep)`) are bit-identical to the
  original's fori_loop version for the same inputs. Confirmed by swapping
  each conversion back to fori_loop independently and re-running the
  50-restart/150-iteration multistep demo config -- NaN count was unchanged
  (13/50) either way, proving the loop mechanics were not the source of the
  divergence described next.

## Bug found and fixed while porting

Initially, `overlap_size_lax` (the original's `lax.cond`-branching pairwise
overlap volume, imported from `examples/faulty_car/interval_functions.py`)
was replaced with the branchless clip-and-product `_overlap_volume` used by
`unicycle`/`admire`/`nonlinear_chain`/`quadrotor_fault_diagnosis`, to match
this project's usual convention and drop the `examples/faulty_car`
dependency. **This introduced a real regression**: at this module's default
multistep demo config (50 restarts x 150 GD iterations), 13/50 restarts'
gradients went NaN mid-optimization, vs. 0/50 with the original's
`overlap_size_lax`. Isolated empirically (holding everything else fixed,
including the scan conversions above) to the overlap-volume formula itself:
`lax.cond` never differentiates through the `jnp.prod(upper-lower)` branch
when the boxes are actually disjoint (the `no_overlap` branch just returns a
constant `0.0`), whereas the branchless `jnp.maximum(diff, 0.0)`-then-`prod`
form always sends a gradient through `jnp.prod` even when several axes are
clipped to exactly 0 -- for this module's dynamics/embedding that back-
propagates to NaN through a meaningful fraction of restarts (the forward
LOSS value is identical either way; only the gradient differs).

**Fix**: `_overlap_volume` in this folder's `go2_separating_input.py` is the
original's `overlap_size_lax` logic, inlined (so there is still no
dependency on `examples/faulty_car`) rather than the branchless form used
elsewhere. After the fix, the 50-restart/150-iteration demo config
reproduces the original's result bit-for-bit (loss `0.04477577283978462`,
0/50 NaN restarts).

## Known pre-existing issue (not introduced here, not fixed)

`TestGo2SensorFaultSystem::test_noise_parameter_shifts_lateral` and
`::test_yaw_rate_unaffected_by_noise` fail identically in both this folder
and the original `examples/go2/go2_separating_input_immrax.py` -- confirmed
by running the original's own test file directly. Both tests call
`self.sys.f(0., x, u, jnp.array([<scalar>]))` with a 1-element `p`, but
`Go2SensorFaultSystem.f` reads `p[0]`, `p[1]`, `p[2]` (vy_noise, alpha,
beta); JAX's default silent out-of-bounds index clamping makes `p[1]`/`p[2]`
both alias `p[0]`, so `beta` ends up equal to `vy_noise` and the yaw rate
picks up a spurious noise dependence the test assumes doesn't exist. This
predates this folder (stale relative to whenever `Go2SensorFaultSystem`
gained its 3rd parameter) and is out of scope to fix here -- left as-is to
avoid guessing at the original module's intended fix.

## File layout

```
examples/go2_fault_diagnosis/
  PLAN.md                          (this file)
  go2_separating_input.py          core module (systems, scenarios, 2 layers)
  single_step_demo.py              script 1: single-step separating input
  multistep_unrefined_demo.py      script 2: multistep, no refinement
  plot_baseline_intervals.py       static plot: overlap bar chart + interval boxes
  plot_position_intervals_3d.py    static plot: 3-D interval history over time
  tests/
    test_go2_separating_input.py
```
