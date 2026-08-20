# Plan: Clean Faulty-Car Separating Input

Companion/replacement for `examples/faulty_car/faulty_car_separating_input.py`,
built on the same immrax natural-embedding separating-input machinery, in a
dedicated folder with three standalone demo scripts (mirroring
`examples/integrator_chain/`'s `single_step_demo.py` /
`multistep_demo.py` / `refinement_demo.py` split) plus one animation script.

## Why a new folder

`examples/faulty_car/` accreted ~30 files (~10k lines) around one clean core
module: 7 near-duplicate animation scripts (~4000 lines, 60-90% copy-pasted),
5 overlapping static-plot scripts, a dead/broken 4-state car model, a
bug-riddled closed-loop CBF module, and several broken/scratch scripts. This
folder is a from-scratch, minimal rewrite of just the separating-input core
(system + 3 algorithmic layers) plus one animation script — no CBF, no
closed-loop/output-feedback controller, no legacy 4-state model.

## Bugs found in the original (verified by direct read), fixed here

1. **Sensor Fault was never actually separable in the single-step or
   multistep-unrefined losses.** `faulty_car_separating_input.py`'s
   `Scenario.obs_offset`/`obs_scale` are defined and the Sensor-Fault
   scenario sets them to `[0.2,0.2]`/`[0.95]`, but `separation_loss` (line
   239) and `separation_loss_multistep` (line 273) both compare **raw
   state position**, never referencing `obs_offset`/`obs_scale`. Since
   Sensor Fault's `p_interval` (alpha=1) is identical to Nominal's, its raw
   propagated position is *always* identical to Nominal's too, making
   `overlap(Nominal, SensorFault)` an irreducible constant no choice of `u`
   could shrink. Only the refined layer applied the observation model.
   **Fix**: all three layers here compute overlap on `observed_output(...)`,
   not raw state.
2. **Wrong-scenario `obs_scale` used when inverting an observation**, in
   `propagate_with_refinement`'s step body: both `x_ref_i` and `x_ref_j`
   divided by `scenarios[j].obs_scale[0]` — `x_ref_i` should divide by
   `scenarios[i].obs_scale[0]`. Invisible only because every
   non-Sensor-Fault scenario has `obs_scale=1`. **Fix**:
   `_invert_observation(y, scenario, ...)` always uses its own `scenario`'s
   `obs_scale`.
3. **The original animation's refinement math had silently drifted from
   the optimizer's.** `animate_refinement_3d.py`'s
   `collect_refinement_history` independently re-derived the per-pair
   refine-then-propagate step instead of calling
   `propagate_with_refinement`, and its inversion didn't divide by
   `obs_scale` **at all** (only subtracted `obs_offset`) — a second,
   different bug from #2, meaning the visualized animation and the
   optimized loss were computing two different things. **Fix**: one
   `_refine_and_step_pair(...)` helper is used by both the jittable loss
   (`propagate_with_refinement`) and the animation's history collector
   (`collect_refinement_history`, defined once, here, and imported by the
   animation script) — they cannot drift apart again.
4. **No-overlap fallback state was ill-defined.** The original's no-overlap
   branch computed a "safe" `y` from `x_curr_i`'s own center alone (not a
   real intersection), then inverted *both* scenarios' next state from
   that single bogus value, corrupting the carried-forward interval used
   by later refinement steps (the animation copied this same pattern).
   **Fix**: matches the already-proven pattern from this repo's
   `nonlinear_chain`/`quadrotor_fault_diagnosis` modules — on no-overlap,
   `x_ref_i`/`x_ref_j` fall back to the **unchanged** `x_curr_i`/`x_curr_j`
   via `jnp.where` gating the *state* directly, not a derived fallback `y`.
5. Six near-duplicate `overlap_size*` functions in `interval_functions.py`
   (branch-based, via `lax.cond`) are replaced by the branchless
   clip-and-product `_overlap_volume` already used in
   `nonlinear_chain_separating_input.py` / `quadrotor_separating_input.py`
   — no dependency on `interval_functions.py` at all.

## System (unchanged from the original)

```
State   x = [px, py, phi]   (2-D position + heading)
Control u = [v, omega]       (forward velocity + steering rate), box [-1,1]^2
Params  p = [alpha]           (steering effectiveness; alpha=1 -> nominal)

Dynamics:
    px_dot = v * cos(phi)
    py_dot = v * sin(phi)
    phi_dot = alpha * omega
```

Dynamics are cheap (plain `sin`/`cos` of *state*, no division, no `tan`) --
no special compile-cost handling needed, unlike quadrotor_fault_diagnosis's
trig-in-denominator problem. Plain `jax.lax.fori_loop`/`jax.lax.scan`
throughout.

## Scenarios (3 total, fixed -- unchanged from the original)

```
Nominal          alpha=1,           y = [px, py]
Actuator Fault   alpha in [0,0.5],  y = [px, py]
Sensor Fault     alpha=1,           y = 0.95*[px, py] + [0.2, 0.2]
```

## File layout

```
examples/unicycle/
  PLAN.md                          (this file)
  car_separating_input.py          core module (system, scenarios, all 3 layers)
  single_step_demo.py              script 1: single-step separating input
  multistep_unrefined_demo.py      script 2: multistep, no refinement
  multistep_refined_demo.py        script 3: multistep, intersection-refinement
  animate_refinement_3d.py         animation, reproduces the original's output
  tests/
    test_car_separating_input.py
```
