# Tasks: Decoupled Cubic-Drift Chain

Execution checklist for `PLAN.md`. Ordered so each task's output is usable
by the next (system -> scenarios -> single-step -> multistep -> refinement
-> tests). Mirrors the section structure of
`examples/integrator_chain/integrator_separating_input.py`.

## 0. Scaffolding
- [ ] Create `examples/nonlinear_chain/nonlinear_chain_separating_input.py`
      with the same sys.path setup as integrator_chain
      (`parents[1]` -> `examples/`, for parity even if unused initially).
- [ ] Create `examples/nonlinear_chain/tests/` directory.

## 1. System definition
- [ ] Implement `CubicChainSystem(irx.System)` per PLAN.md §2:
      `f(t, x, u, p) = a*x - b*x**3 + p*u`, `evolution='continuous'`,
      `xlen=N`, with `a`, `b` stored as instance arrays.
- [ ] Decide and implement `a`, `b` provenance (PLAN.md §6 recommends
      `jnp.linspace(a_lo, a_hi, N)` / `jnp.linspace(b_lo, b_hi, N)` for
      determinism) as a helper, e.g. `default_channel_params(N, a_lo=0.5,
      a_hi=1.5, b_lo=0.2, b_hi=0.8)`.
- [ ] Implement `get_system_and_embedding(N, a, b)`, cached by
      `(N, tuple(a), tuple(b))` (extends integrator_chain's cache-by-`N`
      since `a`/`b` are now part of the traced system).
- [ ] Sanity check: build the embedding for one small `N` (e.g. 3) and
      confirm `irx.natemb(...)` succeeds and `emb.f(...)` runs on an
      interval input without shape errors, before building anything on
      top of it.

## 2. Scenarios & output model
- [ ] Port `observed_output` / `_invert_observation` from integrator_chain
      unchanged (already elementwise over `N`).
- [ ] Implement `Scenario` dataclass (PLAN.md §4).
- [ ] Implement `create_scenarios(N, a, b, actuator_alpha_lo,
      actuator_alpha_hi, sensor_beta_lo, sensor_beta_hi, sensor_xi_bound)`
      producing `Nominal + N ActuatorFault_i + N SensorFault_i` (2N+1
      total), per the code sketch in PLAN.md §4.
- [ ] Assert `beta.lower > 0` for every scenario at construction time (same
      guard as integrator_chain, needed for refinement-layer invertibility).
- [ ] Unit-check (can fold into pytest, see §6): for `ActuatorFault_i`,
      confirm `p_interval.lower[j] == p_interval.upper[j] == 1.0` for all
      `j != i`, and only index `i` carries the fault interval; symmetric
      check for `SensorFault_i` on `beta`/`xi`.

## 3. Single-step separating input
- [ ] Port `euler_step`, `_propagate_by_params`, `propagate_scenario`,
      `_propagate_all_scenarios` — update control box constant to
      `_U_LO`/`_U_HI` shape `(N,)` (was `(1,)`).
- [ ] Port `separation_loss` (pairwise overlap sum over all `2N+1`
      scenarios) and `_overlap_volume` (verbatim, generic already).
- [ ] Port `SeparatingInputOptimizer`, `optimize_parallel_gpu`,
      `optimize_parallel_gpu_rejit`, updating `u_init`/`u0` sampling shape
      from `(1,)`/`(num_restarts, 1)` to `(N,)`/`(num_restarts, N)`.
- [ ] Smoke test at small `N` (e.g. `N=3`, so 7 scenarios / 21 pairs):
      run `optimize_parallel_gpu` for a handful of iterations, confirm loss
      decreases and no NaNs (cubic drift can blow up faster than the linear
      integrator chain — watch for interval blowup at large `dt`/`num_steps`
      and tune the demo's `dt` down if needed).

## 4. Multistep (unrefined)
- [ ] Port `_propagate_history`, `propagate_scenario_multistep`,
      `separation_loss_multistep`, `MultistepSequenceOptimizer`,
      `optimize_multistep_gpu`, `optimize_multistep_gpu_rejit`,
      `optimize_multistep` — same `(N,)`-shape updates as §3, `u_seq` shape
      `(num_segments, N)`.

## 5. Multistep (intersection-refinement)
- [ ] Port `propagate_with_refinement`, `refined_overlap_loss`,
      `optimize_refined_gpu` — verify `_invert_observation`/`ivl_to_arr`/
      `arr_to_ivl` need no changes beyond `N` (they're already generic).
- [ ] Confirm the per-pair carry shape `(n_pairs, 2N)` is sane at the
      target demo `N` — flag if `n_pairs = C(2N+1,2)` makes this loop's
      compile time unreasonable (see PLAN.md §6 O(N^2) risk) and consider
      restricting refinement to a curated pair subset if so.

## 6. Tests
- [ ] `examples/nonlinear_chain/tests/test_nonlinear_chain_separating_input.py`,
      mirroring `examples/integrator_chain/tests/test_integrator_separating_input.py`'s
      coverage:
  - System/embedding construction for a couple of `N` values.
  - Scenario construction: correct count (`2N+1`), correct per-channel
    fault isolation (§2 checks above), `beta>0` guard raises on bad input.
  - `euler_step`/`propagate_scenario` shape and basic sanity (e.g. nominal
    trajectory from `x0=0, u=0` stays at 0, since `f(0)=0` and `g(0)=0`).
  - `separation_loss` single-step: decreases (or is near-minimal) for a
    hand-picked separating `u`; is symmetric/well-defined for `u=0`.
  - Multistep unrefined path: shapes, at least one optimize-then-evaluate
    smoke test at small `N`/`num_restarts`/`num_iters`.
  - Refinement path: shapes, at least one smoke test comparing refined vs.
    unrefined loss on the same scenario set (refined should be `<=`
    unrefined, matching integrator_chain's invariant if it asserts that).
  - Run via `JAX_PLATFORMS=cpu /home/user/immrax-venv/bin/python -m pytest`
    per project convention.

## 7. Demo (optional, follow-up)
- [ ] Only after the above passes: consider a
      `nonlinear_chain_separating_input_demo.ipynb` or a `__main__` block,
      following the go2/admire/faulty_car precedent of one demo notebook
      per system — not started as part of this task list, listed here so
      it isn't forgotten.

## 8. Memory
- [ ] After implementation lands, update
      `/home/user/.claude/projects/-home-user-output-feedback-nhholyap/memory/`
      (Key Files section) with the new module path and its scenario-count
      formula (`2N+1`), the way integrator_chain's per-state sensor model
      is already documented there.
