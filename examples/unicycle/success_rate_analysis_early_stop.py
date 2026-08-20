"""
Success-rate analysis for unicycle, using early-stopping GD
(car_separating_input.gd_early_stop) instead of success_rate_analysis.py's
fixed-NUM_ITERS jax.lax.scan loop.

This is a SEPARATE script, not a modification of success_rate_analysis.py --
it imports that module's config grid, scenario construction, and
history/CSV-row/npz-saving helpers UNCHANGED (duplicating that logic here
would risk exactly the kind of "two implementations silently drift apart"
bug this codebase has already hit and fixed once, see PLAN.md bug #3). The
ONLY thing this script changes is the GD optimizer core:

  success_rate_analysis.run_batched_sweep : jax.lax.scan, exactly NUM_ITERS
    steps every time, regardless of whether the loss already hit 0.
  run_batched_sweep_early_stop (here)     : car_separating_input.gd_early_stop,
    jax.lax.while_loop, MAX_ITERS is an UPPER BOUND -- each (config, restart)
    instance stops the moment ITS OWN loss hits exactly 0.

Same double-vmap-then-one-jit structure and the same constant-control
restart seeding (for the two multistep methods -- see success_rate_analysis's
full_sweep for why) are kept, so the two scripts are apples-to-apples: any
difference in success rate between this script's output and
success_rate_summary.csv isolates the effect of the stopping condition
itself.

Why this can change results, not just save compute
----------------------------------------------------
The pairwise overlap loss is `jnp.maximum(..., 0.0)`-clipped -- it has a
kink exactly at the loss==0 boundary. A fixed-iteration loop has no way to
know it already found a perfect (loss==0) solution and keeps applying
gradient updates for all NUM_ITERS steps regardless; if the (sub)gradient
at that kink is nonzero, those extra updates can push u back into positive-
overlap territory, so the ORIGINAL script's reported "final" loss for a
restart can be WORSE than a loss that restart actually passed through
mid-optimization. gd_early_stop can't regress this way -- it returns the
state at the exact iteration loss first hit 0.

Caveat (see gd_early_stop's docstring in car_separating_input.py): JAX's
vmap batching rule for lax.while_loop keeps the WHOLE vmapped batch's loop
running until EVERY lane's condition is false, so a batch containing any
never-converges-to-exactly-0 instance (e.g. an infeasible wide-x0_width
config) still costs MAX_ITERS iterations of wall-clock time -- this script
is not expected to be faster than success_rate_analysis.py, only to
(potentially) find different/better final losses per instance, plus it
records each winning restart's actual convergence-iteration count for free.

Output: success_rate_summary_early_stop.csv and
success_rate_data_early_stop_<method>.npz (adds an `iters_used` array vs.
the original's data files) in this directory.
"""

import csv
import sys
import time
from functools import partial
from pathlib import Path
from typing import List, Tuple

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import numpy as np
import jax
import jax.numpy as jnp
import immrax as irx

from car_separating_input import gd_early_stop, time_jit, _project_u

import success_rate_analysis as sra
from success_rate_analysis import (
    X0_CENTERS, X0_WIDTHS, ACTUATOR_RANGES, SENSOR_SEVERITIES,
    SUCCESS_THRESHOLD, DT, SINGLE_STEP_NUM_STEPS,
    UNREFINED_STEPS_PER_SEGMENT, UNREFINED_NUM_SEGMENTS, REFINED_NUM_STEPS,
    NUM_RESTARTS, LEARNING_RATE, SEED,
    build_config_grid, _config_loss, _as_u_seq,
    batched_history, refined_history_per_config, _METHOD_SPECS,
)

# Deliberately larger than success_rate_analysis.py's NUM_ITERS=30 -- unlike
# a fixed jax.lax.scan, gd_early_stop only pays for the extra ceiling on
# instances that actually need it (an already-converged instance's loop
# freezes as soon as it hits 0, per gd_early_stop's docstring), so raising
# this mainly costs wall-clock time on configs that are still improving
# past iteration 30, not on the ones that already succeed quickly.
MAX_ITERS = 200

OUT_CSV = _HERE / "success_rate_summary_early_stop.csv"


# ══════════════════════════════════════════════════════════════════════════════
# Double-vmapped (config x restart) batched sweep optimizer, early-stop GD core
# ══════════════════════════════════════════════════════════════════════════════

def run_batched_sweep_early_stop(loss_kind: str, u_shape_per_restart: Tuple[int, ...],
                                 config_arrays, num_restarts: int, max_iters: int,
                                 learning_rate: float, seed: int, dt: float, **loss_kwargs):
    """Same contract as success_rate_analysis.run_batched_sweep, but the GD
    core is gd_early_stop instead of a fixed-length jax.lax.scan.

    Returns (jitted_fn, u_final, losses_final, iters_final, compile_t, run_t, mem):
    u_final has shape (num_configs, num_restarts, *u_shape_per_restart);
    losses_final and iters_final have shape (num_configs, num_restarts) --
    iters_final is gd_early_stop's per-instance convergence step count
    (== max_iters for instances that never hit loss==0).
    """
    x0_center, x0_width, alpha_lo, alpha_hi, sensor_offset, sensor_scale = config_arrays
    num_configs = x0_center.shape[0]

    _loss = partial(_config_loss, loss_kind=loss_kind, dt=dt, **loss_kwargs)

    def optimize_one(u0, x0_center, x0_width, alpha_lo, alpha_hi, sensor_offset, sensor_scale):
        loss_fn = lambda u: _loss(u, x0_center, x0_width, alpha_lo, alpha_hi, sensor_offset, sensor_scale)
        return gd_early_stop(loss_fn, u0, learning_rate, max_iters)

    # inner: batch over restarts for ONE fixed config (config args: in_axes=None)
    inner = jax.vmap(optimize_one, in_axes=(0, None, None, None, None, None, None))
    # outer: batch over configs (every arg, including u0, has a leading config axis)
    batched = jax.vmap(inner, in_axes=(0, 0, 0, 0, 0, 0, 0))

    def full_sweep(seed_val):
        key = jax.random.PRNGKey(seed_val)

        if len(u_shape_per_restart) == 2:
            # Multistep methods: same constant-control restart seeding as
            # success_rate_analysis.full_sweep (see that function's comment
            # for why) -- kept here so the two scripts differ ONLY in the
            # GD stopping condition, not in what restarts are tried.
            key_indep, key_const = jax.random.split(key)
            num_steps_dim = u_shape_per_restart[0]
            u0_indep = (jax.random.normal(key_indep, (num_configs, num_restarts) + u_shape_per_restart) * 0.3
                        + jnp.array([0.5, 0.3]))
            u0_const_base = (jax.random.normal(key_const, (num_configs, num_restarts, 2)) * 0.3
                              + jnp.array([0.5, 0.3]))
            u0_const = jnp.broadcast_to(
                u0_const_base[:, :, None, :], (num_configs, num_restarts, num_steps_dim, 2)
            )
            is_const_restart = jnp.arange(num_restarts) < (num_restarts // 2)
            u0 = jnp.where(is_const_restart[None, :, None, None], u0_const, u0_indep)
        else:
            u0 = (jax.random.normal(key, (num_configs, num_restarts) + u_shape_per_restart) * 0.3
                  + jnp.array([0.5, 0.3]))

        u_final, losses_final, iters_final = batched(
            u0, x0_center, x0_width, alpha_lo, alpha_hi, sensor_offset, sensor_scale
        )
        return u_final, losses_final, iters_final

    jitted_fn, compile_t, run_t, mem = time_jit(full_sweep, seed)
    u_final, losses_final, iters_final = jitted_fn(seed)
    return jitted_fn, u_final, losses_final, iters_final, compile_t, run_t, mem


# ══════════════════════════════════════════════════════════════════════════════
# Per-method driver
# ══════════════════════════════════════════════════════════════════════════════

def run_method_early_stop(name: str, loss_kind: str, u_shape_per_restart, loss_kwargs,
                          config_arrays, combos):
    print(f"\n{'=' * 78}\n{name} (early-stop GD)\n{'=' * 78}")

    jitted_fn, u_final, losses_final, iters_final, compile_t, run_t, mem = run_batched_sweep_early_stop(
        loss_kind, u_shape_per_restart, config_arrays,
        num_restarts=NUM_RESTARTS, max_iters=MAX_ITERS, learning_rate=LEARNING_RATE,
        seed=SEED, dt=DT, **loss_kwargs,
    )
    num_configs = losses_final.shape[0]

    best_idx = jnp.argmin(losses_final, axis=1)                        # (num_configs,)
    best_loss = losses_final[jnp.arange(num_configs), best_idx]         # (num_configs,)
    best_u = u_final[jnp.arange(num_configs), best_idx]                  # (num_configs, *u_shape)
    best_iters = iters_final[jnp.arange(num_configs), best_idx]          # (num_configs,)
    restart_success_rate = jnp.mean(losses_final < SUCCESS_THRESHOLD, axis=1)   # (num_configs,)
    config_success = best_loss < SUCCESS_THRESHOLD

    print(f"compile {compile_t * 1e3:8.2f} ms total ({compile_t / num_configs * 1e3:7.3f} ms/config)   "
          f"run {run_t * 1e3:7.3f} ms total ({run_t / num_configs * 1e3:7.3f} ms/config)   mem {mem}")
    print(f"Config success rate: {int(jnp.sum(config_success))}/{num_configs} "
          f"({100 * float(jnp.mean(config_success)):.1f}%)")
    frac_early = float(jnp.mean(iters_final < MAX_ITERS))
    print(f"Restart-level early-stop rate: {100 * frac_early:.1f}% of all "
          f"(config x restart) instances stopped before MAX_ITERS={MAX_ITERS} "
          f"(mean iters used = {float(jnp.mean(iters_final)):.2f}, "
          f"winning-restart mean iters used = {float(jnp.mean(best_iters)):.2f})")

    u_seq_for_history = _as_u_seq(best_u, loss_kind)
    if loss_kind == "multistep_refined":
        obs_i_lo, obs_i_hi, obs_j_lo, obs_j_hi, has_overlap, pairs = refined_history_per_config(
            config_arrays, u_seq_for_history, DT, REFINED_NUM_STEPS
        )
        history_payload = dict(obs_i_lo=obs_i_lo, obs_i_hi=obs_i_hi,
                               obs_j_lo=obs_j_lo, obs_j_hi=obs_j_hi,
                               has_overlap=has_overlap, pairs=np.array(pairs))
    else:
        steps_per_segment = 1 if loss_kind == "single_step" else UNREFINED_STEPS_PER_SEGMENT
        y_lower, y_upper = batched_history(config_arrays, u_seq_for_history, DT, steps_per_segment)
        history_payload = dict(y_lower=y_lower, y_upper=y_upper)

    rows = []
    for c, combo in enumerate(combos):
        (px, py, phi), width, (a_lo, a_hi), (offset, scale) = combo
        rows.append({
            'method': name, 'config_idx': c,
            'x0_px': px, 'x0_py': py, 'x0_phi': phi, 'x0_width': width,
            'alpha_lo': a_lo, 'alpha_hi': a_hi,
            'sensor_offset': offset[0], 'sensor_scale': scale,
            'success': bool(config_success[c]), 'final_loss': float(best_loss[c]),
            'restart_success_rate': float(restart_success_rate[c]),
            'iters_used': int(best_iters[c]),
        })

    npz_path = _HERE / f"success_rate_data_early_stop_{loss_kind}.npz"
    np.savez(
        npz_path,
        u_opt=np.array(best_u), losses_final_all_restarts=np.array(losses_final),
        iters_final_all_restarts=np.array(iters_final),
        config_success=np.array(config_success), restart_success_rate=np.array(restart_success_rate),
        x0_center=np.array(config_arrays[0]), x0_width=np.array(config_arrays[1]),
        alpha_lo=np.array(config_arrays[2]), alpha_hi=np.array(config_arrays[3]),
        sensor_offset=np.array(config_arrays[4]), sensor_scale=np.array(config_arrays[5]),
        compile_time_s=compile_t, run_time_s=run_t,
        **history_payload,
    )
    print(f"Saved full data -> {npz_path}")

    return rows, {'compile_time_s': compile_t, 'run_time_s': run_t, 'mem': mem,
                  'success_rate': float(jnp.mean(config_success)),
                  'mean_iters_used': float(jnp.mean(iters_final))}


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def _load_baseline_success(csv_path: Path):
    """Best-effort load of success_rate_analysis.py's fixed-iteration output
    for a side-by-side comparison. Returns {(method, config_idx): success_bool}
    or None if the baseline CSV isn't there."""
    if not csv_path.exists():
        return None
    with open(csv_path) as f:
        return {(r['method'], int(r['config_idx'])): r['success'] == 'True'
                for r in csv.DictReader(f)}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method", choices=list(_METHOD_SPECS) + ["all"], default="all",
        help="Run only this method; writes success_rate_summary_early_stop_<method>.csv "
             "and reports isolated compile/run/memory numbers for that method alone.",
    )
    args = parser.parse_args()

    print("Devices:", jax.devices())
    combos, config_arrays = build_config_grid()
    num_configs = len(combos)
    print(f"Config grid (imported from success_rate_analysis.py): {len(X0_CENTERS)} x0_centers x "
          f"{len(X0_WIDTHS)} x0_widths x {len(ACTUATOR_RANGES)} actuator_ranges x "
          f"{len(SENSOR_SEVERITIES)} sensor_severities = {num_configs} configs")
    print(f"Per method: {NUM_RESTARTS} restarts x UP TO {MAX_ITERS} GD iters (early-stop "
          f"once loss hits exactly 0), success threshold = {SUCCESS_THRESHOLD} m^2")

    methods_to_run = list(_METHOD_SPECS) if args.method == "all" else [args.method]

    all_rows = []
    summaries = {}
    for loss_kind in methods_to_run:
        display_name, u_shape, kwargs = _METHOD_SPECS[loss_kind]
        rows, summary = run_method_early_stop(display_name, loss_kind, u_shape, kwargs, config_arrays, combos)
        all_rows += rows
        summaries[loss_kind] = summary

    out_csv = OUT_CSV if args.method == "all" else _HERE / f"success_rate_summary_early_stop_{args.method}.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nWrote summary CSV -> {out_csv}")

    print(f"\n{'=' * 78}\nSUMMARY ({'isolated, single-process run' if args.method != 'all' else 'all methods, one process -- see --method for isolated numbers'})\n{'=' * 78}")
    print(f"  (compile/run reported per-config, i.e. total / {num_configs} configs)")
    for method, s in summaries.items():
        print(f"  {method:22s}: success {100*s['success_rate']:5.1f}%   "
              f"compile {s['compile_time_s']/num_configs*1e3:7.3f} ms/config   "
              f"run {s['run_time_s']/num_configs*1e3:7.3f} ms/config   "
              f"mean iters used {s['mean_iters_used']:5.2f}/{MAX_ITERS}   "
              f"mem {s['mem']}")

    baseline = _load_baseline_success(sra.OUT_CSV)
    if baseline is not None and args.method == "all":
        print(f"\n{'=' * 78}\nvs. FIXED-ITERATION baseline ({sra.OUT_CSV.name})\n{'=' * 78}")
        for loss_kind in methods_to_run:
            display_name = _METHOD_SPECS[loss_kind][0]
            method_rows = [r for r in all_rows if r['method'] == display_name]
            flips_to_success, flips_to_failure, matched = 0, 0, 0
            for r in method_rows:
                key = (display_name, r['config_idx'])
                if key not in baseline:
                    continue
                base_success = baseline[key]
                if r['success'] and not base_success:
                    flips_to_success += 1
                elif not r['success'] and base_success:
                    flips_to_failure += 1
                else:
                    matched += 1
            print(f"  {display_name:22s}: {matched} unchanged, "
                  f"{flips_to_success} newly succeed under early-stop, "
                  f"{flips_to_failure} newly fail under early-stop")
    elif args.method == "all":
        print(f"\n(No baseline found at {sra.OUT_CSV} -- run success_rate_analysis.py "
              f"first for a side-by-side comparison.)")


if __name__ == "__main__":
    main()
