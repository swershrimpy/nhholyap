"""
Success-rate analysis for ADMIRE actuator-fault diagnosis, mirroring
car_fault_diagnosis/success_rate_analysis.py's double-vmapped
(config x restart), one-jit-per-method sweep design.

Fault modes: UNCHANGED from the folder's existing setup -- all 11 scenarios
from admire_separating_input.create_scenarios() (Nominal + one complete-loss
fault per one of the 10 control surfaces). Nothing here adds, removes, or
weakens any fault mode.

Swept axes
----------
  input_limit : [0.1, 0.05, 0.025] rad -- symmetric control box +-limit on
                 all 10 surfaces. 0.05 is admire_separating_input.py's
                 existing hardcoded _U_LO/_U_HI; 0.1/0.025 are 2x wider/
                 narrower.
  horizon     : [0.5, 1.0, 2.0] s, at the folder's existing dt=0.1
                 (admire_separating_input.py's __main__ example) -> 5, 10,
                 20 Euler steps. dt itself is left UNCHANGED. Chosen to stay
                 within (or barely past) horizons ALREADY exercised
                 elsewhere in this folder, not extrapolated: single-step's
                 own __main__ demo already runs num_steps=20 (2.0s);
                 admire_refined_sequence_optimizer.py's default/__main__
                 runs num_steps=5 (0.5s). See the resource note below --
                 this replaces an earlier draft that swept up to 10s
                 (100 steps), which had no precedent in this codebase at all.
  x0_width    : [0.005, 0.01, 0.05] -- icentpert half-width applied
                 uniformly to all 9 states around the folder's existing
                 nominal center (Vt=343*0.3, all else 0). 0.01 matches
                 admire_separating_input.py's __main__ example exactly;
                 0.005/0.05 are 2x narrower / 5x wider.

3 x 3 x 3 = 27 configs per method, all 3 methods (single-step, multistep
unrefined, multistep refined) reusing admire_separating_input.py's and
admire_refined_sequence_optimizer.py's existing loss/propagation functions
unchanged. multistep-unrefined uses steps_per_segment=1 (one control
decision per Euler step, same resolution as multistep-refined), so at
these horizons its u_seq shape matches refined's exactly: (num_steps, 10).

WHY THIS IS ORGANIZED AS ONE JIT PER (METHOD, HORIZON), NOT PER METHOD
-----------------------------------------------------------------------
car_fault_diagnosis's script keeps horizon (dt, num_steps) STATIC across
its whole sweep, so every config fits in one jax.jit via double-vmap. This
sweep varies horizon itself, and num_steps is a Python-level loop-trip
count (jax.lax.scan/fori_loop length), not a traceable value -- it cannot
be vmapped over. So each horizon needs its own jit per method: 3 horizons
x 3 methods = 9 total jax.jit compiles, each double-vmapping the
remaining 9 (input_limit x x0_width) configs x NUM_RESTARTS restarts.

Resource note (see the earlier resource estimate for the fuller writeup)
--------------------------------------------------------------------------
AdmireNineDoFLinAct.f (admire.py) divides FIVE times per Euler step
(Vt_der, alpha_der, beta_der, psi_der all divide; phi_der uses tan(theta)),
the same category of operation that made quadrotor_fault_diagnosis's
gradient compile go from ~2s/1 step to ~56s/4.3GB at 5 steps (project
memory) -- so this is not risk-free. But unlike that first draft (10s /
100 steps, with no precedent anywhere in this codebase), every horizon
swept here is within 4x of a horizon this exact ADMIRE refined loss has
already been run at (5 steps), and single-step's 20-step case is EXACTLY
what admire_separating_input.py's own demo already runs. single-step and
multistep-unrefined still have no jax.checkpoint (fori_loop/scan,
admire_separating_input.py); only multistep-refined does
(admire_refined_sequence_optimizer.py). --confirm-expensive is still
required above 2.0s, as a guard for anyone who edits HORIZONS_S upward
later -- it does not trigger for the current axis.

Output: admire_success_rate_summary.csv and
admire_success_rate_data_<method>_<horizon>s.npz in this directory.
"""

import csv
import sys
import time
import resource
from pathlib import Path
from functools import partial
from typing import Tuple, Dict

_HERE = Path(__file__).resolve().parent
_EXAMPLES_DIR = _HERE.parent
for _p in (str(_HERE), str(_EXAMPLES_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import jax
import jax.numpy as jnp
import immrax as irx

from admire_separating_input import (
    create_scenarios, separation_loss, separation_loss_multistep,
)
from admire_refined_sequence_optimizer import propagate_with_refinement_admire

# ══════════════════════════════════════════════════════════════════════════════
# Sweep axes
# ══════════════════════════════════════════════════════════════════════════════

INPUT_LIMITS = [0.1, 0.05, 0.025]      # rad, symmetric box on all 10 surfaces
HORIZONS_S = [0.5, 1.0, 2.0]           # seconds -- see module docstring for the precedent behind these
DT = 0.1                               # unchanged from the folder's existing default
NUM_STEPS_FOR_HORIZON = {0.5: 5, 1.0: 10, 2.0: 20}   # horizon / DT, exact integers

# multistep-unrefined: one control decision per Euler step (steps_per_segment=1),
# same resolution as multistep-refined -- at these short horizons (max 20 steps,
# 200 free params/restart) there's no need for the coarser segment spacing a
# longer sweep would want; keeping both multistep methods at the same
# resolution also makes them directly comparable.
STEPS_PER_SEGMENT = 1
NUM_SEGMENTS_FOR_HORIZON = NUM_STEPS_FOR_HORIZON   # segment == step

X0_WIDTHS = [0.005, 0.01, 0.05]        # narrower / folder-default / wider
X0_NOM = jnp.zeros(9).at[0].set(343.0 * 0.3)   # matches admire_separating_input.py's __main__

SUCCESS_THRESHOLD = 1e-6   # rad^2 (refined) / rad^3 (single-step, multistep -- see overlap_size_log); loss below this = "fully separated"

NUM_RESTARTS = 20     # smaller than car_fault_diagnosis's 100: 10-D control
NUM_ITERS = 100        # (up to 200 free params/restart at the 2.0s horizon)
LEARNING_RATE = 0.05   # and higher per-restart cost -- see resource estimate
SEED = 42

SCENARIOS = create_scenarios()   # 11 scenarios: Nominal + 10 total-loss faults (UNCHANGED)
N_SCENARIOS = len(SCENARIOS)

def _out_csv_for_method(method: str) -> Path:
    """Method-specific summary CSV path.

    Needed so that running the three methods as separate concurrent
    processes (e.g. one PACE job per method) doesn't have them race on
    writing the same file -- each method's own run only ever touches its
    own CSV. `--method all` keeps the original shared filename."""
    suffix = "" if method == "all" else f"_{method}"
    return _HERE / f"admire_success_rate_summary{suffix}.csv"


# ══════════════════════════════════════════════════════════════════════════════
# Timing / memory helper (same pattern as car_separating_input.time_jit)
# ══════════════════════════════════════════════════════════════════════════════

def _memory_snapshot() -> Dict[str, float]:
    try:
        stats = jax.devices()[0].memory_stats()
        if stats:
            return {k: float(v) for k, v in stats.items() if 'bytes' in k}
    except Exception:
        pass
    return {'ru_maxrss_kb': float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)}


def time_jit(fn, *args, **kwargs):
    jitted_fn = jax.jit(fn)
    t0 = time.perf_counter()
    jax.block_until_ready(jitted_fn(*args, **kwargs))
    compile_time_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    jax.block_until_ready(jitted_fn(*args, **kwargs))
    run_time_s = time.perf_counter() - t0
    return jitted_fn, compile_time_s, run_time_s, _memory_snapshot()


def _scan_loop(step_fn, init, n: int):
    def body(carry, _):
        return step_fn(carry), None
    carry, _ = jax.lax.scan(body, init, xs=None, length=n)
    return carry


# ══════════════════════════════════════════════════════════════════════════════
# Config grid (input_limit x x0_width) -- horizon is NOT here, see module docstring
# ══════════════════════════════════════════════════════════════════════════════

def build_config_grid():
    combos = [(lim, w) for lim in INPUT_LIMITS for w in X0_WIDTHS]
    input_limit = jnp.array([c[0] for c in combos])
    x0_width = jnp.array([c[1] for c in combos])
    return combos, (input_limit, x0_width)


def _reshape_per_config(limit_per_config: jnp.ndarray, ndim: int) -> jnp.ndarray:
    """limit_per_config: (num_configs,) -> broadcastable against a
    (num_configs, num_restarts, ...) array of the given ndim."""
    shape = (limit_per_config.shape[0],) + (1,) * (ndim - 1)
    return limit_per_config.reshape(shape)


def _clip_per_config(u: jnp.ndarray, limit_per_config: jnp.ndarray) -> jnp.ndarray:
    """u: (num_configs, num_restarts, ..., 10); limit_per_config: (num_configs,)."""
    lim = _reshape_per_config(limit_per_config, u.ndim)
    return jnp.clip(u, -lim, lim)


def _scale_per_config(u_signed: jnp.ndarray, limit_per_config: jnp.ndarray) -> jnp.ndarray:
    """u_signed: (num_configs, num_restarts, ...) with values in [-1, 1];
    scale (not clip) elementwise by each config's own limit, so restart
    inits spread across the FULL feasible box instead of saturating to its
    boundary (clipping a [-1,1] draw against a narrow limit like 0.025
    would put nearly every restart exactly at +-0.025)."""
    lim = _reshape_per_config(limit_per_config, u_signed.ndim)
    return u_signed * lim


# ══════════════════════════════════════════════════════════════════════════════
# Per-method, per-horizon batched sweeps
# ══════════════════════════════════════════════════════════════════════════════

def _single_step_loss(u, x0_width, *, num_steps, dt):
    x0_ivl = irx.icentpert(X0_NOM, jnp.full(9, x0_width))
    return separation_loss(u, x0_ivl, SCENARIOS, dt=dt, num_steps=num_steps)


def _multistep_loss(u_seq, x0_width, *, steps_per_segment, dt):
    x0_ivl = irx.icentpert(X0_NOM, jnp.full(9, x0_width))
    return separation_loss_multistep(u_seq, x0_ivl, SCENARIOS, dt=dt, steps_per_segment=steps_per_segment)


def _refined_loss(u_seq, x0_width, *, num_steps, dt):
    x0_ivl = irx.icentpert(X0_NOM, jnp.full(9, x0_width))
    return propagate_with_refinement_admire(u_seq, x0_ivl, SCENARIOS, dt=dt, num_steps=num_steps)


def run_batched_sweep(loss_fn_single, u_shape_per_restart: Tuple[int, ...],
                      config_arrays, num_restarts: int, num_iters: int,
                      learning_rate: float, seed: int):
    """Double-vmapped (config x restart) GD sweep, ONE jax.jit call.

    loss_fn_single(u, x0_width) -> scalar; input_limit is used only for
    per-config box projection/init range, not inside the loss.
    """
    input_limit, x0_width = config_arrays
    num_configs = input_limit.shape[0]

    inner_loss = jax.vmap(loss_fn_single, in_axes=(0, None))
    inner_grad = jax.vmap(jax.grad(loss_fn_single), in_axes=(0, None))
    batched_loss = jax.vmap(inner_loss, in_axes=(0, 0))
    batched_grad = jax.vmap(inner_grad, in_axes=(0, 0))

    def full_sweep(seed_val):
        key = jax.random.PRNGKey(seed_val)
        u01 = jax.random.uniform(key, (num_configs, num_restarts) + u_shape_per_restart)  # [0,1)
        u0 = _scale_per_config(u01 * 2 - 1, input_limit)   # scale (not clip) to +-input_limit per config

        def body(u_batch):
            g = batched_grad(u_batch, x0_width)
            return _clip_per_config(u_batch - learning_rate * g, input_limit)

        u_final = _scan_loop(body, u0, num_iters)
        losses_final = batched_loss(u_final, x0_width)
        return u_final, losses_final

    jitted_fn, compile_t, run_t, mem = time_jit(full_sweep, seed)
    u_final, losses_final = jitted_fn(seed)
    return u_final, losses_final, compile_t, run_t, mem


_METHOD_LOSS_BUILDERS = {
    "single_step": lambda num_steps, num_segments, dt: (
        partial(_single_step_loss, num_steps=num_steps, dt=dt), (10,)
    ),
    "multistep_unrefined": lambda num_steps, num_segments, dt: (
        partial(_multistep_loss, steps_per_segment=STEPS_PER_SEGMENT, dt=dt), (num_segments, 10)
    ),
    "multistep_refined": lambda num_steps, num_segments, dt: (
        partial(_refined_loss, num_steps=num_steps, dt=dt), (num_steps, 10)
    ),
}
_METHOD_DISPLAY = {
    "single_step": "SINGLE-STEP",
    "multistep_unrefined": "MULTISTEP UNREFINED",
    "multistep_refined": "MULTISTEP REFINED",
}


def run_method_for_horizon(loss_kind: str, horizon_s: float, config_arrays, combos):
    num_steps = NUM_STEPS_FOR_HORIZON[horizon_s]
    num_segments = NUM_SEGMENTS_FOR_HORIZON[horizon_s]
    loss_fn_single, u_shape = _METHOD_LOSS_BUILDERS[loss_kind](num_steps, num_segments, DT)
    name = _METHOD_DISPLAY[loss_kind]

    print(f"\n{'=' * 78}\n{name}  |  horizon={horizon_s}s ({num_steps} steps)\n{'=' * 78}")

    u_final, losses_final, compile_t, run_t, mem = run_batched_sweep(
        loss_fn_single, u_shape, config_arrays,
        num_restarts=NUM_RESTARTS, num_iters=NUM_ITERS,
        learning_rate=LEARNING_RATE, seed=SEED,
    )
    num_configs = losses_final.shape[0]

    best_idx = jnp.argmin(losses_final, axis=1)
    best_loss = losses_final[jnp.arange(num_configs), best_idx]
    restart_success_rate = jnp.mean(losses_final < SUCCESS_THRESHOLD, axis=1)
    config_success = best_loss < SUCCESS_THRESHOLD

    print(f"compile {compile_t * 1e3:9.2f} ms total ({compile_t / num_configs * 1e3:8.3f} ms/config)   "
          f"run {run_t * 1e3:8.3f} ms total ({run_t / num_configs * 1e3:7.3f} ms/config)   mem {mem}")
    print(f"Config success rate: {int(jnp.sum(config_success))}/{num_configs} "
          f"({100 * float(jnp.mean(config_success)):.1f}%)")

    rows = []
    for c, (lim, w) in enumerate(combos):
        rows.append({
            'method': name, 'horizon_s': horizon_s, 'config_idx': c,
            'input_limit': lim, 'x0_width': w,
            'success': bool(config_success[c]), 'final_loss': float(best_loss[c]),
            'restart_success_rate': float(restart_success_rate[c]),
        })

    npz_path = _HERE / f"admire_success_rate_data_{loss_kind}_{horizon_s}s.npz"
    np.savez(
        npz_path,
        losses_final_all_restarts=np.array(losses_final),
        config_success=np.array(config_success), restart_success_rate=np.array(restart_success_rate),
        input_limit=np.array(config_arrays[0]), x0_width=np.array(config_arrays[1]),
        compile_time_s=compile_t, run_time_s=run_t,
    )
    print(f"Saved full data -> {npz_path}")

    return rows, {'compile_time_s': compile_t, 'run_time_s': run_t, 'mem': mem,
                  'success_rate': float(jnp.mean(config_success))}


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=list(_METHOD_LOSS_BUILDERS) + ["all"], default="all")
    parser.add_argument("--horizon", type=float, default=None,
                        help="Run only this horizon (seconds). Omit to run all of HORIZONS_S.")
    parser.add_argument("--confirm-expensive", action="store_true",
                        help="Required only if HORIZONS_S (or --horizon) is edited to include "
                             "something above 2.0s -- the current default axis (0.5/1.0/2.0s) "
                             "does not trigger this. See this file's module docstring for why "
                             "2.0s was chosen as the line: it matches precedent already "
                             "exercised elsewhere in this folder, not extrapolated.")
    args = parser.parse_args()

    horizons_to_run = [args.horizon] if args.horizon is not None else HORIZONS_S
    if any(h > 2.0 for h in horizons_to_run) and not args.confirm_expensive:
        print(
            "Refusing to run: horizon > 2.0s requested without --confirm-expensive.\n"
            "ADMIRE's dynamics divide 5x per Euler step (see module docstring) -- "
            "reverse-mode gradient compile cost is known (from this codebase's "
            "quadrotor precedent) to compound sharply with step count for this kind "
            "of dynamics, and this horizon has no precedent in this folder. Pass "
            "--confirm-expensive if you've already sized this."
        )
        sys.exit(1)

    print("Devices:", jax.devices())
    combos, config_arrays = build_config_grid()
    num_configs = len(combos)
    print(f"Config grid: {len(INPUT_LIMITS)} input_limits x {len(X0_WIDTHS)} x0_widths "
          f"= {num_configs} configs, per (method, horizon)")
    print(f"Horizons: {horizons_to_run}  |  {N_SCENARIOS} fault scenarios (unchanged)")
    print(f"Per (method, horizon): {NUM_RESTARTS} restarts x {NUM_ITERS} GD iters, "
          f"success threshold = {SUCCESS_THRESHOLD}")

    methods_to_run = list(_METHOD_LOSS_BUILDERS) if args.method == "all" else [args.method]

    # Written incrementally, one (method, horizon)'s rows appended right
    # after run_method_for_horizon returns them -- NOT collected in memory
    # and written once at the very end. A compile that runs for tens of
    # minutes per horizon (see module docstring) is a real OOM/walltime/
    # maintenance-reclaim target; the old all-at-the-end write meant a kill
    # during the LAST horizon lost the summary for every horizon that had
    # already finished, even though their raw data was already safely on
    # disk in per-horizon .npz files (see run_method_for_horizon). Start
    # from a clean file each invocation (truncate any stale CSV from a
    # previous run) so this can't silently append onto old data.
    out_csv = _out_csv_for_method(args.method)
    out_csv.unlink(missing_ok=True)
    csv_fieldnames = None
    summaries = {}
    for horizon_s in horizons_to_run:
        for loss_kind in methods_to_run:
            rows, summary = run_method_for_horizon(loss_kind, horizon_s, config_arrays, combos)
            summaries[(loss_kind, horizon_s)] = summary

            is_new_file = csv_fieldnames is None
            if is_new_file:
                csv_fieldnames = list(rows[0].keys())
            with open(out_csv, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=csv_fieldnames)
                if is_new_file:
                    writer.writeheader()
                writer.writerows(rows)
                f.flush()
            print(f"Appended {len(rows)} rows -> {out_csv}")

    print(f"\nWrote summary CSV -> {out_csv}")

    print(f"\n{'=' * 78}\nSUMMARY\n{'=' * 78}")
    for (method, horizon_s), s in summaries.items():
        print(f"  {method:22s} @ {horizon_s:5.1f}s: success {100*s['success_rate']:5.1f}%   "
              f"compile {s['compile_time_s']:7.2f}s total   run {s['run_time_s']*1e3:8.2f} ms total   "
              f"mem {s['mem']}")


if __name__ == "__main__":
    main()
