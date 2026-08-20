"""
ADMIRE -- number of still-valid (non-zero output intersection) model pairs
over time, unrefined vs. refined.

Reproduces examples/admire/plot_refinement_3d.py's output (same plot:
count of pairs whose predicted [pb,qb,rb] output intervals still intersect,
at each time step, unrefined vs. refined), but the unrefined history uses
this folder's shared `euler_step`/`output_interval`, and the refined
history is built from `collect_refinement_history` -- the SAME per-pair
helper (`_step_one_pair`) used by `propagate_with_refinement`'s jittable
loss -- instead of a third independent reimplementation of the refinement
math. This closes the "two/three independently-maintained
reimplementations can silently drift apart" gap flagged as bug #3 in
examples/unicycle/car_separating_input.py's PLAN.md, which the original
examples/admire/plot_refinement_3d.py was exposed to (it re-derived the
refinement step locally, separately from admire_refined_sequence_optimizer.py).

Solves for its own separating control sequence via optimize_multistep_gpu
(rather than requiring a pre-saved admire_u_opt.npz, unlike the original --
keeps this folder self-contained with no notebook dependency).

Usage:
  /home/user/immrax-venv/bin/python examples/admire_fault_diagnosis/plot_refinement_3d.py
"""
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import immrax as irx
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from admire_separating_input import (
    create_scenarios,
    euler_step,
    output_interval,
    optimize_multistep_gpu,
    collect_refinement_history,
)

DT = 0.1
NUM_STEPS = 10
NUM_RESTARTS = 4
NUM_ITERS = 10
SEED = 42


def _has_intersection(a, b):
    lo = jnp.maximum(a.lower, b.lower)
    hi = jnp.minimum(a.upper, b.upper)
    return bool(jnp.all(hi >= lo))


def main():
    print(f"Devices: {jax.devices()}")
    scenarios = create_scenarios(fault_effectiveness=0.0)
    print(f"{len(scenarios)} scenarios: " + ", ".join(s.name for s in scenarios))

    x0_nom = jnp.zeros(9).at[0].set(343.0 * 0.3)
    x0_ivl = irx.icentpert(x0_nom, jnp.ones(9) * 0.01)

    print(f"\nSolving separating control sequence ({NUM_STEPS} x {DT}s = "
          f"{NUM_STEPS*DT:.1f}s, {NUM_RESTARTS} restarts x {NUM_ITERS} iters) ...")
    print("Compiling (division-heavy dynamics -- may take a while) ...")
    t0 = time.perf_counter()
    u_seq, loss_opt, _, _ = jax.jit(
        lambda seed: optimize_multistep_gpu(
            x0_ivl=x0_ivl, scenarios=scenarios, dt=DT, steps_per_segment=1,
            num_segments=NUM_STEPS, num_restarts=NUM_RESTARTS, num_iters=NUM_ITERS,
            seed=seed,
        )
    )(SEED)
    jax.block_until_ready(loss_opt)
    print(f"  done in {time.perf_counter()-t0:.1f}s, loss={float(loss_opt):.6f}")

    pairs = [(i, j) for i in range(len(scenarios)) for j in range(i + 1, len(scenarios))]
    n_pairs = len(pairs)

    # -- Unrefined: propagate each scenario independently --
    print("Collecting unrefined output histories ...")
    output_histories = []
    for s in scenarios:
        hist = [output_interval(x0_ivl)]
        x = x0_ivl
        for step in range(NUM_STEPS):
            x = euler_step(s.emb_system, x, u_seq[step], s.p_interval, DT)
            hist.append(output_interval(x))
        output_histories.append(hist)

    t_unref = [k * DT for k in range(NUM_STEPS + 1)]
    unref_valid = [True] * n_pairs
    unref_counts = []
    for t_idx in range(NUM_STEPS + 1):
        for k, (i, j) in enumerate(pairs):
            if unref_valid[k] and not _has_intersection(output_histories[i][t_idx], output_histories[j][t_idx]):
                unref_valid[k] = False
        unref_counts.append(sum(unref_valid))

    # -- Refined: shared collect_refinement_history (no local reimplementation) --
    print("Collecting refined histories (shared _step_one_pair helper) ...")
    ref_steps, ref_pairs = collect_refinement_history(x0_ivl, u_seq, scenarios, DT, NUM_STEPS)
    assert ref_pairs == pairs

    t_ref = [0.0] + [step['t'] for step in ref_steps]
    ref_valid = [True] * n_pairs
    ref_counts = [n_pairs]
    for step in ref_steps:
        for k, (obs_i, obs_j, _inter) in enumerate(step['pair_obs']):
            if ref_valid[k] and not _has_intersection(obs_i, obs_j):
                ref_valid[k] = False
        ref_counts.append(sum(ref_valid))

    print(f"  unrefined counts: {unref_counts}")
    print(f"  refined counts  : {ref_counts}")

    # -- Plot --
    plt.rcParams.update({'font.family': 'serif', 'font.size': 11})
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(t_unref, unref_counts, color='steelblue', linewidth=2,
            marker='o', markersize=4, label='Baseline')
    ax.plot(t_ref, ref_counts, color='darkorange', linewidth=2,
            marker='s', markersize=4, label='Intersection Refinement')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Number of unseparated model pairs')
    ax.set_ylim(-0.5, n_pairs + 0.5)
    ax.legend(frameon=True, fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    fname = HERE / 'admire_valid_pairs.pdf'
    plt.savefig(fname, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {fname}")


if __name__ == "__main__":
    main()
