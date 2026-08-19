"""
Synthesize a separating spoof-bias sequence for the LEGACY 4-controller bank
under a WIDE (realistic, physically-scaled), not tight/degenerate, initial
state uncertainty -- the legacy-bank analog of run_firmware_discrimination.py's
WIDE_WIDTH regime. synthesize_spoof.py's existing run used a uniform,
unrealistically tight width=1e-3 box (fine for the mechanism demo, not
representative of an actual converged state estimate).

Diagnostic run (see conversation) before picking these numbers: propagating
the shared x0_ivl box under ZERO bias showed this bank's interval reachable
sets grow far faster than the firmware bank's ~2x/step "wrapping effect" --
`qps_snap_chain`'s max state-interval width hit 4170 after 30 steps starting
from a modest (0.05m position / 0.1 velocity-scale) box, because its real
gain (k_p=1680) multiplies straight through a SINGLE forward-Euler step of
the *jerk* derivative (`j_dot` includes `-k_p*e_p`), so even a few-cm
position uncertainty becomes a huge jerk-rate uncertainty in one tick --
much more aggressive than the firmware bank's implicit, eigenvalue-bounded
discrete step. This means: (a) WIDE_WIDTH here must be noticeably smaller
than the firmware bank's 5cm-scale numbers or the tubes are meaningless
within a handful of steps, and (b) the horizon must be short (NUM_STEPS=10,
not the tight-regime's 30) for the same reason. Also found: at ANY
remotely-realistic width, all 4 candidates separate almost immediately at
ZERO bias -- even the notoriously hard `pd_pos_vel`/`pid_pos_vel_i` pair,
whose zero-bias overlap at this width/horizon is ~2e-9 (negligible), because
interval reachability itself over-approximates enough that the boxes are
already all but disjoint. This is a STRONGER version of PLAN.md Sec 7's
already-documented finding ("the four candidates separate within a few
Euler steps even at zero spoof bias" for the *tight* regime) -- reported
honestly here rather than hand-picking a width that manufactures a "hard"
optimization problem that doesn't reflect this bank's actual behavior.

Also reports COMPILE time vs RUNTIME for `optimize_refined_gpu`: since it's
not `@jax.jit`-decorated itself (the compile happens inside the `jax.lax.scan`
call it makes), the first call in this process pays JIT compilation on top
of the actual optimization; a second, immediately-following call with
IDENTICAL array shapes hits JAX's compilation cache, isolating steady-state
runtime. (first_call_time - second_call_time) is reported as the compile-time
estimate.

Run with:
  JAX_PLATFORMS=cpu /home/user/immrax-venv/bin/python examples/adaptive_spoofing/synthesize_spoof_wide.py
"""
import sys
import time
from pathlib import Path

import numpy as np
import jax.numpy as jnp
import jax
import immrax as irx

_EXAMPLES_DIR = Path(__file__).resolve().parents[1]
if str(_EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES_DIR))

from adaptive_spoofing.crazyflie_chain_controllers import (
    create_scenarios, hover_reference, QPS_DT, _CANDIDATE_THETA,
    optimize_refined_gpu, discriminate_controller, simulate_true_trajectory,
    refined_overlap_loss, _project_u,
)

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Realistic per-axis uncertainty for a converged state estimate on THIS
# bank's [p,v,a,j,integ,cmd] state -- deliberately smaller than the firmware
# bank's WIDE_WIDTH (see module docstring: this bank's gains amplify a
# forward-Euler step's interval width far more aggressively).
WIDE_WIDTH = jnp.concatenate([
    jnp.full(3, 0.02),   # position (m)
    jnp.full(3, 0.02),   # velocity (m/s)
    jnp.full(3, 0.05),   # acceleration (m/s^2) -- higher derivative, looser
    jnp.full(3, 0.05),   # jerk (m/s^3) -- higher derivative, looser
    jnp.full(3, 0.01),   # integ (hidden controller memory)
    jnp.full(3, 0.02),   # cmd (hidden controller memory)
])
# Short horizon (vs. the tight regime's 30): this bank's wrapping effect
# makes reachable-set widths balloon past any physically-meaningful scale
# well before 30 steps once x0 isn't a near-degenerate box (see docstring).
NUM_STEPS = 10
HOVER_POS = jnp.array([0.0, 0.0, 1.0])


def main():
    ref15 = hover_reference(HOVER_POS)
    scenarios = create_scenarios(ref15)
    x0_ivl = irx.icentpert(jnp.zeros(18), WIDE_WIDTH)
    u_zero = jnp.zeros((NUM_STEPS, 3))

    zero_bias_loss = float(refined_overlap_loss(u_zero, x0_ivl, scenarios, QPS_DT, NUM_STEPS))
    print(f"WIDE regime: {NUM_STEPS} steps, dt={QPS_DT}")
    print(f"separation_loss (refined) at ZERO bias: {zero_bias_loss:.6e}")

    # ── Compile-vs-runtime: optimize_refined_gpu itself has NO @jax.jit
    # boundary (the compile happens inside the jax.lax.scan call it makes
    # each time it's invoked), so calling IT twice recompiles from scratch
    # both times -- confirmed empirically (first call 47.8s, second 73.3s,
    # i.e. NOT faster; see module docstring history). To get a real
    # compile/runtime split, reimplement its exact GD loop here under an
    # explicit @jax.jit, which DOES get a persistent cache keyed by argument
    # shape -- so a second call with the same-shaped input is a genuine
    # cache hit. ──
    num_restarts, learning_rate, num_iters, seed = 48, 0.03, 250, 0

    def loss_fn_refined(u_seq):
        return refined_overlap_loss(u_seq, x0_ivl, scenarios, QPS_DT, NUM_STEPS)

    batched_loss = jax.vmap(loss_fn_refined)
    batched_grad = jax.vmap(jax.grad(loss_fn_refined))

    @jax.jit
    def run_gd(u0):
        def body(u_batch, _i):
            g = batched_grad(u_batch)
            return _project_u(u_batch - learning_rate * g), None
        u_final, _ = jax.lax.scan(body, u0, xs=None, length=num_iters)
        losses = batched_loss(u_final)
        best_idx = jnp.argmin(losses)
        return u_final[best_idx], losses[best_idx], u_final, losses

    key = jax.random.PRNGKey(seed)
    noise_scale = 0.3 * 0.5   # matches _BIAS_LIM * 0.5 used by optimize_refined_gpu
    u0 = jax.random.normal(key, (num_restarts, NUM_STEPS, 3)) * noise_scale

    t0 = time.perf_counter()
    u_star, loss_star, u_all, losses_all = run_gd(u0)
    jax.block_until_ready(u_star)
    t1 = time.perf_counter()
    first_call_s = t1 - t0

    u_star2, loss_star2, _, _ = run_gd(u0)   # same shape -> cache hit
    jax.block_until_ready(u_star2)
    t2 = time.perf_counter()
    second_call_s = t2 - t1

    compile_estimate_s = max(first_call_s - second_call_s, 0.0)
    print(f"\nFirst call  (compile + run): {first_call_s:.2f}s")
    print(f"Second call (cached, run only): {second_call_s:.2f}s")
    print(f"Estimated compile time: {compile_estimate_s:.2f}s")
    print(f"Estimated steady-state runtime: {second_call_s:.2f}s")

    assert bool(jnp.allclose(u_star, u_star2)), "second call diverged from first -- not a cache hit"

    print(f"\nBest refined loss = {float(loss_star):.6e} "
         f"(median over restarts = {float(jnp.median(losses_all)):.6e})")
    print(f"max|bias| = {float(jnp.max(jnp.abs(u_star))):.4f} m")
    print(f"Reduction vs. zero bias: {zero_bias_loss:.3e} -> {float(loss_star):.3e}")

    print("\nSanity check -- discriminate_controller against each candidate as ground truth:")
    all_ok = True
    for true_name in _CANDIDATE_THETA:
        true_theta = jnp.array(_CANDIDATE_THETA[true_name])
        observed = simulate_true_trajectory(jnp.zeros(18), u_star, true_theta, ref15, dt=QPS_DT)
        result = discriminate_controller(x0_ivl, u_star, observed, scenarios, dt=QPS_DT)
        survivors = result["survivors"]
        ok = survivors == [true_name]
        all_ok &= ok
        print(f"  true={true_name:16s}  survivors={survivors}  {'OK' if ok else 'AMBIGUOUS/WRONG'}")

    np.savez(
        RESULTS_DIR / "synthesized_spoof_wide.npz",
        u_seq=np.array(u_star),
        loss_star=float(loss_star),
        zero_bias_loss=zero_bias_loss,
        dt=QPS_DT,
        num_steps=NUM_STEPS,
        hover_pos=np.array(HOVER_POS),
        ref15=np.array(ref15),
        x0_ivl_lower=np.array(x0_ivl.lower),
        x0_ivl_upper=np.array(x0_ivl.upper),
        candidate_names=np.array(list(_CANDIDATE_THETA.keys())),
        candidate_thetas=np.array([_CANDIDATE_THETA[n] for n in _CANDIDATE_THETA]),
        all_uniquely_discriminated=all_ok,
        first_call_s=first_call_s,
        second_call_s=second_call_s,
        compile_estimate_s=compile_estimate_s,
    )
    print(f"\nSaved {RESULTS_DIR / 'synthesized_spoof_wide.npz'}")


if __name__ == "__main__":
    main()
