"""
Post-process the synthesized spoof bias (synthesize_spoof.py's raw output)
into something safe to inject into the REAL QPS simulator.

The raw optimized u_seq is a per-step-independent GD solution with no
smoothness penalty -- it jumps by up to the full 0.6m range in a single
0.02s step. QPS's own rq3_killswitch.py explicitly treats a large bias jump
in one step as itself dangerous (a violent transient against the real
attitude loop), so injecting the raw signal into the real simulator would be
both a poor demo and a plausible source of a bogus-looking flight. This
script moving-average-smooths u_seq, re-clips to the box, and re-verifies
(using the SAME discriminate_controller Section 6 uses) that all four
controllers are still uniquely discriminated before anything touches QPS.

Run with:
  JAX_PLATFORMS=cpu /home/user/immrax-venv/bin/python examples/adaptive_spoofing/smooth_and_verify_spoof.py
"""
import sys
from pathlib import Path

import numpy as np
import jax.numpy as jnp
import immrax as irx

_EXAMPLES_DIR = Path(__file__).resolve().parents[1]
if str(_EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES_DIR))

from adaptive_spoofing.crazyflie_chain_controllers import (
    create_scenarios, hover_reference, QPS_DT, _CANDIDATE_THETA, _BIAS_LIM,
    discriminate_controller, simulate_true_trajectory,
)

RESULTS_DIR = Path(__file__).resolve().parent / "results"
SMOOTH_WINDOW = 5   # steps (0.1s moving-average window at dt=0.02s)


def moving_average(u_seq: np.ndarray, window: int) -> np.ndarray:
    pad = window // 2
    padded = np.pad(u_seq, ((pad, pad), (0, 0)), mode="edge")
    kernel = np.ones(window) / window
    return np.stack([np.convolve(padded[:, d], kernel, mode="valid") for d in range(u_seq.shape[1])], axis=1)


def main():
    data = dict(np.load(RESULTS_DIR / "synthesized_spoof.npz", allow_pickle=True))
    u_raw = data["u_seq"]
    dt = float(data["dt"])
    ref15 = jnp.array(data["ref15"])
    x0_ivl = irx.Interval(lower=jnp.array(data["x0_ivl_lower"]), upper=jnp.array(data["x0_ivl_upper"]))

    u_smooth = moving_average(u_raw, SMOOTH_WINDOW)
    u_smooth = np.clip(u_smooth, -_BIAS_LIM, _BIAS_LIM)

    max_step_raw = np.max(np.abs(np.diff(u_raw, axis=0)))
    max_step_smooth = np.max(np.abs(np.diff(u_smooth, axis=0)))
    print(f"Max per-step bias jump: raw={max_step_raw:.4f} m, smoothed={max_step_smooth:.4f} m")

    scenarios = create_scenarios(ref15)
    u_smooth_j = jnp.array(u_smooth)

    print("\nRe-verifying discrimination on the SMOOTHED bias sequence:")
    all_ok = True
    for true_name in _CANDIDATE_THETA:
        true_theta = jnp.array(_CANDIDATE_THETA[true_name])
        observed = simulate_true_trajectory(jnp.zeros(18), u_smooth_j, true_theta, ref15, dt=QPS_DT)
        result = discriminate_controller(x0_ivl, u_smooth_j, observed, scenarios, dt=QPS_DT)
        survivors = result["survivors"]
        ok = survivors == [true_name]
        all_ok &= ok
        print(f"  true={true_name:16s}  survivors={survivors}  {'OK' if ok else 'AMBIGUOUS/WRONG'}")

    if not all_ok:
        print("\nWARNING: smoothing broke unique discrimination for at least one controller. "
             "Not overwriting u_seq -- inspect before proceeding to the QPS run.")
        return

    data["u_seq_raw"] = u_raw
    data["u_seq"] = u_smooth   # downstream scripts (plots, QPS run) use the smoothed version
    data["smooth_window"] = SMOOTH_WINDOW
    data["all_uniquely_discriminated_smoothed"] = all_ok
    np.savez(RESULTS_DIR / "synthesized_spoof.npz", **data)
    print(f"\nSmoothing preserves discrimination for all 4 controllers. "
         f"Saved smoothed u_seq (raw kept as u_seq_raw) to {RESULTS_DIR / 'synthesized_spoof.npz'}")


if __name__ == "__main__":
    main()
