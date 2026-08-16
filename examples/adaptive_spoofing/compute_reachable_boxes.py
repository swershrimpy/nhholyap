"""
Compute per-controller reachable POSITION boxes for video overlay.

For each of the 4 candidate controllers, propagate the SAME x0_ivl/u_seq used
by run_discrimination.py (Section 6) one euler_step at a time -- but here we
keep every candidate's own unrefined tube (no cross-candidate intersection,
no falsification), so what gets visualized is exactly "where the drone would
be if it were running controller X", not the discriminator's verdict. The
real QPS drone should visually stay inside the qps_snap_chain box and drift
out of the other three, since qps_snap_chain is QPS's actual controller
(confirmed by run_discrimination.py).

Only position (indices 0:3 of the 18-dim chain state) is saved -- that's all
a 3D box overlay on the video needs.

Run with:
  JAX_PLATFORMS=cpu /home/user/immrax-venv/bin/python examples/adaptive_spoofing/compute_reachable_boxes.py
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
    create_scenarios, euler_step, _CANDIDATE_THETA,
)

RESULTS_DIR = Path(__file__).resolve().parent / "results"

# Same choices as run_discrimination.py -- see that script's docstring for
# why (real settled state, not zeros; observable/hidden width split).
OBSERVABLE_WIDTH = 1e-2
HIDDEN_WIDTH = 1e-3


def main():
    qps_data = np.load(RESULTS_DIR / "qps_observed_trajectory.npz")
    u_seq = jnp.array(qps_data["u_seq"])
    dt = float(qps_data["dt"])
    settled_state = qps_data["settled_state"]
    num_steps = u_seq.shape[0]

    spoof_data = np.load(RESULTS_DIR / "synthesized_spoof.npz", allow_pickle=True)
    ref15 = jnp.array(spoof_data["ref15"])
    scenarios = create_scenarios(ref15)
    names = [s.name for s in scenarios]

    x0_center = jnp.concatenate([jnp.array(settled_state), jnp.zeros(6)])
    x0_width = jnp.concatenate([jnp.full(12, OBSERVABLE_WIDTH), jnp.full(6, HIDDEN_WIDTH)])
    x0_ivl = irx.icentpert(x0_center, x0_width)

    n = len(scenarios)
    pos_lower = np.zeros((num_steps + 1, n, 3))
    pos_upper = np.zeros((num_steps + 1, n, 3))
    pos_lower[0] = np.array(x0_ivl.lower[:3])[None, :].repeat(n, axis=0)
    pos_upper[0] = np.array(x0_ivl.upper[:3])[None, :].repeat(n, axis=0)

    x_ivls = [x0_ivl] * n
    for k in range(num_steps):
        new_ivls = []
        for i, s in enumerate(scenarios):
            x_next = euler_step(s.emb_system, x_ivls[i], u_seq[k], s.p_interval, dt)
            new_ivls.append(x_next)
            pos_lower[k + 1, i] = np.array(x_next.lower[:3])
            pos_upper[k + 1, i] = np.array(x_next.upper[:3])
        x_ivls = new_ivls

    np.savez(
        RESULTS_DIR / "reachable_boxes.npz",
        pos_lower=pos_lower,     # (num_steps+1, n_scenarios, 3)
        pos_upper=pos_upper,
        scenario_names=np.array(names),
        dt=dt,
        num_steps=num_steps,
    )
    print(f"Saved reachable position boxes to {RESULTS_DIR / 'reachable_boxes.npz'}")
    print(f"Box widths at final step (should differ a lot per controller by t={num_steps*dt:.2f}s):")
    for i, name in enumerate(names):
        w = pos_upper[-1, i] - pos_lower[-1, i]
        print(f"  {name:16s}  width(x,y,z)={np.round(w, 4)}")


if __name__ == "__main__":
    main()
