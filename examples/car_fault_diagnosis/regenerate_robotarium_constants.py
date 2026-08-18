"""
Regenerates the embedded constants (U_OPT, X0_CENTER, ALPHA_LO/HI,
SENSOR_OFFSET/SCALE, TUBE_LO/TUBE_HI) used by the robotarium_submit_*.py
scripts, for a chosen (METHOD, CONFIG_IDX) pair from this project's
success_rate_data_*.npz sweep results.

Not run on the Robotarium submission server -- this needs jax/immrax and
this project's own car_separating_input.py, which the submission
environment cannot be assumed to have (see robotarium_submit_diagnosis_
all_modes.py's module docstring). Run this locally, then paste the
printed block into a submission script's constants section, or import
`compute_constants` from another local script (see
generate_all_mode_scripts.py for the batch-generation usage).

Three methods, one shared DT=0.5s / 10-decision-point / 5s horizon
(success_rate_analysis.py's own module docstring: "5s total for every
method: single-step gets ONE control decision held constant for the whole
horizon; both multistep methods get 10 independent control decisions").
`single_step`'s saved u_opt is shape (2,) (one [v,omega]); this script
tiles it to (10,2) before computing tubes, which is mathematically
identical to that method's own native constant-control propagation
(car_separating_input.py's `separation_loss`/`propagate_scenario`) since
both are the same forward-Euler step repeated 10 times at the same dt.
This lets ALL THREE methods share one tube-computation code path here.

Two things this script gets right that an earlier draft of
robotarium_submit_diagnosis_demo.py got wrong (both caught by dry-running
that script against the local rps.robotarium.Robotarium simulator before
treating it as submission-ready) -- see that script and
robotarium_submit_diagnosis_all_modes.py's own docstrings for the full
story:

1. FINE integration for the reachable tubes (TUBE_SUBSTEPS=25 substeps/
   segment, dt=DT/25=0.02s), NOT the coarse single-Euler-step-per-segment
   resolution the optimizer itself used to find u_opt.
2. CarNomActSystem's actuator fault scales OMEGA ONLY (phi_dot =
   alpha*omega; car_separating_input.py), not the full control vector.
   ALPHA_TRUE is computed and printed here for a submission script's own
   drive loop to apply to omega only.

Usage:
  cd examples/car_fault_diagnosis
  JAX_PLATFORMS=cpu /home/user/immrax-venv/bin/python regenerate_robotarium_constants.py <METHOD> <CONFIG_IDX>
  METHOD in {single_step, multistep_unrefined, multistep_refined} (the
  early_stop_* data files -- see success_rate_analysis_early_stop.py --
  are used, since that GD variant is what the earlier RESULTS.md-style
  notes in this project's memory found to have the best success rates).
"""
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import numpy as np
import jax.numpy as jnp
import immrax as irx
from car_separating_input import create_scenarios, _propagate_history, observed_output

DATA_FILES = {
    "single_step": _HERE / "success_rate_data_early_stop_single_step.npz",
    "multistep_unrefined": _HERE / "success_rate_data_early_stop_multistep_unrefined.npz",
    "multistep_refined": _HERE / "success_rate_data_early_stop_multistep_refined.npz",
}
DT = 0.5                # segment length (s) u_opt was optimized at
NUM_STEPS = 10           # shared decision-point count across all 3 methods
TUBE_SUBSTEPS = 25       # fine resolution for the CHECKED tube -- see module docstring point 1

# Real Robotarium hardware limits (rps/robotarium_abc.py)
MAX_V = 0.2
MAX_OMEGA = 2 * (0.016 / 0.11) * (0.2 / 0.016)


def _fmt_rows(arr: np.ndarray) -> str:
    lines = ["    " + repr(row.tolist()) + "," for row in arr]
    return "\n".join(lines)


def compute_constants(method: str, config_idx: int) -> dict:
    """Returns a dict with all the embedded-constant values (as numpy
    arrays / plain floats), plus a `fits_hw` bool. Raises AssertionError
    if config_success is False for this (method, config_idx)."""
    d = np.load(DATA_FILES[method], allow_pickle=True)
    assert d["config_success"][config_idx], \
        f"{method}/config {config_idx} did not converge (config_success=False) -- pick another"

    u_opt_raw = d["u_opt"][config_idx]
    u_seq = np.tile(u_opt_raw, (NUM_STEPS, 1)) if u_opt_raw.ndim == 1 else u_opt_raw
    x0_center = d["x0_center"][config_idx]
    x0_width = float(d["x0_width"][config_idx])
    alpha_lo = float(d["alpha_lo"][config_idx])
    alpha_hi = float(d["alpha_hi"][config_idx])
    sensor_offset = d["sensor_offset"][config_idx]
    sensor_scale = float(d["sensor_scale"][config_idx])
    alpha_true = 0.5 * (alpha_lo + alpha_hi)

    max_v, max_omega = np.abs(u_seq[:, 0]).max(), np.abs(u_seq[:, 1]).max()
    fits_hw = bool((max_v <= MAX_V) and (max_omega <= MAX_OMEGA))

    scenarios = create_scenarios(alpha_lo, alpha_hi, tuple(sensor_offset.tolist()), sensor_scale)
    x0_ivl = irx.icentpert(jnp.array(x0_center), jnp.full(3, x0_width))
    u_seq_j = jnp.array(u_seq)
    dt_fine = DT / TUBE_SUBSTEPS

    tube_lo, tube_hi = {}, {}
    for s in scenarios:
        x_hist = _propagate_history(x0_ivl, u_seq_j, s.emb_system, s.p_interval, dt_fine, TUBE_SUBSTEPS)
        lo_rows, hi_rows = [], []
        for k in range(x_hist.lower.shape[0]):
            x_ivl_k = irx.Interval(lower=x_hist.lower[k], upper=x_hist.upper[k])
            y_ivl_k = observed_output(x_ivl_k, s)
            lo_rows.append(np.array(y_ivl_k.lower))
            hi_rows.append(np.array(y_ivl_k.upper))
        tube_lo[s.name] = np.stack(lo_rows)
        tube_hi[s.name] = np.stack(hi_rows)

    return dict(
        method=method, config_idx=config_idx, u_opt=u_seq, x0_center=np.array(x0_center),
        x0_width=x0_width, alpha_lo=alpha_lo, alpha_hi=alpha_hi, alpha_true=alpha_true,
        sensor_offset=np.array(sensor_offset), sensor_scale=sensor_scale,
        max_v=float(max_v), max_omega=float(max_omega), fits_hw=fits_hw,
        tube_lo=tube_lo, tube_hi=tube_hi,
    )


def print_constants(c: dict):
    print(f"# method={c['method']}  config_idx={c['config_idx']}  "
         f"max|v|={c['max_v']:.4f} (limit {MAX_V})  "
         f"max|omega|={c['max_omega']:.4f} (limit {MAX_OMEGA:.4f})  fits_hardware={c['fits_hw']}")
    if not c["fits_hw"]:
        print("# WARNING: this config's u_opt exceeds real Robotarium hardware limits --\n"
             "# it would be silently clipped/distorted if submitted. Pick a different config.")

    print(f"\nU_OPT = np.array([\n{_fmt_rows(c['u_opt'])}\n])")
    print(f"\nX0_CENTER = np.array({c['x0_center'].tolist()!r})")
    print(f"ALPHA_LO, ALPHA_HI = {c['alpha_lo']}, {c['alpha_hi']}")
    print(f"SENSOR_OFFSET = np.array({c['sensor_offset'].tolist()!r})")
    print(f"SENSOR_SCALE = {c['sensor_scale']}")

    print("\nTUBE_LO = {")
    for name in ("Nominal", "Actuator Fault", "Sensor Fault"):
        print(f'    "{name}": np.array([\n{_fmt_rows(c["tube_lo"][name])}\n    ]),')
    print("}")
    print("TUBE_HI = {")
    for name in ("Nominal", "Actuator Fault", "Sensor Fault"):
        print(f'    "{name}": np.array([\n{_fmt_rows(c["tube_hi"][name])}\n    ]),')
    print("}")


if __name__ == "__main__":
    method = sys.argv[1] if len(sys.argv) > 1 else "multistep_refined"
    config_idx = int(sys.argv[2]) if len(sys.argv) > 2 else 32
    print_constants(compute_constants(method, config_idx))
