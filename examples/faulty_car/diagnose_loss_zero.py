"""
diagnose_loss_zero.py
=====================
Re-evaluates tracking_cbf_loss step-by-step (no vmap / lax.scan) using the
controller saved in theta_track_opt.npz, printing every intermediate variable
so that the reason loss = 0 is unambiguous.

Run from nhholyap/examples/faulty_car/ with the immrax-venv Python:
  /home/user/immrax-venv/bin/python3 diagnose_loss_zero.py
"""

import sys
sys.path.insert(0, '.')

import numpy as np
import jax.numpy as jnp
import immrax as irx
from pathlib import Path

from faulty_car_output_feedback_cbf import (
    create_track_cl_scenarios,
    cl_euler_multistep,
    cbf_penalty_interval,
    _obs_interval_raw,
)
from interval_functions import overlap_size_lax

# ── Load the saved controller ─────────────────────────────────────────────────
_FAULT_DIR = Path('../../../robotarium_python_simulator/rps/examples/fault_diagnosis')
npz = np.load(_FAULT_DIR / 'theta_track_opt.npz')

theta_seq   = jnp.array(npz['theta_seq'])     # (num_steps, 6)
y_hat_seq   = jnp.array(npz['y_hat_seq'])     # (num_steps, 2)
dt          = float(npz['dt'])
obstacles   = jnp.array(npz['obstacles'])
x0_center   = jnp.array(npz['x0_center'])
x0_pert     = jnp.array(npz['x0_pert'])
num_steps   = int(theta_seq.shape[0])
cbf_weight  = 2.0           # same as used in Cell 64
num_substeps = 10           # same as used in Cell 64

cl_scenarios = create_track_cl_scenarios(actuator_alpha_lo=0.0, actuator_alpha_hi=0.5)
scenario_names = [s.name for s in cl_scenarios]
n = len(cl_scenarios)
xlen = 3   # state dim: [px, py, phi]

x0_ivl = irx.icentpert(x0_center, x0_pert)

PAIRS = [(i, j) for i in range(n) for j in range(i + 1, n)]
PAIR_NAMES = [f'{scenario_names[i]} vs {scenario_names[j]}' for i, j in PAIRS]
P = len(PAIRS)

# ── Helpers that mirror tracking_cbf_loss exactly ────────────────────────────
emb_systems = cl_scenarios[0].emb_system   # same as in tracking_cbf_loss (Bug 2)

p_lowers = jnp.stack([s.p_interval.lower for s in cl_scenarios])   # (n, 1)
p_uppers = jnp.stack([s.p_interval.upper for s in cl_scenarios])   # (n, 1)
obs_scales  = jnp.stack([s.obs_scale  for s in cl_scenarios])      # (n, 1)
obs_offsets = jnp.stack([s.obs_offset for s in cl_scenarios])      # (n, 2)

def ivl_to_arr(ivl):
    return jnp.concatenate([ivl.lower, ivl.upper])          # (2*xlen,)

def arr_to_ivl(arr):
    return irx.Interval(lower=arr[:xlen], upper=arr[xlen:])

def obs_arr_to_ivl(arr):
    return irx.Interval(lower=arr[:2], upper=arr[2:])

def pack(k):
    return jnp.concatenate([theta_seq[k, :4], y_hat_seq[k], theta_seq[k, 4:6]])  # 8-D

def prop_one(x_arr, p_lower, p_upper, full_theta):
    """Matches prop_one in tracking_cbf_loss: uses shared emb_systems."""
    x_ivl  = arr_to_ivl(x_arr)
    p_ivl  = irx.Interval(lower=p_lower, upper=p_upper)
    x_next = cl_euler_multistep(emb_systems, x_ivl, full_theta, p_ivl, dt, num_substeps)
    return ivl_to_arr(x_next)

def obs_one(x_arr, obs_scale, obs_offset):
    x_ivl   = arr_to_ivl(x_arr)
    obs_ivl = _obs_interval_raw(x_ivl, obs_scale, obs_offset)
    return jnp.concatenate([obs_ivl.lower, obs_ivl.upper])   # (4,) = (2*ylen,)

def pair_overlap_and_refine(pxi_arr, pxj_arr,
                             obs_scale_i, obs_scale_j,
                             obs_offset_i, obs_offset_j,
                             full_theta,
                             p_lower_i, p_upper_i,
                             p_lower_j, p_upper_j,
                             pair_name, step_label):
    """Exact replica of pair_overlap_and_refine in tracking_cbf_loss, with prints."""
    xi = arr_to_ivl(pxi_arr)
    xj = arr_to_ivl(pxj_arr)

    obs_i_arr = obs_one(pxi_arr, obs_scale_i, obs_offset_i)
    obs_j_arr = obs_one(pxj_arr, obs_scale_j, obs_offset_j)
    obs_i = obs_arr_to_ivl(obs_i_arr)
    obs_j = obs_arr_to_ivl(obs_j_arr)

    y_lo = jnp.maximum(obs_i.lower, obs_j.lower)
    y_hi = jnp.minimum(obs_i.upper, obs_j.upper)
    has_overlap = bool(jnp.all(y_hi >= y_lo))

    print(f"  [{pair_name}]  {step_label}")
    print(f"    input xi state:  lo={np.array(xi.lower).round(4)}  hi={np.array(xi.upper).round(4)}")
    print(f"    input xj state:  lo={np.array(xj.lower).round(4)}  hi={np.array(xj.upper).round(4)}")
    print(f"    obs_i (output):  lo={np.array(obs_i.lower).round(4)}  hi={np.array(obs_i.upper).round(4)}")
    print(f"    obs_j (output):  lo={np.array(obs_j.lower).round(4)}  hi={np.array(obs_j.upper).round(4)}")
    print(f"    output intersect: lo={np.array(y_lo).round(4)}  hi={np.array(y_hi).round(4)}")
    print(f"    has_overlap = {has_overlap}")

    # fallback used when has_overlap=False (mirrors JAX logic)
    fallback  = (xi.lower[:2] + xi.upper[:2]) / 2
    y_lo_safe = jnp.where(has_overlap, y_lo, fallback)
    y_hi_safe = jnp.where(has_overlap, y_hi, fallback)

    si_s = obs_scale_i[0]
    sj_s = obs_scale_j[0]

    xi_ref = irx.Interval(
        lower=jnp.array([(y_lo_safe[0] - obs_offset_i[0]) / si_s,
                          (y_lo_safe[1] - obs_offset_i[1]) / si_s,
                          xi.lower[2]]),
        upper=jnp.array([(y_hi_safe[0] - obs_offset_i[0]) / si_s,
                          (y_hi_safe[1] - obs_offset_i[1]) / si_s,
                          xi.upper[2]]),
    )
    xj_ref = irx.Interval(
        lower=jnp.array([(y_lo_safe[0] - obs_offset_j[0]) / sj_s,
                          (y_lo_safe[1] - obs_offset_j[1]) / sj_s,
                          xj.lower[2]]),
        upper=jnp.array([(y_hi_safe[0] - obs_offset_j[0]) / sj_s,
                          (y_hi_safe[1] - obs_offset_j[1]) / sj_s,
                          xj.upper[2]]),
    )
    print(f"    xi_ref:          lo={np.array(xi_ref.lower).round(4)}  hi={np.array(xi_ref.upper).round(4)}")
    print(f"    xj_ref:          lo={np.array(xj_ref.lower).round(4)}  hi={np.array(xj_ref.upper).round(4)}")

    # Propagate refined intervals with full_theta
    xn_i_arr = prop_one(ivl_to_arr(xi_ref), p_lower_i, p_upper_i, full_theta)
    xn_j_arr = prop_one(ivl_to_arr(xj_ref), p_lower_j, p_upper_j, full_theta)

    obs_ni = obs_arr_to_ivl(obs_one(xn_i_arr, obs_scale_i, obs_offset_i))
    obs_nj = obs_arr_to_ivl(obs_one(xn_j_arr, obs_scale_j, obs_offset_j))

    raw_cost = float(overlap_size_lax(obs_ni, obs_nj))
    overlap  = raw_cost if has_overlap else 0.0

    print(f"    propagated obs_ni: lo={np.array(obs_ni.lower).round(4)}  hi={np.array(obs_ni.upper).round(4)}")
    print(f"    propagated obs_nj: lo={np.array(obs_nj.lower).round(4)}  hi={np.array(obs_nj.upper).round(4)}")
    print(f"    raw_cost (overlap of prop. refined) = {raw_cost:.6f}")
    print(f"    final overlap contribution = {overlap:.6f}  "
          f"({'has_overlap=True → uses raw_cost' if has_overlap else 'has_overlap=False → forced to 0'})")
    print()

    return overlap, xn_i_arr, xn_j_arr


# ══════════════════════════════════════════════════════════════════════════════
# MAIN TRACE
# ══════════════════════════════════════════════════════════════════════════════

SEP = "=" * 72

print(SEP)
print("SETUP")
print(SEP)
print(f"  num_steps={num_steps}  dt={dt}  num_substeps={num_substeps}  cbf_weight={cbf_weight}")
print(f"  x0_center = {np.array(x0_center)}")
print(f"  x0_pert   = {np.array(x0_pert)}")
print(f"  x0_ivl    lower={np.array(x0_ivl.lower).round(4)}  upper={np.array(x0_ivl.upper).round(4)}")
print(f"  scenarios : {scenario_names}")
print(f"  p_lowers  : {np.array(p_lowers).flatten()}")
print(f"  p_uppers  : {np.array(p_uppers).flatten()}")
print(f"  obs_scales : {np.array(obs_scales).flatten()}")
print(f"  obs_offsets: {np.array(obs_offsets)}")
print(f"  emb_systems (shared, Bug 2): {type(emb_systems).__name__}")
print(f"  pair_names : {PAIR_NAMES}")
print()

print(SEP)
print("CONTROLLER (K and u_ff per step)")
print(SEP)
for k in range(num_steps):
    K   = np.array(theta_seq[k, :4]).reshape(2, 2)
    uff = np.array(theta_seq[k, 4:6])
    yh  = np.array(y_hat_seq[k])
    print(f"  step {k}: K=[[{K[0,0]:+.4f},{K[0,1]:+.4f}],[{K[1,0]:+.4f},{K[1,1]:+.4f}]]"
          f"  u_ff=[{uff[0]:+.4f},{uff[1]:+.4f}]  y_hat=[{yh[0]:.4f},{yh[1]:.4f}]")
print()

# ── Initial propagation (step 0 with theta_0) ────────────────────────────────
print(SEP)
print("STEP 0  — propagate x0 → state intervals at t=1s  (theta_0 applied once)")
print(SEP)
full_0   = pack(0)
x0_arr   = ivl_to_arr(x0_ivl)
x_arrs   = [prop_one(x0_arr, p_lowers[i], p_uppers[i], full_0) for i in range(n)]
cbf_pens = [float(cbf_penalty_interval(arr_to_ivl(x_arrs[i]), obstacles)) for i in range(n)]
cbf_pen  = sum(cbf_pens)

for i, s in enumerate(cl_scenarios):
    xi = arr_to_ivl(x_arrs[i])
    oi = obs_arr_to_ivl(obs_one(x_arrs[i], obs_scales[i], obs_offsets[i]))
    print(f"  {s.name}:")
    print(f"    state  lo={np.array(xi.lower).round(4)}  hi={np.array(xi.upper).round(4)}")
    print(f"    output lo={np.array(oi.lower).round(4)}  hi={np.array(oi.upper).round(4)}")
    print(f"    CBF penalty = {cbf_pens[i]:.6f}")

# ── step0_overlaps: refine and propagate pairs (still using theta_0) ──────────
print()
print(SEP)
print("STEP 0  — pair_overlap_and_refine  (refine intersection at t=1s, propagate with theta_0)")
print("NOTE: This applies theta_0 a SECOND time (the off-by-one / Bug 3 discussed in debug_plan.md)")
print(SEP)

pxi_arrs = [x_arrs[i] for i, j in PAIRS]
pxj_arrs = [x_arrs[j] for i, j in PAIRS]

step0_overlaps = []
new_pxi_arrs, new_pxj_arrs = [], []
for pi, (ia, ib) in enumerate(PAIRS):
    ov, nxi, nxj = pair_overlap_and_refine(
        pxi_arrs[pi], pxj_arrs[pi],
        obs_scales[ia], obs_scales[ib],
        obs_offsets[ia], obs_offsets[ib],
        full_0,
        p_lowers[ia], p_uppers[ia],
        p_lowers[ib], p_uppers[ib],
        PAIR_NAMES[pi], "refine(t=1s) → propagate with theta_0 → overlap at 'step_0_refined'",
    )
    step0_overlaps.append(ov)
    new_pxi_arrs.append(nxi)
    new_pxj_arrs.append(nxj)

min_sep = list(step0_overlaps)
print("After step0_overlaps:")
for pi, name in enumerate(PAIR_NAMES):
    print(f"  min_sep[{name}] = {min_sep[pi]:.6f}")
print(f"  cbf_pen so far = {cbf_pen:.6f}")
print()

# ── lax.scan over steps 1..num_steps-1 ───────────────────────────────────────
pxi_arrs = new_pxi_arrs
pxj_arrs = new_pxj_arrs
full_theta_seq = [pack(k) for k in range(1, num_steps)]   # thetas 1..T-1

for scan_k, full_theta_k in enumerate(full_theta_seq):
    real_k = scan_k + 1   # theta index
    print(SEP)
    print(f"SCAN step {scan_k}  (theta_{real_k} / t={real_k}s→{real_k+1}s)")
    print(SEP)

    # 1. Propagate ALL full scenarios one step (for CBF tracking)
    x_arrs = [prop_one(x_arrs[i], p_lowers[i], p_uppers[i], full_theta_k) for i in range(n)]
    step_cbf = sum(float(cbf_penalty_interval(arr_to_ivl(x_arrs[i]), obstacles)) for i in range(n))
    cbf_pen += step_cbf
    print(f"  Full scenario state intervals (for CBF):")
    for i, s in enumerate(cl_scenarios):
        xi = arr_to_ivl(x_arrs[i])
        oi = obs_arr_to_ivl(obs_one(x_arrs[i], obs_scales[i], obs_offsets[i]))
        print(f"    {s.name}: state lo={np.array(xi.lower).round(4)} hi={np.array(xi.upper).round(4)}")
        print(f"    {s.name}: output lo={np.array(oi.lower).round(4)} hi={np.array(oi.upper).round(4)}")
    print(f"  CBF penalty this step = {step_cbf:.6f}  (cumulative = {cbf_pen:.6f})")
    print()

    # 2. Refine + propagate pairs
    print(f"  Pair refinement (carry pxi/pxj from previous step → propagate with theta_{real_k}):")
    overlaps = []
    new_pxi_arrs, new_pxj_arrs = [], []
    for pi, (ia, ib) in enumerate(PAIRS):
        ov, nxi, nxj = pair_overlap_and_refine(
            pxi_arrs[pi], pxj_arrs[pi],
            obs_scales[ia], obs_scales[ib],
            obs_offsets[ia], obs_offsets[ib],
            full_theta_k,
            p_lowers[ia], p_uppers[ia],
            p_lowers[ib], p_uppers[ib],
            PAIR_NAMES[pi], f"scan_step={scan_k} theta_{real_k}",
        )
        overlaps.append(ov)
        new_pxi_arrs.append(nxi)
        new_pxj_arrs.append(nxj)

    # Update min
    for pi in range(P):
        min_sep[pi] = min(min_sep[pi], overlaps[pi])

    print(f"  After scan step {scan_k}: overlaps this step = {[round(o, 6) for o in overlaps]}")
    print(f"  min_sep (running min across all steps so far) = {[round(m, 6) for m in min_sep]}")
    print()

    pxi_arrs = new_pxi_arrs
    pxj_arrs = new_pxj_arrs

# ── Final result ──────────────────────────────────────────────────────────────
sep_cost = sum(min_sep)
total_loss = sep_cost + cbf_weight * cbf_pen

print(SEP)
print("FINAL RESULT")
print(SEP)
print(f"  min_sep_per_pair (final):")
for pi, name in enumerate(PAIR_NAMES):
    print(f"    {name}: {min_sep[pi]:.6f}")
print(f"  sep_cost   = sum(min_sep) = {sep_cost:.6f}")
print(f"  cbf_pen    = {cbf_pen:.6f}")
print(f"  total_loss = sep_cost + {cbf_weight} * cbf_pen = {total_loss:.6f}")
print()

print(SEP)
print("CONCLUSION")
print(SEP)
if sep_cost == 0.0:
    zero_pairs = [PAIR_NAMES[pi] for pi in range(P) if min_sep[pi] == 0.0]
    nonzero_pairs = [PAIR_NAMES[pi] for pi in range(P) if min_sep[pi] > 0.0]
    print(f"  sep_cost = 0 because min_sep = 0 for ALL pairs.")
    print(f"  Pairs where min_sep=0: {zero_pairs}")
    if nonzero_pairs:
        print(f"  Pairs where min_sep>0: {nonzero_pairs}")
    print()
    print("  For each zero pair, trace back to which scan step first produced overlap=0.")
    print("  This happens either because:")
    print("    (A) has_overlap was False (intersection was empty) → overlap forced to 0.0")
    print("    (B) raw_cost was 0 (propagated refined intervals do not overlap)")
    print()
    print("  See the scan step outputs above for 'has_overlap=False' to find (A),")
    print("  or 'raw_cost = 0.0' with 'has_overlap=True' to find (B).")
else:
    print(f"  sep_cost = {sep_cost:.6f} != 0  (loss is not zero with this controller).")
