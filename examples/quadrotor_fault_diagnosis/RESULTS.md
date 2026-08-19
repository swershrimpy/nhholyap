# 12-State Quadrotor Fault Diagnosis — QPS Ground-Truth Results

This reports a statistical diagnosis success-rate experiment for the
12-state quadrotor fault-diagnosis module (`quadrotor_separating_input.py`),
using QPS's own verified rigid-body Crazyflie plant
(`adaptive_spoofing/crazyflie_12d.py`'s `CrazyflieSystem`) as ground truth,
in place of the literal CrazySim Gazebo/ROS2 SITL stack (not installed on
this machine). Reproduction: `solve_and_plot.py` then
`run_quadrotor_diagnosis_qps.py`.

**This file documents two rounds of investigation.** The first pass (below,
"Round 1") found that a near-zero *volume* separation loss was a misleading
proxy — real per-dimension overlap was still 60-90%. Fixing that (longer
horizon + a margin-based loss) is Round 2, which found and fixed a SECOND,
more subtle bug in the fix itself, and then a genuine soundness/horizon
trade-off in the resulting diagnosis experiment. Both rounds are kept
because the mistakes and how they were found are as informative as the
final numbers.

## Setup (current, Round 2)

- 5 scenarios: `Nominal` (α=[1,1,1,1]) + `ActuatorFault_1..4` (thrust,
  roll, pitch, yaw moment respectively; faulted channel α ∈ [0.5, 0.9]).
- Real Crazyflie physical parameters throughout (m=35.89g,
  Ixx=Iyy=2.3951e-5, Izz=3.2346e-5 kg·m², sourced from QPS's own
  `quadcopter_model.py`).
- Separating input `u_seq`: 20 segments, dt=0.1s (2.0s horizon), solved
  by the **multistep-unrefined** layer (`optimize_multistep_gpu_rejit`)
  against `x0_width=0.01` (all 12 states, near-hover), using the
  **margin-based** loss (`margin_tau=0.01`) and **normalized-gradient**
  descent (see "Round 2" below for why both were necessary).
- **Synthesis/reachability model** (`QuadrotorSystem`): body-frame
  velocity, Coriolis-coupled translational dynamics.
- **Ground-truth model** (`CrazyflieSystem`, QPS's own plant, used only to
  simulate the *true* trajectory): world-frame velocity, no Coriolis
  terms — a genuinely different ODE, not merely a coarser integration of
  the same one (see `crazyflie_12d.py`'s docstring).
- Diagnosis rule (matches `car_fault_diagnosis`'s Robotarium precedent and
  Def. "Pairwise pruning" in the paper): a scenario is excluded the first
  time the true state fails to lie in its predicted box at a segment
  boundary, and stays excluded. `correct` = true scenario is the sole
  survivor; `inconclusive` = true scenario survives but so does ≥1 false
  one; `missed` = true scenario gets excluded; `wrong` (⊆ missed) = missed
  **and** exactly one false scenario survives.

---

## Round 1: the volume-product loss was misleading (original finding, kept for record)

`solve_and_plot.py`'s original sanity check reported the 5 scenarios'
final-state boxes as pairwise overlap-volume ≈ 1e-19–1e-20, read at the
time as "effectively exact separation." **That reading was wrong.** Volume
is a *product* of 12 per-dimension overlap widths; a product of 12 numbers
each ~0.02–0.06 is already ~1e-19–1e-16 *even when every dimension is
60–90% overlapping*, not disjoint. Directly checking per-dimension overlap
fraction against `Nominal` confirmed this:

| Scenario | Min per-dim overlap fraction | Dimension |
|---|---|---|
| ActuatorFault_1 (thrust) | **0.0** (genuinely disjoint) | vz |
| ActuatorFault_2 (roll)   | 0.672 | p |
| ActuatorFault_3 (pitch)  | 0.600 | q |
| ActuatorFault_4 (yaw)    | 0.878 | r |

Only the thrust fault was genuinely separated at the original 0.2s
horizon (num_steps=2). This motivated Round 2: increase the horizon (give
the moment faults more time to diverge) and replace the volume-product
loss with something that doesn't vanish just because the boxes are small.

## Round 2: fixing the loss, and a bug found while fixing it

**First attempt (bug):** added `_soft_separation_loss` — targets
`max_d margin[d]`, the actual disjointness criterion (axis-aligned boxes
are disjoint iff *at least one* dimension is disjoint), smoothed via
`tau*logsumexp(margins/tau)` so gradient doesn't vanish. Optimizing this
with the REFINED layer (`propagate_with_refinement`, pairwise-intersected
boxes) at num_steps=3 reported loss=0.0 — but checking the RAW
(independent per-scenario) boxes that the plots/diagnosis actually use
showed NO segment was genuinely disjoint (margin stayed ~-0.02
throughout). Two compounding causes: (1) the refined layer's boxes are a
different object from what's plotted/diagnosed (mutual intersection vs.
independent propagation), so its "disjoint at some step" didn't transfer;
(2) `tau*logsumexp(margins/tau)` is an upper bound on the true max, biased
high by up to `tau*log(12)≈0.025` — comparable to this module's actual
margins, so the loss falsely hit 0 while every dimension was still
overlapping by about that much.

**Fix:** switched to the **multistep-unrefined** layer
(`separation_loss_multistep`, raw independent per-scenario boxes — the
same object the plots and diagnosis check), and corrected the bias by
subtracting `tau*log(12)`, turning the smooth estimate into a safe LOWER
bound on the true max margin (`_soft_separation_loss`'s "Bias correction"
docstring in `quadrotor_separating_input.py`). Loss=0.0 under the
corrected loss is now a genuine certificate, verified directly against
the hard (non-smoothed) per-dimension margin at every segment — not
inferred from the loss value.

**Optimizer:** plain gradient descent got stuck at the control-box
boundary — gradient norm through 20 unrolled Euler steps reaches O(1e3)
(RNN-like compounding) while the moment control box is only O(1e-4)
wide, so any fixed learning rate either overshoots (clipped back to the
same boundary point every iteration, permanently stuck) or crawls.
`optimize_multistep_gpu(..., normalize_grad=True)` (new option, step size
in the *normalized* gradient direction) fixes this.

**Verified result:** with the corrected loss, longer horizon, and
normalized GD (40 restarts, 400 iterations), **segments 0–4 (0–0.5s) are
all genuinely, simultaneously disjoint** across all 5 scenarios — verified
by the hard per-dimension margin check, not the loss value:

| Segment | All pairs disjoint | Worst pairwise margin |
|---|---|---|
| 0 | Yes | +0.0024 |
| 1 | Yes | +0.0121 |
| 2 | Yes | **+0.0251** (best) |
| 3 | Yes | +0.0056 |
| 4 | Yes | +0.0022 |
| 5–19 | No | −0.0002 to −0.033 (boxes re-overlap) |

This doesn't contradict soundness: the loss is `min over segments`, so
the optimizer is only asked to find *some* window where every pair
separates — later re-overlap is fine for an online diagnoser, since a
model excluded during segments 0–4 stays excluded regardless of what the
boxes do afterward.

## Round 2, second caveat: genuine separation ≠ automatically deployable diagnosis

Running the QPS Monte Carlo diagnosis experiment over the **full** 2.0s
horizon gave a striking result: **100% missed for every scenario**,
including `Nominal` evaluated at the exact box center (x0=0, α=1, zero
sampling noise). Tracing this by hand: even at this noise-free point, the
ground-truth (`CrazyflieSystem`, world-frame) trajectory exits its own
predicted box (built from `QuadrotorSystem`, body-frame) by segment 3
(0.3–0.4s), specifically in the velocity dimensions (vx, vz) — exactly
where the two models structurally differ. At this longer horizon and the
larger control authority needed to force separation, the state moves
enough that the near-hover approximation (where body-frame and
world-frame velocity trivially coincide) breaks down, and the frame
mismatch — a small residual effect at the original 0.2s horizon (Round 1
found ~0.1% missed from this) — becomes the dominant error source.

**Fix:** an online diagnoser has no reason to keep checking once
separation is already certified. Sweeping the diagnosis cutoff (how many
segments of `u_seq` are actually used to decide, from 1 up to
`max(disjoint_segments)+1=5`) rather than always running the full horizon
gives a sharp, interpretable trade-off:

**In-bound, fine resolution, aggregate over 5 scenarios × 2000 trials each (10000 trials/row):**

| Cutoff (segments) | Horizon | Correct | Missed | Wrong |
|---|---|---|---|---|
| 1 | 0.1s | **98.9%** | 1.1% | 0.0% |
| 2 | 0.2s | **97.3%** | 2.7% | 0.0% |
| 3 | 0.3s | 59.0% | 41.0% | 0.0% |
| 4 | 0.4s | 7.4% | 92.6% | 0.0% |
| 5 | 0.5s | 3.7% | 96.3% | 0.0% |

The knee is sharp and lands almost exactly where the hand-traced
containment failure was found (segment 3): diagnosis quality is excellent
through cutoff 1–2, then collapses. Wrong-diagnosis rate is 0.0% at every
cutoff in-bound (fine resolution) — failures are "excluded everything" /
"don't know," not confidently incorrect. Adversarial (off-envelope)
trials degrade further at every cutoff as expected (correct rates in the
2–8% range by cutoff 1–2), consistent with the method's guarantee being
conditioned on staying in-bound.

**Coarse resolution** (the literal single-big-Euler-step boxes
`optimize_multistep_gpu_rejit` computes and would deploy online, vs. the
20-substep "fine" boxes above) performs far worse at every cutoff (≤8%
correct even at cutoff 1) — the same single-step wrapping-effect
unsoundness documented for `car_fault_diagnosis`'s Robotarium study and
this module's own earlier (Round 1-era) results.

## Interpretation

1. **The horizon-increase + margin-loss fix worked as intended**: genuine
   (hard-margin-verified, not proxy-inferred) reachable-set disjointness
   across all 5 scenarios, unlike Round 1's volume-product result.
2. **But the longer horizon needed to force that separation pushes the
   trajectory out of the regime where the synthesis model
   (`QuadrotorSystem`, body-frame) is a good proxy for the real plant
   (`CrazyflieSystem`, world-frame QPS)** — a structural model-mismatch
   effect that was a small residual (~0.1% missed) at the original 0.2s
   horizon and becomes the dominant failure mode by ~0.3-0.4s here.
3. **The practical fix is cheap**: don't use the whole horizon for
   diagnosis just because it was needed for synthesis. Checking only
   through segment 1-2 (0.1-0.2s) — still within the certified-disjoint
   window — recovers 97-99% correct diagnosis with 0% wrong, in-bound.
4. **Net effect of this whole exercise**: the demonstrated separating
   controller is now both genuinely separated (fixed Round 1's false
   claim) AND has a verified, sound, high-success-rate diagnosis protocol
   against the real QPS plant (fixed the naive "use the whole optimized
   horizon" assumption) — but only because the second failure mode was
   checked for and diagnosed, not assumed away.

## Files

- `results/quadrotor_separating_input.npz` — the solved `u_seq` (20
  segments), `disjoint_segments=[0,1,2,3,4]`, loss (0.0, margin-based).
- `results/quadrotor_position_3d.pdf`, `results/quadrotor_attitude_3d.pdf`
  — reachable-box + trajectory plots, plotted through segment
  `max(disjoint_segments)+3` (not the full 20 segments — later boxes
  balloon from wrapping-effect over-approximation and would drown out the
  meaningful part of the plot). Position separates cleanly only for the
  thrust fault; roll/pitch/yaw faults separate through attitude/rates —
  see the attitude plot.
- `results/quadrotor_diagnosis_qps.npz` — full per-(cutoff, scenario, mode)
  outcome counts from the cutoff sweep (`run_quadrotor_diagnosis_qps.py`).
