# adaptive_spoofing: controller-discrimination pipeline

## 0. What's implemented so far

- `crazyflie_12d.py` — `CrazyflieSystem`, an immrax `System` that reproduces
  QPS's actual rigid-body plant (`qps/utilities/quadcopter_model.py`,
  `forward_model()`) with the real Crazyflie physical constants. Verified
  bit-for-bit (mod float precision) against a transcription of QPS's own code
  across 8 random states (`tests/test_crazyflie_12d.py`, 19 tests passing).
- Findings that motivated it (full detail in the module docstring):
  - **Dynamics**: `quadrotor_fault_diagnosis/quadrotor_separating_input.py`'s
    `QuadrotorSystem` uses BODY-frame velocity state (Coriolis-coupled
    translational dynamics). QPS's real Crazyflie plant uses WORLD-frame
    velocity state (no Coriolis coupling, position integrates velocity
    directly). Same rotational dynamics and Euler kinematics, different
    translational-kinematics equations — a structurally different ODE, not a
    relabeling.
  - **Parameters**: `QuadrotorSystem`'s defaults (m=0.468 kg, Ixx=Iyy=4.856e-3,
    Izz=8.801e-3) are generic literature quadrotor values, ~13x too heavy and
    ~2 orders of magnitude too much inertia for the real Crazyflie QPS models
    (m=0.03589 kg, Ixx=Iyy=2.3951e-5, Izz=3.2346e-5).

## 1. A subtlety: there are TWO "12-state" models in `~/adaptive_spoofing`

This matters for everything below, so it's worth stating explicitly before
the pipeline design.

1. **The rigid-body plant** (`CrazyflieSystem`, §0): position, Euler angles,
   world-frame velocity, body rates. This is the "real physics" QPS
   ultimately integrates every step.
2. **The outer-loop flat-output chain** — ALSO 12-dimensional, but a totally
   different state: `[pos3, vel3, acc3, jerk3]`, driven by a commanded
   *snap* (4th derivative of position). This is a **virtual/reference
   state**, not the physical one — it's what QPS's snap-tracking law
   (`snap_input.py`) integrates as `xd = AA@x + bb@u`, and what
   `quadrotarium.py:513` then splices the REAL position back into every step
   (`self.x_state[i][0,:] = self.pose_real[i]`). The four candidate
   controllers in `experiments/RQ3/rq3_model_bank.py` (and RQ3's own JAX
   surrogate, `experiments/RQ3/qps_surrogate.py`) are written entirely in
   terms of THIS state, not the rigid-body one.

Between the flat-output chain and the rigid body sits a **third layer**: an
inner geometric SO(3) + PID attitude/position controller
(`QuadcopterObject.obtain_desired_inputs`, Mellinger/Lee-style) that converts
the chain's `[pos,vel,acc]` triple into actual thrust+moments, plus an
exponential-CBF safety QP (`barrier_certificates.py`) that filters the
commanded snap before it's integrated. RQ3's own design doc treats this inner
loop as "a bounded actuation lag... approximately transparent"
(`docs/rq3_transferability_design.md` §4) and does NOT model it in the JAX
surrogate used for synthesis — it only ever perturbs the position term feeding
the OUTER loop.

**This is the central design fork for the whole pipeline** — see §4 Q1.

## 2. The four surrogate controllers (from `rq3_model_bank.py`)

All four predict commanded snap as a linear function of flat-output tracking
errors `e_p = p-p_ref, e_v = v-v_ref, e_a = a-a_ref, e_j = j-j_ref`:

| Candidate | θ | Regressor (Φ) | Notes |
|---|---|---|---|
| `qps_snap_chain` | k_p,k_v,k_a,k_j | [-e_p,-e_v,-e_a,-e_j] | QPS's real law, θ_true=(1680,1066,251,26), poles -5..-8 |
| `pd_pos_vel` | k_p,k_v | [-e_p,-e_v] | cascaded PID, no acc/jerk feedback |
| `pid_pos_vel_i` | k_p,k_v,k_i | [-e_p,-e_v,-∫e_p dt] | PD + position-error integral (needs **memory state**) |
| `indi_jerk` | k_j | [-e_j] | snap_k = snap_{k-1} - k_j·e_j (needs **previous-snap memory**, inherently discrete/incremental) |

All four are **linear in their own θ**, and all four are a *subset* of the
same 5-term family `[-e_p,-e_v,-e_a,-∫e_p dt,-e_j]` (dropping the incremental
`indi_jerk` term, which acts on a *difference* rather than the state itself).
That near-unification is the basis for the parameterization proposal in §4 Q2.

## 3. Proposed pipeline (mirrors this repo's existing separating-input +
refinement machinery — see `quadrotor_fault_diagnosis/quadrotor_separating_input.py`,
`car_fault_diagnosis/car_separating_input.py`)

1. **Scenario = controller hypothesis.** Replace "fault type" with
   "controller structure" as the thing that differentiates scenarios. Four
   `Scenario`s (one per candidate above), reusing the existing
   `_propagate_all_scenarios` / vmap-over-stacked-params pattern — contingent
   on §4 Q2's answer (whether all four can share one traced `emb_system`).
2. **Spoofing signal = the separating input.** QPS already has exactly this
   hook: `quadrotarium.py`'s `spoof_bias`, a position bias added to what the
   controller reads before it computes its control law (`x_ctrl = x_state +
   bias`), leaving the true state/plant untouched — this is the direct analog
   of `car_fault_diagnosis`'s sensor-fault observation model
   (`y = 0.95*[px,py] + [0.2,0.2]`), except here the "fault" is the attacker's
   deliberate design variable. So the optimization variable `u` in
   `separation_loss(u, ...)` becomes the bias trajectory `b(t)` (or
   `qps_surrogate.py`'s bias-*rate*, `db/dt`, if we want it dynamically
   smooth/kill-switch-safe — see `rq3_killswitch.py`'s ramp-unwind logic for
   why bias-rate, not raw bias, is the safer design variable).
   Optimize `b(t)` to MAXIMIZE pairwise reachable-set separation across the
   four controller scenarios — literally `separation_loss` /
   `separation_loss_multistep` unchanged, just with `u` reinterpreted as bias
   and the plant swapped for whichever layer §4 Q1 picks.
3. **React = the existing refinement/monitoring machinery, run online.**
   "Which controller is active" is exactly what `propagate_with_refinement`
   already computes the auxiliary information for: after injecting the
   optimized `b(t)` and observing the drone's real response, intersect the
   observation against each scenario's predicted reachable tube step by step
   (same `jnp.where(has_overlap, refined, x_curr)` pattern already used for
   fault narrowing) — whichever scenario's tube keeps containing the real
   trajectory survives; the others are falsified exactly the way
   `rq3_sme.py`'s "Theta is empty ⇒ candidate rejected" falsification works,
   but from PROACTIVE reachable sets instead of PASSIVE regression residuals.
   This is the "react ... using existing code for model discrimination" the
   task asked for — no new discrimination algorithm, just repointing the
   existing refinement loop's inputs.

## 4. Roadblocks found while digging into the API, and proposed fixes

**R1 — Two-layer state means two candidate "plants" for the optimization,
with very different compile cost.** Doing the discrimination optimization
directly on `CrazyflieSystem` (rigid body) means differentiating through the
SAME inner geometric-PID + CBF-QP + rigid-body cascade every unrolled step;
this repo's own `quadrotor_fault_diagnosis` module already hit a wall here
(gradient compile time ~2s at 1 unrolled step → ~56s/4.3GB at 5, purely from
the `tan`/`1/cos` terms already present in `CrazyflieSystem` too — see its
memory note). The inner CBF QP isn't even naturally differentiable/jittable
at all without a custom implicit-diff rule. **Fix (recommended): do the
discrimination optimization at the flat-output chain layer** (linear
`ṗ=v,v̇=a,ȧ=j,j̇=snap`, no trig, no QP) — exactly what RQ3 itself does in
`qps_surrogate.py`, and exactly what the four candidates are natively defined
over. Keep `CrazyflieSystem` for final physical validation / rendering only
(mirroring how `quadrotor_fault_diagnosis/simulate_and_render.py` sits
downstream of the reachability optimization, not inside it). See Q1.

**R2 — "Controller identity" isn't naturally a jit-friendly interval
parameter.** immrax's vmap-over-scenarios trick
(`_propagate_all_scenarios`) requires every scenario to share ONE traced
`emb_system` and differ only in a stacked `p_interval`. A discrete switch
between 4 different symbolic control laws (different numbers of terms, one
with a `cumsum` integral, one with a previous-step reference instead of the
current state) does NOT fit that mold directly. **Fix (recommended):**
exploit §2's near-unification — write ONE controller law
`snap = rsnap - θ·[-e_p,-e_v,-e_a,-∫e_p dt,-e_j]` with a 5-vector θ, and let
each "scenario" be a MASKED θ interval (zero out the entries the candidate
doesn't use, e.g. `pd_pos_vel` = `θ=(k_p,k_v,0,0,0)`). This makes "which
controller" literally just another `p_interval`, so the whole existing
vmap-over-stacked-params/vmap-over-pairs machinery (§3 pt. 1) applies
unchanged. Cost: (a) `indi_jerk`'s recursive `snap_{k-1}` term doesn't fit
this state-feedback mold cleanly — it needs `snap` itself as an extra memory
state (see R3); (b) this reshapes the bank from "4 distinct hypotheses" into
"one family + a masking convention", which is a modeling choice worth
confirming rather than assuming. See Q2.

**R3 — Two candidates need extra memory state beyond the 12-dim chain.**
`pid_pos_vel_i` needs `∫e_p dt` (+3 states); `indi_jerk` needs `snap_{k-1}`
(+3 states, and is inherently a discrete recursion, not a clean continuous
ODE — awkward to fold into `irx.natemb`'s continuous-time embedding the same
way `nonlinear_chain`/`quadrotor_fault_diagnosis` do). **Fix:** augment the
state to `[pos,vel,acc,jerk, ∫e_p, snap_prev]` (18-dim) for every scenario
(unused memory just sits at its initial value for candidates that don't read
it), and treat the whole chain as **discrete-time at QPS's own dt=0.02s**
(a plain recursion, matching `qps_surrogate.py`'s `qps_closed_loop_step`)
rather than routing it through `irx.natemb`'s continuous embedding + Euler
step at all — the chain dynamics are already exactly what QPS discretizes,
so there's no continuous ODE being approximated here the way there is for
`CrazyflieSystem`. This sidesteps R1's compile-cost concern entirely for the
discrimination layer (linear recursion, nothing to case-split).

**R4 — Gain-scale mismatch.** The real controller's gains
(k=1680,1066,251,26) are 2-3 orders of magnitude larger than what the other
three candidates would plausibly fit (their own PLAN.md analogues in this
repo use O(1) gains). If θ is literally a free optimization/interval
variable, gradient descent on `separation_loss` will be poorly conditioned
across scenarios of such different scale. **Fix:** fix θ intervals from
domain knowledge per candidate (as `rq3_model_bank.py` already does via
`theta_true`/typical tuning ranges) rather than optimizing over θ — θ is a
descriptive parameter of each hypothesis, not something the spoofing
optimizer should be searching over (only `b(t)` should be optimized, exactly
mirroring `alpha` being fixed-per-scenario in `quadrotor_fault_diagnosis`).

**R5 — Bias-only spoof surface.** QPS's `spoof_bias` only corrupts the
3-dim position reading, not the full 12-dim chain — the design variable `u`
in the discrimination optimizer should be a 3-vector (or a short sequence of
them for the multistep/refinement layers), much smaller than existing
modules' 4-dim actuator/moment inputs. No fix needed, just flagging so the
control-box / learning-rate defaults get re-tuned rather than copy-pasted
from `quadrotor_fault_diagnosis`'s thrust-scale numbers.

**R6 — Safety envelope.** QPS's own `rq3_killswitch.py` treats a raw
position-bias jump as itself dangerous (arena bounds, `bias_lim`, ramped
unwind) — any optimized `b(t)` this pipeline proposes should respect a
comparable box constraint (`_project_u`-style clipping, as every existing
`SeparatingInputOptimizer` in this repo already does for its control input)
before being described as usable against a real/simulated QPS run.

## 5. Suggested phased build order (pending answers in §6)

1. `crazyflie_chain_controllers.py`: the masked-θ chain-of-integrator system
   (R2/R3's 18-dim discrete recursion), four `Scenario`s, single-step
   separating spoof-bias optimizer (mirrors `quadrotor_fault_diagnosis` §3).
2. Multistep + refinement layers (mirrors §4/§5 of the same file) — the
   "react" / online discrimination step.
3. Validation harness: replay the optimized `b(t)` through the REAL
   two/three-layer QPS-faithful stack (`CrazyflieSystem` + the actual
   `nominal_snap_input_u`/`obtain_desired_inputs` controllers, transcribed or
   imported from `~/adaptive_spoofing`) to confirm the chain-level
   abstraction's predictions hold up against the full physics — analogous to
   `quadrotor_fault_diagnosis/simulate_and_render.py`.

## 6. Decisions (confirmed with user, 2026-08-09)

- **Q1 fidelity**: flat-output chain only for the optimization; `CrazyflieSystem`
  used downstream for validation/rendering, not inside the optimizer. (R1)
- **Q2 controller param**: unified masked-θ family, one shared traced
  `emb_system`, "which controller" = a `p_interval` mask. (R2)
- **Q3 spoof surface**: position bias only, matching QPS's `spoof_bias` hook
  and RQ3's attacker model. (R5)
- **Q4 reaction scope**: discrimination only for now (identify which
  controller is active from the observed response and falsify the rest).
  Follow-on attack synthesis against the identified controller (RQ3 Stage 2 /
  A1-A2 arms) is explicitly OUT of scope — left as a future to-do, not to be
  built as part of this pass.

Phase 1 (§5 item 1) and Phase 2 (§5 item 2) are now unblocked and match the
scope confirmed above; Phase 3 (validation harness) and any attack-synthesis
extension remain future work.

## 7. Phase 1 implementation (done)

`crazyflie_chain_controllers.py` + `tests/test_crazyflie_chain_controllers.py`
(31 tests, all passing). Implements exactly the Q1/Q2/Q3 decisions above:
`ChainControllerSystem` (18-state, `continuous` evolution, matching this
repo's universal convention — see the module docstring for why `discrete`
evolution was considered and rejected: the chain dynamics are pure
linear/polynomial, so there's no compile-cost reason to deviate, and
`continuous`+`natemb`+`euler_step` is what every other system here already
does), four `Scenario`s sharing one traced `emb_system` via masked-θ
`p_interval`s, and a `SeparatingInputOptimizer`/`optimize_parallel_gpu` pair
mirroring `quadrotor_fault_diagnosis`'s Section 3 exactly, with `u`
reinterpreted as the position-spoof bias.

**Finding from testing (not a bug, a property of the bank):** the four
candidates' gains differ by orders of magnitude (26 vs 1680), so from a
tight initial box they separate within a few Euler steps even at **zero**
spoof bias — the optimizer's `separation_loss` is already ~0 for the full
4-way bank without any attack. The bias only does real work for the
*genuinely hard* pairs: `pd_pos_vel` vs. `pid_pos_vel_i` are **exactly
identical** for one step (the integral state starts at a point, so
`k_i·integ` hasn't diverged from 0 yet — proven by
`test_pd_and_pid_are_identical_after_one_step`) and only separate once the
integral term accumulates over several steps
(`test_pd_and_pid_diverge_after_several_steps`). This means:
- **num_steps must be > 1** for the pipeline to do anything nontrivial
  against a bank that includes an integral-action candidate — a single-step
  spoof cannot possibly discriminate `pd_pos_vel` from `pid_pos_vel_i`, no
  matter how it's optimized. Phase 2 (multistep/refinement) is not optional
  polish; for this specific bank it's necessary for the hardest pair.
- The illustrative placeholder gains for `pd_pos_vel`/`pid_pos_vel_i`/
  `indi_jerk` (§2 — not published anywhere in `rq3_model_bank.py`, only
  `qps_snap_chain`'s are real) directly control how "hard" discrimination
  looks in this demo. Before trusting any of these numbers past a mechanism
  demo, they should be replaced with gains fit from real/simulated benign
  QPS flight (e.g. via `rq3_sme.py`'s own LP fit) rather than left as
  placeholders.

## 8. Phase 2 implementation (done) — output-anticipating refinement + reaction

Added to `crazyflie_chain_controllers.py` (70 tests total, all passing):
Section 4 (multistep unrefined sequence), Section 5 (output-anticipating
intersection-refinement), Section 6 (online reaction/discrimination). Attack
synthesis on the identified controller is explicitly OUT of scope here per
your instruction — left for Phase 3.

**Observability fix (applied retroactively to Phase 1 too).** Building
Section 5/6 surfaced that Phase 1's `separation_loss` was maximizing
separation over the FULL 18-dim state, including `integ`/`cmd` —
controller-internal memory a real observer never sees. That's not just an
"output-anticipating" gap in the new sections, it was already a latent
correctness issue in Phase 1 (those hidden dims existed from the start).
Fixed by adding `observed_output()` (projects to the 12 physically-observable
dims) and switching `separation_loss`/`evaluate` (Section 3) to use it too,
so the whole pipeline is now consistently "what can actually be seen," not
just the new sections. All Phase 1 tests still pass unchanged after the fix.

**Section 5 — output-anticipating refinement.** `propagate_with_refinement`
mirrors `quadrotor_fault_diagnosis`'s vmap-over-pairs pattern exactly, with
one structural change: refinement only ever intersects the OBSERVABLE
12-dim sub-block of a pair's state; the hidden 6 dims (`integ`, `cmd`) are
carried forward unrefined per scenario, since there's no way to infer a
hidden state from an output-space intersection without an inverse map that
doesn't exist here. This is the correct generalization of
`quadrotor_fault_diagnosis`'s identity-observation case, not an
approximation of it.

**Section 6 — reaction — and a real bug it caught.** `discriminate_controller`
propagates each candidate forward under the actually-injected bias sequence
and checks whether the true observed output stays inside its predicted
tube, falsifying candidates that don't (the reachable-set analog of
`rq3_sme.py`'s "Theta is empty ⇒ rejected"). First implementation refined a
surviving candidate's observable state to an EXACT point (`lower=upper=y_true`)
at every consistent step. This looked right but **falsified the TRUE
generating controller** (`qps_snap_chain`) at step 4 of a 5-step demo — root
cause: `simulate_true_trajectory` (a plain point rollout) and the interval
propagation through `natemb` are two independently-coded Euler integrations
of the same math, so they don't reproduce each other bit-for-bit; collapsing
to an exact-width point after every successful step compounds ~1e-7 float32
rounding until it exceeds a truly zero-width box. Fixed with `w_bar`
(default `1e-5`) — refine to `[y_true-w_bar, y_true+w_bar]`, not a point,
and add the same slack to the containment check. This isn't a numerical
hack bolted on after the fact: it's exactly `rq3_sme.py`'s own `w_bar`
(bounded, not stochastic, measurement/model-mismatch budget) — the fix is
also the theoretically correct model, we just hadn't included it yet.
`test_true_controller_always_survives` (parametrized over all 4 candidates
as the "true" one) is the regression guard: any future change that makes
the true generator falsifiable by its own trajectory fails loudly.

**Remaining for Phase 3 (not built, per your instruction):** attack
synthesis against the identified survivor(s) (RQ3's own Stage 2 / A1-A2
comparison arms), and the Phase-3-item-3 validation harness from §5 that
replays the optimized bias sequence through the REAL QPS/`CrazyflieSystem`
stack to confirm the chain-level abstraction's predictions hold up.

## 9. Phase 3 validation harness (done, minus attack synthesis)

Synthesized an actual separating spoof bias, validated it against the REAL
QPS simulator (not just our own point-simulation ground truth), and produced
a video. Scripts (run in this order):

1. `synthesize_spoof.py` (immrax-venv) — runs `optimize_refined_gpu` over
   the full 4-controller bank at a 30-step (0.6s) horizon, long enough for
   the hard pair (`pd_pos_vel`/`pid_pos_vel_i`) to separate (Sec 7). Saves
   `results/synthesized_spoof.npz`. Sanity-checked online: `discriminate_controller`
   uniquely identifies each of the 4 candidates as ground truth from its own
   generated trajectory.
2. `smooth_and_verify_spoof.py` (immrax-venv) — the raw GD solution jumps by
   up to 0.54m in a single 0.02s step (no smoothness penalty in the
   optimizer). QPS's own `rq3_killswitch.py` explicitly treats a bias jump
   like that as itself dangerous. Moving-average-smooths `u_seq` (window=5,
   max step drops to 0.087m) and RE-VERIFIES discrimination still holds
   before anything touches the real simulator — it does, unique survivor for
   all 4 candidates unchanged.
3. `plot_separation.py` (immrax-venv) — `results/pairwise_overlap.png`
   (all 6 pairs' observed-output overlap collapsing to ~0, optimized vs.
   zero-bias baseline — the hard pair visibly needs the full window, the
   other 5 separate almost immediately), `results/reachable_tubes.png`
   (per-controller position/velocity reachable tubes diverging), and
   `results/spoof_bias_signal.png`.
4. `run_qps_validation.py` (dedicated **`~/qps-venv`**, NOT immrax-venv —
   QPS needs `cvxopt`/`control`/`matplotlib<3.10`, none of which belong in
   the JAX venv; `setup.py`'s unconstrained `matplotlib` dependency initially
   pulled 3.10.9, which crashes QPS's own 3D patch drawing code
   (`AttributeError: 'PathPatch3D' object has no attribute '_axlim_clip'`) —
   had to reinstall matplotlib to the `<3.10` `requirements_venv.txt` range).
   Flies the REAL QPS Crazyflie (real rigid-body plant, real
   `nominal_snap_input_u` controller, real CBF safety filter) from an offset
   start to a `[0,0,1]` hover, injects the smoothed spoof bias once settled,
   records `get_estimated_states()` (which turns out to already be exactly
   `[p,v,a,j]` — identical layout to our own `observed_output`, no
   reformatting needed) and a video via ffmpeg.

   **Bug found and fixed:** `Quadrotarium.quad_mission_state` defaults to
   `"HOVERING"` before any `set_poses`/`step()` call, so a settle loop
   written as `while state != "HOVERING": step()` exits immediately without
   commanding the quad anywhere — the first run "settled" after 0 steps at
   the START pose, 0.37m from the actual hover target. Fixed by always
   taking at least one step before checking the (now real,
   waypoint-logic-derived) state.
5. `run_discrimination.py` (immrax-venv) — feeds the REAL recorded
   trajectory into `discriminate_controller`. **Two more real mismatches
   found and fixed, not assumed away:**
   - `x0_ivl` must center on the REAL settled state (nonzero residual
     jerk, ~0.05-0.18), not the zeros `synthesize_spoof.py` assumed at
     design time — reacting with a zero-centered prior mismatches the real
     trajectory from the first step.
   - `w_bar=1e-5` (calibrated for float32 rounding between two idealized
     computations of the SAME equations, Sec 8) falsified the TRUE
     controller at step 1 against real data. Root cause: QPS's real closed
     loop runs a CBF safety filter and real physical/inner-loop dynamics
     our chain abstraction deliberately doesn't model (Sec 6 Q1) — genuine,
     honest model-mismatch, not numerical noise. Swept `w_bar` from 1e-5 to
     1.0: `w_bar=1e-3` is the smallest value where `qps_snap_chain`
     survives, and the other three are falsified at step 0 regardless of
     `w_bar` across that entire range — so this isn't "widen until the test
     is vacuous," the true/false gap is real and comfortably larger than
     the model-mismatch scale.

**Result:** `results/discrimination_result.json` —
`survivors=["qps_snap_chain"]`, the other three falsified at step 0,
against a trajectory flown by the actual QPS simulator (not our own
surrogate). `results/qps_spoof_flight.mp4` is the flight video;
`results/qps_reference_log.json` records the position/velocity reference
(hover keypoint, start pose, settled pose, and the full observed
position/velocity trace during the run).

**Still not built (Phase 4, if wanted):** attack synthesis against the
identified controller, and generalizing beyond the single static-hover
reference this whole pipeline assumes.

## 10. Video overlay: reachable boxes + zoom

Two more scripts on top of §9: `compute_reachable_boxes.py` (immrax-venv)
propagates all 4 candidates' own unrefined position tube (no cross-candidate
intersection, no falsification -- just "where would the drone be if
controller X were true") through the same x0/u_seq `run_discrimination.py`
uses, and saves position-only (indices 0:3) lower/upper per step. Reused
`run_qps_validation.py` (qps-venv) then draws each as a colored 3D wireframe
box (`Line3DCollection`, 12 edges) on the SAME `q.axes` the real drone
renders on, and switches the camera to a fixed ~0.37m half-width window
centered on the hover setpoint once the attack starts (the boxes are cm-to-dm
scale against a ~0.2-0.3m drone body -- see `quadcopter_plot.py`'s
`radius0`/`radius1` -- invisible at the full-arena view used for the
settle-phase flythrough).

Confirmed visually (extracted frames mid- and late-attack): early in the
attack all 4 boxes are too small (~0.02-0.03m) to see against the drone body
-- correct, matches Sec 7's finding that separation takes time to develop,
not a rendering bug. By late attack, the real drone sits inside a large blue
`qps_snap_chain` box while a much smaller, nearly-coincident orange/green
box (`pd_pos_vel`/`pid_pos_vel_i` -- consistent with them being
near-indistinguishable, Sec 7) and a tiny red `indi_jerk` box sit elsewhere
-- exactly the discrimination signal made visible.

Since QPS's simulation is deterministic (no randomness in the controller or
plant), the boxes (computed from a prior run's settled state) realigned with
zero drift on the next run -- the script prints a warning if a future run's
settled state ever diverges from what the boxes assume by >1cm.

## 11. Direct waypoint control (no GPS spoofing)

**Objective:** achieve the same controller discrimination outcome as the
GPS-spoof pipeline (§§7-10), but through legitimate flight control — a
designed sequence of waypoints via QPS's actual `set_poses()` API — with no
sensor deception at all.

**Math-equivalence finding:** in `ChainControllerSystem.f`, the tracking
error is `e_p = (pos + b) - rp`, which is algebraically identical to
`e_p = pos - (rp - b)`. "Corrupt what the controller perceives" (bias `b`)
and "move the commanded target" (`rp`) enter the closed-loop ODE identically.
The existing discrimination machinery (`separation_loss_multistep`,
`propagate_with_refinement`, `discriminate_controller`) therefore already
computes the correct dynamics for a legitimate-waypoint framing; what changes
is:
  (1) The reference is now genuinely **time-varying** (a real flown mission,
      not a fixed hover + tiny bias);
  (2) Box constraints are arena-scale (±0.5m x/y, ±0.3m z around the
      nominal cruise line), not stealth-scale (±0.3m);
  (3) Deployment goes through QPS's actual `set_poses()` waypoint API
      instead of `spoof_bias`.

**Confirmed decisions:**
- Reference fidelity: full smooth polynomial — a JAX transcription of QPS's
  `nominal_traj.py` (`traj_coeffs_from_state_and_waypoint`/`trajectory_generator`),
  validated bit-for-bit against a NumPy oracle.
- Mission profile: RQ3's cruise line P0=(-1.2,0,1.0)→P1=(1.2,0,1.0), K=5
  intermediate waypoints, T_hop=4.0s (QPS's default).
- Real-QPS deployment via `q.set_poses()` + `q.waypoint_time = T_hop`.
- Structure: new modules that import/reuse helpers from
  `crazyflie_chain_controllers.py` — the GPS-spoof pipeline stays untouched.

**New files:**
- `crazyflie_waypoint_trajectory.py` — JAX-differentiable 7th-order
  polynomial trajectory generation, mirroring `nominal_traj.py`. Functions:
  `traj_coeffs_from_waypoint`, `sample_reference`, `build_mission_reference`.
- `crazyflie_waypoint_controllers.py` — `WaypointTrackingSystem` (same 18-state
  layout, ref15 passed as `u` per step instead of baked in), scenario
  construction, waypoint-space optimizer (`optimize_waypoints_gpu`),
  and `discriminate_controller_waypoint`.
- `tests/test_crazyflie_waypoint_trajectory.py` — 25 tests (NumPy oracle
  validation, boundary conditions, differentiability).
- `tests/test_crazyflie_waypoint_controllers.py` — 34 tests (dynamics match,
  embedding sanity, scenario construction, loss/optimizer, discrimination
  regression guard).
- `synthesize_waypoints.py` — optimizer script, saves
  `results/synthesized_waypoints.npz`.
- `run_qps_waypoint_validation.py` (qps-venv) — flies real QPS simulator
  with `set_poses()`, records trajectory + video.
- `run_waypoint_discrimination.py` (immrax-venv) — feeds real trajectory
  into `discriminate_controller_waypoint`.
- `plot_waypoint_separation.py` — pairwise overlap, reachable tubes, 3D
  waypoint visualization.

**Design notes:**
- The loss function uses sum-of-per-axis-overlap-widths (not volume product)
  for gradient stability over long multi-hop horizons where tubes separate
  quickly in most dimensions.
- The placeholder gains for `pd_pos_vel`/`pid_pos_vel_i` (k_p=400) are
  marginally unstable on 4-second hops — they diverge beyond ~150 Euler steps.
  The real QPS controller (`qps_snap_chain`, k_p=1680) is stable because its
  additional acceleration/jerk feedback terms provide damping. This is an
  expected property of the placeholder gains (see §7), not a pipeline bug —
  for real deployment, gains should be fit from data (`rq3_sme.py`).
- `w_bar` for the discrimination test was planned at 1e-3 (same as the spoof
  pipeline's real-QPS validation in §9); §12 below found this needed revising
  once actually run against a real recorded flight.

## 12. Real-QPS deployment, executed and verified (2026-08-10)

Ran the full pipeline end-to-end: `synthesize_waypoints.py` (immrax-venv,
K=5, T_hop=4.0s, 48 restarts × 200 iters, ~15min) →
`run_qps_waypoint_validation.py` (qps-venv, real simulator) →
`run_waypoint_discrimination.py` (immrax-venv). All 129 existing +
new tests pass first. **Result: `qps_snap_chain` (QPS's actual controller)
is correctly and uniquely identified as the sole survivor from a real flight
driven entirely by `q.set_poses()` — no `spoof_bias` set at any point.**
`pd_pos_vel`/`pid_pos_vel_i` falsified at step 141, `indi_jerk` at step 0.

Two real bugs found and fixed in `run_qps_waypoint_validation.py` while
getting the real deployment to run (not present in the JAX-only demo, since
they only manifest against the actual `Quadrotarium` object):
1. **NaN hover state from starting exactly at the target.** The settle phase
   initialized `initial_conditions=P0` and immediately called
   `set_poses(P0)` — the exact case `run_qps_validation.py`'s own docstring
   already warns about (§9's `start_pose` comment): commanding the exact
   starting position never registers as a "new waypoint"
   (`quadrotarium.py`'s 0.05m `min_distance_between_waypoints` check), so
   `hover_positions` stays `None` and `nominal_snap_input_u`'s hover branch
   silently produces NaN. Fixed by starting from a small offset
   (`P0 + [0.1, 0.05, -0.05]`) and settling to `P0` via a real `EXECUTING`
   trajectory first, mirroring `run_qps_validation.py`'s `start_pose` pattern
   exactly.
2. **Wrong state indexing / wrong figure attribute.**
   `q.get_estimated_states()[0]` (selects row 0 = a length-1 slice of just
   `px`, not the 12-vector) should be `q.get_estimated_states()[:, 0]`
   (column 0 = quad 0's full state) — copied incorrectly from
   `run_qps_validation.py`'s correct usage. Same for `q.fig.savefig(...)`,
   which should be `q.figure.savefig(...)` (no `fig` attribute exists on
   `Quadrotarium`). Both silently produced garbage/crashed rather than
   raising an obviously-wrong error, which is why they weren't caught by the
   JAX-only demo/tests (that path never touches the real `Quadrotarium`).

One `w_bar` retuning, found by sweeping the real recorded flight (mirroring
§9's own w_bar sweep methodology exactly): at the previously-planned
`w_bar=1e-3`, the TRUE controller (`qps_snap_chain`) was itself falsified at
step 41 — by ~1.1mm/s³ in one jerk dimension, the same class of honest
chain-abstraction-vs-real-QPS-physics mismatch already documented in §9 (no
CBF/inner loop modeled at this layer), just larger here because this
mission is more aggressive (20s, ~2.4m straight-line span, 5 hops) than §9's
single slow hover correction. Swept `w_bar` from 1e-3 to 1e-1:
`w_bar=4e-3` is the smallest value where `qps_snap_chain` survives the
*whole* mission while the other three remain cleanly falsified with margin
(`pid_pos_vel_i` starts surviving too at `w_bar=1e-2`, everything survives
vacuously by `w_bar=2e-2`) — not "widen until everything passes." Updated
`run_waypoint_discrimination.py`'s `W_BAR` constant accordingly, with the
sweep documented in its module docstring.

Artifacts: `results/synthesized_waypoints.npz`,
`results/qps_waypoint_observed_trajectory.npz`,
`results/qps_waypoint_flight.mp4`, `results/qps_waypoint_reference_log.json`,
`results/waypoint_discrimination_result.json`.
