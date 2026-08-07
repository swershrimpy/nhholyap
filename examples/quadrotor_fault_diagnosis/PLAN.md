# Plan: 12-DOF Quadrotor — Active Actuator-Fault Diagnosis

Companion module to `examples/admire/admire_separating_input.py` and
`examples/nonlinear_chain/nonlinear_chain_separating_input.py`, built on the
same immrax natural-embedding separating-input machinery, applied to the
standard 12-state rigid-body quadrotor model with per-actuator faults on the
four virtual control inputs (thrust + roll/pitch/yaw moment).

This document was written after an `AskUserQuestion` clarification round;
see "Decisions" below.

## Decisions

- **Velocity frame: body-frame `[u,v,w]`** (thesis-faithful form of the
  model, includes Coriolis-like `ω×V` coupling in translational dynamics) —
  chosen over a simpler inertial-frame-velocity variant.
- **Parameters: standard literature defaults**, not fetched from the KTH
  thesis PDF directly (that fetch was not carried out): `m=0.468 kg`,
  `g=9.81 m/s²`, `Ixx=Iyy=4.856e-3 kg·m²`, `Izz=8.801e-3 kg·m²` — values
  that circulate with this exact model across most quadrotor-control
  code/papers (e.g. Luukkonen's widely-reproduced derivation of the same
  Newton-Euler model). Documented here as assumptions; trivially
  overridable via `QuadrotorSystem.__init__` kwargs later if exact
  thesis-table values are needed.
- **Trig kinematics are not a numerical-soundness risk**:
  `examples/admire/admire.py` already differentiates through
  `sin`/`cos`/`tan` (including `1/cos(theta)` and `tan(theta)*(...)` terms
  in its own Euler-angle kinematics) inside `irx.natemb(...)` successfully
  in this exact repo, so immrax's natural embedding is proven to handle
  this model's trig terms soundly.
- **No sensor-fault layer** (per the prompt): dropped `beta`/`xi`/
  `observed_output` entirely rather than keeping unused/no-op machinery —
  separation loss operates directly on the propagated 12-state interval.

## 1. Dynamics model

**State** `x = [x, y, z, φ, θ, ψ, u, v, w, p, q, r]` (12-dim): inertial
position, Euler angles (roll/pitch/yaw), body-frame linear velocity,
body-frame angular velocity.

**Control** `u = [U1, U2, U3, U4]`: total thrust (N) + roll/pitch/yaw
moment (N·m) — already-mixed "virtual" actuator commands; no rotor-speed
mixing matrix is modeled (matches the prompt's `u1..u4` framing).

**Fault params** `p = alpha = [α1, α2, α3, α4]`, `αi=1` nominal, applied as
`u_eff = alpha * u` (elementwise) before it enters the dynamics — the same
idea as `admire.py`'s `jnp.diag(p) @ u`, diagonal-by-construction here since
each `Ui` independently drives exactly one branch of the dynamics below, so
no explicit mixing matrix is needed.

**Equations** (standard body-frame Newton-Euler quadrotor model, ZYX Euler
angles, rotation body->inertial `R = Rz(ψ)Ry(θ)Rx(φ)`):

```
sφ,cφ = sin(φ),cos(φ)   sθ,cθ,tθ = sin(θ),cos(θ),tan(θ)   sψ,cψ = sin(ψ),cos(ψ)
U1,U2,U3,U4 = alpha * u

# position kinematics: inertial velocity = R_body->inertial @ [u,v,w]
ẋ = cθcψ·u + (sφsθcψ - cφsψ)·v + (cφsθcψ + sφsψ)·w
ẏ = cθsψ·u + (sφsθsψ + cφcψ)·v + (cφsθsψ - sφcψ)·w
ż = -sθ·u + sφcθ·v + cφcθ·w

# Euler-angle kinematics
φ̇ = p + sφ·tθ·q + cφ·tθ·r
θ̇ = cφ·q - sφ·r
ψ̇ = (sφ/cθ)·q + (cφ/cθ)·r

# body-frame translational dynamics (with Coriolis coupling ω×V)
u̇ = g·sθ - q·w + r·v
v̇ = -g·sφ·cθ - r·u + p·w
ẇ = U1/m - g·cφ·cθ - p·v + q·u

# rotational dynamics (Euler's equations; no rotor gyroscopic term -- there
# is no rotor-speed state since u1..u4 are already-mixed virtual controls,
# so this simplification is unavoidable given the prompt's u1..u4 framing)
ṗ = ((Iyy-Izz)/Ixx)·q·r + U2/Ixx
q̇ = ((Izz-Ixx)/Iyy)·p·r + U3/Iyy
ṙ = ((Ixx-Iyy)/Izz)·p·q + U4/Izz
```

**Hover equilibrium sanity check** (implemented as a test): at
`φ=θ=ψ=0, u=v=w=0, p=q=r=0, u_input=[mg,0,0,0], alpha=[1,1,1,1]`, every
derivative above evaluates to exactly `0` — thrust exactly cancels gravity
(`U1/m - g·1·1 = g - g = 0`) and every coupling/trig term vanishes. A clean,
exact regression check with no floating-point tolerance tuning needed.

**Known singularity (documented, not "fixed")**: `θ̇`/`ψ̇` blow up at
`θ=±90°` (classical Euler-angle gimbal lock). Demo/test initial intervals
and control bounds (below) are chosen to keep `θ` well under that range
over the horizons used; this is an explicit operating-envelope assumption
(same class of caveat `admire.py` implicitly relies on for
small-perturbation flight), not a runtime-enforced guard.

**Compile-cost finding (from implementation, not anticipated in the
original plan)**: `tan(θ)` and `1/cos(θ)` in the Euler-angle kinematics
make this system's natural embedding far more expensive to differentiate
through than `nonlinear_chain`'s polynomial dynamics or `admire`'s plain
(non-divided) `sin`/`cos`/`tan` terms — interval division requires
sign/pole case-splitting, and that compounds badly under reverse-mode AD
across unrolled Euler steps. Measured (single-step layer, single restart):
`num_steps=1` ~2s compile, `num_steps=2` ~16s, `num_steps=5` ~56s and
4.3GB. Per user decision, the dynamics were kept exactly as specified
(no small-angle linearization, no reformulation) and horizons are capped
small instead — single-step/multistep-unrefined layers stay tractable
this way since they already `vmap` propagation across all 5 scenarios
(compiled once, reused, cost roughly independent of scenario count).

The refinement layer needed an actual code fix, not just smaller horizons:
its per-pair loop (`nonlinear_chain`'s original pattern) traces one
`euler_step` call per pair in plain Python, `len(pairs)*2` separately-traced
expensive calls -- didn't finish compiling in 90s even at the minimum
useful `num_steps=2`. Fixed by vmapping the pairwise loop the same way
scenario propagation is already vmapped (stack each pair's `(p_i, p_j)`,
`vmap euler_step` over the pair axis) -- brought `num_steps=2` down to
~22s. See `propagate_with_refinement`'s docstring in the module for detail.

## 2. Fault scenarios (5 total, fixed)

Control dimension is fixed at 4, so there are always exactly `1 + 4 = 5`
scenarios -- no scenario-count parameter (that complexity in
`nonlinear_chain` was specific to its N-scaling task):

```
Nominal          alpha = [1,1,1,1]
ActuatorFault_1  alpha[0] in [lo,hi], others 1   (thrust effectiveness loss)
ActuatorFault_2  alpha[1] in [lo,hi], others 1   (roll-moment authority loss)
ActuatorFault_3  alpha[2] in [lo,hi], others 1   (pitch-moment authority loss)
ActuatorFault_4  alpha[3] in [lo,hi], others 1   (yaw-moment authority loss)
```

Default `alpha_lo=0.5, alpha_hi=0.9`, matching every other module's
actuator-fault-range convention in this repo.

## 3. Algorithmic layers — ported from `nonlinear_chain_separating_input.py`

Same 3 layers as every other module in this repo, near-verbatim port of the
already-optimized, generic machinery (`_run_unrolled_or_loop`,
`_run_unrolled_or_loop_nocheckpoint`, `_plain_unroll`, `_overlap_volume`,
`time_jit`, `_memory_snapshot` copied unchanged):

1. **Single-step**: `euler_step`, `propagate_scenario`,
   `_propagate_all_scenarios`, `separation_loss` (`C(5,2)=10` pairwise
   overlaps -- small and safe, no risk of the combinatorial compile blowup
   hit during `nonlinear_chain`'s scenario-count sweep since scenario count
   is fixed at 5 here), `SeparatingInputOptimizer`, `optimize_parallel_gpu(_rejit)`.
2. **Multistep unrefined**: `_propagate_history`,
   `propagate_scenario_multistep`, `separation_loss_multistep`,
   `MultistepSequenceOptimizer`, `optimize_multistep_gpu(_rejit)`,
   `optimize_multistep`.
3. **Multistep refined**: simpler than `nonlinear_chain`'s version -- with
   no sensor fault, the observation map is the identity, so refinement is a
   **direct state-interval intersection** between a scenario pair's current
   state intervals (`y_int = intersect(x_i, x_j)`, refine both to
   `x_i ∩ y_int` / `x_j ∩ y_int` when they overlap, else carry forward
   unrefined), then propagate one Euler step. Same
   `jax.checkpoint`-wrapped loop shape as before, flat-array
   `(n_pairs, 2*12)` carry.

**Control box**: `u1 ∈ [0.5·mg, 1.5·mg]` N (thrust, centered on hover,
`mg ≈ 4.59 N` with the defaults above), `u2,u3,u4 ∈ [-0.02, 0.02]` N·m
(moments -- sized so full-authority input produces a modest attitude-rate
change over the short horizons used here, staying well clear of gimbal
lock; a tunable default, not a physical spec). `_project_u` clips per
channel via a `(4,)` lo/hi array, not a single scalar box.

Multi-start GD initialization is centered at hover `[mg, 0, 0, 0]` rather
than a generic `0.5`/`[-1,1]`-box default, since `u1=0` is physically
degenerate (free-fall) and a poor starting point for gradient descent.

## 4. File layout

```
examples/quadrotor_fault_diagnosis/
  PLAN.md                                  (this file)
  quadrotor_separating_input.py            module
  tests/
    test_quadrotor_separating_input.py
```
