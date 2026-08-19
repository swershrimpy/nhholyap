# Firmware controller equations behind `results/firmware_*`

Exact mathematical definition of the four candidate controllers used to
produce `results/firmware_pairwise_overlap.{png,pdf}`,
`results/firmware_spoof_bias.{png,pdf}`, and
`results/firmware_simulated_trajectories*.png` — `cf_pid`, `cf_mellinger`,
`cf_indi`, `cf_brescianini`.

**Provenance.** Every number below is transcribed, not hand-derived: it was
pulled by importing and executing
`~/adaptive_spoofing/experiments/RQ3_transferability_pipeline/rq3_crazyflie_surrogates.py`
(the project's own source of truth for these gains, itself transcribed from
the real Crazyflie firmware — `libs/CrazySim/crazyflie-firmware` at the
commit the submodule pins — not tuned) and printing the constructed
matrices/vectors directly, so this file matches what actually ran, not a
re-derivation that could drift from it.
`examples/adaptive_spoofing/crazyflie_firmware_controllers.py` is the JAX/
immrax port of the same equations (§2 below); it is cross-checked against
`rq3_crazyflie_surrogates` at random states in
`tests/test_crazyflie_firmware_controllers.py`.

Two representations are given for each controller: the **firmware-level
cascade**, in the natural units and gains the C source itself uses (§3–6,
"Firmware law"), and the **affine virtual-control regressor**,
`u_virtual = u_offset + Φᵤθ`, that is what `crazyflie_firmware_controllers.py`
actually evaluates every tick and what generates the reachable-set /
trajectory plots (§3–6, "Regressor form"). The two are algebraically the
same law; §1.3 explains the reduction connecting them.

---

## 1. Shared quantities

### 1.1 State, control, reference

Physical state `x ∈ ℝ¹²` (indices as in the code):

| Slice | Meaning |
|---|---|
| `x[0:3]` | position `p = (p₀,p₁,p₂)` |
| `x[3:6]` | velocity `v = (v₀,v₁,v₂)` |
| `x[6:9]` | attitude `(roll, pitch, yaw)`, hover-linearized |
| `x[9:12]` | body rates `(p_rate, q_rate, r_rate)` |

Control `u = bias ∈ ℝ³`: the attacker's position-spoof bias. The
controller reads `p + bias` wherever it reads position; the plant always
integrates the true `p`. Concretely, every error term below uses
`e_p = r_pos − (p + bias)`, never `r_pos − p`.

Reference `r ∈ ℝ¹⁵ = [pos_sp(3), vel_sp(3), acc_sp(3), jerk_sp(3), snap_sp(3)]`;
for a hover at `pos_sp`, `r = [pos_sp, 0, 0, 0, 0]`
(`hover_reference()`).

Controller memory `mem` (kind depends on the controller — velocity-error
integral, position-error integral, incremental filtered-acceleration state,
or none — see §3–6).

Sample time: `dt = 0.02` s (QPS's tick, `QPS_DT`) for every number in this
file. Gravity `g = 9.81` m/s² (firmware's `GRAVITY_MAGNITUDE`, not the
more precise 9.80665).

### 1.2 Shared hover-linearized plant

**This is a third, distinct 12-state model — not `CrazyflieSystem`.** The
project has (at least) three different "12-state quadrotor" representations
and this is the one these controllers actually act on: `HoverPlant` here is
a **linearized, discrete-time, semi-implicit-Euler affine surrogate**
(`x_next = Ax+Bu`, this section), sharing only the state *layout*
(position/attitude/velocity/rates) with `crazyflie_12d.py`'s
`CrazyflieSystem` — the verified-bit-for-bit-against-QPS **nonlinear**
rigid-body plant used for "final physical validation / rendering only"
(`PLAN.md` §5). It is also unrelated to the flat-output `[pos,vel,acc,jerk]`
chain state `crazyflie_chain_controllers.py` uses. `thrust_accel_gain` and
`torque_angular_accel_gain` below are fitted lumped actuator gains standing
in for "commanded units → physical acceleration," not `CrazyflieSystem`'s
real mass/inertia.


All four controllers close the loop around the **same** discrete affine
plant, `x_next = A x + B u_virtual`, where `u_virtual ∈ ℝ⁶ =
[τ_roll, τ_pitch, τ_yaw, unused, unused, a_z]` — three body-axis torque
commands, two structurally-reserved-but-never-written slots, and one
vertical (mass-normalized thrust) acceleration command, in that column
order (confirmed directly from `B`'s nonzero entries below, not assumed —
column 0 drives `roll`/`p_rate`, column 1 drives `pitch`/`q_rate`, column 2
drives `yaw`/`r_rate`, columns 3–4 are all-zero in every row, column 5
drives `v₂`/`p₂`). Effectively this is the classic 4-input quadrotor
control `[τ_roll, τ_pitch, τ_yaw, thrust]`, carried in a 6-slot vector for
code uniformity with `Φᵤ`'s 6 rows (see §1.3) — **not** `[a₀,a₁,a₂,τ_roll,
τ_pitch,τ_yaw]` as an earlier draft of this file mislabeled it. `A, B`
come from **one step of
semi-implicit Euler** — rates, then attitude from the new rates, then
velocity from the new attitude, then position from the new velocity, in
that order — not fully explicit Euler: the source's own docstring notes
that fully explicit integration destabilizes `cf_brescianini`'s cascade
(oscillates at ≈0.6 Hz) because four stacked half-tick lags are enough to
break a continuous-time-stable design; semi-implicit costs nothing (each
stage is still linear, so the composition is still exactly affine in θ) and
removes the artificial lag.

Two lumped actuator gains stand in for "commanded units → physical
acceleration" (not firmware numbers — the interface between the affine
surrogate and the real vehicle):
`thrust_accel_gain (k_t) = (20.0, 20.0, 20.0)`,
`torque_angular_accel_gain (k_τ) = (50.0, 50.0, 20.0)`.

At `dt = 0.02`, evaluated directly from `HoverPlant.matrices()`:

```
A =
[ 1        0        0        0.02     0        0        0        -0.003924  0        0        -0.000078  0      ]
[ 0        1        0        0        0.02     0        0.003924  0        0        0.000078   0        0      ]
[ 0        0        1        0        0        0.02     0        0        0        0           0        0      ]
[ 0        0        0        1        0        0        0        -0.1962   0        0        -0.003924  0      ]
[ 0        0        0        0        1        0        0.1962    0        0        0.003924   0        0      ]
[ 0        0        0        0        0        1        0        0        0        0           0        0      ]
[ 0        0        0        0        0        0        1        0        0        0.02        0        0      ]
[ 0        0        0        0        0        0        0        1        0        0           0.02     0      ]
[ 0        0        0        0        0        0        0        0        1        0           0        0.02   ]
[ 0        0        0        0        0        0        0        0        0        1           0        0      ]
[ 0        0        0        0        0        0        0        0        0        0           1        0      ]
[ 0        0        0        0        0        0        0        0        0        0           0        1      ]

B =
[ 0         -0.000078  0        0  0  0    ]
[ 0.000078   0         0        0  0  0    ]
[ 0          0         0        0  0  0.008]
[ 0         -0.003924  0        0  0  0    ]
[ 0.003924   0         0        0  0  0    ]
[ 0          0         0        0  0  0.4  ]
[ 0.02       0         0        0  0  0    ]
[ 0          0.02      0        0  0  0    ]
[ 0          0         0.008    0  0  0    ]
[ 1          0         0        0  0  0    ]
[ 0          1         0        0  0  0    ]
[ 0          0         0.4      0  0  0    ]
```
(rows/cols indexed by `[p₀,p₁,p₂,v₀,v₁,v₂,roll,pitch,yaw,p_rate,q_rate,r_rate]`
for `A`, and `u_virtual = [a₀,a₁,a₂,τ_roll,τ_pitch,τ_yaw]` for `B`'s columns).

### 1.3 From the firmware cascade to the affine regressor

Every one of the four controllers reduces, after **(a)** hover-linearizing
the attitude/thrust map and **(b)** capping the inner attitude/rate loop's
bandwidth to the fastest a single `dt = 0.02` s explicit step can represent
(`cap_second_order`, keeping each loop's damping ratio but lowering only
its natural frequency — the real vehicle's inner loop is faster than this,
and that gap is treated as bounded model error, not asserted away), to the
same three-line shape:

```
att_cmd = A_p · e_p + A_v · e_v + A_i · integ        (roll/pitch, [rad])
ω_cmd   = ρ · (att_cmd − attitude)
u       = σ · (ω_cmd − ω)
     ⟹  u = (σρA_p)·e_p + (σρA_v)·e_v + (σρA_i)·integ + (σρ)·(−attitude) + σ·(−ω)
```

so the quantities an affine regressor can identify are the **products**
`σρA_p, σρA_v, σρA_i, σρ, σ` — never the individual firmware gains — which
is why `θ` below is named `kappa_*`/`rho_*`/`sigma_*` (products), not the
raw C constants. `(ρ, σ)` (`_inner_pair`) are **not** transcribed from the
firmware directly (units aren't comparable across controllers — e.g.
Mellinger's inner gain is a PWM-count torque, Brescianini's is a time
constant); instead each controller's inner loop is placed at a fixed
bandwidth separation (`INNER_SEPARATION = 4×`) above its **own** outer
loop, then capped the same way. The vertical (thrust) channel has no
attitude stage in the middle, so its two coefficients are just the capped
second-order pair divided by `k_t` (`_vertical_pair`).

Every controller shares this **regressor → plant** wiring in
`crazyflie_firmware_controllers.py`:

```
u_offset, Φᵤ = <controller-specific virtual-control terms>(x, r, mem, bias)
u_virtual    = u_offset + Φᵤ · θ_true
x_next       = A · x + B · u_virtual
mem_next     = <controller-specific memory update>(x, r, mem, bias)
```

`θ_true` is a **fixed, degenerate interval** (`Interval(lower=θ, upper=θ)`)
— these are real published firmware gains, not something the optimizer
searches over.

---

## 2. `cf_pid` — `position_controller_pid.c`

**Firmware law** (nested, saturated cascade — P on position → clamped
velocity setpoint → PI on velocity → clamped angle → P on attitude → P on
rate):

```
v_sp        = KP_POS · e_p + vel_ff                          KP_POS = (2.0, 2.0, 2.0)
v_sp_c      = clip(v_sp, ±PID_POS_VEL_MAX)                    PID_POS_VEL_MAX = 1.0 m/s
e_v         = v_sp_c − v
integ_v    += dt · e_v
angle_cmd   = −(KP_VEL · e_v + KI_VEL · integ_v) · (π/180)    KP_VEL=(25,25,25), KI_VEL=(1,1,15)
angle_cmd_c = clip(angle_cmd, ±PID_VEL_RP_MAX·π/180)           PID_VEL_RP_MAX = 20°
rate_sp     = KP_ATT · (angle_cmd_c − attitude)                KP_ATT = (6,6,6)
τ           = KP_RATE · (rate_sp − ω)                          KP_RATE = (250, 250, 120)
```

Both clamps (velocity setpoint, commanded angle) are real and modelled —
they are what makes `cf_pid` structurally different from `cf_indi` rather
than a re-gained copy of it. `surv ∈ [0,1]³` is the fraction of the
position error that survives the velocity clamp (`_clip_survival`,
`= v_sp_c/v_sp` elementwise, `1` when unsaturated); the affine form below
is exact on the unsaturated branch.

**Regressor form** (`n_θ = 15`; roll driven by axis-1 error, pitch by
axis-0 with a sign flip, per the plant's `B` rows in §1.2):

```
Φᵤ[0,1]  = surv₁·e_p₁        Φᵤ[0,3]  = −v₁            Φᵤ[0,5]  = integ₁     Φᵤ[0,6]  = −roll     Φᵤ[0,9]  = −p_rate
Φᵤ[1,0]  = −surv₀·e_p₀       Φᵤ[1,2]  = v₀             Φᵤ[1,4]  = −integ₀    Φᵤ[1,7]  = −pitch    Φᵤ[1,10] = −q_rate
Φᵤ[2,8]  = −yaw   (yaw_sp = 0)                                                                     Φᵤ[2,11] = −r_rate
Φᵤ[5,12] = surv₂·e_p₂/k_t,z  Φᵤ[5,13] = −v₂/k_t,z      Φᵤ[5,14] = integ₂/k_t,z
u_offset = [0,0,0,0,0, r₈/k_t,z]        (r₈ = acceleration setpoint, z-axis)
```

Memory update: `integ_next = integ + dt · (v_sp_c − v)` (velocity-error
integral — the signature that separates `cf_pid` from `cf_mellinger`,
whose integral is on *position* error).

**θ_true** (order matches `theta_names`):

| `kappa_p0,1` | `kappa_v0,1` | `kappa_i0,1` | `rho_roll` | `rho_pitch` | `rho_yaw` | `sigma_roll` | `sigma_pitch` | `sigma_yaw` | `lambda_z` | `mu_z` | `iota_z` |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 5.106644 | 2.553322 | 0.102133 | 5.851783 | 5.851783 | 14.629457 | 0.734071 | 0.734071 | 1.835178 | 2.567503 | 1.352700 | 0.811620 |

(`kappa_p0=kappa_p1`, `kappa_v0=kappa_v1`, `kappa_i0=kappa_i1` by symmetry
of the x/y gains; listed once each above, both axes equal.)

---

## 3. `cf_mellinger` — `controller_mellinger.c`

**Firmware law** (one-shot desired force vector, direction gives attitude
— no intermediate velocity setpoint or clamp between loops):

```
F_des = m·a_ref + KP·e_p + KD·e_v + KI·∫e_p dt      KP=(0.4,0.4,1.25), KD=(0.2,0.2,0.4), KI=(0.05,0.05,0.05), m=0.027 kg
attitude_cmd = direction(F_des)      (hover-linearized: divide the horizontal components by g)
rate_sp = KP_ATT_MEL · (attitude_cmd − attitude)     (geometric SO(3) attitude law, hover-linearized to P)
τ = KP_RATE_MEL · (rate_sp − ω)
```

Dividing the force gains by mass turns them into accelerations (`KP/m`,
`KD/m`, `KI/m`) — the natural units the 12-state plant speaks — and the
integral is on **position** error, not velocity error (the signature
separating this controller from `cf_pid`).

**Regressor form** (`n_θ = 12`):

```
Φᵤ[0,0] = e_p₁/g       Φᵤ[0,1] = e_v₁/g       Φᵤ[0,2] = integ₁/g      Φᵤ[0,6] = r₇/g − roll   Φᵤ[0,9]  = −p_rate
Φᵤ[1,0] = −e_p₀/g      Φᵤ[1,1] = −e_v₀/g      Φᵤ[1,2] = −integ₀/g     Φᵤ[1,7] = −r₆/g − pitch Φᵤ[1,10] = −q_rate
Φᵤ[2,8] = −yaw                                                                                   Φᵤ[2,11] = −r_rate
Φᵤ[5,3] = e_p₂/k_t,z   Φᵤ[5,4] = e_v₂/k_t,z   Φᵤ[5,5] = integ₂/k_t,z
u_offset = [0,0,0,0,0, r₈/k_t,z]
```

Memory update: `integ_next = integ + dt · (r_pos − (p + bias))` (position-
error integral).

**θ_true**:

| `alpha_p_xy` | `alpha_d_xy` | `alpha_i_xy` | `alpha_p_z` | `alpha_d_z` | `alpha_i_z` | `rho_roll` | `rho_pitch` | `rho_yaw` | `sigma_roll` | `sigma_pitch` | `sigma_yaw` |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 92.592593 | 46.296296 | 11.574074 | 2.314815 | 0.740741 | 0.092593 | 6.25 | 6.25 | 15.625 | 0.75 | 0.75 | 1.875 |

---

## 4. `cf_indi` — `controller_indi.c` + `position_controller_indi.c`

**Firmware law** (incremental — the command is a running increment against
*measured*, not modelled, acceleration):

```
v_ref = K_ξ · e_p                         K_ξ  = (1.0, 1.0, 1.0)     (position P → velocity reference)
a_ref = K_dξ · (v_ref − v)                K_dξ = (5.0, 5.0, 5.0)     (velocity P → acceleration reference)
a_meas_filt = LowPass₂ₙ_Butterworth(a_meas, cutoff = 8 Hz)            (2nd-order, resampled to one pole at dt)
cmd  += G⁻¹ · (a_ref − a_meas_filt) · dt                              (accumulated, leaked slightly)
```

`a_meas` is not directly in the 12-state; it is reconstructed from the
plant's own attitude rows, `a_meas = (−g·pitch, g·roll, 0)`, and pushed
through the same low-pass the firmware uses
(`α = dt/(τ+dt)`, `τ = 1/(2π·8Hz) ⟹ α ≈ 0.5152` at `dt=0.02`). This
reconstruction — it omits the accelerometer noise the real filter also
sees — is documented in the source as the single largest contributor to
this controller's model-mismatch bound.

**Regressor form** (`n_θ = 11`; `ν_xy` — the coefficient on the
accumulated increment `mem.prev_cmd` — exists in **no other** bank member;
it is the coordinate that structurally separates `cf_indi` from `cf_pid`,
whose outer P-P cascade is otherwise identical):

```
Φᵤ[0,0] = e_p₁    Φᵤ[0,1] = e_v₁    Φᵤ[0,2] = prev_cmd₁   Φᵤ[0,5] = −roll    Φᵤ[0,8]  = −p_rate
Φᵤ[1,0] = −e_p₀   Φᵤ[1,1] = −e_v₀   Φᵤ[1,2] = −prev_cmd₀  Φᵤ[1,6] = −pitch   Φᵤ[1,9]  = −q_rate
Φᵤ[2,7] = −yaw                                                               Φᵤ[2,10] = −r_rate
Φᵤ[5,3] = e_p₂/k_t,z    Φᵤ[5,4] = e_v₂/k_t,z
u_offset = [0,0,0,0,0, r₈/k_t,z]
```

Memory update (two states — `accel_filt`, `prev_cmd`):

```
accel_filt_next = accel_filt + α · (a_meas − accel_filt)
a_ref           = K_dξ · (K_ξ·e_p + r_vel − v)
prev_cmd_next   = (1−α) · prev_cmd + dt · (a_ref − accel_filt_next)
```

**θ_true**:

| `kappa_pp_xy` | `kappa_dd_xy` | `nu_xy` | `kappa_pp_z` | `kappa_dd_z` | `rho_roll` | `rho_pitch` | `rho_yaw` | `sigma_roll` | `sigma_pitch` | `sigma_yaw` |
|---|---|---|---|---|---|---|---|---|---|---|
| 2.279376 | 2.279376 | 0.014356 | 0.25 | 0.25 | 4.472136 | 4.472136 | 11.180340 | 0.678885 | 0.678885 | 1.697214 |

---

## 5. `cf_brescianini` — `controller_brescianini.c`

**Firmware law** (second-order reference model in time constants and
damping ratios — no integral anywhere):

```
a_des     = a_ref + (1/τ²)·e_p + (2ζ/τ)·e_v          τ_xy=0.3s, ζ_xy=0.85;  τ_z=0.3s, ζ_z=0.85
ω_cmd     = (2/τ_rp) · attitude_error                 τ_rp = 0.25 s
τ_torque  = (ω_cmd − ω) / τ_rp,rate                   τ_rp,rate = 0.015 s (roll/pitch), 0.0075 s (yaw)
```

No integrator: under a constant spoof bias, this controller alone reaches
a steady-state position offset of exactly `−bias` with **no wind-up
transient**, while `cf_pid`/`cf_mellinger` fight the bias through their
integrators first — documented in the source as "the cleanest
discriminator in the bank."

**Regressor form** (`n_θ = 10`):

```
Φᵤ[0,0] = e_p₁/g    Φᵤ[0,1] = e_v₁/g    Φᵤ[0,4] = r₇/g − roll   Φᵤ[0,7] = −p_rate
Φᵤ[1,0] = −e_p₀/g   Φᵤ[1,1] = −e_v₀/g   Φᵤ[1,5] = −r₆/g − pitch Φᵤ[1,8] = −q_rate
Φᵤ[2,6] = −yaw                                                    Φᵤ[2,9] = −r_rate
Φᵤ[5,2] = e_p₂/k_t,z   Φᵤ[5,3] = e_v₂/k_t,z
u_offset = [0,0,0,0,0, r₈/k_t,z]
```

Memory update: **none** (`mem_next = mem` — memoryless, `memory_kind =
"none"`).

**θ_true**:

| `omega2_xy` | `two_zeta_omega_xy` | `omega2_z` | `two_zeta_omega_z` | `rho_roll` | `rho_pitch` | `rho_yaw` | `sigma_roll` | `sigma_pitch` | `sigma_yaw` |
|---|---|---|---|---|---|---|---|---|---|
| 69.444444 | 35.416667 | 0.555556 | 0.283333 | 6.25 | 6.25 | 15.625 | 0.75 | 0.75 | 1.875 |

(`omega2_xy = 1/τ_xy²`, `two_zeta_omega_xy = 2ζ_xy/τ_xy` — Brescianini is
written directly in `(τ,ζ)` by the firmware, so unlike the other three
controllers its natural parameters and its affine ones are related by a
*nonlinear* map, which is what `stage_c_theta_names` in the source converts
back and forth.)

---

## 6. Common outer-loop bandwidth, for reference

`(ω_n, ζ)` of each controller's **outer position loop**, before the
inner-loop capping of §1.3 — this is each controller's real behavioral
identity (its response speed/shape to a position error), computed directly
from the firmware gains above:

| Controller | `ω_n` (rad/s) | `ζ` |
|---|---|---|
| `cf_pid` | 2.9259 | 0.7315 |
| `cf_mellinger` | 3.8490 | 0.9623 |
| `cf_indi` | 2.2361 | 1.1180 |
| `cf_brescianini` | 3.3333 (`=1/τ_xy`) | 0.85 |

All four share the same inner-loop separation rule (`INNER_SEPARATION =
4×` the controller's own outer `ω_n`, then capped at `dt=0.02` by
`cap_second_order`) and the same shared plant (§1.2) — the *only* thing
that differs across the bank is the outer-loop law each controller closes
with, which is exactly what these plots are testing whether a spoof bias
can discriminate.
