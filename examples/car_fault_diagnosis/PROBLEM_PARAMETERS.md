# Problem Parameters: `car_fault_diagnosis`

Catalogue of the control horizon, control-input bounds, disturbance
(fault-parameter) bounds, and initial-state interval used by
`car_separating_input.py`'s three separating-input methods and by
`success_rate_analysis.py`'s 24-config sweep. All values pulled directly
from source, not re-derived.

## System

State `x = [px, py, phi]` (position, heading). Control `u = [v, omega]`
(linear/angular velocity). Fault parameter `p = [alpha]` (actuator
authority, `alpha=1` = nominal). Integration: forward Euler via immrax's
natural embedding (`euler_step`, `car_separating_input.py:277`).

`dt = 0.1` s, fixed across every method and the whole sweep
(`success_rate_analysis.py:110`).

## Control horizon / steps available to the control law

The three methods give the optimizer different amounts of control
authority over the same wall-clock horizon:

| Method | Free control decisions | Steps held per decision | Total Euler steps | Horizon (s) |
|---|---|---|---|---|
| Single-step | 1 (`u`, shape `(2,)`) | held constant for all steps | 8 (`SINGLE_STEP_NUM_STEPS`) | 0.8 |
| Multistep unrefined | 10 (`u_seq`, shape `(10,2)`, `UNREFINED_NUM_SEGMENTS`) | 2 (`UNREFINED_STEPS_PER_SEGMENT`) each | 20 | 2.0 |
| Multistep refined | 8 (`u_seq`, shape `(8,2)`, `REFINED_NUM_STEPS`) | 1 each | 8 | 0.8 |

- Single-step: one control vector optimized, then propagated through
  `propagate_scenario`'s 8-step `fori_loop` unchanged
  (`car_separating_input.py:291-297`; tiled back out to 8 steps for
  history/plotting via `_as_u_seq`, `success_rate_analysis.py:445-448`).
- Unrefined: `separation_loss_multistep` treats `u_seq` as one control per
  *segment*, each segment = `steps_per_segment` Euler steps
  (`car_separating_input.py:408-425`) — the control can change 10 times
  but the state/loss is only evaluated at segment boundaries.
- Refined: `propagate_with_refinement` takes one control per Euler step
  and re-refines the per-pair state intervals after every step
  (`car_separating_input.py:622-679`) — the finest-grained control and the
  only method with per-step (not per-segment) state refinement.

Note single-step and refined share the same 0.8 s horizon but differ in
how many independent decisions the optimizer gets over it (1 vs. 8).

## Control-input bounds

Box constraint enforced by `_project_u` (`jnp.clip`) after every gradient
step, for all three methods:

```
v, omega ∈ [-1, 1]        (car_separating_input.py:76-83, _U_LO/_U_HI)
```

## Disturbance (fault-parameter) bounds

Three scenarios, all sharing `CarNomActSystem`'s dynamics
(`px_dot = v*cos(phi)`, `py_dot = v*sin(phi)`, `phi_dot = alpha*omega`):

| Scenario | `alpha` (actuator authority) | Observed output `y = obs_scale*[px,py] + obs_offset` |
|---|---|---|
| Nominal | point value 1.0 (`icentpert(1.0, 0)` — zero-width interval) | `obs_scale=1`, `obs_offset=[0,0]` |
| Actuator Fault | interval `[alpha_lo, alpha_hi]`, swept | `obs_scale=1`, `obs_offset=[0,0]` |
| Sensor Fault | point value 1.0 (same as Nominal — dynamics identical) | `obs_scale`, `obs_offset` swept |

`success_rate_analysis.py`'s sweep varies the fault severity across 2
levels per fault type (`ACTUATOR_RANGES`, `SENSOR_SEVERITIES`,
lines 101-105):

```
alpha ∈ [0.0, 0.5]    (wide/easy — far from nominal alpha=1)
alpha ∈ [0.35, 0.65]  (narrow/hard — closer to nominal, harder to detect)

sensor: offset=(0.2, 0.2),   scale=0.95   (moderate/easy)
sensor: offset=(0.03, 0.03), scale=0.99   (subtle/hard — near-identity map)
```

(Outside the sweep, `create_scenarios`'s defaults are
`alpha ∈ [0.0, 0.5]`, `sensor_offset=(0.2,0.2)`, `sensor_scale=0.95` —
`car_separating_input.py:203-207`.)

## Initial-state interval

`x0_ivl = irx.icentpert(x0_center, jnp.full(3, x0_width))`
(`success_rate_analysis.py:169`, same half-width applied to all 3 state
dims: px, py, phi). Sweep varies center (3 values) and half-width
(2 values), lines 95-100:

```
x0_center ∈ {(0.1, 0.1, 0.0), (0.5, -0.3, 1.0), (-0.2, 0.4, -0.8)}   [px, py, phi]
x0_width  ∈ {0.02, 0.15}   (half-width, same on all 3 dims -> tight vs. wide interval)
```

E.g. center `(0.1, 0.1, 0.0)` at `x0_width=0.15` gives
`px ∈ [-0.05, 0.25]`, `py ∈ [-0.05, 0.25]`, `phi ∈ [-0.15, 0.15]`.

## Sweep grid size

`3 x0_centers × 2 x0_widths × 2 actuator_ranges × 2 sensor_severities = 24
configs` (`success_rate_analysis.py:486-488`), each run through all 3
methods with `NUM_RESTARTS=100` random control-sequence initializations
and `NUM_ITERS=30` gradient-descent steps (`learning_rate=0.05`).

See the investigation notes in project memory (`success_rate_analysis`
entry) for how `x0_width=0.15` and the subtle sensor severity correlate
with 0% success across all three methods — that's this table's wide/hard
rows hitting a genuine reachable-set-overlap floor, not an optimizer bug.
