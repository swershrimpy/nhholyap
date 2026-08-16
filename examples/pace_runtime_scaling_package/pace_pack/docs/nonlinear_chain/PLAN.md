# Plan: Decoupled Cubic-Drift Chain — Active Fault Diagnosis

Companion module to `examples/integrator_chain/integrator_separating_input.py`,
built on the same three-layer separating-input machinery (single-step,
multistep-unrefined, multistep-refined-via-intersection), but with a
genuinely nonlinear, per-channel-heterogeneous drift instead of a pure
integrator chain, and a richer per-channel fault scenario set.

This document was written after an `AskUserQuestion` clarification round;
the choices below are the ones the user selected (recorded in "Decisions"
below with the alternatives that were on the table).

## 1. System

State `x = [x_1, ..., x_N]`, control `u = [u_1, ..., u_N]` (**not** scalar —
one control input per channel, unlike integrator_chain's shared scalar `u`).
Channels are dynamically **decoupled**: `x_i` depends only on `x_i` and
`u_i`, never on `x_j`, `j != i`. Coupling only happens through the shared
fault-diagnosis objective (separating scenarios' reachable sets), exactly
as go2/admire/faulty_car couple otherwise-independent fault hypotheses.

### Per-channel dynamics

```
x_i' = f(x_i) + alpha_i * g(u_i)     for i = 1..N

f(x_i)  = a_i * x_i  -  b_i * x_i**3        (cubic / Duffing-type drift)
g(u_i)  = u_i                                (linear input map)

  =>  x_i' = a_i * x_i - b_i * x_i**3 + alpha_i * u_i
```

Vectorized (no Python loop over N, matching integrator_chain's style):

```python
x_dot = a * x - b * x**3 + alpha * u   # a, b: fixed constants; alpha: fault param (p)
```

- `a_i in [0.5, 1.5]`, `b_i in [0.2, 0.8]` — **fixed per-channel physical
  constants** (drawn once at system-construction time, e.g.
  `np.linspace` or a seeded RNG over `[a_lo,a_hi] x [b_lo,b_hi]`), stored as
  plain arrays on the `System` instance. They are *not* part of `p` —
  they are not something a scenario disputes, they just make channel `i`
  physically different from channel `j` (the "vary with the state/channel"
  requirement from the prompt). With `a_i > 0` the origin is an unstable
  equilibrium for every channel, so — like integrator_chain's undamped
  chain — reachable sets grow without a stabilizing control, giving
  nontrivial (non-degenerate) separation geometry to exploit.
- `alpha_i` — **actuator-fault authority for channel i**, this system's `p`,
  shape `(N,)`, interval-valued. `alpha_i = 1` is nominal (full authority).
  This is the only fault-relevant system parameter, exactly mirroring
  integrator_chain's scalar `alpha` but promoted to one entry per channel
  (see admire's per-actuator fault pattern).

### Why cubic drift, why linear `g`

Both are the `AskUserQuestion`-selected options (alternatives on the table:
saturating `-a*tanh(x)`, pendulum `-a*sin(x)`, quadratic drag
`-a*x*|x|` for `f`; saturating `u_max*tanh(u/u_max)` for `g`). Cubic drift
was preferred because it is polynomial — immrax's natural embedding derives
a sound interval extension automatically from the JAX-traced expression, no
special-cased monotone bound is needed the way `tanh`/`sin` would want one
if bounds got wide. Linear `g` keeps the input map identical in spirit to
integrator_chain's `alpha * u`, isolating "what's new here" to the drift
term and the fault-scenario structure.

## 2. immrax System class

```python
class CubicChainSystem(irx.System):
    """Decoupled per-channel cubic-drift chain with actuator-fault authority alpha.

    State   x = [x_1, ..., x_N]
    Control u = [u_1, ..., u_N]           (shape (N,), NOT scalar)
    Params  p = alpha                      (shape (N,), alpha_i=1 -> nominal)

    Dynamics (vectorized, no Python loop over N):
        x_dot = a * x - b * x**3 + alpha * u
    """

    def __init__(self, N: int, a: jnp.ndarray, b: jnp.ndarray):
        self.evolution = 'continuous'
        self.xlen = N
        self.N = N
        self.a = a   # shape (N,), fixed per-channel linear-drift coefficient
        self.b = b   # shape (N,), fixed per-channel cubic-damping coefficient

    def f(self, t, x, u, p):
        alpha = p
        return self.a * x - self.b * x**3 + alpha * u
```

Matches the go2/ADMIRE/integrator_chain pattern from project memory: no
`super().__init__(xlen=...)` call, `evolution`/`xlen` set directly,
`f(self, t, x, u, p)` signature with no `w`. One `irx.natemb(sys_)` embedding
per distinct `(N, a, b)` triple, cached the same way as
`get_system_and_embedding` in integrator_chain (keyed by `(N, tuple(a),
tuple(b))` instead of just `N`, since `a`/`b` are now part of the traced
system rather than universal constants).

## 3. Sensor / output model

Unchanged functional form from integrator_chain, all `N` states directly
observed:

```
y = beta ⊙ x + xi          (elementwise, per-channel)
```

`beta`, `xi` are per-state interval parameters. `beta` must stay strictly
positive in every scenario (same invertibility requirement as
integrator_chain, needed by the refinement layer's `_invert_observation`).
`observed_output` / `_invert_observation` port essentially verbatim from
integrator_chain — both already operate elementwise over an `N`-vector, so
no change is needed to make them work per-channel.

## 4. Fault scenarios — the actual new design point

Two independent `AskUserQuestion` decisions shape this:

- **Actuator faults**: `alpha_i in [alpha_lo, alpha_hi]` on **exactly one**
  channel at a time, all other channels' `alpha_j = 1`. Scenario set is
  `Nominal + N` single-channel actuator-fault scenarios (chosen over a
  fixed small demo subset) — mirrors admire's "nominal + 10 single-actuator
  faults" pattern exactly, and matches the prompt's "there can be n
  actuator faults for n-dim" literally.
- **Sensor faults**: `beta_i in [beta_lo, beta_hi]`, `xi_i in [-eps, eps]`
  on **exactly one** output channel at a time, all other channels
  `beta_j=1, xi_j=0` — chosen over a single global vector-fault scenario
  (integrator_chain's style) specifically so sensor faults are symmetric
  with the per-channel actuator-fault structure: `N` additional scenarios.

**Total scenario count: `2N + 1`** (1 Nominal + N ActuatorFault_i + N
SensorFault_i). This is new relative to every existing module in this repo
(go2/admire/faulty_car/integrator_chain all use a small fixed scenario
count independent of N) — it's the first module where the scenario count
itself scales with the state dimension. See §6 (Open design questions /
risks) for the consequence of this.

```python
@dataclass
class Scenario:
    name: str
    emb_system: object          # shared across ALL 2N+1 scenarios (a,b are
                                 # fixed system constants, not part of p —
                                 # only p=alpha differs across scenarios)
    p_interval: irx.Interval    # alpha, shape (N,)
    beta: irx.Interval          # shape (N,)
    xi: irx.Interval            # shape (N,)


def create_scenarios(N, a, b,
                     actuator_alpha_lo=0.5, actuator_alpha_hi=0.9,
                     sensor_beta_lo=0.8, sensor_beta_hi=1.2,
                     sensor_xi_bound=0.1) -> List[Scenario]:
    _, emb = get_system_and_embedding(N, a, b)
    ones_N, zeros_N = jnp.ones(N), jnp.zeros(N)

    scenarios = [Scenario("Nominal", emb,
                          point_ivl(ones_N), point_ivl(ones_N), point_ivl(zeros_N))]

    for i in range(N):
        alpha_lo = ones_N.at[i].set(actuator_alpha_lo)
        alpha_hi = ones_N.at[i].set(actuator_alpha_hi)
        scenarios.append(Scenario(f"ActuatorFault_{i}", emb,
                                  irx.Interval(alpha_lo, alpha_hi),
                                  point_ivl(ones_N), point_ivl(zeros_N)))

    for i in range(N):
        beta_lo = ones_N.at[i].set(sensor_beta_lo)
        beta_hi = ones_N.at[i].set(sensor_beta_hi)
        xi_lo = zeros_N.at[i].set(-sensor_xi_bound)
        xi_hi = zeros_N.at[i].set(sensor_xi_bound)
        scenarios.append(Scenario(f"SensorFault_{i}", emb,
                                  point_ivl(ones_N),
                                  irx.Interval(beta_lo, beta_hi),
                                  irx.Interval(xi_lo, xi_hi)))
    return scenarios
```

Because every scenario shares one `emb_system` (only `p=alpha` and the
output map `(beta,xi)` differ), the existing "stack `p_interval`s and
`vmap` a single `_propagate_by_params`" trick from integrator_chain still
applies unchanged across all `2N+1` scenarios — no per-scenario Python loop
over propagation is needed, only over pairwise-overlap bookkeeping.

## 5. Algorithmic layers (ported, not redesigned)

Same three layers as integrator_chain, reused near-verbatim since the
propagation/loss/optimizer machinery is generic over `(x0_ivl, u, scenarios,
dt, num_steps)` and doesn't care that `f` is now cubic or that `u` is
now `(N,)`-shaped instead of `(1,)`-shaped:

1. **Single-step** (`euler_step`, `propagate_scenario`, `separation_loss`,
   `SeparatingInputOptimizer`, `optimize_parallel_gpu`) — control box
   becomes `u in [-1,1]^N` (was `[-1,1]^1`); everything else is a direct
   port.
2. **Multistep unrefined** (`_propagate_history`,
   `separation_loss_multistep`, `MultistepSequenceOptimizer`,
   `optimize_multistep_gpu`) — direct port, `u_seq` shape becomes
   `(num_segments, N)`.
3. **Multistep refined** (`propagate_with_refinement`,
   `refined_overlap_loss`, `optimize_refined_gpu`) — direct port;
   `_invert_observation` is already elementwise over the state vector so it
   needs no change for the larger `N` or the per-channel `beta`/`xi`.

## 6. Open design questions / risks (flagged, not yet resolved)

- **O(N^2) pairwise overlap cost**: `2N+1` scenarios means
  `C(2N+1, 2) ~ 2N^2` pairwise-overlap terms per loss evaluation (vs. a
  constant 6 pairs for integrator_chain/faulty_car's 4 scenarios). For
  small demo `N` (3-5) this is fine (~20-60 pairs); it will need either a
  smaller demo `N`, or restricting `separation_loss` to only the pairs that
  are actually confusable (e.g. `Nominal` vs. each fault, and
  same-fault-type pairs), before pushing `N` much past ~8-10. Flagging for
  the user rather than deciding now.
- **Control dimensionality**: `u` is now `(N,)` instead of integrator_chain's
  `(1,)`. Multi-start GD (`num_restarts x N`) and the refinement loop's
  per-pair carry (`n_pairs x 2N`) both grow with `N`; no algorithmic change
  needed, just more memory per restart, worth a note in the demo's runtime
  estimate.
- **Should there be a "simultaneous fault" scenario** (one actuator fault +
  one sensor fault at once, per integrator_chain's `Simultaneous Fault`)?
  Not requested by the prompt and not asked in the clarification round —
  left out of the `2N+1` baseline; can be added later as
  `SimultaneousFault_{i}_{j}` if wanted (would push scenario count to
  `O(N^2)` on its own, so likely opt-in via a flag rather than default-on).
- **`a`, `b` provenance**: plan above draws them once at construction
  (deterministic default, e.g. `jnp.linspace`, or a seeded RNG) rather than
  hardcoding one shared `(a,b)` for all channels — needs a concrete default
  policy decided during implementation (linspace is simplest/most
  reproducible without a seed argument; RNG is more "generic heterogeneous
  chain" flavored). Recommend `jnp.linspace(a_lo, a_hi, N)` /
  `jnp.linspace(b_lo, b_hi, N)` for determinism and zero extra API surface.

## 7. File layout

```
examples/nonlinear_chain/
  PLAN.md                                  (this file)
  TASKS.md                                 (execution checklist)
  nonlinear_chain_separating_input.py       (module, mirrors integrator_separating_input.py)
  tests/
    test_nonlinear_chain_separating_input.py
```
