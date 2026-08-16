"""
12-State Rigid-Body Crazyflie -- immrax System Definition
===========================================================
This module answers the first two checks requested for the adaptive_spoofing
pipeline:

1. Does `examples/quadrotor_fault_diagnosis/quadrotor_separating_input.py`'s
   `QuadrotorSystem` already match the dynamics QPS (Quadrotarium Python
   Simulator, `~/adaptive_spoofing/libs/quadrotarium_python_simulator`) uses
   for its Crazyflie plant?  -- NO, see "Dynamics comparison" below.
2. Are that system's physical parameters (mass, inertia) correct for a
   Crazyflie?  -- NO, see "Parameter comparison" below.  `QuadrotorSystem`
   uses generic literature quadrotor defaults (m=0.468 kg, I~1e-3 kg*m^2);
   the real Crazyflie is ~13x lighter with inertia ~2 orders of magnitude
   smaller.

`CrazyflieSystem` below is a corrected replacement, scoped to exactly what
QPS's own rigid-body plant computes -- see
`~/adaptive_spoofing/libs/quadrotarium_python_simulator/qps/utilities/quadcopter_model.py`,
`QuadcopterObject.forward_model()`.

Dynamics comparison
--------------------
QPS's `forward_model()` and this repo's existing `QuadrotorSystem.f()` share
the SAME state layout (position, ZYX Euler angles, a 3-vector "velocity", body
rates) and the SAME rotational dynamics (Euler's equations) and Euler-angle
kinematics (the `Twb` matrix). They differ in exactly one place: what frame
the velocity state (indices 6:9) is in.

- `QuadrotorSystem` (existing): state[6:9] = BODY-frame velocity (u,v,w).
  Position kinematics rotate it into the world frame (`xdot = R @ [u,v,w]`),
  and the translational-acceleration equations carry Coriolis coupling terms
  (`-q*w+r*v` etc.) because differentiating a body-frame vector picks up
  `omega x v`. This is the standard "body-frame Newton-Euler" textbook form
  (e.g. Beard & McLain).

- QPS (`forward_model`): state[6:9] is labelled "Linear Velocity of body
  (u,v,w)" in a comment, but `state_d[0:3] = vel` uses it UNROTATED as the
  position rate, and `state_d[6:9] = (-m*g*z_w + u[0]*z_b) / m` computes
  acceleration directly in the WORLD frame (no Coriolis terms). So despite
  the comment, QPS's velocity state is actually WORLD-frame, not body-frame.
  This is a common simplified quadrotor formulation, but it is a genuinely
  different ODE from `QuadrotorSystem`'s -- not just a relabeling. A
  spoofing/discrimination pipeline built on the wrong frame convention would
  silently diverge from QPS every step (no error, no crash -- the equations
  are simply modeling different physics), which is why this module exists.

  (Sanity check that this reading of `forward_model` is right: expand QPS's
  world-frame `xdot=vel` composed with a body-frame integrator and you get
  DIFFERENT accelerations under nonzero roll/pitch than `QuadrotorSystem`
  would give the SAME state+input -- verified numerically in
  `tests/test_crazyflie_12d.py::test_matches_qps_forward_model`, which runs a
  handful of random states through both this module's `f()` and a
  transcription of QPS's actual `forward_model()` and requires bit-for-bit
  agreement, so any future edit to either side that breaks the correspondence
  fails loudly.)

Everything else -- rotational dynamics, Euler kinematics, and therefore the
`theta=+-90deg` gimbal-lock singularity in phidot/psidot -- is identical to
`QuadrotorSystem` and inherits the same compile-cost caveat (see this
project's memory: tan(theta)/1-cos(theta) terms are expensive to
reverse-mode-differentiate through when unrolled over many Euler steps).

Parameter comparison (QPS values, from `quadcopter_model.py` /
`docs/rq3_transferability_design.md` Sec 8.1)
------------------------------------------------------------------------------
                        QuadrotorSystem (existing, WRONG for Crazyflie)   QPS Crazyflie (source of truth)
  mass m                0.468 kg                                         0.03589 kg  (35.89 g)
  gravity g              9.81                                             9.81
  Ixx                    4.856e-3 kg*m^2                                  2.3951e-5 kg*m^2
  Iyy                    4.856e-3 kg*m^2                                  2.3951e-5 kg*m^2
  Izz                    8.801e-3 kg*m^2                                  3.2346e-5 kg*m^2
  hover thrust (m*g)     ~4.59 N                                          ~0.352 N
  dt (QPS sim step)      n/a                                              0.02 s
  arena bounds           n/a                                              x,y in [-1.5,1.5], z in [0.1,1.8]

Scope of this module
---------------------
Only the rigid-body PLANT (state, dynamics, embedding) -- matching what the
task asked for ("make another system called 12dCrazyflie"). The outer-loop
control laws (the four surrogate controllers), the spoofing-signal design,
and the discrimination pipeline that consumes this system are design
questions with several open decisions -- see PLAN.md in this directory.
"""

import sys
from pathlib import Path
from typing import Dict, Tuple

# File is at examples/adaptive_spoofing/<name>.py -> parents[1] = examples/
_EXAMPLES_DIR = Path(__file__).resolve().parents[1]
if str(_EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES_DIR))

import jax
import jax.numpy as jnp
import immrax as irx

# ══════════════════════════════════════════════════════════════════════════════
# Physical parameters -- real Crazyflie values from QPS (NOT the generic
# literature defaults in quadrotor_fault_diagnosis/quadrotor_separating_input.py)
# ══════════════════════════════════════════════════════════════════════════════
_M = 35.89e-3       # mass, kg  (quadcopter_model.py:39)
_G = 9.81           # gravity, m/s^2  (quadcopter_model.py:40)
_IXX = 2.3951e-5    # roll-axis moment of inertia, kg*m^2  (quadcopter_model.py:35)
_IYY = 2.3951e-5    # pitch-axis moment of inertia, kg*m^2
_IZZ = 3.2346e-5    # yaw-axis moment of inertia, kg*m^2

_HOVER_THRUST = _M * _G   # ~0.3521 N

# QPS's own simulation step and arena bounds (quadrotarium.py / RQ3 design doc
# Sec 8.1) -- useful defaults for demos/tests built on this system.
QPS_DT = 0.02
ARENA_X_HALF = 1.5
ARENA_Y_HALF = 1.2
ARENA_Z_RANGE = (0.1, 1.6)


class CrazyflieSystem(irx.System):
    """12-state rigid-body Crazyflie, matching QPS's `forward_model()` exactly.

    State   x = [x, y, z, phi, theta, psi, vx, vy, vz, p, q, r]   (12 states)
              position (world) / Euler angles (roll,pitch,yaw) /
              WORLD-frame linear velocity / body-frame angular velocity.
              (Contrast with quadrotor_fault_diagnosis's QuadrotorSystem,
              whose indices 6:9 are BODY-frame velocity -- see module
              docstring "Dynamics comparison".)
    Control u = [U1, U2, U3, U4]   thrust (N) + roll/pitch/yaw moment (N*m),
              already-mixed virtual actuator commands (QPS's `go_to()` ->
              `obtain_desired_inputs()` output), no rotor-speed mixing matrix.
    Params  p = alpha = [a1,a2,a3,a4]   per-channel actuator authority,
              applied as u_eff = alpha * u (elementwise), ai=1 -> nominal.
              (Kept for API parity with QuadrotorSystem/other fault-diagnosis
              modules in this repo; QPS itself has no actuator-fault model,
              so alpha=[1,1,1,1] is QPS-faithful. See PLAN.md for how the
              controller-discrimination pipeline's own "which controller"
              parameter is a SEPARATE concern from this actuator-fault knob.)
    """

    def __init__(self, m: float = _M, g: float = _G,
                 Ixx: float = _IXX, Iyy: float = _IYY, Izz: float = _IZZ):
        self.evolution = 'continuous'
        self.xlen = 12
        self.m = m
        self.g = g
        self.Ixx = Ixx
        self.Iyy = Iyy
        self.Izz = Izz

    def f(self, t, x, u, p):
        alpha = p
        U1, U2, U3, U4 = alpha * u

        phi, theta, psi = x[3], x[4], x[5]
        vx, vy, vz = x[6], x[7], x[8]
        pr, qr, rr = x[9], x[10], x[11]

        sphi, cphi = jnp.sin(phi), jnp.cos(phi)
        stheta, ctheta, ttheta = jnp.sin(theta), jnp.cos(theta), jnp.tan(theta)
        spsi, cpsi = jnp.sin(psi), jnp.cos(psi)

        # position kinematics: WORLD-frame velocity integrates directly
        # (QPS forward_model: state_d[0:3] = vel, no rotation) -- this is the
        # one line that differs structurally from QuadrotorSystem.
        xdot = vx
        ydot = vy
        zdot = vz

        # Euler-angle kinematics (Twb @ [p,q,r]) -- identical to QuadrotorSystem
        phidot = pr + sphi * ttheta * qr + cphi * ttheta * rr
        thetadot = cphi * qr - sphi * rr
        psidot = (sphi / ctheta) * qr + (cphi / ctheta) * rr

        # world-frame translational acceleration: thrust along body z-axis
        # (R's third column, same rotation convention as QuadrotorSystem)
        # minus gravity -- no Coriolis terms since the velocity state is
        # already world-frame (QPS forward_model: state_d[6:9] =
        # (-m*g*z_w + u[0]*z_b)/m).
        z_b_x = cphi * stheta * cpsi + sphi * spsi
        z_b_y = cphi * stheta * spsi - sphi * cpsi
        z_b_z = cphi * ctheta

        vxdot = (U1 / self.m) * z_b_x
        vydot = (U1 / self.m) * z_b_y
        vzdot = (U1 / self.m) * z_b_z - self.g

        # rotational dynamics (Euler's equations) -- identical to QuadrotorSystem
        prdot = ((self.Iyy - self.Izz) / self.Ixx) * qr * rr + U2 / self.Ixx
        qrdot = ((self.Izz - self.Ixx) / self.Iyy) * pr * rr + U3 / self.Iyy
        rrdot = ((self.Ixx - self.Iyy) / self.Izz) * pr * qr + U4 / self.Izz

        return jnp.array([
            xdot, ydot, zdot, phidot, thetadot, psidot,
            vxdot, vydot, vzdot, prdot, qrdot, rrdot,
        ])


# Module-level cache: one CrazyflieSystem + one irx.natemb(...) per distinct
# (m, g, Ixx, Iyy, Izz) tuple -- mirrors quadrotor_separating_input.py's
# get_system_and_embedding caching pattern.
_EMB_CACHE: Dict[Tuple[float, float, float, float, float], Tuple[CrazyflieSystem, object]] = {}


def get_system_and_embedding(m: float = _M, g: float = _G, Ixx: float = _IXX,
                             Iyy: float = _IYY, Izz: float = _IZZ) -> Tuple[CrazyflieSystem, object]:
    """Return (system, natural_embedding) for the given physical parameters, cached."""
    key = (m, g, Ixx, Iyy, Izz)
    if key not in _EMB_CACHE:
        sys_ = CrazyflieSystem(m, g, Ixx, Iyy, Izz)
        emb = irx.natemb(sys_)
        _EMB_CACHE[key] = (sys_, emb)
    return _EMB_CACHE[key]


def euler_step(emb_sys, x_ivl: irx.Interval, u: jnp.ndarray,
               p_ivl: irx.Interval, dt: float) -> irx.Interval:
    """One forward-Euler interval step via the natural embedding. `t` must be
    a JAX array, not a Python scalar -- see this project's immrax API notes
    (memory: eqx.filter_make_jaxpr folds Python scalars as constants)."""
    _t = jnp.zeros(())
    x_ut = irx.i2ut(x_ivl)
    dx_ut = emb_sys.f(_t, x_ut, u, p_ivl)
    return irx.ut2i(dx_ut * dt + x_ut)


if __name__ == "__main__":
    # Hover-equilibrium sanity check: at x=0 (level attitude, zero rates),
    # u=[m*g,0,0,0] should give f(x,u)=0 exactly -- same regression check
    # QuadrotorSystem's PLAN.md uses.
    sys_, emb = get_system_and_embedding()
    x0 = jnp.zeros(12)
    u_hover = jnp.array([_HOVER_THRUST, 0.0, 0.0, 0.0])
    p_nominal = jnp.ones(4)
    xdot = sys_.f(jnp.zeros(()), x0, u_hover, p_nominal)
    print("Hover residual (should be ~0):", xdot)
    assert jnp.allclose(xdot, 0.0, atol=1e-5), "Hover point is not an equilibrium!"
    print("OK: x=0, u=[m*g,0,0,0] is an exact fixed point.")
