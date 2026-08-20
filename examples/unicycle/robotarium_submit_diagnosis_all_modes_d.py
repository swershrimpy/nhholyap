"""
Robotarium SUBMISSION script: single-robot, single-submission demonstration
of ALL THREE fault hypotheses (Nominal / Actuator Fault / Sensor Fault) for
ONE unicycle config (multistep_unrefined/config_idx=9), run back-to-back in one Robotarium session.
Between modes the robot is physically DRIVEN back to the original start
pose (a real robot cannot be teleported/reset in software -- see "Reset
between modes" below), then the next mode's u_opt is played and checked.

Supersedes robotarium_submit_diagnosis_demo.py's TRUE_MODE-switch-and-
resubmit-3-times workflow: that script needed 3 separate submissions (one
per TRUE_MODE) for the same config. This script needs exactly ONE
submission to demonstrate all 3 modes for this config. One config per
submission still holds -- to demo a DIFFERENT config, regenerate the
embedded constants below (see "Regenerating the embedded constants") and
submit this file again.

Files needed for this submission: THIS FILE ONLY. All project-specific
data (u_opt, x0, alpha range, sensor transform, the 3 reachable tubes) is
precomputed offline and embedded below as plain numpy arrays -- no jax/
immrax/car_separating_input.py import, since the Robotarium submission
environment cannot be assumed to have those installed, only `rps` + numpy
+ matplotlib (see robotarium_submit_diagnosis_demo.py's module docstring
for the same reasoning; unchanged here).

Structural precedent: this machine's existing Robotarium-submittable
examples (~/output_feedback/robotarium_python_simulator/rps/examples/
fault_diagnosis/uni_afd.py) and the go-to-pose precedent (rps/examples/
plotting/uni_go_to_pose_hybrid_with_plotting.py, for the reset routine).

Reset between modes
---------------------
A real robot cannot be reset in software -- "resetting to the original
initial state" means physically DRIVING it back to X0_CENTER's full pose
(position AND orientation; the tubes below assume that exact starting
phi). Done with rps.utilities.controllers.create_hybrid_unicycle_pose_
controller (drives straight to the target position, then rotates to the
target heading) plus rps.utilities.misc.at_pose to detect arrival, capped
at RESET_MAX_ITERS iterations as a safety net (prints a warning and moves
on rather than looping forever if convergence is slower than expected).
No unicycle_barrier_certificate wraps the reset drive -- see the comment
above the Robotarium constructor call for why one was tried and then
removed (it actively caused hardware-limit violations for N=1, not a
harmless no-op). The u_opt PLAYBACK phases are likewise unwrapped, for
the more basic reason that wrapping them would alter the exact commanded
values the embedded reachable tubes were computed for.

Three bugs were found and fixed while building this reset routine, each
caught by dry-running this exact script against the local
rps.robotarium.Robotarium simulator, not by inspection:
1. X0_CENTER.reshape(3,1) is a VIEW, and Robotarium's constructor does
   self.poses = self.initial_conditions with no copy, then mutates
   self.poses in place every step() -- an uncopied view silently
   corrupted X0_CENTER itself as the robot moved, so every reset's
   target_pose was actually just the robot's current (already-drifted)
   pose. Fixed with .copy() at both use sites.
2. create_hybrid_unicycle_pose_controller's "stop translating" check is
   `norm_ > (position_error - position_epsilon)`; its default
   position_epsilon=0.03 combined with a tighter position_error=0.02 (not
   also overriding position_epsilon) made that threshold negative, so it
   was always true and the controller got stuck translating forever,
   never entering its rotate-to-heading phase. Fixed by keeping
   RESET_POSITION_EPSILON strictly less than RESET_POSITION_ERROR.
3. The controller's rotate-in-place phase computes
   dxu[1,i]=angular_velocity_gain*wrapped with NO clamp to its own
   angular_velocity_limit parameter (that limit only applies inside the
   translate phase's si_to_uni_dyn conversion) -- with the library's
   default gain=2 and a worst-case heading error of pi, commanded omega
   could reach ~6.28 rad/s against real hardware's ~3.64 rad/s limit.
   Fixed by lowering RESET_ANGULAR_VELOCITY_GAIN so gain*pi stays under
   the limit with margin.

Fault emulation on real hardware (see robotarium_submit_diagnosis_demo.py
for the full derivation -- unchanged here): CarNomActSystem's fault model
is phi_dot = ALPHA*omega -- alpha scales STEERING (omega), not throttle
(v). So "Actuator Fault" sends v_opt unchanged and ALPHA_TRUE*omega_opt;
"Sensor Fault" sends u_opt unchanged and instead transforms the RECORDED
pose (obs_scale*[px,py] + obs_offset) before checking it against the
tubes -- the robot's true motion is identical to Nominal.

Config selection: method=multistep_unrefined, config_idx=9 from
success_rate_data_early_stop_multistep_unrefined.npz -- one of 53 (method, config_idx)
pairs found (across all 6 success_rate_data*.npz sweep files) whose u_opt
satisfies real Robotarium hardware velocity limits (max|v|=0.0953 m/s,
max|omega|=0.8452 rad/s, vs. limits 0.2 m/s / 3.6364 rad/s).
See robotarium_submit_diagnosis_all_modes.py (config 32/multistep_refined)
for the original of this family and the full derivation of every fix
applied below -- unchanged here, only the config differs.

Regenerating the embedded constants (for a different CONFIG_IDX)
---------------------------------------------------------------------
  cd examples/unicycle
  JAX_PLATFORMS=cpu /home/user/immrax-venv/bin/python regenerate_robotarium_constants.py <METHOD> <CONFIG_IDX>
Paste the printed U_OPT / X0_CENTER / ALPHA_LO,HI / SENSOR_OFFSET,SCALE /
TUBE_LO / TUBE_HI block over the corresponding constants below.
"""
import rps.robotarium as robotarium
from rps.utilities.transformations import *
from rps.utilities.barrier_certificates import *
from rps.utilities.misc import *
from rps.utilities.controllers import *

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ══════════════════════════════════════════════════════════════════════════
# Vendored copies of at_pose / create_hybrid_unicycle_pose_controller (plus
# its create_si_to_uni_dynamics dependency), defined locally to SHADOW
# whatever the `from rps.utilities.* import *` lines above happened to pull
# in. The Robotarium hardware submission environment's installed rps build
# does not define create_hybrid_unicycle_pose_controller at all (NameError
# observed on an actual submission run, even though the same wildcard
# import resolved everything else this script needs -- consistent with the
# earlier `.axes` AttributeError also seen on that build: its installed rps
# is a different/older version than the local rps.robotarium simulator this
# script was dry-run against). Don't depend on the installed library having
# these two functions; these are pure-numpy copies matching
# rps.utilities.transformations.create_si_to_uni_dynamics,
# rps.utilities.controllers.create_hybrid_unicycle_pose_controller, and
# rps.utilities.misc.at_pose's own implementations (argument-checking
# asserts dropped since call sites here are internally controlled).
# ══════════════════════════════════════════════════════════════════════════
def create_si_to_uni_dynamics(linear_velocity_gain=1, angular_velocity_limit=np.pi):
    def si_to_uni_dyn(dxi, poses):
        a = np.cos(poses[2, :])
        b = np.sin(poses[2, :])
        dxu = np.zeros((2, dxi.shape[1]))
        dxu[0, :] = linear_velocity_gain * (a * dxi[0, :] + b * dxi[1, :])
        dxu[1, :] = angular_velocity_limit * np.arctan2(-b * dxi[0, :] + a * dxi[1, :], dxu[0, :]) / (np.pi / 2)
        return dxu
    return si_to_uni_dyn


def create_hybrid_unicycle_pose_controller(linear_velocity_gain=1, angular_velocity_gain=2,
                                            velocity_magnitude_limit=0.15, angular_velocity_limit=np.pi,
                                            position_error=0.05, position_epsilon=0.03, rotation_error=0.05):
    si_to_uni_dyn = create_si_to_uni_dynamics(linear_velocity_gain=linear_velocity_gain,
                                               angular_velocity_limit=angular_velocity_limit)

    def pose_uni_hybrid_controller(states, poses, approach_state=np.empty([0, 0])):
        N = states.shape[1]
        dxu = np.zeros((2, N))
        if approach_state.shape[1] != N:
            approach_state = np.ones((1, N))[0]
        for i in range(N):
            wrapped = poses[2, i] - states[2, i]
            wrapped = np.arctan2(np.sin(wrapped), np.cos(wrapped))
            dxi = poses[:2, [i]] - states[:2, [i]]
            norm_ = np.linalg.norm(dxi)
            if norm_ > (position_error - position_epsilon) and approach_state[i]:
                if norm_ > velocity_magnitude_limit:
                    dxi = velocity_magnitude_limit * dxi / norm_
                dxu[:, [i]] = si_to_uni_dyn(dxi, states[:, [i]])
            elif np.absolute(wrapped) > rotation_error:
                approach_state[i] = 0
                if norm_ > position_error:
                    approach_state = 1
                dxu[0, i] = 0
                dxu[1, i] = angular_velocity_gain * wrapped
            else:
                dxu[:, [i]] = np.zeros((2, 1))
        return dxu

    return pose_uni_hybrid_controller


def at_pose(states, poses, position_error=0.05, rotation_error=0.2):
    res = states[2, :] - poses[2, :]
    res = np.abs(np.arctan2(np.sin(res), np.cos(res)))
    pes = np.linalg.norm(states[:2, :] - poses[:2, :], 2, 0)
    done = np.nonzero((res <= rotation_error) & (pes <= position_error))
    return done


# ══════════════════════════════════════════════════════════════════════════
# Config constants -- see module docstring "Config selection" /
# "Regenerating the embedded constants". method=multistep_unrefined, config_idx=9.
# ══════════════════════════════════════════════════════════════════════════
U_OPT = np.array([
    [0.07400655001401901, 0.8452270030975342],
    [0.07439909875392914, 0.8450517654418945],
    [0.07782936841249466, 0.8446298241615295],
    [0.08334370702505112, 0.844067394733429],
    [0.0892975702881813, 0.8435457944869995],
    [0.09377918392419815, 0.8432255983352661],
    [0.09530927985906601, 0.8432639837265015],
    [0.09272360801696777, 0.8437382578849792],
    [0.07434543967247009, 0.8437382578849792],
    [0.07434543967247009, 0.8437382578849792],
])   # (10, 2) = [v (m/s), omega (rad/s)] per 0.5s segment
X0_CENTER = np.array([0.10000000149011612, 0.10000000149011612, 0.0])   # [px, py, phi]
X0_WIDTH = 0.05999999865889549   # half-width car_separating_input.py's icentpert()
                                  # assumed on px/py/phi alike when computing the
                                  # tubes below -- see RESET_POSITION_ERROR/
                                  # RESET_ROTATION_ERROR further down for why this
                                  # value (not a fixed constant) has to drive the
                                  # reset tolerance.
ALPHA_LO, ALPHA_HI = 0.3499999940395355, 0.6499999761581421
ALPHA_TRUE = 0.5 * (ALPHA_LO + ALPHA_HI)     # concrete point value used to
                                              # emulate "Actuator Fault"
SENSOR_OFFSET = np.array([0.20000000298023224, 0.20000000298023224])
SENSOR_SCALE = 0.949999988079071

# Per-scenario predicted OBSERVED-output ([px,py]) reachable box at each of
# the 10 segment boundaries, at 25 Euler substeps/segment (dt=0.02s) -- see
# robotarium_submit_diagnosis_demo.py's module docstring for why fine (not
# the optimizer's own coarse) resolution is required here.
TUBE_LO = {
    "Nominal": np.array([
    [0.07546740770339966, 0.04522906616330147],
    [0.10405202209949493, 0.06501197814941406],
    [0.12130215764045715, 0.09725918620824814],
    [0.12298795580863953, 0.13801802694797516],
    [0.10650214552879333, 0.1791568547487259],
    [0.07302191108465195, 0.2114892452955246],
    [0.028527654707431793, 0.22754646837711334],
    [-0.017704801633954048, 0.22407948970794678],
    [-0.05200789496302605, 0.20647776126861572],
    [-0.07773757725954056, 0.17714063823223114],
    ]),
    "Actuator Fault": np.array([
    [0.07621043175458908, 0.04040662944316864],
    [0.10933052003383636, 0.04628739878535271],
    [0.1379489302635193, 0.058027394115924835],
    [0.15984874963760376, 0.07630568742752075],
    [0.1721935123205185, 0.10157358646392822],
    [0.17251767218112946, 0.13349856436252594],
    [0.15997833013534546, 0.17071504890918732],
    [0.1361667960882187, 0.20929934084415436],
    [0.10918781906366348, 0.23470251262187958],
    [0.07633780688047409, 0.25185084342956543],
    ]),
    "Sensor Fault": np.array([
    [0.27169403433799744, 0.2429676204919815],
    [0.2988494336605072, 0.2617613673210144],
    [0.31523704528808594, 0.29239621758461],
    [0.3168385624885559, 0.3311171233654022],
    [0.3011770248413086, 0.3701990246772766],
    [0.26937082409858704, 0.4009147882461548],
    [0.22710126638412476, 0.4161691665649414],
    [0.18318043649196625, 0.41287553310394287],
    [0.15059250593185425, 0.39615386724472046],
    [0.12614929676055908, 0.36828362941741943],
    ]),
}
TUBE_HI = {
    "Nominal": np.array([
    [0.19635944068431854, 0.1695435792207718],
    [0.22753651440143585, 0.19291645288467407],
    [0.2487998604774475, 0.22747723758220673],
    [0.2554212212562561, 0.2688855230808258],
    [0.24397879838943481, 0.31170207262039185],
    [0.21460998058319092, 0.34780988097190857],
    [0.1723579317331314, 0.3690781891345978],
    [0.12672020494937897, 0.37112957239151],
    [0.09429103136062622, 0.35753652453422546],
    [0.07191243022680283, 0.3310892879962921],
    ]),
    "Actuator Fault": np.array([
    [0.19697776436805725, 0.16703368723392487],
    [0.2336752712726593, 0.18371309340000153],
    [0.27073967456817627, 0.20990216732025146],
    [0.3081468641757965, 0.2452022284269333],
    [0.3449087142944336, 0.2879653573036194],
    [0.379193514585495, 0.3347700536251068],
    [0.4088883399963379, 0.3824247121810913],
    [0.43213826417922974, 0.4287866950035095],
    [0.4458514451980591, 0.4659591317176819],
    [0.4543376564979553, 0.5031315684318542],
    ]),
    "Sensor Fault": np.array([
    [0.386541485786438, 0.36106640100479126],
    [0.4161596894264221, 0.38327062129974365],
    [0.43635988235473633, 0.4161033630371094],
    [0.4426501393318176, 0.4554412364959717],
    [0.4317798614501953, 0.4961169958114624],
    [0.40387946367263794, 0.5304194092750549],
    [0.3637400269508362, 0.5506242513656616],
    [0.3203842043876648, 0.5525730848312378],
    [0.2895764708518982, 0.5396596789360046],
    [0.2683168053627014, 0.5145348310470581],
    ]),
}
TUBE_COLOR = {"Nominal": "green", "Actuator Fault": "blue", "Sensor Fault": "orange"}
MODE_MARKER = {"Nominal": "o", "Actuator Fault": "s", "Sensor Fault": "^"}

DT = 0.5              # segment length (s) u_opt was optimized at
N_SUB = 15             # real Robotarium steps per segment: round(DT / time_step=0.033s)
RESET_MAX_ITERS = 900  # safety cap on the go-back-to-start drive (~30s at time_step=0.033s)
# Must be tighter than the config's OWN assumed x0 uncertainty (X0_WIDTH,
# applied uniformly to px/py/phi -- car_separating_input.py's
# icentpert(x0_center, x0_width)), not just "close enough" by eye, and not a
# FIXED constant picked once and reused across configs either: a flat 0.03
# tolerance is fine for configs with a wide X0_WIDTH but is actually LOOSER
# than X0_WIDTH itself for a tight one (config_idx=57 in this family has
# X0_WIDTH=0.02) -- i.e. the reset routine would consider itself "done" at a
# residual pose error the tube computation never allowed for at all, which
# repeat dry runs against the local simulator showed intermittently missing
# Nominal's own tube (the residual error only sometimes lands outside the
# tube by chance, since Robotarium's un-seeded initialize() drive-to-start
# takes a different path -- and so converges to a different residual offset
# within its own tolerance -- on every run). Deriving the reset tolerance as
# a fraction of X0_WIDTH instead gives every config real headroom (60%) below
# what its own tube assumed, while never loosening past the original 0.03 for
# configs that didn't need it.
#
# RESET_POSITION_EPSILON MUST be strictly less than RESET_POSITION_ERROR:
# create_hybrid_unicycle_pose_controller's own "close enough to stop
# translating" check is `norm_ > (position_error - position_epsilon)`; an
# unthinkingly tighter position_error without also tightening
# position_epsilon can make that threshold NEGATIVE, so the check is always
# true and the controller gets stuck translating forever, never entering its
# rotate-to-heading phase -- confirmed by tracing pos_err converging to ~0
# while rot_err sat at a nonzero plateau (1.3-2.3 rad) for 900 iterations
# straight, on an earlier fixed-constant version of this bug. Scaling
# EPSILON off of ERROR (same 1:3 ratio as the original 0.01/0.03) keeps this
# invariant automatically as X0_WIDTH varies across configs.
RESET_POSITION_ERROR = min(0.03, 0.4 * X0_WIDTH)
RESET_POSITION_EPSILON = RESET_POSITION_ERROR / 3
RESET_ROTATION_ERROR = min(0.03, 0.4 * X0_WIDTH)

# create_hybrid_unicycle_pose_controller's rotate-in-place phase computes
# dxu[1,i] = angular_velocity_gain*wrapped DIRECTLY, with no clamp to its
# own angular_velocity_limit parameter (that limit is only applied inside
# the TRANSLATE phase's si_to_uni_dyn conversion -- rps/utilities/
# controllers.py). With the library's default angular_velocity_gain=2 and
# a worst-case heading error of pi, the commanded omega can reach ~6.28
# rad/s, exceeding real hardware's max_angular_velocity~=3.64 rad/s --
# confirmed by a dry run against the local simulator reporting "actuator
# limits... thresholded to their maximum rotational velocity" (Robotarium
# safely clips this itself, but it is flagged as an error in
# call_at_scripts_end's report, which also suppresses the "Acceptance of
# your experiment is likely" message). Lowering the gain so gain*pi stays
# under MAX_OMEGA with margin removes the warning outright rather than
# relying on the simulator's own clipping as a safety net.
RESET_ANGULAR_VELOCITY_GAIN = 1.0

# SEPARATE from RESET_ANGULAR_VELOCITY_GAIN above -- easily confused, and
# confusing them was itself a bug caught by dry-running this script:
# angular_velocity_gain only bounds the ROTATE-in-place phase (fixed
# above); the TRANSLATE phase uses create_si_to_uni_dynamics's OWN
# angular_velocity_limit parameter instead, and that function's formula is
# `angular_velocity_limit * arctan2(...) / (pi/2)` (rps/utilities/
# transformations.py) -- since arctan2 ranges over [-pi,pi], dividing by
# pi/2 means the ACTUAL max output is 2*angular_velocity_limit, not
# angular_velocity_limit itself. Leaving this at its own default (pi) left
# the translate phase free to command omega up to ~2*pi~=6.28 rad/s (seen
# directly: it produced om=5.54 rad/s while driving back to start after
# Actuator Fault), a second, independent source of the same "actuator
# limits... thresholded" warning after RESET_ANGULAR_VELOCITY_GAIN alone
# had already fixed the rotate-phase source. Set so 2x this stays under
# MAX_OMEGA with margin.
RESET_ANGULAR_VELOCITY_LIMIT = 1.5

# Real Robotarium hardware limits (rps/robotarium_abc.py) -- sanity check
# before ever sending anything, not just at design time.
MAX_V = 0.2
MAX_OMEGA = 2 * (0.016 / 0.11) * (0.2 / 0.016)
assert np.all(np.abs(U_OPT[:, 0]) <= MAX_V + 1e-6), "U_OPT exceeds max_linear_velocity"
assert np.all(np.abs(U_OPT[:, 1]) <= MAX_OMEGA + 1e-6), "U_OPT exceeds max_angular_velocity"

print(f"x0_center={X0_CENTER}  alpha_true={ALPHA_TRUE:.3f}  "
     f"max|v|={np.abs(U_OPT[:,0]).max():.3f} (limit {MAX_V})  "
     f"max|omega|={np.abs(U_OPT[:,1]).max():.3f} (limit {MAX_OMEGA:.3f})")

# ══════════════════════════════════════════════════════════════════════════
# Robotarium setup
# ══════════════════════════════════════════════════════════════════════════
N = 1
initial_conditions = X0_CENTER.reshape(3, 1).copy()   # .copy() is load-bearing:
# reshape() returns a VIEW, and Robotarium's constructor does
# self.poses = self.initial_conditions with NO copy (robotarium_abc.py), then
# mutates self.poses IN PLACE every step() -- an uncopied view here would
# silently corrupt X0_CENTER itself as the robot moves. Caught by
# dry-running this script: without .copy(), reset_to_start() below found
# target_pose == the robot's CURRENT (already-drifted) pose on iteration 0
# every time, because X0_CENTER had already been overwritten by Robotarium's
# internal state -- the reset never actually drove anywhere, and Actuator
# Fault / Sensor Fault silently started from wherever the PREVIOUS mode's
# trajectory happened to end instead of from the true x0.
r = robotarium.Robotarium(number_of_robots=N, show_figure=True,
                          initial_conditions=initial_conditions, sim_in_real_time=True)
# The Robotarium hardware submission environment's rps build does not
# expose `.axes` on the Robotarium object (AttributeError observed on
# an actual submission run), even though show_figure=True still builds a
# matplotlib figure internally (confirmed by that run's own
# "FigureCanvasAgg is non-interactive" warning from its internal
# plt.show() call) -- so locate the Axes defensively instead of
# assuming an attribute name, falling all the way back to a standalone
# figure if the Robotarium object exposes neither .axes nor .figure.
ax = getattr(r, "axes", None)
if ax is None:
    _r_fig = getattr(r, "figure", None)
    if _r_fig is not None and getattr(_r_fig, "axes", None):
        ax = _r_fig.axes[0]
if ax is None:
    _, ax = plt.subplots()
# No unicycle_barrier_certificate: not needed for N=1 (no collision risk),
# and confirmed actively harmful here -- wrapping the reset drive's dxu
# through it introduced a small spurious linear-velocity component
# alongside an already near-maximum rotation-only omega command, and their
# COMBINED per-wheel demand (Robotarium's differential-drive wheel
# velocities couple v and omega together, not independently limited)
# exceeded the physical wheel-speed limit even though v and omega were
# each individually within their own separate limits -- one of several
# compounding sources of the "actuator limits... thresholded" events
# call_at_scripts_end() reported (11, before this was removed) while this
# routine was being built; RESET_ANGULAR_VELOCITY_GAIN/_LIMIT below fixed
# two more independent sources, and _clamp_to_wheel_limits (further below)
# is what finally brought the count to zero -- see that function's own
# docstring for why tuning gains alone could not reliably guarantee this.

legend_handles = []
for name in ("Nominal", "Actuator Fault", "Sensor Fault"):
    for k in range(TUBE_LO[name].shape[0]):
        w = TUBE_HI[name][k, 0] - TUBE_LO[name][k, 0]
        h = TUBE_HI[name][k, 1] - TUBE_LO[name][k, 1]
        rect = patches.Rectangle((TUBE_LO[name][k, 0], TUBE_LO[name][k, 1]), w, h,
                                 facecolor=TUBE_COLOR[name], alpha=0.15,
                                 edgecolor=TUBE_COLOR[name], linewidth=0.8)
        ax.add_patch(rect)
    legend_handles.append(patches.Patch(facecolor=TUBE_COLOR[name], alpha=0.3,
                                        edgecolor=TUBE_COLOR[name], label=name))

# ══════════════════════════════════════════════════════════════════════════
# Live per-mode trajectory lines, drawn directly on the Robotarium arena axes
# (`ax` -- literally the main robot-area plot, not a separate window) and
# updated after every real Robotarium step instead of being built once and
# plotted only after all 3 modes finish. `r.step()` already redraws `ax` on
# every call (Robotarium._draw_robots() -> canvas.draw_idle()/flush_events(),
# rps/robotarium_abc.py), so a Line2D already registered on `ax` picks up new
# `set_data()` calls automatically -- no extra plt.pause()/canvas call needed
# here, and this stays correct whether show_figure ends up interactive or
# (as on the real submission server) an Agg canvas that just no-ops draws.
#
# For "Sensor Fault", the line tracks the SENSOR-TRANSFORMED point (what the
# tube/diagnosis check actually compares against -- observed_by_mode below),
# not the robot's true position -- the fault lives in the sensor model, not
# the true motion (module docstring "Fault emulation on real hardware"), so
# plotting true state here would show a trajectory the diagnoser never
# actually sees and make the tube-containment picture misleading.
# ══════════════════════════════════════════════════════════════════════════
def _observed_point(mode, xy):
    if mode == "Sensor Fault":
        return SENSOR_SCALE * xy + SENSOR_OFFSET
    return xy


traj_lines = {}
for mode in ("Nominal", "Actuator Fault", "Sensor Fault"):
    label = "Observed (Sensor Fault)" if mode == "Sensor Fault" else f"Measured ({mode})"
    (line,) = ax.plot([], [], color="black", linewidth=1.5, marker=MODE_MARKER[mode],
                      markersize=6, zorder=10, label=label)
    traj_lines[mode] = line
ax.legend(handles=legend_handles + list(traj_lines.values()), loc="upper left", fontsize=7)


# ══════════════════════════════════════════════════════════════════════════
# Wheel-velocity clamp: replicates Robotarium's OWN internal _threshold /
# _uni_to_diff / _diff_to_uni math (rps/robotarium_abc.py) so a command can
# be made safe BEFORE sending it, rather than relying on Robotarium's own
# after-the-fact clipping (which is safe, but increments its internal
# error counter and suppresses the "Acceptance of your experiment is
# likely" message). Needed because MAX_V/MAX_OMEGA are each independently-
# derived assuming the OTHER is zero (pure translation or pure rotation);
# the actual physical constraint is on each WHEEL's combined (v,omega)
# demand, which the hybrid pose controller's translate phase can violate
# even with v and omega each individually within their own separate
# limits (observed directly: v=-0.14 m/s with omega=2.64 rad/s together,
# both comfortably under MAX_V/MAX_OMEGA alone, still exceeded the wheel
# limit) -- tuning gains to avoid this indirectly turned out to be
# unreliable across the different (v,omega) combinations this reset drive
# actually visits, so this clamps the physical quantity directly instead.
_WHEEL_RADIUS = 0.016
_BASE_LENGTH = 0.105
_MAX_WHEEL_VELOCITY = MAX_V / _WHEEL_RADIUS


def _clamp_to_wheel_limits(dxu):
    v, om = dxu[0, 0], dxu[1, 0]
    wl = (2 * v - _BASE_LENGTH * om) / (2 * _WHEEL_RADIUS)
    wr = (2 * v + _BASE_LENGTH * om) / (2 * _WHEEL_RADIUS)
    scale = max(abs(wl), abs(wr), _MAX_WHEEL_VELOCITY) / _MAX_WHEEL_VELOCITY
    wl, wr = wl / scale, wr / scale
    v_safe = _WHEEL_RADIUS / 2 * (wl + wr)
    om_safe = _WHEEL_RADIUS / _BASE_LENGTH * (wr - wl)
    return np.array([[v_safe], [om_safe]])


# ══════════════════════════════════════════════════════════════════════════
# Reset routine: physically drive back to X0_CENTER's full pose
# (position + orientation) -- see module docstring "Reset between modes".
# ══════════════════════════════════════════════════════════════════════════
def reset_to_start():
    target_pose = X0_CENTER.reshape(3, 1).copy()   # .copy() is load-bearing -- see
    # initial_conditions' comment above; at_pose()/the pose controller below
    # both take this as an argument they only read, but copying defensively
    # here too costs nothing and removes any dependence on X0_CENTER not
    # having been aliased elsewhere.
    pose_controller = create_hybrid_unicycle_pose_controller(
        position_error=RESET_POSITION_ERROR, position_epsilon=RESET_POSITION_EPSILON,
        rotation_error=RESET_ROTATION_ERROR, angular_velocity_gain=RESET_ANGULAR_VELOCITY_GAIN,
        angular_velocity_limit=RESET_ANGULAR_VELOCITY_LIMIT)
    for it in range(RESET_MAX_ITERS):
        x = r.get_poses()
        if np.size(at_pose(x, target_pose, position_error=RESET_POSITION_ERROR,
                           rotation_error=RESET_ROTATION_ERROR)) == N:
            break
        dxu = pose_controller(x, target_pose)
        dxu = _clamp_to_wheel_limits(dxu)
        r.set_velocities(np.arange(N), dxu)
        r.step()
    else:
        print(f"WARNING: reset did not converge within {RESET_MAX_ITERS} iterations "
             f"(position_error={RESET_POSITION_ERROR}, rotation_error={RESET_ROTATION_ERROR}); "
             f"proceeding to the next mode from wherever the robot currently is.")
        r.get_poses()   # required before the next step() call -- see robotarium.py's
                        # "Make sure to call get_poses before calling step() again"
                        # protocol; the for-loop's own last iteration already
                        # called step() without a following get_poses() when the
                        # loop is exhausted (as opposed to broken out of).
    # Fully stop before handing off to the next mode's playback loop.
    r.set_velocities(np.arange(N), np.zeros((2, N)))
    r.step()


# ══════════════════════════════════════════════════════════════════════════
# Run all three modes back-to-back, resetting to X0_CENTER between each --
# AND before the first one. See module docstring "Fault emulation on real
# hardware".
#
# Reset-before-Nominal is load-bearing, not defensive boilerplate:
# Robotarium.__init__()/initialize() (robotarium.py) only drives to
# `initial_conditions` within ITS OWN position_error=0.05m/rotation_error=
# 0.2rad tolerance -- much looser than this config's own x0_width that the
# tubes below were actually computed assuming. Dry-running an earlier
# version of this script (no reset before the first mode) against the local
# rps.robotarium.Robotarium simulator caught this directly on config 32
# (robotarium_submit_diagnosis_all_modes.py): Nominal (the only mode that
# skipped reset_to_start()) started far enough outside the assumed x0 box
# that its OWN tube failed to contain its own measured trajectory (MISSED),
# while Actuator Fault and Sensor Fault -- both already benefiting from
# reset_to_start()'s tighter RESET_POSITION_ERROR/RESET_ROTATION_ERROR --
# were both CORRECT. All three modes must start from the same verified
# tight-tolerance pose; this config was re-verified against the local
# simulator with the fix applied (see PLAN.md/commit history).
# ══════════════════════════════════════════════════════════════════════════
measured_by_mode = {}
observed_by_mode = {}

x = r.get_poses()
r.step()
print("Driving to verified start pose before Nominal...")
reset_to_start()

for mode in ("Nominal", "Actuator Fault", "Sensor Fault"):
    command_scale = ALPHA_TRUE if mode == "Actuator Fault" else 1.0
    line = traj_lines[mode]
    obs_xs, obs_ys = [], []

    measured_xy = []
    for k in range(U_OPT.shape[0]):
        v_cmd = float(U_OPT[k, 0])
        omega_cmd = float(U_OPT[k, 1]) * command_scale
        for _ in range(N_SUB):
            x = r.get_poses()
            r.set_velocities(np.arange(N), np.array([[v_cmd], [omega_cmd]]))
            r.step()
            # Live update: append this substep's OBSERVED point (module
            # docstring above -- true position for Nominal/Actuator Fault,
            # sensor-transformed for Sensor Fault) and let the next r.step()
            # pick it up on its next canvas redraw.
            obs_pt = _observed_point(mode, x[:2, 0])
            obs_xs.append(float(obs_pt[0]))
            obs_ys.append(float(obs_pt[1]))
            line.set_data(obs_xs, obs_ys)
        measured_xy.append(x[:2, 0].copy())
    measured_xy = np.array(measured_xy)   # (10, 2) -- TRUE robot position at each segment boundary
    measured_by_mode[mode] = measured_xy
    observed_by_mode[mode] = _observed_point(mode, measured_xy)

    print(f"Finished playback for TRUE_MODE={mode}; driving back to start pose...")
    reset_to_start()

# ══════════════════════════════════════════════════════════════════════════
# Online diagnosis check per mode (post-hoc, from the REAL recorded
# trajectory): a scenario is excluded the first segment its tube fails to
# contain the observed trajectory, and stays excluded -- same "Pairwise
# pruning" rule robotarium_diagnosis_mc.py checks in local simulation.
# ══════════════════════════════════════════════════════════════════════════
print("\n=== Diagnosis results ===")
for true_mode in ("Nominal", "Actuator Fault", "Sensor Fault"):
    observed_xy = observed_by_mode[true_mode]
    alive = {name: True for name in TUBE_COLOR}
    for k in range(observed_xy.shape[0]):
        for name in TUBE_COLOR:
            if alive[name]:
                inside = (np.all(observed_xy[k] >= TUBE_LO[name][k]) and
                         np.all(observed_xy[k] <= TUBE_HI[name][k]))
                if not inside:
                    alive[name] = False
    survivors = [n for n, a in alive.items() if a]
    print(f"True mode: {true_mode:16s}  surviving hypotheses: {survivors}", end="  ")
    if survivors == [true_mode]:
        print("-> CORRECT")
    elif true_mode in survivors:
        print("-> INCONCLUSIVE")
    else:
        print("-> MISSED")

# Call at end of script to print debug information and for the script to run
# on the Robotarium server properly. The Robotarium hardware submission
# environment's installed rps build was observed (on an actual submission
# run) to not define call_at_scripts_end at all -- AttributeError: 'Robotarium'
# object has no attribute 'call_at_scripts_end' -- consistent with the
# earlier .axes / create_hybrid_unicycle_pose_controller discrepancies
# documented above: that build is a different/older rps version than the
# local simulator this script was dry-run against. Guard defensively rather
# than assuming it exists.
if hasattr(r, "call_at_scripts_end"):
    r.call_at_scripts_end()
plt.show(block=True)
