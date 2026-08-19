"""
Robotarium SUBMISSION script: single-robot, single-submission demonstration
of ALL THREE fault hypotheses (Nominal / Actuator Fault / Sensor Fault) for
ONE car_fault_diagnosis config (single_step/config_idx=62), run back-to-back in one Robotarium session.
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

Config selection: method=single_step, config_idx=62 from
success_rate_data_early_stop_single_step.npz -- one of 53 (method, config_idx)
pairs found (across all 6 success_rate_data*.npz sweep files) whose u_opt
satisfies real Robotarium hardware velocity limits (max|v|=0.0837 m/s,
max|omega|=0.3092 rad/s, vs. limits 0.2 m/s / 3.6364 rad/s).
See robotarium_submit_diagnosis_all_modes.py (config 32/multistep_refined)
for the original of this family and the full derivation of every fix
applied below -- unchanged here, only the config differs.

Regenerating the embedded constants (for a different CONFIG_IDX)
---------------------------------------------------------------------
  cd examples/car_fault_diagnosis
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
import matplotlib.lines as mlines

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
# "Regenerating the embedded constants". method=single_step, config_idx=62.
# ══════════════════════════════════════════════════════════════════════════
U_OPT = np.array([
    [0.08370403945446014, -0.30915477871894836],
    [0.08370403945446014, -0.30915477871894836],
    [0.08370403945446014, -0.30915477871894836],
    [0.08370403945446014, -0.30915477871894836],
    [0.08370403945446014, -0.30915477871894836],
    [0.08370403945446014, -0.30915477871894836],
    [0.08370403945446014, -0.30915477871894836],
    [0.08370403945446014, -0.30915477871894836],
    [0.08370403945446014, -0.30915477871894836],
    [0.08370403945446014, -0.30915477871894836],
])   # (10, 2) = [v (m/s), omega (rad/s)] per 0.5s segment
X0_CENTER = np.array([-0.20000000298023224, 0.4000000059604645, -0.800000011920929])   # [px, py, phi]
ALPHA_LO, ALPHA_HI = 0.0, 0.5
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
    [-0.22481021285057068, 0.31662946939468384],
    [-0.20505861937999725, 0.27977854013442993],
    [-0.19121623039245605, 0.24032601714134216],
    [-0.18361318111419678, 0.1992126852273941],
    [-0.1824307143688202, 0.15741366147994995],
    [-0.18769709765911102, 0.11561167240142822],
    [-0.19928668439388275, 0.07448327541351318],
    [-0.21692314743995667, 0.03500326722860336],
    [-0.24018584191799164, -0.001886886078864336],
    [-0.2685200273990631, -0.03530745953321457],
    ]),
    "Actuator Fault": np.array([
    [-0.22357018291950226, 0.3175624907016754],
    [-0.19972383975982666, 0.28318122029304504],
    [-0.17860335111618042, 0.2470613718032837],
    [-0.1603347659111023, 0.20941853523254395],
    [-0.14502723515033722, 0.17047755420207977],
    [-0.13277211785316467, 0.130470871925354],
    [-0.12364258617162704, 0.0896373838186264],
    [-0.11769317835569382, 0.048220906406641006],
    [-0.11495941877365112, 0.006468686740845442],
    [-0.11545760184526443, -0.03538161516189575],
    ]),
    "Sensor Fault": np.array([
    [-0.013569697737693787, 0.5007979869842529],
    [0.005194321274757385, 0.4657896161079407],
    [0.018344581127166748, 0.42830973863601685],
    [0.025567486882209778, 0.38925206661224365],
    [0.026690825819969177, 0.3495429754257202],
    [0.021687760949134827, 0.30983108282089233],
    [0.010677650570869446, 0.27075910568237305],
    [-0.006076991558074951, 0.2332531064748764],
    [-0.028176546096801758, 0.19820746779441833],
    [-0.05509401857852936, 0.1664579212665558],
    ]),
}
TUBE_HI = {
    "Nominal": np.array([
    [-0.1216045394539833, 0.419310986995697],
    [-0.0982726514339447, 0.38461607694625854],
    [-0.08056069910526276, 0.3467426598072052],
    [-0.06889107823371887, 0.30659380555152893],
    [-0.06354209035634995, 0.2651269733905792],
    [-0.06464125961065292, 0.22364959120750427],
    [-0.07216241210699081, 0.18347756564617157],
    [-0.08592615276575089, 0.1455688625574112],
    [-0.10560426861047745, 0.11082755774259567],
    [-0.13072749972343445, 0.08008211106061935],
    ]),
    "Actuator Fault": np.array([
    [-0.11937737464904785, 0.4214717447757721],
    [-0.08875474333763123, 0.39294347167015076],
    [-0.0581321120262146, 0.3644151985645294],
    [-0.027509473264217377, 0.3358869254589081],
    [0.003113190643489361, 0.30735865235328674],
    [0.03373585268855095, 0.2788303792476654],
    [0.06435848772525787, 0.25030210614204407],
    [0.0949811190366745, 0.22177420556545258],
    [0.12560375034809113, 0.19324630498886108],
    [0.15622638165950775, 0.1647184044122696],
    ]),
    "Sensor Fault": np.array([
    [0.08447568863630295, 0.5983454585075378],
    [0.10664098709821701, 0.5653852820396423],
    [0.1234673410654068, 0.5294055342674255],
    [0.1345534771680832, 0.49126410484313965],
    [0.13963502645492554, 0.4518706202507019],
    [0.13859081268310547, 0.4124671220779419],
    [0.13144570589065552, 0.3743036985397339],
    [0.1183701604604721, 0.33829042315483093],
    [0.09967594593763351, 0.305286169052124],
    [0.07580888271331787, 0.2760780155658722],
    ]),
}
TUBE_COLOR = {"Nominal": "green", "Actuator Fault": "blue", "Sensor Fault": "orange"}
MODE_MARKER = {"Nominal": "o", "Actuator Fault": "s", "Sensor Fault": "^"}

DT = 0.5              # segment length (s) u_opt was optimized at
N_SUB = 15             # real Robotarium steps per segment: round(DT / time_step=0.033s)
RESET_MAX_ITERS = 900  # safety cap on the go-back-to-start drive (~30s at time_step=0.033s)
# Must be tighter than the config's OWN assumed x0 uncertainty
# (X0_WIDTH=0.04, applied uniformly to px/py/phi -- car_separating_input.py's
# icentpert(x0_center, x0_width)), not just "close enough" by eye: Actuator
# Fault / Sensor Fault modes missed their own diagnosis until this was
# fixed. RESET_POSITION_EPSILON MUST be strictly less than
# RESET_POSITION_ERROR: create_hybrid_unicycle_pose_controller's own
# "close enough to stop translating" check is `norm_ > (position_error -
# position_epsilon)`; its default position_epsilon=0.03 combined with an
# unthinkingly tighter position_error=0.02 (not also overriding
# position_epsilon) made that threshold NEGATIVE, so the check was always
# true and the controller got stuck translating forever, never entering
# its rotate-to-heading phase -- confirmed by tracing pos_err converging
# to ~0 while rot_err sat at a nonzero plateau (1.3-2.3 rad) for 900
# iterations straight. Caught by dry-running this exact script against
# the local simulator (same discipline that caught the v/omega actuator-
# fault bug and the initial_conditions aliasing bug elsewhere in this
# module -- see below).
RESET_POSITION_ERROR = 0.03
RESET_POSITION_EPSILON = 0.01
RESET_ROTATION_ERROR = 0.03

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
# Run all three modes back-to-back, resetting to X0_CENTER between each.
# See module docstring "Fault emulation on real hardware".
# ══════════════════════════════════════════════════════════════════════════
measured_by_mode = {}
observed_by_mode = {}

x = r.get_poses()
r.step()
for mode in ("Nominal", "Actuator Fault", "Sensor Fault"):
    command_scale = ALPHA_TRUE if mode == "Actuator Fault" else 1.0

    measured_xy = []
    for k in range(U_OPT.shape[0]):
        v_cmd = float(U_OPT[k, 0])
        omega_cmd = float(U_OPT[k, 1]) * command_scale
        for _ in range(N_SUB):
            x = r.get_poses()
            r.set_velocities(np.arange(N), np.array([[v_cmd], [omega_cmd]]))
            r.step()
        measured_xy.append(x[:2, 0].copy())
    measured_xy = np.array(measured_xy)   # (10, 2) -- TRUE robot position at each segment boundary
    measured_by_mode[mode] = measured_xy

    # Sensor-fault emulation: transform the RECORDED position, not the
    # robot's actual motion (module docstring).
    if mode == "Sensor Fault":
        observed_by_mode[mode] = SENSOR_SCALE * measured_xy + SENSOR_OFFSET
    else:
        observed_by_mode[mode] = measured_xy

    print(f"Finished playback for TRUE_MODE={mode}; driving back to start pose...")
    reset_to_start()

# ══════════════════════════════════════════════════════════════════════════
# Plot all 3 measured trajectories over the background tubes.
# ══════════════════════════════════════════════════════════════════════════
for mode in ("Nominal", "Actuator Fault", "Sensor Fault"):
    traj = measured_by_mode[mode]
    ax.plot(traj[:, 0], traj[:, 1], color="black", linewidth=1.5,
               marker=MODE_MARKER[mode], markersize=6, zorder=10)
    legend_handles.append(mlines.Line2D([0], [0], color="black", marker=MODE_MARKER[mode],
                                        linestyle="-", label=f"Measured ({mode})"))
ax.legend(handles=legend_handles, loc="upper left", fontsize=7)

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

# Call at end of script to print debug information and for the script to
# run on the Robotarium server properly.
r.call_at_scripts_end()
plt.show(block=True)
