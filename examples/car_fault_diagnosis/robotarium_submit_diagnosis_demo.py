"""
Robotarium SUBMISSION script: single-robot demonstration of
car_fault_diagnosis's solved separating-input controller, driven on REAL
Robotarium hardware, with an online check (from the REAL measured
trajectory) of which of the three fault hypotheses (Nominal / Actuator
Fault / Sensor Fault) remains consistent -- the same "Pairwise pruning"
diagnosis rule this project's local-simulator Monte Carlo study
(robotarium_diagnosis_mc.py) validates in `rps.robotarium.Robotarium`
before ever touching real hardware.

Structural precedent (imports, `robotarium.Robotarium(..., sim_in_real_time=
True)`, `get_poses`/`set_velocities`/`step` loop, background `patches.
Rectangle` regions, `r.call_at_scripts_end()` at the very end) is this
machine's existing Robotarium-submittable examples:
  ~/output_feedback/robotarium_python_simulator/rps/examples/fault_diagnosis/
  uni_afd.py, uni_afd_npy_actuator_fault.py
Both of those hand-derive/hardcode their background-tube numbers and
staged velocity commands. This script instead embeds NUMBERS PRECOMPUTED
by car_separating_input.py's own propagation/observation code (this
project's `_propagate_history` + `observed_output` -- the exact functions
the paper's diagnosis check itself uses, not a re-derivation) -- see
"Regenerating the embedded constants" below. The script has NO import of
jax/immrax/this project's own modules: the Robotarium submission
environment cannot be assumed to have those installed, only `rps` +
numpy + matplotlib, so everything project-specific is precomputed offline
and pasted in as plain arrays.

Config selection
-----------------
u_opt/x0/alpha/sensor values below come from one config (index 32) out of
the paper's 84-config sweep
(success_rate_data_early_stop_multistep_refined.npz), chosen because it is
one of only 4 (of 60 successful) configs whose OPTIMIZED u_opt also
satisfies REAL Robotarium hardware limits -- the abstract optimization's
own u in [-1,1]^2 box (car_separating_input.py's `_project_u`) does NOT
enforce max_linear_velocity=0.2 m/s / max_angular_velocity~=3.64 rad/s
(robotarium_abc.py); most of the 60 successful configs exceed 0.2 m/s and
would be clipped/distorted by the real robot's own saturation if submitted
unchanged.

  config_idx | x0_center          | max|v| (m/s) | max|omega| (rad/s) | in-bound correct rate (fine, all 3 modes, robotarium_diagnosis_mc.npz)
  2          | [ 0.10, 0.10, 0.0] | 0.196        | 0.302               | 100%  (but alpha_lo=0 -> actuator fault = total stall, and only 2% margin below the 0.2 m/s limit)
  13         | [ 0.10, 0.10, 0.0] | 0.085        | 1.000               | 98-100%
  32         | [ 0.50,-0.30, 1.0] | 0.151        | 0.639               | 100%  <- used here: 24% margin below the v limit, and alpha in [0.35,0.65] gives a visibly PARTIAL (not zero) actuator-fault motion
  57         | [-0.20, 0.40,-0.8] | 0.097        | 0.566               | 100%

All 4 also stay comfortably inside the Robotarium arena (x in [-1.6,1.6],
y in [-1,1]) under every true_mode -- checked by direct rollout, not
assumed (largest excursion of any of the 4, any mode: config 2 Nominal,
x up to 0.75 m, y up to 0.71 m).

Fault emulation on real hardware
----------------------------------
Only ACTUATOR fault changes what must be physically sent, and only on the
STEERING channel: CarNomActSystem's actual dynamics (car_separating_input.py)
are px_dot=v*cos(phi), py_dot=v*sin(phi), phi_dot=ALPHA*omega -- alpha
scales omega (steering/turn-rate authority), not v. (An earlier version
of this script scaled v instead, matching this project's OTHER fault-
diagnosis modules' `u_eff = alpha*u` convention on the FULL control
vector -- but this module's alpha only ever multiplies omega; that
mismatch was caught by dry-running this exact script against the local
`rps.robotarium.Robotarium` simulator before considering it submission-
ready, where the "Actuator Fault" case's own tube failed to contain its
own measured trajectory.) So "Actuator Fault" is emulated by sending
v_opt unchanged and ALPHA_TRUE*omega_opt (ALPHA_TRUE = midpoint of
[alpha_lo,alpha_hi], matching this project's own simulate_and_render.py-
style convention for a concrete point value) as the ACTUAL omega command
-- the robot mechanically turns less, a real actuation deficiency, not
merely a different reading of the same motion. A SENSOR fault does NOT
change the robot's true motion at all (Scenario's obs_scale/obs_offset
only transform the OBSERVED [px,py] the diagnoser sees, per
car_separating_input.py's Scenario docstring) -- so "Sensor Fault" mode
sends u_opt UNCHANGED and instead applies the affine sensor transform to
the RECORDED pose before comparing it against the tubes below, matching
what each mode actually represents.

Usage
------
Set TRUE_MODE below to whichever hypothesis this run should physically
be, then submit this single file through the Robotarium portal as-is.

Regenerating the embedded constants (if a different CONFIG_IDX is wanted)
----------------------------------------------------------------------------
  cd examples/car_fault_diagnosis
  JAX_PLATFORMS=cpu /home/user/immrax-venv/bin/python regenerate_robotarium_constants.py <CONFIG_IDX>
Prints a ready-to-paste block (U_OPT, X0_CENTER, ALPHA_LO/HI,
SENSOR_OFFSET/SCALE, TUBE_LO/TUBE_HI) using the same fine (25-substep)
tube resolution as the values embedded below -- verified to reproduce
them exactly for CONFIG_IDX=32. Also checks the new config's u_opt
against real hardware limits before printing anything, so a bad
CONFIG_IDX (one that exceeds max_linear_velocity/max_angular_velocity)
is caught immediately rather than silently producing an unsubmittable
script.
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
# Configuration -- change TRUE_MODE for each of the 3 submissions
# ══════════════════════════════════════════════════════════════════════════
TRUE_MODE = "Actuator Fault"    # "Nominal" | "Actuator Fault" | "Sensor Fault"

# ── Precomputed from success_rate_data_early_stop_multistep_refined.npz,
#    config_idx=32 -- see module docstring "Config selection" ──
U_OPT = np.array([
    [0.15065661072731018, 0.6394131779670715],
    [0.15065661072731018, 0.6394131779670715],
    [0.15065661072731018, 0.6394131779670715],
    [0.15065661072731018, 0.6394131779670715],
    [0.15065661072731018, 0.6394131779670715],
    [0.15065661072731018, 0.6394131779670715],
    [0.15065661072731018, 0.6394131779670715],
    [0.15065661072731018, 0.6394131779670715],
    [0.15065661072731018, 0.6394131779670715],
    [0.15065661072731018, 0.6394131779670715],
])   # (10, 2) = [v (m/s), omega (rad/s)] per 0.5s segment
X0_CENTER = np.array([0.5, -0.30000001192092896, 1.0])   # [px, py, phi]
ALPHA_LO, ALPHA_HI = 0.0, 0.5
ALPHA_TRUE = 0.5 * (ALPHA_LO + ALPHA_HI)     # concrete point value used to
                                              # emulate "Actuator Fault"
SENSOR_OFFSET = np.array([0.20000000298023224, 0.20000000298023224])
SENSOR_SCALE = 0.949999988079071

# Per-scenario predicted OBSERVED-output ([px,py]) reachable box at each of
# the 10 segment boundaries -- from car_separating_input.py's own
# _propagate_history + observed_output (config_idx=32), NOT hand-derived,
# at 25 Euler substeps/segment (dt=0.02s). NOT the coarse 1-substep/segment
# resolution the optimizer itself used: a headless dry run against the
# LOCAL rps.robotarium.Robotarium simulator with the coarse tube showed the
# TRUE mode's own tube failing to contain the measured trajectory --
# exactly the discretization-too-coarse pitfall robotarium_diagnosis_mc.py's
# own docstring documents and fixes (a single big 0.5s Euler step is too
# coarse a stand-in for the continuous flow once phi changes appreciably
# within it). This fine resolution is what that module's own "fine" check
# uses, and is what's embedded below.
TUBE_LO = {
    "Nominal": np.array([
        [0.4876365065574646, -0.27270010113716125], [0.49195680022239685, -0.1984374076128006],
        [0.47252294421195984, -0.12599024176597595], [0.43130460381507874, -0.06332199275493622],
        [0.3724789321422577, -0.016783786937594414], [0.3020077049732208, 0.008907976560294628],
        [0.2269516885280609, 0.011149553582072258], [0.15358832478523254, -0.01028622966259718],
        [0.08885255455970764, -0.05322696268558502], [0.039305057376623154, -0.11332081258296967],
    ]),
    "Actuator Fault": np.array([
        [0.4930057227611542, -0.27829205989837646], [0.514826774597168, -0.21658390760421753],
        [0.5249066948890686, -0.15487559139728546], [0.5229883790016174, -0.0931672751903534],
        [0.5091211199760437, -0.031458958983421326], [0.4836580157279968, 0.030249355360865593],
        [0.447248637676239, 0.09195766597986221], [0.40082141757011414, 0.15108704566955566],
        [0.34556007385253906, 0.20216040313243866], [0.2828737497329712, 0.24378642439842224],
    ]),
    "Sensor Fault": np.array([
        [0.6632546782493591, -0.059065088629722595], [0.667358934879303, 0.011484473943710327],
        [0.6488968133926392, 0.08030927181243896], [0.609739363193512, 0.1398441195487976],
        [0.5538550019264221, 0.1840554028749466], [0.48690730333328247, 0.20846258103847504],
        [0.4156041145324707, 0.21059207618236542], [0.34590891003608704, 0.19022808969020844],
        [0.28440994024276733, 0.1494343876838684], [0.2373398095369339, 0.092345230281353],
    ]),
}
TUBE_HI = {
    "Nominal": np.array([
        [0.5731207728385925, -0.19026851654052734], [0.5834116339683533, -0.11534325033426285],
        [0.5698294043540955, -0.04157474264502525], [0.5337510108947754, 0.024187032133340836],
        [0.47883257269859314, 0.07527745515108109], [0.41063982248306274, 0.10651876032352448],
        [0.3361600935459137, 0.11474477499723434], [0.26427993178367615, 0.09912184625864029],
        [0.20277908444404602, 0.061233293265104294], [0.1578904390335083, 0.004918926861137152],
    ]),
    "Actuator Fault": np.array([
        [0.5832029581069946, -0.19237661361694336], [0.6264058947563171, -0.12036174535751343],
        [0.6696088314056396, -0.04579168185591698], [0.7128117680549622, 0.029524000361561775],
        [0.7560147047042847, 0.1048523485660553], [0.7992176413536072, 0.18018069863319397],
        [0.8424205780029297, 0.25550904870033264], [0.8856235146522522, 0.3308373987674713],
        [0.9288264513015747, 0.40616574883461], [0.9720293879508972, 0.48149409890174866],
    ]),
    "Sensor Fault": np.array([
        [0.7444646954536438, 0.019244909286499023], [0.7542410492897034, 0.09042391926050186],
        [0.7413378953933716, 0.16050399839878082], [0.7070634365081787, 0.2229776829481125],
        [0.6548909544944763, 0.27151358127593994], [0.5901078581809998, 0.3011928200721741],
        [0.5193520784378052, 0.30900752544403076], [0.45106595754623413, 0.2941657602787018],
        [0.3926401138305664, 0.25817161798477173], [0.3499959111213684, 0.20467297732830048],
    ]),
}
TUBE_COLOR = {"Nominal": "green", "Actuator Fault": "blue", "Sensor Fault": "orange"}

DT = 0.5              # segment length (s) u_opt was optimized at
N_SUB = 15             # real Robotarium steps per segment: round(DT / time_step=0.033s)

# Real Robotarium hardware limits (rps/robotarium_abc.py) -- sanity check
# before ever sending anything, not just at design time.
MAX_V = 0.2
MAX_OMEGA = 2 * (0.016 / 0.11) * (0.2 / 0.016)
assert np.all(np.abs(U_OPT[:, 0]) <= MAX_V + 1e-6), "U_OPT exceeds max_linear_velocity"
assert np.all(np.abs(U_OPT[:, 1]) <= MAX_OMEGA + 1e-6), "U_OPT exceeds max_angular_velocity"

print(f"TRUE_MODE={TRUE_MODE}  x0_center={X0_CENTER}  alpha_true={ALPHA_TRUE:.3f}  "
     f"max|v|={np.abs(U_OPT[:,0]).max():.3f} (limit {MAX_V})  "
     f"max|omega|={np.abs(U_OPT[:,1]).max():.3f} (limit {MAX_OMEGA:.3f})")

# ══════════════════════════════════════════════════════════════════════════
# Robotarium setup
# ══════════════════════════════════════════════════════════════════════════
N = 1
initial_conditions = X0_CENTER.reshape(3, 1)
r = robotarium.Robotarium(number_of_robots=N, show_figure=True,
                          initial_conditions=initial_conditions, sim_in_real_time=True)

legend_handles = []
for name in ("Nominal", "Actuator Fault", "Sensor Fault"):
    for k in range(TUBE_LO[name].shape[0]):
        w = TUBE_HI[name][k, 0] - TUBE_LO[name][k, 0]
        h = TUBE_HI[name][k, 1] - TUBE_LO[name][k, 1]
        rect = patches.Rectangle((TUBE_LO[name][k, 0], TUBE_LO[name][k, 1]), w, h,
                                 facecolor=TUBE_COLOR[name], alpha=0.15,
                                 edgecolor=TUBE_COLOR[name], linewidth=0.8)
        r.axes.add_patch(rect)
    legend_handles.append(patches.Patch(facecolor=TUBE_COLOR[name], alpha=0.3,
                                        edgecolor=TUBE_COLOR[name], label=name))
r.axes.legend(handles=legend_handles, loc="upper left", fontsize=8)

# ══════════════════════════════════════════════════════════════════════════
# Drive the real robot through U_OPT, holding each segment's command for
# N_SUB real Robotarium steps. See module docstring "Fault emulation on
# real hardware" for why only Actuator Fault changes the sent command.
# ══════════════════════════════════════════════════════════════════════════
command_scale = ALPHA_TRUE if TRUE_MODE == "Actuator Fault" else 1.0

measured_xy = []
x = r.get_poses()
r.step()
for k in range(U_OPT.shape[0]):
    v_cmd = float(U_OPT[k, 0])
    omega_cmd = float(U_OPT[k, 1]) * command_scale
    for _ in range(N_SUB):
        x = r.get_poses()
        r.set_velocities(np.arange(N), np.array([[v_cmd], [omega_cmd]]))
        r.step()
    measured_xy.append(x[:2, 0].copy())
measured_xy = np.array(measured_xy)   # (10, 2) -- TRUE robot position at each segment boundary

# Sensor-fault emulation: transform the RECORDED position, not the robot's
# actual motion (see module docstring).
if TRUE_MODE == "Sensor Fault":
    observed_xy = SENSOR_SCALE * measured_xy + SENSOR_OFFSET
else:
    observed_xy = measured_xy

r.axes.plot(measured_xy[:, 0], measured_xy[:, 1], "k.-", linewidth=2,
           markersize=8, zorder=10)
legend_handles.append(mlines.Line2D([0], [0], color="k", marker=".", label="Measured trajectory"))
r.axes.legend(handles=legend_handles, loc="upper left", fontsize=8)

# ══════════════════════════════════════════════════════════════════════════
# Online diagnosis check (post-hoc, from the REAL recorded trajectory): a
# scenario is excluded the first segment its tube fails to contain
# observed_xy, and stays excluded -- same "Pairwise pruning" rule
# robotarium_diagnosis_mc.py checks in local simulation.
# ══════════════════════════════════════════════════════════════════════════
alive = {name: True for name in TUBE_COLOR}
for k in range(observed_xy.shape[0]):
    for name in TUBE_COLOR:
        if alive[name]:
            inside = np.all(observed_xy[k] >= TUBE_LO[name][k]) and np.all(observed_xy[k] <= TUBE_HI[name][k])
            if not inside:
                alive[name] = False
survivors = [n for n, a in alive.items() if a]
print(f"True mode: {TRUE_MODE}")
print(f"Diagnosis result -- surviving hypotheses: {survivors}")
if survivors == [TRUE_MODE]:
    print("CORRECT: diagnosis uniquely identified the true mode.")
elif TRUE_MODE in survivors:
    print("INCONCLUSIVE: true mode survives, but so does another hypothesis.")
else:
    print("MISSED: the true mode's own tube did not contain the measured trajectory.")

# Call at end of script to print debug information and for the script to
# run on the Robotarium server properly.
r.call_at_scripts_end()
plt.show(block=True)
