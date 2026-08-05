"""
State-space plot: output-feedback CBF controller on faulty car.

Shows reachable position intervals at every time step for all three fault
scenarios, overlaid with the obstacle disk.  The controller simultaneously
separates the scenarios (fault diagnosis) and keeps every interval clear of
the obstacle (collision avoidance).
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXAMPLES = HERE.parent
for _p in (str(HERE), str(EXAMPLES)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import jax
import jax.numpy as jnp
import immrax as irx
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
from matplotlib.collections import PatchCollection

from faulty_car_output_feedback_cbf import (
    create_cl_scenarios,
    cl_euler_step,
    optimize_output_feedback_cbf_gpu,
    theta_to_K_r,
    K_r_to_theta,
    separation_cbf_loss_refined,
    cbf_min_over_interval,
    _obs_interval,
)

# ── Problem setup ─────────────────────────────────────────────────────────────

x0_ivl = irx.icentpert(
    jnp.array([0.1,  0.1,  0.0]),
    jnp.array([0.08, 0.08, 0.08]),
)

# Obstacle disk
OBS_CX, OBS_CY, OBS_R = 0.8, 0.55, 0.20

obstacles = jnp.array([[OBS_CX, OBS_CY, OBS_R]])

DT         = 0.1
NUM_STEPS  = 8
CBF_WEIGHT = 5.0

cl_scenarios = create_cl_scenarios(actuator_alpha_lo=0.0, actuator_alpha_hi=0.5)

# ── Optimise ──────────────────────────────────────────────────────────────────

print("Optimising output-feedback + CBF controller …")
best_theta, best_loss, _, all_losses = optimize_output_feedback_cbf_gpu(
    cl_scenarios=cl_scenarios,
    x0_ivl=x0_ivl,
    dt=DT,
    num_steps=NUM_STEPS,
    obstacles=obstacles,
    cbf_weight=CBF_WEIGHT,
    num_restarts=50,
    learning_rate=0.02,
    num_iters=150,
    seed=0,
)
K, r = theta_to_K_r(best_theta)
print(f"  loss = {float(best_loss):.6f}")
print(f"  K =\n{np.array(K)}")
print(f"  r = {np.array(r)}")

# ── Collect per-step position intervals ───────────────────────────────────────

def collect_pos_history(theta, x0_ivl, scenarios, dt, num_steps):
    """Return list-of-lists: history[scenario_idx][step] = irx.Interval (2-D pos)."""
    history = [[] for _ in scenarios]
    for si, s in enumerate(scenarios):
        x = x0_ivl
        history[si].append(irx.Interval(lower=x.lower[:2], upper=x.upper[:2]))
        for _ in range(num_steps):
            x = cl_euler_step(s.emb_system, x, theta, s.p_interval, dt)
            history[si].append(irx.Interval(lower=x.lower[:2], upper=x.upper[:2]))
    return history

history = collect_pos_history(best_theta, x0_ivl, cl_scenarios, DT, NUM_STEPS)

# Verify CBF: all intervals clear of obstacle
print("\nCBF verification (h_min over final position interval):")
for si, s in enumerate(cl_scenarios):
    final_ivl = history[si][-1]
    # Reconstruct 3D interval for cbf_min (only px, py used)
    x3d = irx.Interval(
        lower=jnp.concatenate([final_ivl.lower, jnp.zeros(1)]),
        upper=jnp.concatenate([final_ivl.upper, jnp.zeros(1)]),
    )
    h_min = float(cbf_min_over_interval(x3d, obstacles)[0])
    status = "SAFE" if h_min >= 0 else "VIOLATED"
    print(f"  {s.name}: h_min = {h_min:.4f} m²  [{status}]")

# ── Plot ──────────────────────────────────────────────────────────────────────

COLORS = ['#2196F3', '#FF9800', '#4CAF50']   # blue, orange, green
ALPHA_BOX  = 0.18
ALPHA_LAST = 0.60

fig, ax = plt.subplots(figsize=(7, 6))

# -- Draw position-interval boxes at every time step --------------------------
for si, (s, color) in enumerate(zip(cl_scenarios, COLORS)):
    n_steps = len(history[si])
    centers_x, centers_y = [], []

    for k, ivl in enumerate(history[si]):
        xl = float(ivl.lower[0])
        xh = float(ivl.upper[0])
        yl = float(ivl.lower[1])
        yh = float(ivl.upper[1])

        is_last = (k == n_steps - 1)
        alpha   = ALPHA_LAST if is_last else ALPHA_BOX
        lw      = 1.4        if is_last else 0.4
        zorder  = 3          if is_last else 2

        rect = mpatches.FancyBboxPatch(
            (xl, yl), xh - xl, yh - yl,
            boxstyle="square,pad=0",
            linewidth=lw, edgecolor=color,
            facecolor=color, alpha=alpha,
            zorder=zorder,
        )
        ax.add_patch(rect)

        cx = (xl + xh) / 2
        cy = (yl + yh) / 2
        centers_x.append(cx)
        centers_y.append(cy)

    # Trajectory line through interval centres (thin, semi-transparent)
    ax.plot(centers_x, centers_y, color=color, linewidth=1.0,
            alpha=0.55, zorder=4)

    # Arrow on trajectory to show direction
    if len(centers_x) > 3:
        mid = len(centers_x) // 2
        ax.annotate(
            "", xy=(centers_x[mid + 1], centers_y[mid + 1]),
            xytext=(centers_x[mid], centers_y[mid]),
            arrowprops=dict(arrowstyle="-|>", color=color,
                            lw=1.5, mutation_scale=12),
            zorder=5,
        )

# -- Initial state interval (black outline) -----------------------------------
ix0l, ix0h = float(x0_ivl.lower[0]), float(x0_ivl.upper[0])
iy0l, iy0h = float(x0_ivl.lower[1]), float(x0_ivl.upper[1])
init_rect = mpatches.FancyBboxPatch(
    (ix0l, iy0l), ix0h - ix0l, iy0h - iy0l,
    boxstyle="square,pad=0",
    linewidth=2.0, edgecolor='black', facecolor='lightgrey', alpha=0.7,
    zorder=6,
)
ax.add_patch(init_rect)
ax.text((ix0l + ix0h) / 2, (iy0l + iy0h) / 2 - 0.035,
        r'$\mathcal{X}_0$', ha='center', va='top', fontsize=10,
        color='black', zorder=7)

# -- Obstacle disk ------------------------------------------------------------
obs_patch = plt.Circle(
    (OBS_CX, OBS_CY), OBS_R,
    color='#F44336', alpha=0.85, zorder=8,
)
ax.add_patch(obs_patch)
# Safety margin ring (dashed)
margin_ring = plt.Circle(
    (OBS_CX, OBS_CY), OBS_R,
    fill=False, linestyle='--', linewidth=1.5, edgecolor='#B71C1C',
    alpha=0.6, zorder=9,
)
ax.add_patch(margin_ring)
ax.text(OBS_CX, OBS_CY, 'Obstacle', ha='center', va='center',
        fontsize=8, color='white', fontweight='bold', zorder=10)

# -- Axis labels and formatting -----------------------------------------------
ax.set_xlabel(r'$p_x$ (m)', fontsize=12)
ax.set_ylabel(r'$p_y$ (m)', fontsize=12)
ax.set_aspect('equal', adjustable='datalim')
ax.grid(True, alpha=0.3, linewidth=0.7)
ax.tick_params(labelsize=10)

# -- Legend -------------------------------------------------------------------
legend_handles = [
    mpatches.Patch(facecolor=COLORS[i], edgecolor=COLORS[i],
                   alpha=0.7, label=cl_scenarios[i].name)
    for i in range(len(cl_scenarios))
]
legend_handles.append(
    mpatches.Patch(facecolor='lightgrey', edgecolor='black',
                   alpha=0.7, label='Initial set $\\mathcal{X}_0$')
)
legend_handles.append(
    mpatches.Patch(facecolor='#F44336', edgecolor='#F44336',
                   alpha=0.85, label='Obstacle')
)
ax.legend(handles=legend_handles, fontsize=9, loc='upper right',
          framealpha=0.9)

# -- Title annotation with CBF verification result ----------------------------
all_safe = all(
    float(cbf_min_over_interval(
        irx.Interval(
            lower=jnp.concatenate([ivl.lower, jnp.zeros(1)]),
            upper=jnp.concatenate([ivl.upper, jnp.zeros(1)]),
        ),
        obstacles,
    )[0]) >= 0
    for hist in history
    for ivl in hist
)
safety_text = "All reachable sets clear of obstacle" if all_safe else "WARNING: CBF violated"
safety_color = '#1B5E20' if all_safe else '#B71C1C'

ax.set_title(
    f"Output-Feedback CBF Controller — Faulty Nonholonomic Car\n"
    f"$T={NUM_STEPS}\\times{DT}={NUM_STEPS*DT:.2f}$ s | "
    f"$K_{{\\max}}=5$, $w_{{\\rm cbf}}={CBF_WEIGHT}$\n"
    f"\\textit{{{safety_text}}}",
    usetex=False, fontsize=10,
)
# Stamp safety verdict bottom-left
ax.text(0.02, 0.02, f"✓ {safety_text}" if all_safe else f"✗ {safety_text}",
        transform=ax.transAxes, fontsize=9, color=safety_color,
        va='bottom', ha='left',
        bbox=dict(facecolor='white', alpha=0.8, edgecolor=safety_color, pad=3),
        zorder=11)

# Autoscale with a bit of padding
all_xs = ([float(x0_ivl.lower[0]), float(x0_ivl.upper[0])] +
          [float(ivl.lower[0]) for hist in history for ivl in hist] +
          [float(ivl.upper[0]) for hist in history for ivl in hist] +
          [OBS_CX - OBS_R - 0.05, OBS_CX + OBS_R + 0.05])
all_ys = ([float(x0_ivl.lower[1]), float(x0_ivl.upper[1])] +
          [float(ivl.lower[1]) for hist in history for ivl in hist] +
          [float(ivl.upper[1]) for hist in history for ivl in hist] +
          [OBS_CY - OBS_R - 0.05, OBS_CY + OBS_R + 0.05])
pad = 0.07
ax.set_xlim(min(all_xs) - pad, max(all_xs) + pad)
ax.set_ylim(min(all_ys) - pad, max(all_ys) + pad)

plt.tight_layout()
fname = HERE / 'output_feedback_cbf_statespace.pdf'
plt.savefig(fname, bbox_inches='tight')
print(f"\nSaved: {fname}")
plt.close(fig)
