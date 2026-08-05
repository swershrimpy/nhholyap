"""
Go2 System Report – Plot Generator
====================================
Generates publication-quality figures for go2_system_report.tex.

Figures produced (all in examples/go2/):
  fig1_go2_nominal_trajectory.pdf       -- nominal circular trajectory + control inputs
  fig2_go2_fault_scenarios.pdf          -- fault scenario parameter diagram
  fig3_go2_reachable_sets_openloop.pdf  -- position reachable sets under u_nom only
  fig4_go2_reachable_sets_feedback.pdf  -- position reachable sets under optimised feedback
  fig5_go2_feedback_gains.pdf           -- K[t] entries over time
  fig6_go2_overlap_comparison.pdf       -- pairwise overlap before/after optimisation

Usage:
    JAX_PLATFORMS=cpu python go2_system_report_plots.py
"""

import os; os.environ['JAX_PLATFORMS'] = 'cpu'
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import jax
import jax.numpy as jnp

HERE = Path(__file__).resolve().parent
plt.rcParams.update({'font.family': 'serif', 'font.size': 10})

COLORS = {
    'Nominal':        '#3498db',
    'Actuator Fault': '#2ecc71',
    'Sensor Fault':   '#e74c3c',
}

# ══════════════════════════════════════════════════════════════════════════════
# Interval helpers (JAX-grad-safe — no jnp.array([traced, ...]))
# ══════════════════════════════════════════════════════════════════════════════

def _cos_bounds(tl, tu):
    """Tight [cos_lo, cos_hi] over theta ∈ [tl, tu]."""
    c1, c2 = jnp.cos(tl), jnp.cos(tu)
    c_lo = jnp.minimum(c1, c2)
    c_hi = jnp.maximum(c1, c2)
    two_pi = 2.0 * jnp.pi
    # contains 0 + 2πk  →  cos_hi = 1
    has_max = jnp.ceil((tl) / two_pi) <= jnp.floor((tu) / two_pi)
    # contains π + 2πk  →  cos_lo = -1
    has_min = jnp.ceil((tl - jnp.pi) / two_pi) <= jnp.floor((tu - jnp.pi) / two_pi)
    c_hi = jnp.where(has_max, 1.0, c_hi)
    c_lo = jnp.where(has_min, -1.0, c_lo)
    return c_lo, c_hi

def _sin_bounds(tl, tu):
    """Tight [sin_lo, sin_hi] over theta ∈ [tl, tu]."""
    s1, s2 = jnp.sin(tl), jnp.sin(tu)
    s_lo = jnp.minimum(s1, s2)
    s_hi = jnp.maximum(s1, s2)
    two_pi = 2.0 * jnp.pi
    half_pi = jnp.pi / 2.0
    has_max = jnp.ceil((tl - half_pi) / two_pi) <= jnp.floor((tu - half_pi) / two_pi)
    has_min = jnp.ceil((tl + half_pi) / two_pi) <= jnp.floor((tu + half_pi) / two_pi)
    s_hi = jnp.where(has_max,  1.0, s_hi)
    s_lo = jnp.where(has_min, -1.0, s_lo)
    return s_lo, s_hi


def euler_step_nom(xlo, xhi, u, alpha_lo, alpha_hi, dt):
    """Euler step for nominal/actuator-fault unicycle.

    ẋ = vx·cos θ − α·vy·sin θ
    ẏ = vx·sin θ + α·vy·cos θ
    θ̇ = β·ω          (β = alpha here for simplicity)
    """
    tl, tu = xlo[2], xhi[2]
    vx, vy, omega = u[0], u[1], u[2]

    cos_lo, cos_hi = _cos_bounds(tl, tu)
    sin_lo, sin_hi = _sin_bounds(tl, tu)

    # dpx/dt
    vx_cos_lo = jnp.minimum(vx * cos_lo, vx * cos_hi)
    vx_cos_hi = jnp.maximum(vx * cos_lo, vx * cos_hi)
    mvy_sin_lo = jnp.minimum(-vy * sin_lo, -vy * sin_hi)
    mvy_sin_hi = jnp.maximum(-vy * sin_lo, -vy * sin_hi)
    dpx_lo = vx_cos_lo + mvy_sin_lo
    dpx_hi = vx_cos_hi + mvy_sin_hi

    # dpy/dt
    vx_sin_lo = jnp.minimum(vx * sin_lo, vx * sin_hi)
    vx_sin_hi = jnp.maximum(vx * sin_lo, vx * sin_hi)
    vy_cos_lo = jnp.minimum(vy * cos_lo, vy * cos_hi)
    vy_cos_hi = jnp.maximum(vy * cos_lo, vy * cos_hi)
    dpy_lo = vx_sin_lo + vy_cos_lo
    dpy_hi = vx_sin_hi + vy_cos_hi

    # dθ/dt
    dth_lo = jnp.minimum(alpha_lo * omega, alpha_hi * omega)
    dth_hi = jnp.maximum(alpha_lo * omega, alpha_hi * omega)

    nlo = xlo + dt * jnp.stack([dpx_lo, dpy_lo, dth_lo])
    nhi = xhi + dt * jnp.stack([dpx_hi, dpy_hi, dth_hi])
    return nlo, nhi


def euler_step_sensor(xlo, xhi, u, vy_noise_lo, vy_noise_hi, dt):
    """Euler step for sensor-fault dead-reckoned dynamics.

    vy_corrupted = (1−ω²)·vy + ω²·vy_noise
    """
    tl, tu = xlo[2], xhi[2]
    vx, vy, omega = u[0], u[1], u[2]
    w2 = omega ** 2

    # vy_corrupted interval
    nom_part = (1 - w2) * vy
    noise_lo = w2 * vy_noise_lo
    noise_hi = w2 * vy_noise_hi
    vyc_lo = nom_part + noise_lo
    vyc_hi = nom_part + noise_hi

    cos_lo, cos_hi = _cos_bounds(tl, tu)
    sin_lo, sin_hi = _sin_bounds(tl, tu)

    # dpx/dt = vx·cos θ − vy_c·sin θ
    t1_lo = jnp.minimum(vx * cos_lo, vx * cos_hi)
    t1_hi = jnp.maximum(vx * cos_lo, vx * cos_hi)
    # −vy_c·sin θ: four combinations
    ns_lo = jnp.minimum(jnp.minimum(-vyc_lo * sin_lo, -vyc_lo * sin_hi),
                        jnp.minimum(-vyc_hi * sin_lo, -vyc_hi * sin_hi))
    ns_hi = jnp.maximum(jnp.maximum(-vyc_lo * sin_lo, -vyc_lo * sin_hi),
                        jnp.maximum(-vyc_hi * sin_lo, -vyc_hi * sin_hi))
    dpx_lo = t1_lo + ns_lo
    dpx_hi = t1_hi + ns_hi

    # dpy/dt = vx·sin θ + vy_c·cos θ
    t2_lo = jnp.minimum(vx * sin_lo, vx * sin_hi)
    t2_hi = jnp.maximum(vx * sin_lo, vx * sin_hi)
    yc_lo = jnp.minimum(jnp.minimum(vyc_lo * cos_lo, vyc_lo * cos_hi),
                        jnp.minimum(vyc_hi * cos_lo, vyc_hi * cos_hi))
    yc_hi = jnp.maximum(jnp.maximum(vyc_lo * cos_lo, vyc_lo * cos_hi),
                        jnp.maximum(vyc_hi * cos_lo, vyc_hi * cos_hi))
    dpy_lo = t2_lo + yc_lo
    dpy_hi = t2_hi + yc_hi

    dth_lo = omega
    dth_hi = omega

    nlo = xlo + dt * jnp.stack([dpx_lo, dpy_lo, dth_lo])
    nhi = xhi + dt * jnp.stack([dpx_hi, dpy_hi, dth_hi])
    return nlo, nhi


def overlap_1d(alo, ahi, blo, bhi):
    return jnp.maximum(0.0, jnp.minimum(ahi, bhi) - jnp.maximum(alo, blo))

def overlap_2d(alo, ahi, blo, bhi):
    """2-D position interval overlap volume."""
    dx = overlap_1d(alo[0], ahi[0], blo[0], bhi[0])
    dy = overlap_1d(alo[1], ahi[1], blo[1], bhi[1])
    return dx * dy


# ══════════════════════════════════════════════════════════════════════════════
# Scenario propagation (returns list of (lo, hi) pairs per time step)
# ══════════════════════════════════════════════════════════════════════════════

# Scenario fault parameters
NOM_ALPHA = (1.0, 1.0)
ACT_ALPHA = (0.60, 0.80)
SENSOR_VY_NOISE = (-0.25, 0.25)
U_LO = jnp.array([-0.6, -0.6, -0.6])
U_HI = jnp.array([ 0.6,  0.6,  0.6])


def propagate_nom(x0lo, x0hi, u_seq, dt):
    xlo, xhi = x0lo, x0hi
    hist = [(xlo, xhi)]
    for k in range(len(u_seq)):
        xlo, xhi = euler_step_nom(xlo, xhi, u_seq[k], NOM_ALPHA[0], NOM_ALPHA[1], dt)
        hist.append((xlo, xhi))
    return hist


def propagate_act(x0lo, x0hi, u_seq, dt):
    xlo, xhi = x0lo, x0hi
    hist = [(xlo, xhi)]
    for k in range(len(u_seq)):
        xlo, xhi = euler_step_nom(xlo, xhi, u_seq[k], ACT_ALPHA[0], ACT_ALPHA[1], dt)
        hist.append((xlo, xhi))
    return hist


def propagate_sensor(x0lo, x0hi, u_seq, dt):
    xlo, xhi = x0lo, x0hi
    hist = [(xlo, xhi)]
    for k in range(len(u_seq)):
        xlo, xhi = euler_step_sensor(xlo, xhi, u_seq[k],
                                     SENSOR_VY_NOISE[0], SENSOR_VY_NOISE[1], dt)
        hist.append((xlo, xhi))
    return hist


def separation_loss(u_seq_flat, x0lo, x0hi, dt, N):
    """Sum of pairwise 2-D position overlaps at the final time step."""
    u_seq = jnp.clip(u_seq_flat.reshape(N, 3), U_LO, U_HI)

    # Final state intervals (we only need the last step for the loss)
    xlo_n, xhi_n = x0lo, x0hi
    xlo_a, xhi_a = x0lo, x0hi
    xlo_s, xhi_s = x0lo, x0hi
    for k in range(N):
        xlo_n, xhi_n = euler_step_nom(xlo_n, xhi_n, u_seq[k], NOM_ALPHA[0], NOM_ALPHA[1], dt)
        xlo_a, xhi_a = euler_step_nom(xlo_a, xhi_a, u_seq[k], ACT_ALPHA[0], ACT_ALPHA[1], dt)
        xlo_s, xhi_s = euler_step_sensor(xlo_s, xhi_s, u_seq[k],
                                          SENSOR_VY_NOISE[0], SENSOR_VY_NOISE[1], dt)

    loss = (overlap_2d(xlo_n, xhi_n, xlo_a, xhi_a)
          + overlap_2d(xlo_n, xhi_n, xlo_s, xhi_s)
          + overlap_2d(xlo_a, xhi_a, xlo_s, xhi_s))
    return loss


def separation_loss_fb(K_flat, u_nom_seq, x0lo, x0hi, y_nom_seq, dt, N,
                       lambda_track=0.01, obs_unc=0.05):
    """Separation loss under output feedback u = u_nom + K[k] @ (y - y_nom)."""
    K_seq = K_flat.reshape(N, 3, 3)

    def _propagate_nom_fb(x0lo, x0hi):
        xlo, xhi = x0lo, x0hi
        for k in range(N):
            # y interval = x interval + obs_uncertainty
            y_lo = xlo - obs_unc
            y_hi = xhi + obs_unc
            dy_lo = y_lo - y_nom_seq[k]
            dy_hi = y_hi - y_nom_seq[k]
            # K @ dy interval (row-wise)
            Kk = K_seq[k]
            fb_lo = jnp.stack([
                jnp.sum(jnp.where(Kk[i] >= 0, Kk[i] * dy_lo, Kk[i] * dy_hi))
                for i in range(3)
            ])
            fb_hi = jnp.stack([
                jnp.sum(jnp.where(Kk[i] >= 0, Kk[i] * dy_hi, Kk[i] * dy_lo))
                for i in range(3)
            ])
            u_lo = jnp.clip(u_nom_seq[k] + fb_lo, U_LO, U_HI)
            u_hi = jnp.clip(u_nom_seq[k] + fb_hi, U_LO, U_HI)
            # propagate with both extremes and union
            nlo1, nhi1 = euler_step_nom(xlo, xhi, u_lo, NOM_ALPHA[0], NOM_ALPHA[1], dt)
            nlo2, nhi2 = euler_step_nom(xlo, xhi, u_hi, NOM_ALPHA[0], NOM_ALPHA[1], dt)
            xlo = jnp.minimum(nlo1, nlo2)
            xhi = jnp.maximum(nhi1, nhi2)
        return xlo, xhi

    def _propagate_act_fb(x0lo, x0hi):
        xlo, xhi = x0lo, x0hi
        for k in range(N):
            y_lo = xlo - obs_unc
            y_hi = xhi + obs_unc
            dy_lo = y_lo - y_nom_seq[k]
            dy_hi = y_hi - y_nom_seq[k]
            Kk = K_seq[k]
            fb_lo = jnp.stack([
                jnp.sum(jnp.where(Kk[i] >= 0, Kk[i] * dy_lo, Kk[i] * dy_hi))
                for i in range(3)
            ])
            fb_hi = jnp.stack([
                jnp.sum(jnp.where(Kk[i] >= 0, Kk[i] * dy_hi, Kk[i] * dy_lo))
                for i in range(3)
            ])
            u_lo = jnp.clip(u_nom_seq[k] + fb_lo, U_LO, U_HI)
            u_hi = jnp.clip(u_nom_seq[k] + fb_hi, U_LO, U_HI)
            nlo1, nhi1 = euler_step_nom(xlo, xhi, u_lo, ACT_ALPHA[0], ACT_ALPHA[1], dt)
            nlo2, nhi2 = euler_step_nom(xlo, xhi, u_hi, ACT_ALPHA[0], ACT_ALPHA[1], dt)
            xlo = jnp.minimum(nlo1, nlo2)
            xhi = jnp.maximum(nhi1, nhi2)
        return xlo, xhi

    def _propagate_sensor_fb(x0lo, x0hi):
        xlo, xhi = x0lo, x0hi
        for k in range(N):
            # sensor fault: observed state has sensor-offset theta (we add obs_unc)
            y_lo = xlo - obs_unc
            y_hi = xhi + obs_unc
            dy_lo = y_lo - y_nom_seq[k]
            dy_hi = y_hi - y_nom_seq[k]
            Kk = K_seq[k]
            fb_lo = jnp.stack([
                jnp.sum(jnp.where(Kk[i] >= 0, Kk[i] * dy_lo, Kk[i] * dy_hi))
                for i in range(3)
            ])
            fb_hi = jnp.stack([
                jnp.sum(jnp.where(Kk[i] >= 0, Kk[i] * dy_hi, Kk[i] * dy_lo))
                for i in range(3)
            ])
            u_lo = jnp.clip(u_nom_seq[k] + fb_lo, U_LO, U_HI)
            u_hi = jnp.clip(u_nom_seq[k] + fb_hi, U_LO, U_HI)
            nlo1, nhi1 = euler_step_sensor(xlo, xhi, u_lo,
                                           SENSOR_VY_NOISE[0], SENSOR_VY_NOISE[1], dt)
            nlo2, nhi2 = euler_step_sensor(xlo, xhi, u_hi,
                                           SENSOR_VY_NOISE[0], SENSOR_VY_NOISE[1], dt)
            xlo = jnp.minimum(nlo1, nlo2)
            xhi = jnp.maximum(nhi1, nhi2)
        return xlo, xhi

    nlo, nhi = _propagate_nom_fb(x0lo, x0hi)
    alo, ahi = _propagate_act_fb(x0lo, x0hi)
    slo, shi = _propagate_sensor_fb(x0lo, x0hi)

    sep_loss = (overlap_2d(nlo, nhi, alo, ahi)
              + overlap_2d(nlo, nhi, slo, shi)
              + overlap_2d(alo, ahi, slo, shi))
    track_loss = jnp.mean(jnp.square(K_flat))
    return sep_loss + lambda_track * track_loss


def propagate_all_fb(K_flat, u_nom_seq, x0lo, x0hi, y_nom_seq, dt, N, obs_unc=0.05):
    """Return full histories under output-feedback for all three scenarios."""
    K_seq = K_flat.reshape(N, 3, 3)

    def _run(x0lo, x0hi, step_fn):
        xlo, xhi = x0lo, x0hi
        hist = [(xlo, xhi)]
        for k in range(N):
            y_lo = xlo - obs_unc
            y_hi = xhi + obs_unc
            dy_lo = y_lo - y_nom_seq[k]
            dy_hi = y_hi - y_nom_seq[k]
            Kk = K_seq[k]
            fb_lo = jnp.stack([
                jnp.sum(jnp.where(Kk[i] >= 0, Kk[i] * dy_lo, Kk[i] * dy_hi))
                for i in range(3)
            ])
            fb_hi = jnp.stack([
                jnp.sum(jnp.where(Kk[i] >= 0, Kk[i] * dy_hi, Kk[i] * dy_lo))
                for i in range(3)
            ])
            u_lo = jnp.clip(u_nom_seq[k] + fb_lo, U_LO, U_HI)
            u_hi = jnp.clip(u_nom_seq[k] + fb_hi, U_LO, U_HI)
            nlo1, nhi1 = step_fn(xlo, xhi, u_lo)
            nlo2, nhi2 = step_fn(xlo, xhi, u_hi)
            xlo = jnp.minimum(nlo1, nlo2)
            xhi = jnp.maximum(nhi1, nhi2)
            hist.append((xlo, xhi))
        return hist

    nom_hist = _run(x0lo, x0hi,
        lambda xl, xh, u: euler_step_nom(xl, xh, u, NOM_ALPHA[0], NOM_ALPHA[1], dt))
    act_hist = _run(x0lo, x0hi,
        lambda xl, xh, u: euler_step_nom(xl, xh, u, ACT_ALPHA[0], ACT_ALPHA[1], dt))
    sen_hist = _run(x0lo, x0hi,
        lambda xl, xh, u: euler_step_sensor(xl, xh, u,
                                            SENSOR_VY_NOISE[0], SENSOR_VY_NOISE[1], dt))
    return {'Nominal': nom_hist, 'Actuator Fault': act_hist, 'Sensor Fault': sen_hist}


# ══════════════════════════════════════════════════════════════════════════════
# Shared setup
# ══════════════════════════════════════════════════════════════════════════════
N         = 20
DT        = 0.5
t_arr     = np.linspace(0, (N - 1) * DT, N)
radius    = 2.0
omega_nom = 0.3

x_nom_arr  = radius * np.cos(omega_nom * t_arr)
y_nom_arr  = radius * np.sin(omega_nom * t_arr)
theta_nom  = omega_nom * t_arr + np.pi / 2

x_nom_traj = np.column_stack([x_nom_arr, y_nom_arr, theta_nom])
vx_nom     = -radius * omega_nom * np.sin(omega_nom * t_arr)
vy_nom     =  radius * omega_nom * np.cos(omega_nom * t_arr)
u_nom_seq  = jnp.array(np.column_stack([vx_nom, vy_nom, np.ones(N) * omega_nom]))
y_nom_seq  = u_nom_seq  # C = I, so y_nom = x_nom (state = observation here)

x0lo = jnp.array([-0.05, -0.05, -0.02])
x0hi = jnp.array([ 0.05,  0.05,  0.02])

scenario_names = ['Nominal', 'Actuator Fault', 'Sensor Fault']


# ══════════════════════════════════════════════════════════════════════════════
# Figure 1 — Nominal trajectory
# ══════════════════════════════════════════════════════════════════════════════
print("Figure 1: nominal trajectory …")
fig1, (ax1a, ax1b) = plt.subplots(1, 2, figsize=(11, 4.5))

ax1a.plot(x_nom_arr, y_nom_arr, 'b-o', linewidth=2, markersize=4, label='Nominal path')
ax1a.plot(x_nom_arr[0],  y_nom_arr[0],  'go', markersize=10, zorder=5, label='Start')
ax1a.plot(x_nom_arr[-1], y_nom_arr[-1], 'rs', markersize=10, zorder=5, label='End')
for idx in [0, 4, 8, 12, 16]:
    dx = 0.28 * np.cos(theta_nom[idx])
    dy = 0.28 * np.sin(theta_nom[idx])
    ax1a.annotate('', xy=(x_nom_arr[idx]+dx, y_nom_arr[idx]+dy),
                  xytext=(x_nom_arr[idx], y_nom_arr[idx]),
                  arrowprops=dict(arrowstyle='->', color='navy', lw=1.8))
ax1a.set_xlabel('$p_x$ (m)'); ax1a.set_ylabel('$p_y$ (m)')
ax1a.set_title('Nominal Trajectory', fontweight='bold')
ax1a.legend(fontsize=9); ax1a.grid(True, alpha=0.3); ax1a.set_aspect('equal')

ax1b.plot(t_arr, vx_nom,              'b-',  linewidth=2, label='$v_x$ (m/s)')
ax1b.plot(t_arr, vy_nom,              'r--', linewidth=2, label='$v_y$ (m/s)')
ax1b.plot(t_arr, np.ones(N)*omega_nom,'g-.', linewidth=2, label='$\\omega$ (rad/s)')
ax1b.set_xlabel('Time (s)'); ax1b.set_ylabel('Control value')
ax1b.set_title('Nominal Control Inputs', fontweight='bold')
ax1b.legend(fontsize=9); ax1b.grid(True, alpha=0.3)

plt.tight_layout()
f1 = HERE / 'fig1_go2_nominal_trajectory.pdf'
plt.savefig(f1, bbox_inches='tight'); plt.close(fig1)
print(f"  Saved {f1.name}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 2 — Fault scenario diagram
# ══════════════════════════════════════════════════════════════════════════════
print("Figure 2: fault scenario diagram …")
fig2, axes2 = plt.subplots(1, 3, figsize=(13, 4.5))

sc_info = [
    dict(name='Nominal',        color='#3498db',
         params=[r'$\alpha = \beta = 1$', r'$v_y^{\rm noise}=0$'],
         eff=1.0),
    dict(name='Actuator Fault', color='#2ecc71',
         params=[r'$\alpha,\beta \in [0.6,\,0.8]$', 'Reduced lateral/yaw authority'],
         eff=0.7),
    dict(name='Sensor Fault',   color='#e74c3c',
         params=[r'$v_y^{\rm noise}\in[-\varepsilon,+\varepsilon]$',
                 r'$\varepsilon=0.25$ m/s'],
         eff=1.0),
]
for ax, sd in zip(axes2, sc_info):
    ax.set_xlim(0, 5); ax.set_ylim(0, 5); ax.set_aspect('equal')
    ax.set_facecolor('#f8f9fa'); ax.grid(True, alpha=0.25)
    rx, ry = 1.6, 2.5
    circle = plt.Circle((rx, ry), 0.4, color=sd['color'], alpha=0.75, zorder=3)
    ax.add_patch(circle)
    ax.annotate('', xy=(rx+0.75, ry), xytext=(rx+0.4, ry),
                arrowprops=dict(arrowstyle='->', color='black', lw=2.2), zorder=4)
    vy_draw = 0.7 * sd['eff']
    ax.annotate('', xy=(rx, ry+vy_draw), xytext=(rx, ry+0.4),
                arrowprops=dict(arrowstyle='->', color='darkgreen', lw=2.0), zorder=4)
    ax.text(rx+0.12, ry+0.4+vy_draw/2, '$v_y$', fontsize=9, color='darkgreen')
    if sd['name'] == 'Sensor Fault':
        ax.text(rx-0.3, ry-0.88, 'sensor fault', fontsize=8, ha='center',
                color='darkred', style='italic',
                bbox=dict(boxstyle='round,pad=0.2', fc='#ffdddd', ec='darkred', alpha=0.8))
    ax.text(2.85, 2.5, '\n'.join(sd['params']), fontsize=9, va='center',
            bbox=dict(boxstyle='round,pad=0.45', fc='lightyellow', ec='gray', alpha=0.95))
    ax.set_title(sd['name'], fontweight='bold', color=sd['color'], fontsize=11)
    ax.set_xlabel('$p_x$ (m)'); ax.set_ylabel('$p_y$ (m)'); ax.tick_params(labelsize=8)

plt.suptitle('Go2 Fault Scenarios', fontsize=13, fontweight='bold')
plt.tight_layout()
f2 = HERE / 'fig2_go2_fault_scenarios.pdf'
plt.savefig(f2, bbox_inches='tight'); plt.close(fig2)
print(f"  Saved {f2.name}")


# ══════════════════════════════════════════════════════════════════════════════
# Propagate open-loop (u_nom only)
# ══════════════════════════════════════════════════════════════════════════════
print("Propagating open-loop …")
hist_ol = {
    'Nominal':        propagate_nom(x0lo, x0hi, u_nom_seq, DT),
    'Actuator Fault': propagate_act(x0lo, x0hi, u_nom_seq, DT),
    'Sensor Fault':   propagate_sensor(x0lo, x0hi, u_nom_seq, DT),
}

def _pairwise_overlap(hist):
    names = list(hist.keys())
    total = 0.0
    result = {}
    for ii in range(len(names)):
        for jj in range(ii+1, len(names)):
            ni, nj = names[ii], names[jj]
            lo_i, hi_i = hist[ni][-1]
            lo_j, hi_j = hist[nj][-1]
            ov = float(overlap_2d(lo_i, hi_i, lo_j, hi_j))
            result[(ni, nj)] = ov
            total += ov
    return result, total

ov_ol, total_ol = _pairwise_overlap(hist_ol)
print(f"  Open-loop total overlap: {total_ol:.6f} m²")


# ══════════════════════════════════════════════════════════════════════════════
# Optimise open-loop separating input (constant u per step)
# then separately optimise output-feedback gains K
# ══════════════════════════════════════════════════════════════════════════════

# -- Step 1: warm-start from the nominal control, optimise open-loop u
print("Optimising open-loop separating input …")
loss_fn_ol = jax.jit(lambda u_flat: separation_loss(u_flat, x0lo, x0hi, DT, N))
grad_fn_ol = jax.jit(jax.grad(loss_fn_ol))

u_flat = u_nom_seq.flatten()
lr_ol = 0.05
for it in range(200):
    g = grad_fn_ol(u_flat)
    u_flat = u_flat - lr_ol * g
    if it % 50 == 0:
        print(f"  ol iter {it:3d}: loss={float(loss_fn_ol(u_flat)):.6f}")
u_opt = u_flat.reshape(N, 3)
print(f"  Open-loop optimised loss: {float(loss_fn_ol(u_flat)):.6f}")


# -- Step 2: optimise output-feedback K (warm-start K=0, lr=1e-3)
print("Optimising output-feedback K …")
loss_fn_fb = lambda K_flat: separation_loss_fb(
    K_flat, u_nom_seq, x0lo, x0hi, y_nom_seq, DT, N,
    lambda_track=0.01, obs_unc=0.05)
grad_fn_fb = jax.grad(loss_fn_fb)

K_flat = jnp.zeros(N * 3 * 3)
lr_fb = 0.005
for it in range(150):
    g = grad_fn_fb(K_flat)
    K_flat = K_flat - lr_fb * g
    if it % 50 == 0:
        print(f"  fb iter {it:3d}: loss={float(loss_fn_fb(K_flat)):.6f}")
K_opt_flat = K_flat
K_opt = np.array(K_opt_flat.reshape(N, 3, 3))
loss_fb = float(loss_fn_fb(K_opt_flat))
print(f"  Feedback optimised loss: {loss_fb:.6f}")


# propagate full histories under optimised feedback
hist_fb = propagate_all_fb(K_opt_flat, u_nom_seq, x0lo, x0hi, y_nom_seq, DT, N)
ov_fb, total_fb = _pairwise_overlap(hist_fb)
print(f"  Feedback total overlap: {total_fb:.6f} m²")


# ══════════════════════════════════════════════════════════════════════════════
# Helper: draw position interval histories
# ══════════════════════════════════════════════════════════════════════════════
def _draw_pos_intervals(ax, hist, x_nom, y_nom, title):
    all_px, all_py = list(x_nom), list(y_nom)
    for name in scenario_names:
        col  = COLORS[name]
        h    = hist[name]
        for k, (lo, hi) in enumerate(h):
            xl, xh = float(lo[0]), float(hi[0])
            yl, yh = float(lo[1]), float(hi[1])
            all_px += [xl, xh]; all_py += [yl, yh]
            alpha_r = 0.12 + 0.22 * (k / len(h))
            rect = Rectangle((xl, yl), xh-xl, yh-yl,
                              linewidth=0.5, edgecolor=col, facecolor=col, alpha=alpha_r)
            ax.add_patch(rect)
        cx = [(float(lo[0])+float(hi[0]))/2 for lo, hi in h]
        cy = [(float(lo[1])+float(hi[1]))/2 for lo, hi in h]
        ax.plot(cx, cy, '-', color=col, linewidth=2.0, label=name, zorder=3)

    ax.plot(x_nom, y_nom, 'k--', linewidth=1.2, alpha=0.55, label='Nominal', zorder=2)
    ax.plot(x_nom[0],  y_nom[0],  'k^', markersize=7, zorder=4)
    ax.plot(x_nom[-1], y_nom[-1], 'ks', markersize=7, zorder=4)
    pad = 0.35
    ax.set_xlim(min(all_px)-pad, max(all_px)+pad)
    ax.set_ylim(min(all_py)-pad, max(all_py)+pad)
    ax.set_aspect('equal')
    ax.set_xlabel('$p_x$ (m)'); ax.set_ylabel('$p_y$ (m)')
    ax.set_title(title, fontweight='bold')
    ax.legend(fontsize=9, loc='upper right'); ax.grid(True, alpha=0.3)


# ══════════════════════════════════════════════════════════════════════════════
# Figure 3 — Reachable sets, open-loop (u_nom)
# ══════════════════════════════════════════════════════════════════════════════
print("Figure 3: reachable sets open-loop …")
fig3, ax3 = plt.subplots(figsize=(6.5, 6))
_draw_pos_intervals(ax3, hist_ol, x_nom_arr, y_nom_arr,
                    f'Reachable Sets — Open-loop ($u_{{\\rm nom}}$)\n'
                    f'Total overlap: {total_ol:.4f} m$^2$')
plt.tight_layout()
f3 = HERE / 'fig3_go2_reachable_sets_openloop.pdf'
plt.savefig(f3, bbox_inches='tight'); plt.close(fig3)
print(f"  Saved {f3.name}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 4 — Reachable sets, optimised feedback
# ══════════════════════════════════════════════════════════════════════════════
print("Figure 4: reachable sets with feedback …")
fig4, ax4 = plt.subplots(figsize=(6.5, 6))
_draw_pos_intervals(ax4, hist_fb, x_nom_arr, y_nom_arr,
                    f'Reachable Sets — Optimised Feedback ($K_{{\\rm opt}}$)\n'
                    f'Total overlap: {total_fb:.4f} m$^2$')
plt.tight_layout()
f4 = HERE / 'fig4_go2_reachable_sets_feedback.pdf'
plt.savefig(f4, bbox_inches='tight'); plt.close(fig4)
print(f"  Saved {f4.name}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 5 — Feedback gain evolution
# ══════════════════════════════════════════════════════════════════════════════
print("Figure 5: feedback gain evolution …")
ctrl_sym = ['vx', 'vy', 'omega']
obs_sym  = ['px', 'py', 'theta']
ctrl_tex = [r'$v_x$', r'$v_y$', r'$\omega$']
obs_tex  = [r'$p_x$', r'$p_y$', r'$\theta$']

fig5, axes5 = plt.subplots(3, 3, figsize=(11, 8), sharex=True)
fig5.suptitle('Optimised Feedback Gain Matrix $K[t]$', fontsize=11, fontweight='bold')

for i in range(3):
    for j in range(3):
        ax = axes5[i, j]
        ax.plot(t_arr, K_opt[:, i, j], 'b-', linewidth=1.8)
        ax.axhline(0, color='k', linestyle='--', linewidth=0.7, alpha=0.4)
        ax.set_title(f'$K_{{{ctrl_sym[i]},{obs_sym[j]}}}[t]$  '
                     f'({ctrl_tex[i]} gain on {obs_tex[j]})', fontsize=9)
        ax.grid(True, alpha=0.3); ax.tick_params(labelsize=8)
        if i == 2: ax.set_xlabel('Time (s)')
        if j == 0: ax.set_ylabel('Gain value')

plt.tight_layout()
f5 = HERE / 'fig5_go2_feedback_gains.pdf'
plt.savefig(f5, bbox_inches='tight'); plt.close(fig5)
print(f"  Saved {f5.name}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 6 — Pairwise overlap comparison
# ══════════════════════════════════════════════════════════════════════════════
print("Figure 6: overlap comparison …")

keys     = list(ov_ol.keys())
x_pos    = np.arange(len(keys))
w        = 0.35

fig6, ax6 = plt.subplots(figsize=(9, 5))
bars_ol = ax6.bar(x_pos - w/2, [ov_ol[k] for k in keys], w,
                  color='#95a5a6', edgecolor='black', linewidth=1,
                  label='Open-loop ($u_{\\rm nom}$)')
bars_fb = ax6.bar(x_pos + w/2, [ov_fb[k] for k in keys], w,
                  color='#e74c3c', edgecolor='black', linewidth=1,
                  label='Optimised feedback ($K_{\\rm opt}$)')

ax6.set_xticks(x_pos)
ax6.set_xticklabels([f'{a[:3]}…\nvs {b[:3]}…' for a, b in keys], fontsize=9)
ax6.set_ylabel('Overlap volume (m²)', fontsize=11)
ax6.set_title('Pairwise Position-Interval Overlap at $t = T$\n'
              '(zero = fully separated, fault is uniquely identifiable)',
              fontsize=11, fontweight='bold')
ax6.legend(fontsize=10); ax6.grid(True, alpha=0.3, axis='y')

for bar, val in zip(list(bars_ol)+list(bars_fb),
                    [ov_ol[k] for k in keys]+[ov_fb[k] for k in keys]):
    h = bar.get_height()
    ax6.text(bar.get_x() + bar.get_width()/2, h + 2e-4,
             f'{h:.4f}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
f6 = HERE / 'fig6_go2_overlap_comparison.pdf'
plt.savefig(f6, bbox_inches='tight'); plt.close(fig6)
print(f"  Saved {f6.name}")

print("\nAll figures saved.")
print(f"  Open-loop total overlap : {total_ol:.6f} m²")
print(f"  Feedback total overlap  : {total_fb:.6f} m²")
if total_ol > 0:
    print(f"  Reduction              : {(total_ol - total_fb)/total_ol*100:.1f}%")
