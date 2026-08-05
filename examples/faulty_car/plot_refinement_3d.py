"""
Self-contained script: 3-D refinement interval plot (Section 13 style).
Saves one PDF per scenario pair in the current directory.
"""
import sys
from pathlib import Path

import jax.numpy as jnp
import immrax as irx
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from faulty_car_separating_input import (
    create_scenarios,
    euler_step,
    _obs_interval,
    optimize_refined_gpu,
)

# ── Setup ────────────────────────────────────────────────────────────────────
scenarios = create_scenarios(actuator_alpha_lo=0.0, actuator_alpha_hi=0.5)
x0_ivl    = irx.icentpert(jnp.array([0.1, 0.1, 0.0]), jnp.array([0.1, 0.1, 0.1]))
dt        = 0.5
num_steps = 10



# ── Optimise ─────────────────────────────────────────────────────────────────
print("Optimising separating input …")
u_opt, loss_opt, _, _ = optimize_refined_gpu(
    x0_ivl=x0_ivl, scenarios=scenarios, dt=dt,
    num_restarts=500, learning_rate=1, num_iters=50, seed=42, num_steps=num_steps,
)
print(f"  loss = {loss_opt:.6f},  u_opt shape = {u_opt.shape}")

# ── Refinement history ────────────────────────────────────────────────────────
def _intersect_obs(a, b):
    lo = jnp.maximum(a.lower, b.lower)
    hi = jnp.minimum(a.upper, b.upper)
    return irx.Interval(lower=lo, upper=hi) if bool(jnp.all(hi >= lo)) else None


def collect_refinement_history(x0_ivl, u_seq, scenarios, dt, num_steps):
    n     = len(scenarios)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]

    x1   = [euler_step(s.emb_system, x0_ivl, u_seq[0], s.p_interval, dt) for s in scenarios]
    obs1 = [_obs_interval(xi, s) for xi, s in zip(x1, scenarios)]
    steps = [{'t': dt, 'pair_obs': [
        (obs1[i], obs1[j], _intersect_obs(obs1[i], obs1[j])) for i, j in pairs
    ]}]
    pair_states = [(x1[i], x1[j]) for i, j in pairs]

    for k in range(num_steps):
        new_states, new_pair_obs = [], []
        for (i, j), (xi, xj) in zip(pairs, pair_states):
            oi = _obs_interval(xi, scenarios[i])
            oj = _obs_interval(xj, scenarios[j])
            y_lo = jnp.maximum(oi.lower, oj.lower)
            y_hi = jnp.minimum(oi.upper, oj.upper)
            hov   = bool(jnp.all(y_hi >= y_lo))
            fb    = (xi.lower[:2] + xi.upper[:2]) / 2
            ys_lo = jnp.where(hov, y_lo, fb)
            ys_hi = jnp.where(hov, y_hi, fb)

            xi_ref = irx.Interval(
                lower=jnp.array([ys_lo[0] - scenarios[i].obs_offset[0],
                                  ys_lo[1] - scenarios[i].obs_offset[1], xi.lower[2]]),
                upper=jnp.array([ys_hi[0] - scenarios[i].obs_offset[0],
                                  ys_hi[1] - scenarios[i].obs_offset[1], xi.upper[2]]),
            )
            xj_ref = irx.Interval(
                lower=jnp.array([ys_lo[0] - scenarios[j].obs_offset[0],
                                  ys_lo[1] - scenarios[j].obs_offset[1], xj.lower[2]]),
                upper=jnp.array([ys_hi[0] - scenarios[j].obs_offset[0],
                                  ys_hi[1] - scenarios[j].obs_offset[1], xj.upper[2]]),
            )
            xn_i = euler_step(scenarios[i].emb_system, xi_ref, u_seq[k+1], scenarios[i].p_interval, dt)
            xn_j = euler_step(scenarios[j].emb_system, xj_ref, u_seq[k+1], scenarios[j].p_interval, dt)
            on_i = _obs_interval(xn_i, scenarios[i])
            on_j = _obs_interval(xn_j, scenarios[j])
            new_pair_obs.append((on_i, on_j, _intersect_obs(on_i, on_j)))
            new_states.append((xn_i, xn_j))

        steps.append({'t': (k + 2) * dt, 'pair_obs': new_pair_obs})
        pair_states = new_states

    return steps, pairs


print("Collecting refinement history …")
ref_steps, ref_pairs = collect_refinement_history(x0_ivl, u_opt, scenarios, dt, num_steps)
print(f"  {len(ref_steps)} steps, {len(ref_pairs)} pairs")

# helper to gather output observation intervals over time for a scenario
# (used for unrefined plot and diagnostics)
def collect_output_history(x0_ivl, u_seq, scenario, dt, num_steps):
    history = []
    x = x0_ivl
    # record initial observation then propagate sequentially
    for k in range(num_steps + 1):
        history.append(_obs_interval(x, scenario))
        if k < num_steps:
            x = euler_step(scenario.emb_system, x, u_seq[k], scenario.p_interval, dt)
    return history

# configure computer modern fonts for publication style
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Computer Modern'],
    'text.usetex': False,  # assume not using full LaTeX, cm should still work
})

cmap = plt.colormaps['tab10'].resampled(len(scenarios))

# ── Plot refined vs unrefined intervals ───────────────────────────────────────
# create one combined figure containing all pairs side-by-side (two subplots per
# pair: refined then unrefined). this yields a row of 2*n_pairs axes.
statement = (
    'Refinement noticeably tightens the overlap region between outputs, '
    'demonstrating that the interval refinement strategy effectively reduces uncertainty.'
)

# build histories for unrefined plot
output_histories = [
    collect_output_history(x0_ivl, u_opt, s, dt, num_steps)
    for s in scenarios
]
step_indices = np.arange(0, num_steps+1, 1)
times = [k * dt for k in step_indices]

# now create four side-by-side panels
fig, axs = plt.subplots(1, 4, figsize=(20, 6),
                        subplot_kw={'projection': '3d'},
                        gridspec_kw={'wspace': 0.02, 'hspace': 0})

all_px, all_py = [], []

# panel 0: unrefined intervals for all scenarios, with pairwise overlap boxes
ax0 = axs[0]
for idx, t in zip(step_indices, times):
    # draw coloured rectangles for each scenario at this time step
    for k_sc, (hist, s) in enumerate(zip(output_histories, scenarios)):
        color = cmap(k_sc)
        iv = hist[idx]
        xl, xh = float(iv.lower[0]), float(iv.upper[0])
        yl, yh = float(iv.lower[1]), float(iv.upper[1])
        all_px += [xl, xh]
        all_py += [yl, yh]
        verts = [[(xl, t, yl), (xh, t, yl), (xh, t, yh), (xl, t, yh)]]
        poly = Poly3DCollection(verts, alpha=0.35)
        poly.set_facecolor(color)
        poly.set_edgecolor(color)
        ax0.add_collection3d(poly)
    # compute overlaps between every pair of scenarios and draw black outline if nonempty
    intervals = [hist[idx] for hist in output_histories]
    for (p, q) in [(0,1), (0,2), (1,2)]:
        iv1 = intervals[p]
        iv2 = intervals[q]
        xl = float(max(iv1.lower[0], iv2.lower[0]))
        xh = float(min(iv1.upper[0], iv2.upper[0]))
        yl = float(max(iv1.lower[1], iv2.lower[1]))
        yh = float(min(iv1.upper[1], iv2.upper[1]))
        if xh >= xl and yh >= yl:
            corners_x = [xl, xh, xh, xl, xl]
            corners_y = [t, t, t, t, t]
            corners_z = [yl, yl, yh, yh, yl]
            ax0.plot(corners_x, corners_y, corners_z, color='black', linewidth=2.0, zorder=5)

# panels 1-3: refined for each pair
for pair_idx, (i, j) in enumerate(ref_pairs):
    ax_ref = axs[pair_idx + 1]
    for step in ref_steps:
        t = step['t']
        obs_i, obs_j, inter = step['pair_obs'][pair_idx]
        # draw coloured obs blocks
        for obs, color in [(obs_i, cmap(i)), (obs_j, cmap(j))]:
            xl, xh = float(obs.lower[0]), float(obs.upper[0])
            yl, yh = float(obs.lower[1]), float(obs.upper[1])
            all_px += [xl, xh]
            all_py += [yl, yh]
            verts = [[(xl, t, yl), (xh, t, yl), (xh, t, yh), (xl, t, yh)]]
            poly = Poly3DCollection(verts, alpha=0.35)
            poly.set_facecolor(color)
            poly.set_edgecolor(color)
            ax_ref.add_collection3d(poly)
        if inter is not None:
            xl, xh = float(inter.lower[0]), float(inter.upper[0])
            yl, yh = float(inter.lower[1]), float(inter.upper[1])
            corners_x = [xl, xh, xh, xl, xl]
            corners_y = [t, t, t, t, t]
            corners_z = [yl, yl, yh, yh, yl]
            ax_ref.plot(corners_x, corners_y, corners_z, color='black', linewidth=2.0, zorder=5)

# set limits and labels
pad = 0.05
t_vals = [s['t'] for s in ref_steps]
for ax in axs:
    ax.set_xlim(min(all_px) - pad, max(all_px) + pad)
    ax.set_ylim(t_vals[0], t_vals[-1])
    ax.set_zlim(min(all_py) - pad, max(all_py) + pad)
    ax.set_xlabel('$p_x$ (m)', fontsize=10, labelpad=8)
    ax.set_ylabel('Time (s)', fontsize=11, labelpad=8)
    ax.set_zlabel('$p_y$ (m)', fontsize=10, labelpad=8)
    ax.view_init(elev=15, azim=20, roll=0)

# legend entries: use scenarios colors plus intersection
lines = []
lines.append(axs[0].plot([], [], [], color=cmap(0), linewidth=3, label=scenarios[0].name)[0])
lines.append(axs[0].plot([], [], [], color=cmap(1), linewidth=3, label=scenarios[1].name)[0])
lines.append(axs[0].plot([], [], [], color=cmap(2), linewidth=3, label=scenarios[2].name)[0])
lines.append(axs[0].plot([], [], [], color='black', linewidth=2, label='Intersection')[0])
fig.legend(handles=lines, bbox_to_anchor=(0.50, 0.80), loc='center', borderaxespad=0., ncol=4, fontsize=10)

labels = ['(A) Original output-reachable intervals after 5s',
          '(B) Nominal and Actuator Fault \n Interval Overlaps After Refinement',
          '(C) Nominal and Sensor Fault \n Interval Overlaps After Refinement',
          '(D) Actuator and Sensor Fault \n Interval Overlaps After Refinement']
for idx, label in enumerate(labels):
    x = (idx + 0.6) / 5 + 0.1
    fig.text(x, 0.72, label, ha='center', fontsize=11)

# fig.suptitle(statement, fontsize=12, fontweight='bold')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

fname = HERE / f'four_panel_refinement_dt{dt}.pdf'
plt.savefig(fname, bbox_inches='tight')
plt.close(fig)
print(f"  Saved {fname}")

# ── Log(overlap area + 1) over time ───────────────────────────────────────────
print("Computing log(overlap area + 1) over time …")

n_sc = len(scenarios)
all_pairs_car = [(i, j) for i in range(n_sc) for j in range(i + 1, n_sc)]
times_all = [k * dt for k in range(num_steps + 1)]

def _output_overlap_area(hist_a, hist_b, t_idx):
    ia, ib = hist_a[t_idx], hist_b[t_idx]
    dx = max(0.0, float(min(ia.upper[0], ib.upper[0])) - float(max(ia.lower[0], ib.lower[0])))
    dy = max(0.0, float(min(ia.upper[1], ib.upper[1])) - float(max(ia.lower[1], ib.lower[1])))
    return dx * dy

log1p_mat = np.zeros((len(all_pairs_car), num_steps + 1))
for k, (i, j) in enumerate(all_pairs_car):
    for t_idx in range(num_steps + 1):
        log1p_mat[k, t_idx] = np.log1p(_output_overlap_area(output_histories[i], output_histories[j], t_idx))

pair_colors = [plt.colormaps['tab10'](k) for k in range(len(all_pairs_car))]
fig_log, ax_log = plt.subplots(figsize=(10, 4))
for k, (i, j) in enumerate(all_pairs_car):
    label = f"{scenarios[i].name}–{scenarios[j].name}"
    ax_log.plot(times_all, log1p_mat[k], color=pair_colors[k], linewidth=1.5, label=label)
ax_log.set_xlabel('Time (s)', fontsize=10)
ax_log.set_ylabel('log(overlap area + 1)', fontsize=10)
ax_log.set_title('Output overlap area — log(area + 1) over time  (unrefined)', fontsize=10, fontweight='bold')
ax_log.legend(fontsize=9)
ax_log.grid(True, alpha=0.3)
plt.tight_layout()

fname_log = HERE / f'car_overlap_log1p_dt{dt}.pdf'
plt.savefig(fname_log, bbox_inches='tight')
plt.close(fig_log)
print(f"  Saved {fname_log}")

print("Done.")

# diagnostics: compute output intervals at t=5s and their overlap
idx5 = int(5 / dt)
nom_iv = output_histories[0][idx5]
act_iv = output_histories[1][idx5]
print("Nominal output interval at t=5s:", nom_iv)
print("Actuator output interval at t=5s:", act_iv)
lo = jnp.maximum(nom_iv.lower, act_iv.lower)
hi = jnp.minimum(nom_iv.upper, act_iv.upper)
overlap = irx.Interval(lower=lo, upper=hi) if bool(jnp.all(hi >= lo)) else None
print("Overlap interval at t=5s:", overlap)

# if there is an overlap, roll out trajectories from its centre
if overlap is not None:
    cx = float((overlap.lower[0] + overlap.upper[0]) / 2)
    cy = float((overlap.lower[1] + overlap.upper[1]) / 2)
    # orientation from final state intervals (same for both scenarios)
    th_nom = float((final_nom.lower[2] + final_nom.upper[2]) / 2)
    th_act = float((final_act.lower[2] + final_act.upper[2]) / 2)

    steps = num_steps
    traj_nom = []
    traj_act = []
    state_nom = jnp.array([cx, cy, th_nom])
    state_act = jnp.array([cx, cy, th_act])
    for k in range(steps):
        traj_nom.append(state_nom[:2])
        traj_act.append(state_act[:2])
        state_nom = euler_step(scenarios[0].emb_system, state_nom, u_opt, scenarios[0].p_interval, dt)
        state_act = euler_step(scenarios[1].emb_system, state_act, u_opt, scenarios[1].p_interval, dt)
    traj_nom = np.stack(traj_nom)
    traj_act = np.stack(traj_act)

    fig, ax = plt.subplots()
    ax.plot(traj_nom[:,0], traj_nom[:,1], '-o', label='Nominal start from overlap centre')
    ax.plot(traj_act[:,0], traj_act[:,1], '-s', label='Actuator start from overlap centre')
    ax.set_xlabel('$p_x$ (m)')
    ax.set_ylabel('$p_y$ (m)')
    ax.set_title('Forward roll‑out trajectories from overlapping output centre')
    ax.legend()
    ax.grid(True)
    plt.gca().set_aspect('equal', adjustable='datalim')
    plt.show()
else:
    print("No overlap available; cannot roll out.")

# diagnostics: output intervals and