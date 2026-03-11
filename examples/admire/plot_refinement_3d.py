"""
ADMIRE separating-input 3-D refinement plot.
Mirrors examples/faulty_car/plot_refinement_3d.py for the ADMIRE aircraft.

10 scenarios: 1 nominal + 9 actuator faults.
Output = state[3:6] = [pb, qb, rb]  (no sensor offset — output IS state).

u_opt is loaded from admire_u_opt.npz produced by admire_separating_input_demo.ipynb.
No JAX tracing/compilation is done: only eager euler_step calls are used.

Saves: admire_refinement_3d.pdf
"""
import os; os.environ['JAX_PLATFORMS'] = 'cpu'
import sys
from pathlib import Path

import jax.numpy as jnp
import immrax as irx
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

HERE     = Path(__file__).resolve().parent
EXAMPLES = HERE.parent
for p in (str(HERE), str(EXAMPLES)):
    if p not in sys.path:
        sys.path.insert(0, p)

from admire_separating_input import (
    create_scenarios,
    euler_step,
    output_interval,    # x[3:6] = (pb, qb, rb)
)

# ── Load pre-optimised result ──────────────────────────────────────────────────
_result_file = HERE / 'admire_u_opt.npz'
if not _result_file.exists():
    raise FileNotFoundError(
        f"{_result_file} not found.\n"
        "Run the save cell in admire_separating_input_demo.ipynb first."
    )

_data = np.load(_result_file)

u_opt_np          = _data['u_opt']                      # (num_steps, 10)
dt                = float(_data['dt'])
num_steps         = int(_data['num_steps'])
fault_effectiveness = float(_data['fault_effectiveness'])
num_scenarios     = int(_data['num_scenarios'])
x0_ivl            = irx.Interval(
    lower=jnp.array(_data['x0_ivl_lower']),
    upper=jnp.array(_data['x0_ivl_upper']),
)

u_seq = jnp.array(u_opt_np)   # (num_steps, 10)

print(f"Loaded {_result_file.name}")
print(f"  u_seq shape : {u_seq.shape}")
print(f"  dt          : {dt} s,  num_steps: {num_steps},  horizon: {num_steps * dt:.2f} s")
print(f"  loss_opt    : {float(_data['loss_opt']):.6f} rad²")

# ── Setup ──────────────────────────────────────────────────────────────────────
scenarios = create_scenarios(fault_effectiveness=fault_effectiveness)[:num_scenarios]
print(f"{len(scenarios)} scenarios: " + ", ".join(s.name for s in scenarios))

times = [k * dt for k in range(num_steps + 1)]

# ── Helpers ────────────────────────────────────────────────────────────────────
def _obs(x_ivl):
    """Output = state[3:6] = (pb, qb, rb); no offset."""
    return output_interval(x_ivl)

def _intersect(a, b):
    lo = jnp.maximum(a.lower, b.lower)
    hi = jnp.minimum(a.upper, b.upper)
    return irx.Interval(lower=lo, upper=hi) if bool(jnp.all(hi >= lo)) else None

# ── Collect unrefined histories ────────────────────────────────────────────────
def collect_output_history(x0_ivl, u_seq, scenario):
    """Returns list of 3-D output intervals (pb,qb,rb) — used for 3D plots."""
    history = [_obs(x0_ivl)]
    x = x0_ivl
    for k in range(num_steps):
        x = euler_step(scenario.emb_system, x, u_seq[k], scenario.p_interval, dt)
        history.append(_obs(x))
    return history

def collect_state_history(x0_ivl, u_seq, scenario):
    """Returns list of full 9-D state intervals — used for volumetric overlap."""
    history = [x0_ivl]
    x = x0_ivl
    for k in range(num_steps):
        x = euler_step(scenario.emb_system, x, u_seq[k], scenario.p_interval, dt)
        history.append(x)
    return history

print("Collecting unrefined output histories …")
output_histories = []
state_histories  = []
for k, s in enumerate(scenarios):
    print(f"  scenario {k+1}/{len(scenarios)}: {s.name}", flush=True)
    output_histories.append(collect_output_history(x0_ivl, u_seq, s))
    state_histories.append(collect_state_history(x0_ivl, u_seq, s))

# ── Collect refinement history (nominal vs each fault) ────────────────────────
def collect_refinement_history(x0_ivl, u_seq, scenarios):
    """Pairs = [(0, j) for j=1..9]: nominal vs each fault.
    u_seq: (num_steps, 10) — u_seq[0] for step 1, u_seq[k+1] for step k+2.
    """
    pairs = [(0, j) for j in range(1, len(scenarios))]

    x1   = [euler_step(s.emb_system, x0_ivl, u_seq[0], s.p_interval, dt) for s in scenarios]
    obs1 = [_obs(xi) for xi in x1]
    steps = [{'t': dt, 'pair_obs': [
        (obs1[i], obs1[j], _intersect(obs1[i], obs1[j])) for i, j in pairs
    ]}]
    pair_states = [(x1[i], x1[j]) for i, j in pairs]

    for k in range(num_steps - 1):
        u_k = u_seq[k + 1]   # control applied at this refinement step
        new_states, new_pair_obs = [], []
        for (i, j), (xi, xj) in zip(pairs, pair_states):
            oi = _obs(xi)
            oj = _obs(xj)
            y_lo = jnp.maximum(oi.lower, oj.lower)
            y_hi = jnp.minimum(oi.upper, oj.upper)
            hov   = bool(jnp.all(y_hi >= y_lo))
            fb    = (xi.lower[3:6] + xi.upper[3:6]) / 2
            ys_lo = jnp.where(hov, y_lo, fb)
            ys_hi = jnp.where(hov, y_hi, fb)

            xi_ref = irx.Interval(
                lower=xi.lower.at[3:6].set(ys_lo),
                upper=xi.upper.at[3:6].set(ys_hi),
            )
            xj_ref = irx.Interval(
                lower=xj.lower.at[3:6].set(ys_lo),
                upper=xj.upper.at[3:6].set(ys_hi),
            )

            xn_i = euler_step(scenarios[i].emb_system, xi_ref, u_k, scenarios[i].p_interval, dt)
            xn_j = euler_step(scenarios[j].emb_system, xj_ref, u_k, scenarios[j].p_interval, dt)
            on_i = _obs(xn_i)
            on_j = _obs(xn_j)
            new_pair_obs.append((on_i, on_j, _intersect(on_i, on_j)))
            new_states.append((xn_i, xn_j))

        steps.append({'t': (k + 2) * dt, 'pair_obs': new_pair_obs})
        pair_states = new_states

    return steps, pairs

print("Collecting refinement histories …")
ref_steps, ref_pairs = collect_refinement_history(x0_ivl, u_seq, scenarios)
print(f"  {len(ref_steps)} steps, {len(ref_pairs)} pairs")

# ── Plotting helpers ──────────────────────────────────────────────────────────
plt.rcParams.update({'font.family': 'serif'})
cmap = plt.colormaps['tab10'].resampled(len(scenarios))

def _draw_rect(ax, obs, t, color, alpha=0.35):
    """Draw floating rectangle: x=pb (output[0]), y=t, z=qb (output[1])."""
    xl, xh = float(obs.lower[0]), float(obs.upper[0])
    yl, yh = float(obs.lower[1]), float(obs.upper[1])
    verts = [[(xl, t, yl), (xh, t, yl), (xh, t, yh), (xl, t, yh)]]
    poly  = Poly3DCollection(verts, alpha=alpha)
    poly.set_facecolor(color)
    poly.set_edgecolor(color)
    ax.add_collection3d(poly)
    return xl, xh, yl, yh

def _draw_outline(ax, obs, t):
    xl, xh = float(obs.lower[0]), float(obs.upper[0])
    yl, yh = float(obs.lower[1]), float(obs.upper[1])
    ax.plot([xl, xh, xh, xl, xl], [t]*5, [yl, yl, yh, yh, yl],
            color='black', linewidth=1.5, zorder=5)

def _set_axes(ax, all_pb, all_qb, xlabel='$p_b$ (rad/s)', zlabel='$q_b$ (rad/s)'):
    pad  = max((max(all_pb) - min(all_pb)) * 0.05, 1e-6)
    padz = max((max(all_qb) - min(all_qb)) * 0.05, 1e-6)
    ax.set_xlim(min(all_pb) - pad, max(all_pb) + pad)
    ax.set_ylim(times[0], times[-1])
    ax.set_zlim(min(all_qb) - padz, max(all_qb) + padz)
    ax.set_xlabel(xlabel,     fontsize=7, labelpad=5)
    ax.set_ylabel('Time (s)', fontsize=7, labelpad=5)
    ax.set_zlabel(zlabel,     fontsize=7, labelpad=5)
    ax.view_init(elev=15, azim=20, roll=0)
    ax.tick_params(labelsize=5)

# ── Dynamic grid: 2 rows × enough cols for 1 unrefined + all refined pairs ─────
n_panels = 1 + len(ref_pairs)                     # total panels needed
n_cols   = (n_panels + 1) // 2                    # ceil(n_panels / 2)
fig, axs = plt.subplots(2, n_cols, figsize=(n_cols * 4.8, 10),
                         subplot_kw={'projection': '3d'},
                         gridspec_kw={'wspace': 0.05, 'hspace': 0.15})
axs_flat = axs.flatten()
# hide any unused trailing panels
for _ax in axs_flat[n_panels:]:
    _ax.set_visible(False)

# Panel 0: unrefined — all 10 scenarios
print("Plotting panel 0 (unrefined) …")
ax0 = axs_flat[0]
all_pb0, all_qb0 = [], []
for k_sc, (hist, s) in enumerate(zip(output_histories, scenarios)):
    color = cmap(k_sc)
    for idx, t in enumerate(times):
        xl, xh, yl, yh = _draw_rect(ax0, hist[idx], t, color)
        all_pb0 += [xl, xh];  all_qb0 += [yl, yh]
# pairwise overlap outlines
for k1 in range(len(scenarios)):
    for k2 in range(k1 + 1, len(scenarios)):
        for idx, t in enumerate(times):
            inter = _intersect(output_histories[k1][idx], output_histories[k2][idx])
            if inter is not None:
                _draw_outline(ax0, inter, t)
_set_axes(ax0, all_pb0, all_qb0)
ax0.set_title('(A) Unrefined — all 10 scenarios', fontsize=8, fontweight='bold')

# Panels 1-9: refined nominal vs fault_j
import string
panel_labels = string.ascii_uppercase[1:]   # 'BC…Z' — enough for any number of pairs
print("Plotting refined pair panels …")
for panel_idx, (pair_row, (i, j)) in enumerate(zip(range(len(ref_pairs)), ref_pairs)):
    ax = axs_flat[panel_idx + 1]
    all_pb, all_qb = [], []
    for step in ref_steps:
        t = step['t']
        obs_i, obs_j, inter = step['pair_obs'][pair_row]
        xl, xh, yl, yh = _draw_rect(ax, obs_i, t, cmap(i));  all_pb += [xl, xh];  all_qb += [yl, yh]
        xl, xh, yl, yh = _draw_rect(ax, obs_j, t, cmap(j));  all_pb += [xl, xh];  all_qb += [yl, yh]
        if inter is not None:
            _draw_outline(ax, inter, t)
    _set_axes(ax, all_pb, all_qb)
    ax.set_title(f'({panel_labels[panel_idx]}) Refined\nNominal vs {scenarios[j].name}',
                 fontsize=7, fontweight='bold')

# Legend
legend_lines = [
    axs_flat[0].plot([], [], [], color=cmap(k), linewidth=2.5, label=s.name)[0]
    for k, s in enumerate(scenarios)
] + [axs_flat[0].plot([], [], [], color='black', linewidth=1.5, label='Intersection')[0]]
fig.legend(handles=legend_lines, loc='lower center', ncol=6, fontsize=7,
           bbox_to_anchor=(0.5, -0.02))

plt.suptitle(
    f'ADMIRE output reachable sets — {len(scenarios)} scenarios  '
    f'({num_steps} × {dt}s = {num_steps*dt:.1f}s)\n'
    '$p_b$ = roll rate,  $q_b$ = pitch rate  |  Black outline = intersection used for refinement',
    fontsize=10, fontweight='bold',
)
plt.tight_layout(rect=[0, 0.06, 1, 0.94])

fname = HERE / 'admire_refinement_3d.pdf'
plt.savefig(fname, bbox_inches='tight')
plt.close(fig)
print(f"Saved {fname}")

# ── All-pairs overlap over time ────────────────────────────────────────────────
print("Computing all-pairs overlap over time …")

n = len(scenarios)
all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
n_all_pairs = len(all_pairs)

def _overlap_vol(ivl_a, ivl_b):
    """Scalar overlap volume of two full 9-D state intervals."""
    lo = np.maximum(np.array(ivl_a.lower), np.array(ivl_b.lower))
    hi = np.minimum(np.array(ivl_a.upper), np.array(ivl_b.upper))
    return float(np.log(np.prod(np.maximum(hi - lo, 0.0))))

# ── Unrefined overlaps: (n_all_pairs, num_steps+1) ────────────────────────────
unref_mat = np.zeros((n_all_pairs, num_steps + 1))
for k, (i, j) in enumerate(all_pairs):
    for t_idx in range(num_steps + 1):
        unref_mat[k, t_idx] = _overlap_vol(state_histories[i][t_idx],
                                            state_histories[j][t_idx])

# ── Refined overlaps for ALL pairs: (n_all_pairs, num_steps) ──────────────────
print("  Running refinement for all pairs …")
ref_mat = np.zeros((n_all_pairs, num_steps))

x1 = [euler_step(s.emb_system, x0_ivl, u_seq[0], s.p_interval, dt) for s in scenarios]
pair_states_all = [(x1[i], x1[j]) for i, j in all_pairs]

for k, ((i, j), (xi, xj)) in enumerate(zip(all_pairs, pair_states_all)):
    ref_mat[k, 0] = _overlap_vol(xi, xj)

for step in range(num_steps - 1):
    u_k = u_seq[step + 1]
    new_pair_states_all = []
    for k, ((i, j), (xi, xj)) in enumerate(zip(all_pairs, pair_states_all)):
        # Refinement still uses observer output (state[3:6]) for intersection
        oi, oj = _obs(xi), _obs(xj)
        y_lo = jnp.maximum(oi.lower, oj.lower)
        y_hi = jnp.minimum(oi.upper, oj.upper)
        hov  = bool(jnp.all(y_hi >= y_lo))
        fb   = (xi.lower[3:6] + xi.upper[3:6]) / 2
        ys_lo = jnp.where(hov, y_lo, fb)
        ys_hi = jnp.where(hov, y_hi, fb)

        xi_ref = irx.Interval(lower=xi.lower.at[3:6].set(ys_lo),
                              upper=xi.upper.at[3:6].set(ys_hi))
        xj_ref = irx.Interval(lower=xj.lower.at[3:6].set(ys_lo),
                              upper=xj.upper.at[3:6].set(ys_hi))

        xn_i = euler_step(scenarios[i].emb_system, xi_ref, u_k, scenarios[i].p_interval, dt)
        xn_j = euler_step(scenarios[j].emb_system, xj_ref, u_k, scenarios[j].p_interval, dt)
        ref_mat[k, step + 1] = _overlap_vol(xn_i, xn_j)   # full 9-D state
        new_pair_states_all.append((xn_i, xn_j))
    pair_states_all = new_pair_states_all

print(f"  Done. Matrix shape: unrefined {unref_mat.shape}, refined {ref_mat.shape}")

# ── Heatmap figure ─────────────────────────────────────────────────────────────
pair_labels = [f"{scenarios[i].name[:3]}–{scenarios[j].name[:3]}"
               for i, j in all_pairs]
t_unref = [k * dt for k in range(num_steps + 1)]
t_ref   = [(k + 1) * dt for k in range(num_steps)]

fig2, (ax_u, ax_r) = plt.subplots(1, 2, figsize=(14, max(6, n_all_pairs * 0.22)),
                                    gridspec_kw={'wspace': 0.05})

def _heatmap(ax, mat, t_vals, title):
    vmax = max(mat.max(), 1e-12)
    im = ax.imshow(mat, aspect='auto', origin='upper',
                   extent=[t_vals[0] - dt/2, t_vals[-1] + dt/2,
                            n_all_pairs - 0.5, -0.5],
                   vmin=0, vmax=vmax, cmap='YlOrRd')
    ax.set_yticks(range(n_all_pairs))
    ax.set_yticklabels(pair_labels, fontsize=5)
    ax.set_xlabel('Time (s)', fontsize=9)
    ax.set_title(title, fontsize=9, fontweight='bold')
    # vertical lines at each time step
    for tv in t_vals:
        ax.axvline(tv, color='white', linewidth=0.3, alpha=0.5)
    return im

im_u = _heatmap(ax_u, unref_mat, t_unref, 'Unrefined overlap volume')
im_r = _heatmap(ax_r, ref_mat,   t_ref,   'Refined overlap volume')
ax_r.set_yticklabels([])   # shared y-axis labels on left only

fig2.colorbar(im_u, ax=ax_u, fraction=0.03, pad=0.02, label='Overlap vol. (all 9 states)')
fig2.colorbar(im_r, ax=ax_r, fraction=0.03, pad=0.02, label='Overlap vol. (all 9 states)')

plt.suptitle(
    f'ADMIRE — all {n_all_pairs} scenario-pair volumetric overlaps over time\n'
    f'({num_steps} steps × {dt} s = {num_steps*dt:.1f} s  |'
    r'  state = $[V_t,\alpha,\beta,p_b,q_b,r_b,\psi,\theta,\phi]$)',
    fontsize=10, fontweight='bold',
)
plt.tight_layout()

fname2 = HERE / 'admire_overlap_over_time.pdf'
plt.savefig(fname2, bbox_inches='tight')
plt.close(fig2)
print(f"Saved {fname2}")

# ── Log(overlap volume + 1) heatmap ───────────────────────────────────────────
# Recover raw volumes from the log matrices (entries of -inf/large-neg → vol≈0).
unref_vol_raw = np.exp(np.clip(unref_mat, -500, 500))
ref_vol_raw   = np.exp(np.clip(ref_mat,   -500, 500))
unref_log1p   = np.log1p(unref_vol_raw)
ref_log1p     = np.log1p(ref_vol_raw)

fig3, (ax_u3, ax_r3) = plt.subplots(1, 2, figsize=(14, max(6, n_all_pairs * 0.22)),
                                      gridspec_kw={'wspace': 0.05})

def _heatmap_log1p(ax, mat, t_vals, title):
    vmax = max(mat.max(), 1e-12)
    im = ax.imshow(mat, aspect='auto', origin='upper',
                   extent=[t_vals[0] - dt/2, t_vals[-1] + dt/2,
                            n_all_pairs - 0.5, -0.5],
                   vmin=0, vmax=vmax, cmap='YlOrRd')
    ax.set_yticks(range(n_all_pairs))
    ax.set_yticklabels(pair_labels, fontsize=5)
    ax.set_xlabel('Time (s)', fontsize=9)
    ax.set_title(title, fontsize=9, fontweight='bold')
    for tv in t_vals:
        ax.axvline(tv, color='white', linewidth=0.3, alpha=0.5)
    return im

im_u3 = _heatmap_log1p(ax_u3, unref_log1p, t_unref, 'Unrefined  log(overlap volume + 1)')
im_r3 = _heatmap_log1p(ax_r3, ref_log1p,   t_ref,   'Refined  log(overlap volume + 1)')
ax_r3.set_yticklabels([])

fig3.colorbar(im_u3, ax=ax_u3, fraction=0.03, pad=0.02, label='log(overlap vol. + 1)')
fig3.colorbar(im_r3, ax=ax_r3, fraction=0.03, pad=0.02, label='log(overlap vol. + 1)')

plt.suptitle(
    f'ADMIRE — log(overlap volume + 1) for all {n_all_pairs} scenario pairs over time\n'
    f'({num_steps} steps × {dt} s = {num_steps*dt:.1f} s  |'
    r'  state = $[V_t,\alpha,\beta,p_b,q_b,r_b,\psi,\theta,\phi]$)',
    fontsize=10, fontweight='bold',
)
plt.tight_layout()

fname3 = HERE / 'admire_overlap_log1p.pdf'
plt.savefig(fname3, bbox_inches='tight')
plt.close(fig3)
print(f"Saved {fname3}")
print("Done.")
