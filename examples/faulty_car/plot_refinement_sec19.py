"""
plot_refinement_sec19.py
========================
Visualises the interval refinement process from Section 19
(Tracking Output-Feedback Separating Controller).

For each of the three scenario pairs the figure shows how the
output-space intersection evolves step-by-step under the optimised
controller, and juxtaposes the *full* interval propagation (what the
simulator actually sees) with the *refined* propagation (what the loss
function optimises).

Run from:  nhholyap/examples/faulty_car/
with the immrax-venv Python:
  /home/user/immrax-venv/bin/python3 plot_refinement_sec19.py
"""

import sys, numpy as np, jax.numpy as jnp
sys.path.insert(0, '.')

import immrax as irx
from faulty_car_output_feedback_cbf import create_track_cl_scenarios, cl_euler_step
from faulty_car_separating_input import _obs_interval

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path

# ── Load controller ───────────────────────────────────────────────────────────
_FAULT_DIR = Path('../../../robotarium_python_simulator/rps/examples/fault_diagnosis')
npz = np.load(_FAULT_DIR / 'theta_track_opt.npz')
theta_seq_opt = jnp.array(npz['theta_seq'])
y_hat_seq     = jnp.array(npz['y_hat_seq'])
dt_cl         = float(npz['dt'])
x0_center     = npz['x0_center']
x0_pert       = npz['x0_pert']
num_steps     = int(theta_seq_opt.shape[0])

cl_scenarios = create_track_cl_scenarios(actuator_alpha_lo=0.0, actuator_alpha_hi=0.5)
x0_ivl       = irx.icentpert(jnp.array(x0_center), jnp.array(x0_pert))

COLORS = {
    'Nominal':        '#2196F3',   # blue
    'Actuator Fault': '#FF9800',   # orange
    'Sensor Fault':   '#4CAF50',   # green
}
PAIR_COLORS = [
    ('#2196F3', '#FF9800', 'purple'),   # Nom vs Act
    ('#2196F3', '#4CAF50', '#C62828'),  # Nom vs SF
    ('#FF9800', '#4CAF50', '#6A1B9A'),  # Act vs SF
]

def _full_theta(k):
    return jnp.concatenate([theta_seq_opt[k, :4], y_hat_seq[k], theta_seq_opt[k, 4:6]])

# ── Step 1: Propagate full intervals for every scenario ───────────────────────
# full_ivl[name][k] = (state_lo, state_hi, out_lo, out_hi)   (all np arrays)
full_ivl = {s.name: [] for s in cl_scenarios}
for s in cl_scenarios:
    x = x0_ivl
    for k in range(num_steps):
        x   = cl_euler_step(s.emb_system, x, _full_theta(k), s.p_interval, dt_cl)
        obs = _obs_interval(x, s)
        full_ivl[s.name].append(dict(
            x_lo  = np.array(x.lower),
            x_hi  = np.array(x.upper),
            o_lo  = np.array(obs.lower[:2]),
            o_hi  = np.array(obs.upper[:2]),
        ))

# ── Step 2: Refined propagation for each pair ─────────────────────────────────
# At each step k we:
#   1. Take the full state intervals of i and j.
#   2. Find the output-space intersection.
#   3. Map the intersection back to state-space ("refine").
#   4. Propagate the refined intervals one more step.
#   5. Record the output intervals of the propagated refined intervals.

pairs = [(0, 1), (0, 2), (1, 2)]    # indices into cl_scenarios

def _map_output_to_state(y_lo, y_hi, x_lo_full, x_hi_full, s):
    """Map output intersection back to state-space (x-space) for scenario s.
    y = scale * [px, py] + offset  →  [px, py] = (y - offset) / scale
    The heading component is kept from the full interval.
    """
    scale  = float(np.array(s.obs_scale)[0])
    offset = np.array(s.obs_offset)
    px_lo = (y_lo[0] - offset[0]) / scale
    px_hi = (y_hi[0] - offset[0]) / scale
    py_lo = (y_lo[1] - offset[1]) / scale
    py_hi = (y_hi[1] - offset[1]) / scale
    return np.array([px_lo, py_lo, x_lo_full[2]]), np.array([px_hi, py_hi, x_hi_full[2]])

# refined_ivl[pair_idx][k] = dict with:
#   has_overlap  bool
#   isect_lo/hi  (2,)  — output intersection at step k
#   ri_o_lo/hi   (2,)  — output interval of propagated refined i at step k+1
#   rj_o_lo/hi   (2,)  — output interval of propagated refined j at step k+1
#   prop_overlap (2,) overlap size of propagated refined outputs (area scalar)
refined_ivl = {pi: [] for pi in range(len(pairs))}

for pi, (ia, ib) in enumerate(pairs):
    sa, sb = cl_scenarios[ia], cl_scenarios[ib]
    for k in range(num_steps):
        da, db = full_ivl[sa.name][k], full_ivl[sb.name][k]
        o_lo_a, o_hi_a = da['o_lo'], da['o_hi']
        o_lo_b, o_hi_b = db['o_lo'], db['o_hi']

        # Intersection in output space
        isect_lo = np.maximum(o_lo_a, o_lo_b)
        isect_hi = np.minimum(o_hi_a, o_hi_b)
        has_overlap = bool(np.all(isect_hi >= isect_lo))

        if has_overlap and k < num_steps - 1:
            # Map intersection back to state space
            xi_lo, xi_hi = _map_output_to_state(
                isect_lo, isect_hi, da['x_lo'], da['x_hi'], sa)
            xj_lo, xj_hi = _map_output_to_state(
                isect_lo, isect_hi, db['x_lo'], db['x_hi'], sb)

            xi_ivl = irx.Interval(lower=jnp.array(xi_lo), upper=jnp.array(xi_hi))
            xj_ivl = irx.Interval(lower=jnp.array(xj_lo), upper=jnp.array(xj_hi))

            # Propagate refined intervals one more step with theta_{k+1}
            k_next = k + 1
            xi_next = cl_euler_step(sa.emb_system, xi_ivl, _full_theta(k_next), sa.p_interval, dt_cl)
            xj_next = cl_euler_step(sb.emb_system, xj_ivl, _full_theta(k_next), sb.p_interval, dt_cl)

            obs_i_next = _obs_interval(xi_next, sa)
            obs_j_next = _obs_interval(xj_next, sb)

            ri_o_lo = np.array(obs_i_next.lower[:2])
            ri_o_hi = np.array(obs_i_next.upper[:2])
            rj_o_lo = np.array(obs_j_next.lower[:2])
            rj_o_hi = np.array(obs_j_next.upper[:2])

            # Overlap area of propagated refined intervals
            prop_ov_lo = np.maximum(ri_o_lo, rj_o_lo)
            prop_ov_hi = np.minimum(ri_o_hi, rj_o_hi)
            prop_area  = float(np.prod(np.maximum(prop_ov_hi - prop_ov_lo, 0)))
        else:
            xi_lo = xi_hi = xj_lo = xj_hi = None
            ri_o_lo = ri_o_hi = rj_o_lo = rj_o_hi = None
            prop_area = None

        refined_ivl[pi].append(dict(
            has_overlap=has_overlap,
            isect_lo=isect_lo,
            isect_hi=isect_hi,
            xi_state=(xi_lo, xi_hi) if has_overlap else None,
            xj_state=(xj_lo, xj_hi) if has_overlap else None,
            ri_o_lo=ri_o_lo, ri_o_hi=ri_o_hi,
            rj_o_lo=rj_o_lo, rj_o_hi=rj_o_hi,
            prop_area=prop_area,
        ))

# ── Helpers ───────────────────────────────────────────────────────────────────

def rect(ax, lo, hi, **kw):
    """Draw a rectangle patch from lo/hi (2,) vectors."""
    w, h = hi[0] - lo[0], hi[1] - lo[1]
    ax.add_patch(mpatches.Rectangle((lo[0], lo[1]), max(w, 1e-4), max(h, 1e-4), **kw))

def square_view(ax, margin=0.05):
    ax.autoscale_view()
    xl, xh = ax.get_xlim()
    yl, yh = ax.get_ylim()
    cx, cy = (xl + xh) / 2, (yl + yh) / 2
    half   = max(xh - xl, yh - yl) / 2 + margin
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)
    ax.set_aspect('equal')

# ── Build PDF ─────────────────────────────────────────────────────────────────
OUT_PDF = Path('refinement_sec19.pdf')

pair_names = [
    ('Nominal', 'Actuator Fault'),
    ('Nominal', 'Sensor Fault'),
    ('Actuator Fault', 'Sensor Fault'),
]

with PdfPages(str(OUT_PDF)) as pdf:

    # ── Cover page ────────────────────────────────────────────────────────────
    fig_cover, ax_cover = plt.subplots(figsize=(8.5, 11))
    ax_cover.axis('off')
    ax_cover.text(0.5, 0.72, 'Section 19 — Tracking Output-Feedback',
                  ha='center', va='center', fontsize=18, fontweight='bold',
                  transform=ax_cover.transAxes)
    ax_cover.text(0.5, 0.64, 'Interval Refinement Process',
                  ha='center', va='center', fontsize=16,
                  transform=ax_cover.transAxes)

    desc = (
        "For each of the three scenario pairs this document shows:\n\n"
        "  Page A  –  Full output-interval propagation\n"
        "             (what the simulator's AFD algorithm uses)\n\n"
        "  Page B  –  Refined propagation\n"
        "             (what loss = 0 actually means)\n\n"
        "Notation\n"
        "────────\n"
        "  Filled box   = full output interval  [o_lo, o_hi]\n"
        "  Hatched box  = intersection of the two scenarios' output intervals\n"
        "  Dashed box   = propagated refined interval (one step forward)\n\n"
        "Bug 1: 'loss = 0' optimises the refined sequence (Page B),\n"
        "not the full sequence (Page A). The full intervals overlap\n"
        "at every step, so the simulator's AFD cannot eliminate modes."
    )
    ax_cover.text(0.5, 0.40, desc, ha='center', va='center', fontsize=10.5,
                  transform=ax_cover.transAxes, family='monospace',
                  bbox=dict(boxstyle='round', facecolor='#f5f5f5', alpha=0.8),
                  verticalalignment='center')

    # Legend
    legend_items = [
        mpatches.Patch(fc=COLORS['Nominal'],        ec='black', alpha=0.45, label='Nominal'),
        mpatches.Patch(fc=COLORS['Actuator Fault'], ec='black', alpha=0.45, label='Actuator Fault'),
        mpatches.Patch(fc=COLORS['Sensor Fault'],   ec='black', alpha=0.45, label='Sensor Fault'),
        mpatches.Patch(fc='#9C27B0', ec='black', alpha=0.55, hatch='xxx', label='Intersection'),
        mpatches.Patch(fc='none', ec='black', linestyle='--', linewidth=1.5,
                       label='Propagated refined interval'),
    ]
    ax_cover.legend(handles=legend_items, loc='lower center', fontsize=9,
                    ncol=3, bbox_to_anchor=(0.5, 0.04),
                    framealpha=0.9, title='Legend')
    plt.tight_layout()
    pdf.savefig(fig_cover, bbox_inches='tight')
    plt.close(fig_cover)

    # ── Per-pair pages ────────────────────────────────────────────────────────
    for pi, (na, nb) in enumerate(pair_names):
        ia, ib = pairs[pi]
        sa, sb = cl_scenarios[ia], cl_scenarios[ib]
        ca, cb, ci_c = PAIR_COLORS[pi]

        for page_type in ('full', 'refined'):
            fig, axes = plt.subplots(
                2, num_steps,
                figsize=(3.2 * num_steps, 6.5),
                gridspec_kw={'hspace': 0.55, 'wspace': 0.35},
            )
            # Top row = output space   Bottom row = "next-step propagated refined"
            title_top = (
                f'Pair: {na}  vs  {nb}\n'
                f'{"Full output intervals" if page_type=="full" else "Refined (intersection) output intervals"}'
                f'  —  output space (px, py)'
            )
            fig.suptitle(title_top, fontsize=11, fontweight='bold', y=1.00)

            for k in range(num_steps):
                ax_top = axes[0, k]
                ax_bot = axes[1, k]

                da = full_ivl[na][k]
                db = full_ivl[nb][k]
                rd = refined_ivl[pi][k]

                # ── Top subplot: output intervals at step k ───────────────
                if page_type == 'full':
                    # Draw the two full output intervals
                    rect(ax_top, da['o_lo'], da['o_hi'],
                         fc=ca, ec=ca, alpha=0.35, linewidth=1.2, label=na)
                    rect(ax_top, db['o_lo'], db['o_hi'],
                         fc=cb, ec=cb, alpha=0.35, linewidth=1.2, label=nb)

                    # Draw intersection (if any)
                    if rd['has_overlap']:
                        rect(ax_top, rd['isect_lo'], rd['isect_hi'],
                             fc=ci_c, ec=ci_c, alpha=0.55, hatch='xxx',
                             linewidth=0.8, label='Intersection')
                        area = float(np.prod(np.maximum(
                            rd['isect_hi'] - rd['isect_lo'], 0)))
                        ax_top.set_title(f't = {k+1} s\nOverlap area\n{area:.4f} m²',
                                         fontsize=8)
                    else:
                        ax_top.set_title(f't = {k+1} s\nNo overlap\n(SEPARATED)',
                                         fontsize=8, color='green', fontweight='bold')
                else:
                    # Refined view: show only the intersection box (what was refined)
                    if rd['has_overlap']:
                        rect(ax_top, rd['isect_lo'], rd['isect_hi'],
                             fc=ci_c, ec=ci_c, alpha=0.6, hatch='xxx',
                             linewidth=1.0, label='Intersection')
                        area = float(np.prod(np.maximum(
                            rd['isect_hi'] - rd['isect_lo'], 0)))
                        ax_top.set_title(f't = {k+1} s\nRefined region\n{area:.4f} m²',
                                         fontsize=8)
                    else:
                        ax_top.set_title(f't = {k+1} s\nNo intersection\n(already separated)',
                                         fontsize=8, color='green', fontweight='bold')

                ax_top.set_xlabel('px (m)', fontsize=7)
                ax_top.set_ylabel('py (m)', fontsize=7)
                ax_top.tick_params(labelsize=6)
                square_view(ax_top, margin=0.04)

                # ── Bottom subplot: propagated refined intervals at step k+1
                ax_bot.set_xlabel('px (m)', fontsize=7)
                ax_bot.set_ylabel('py (m)', fontsize=7)
                ax_bot.tick_params(labelsize=6)

                if page_type == 'refined' and rd['has_overlap'] and k < num_steps - 1:
                    ri_lo, ri_hi = rd['ri_o_lo'], rd['ri_o_hi']
                    rj_lo, rj_hi = rd['rj_o_lo'], rd['rj_o_hi']

                    rect(ax_bot, ri_lo, ri_hi,
                         fc=ca, ec=ca, alpha=0.35, linewidth=1.2,
                         linestyle='--', label=f'{na} (refined, prop.)')
                    rect(ax_bot, rj_lo, rj_hi,
                         fc=cb, ec=cb, alpha=0.35, linewidth=1.2,
                         linestyle='--', label=f'{nb} (refined, prop.)')

                    # Propagated overlap
                    prop_lo = np.maximum(ri_lo, rj_lo)
                    prop_hi = np.minimum(ri_hi, rj_hi)
                    has_prop = bool(np.all(prop_hi >= prop_lo))
                    if has_prop:
                        rect(ax_bot, prop_lo, prop_hi,
                             fc=ci_c, ec=ci_c, alpha=0.55, hatch='xxx', linewidth=0.8)
                        area = float(rd['prop_area'])
                        ax_bot.set_title(f'Propagated to t={k+2} s\nOverlap area\n{area:.4f} m²',
                                         fontsize=8)
                    else:
                        ax_bot.set_title(f'Propagated to t={k+2} s\nSEPARATED ✓',
                                         fontsize=8, color='green', fontweight='bold')
                    square_view(ax_bot, margin=0.04)

                elif page_type == 'full' and rd['has_overlap'] and k < num_steps - 1:
                    # In the 'full' view, bottom row shows the FULL intervals at the next step
                    k2 = k + 1
                    da2 = full_ivl[na][k2]
                    db2 = full_ivl[nb][k2]
                    rd2 = refined_ivl[pi][k2]

                    rect(ax_bot, da2['o_lo'], da2['o_hi'],
                         fc=ca, ec=ca, alpha=0.35, linewidth=1.2)
                    rect(ax_bot, db2['o_lo'], db2['o_hi'],
                         fc=cb, ec=cb, alpha=0.35, linewidth=1.2)
                    if rd2['has_overlap']:
                        rect(ax_bot, rd2['isect_lo'], rd2['isect_hi'],
                             fc=ci_c, ec=ci_c, alpha=0.55, hatch='xxx', linewidth=0.8)
                        area2 = float(np.prod(np.maximum(rd2['isect_hi'] - rd2['isect_lo'], 0)))
                        ax_bot.set_title(f'Full ivl at t={k+2} s\nOverlap area\n{area2:.4f} m²',
                                         fontsize=8)
                    else:
                        ax_bot.set_title(f'Full ivl at t={k+2} s\nNo overlap',
                                         fontsize=8, color='green', fontweight='bold')
                    square_view(ax_bot, margin=0.04)
                else:
                    ax_bot.set_title('—', fontsize=8)
                    ax_bot.text(0.5, 0.5, 'N/A', ha='center', va='center',
                                transform=ax_bot.transAxes, fontsize=9, color='grey')
                    for spine in ax_bot.spines.values():
                        spine.set_visible(False)
                    ax_bot.set_xticks([])
                    ax_bot.set_yticks([])

            plt.tight_layout(rect=[0, 0, 1, 0.97])
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

    # ── Summary page: overlap areas over time for all pairs ───────────────────
    fig_sum, axes_sum = plt.subplots(1, 3, figsize=(13, 4.5), sharey=False)
    fig_sum.suptitle(
        'Overlap Area vs Time — Full intervals (solid) vs Refined propagation (dashed)\n'
        '"loss = 0" optimises the dashed lines; AFD in the simulator uses the solid lines',
        fontsize=10, fontweight='bold',
    )
    steps_t = np.arange(1, num_steps + 1)

    for pi, (na, nb) in enumerate(pair_names):
        ax = axes_sum[pi]
        full_areas    = []
        refined_areas = []
        for k in range(num_steps):
            rd = refined_ivl[pi][k]
            da = full_ivl[na][k]
            db = full_ivl[nb][k]

            # Full overlap area
            f_lo = np.maximum(da['o_lo'], db['o_lo'])
            f_hi = np.minimum(da['o_hi'], db['o_hi'])
            full_areas.append(float(np.prod(np.maximum(f_hi - f_lo, 0))))

            # Refined propagated overlap area (overlap of propagated refined intervals)
            if rd['has_overlap'] and rd['prop_area'] is not None:
                refined_areas.append(rd['prop_area'])
            else:
                refined_areas.append(0.0)

        ax.plot(steps_t, full_areas, 'o-', lw=2, color=PAIR_COLORS[pi][2],
                label='Full interval overlap')
        ax.plot(steps_t, refined_areas, 's--', lw=1.5, color='grey',
                label='Refined propagated overlap')
        ax.axhline(0, color='black', lw=0.8, ls=':')
        ax.set_title(f'{na}\nvs {nb}', fontsize=9, fontweight='bold')
        ax.set_xlabel('t (s)', fontsize=8)
        ax.set_ylabel('Overlap area (m²)', fontsize=8)
        ax.legend(fontsize=7)
        ax.tick_params(labelsize=7)
        ax.set_xticks(steps_t)

    plt.tight_layout()
    pdf.savefig(fig_sum, bbox_inches='tight')
    plt.close(fig_sum)

print(f'PDF saved → {OUT_PDF.resolve()}')
