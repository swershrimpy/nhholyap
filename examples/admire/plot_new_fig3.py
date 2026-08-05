"""
ADMIRE — 3-D reachable-set plot showing a box at every time step considered.

Same data source and eager-propagation approach as plot_refinement_3d.py
(loads admire_u_opt.npz, uses create_scenarios/euler_step from
admire_separating_input.py), but plotted in the same style as the
fig3_reproduction.pdf figure from admire_staged_opt_minimal.ipynb: a single
3-D axes over the angular-rate dimensions [pb, qb, rb] = state[3:6], one
cuboid per scenario per checkpoint time (t = 0, 1, 2, 3, 4, 5 s), instead of
only the final state (plot_3d_final_intervals) or a smooth tube
(plot_3d_interval_history).

Loads:  admire_u_opt.npz
Saves:  new_fig3.pdf
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
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ["Liberation Serif"]
# plt.rcParams['mathtext.fontset'] = 'cm'
# plt.rcParams['axes.formatter.use_mathtext'] = True
# plt.rcParams['axes.unicode_minus'] = False  # cmr10 has no U+2212 glyph

HERE     = Path(__file__).resolve().parent
EXAMPLES = HERE.parent
for p in (str(HERE), str(EXAMPLES)):
    if p not in sys.path:
        sys.path.insert(0, p)

from admire_separating_input import (
    create_scenarios,
    euler_step,
)

# ── Load pre-optimised result (same as plot_refinement_3d.py) ─────────────────
_result_file = HERE / 'admire_u_opt.npz'
if not _result_file.exists():
    raise FileNotFoundError(
        f"{_result_file} not found.\n"
        "Run the save cell in admire_separating_input_demo.ipynb first."
    )

_data = np.load(_result_file)
u_opt_np             = _data['u_opt']
dt                    = float(_data['dt'])
num_steps             = int(_data['num_steps'])
fault_effectiveness   = float(_data['fault_effectiveness'])
num_scenarios         = int(_data['num_scenarios'])
x0_ivl                = irx.Interval(
    lower=jnp.array(_data['x0_ivl_lower']),
    upper=jnp.array(_data['x0_ivl_upper']),
)
u_seq = jnp.array(u_opt_np)   # (num_steps, 10)

print(f"Loaded {_result_file.name}")
print(f"  u_seq shape : {u_seq.shape}")
print(f"  dt={dt}s  num_steps={num_steps}  horizon={num_steps*dt:.2f}s")

scenarios = create_scenarios(fault_effectiveness=fault_effectiveness)[:num_scenarios]
print(f"{len(scenarios)} scenarios: " + ", ".join(s.name for s in scenarios))

dim_names = ["Velocity (m/s)", "AoA (rad)", "Sideslip Angle (rad)", "roll rate (rad/s)", "pitch rate (rad/s)",
             "yaw rate (rad/s)", "heading angle (rad)", "pitch angle (rad)", "roll angle (rad)"]

# ── Checkpoint times considered: t = 0, 1, 2, 3, 4, 5 s ───────────────────────
checkpoint_times = list(range(int(num_steps * dt)) ) + [int(num_steps * dt)]
checkpoint_steps = [round(t / dt) for t in checkpoint_times]
assert checkpoint_steps[-1] <= num_steps, "checkpoint times exceed the loaded horizon"
print(f"Checkpoint times: {checkpoint_times}  (steps {checkpoint_steps})")

# ── Propagate every scenario, recording the full state interval at each checkpoint ──
print("Propagating scenarios and recording checkpoint states …")
history = {}
for s in scenarios:
    x = x0_ivl
    states_at_checkpoints = []
    next_ckpt_idx = 0
    if checkpoint_steps[0] == 0:
        states_at_checkpoints.append(x)
        next_ckpt_idx = 1
    for step in range(1, num_steps + 1):
        x = euler_step(s.emb_system, x, u_seq[step - 1], s.p_interval, dt)
        if next_ckpt_idx < len(checkpoint_steps) and step == checkpoint_steps[next_ckpt_idx]:
            states_at_checkpoints.append(x)
            next_ckpt_idx += 1
    history[s.name] = {"states": states_at_checkpoints}
    print(f"  {s.name}: {len(states_at_checkpoints)} checkpoint boxes")


# ── Plotting: one cuboid per scenario per checkpoint time ─────────────────────
def plot_3d_boxes_over_time(
    history,
    dim_names,
    times,
    color_map=None,
    save_to_pdf=False,
    filename="new_fig3.pdf",
):
    """
    3-D plot of state-interval boxes for dimensions 4, 5, and 6 (indices 3,4,5),
    drawing one cuboid per scenario at *every* checkpoint time in `times`
    (instead of only the final state or a smooth tube). Earlier boxes are
    drawn more transparently; the final box per scenario is the most solid.
    """
    scenarios = list(history.keys())
    if color_map is None:
        fault_names = [name for name in scenarios if name != "Nominal"]
        cmap = plt.cm.get_cmap('viridis', len(fault_names))
        generated_colors = [cmap(i) for i in range(len(fault_names))]
        color_map = {name: color for name, color in zip(fault_names, generated_colors)}
        if "Nominal" in scenarios: color_map["Nominal"] = "blue"

    print(f"Generating 3D plot with boxes at t={times} for dimensions 4, 5, and 6 (indices 3, 4, 5)...")

    # --- Tight axis limits over ALL checkpoint boxes (not just the final one) ---
    min_x, max_x = float('inf'), float('-inf')
    min_y, max_y = float('inf'), float('-inf')
    min_z, max_z = float('inf'), float('-inf')
    for name in scenarios:
        for ivl in history[name]["states"]:
            min_x = min(min_x, ivl.lower[3]); max_x = max(max_x, ivl.upper[3])
            min_y = min(min_y, ivl.lower[4]); max_y = max(max_y, ivl.upper[4])
            min_z = min(min_z, ivl.lower[5]); max_z = max(max_z, ivl.upper[5])

    def get_padded_limits(min_val, max_val, padding_factor=0.05):
        range_val = max_val - min_val
        if range_val == 0: range_val = abs(max_val) * 0.1 or 0.1
        padding = range_val * padding_factor
        return [min_val - padding, max_val + padding]

    x_lims = get_padded_limits(min_x, max_x)
    y_lims = get_padded_limits(min_y, max_y)
    z_lims = get_padded_limits(min_z, max_z)

    fig_3d = plt.figure(figsize=(15, 10))
    ax_3d = fig_3d.add_subplot(111, projection='3d')

    def create_cuboid_faces(x_range, y_range, z_range):
        x0, x1 = x_range; y0, y1 = y_range; z0, z1 = z_range
        v = [(x0,y0,z0),(x1,y0,z0),(x1,y1,z0),(x0,y1,z0),(x0,y0,z1),(x1,y0,z1),(x1,y1,z1),(x0,y1,z1)]
        f = [[v[0],v[1],v[2],v[3]],[v[4],v[5],v[6],v[7]],[v[0],v[1],v[5],v[4]],
             [v[2],v[3],v[7],v[6]],[v[0],v[3],v[7],v[4]],[v[1],v[2],v[6],v[5]]]
        return f

    n_ckpts = len(times)
    for name in scenarios:
        color = color_map[name]
        for idx, ivl in enumerate(history[name]["states"]):
            # earlier boxes fainter, final box most solid.
            alpha = 0.15 + 0.5 * (idx / max(n_ckpts - 1, 1))
            collection = Poly3DCollection(
                create_cuboid_faces(
                    (ivl.lower[3], ivl.upper[3]),
                    (ivl.lower[4], ivl.upper[4]),
                    (ivl.lower[5], ivl.upper[5]),
                ),
                facecolors=color, linewidths=0.8, edgecolors='k', alpha=alpha,
            )
            ax_3d.add_collection3d(collection)

    LABEL_FONTSIZE = 20
    TICK_FONTSIZE = 16

    ax_3d.set_xlim(*x_lims)
    ax_3d.set_ylim(*y_lims)
    ax_3d.set_zlim(*z_lims)

    ax_3d.set_xlabel(dim_names[3], fontsize=LABEL_FONTSIZE, labelpad=15)
    ax_3d.set_ylabel(dim_names[4], fontsize=LABEL_FONTSIZE, labelpad=15)
    ax_3d.set_zlabel(dim_names[5], fontsize=LABEL_FONTSIZE, labelpad=15)

    ax_3d.tick_params(axis='x', labelsize=TICK_FONTSIZE)
    ax_3d.tick_params(axis='y', labelsize=TICK_FONTSIZE)
    ax_3d.tick_params(axis='z', labelsize=TICK_FONTSIZE)

    legend_patches = [Rectangle((0, 0), 1, 1, fc=color_map[name], alpha=0.5, label=name) for name in scenarios]
    ax_3d.legend(handles=legend_patches, fontsize=TICK_FONTSIZE, loc='center left', bbox_to_anchor=(1.05, 0.5))
    ax_3d.grid(True)
    ax_3d.set_title(
        f"ADMIRE reachable sets — box at every time step t = {times} s\n"
        "(fainter = earlier, solid = later)",
        fontsize=18,
    )

    if save_to_pdf:
        fig_3d.savefig(filename, bbox_inches='tight', pad_inches=0.3)
        print(f"Saved {filename}")
    else:
        plt.show()
    plt.close(fig_3d)


out_path = HERE / 'new_fig3.pdf'
plot_3d_boxes_over_time(
    history=history,
    dim_names=dim_names,
    times=checkpoint_times,
    save_to_pdf=True,
    filename=str(out_path),
)
