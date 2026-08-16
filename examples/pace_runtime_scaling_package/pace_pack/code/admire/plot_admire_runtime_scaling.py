"""
Runtime-scaling companion plot for admire_success_rate_analysis.py's
per-(method, horizon) sweep.

Reads the 9 admire_success_rate_data_<method>_<horizon>s.npz files it writes
(3 methods x 3 horizons) and produces admire_runtime_scaling.pdf: a two-panel
figure of compile time and steady-state run time vs. horizon, one line per
method. Unlike unrefined_scenario_scaling.csv / refined_scenario_scaling.csv
in nonlinear_chain, admire_success_rate_summary_<method>.csv does NOT carry
per-config compile/run time -- those are one scalar per (method, horizon),
saved only in the .npz files (see admire_success_rate_analysis.py's
run_method_for_horizon) -- so this reads the .npz files directly rather than
a CSV.

Does not run any JAX code itself -- run the three
`sbatch run_success_rate_analysis.sbatch` jobs first (one per method) to
(re)generate the .npz files, then run this script.
"""

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from admire_success_rate_analysis import HORIZONS_S, _METHOD_LOSS_BUILDERS, _METHOD_DISPLAY

OUT_PDF = _HERE / "admire_runtime_scaling.pdf"

# Categorical slots 1/2/3 from the project's validated palette (light mode):
# blue/orange/aqua, same fixed order used by
# nonlinear_chain/plot_scenario_runtime_scaling.py (slots 1/2) extended by
# one slot for this figure's third series.
_METHOD_COLORS = {
    "single_step": "#2a78d6",           # slot 1 (blue)
    "multistep_unrefined": "#eb6834",   # slot 2 (orange)
    "multistep_refined": "#1baf7a",     # slot 3 (aqua)
}

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Liberation Serif'],
    'text.usetex': False,
})


def load_runtime_data():
    """method -> (horizons_present, compile_times_s, run_times_s), sorted by horizon."""
    data = {}
    missing = []
    for method in _METHOD_LOSS_BUILDERS:
        horizons, compile_ts, run_ts = [], [], []
        for horizon_s in HORIZONS_S:
            npz_path = _HERE / f"admire_success_rate_data_{method}_{horizon_s}s.npz"
            if not npz_path.exists():
                missing.append(npz_path.name)
                continue
            d = np.load(npz_path)
            horizons.append(horizon_s)
            compile_ts.append(float(d["compile_time_s"]))
            run_ts.append(float(d["run_time_s"]))
        order = np.argsort(horizons)
        data[method] = (
            np.array(horizons)[order],
            np.array(compile_ts)[order],
            np.array(run_ts)[order],
        )
    return data, missing


def make_plot(data, out_path: Path = OUT_PDF):
    fig, (ax_compile, ax_run) = plt.subplots(1, 2, figsize=(11, 4.4))

    for method in _METHOD_LOSS_BUILDERS:
        horizons, compile_ts, run_ts = data[method]
        if len(horizons) == 0:
            continue
        color = _METHOD_COLORS[method]
        label = _METHOD_DISPLAY[method]
        ax_compile.plot(horizons, compile_ts, color=color, linewidth=2,
                        marker='o', markersize=6, label=label)
        ax_run.plot(horizons, run_ts * 1e3, color=color, linewidth=2,
                    marker='o', markersize=6, label=label)

    for ax, ylabel, title in (
        (ax_compile, "Compile time (s)", "JIT compile time"),
        (ax_run, "Run time (ms)", "Post-compilation run time"),
    ):
        ax.set_xlabel("Horizon (s)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks(HORIZONS_S)
        ax.grid(True, color="0.85", linewidth=0.8)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    ax_run.legend(frameon=False)
    fig.suptitle("ADMIRE success-rate sweep -- compile/run time vs. horizon, by method\n"
                 "(9 configs x 20 restarts x 100 GD iters per point)")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    data, missing = load_runtime_data()
    for method in _METHOD_LOSS_BUILDERS:
        horizons, compile_ts, run_ts = data[method]
        print(f"{_METHOD_DISPLAY[method]}: {len(horizons)}/{len(HORIZONS_S)} horizons found")
        for h, c, r in zip(horizons, compile_ts, run_ts):
            print(f"  horizon={h:4.1f}s  compile={c:8.2f}s  run={r*1e3:8.3f}ms")
    if missing:
        print(f"\nMissing .npz files (skipped): {missing}")

    pdf_path = make_plot(data)
    print(f"\nWrote plot: {pdf_path}")
