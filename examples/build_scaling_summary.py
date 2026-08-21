"""
Builds scaling_summary.csv -- ONE canonical, minimal CSV covering all three
scaling sweeps in this repo (nonlinear_chain horizon, nonlinear_chain
scenario-count, integrator_chain dimensionality), both optimizers
(unrefined/refined), with just enough columns to reproduce
plot_scaling_summary.py's plots: the swept parameter, the fixed problem
parameters, which config produced the row, compile time, run time, GPU VRAM,
and host RAM. Nothing else -- this is deliberately NOT a re-export of the
raw per-sweep CSVs (those keep every column: elapsed_s, loss, per-fault
breakdowns, etc. -- useful for debugging a sweep, not for reproducing a
plot).

This script and plot_scaling_summary.py are a decoupled pair: this one does
every judgment call (which config wins on run time at each point, which
config RAM/VRAM is read from, which dimensionality-sweep points are
outliers) and bakes the result into the CSV; the plotting script only reads
that CSV and draws lines -- it makes no decisions about the data.

nonlinear_chain horizon / scenario-count sweeps
------------------------------------------------
Both axes pool every non-zeroth-order (num_iters > 0) config CSV run for
that sweep and, at each swept-parameter value, take the config with the
LOWEST run_time_avg_ms as that row's run time. GPU VRAM (memory_kb) and host
RAM (peak_rss_mb) are NOT taken from that winning-run-time config -- if they
were, a point where the winning config switches would show a spurious jump
in VRAM/RAM that has nothing to do with the swept parameter, just a
restart-count baseline change (this exact bug was found and fixed for host
RAM in plot_horizon_runtime_scaling_best.py; the same reasoning applies to
VRAM, which is equally restart-count-dependent, so it gets the same
treatment here). Instead both are read from ONE FIXED reference config
(REFERENCE_CONFIG = "n256i8", 256 restarts x 8 iterations) at every point --
the config that already wins run time at nearly every point in both sweeps,
so it has full parameter-range coverage with no failures.

integrator_chain dimensionality sweep (N=2..100)
-------------------------------------------------
Each order N contributes exactly ONE un-repeated timing call (see
plot_post_compile_runtime_filtered.py's docstring), so the raw series is
contaminated by isolated scheduler/GC-pause spikes. filter_outliers() below
is ported verbatim (same half_win/rel_factor/abs_floor defaults) from that
script's outlier filter -- an iterative local-median filter, flagging a
point if it exceeds max(2x, +floor) of the median of its non-outlier
neighbors within +-5 in N, re-run to convergence. Unlike that script (which
drops outliers per-series only from the PLOT), here a point is dropped from
the CSV entirely -- from all three of its metrics at once -- if ANY of
run_time_ms / vram_mb / host_ram_mb is flagged, so the canonical CSV never
carries a known-bad measurement forward silently.
"""

import csv
import statistics as st
from pathlib import Path

_HERE = Path(__file__).resolve().parent
NLCHAIN = _HERE / "nonlinear_chain"
INTCHAIN = _HERE / "integrator_chain"

OUT_CSV = _HERE / "scaling_summary.csv"

FIELDS = [
    "sweep", "optimizer", "x", "N", "num_scenarios", "horizon_steps",
    "num_restarts", "num_iters", "compile_time_ms", "run_time_ms",
    "vram_mb", "host_ram_mb", "config",
]

REFERENCE_CONFIG = "n256i8"


# ══════════════════════════════════════════════════════════════════════════
# nonlinear_chain horizon / scenario-count sweeps: pool configs, take best
# run time, read RAM+VRAM from the fixed reference config.
# ══════════════════════════════════════════════════════════════════════════

UNREFINED_HORIZON_CONFIGS = [
    ("n20i15", NLCHAIN / "unrefined_horizon_scaling.csv"),
    ("n256i8", NLCHAIN / "unrefined_horizon_scaling_v2.csv"),
    ("n4096i5", NLCHAIN / "unrefined_horizon_scaling_bw_gd5.csv"),
    ("n4096i3", NLCHAIN / "unrefined_horizon_scaling_ms4096.csv"),
]
REFINED_HORIZON_CONFIGS = [
    ("n20i15", NLCHAIN / "refined_horizon_scaling.csv"),
    ("n256i8", NLCHAIN / "refined_horizon_scaling_v2.csv"),
    ("n4096i5", NLCHAIN / "refined_horizon_scaling_bw_gd5.csv"),
    ("n4096i3", NLCHAIN / "refined_horizon_scaling_ms4096.csv"),
]
UNREFINED_SCENARIO_CONFIGS = [
    ("n20i15", NLCHAIN / "unrefined_scenario_scaling.csv"),
    ("n256i8", NLCHAIN / "unrefined_scenario_scaling_v2.csv"),
    ("n4096i5", NLCHAIN / "unrefined_scenario_scaling_bw_gd5.csv"),
    ("n4096i3", NLCHAIN / "unrefined_scenario_scaling_ms4096.csv"),
    ("n1200000i8", NLCHAIN / "unrefined_scenario_scaling_wide_gd8.csv"),
]
REFINED_SCENARIO_CONFIGS = [
    ("n20i15", NLCHAIN / "refined_scenario_scaling.csv"),
    ("n256i8", NLCHAIN / "refined_scenario_scaling_v2.csv"),
    ("n4096i5", NLCHAIN / "refined_scenario_scaling_bw_gd5.csv"),
    ("n4096i3", NLCHAIN / "refined_scenario_scaling_ms4096.csv"),
    ("n200000i8", NLCHAIN / "refined_scenario_scaling_wide_gd8.csv"),
]

# Fixed problem parameters each sweep holds constant -- recorded in the CSV
# so a reader doesn't have to go find the producing script to know them.
HORIZON_FIXED_N = 10
HORIZON_FIXED_NUM_SCENARIOS = 7
SCENARIO_FIXED_N = 10
SCENARIO_FIXED_HORIZON_STEPS = {"unrefined": 4, "refined": 3}   # NUM_SEGMENTS vs. NUM_STEPS


def _load_nlchain_config_csv(csv_path: Path, x_key: str):
    """Read one config's raw sweep CSV, return {x: row_dict} for 'ok' rows.

    x_key must be passed explicitly, not auto-detected from the header: the
    scenario-count sweep's CSVs carry BOTH num_scenarios (the swept axis)
    AND num_segments/num_steps (the sweep's fixed horizon length, recorded
    per row) as columns, and the horizon sweep's CSVs carry both
    num_segments/num_steps (the swept axis) and num_scenarios (fixed). Only
    the caller knows which one is actually varying in a given sweep.
    """
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}
    out = {}
    for r in rows:
        if r["status"] != "ok":
            continue
        x = int(float(r[x_key]))
        out[x] = {
            "compile_time_ms": float(r["compile_time_ms"]),
            "run_time_ms": float(r["run_time_avg_ms"]),
            "vram_mb": float(r["memory_kb"]) / 1024.0,
            "host_ram_mb": float(r["peak_rss_mb"]),
            "num_restarts": int(float(r["num_restarts"])),
            "num_iters": int(float(r["num_iters"])),
        }
    return out


def pool_best(config_csvs, x_key, label):
    """Pool every (config_name, csv_path) sweep; at each x, keep the row
    with the lowest run_time_ms, but overwrite its vram_mb/host_ram_mb with
    REFERENCE_CONFIG's reading at that same x (see module docstring). Drops
    x values REFERENCE_CONFIG doesn't cover. Returns {x: row_dict}."""
    by_x = {}
    ref_by_x = {}
    for name, path in config_csvs:
        if not path.exists():
            print(f"  [{label}] skipping {name} ({path.name}) -- not found")
            continue
        parsed = _load_nlchain_config_csv(path, x_key)
        if name == REFERENCE_CONFIG:
            ref_by_x = parsed
        for x, row in parsed.items():
            row = dict(row, config=name)
            if x not in by_x or row["run_time_ms"] < by_x[x]["run_time_ms"]:
                by_x[x] = row
    out = {}
    for x, row in by_x.items():
        if x not in ref_by_x:
            print(f"  [{label}] dropping x={x}: {REFERENCE_CONFIG} has no 'ok' row there")
            continue
        row = dict(row)
        row["vram_mb"] = ref_by_x[x]["vram_mb"]
        row["host_ram_mb"] = ref_by_x[x]["host_ram_mb"]
        out[x] = row
    return out


def build_nlchain_rows():
    rows = []
    for sweep, optimizer, configs, x_key in [
        ("horizon", "unrefined", UNREFINED_HORIZON_CONFIGS, "num_segments"),
        ("horizon", "refined", REFINED_HORIZON_CONFIGS, "num_steps"),
        ("scenario", "unrefined", UNREFINED_SCENARIO_CONFIGS, "num_scenarios"),
        ("scenario", "refined", REFINED_SCENARIO_CONFIGS, "num_scenarios"),
    ]:
        pooled = pool_best(configs, x_key, f"{sweep}/{optimizer}")
        for x in sorted(pooled):
            d = pooled[x]
            if sweep == "horizon":
                N, num_scenarios, horizon_steps = HORIZON_FIXED_N, HORIZON_FIXED_NUM_SCENARIOS, x
            else:
                N, num_scenarios = SCENARIO_FIXED_N, x
                horizon_steps = SCENARIO_FIXED_HORIZON_STEPS[optimizer]
            rows.append({
                "sweep": sweep, "optimizer": optimizer, "x": x,
                "N": N, "num_scenarios": num_scenarios, "horizon_steps": horizon_steps,
                "num_restarts": d["num_restarts"], "num_iters": d["num_iters"],
                "compile_time_ms": round(d["compile_time_ms"], 3),
                "run_time_ms": round(d["run_time_ms"], 4),
                "vram_mb": round(d["vram_mb"], 4),
                "host_ram_mb": round(d["host_ram_mb"], 2),
                "config": d["config"],
            })
        print(f"  {sweep}/{optimizer}: {len(pooled)} points")
    return rows


# ══════════════════════════════════════════════════════════════════════════
# integrator_chain dimensionality sweep: outlier-filter, drop bad rows.
# ══════════════════════════════════════════════════════════════════════════

_HALF_WIN = 5
_REL_FACTOR = 2.0
_ABS_FLOOR_MS = 10.0
_ABS_FLOOR_VRAM_MB = 10.0
_ABS_FLOOR_HOST_MB = 512.0   # same magnitude as plot_post_compile_runtime_filtered.py's 0.5 GiB floor
_MAX_PASSES = 10

INT_CHAIN_CONFIGS = {
    "unrefined": (INTCHAIN / "integrator_unrefined_runtime.csv", 30, 20),
    "refined": (INTCHAIN / "integrator_refinement_runtime.csv", 30, 20),
}


def filter_outliers(Ns, vals, abs_floor, half_win=_HALF_WIN, rel_factor=_REL_FACTOR, max_passes=_MAX_PASSES):
    """Iterative local-median outlier filter, ported verbatim (algorithm and
    defaults) from plot_post_compile_runtime_filtered.py -- see that
    script's module docstring for the full rationale and the calibration
    that picked rel_factor=2.0 / half_win=5. Returns a boolean list, True
    where the point is an outlier."""
    n = len(vals)
    is_outlier = [False] * n

    def local_median(i):
        lo, hi = max(0, i - half_win), min(n, i + half_win + 1)
        window = [vals[j] for j in range(lo, hi) if j != i and not is_outlier[j]]
        if len(window) < 3:
            window = [vals[j] for j in range(lo, hi) if j != i]
        return st.median(window) if window else vals[i]

    for _ in range(max_passes):
        changed = False
        for i in range(n):
            med = local_median(i)
            flag = vals[i] > max(rel_factor * med, med + abs_floor)
            if flag != is_outlier[i]:
                changed = True
            is_outlier[i] = flag
        if not changed:
            break
    return is_outlier


def build_intchain_rows():
    rows = []
    for optimizer, (csv_path, num_restarts, num_iters) in INT_CHAIN_CONFIGS.items():
        with open(csv_path, newline="") as f:
            data = sorted(
                (int(r["N"]), float(r["compile_time_ms"]), float(r["run_time_ms"]),
                 float(r["peak_gpu_mem_bytes"]) / 1e6, float(r["peak_host_rss_bytes"]) / (1024.0 ** 2))
                for r in csv.DictReader(f)
            )
        Ns = [d[0] for d in data]
        compile_ms = [d[1] for d in data]
        run_ms = [d[2] for d in data]
        vram_mb = [d[3] for d in data]
        host_mb = [d[4] for d in data]

        o_run = filter_outliers(Ns, run_ms, _ABS_FLOOR_MS)
        o_vram = filter_outliers(Ns, vram_mb, _ABS_FLOOR_VRAM_MB)
        o_host = filter_outliers(Ns, host_mb, _ABS_FLOOR_HOST_MB)
        dropped = [Ns[i] for i in range(len(Ns)) if o_run[i] or o_vram[i] or o_host[i]]
        print(f"  dimensionality/{optimizer}: {len(Ns)} points, "
              f"dropping {len(dropped)} outlier(s): N={dropped}")

        for i in range(len(Ns)):
            if o_run[i] or o_vram[i] or o_host[i]:
                continue
            rows.append({
                "sweep": "dimensionality", "optimizer": optimizer, "x": Ns[i],
                "N": Ns[i], "num_scenarios": "", "horizon_steps": "",
                "num_restarts": num_restarts, "num_iters": num_iters,
                "compile_time_ms": round(compile_ms[i], 3),
                "run_time_ms": round(run_ms[i], 4),
                "vram_mb": round(vram_mb[i], 4),
                "host_ram_mb": round(host_mb[i], 2),
                "config": f"n{num_restarts}i{num_iters}",
            })
    return rows


if __name__ == "__main__":
    print("nonlinear_chain sweeps:")
    nlchain_rows = build_nlchain_rows()
    print("\nintegrator_chain dimensionality sweep:")
    intchain_rows = build_intchain_rows()

    all_rows = nlchain_rows + intchain_rows
    all_rows.sort(key=lambda r: (r["sweep"], r["optimizer"], r["x"]))

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nWrote {len(all_rows)} rows to {OUT_CSV}")
