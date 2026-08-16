# PACE runtime-scaling package

Data, code, sbatch job definitions, raw Slurm logs, and rendered plots for
three runtime-scaling sweeps run on PACE Phoenix (GT HPC), organized by
example (`nonlinear_chain`, `admire`, `integrator_chain`). Paths below are
relative to this directory.

## Layout

```
data/     raw CSV/NPZ output from each PACE job -- source of truth
code/     the exact scripts that produced data/ and plots/ (sweep scripts,
          plot scripts, sbatch job definitions, and the library modules
          they import)
docs/     design notes (nonlinear_chain only -- admire/integrator_chain
          reuse pre-existing modules with no separate design doc)
logs/     raw stdout from each Slurm job -- includes the source integrator_chain
          scrapes to build its CSV, since that sweep prints timing to
          stdout only (no CSV/NPZ of its own)
plots/    the rendered PDFs (already generated from data/ via code/)
```

## Per-example summary

### nonlinear_chain -- fault-scenario-count scaling (N=10 fixed)
- **What it measures**: compile time / run time / memory vs. number of
  fault scenarios, for unrefined vs. refined multistep optimization.
- **Data**: `data/nonlinear_chain/{unrefined,refined}_scenario_scaling.csv`
  (one row per scenario count; `status` column is `ok`/`timeout`/`oom`/`error`
  -- only `ok` rows have numeric timing).
- **Regenerate data**: `code/nonlinear_chain/run_{unrefined,refined}_scaling.sbatch`
  (submits `{unrefined,refined}_scenario_scaling.py`, which uses
  `runtime_scaling_common.py` for the worker harness).
- **Plot**: `code/nonlinear_chain/plot_scenario_runtime_scaling.py` reads
  the two CSVs -> `plots/scenario_runtime_scaling.pdf`.
- **Design notes**: `docs/nonlinear_chain/PLAN.md`, `docs/nonlinear_chain/TASKS.md`.

### admire -- success-rate / runtime sweep (3 methods x 3 horizons x 9 configs)
- **What it measures**: for each of 3 methods (single-step, multistep
  unrefined, multistep refined) x 3 horizons (0.5/1.0/2.0s), ONE compile
  time and ONE run time for the entire batched sweep: 9 configs
  (3 `input_limit` x 3 `x0_width`) x 20 restarts x 100 GD iterations, all
  double-vmapped into a single `jax.jit`, where each loss evaluation
  itself propagates all 11 fault scenarios. See
  `code/admire/admire_success_rate_analysis.py`'s module docstring for the
  full sweep-axis rationale and the resource-risk analysis (ADMIRE's
  dynamics divide 5x per Euler step, which is expensive to differentiate).
- **Data**: `data/admire/admire_success_rate_summary_<method>.csv` (per-config
  success/loss rows) and `data/admire/admire_success_rate_data_<method>_<horizon>s.npz`
  (raw per-restart losses + the compile_time_s/run_time_s scalars the
  runtime plot reads). `admire_success_rate_summary.csv` (no method suffix)
  is the older combined-write path kept for reference.
- **Recover a summary CSV from just the .npz files** (e.g. after a killed
  run): `code/admire/rebuild_summary_csv_from_npz.py --method <method>`.
- **Regenerate data**: `code/admire/run_success_rate_analysis.sbatch`,
  submitted once per method (`--export=METHOD=single_step` etc.), which
  runs `admire_success_rate_analysis.py`.
- **Plot**: `code/admire/plot_admire_runtime_scaling.py` reads the 9 .npz
  files directly (the summary CSVs do NOT carry timing) ->
  `plots/admire_runtime_scaling.pdf`.
- **Raw job logs**: `logs/admire_success_rate_admire-<method>_<jobid>.out`.

### integrator_chain -- dimensionality scaling (N=2..100)
- **What it measures**: refined multistart optimizer compile/run time vs.
  integrator-chain order N, N=2..100, 30 restarts x 20 iters per N.
- **Fix included**: `code/integrator_chain/refinement_demo_stable.py` is a
  NaN-safe version of the refined optimizer (see its module docstring) --
  the original hit NaN at high N due to a JAX `jnp.where`-branch autodiff
  gotcha (an unselected branch's gradient can still poison the result via
  `inf * 0 = nan`). This stable version fixes it via a double-`where`
  guard plus `nan_to_num`/`nanargmin`, and is what actually produced
  `data/integrator_chain/integrator_refinement_runtime.csv` (0 NaN across
  all 99 orders).
- **Data**: this sweep prints timing to stdout only, no CSV/NPZ of its
  own -- `logs/integrator_refinement_scaling_stable_11774929.out` is the
  actual source of truth; `data/integrator_chain/integrator_refinement_runtime.csv`
  is a parsed/durable copy of it, with columns
  `N, compile_time_ms, run_time_ms, peak_gpu_mem_bytes`.
  `peak_gpu_mem_bytes` is the XLA allocator's `peak_bytes_in_use` from the
  per-order "Memory snapshot" line (0.148 MB at N=2 -> 148.6 MB at N=100).
  That counter is a process-lifetime running max, but every one of the 99
  orders sets a new record, so each value is that order's own peak -- the
  parser re-checks this and warns if a rerun ever produces a flat (censored)
  stretch. Note this is *live* device memory, not the 25.55 GB the process
  reserved: neither the sbatch script nor `refinement_demo_stable.py` sets
  `XLA_PYTHON_CLIENT_PREALLOCATE=false`, so JAX preallocated 75% of the
  V100's 32 GB and used ~0.6% of it.
  There is **no per-N host RAM series**: `time_jit`'s `_memory_snapshot()`
  returns device stats whenever a GPU is present and only falls back to
  `ru_maxrss` when there isn't one, so host RSS was never sampled per order.
  The only host-memory figure this job produced is the single job-level
  Slurm-epilog scalar `mem=24995472K` (~23.84 GiB peak, whole run). To get a
  host-RAM curve, `_memory_snapshot()` would have to record `ru_maxrss`
  *alongside* the device stats rather than instead of them, and the sweep
  re-run -- it cannot be recovered from this log.
- **Regenerate data**: `code/integrator_chain/run_refinement_scaling_stable.sbatch`
  (runs `refinement_demo_stable.py`).
- **Plots**: `code/integrator_chain/plot_refinement_runtime_scaling.py`
  parses the log above -> writes the CSV -> `plots/integrator_refinement_runtime_scaling.pdf`
  (two panels, compile time and run time, all points).
  `code/integrator_chain/plot_post_compile_runtime_filtered.py` reads that
  CSV -> `plots/integrator_refinement_postcompile_runtime_filtered.pdf`:
  run time (left axis, blue) with peak GPU memory overlaid on a right-hand
  axis (orange), both passed through its local-median outlier filter --
  17/99 run-time points are dropped as scheduler/GC spikes, 0 memory points
  (that series is smooth and strictly increasing -- see the script's
  "Why the memory series is smooth" note for the three checks confirming it
  is a measured allocator trace, not a fitted curve). Every surviving point
  of both series is drawn and connected by lines; no marker subsampling.
  Its two y-scales are
  independent, so curve crossings there carry no meaning; call
  `make_plot(..., twin_axis=False)` for the two-panel version instead.

## Reproducing / replotting

All three plot scripts are standalone: `cd` into the matching `code/<example>/`
directory and run e.g. `python plot_scenario_runtime_scaling.py` -- they
read CSV/NPZ/log paths relative to their own location, so this package's
`data/` and `logs/` layout mirrors the original repo's `examples/<name>/`
and `logs/` directories closely enough that copying `data/<example>/*` and
`logs/*` back next to the matching `code/<example>/*.py` (restoring the
original repo layout) is enough to re-run the plot scripts unchanged.

Font: all plots use `Times New Roman`, falling back to `Liberation Serif`
if not installed (Times New Roman is not installed on PACE; Liberation
Serif is what actually rendered these PDFs).
