# Faulty-car AMD separating-input sweep -- PACE draft

Exact active-model-discrimination (AMD) separating-input MILP for the
faulty-nonholonomic-car 3-mode discrimination problem (Nominal / Actuator
Fault / Sensor Fault), swept over 84 configs (3 initial-interval widths x
4 actuator-fault authority levels x 7 initial-position centres), adapted
to run as a PACE Slurm job array with a much longer per-config timeout
than the 60s smoke-test cap used for the local workstation run.

**This has not been submitted to or tested on PACE.** It is a draft built
from this repo's existing `pace_runtime_scaling_package` conventions
(sbatch structure, account string, resource-sizing writeups) plus a local
dry run of the same MILP on a workstation. Everything under "ASSUMPTIONS
TO VERIFY" below needs checking against the actual cluster before
submitting.

## What's here

| File | Purpose |
|---|---|
| `amd_lib/StateSpace.m`, `amd_lib/Exact_get_u_respon.m` | Byte-identical copies of AMD-code's algorithm/optimizer (md5-verified) -- **not modified**. |
| `faulty_car_config.m` | Builds `(modes, bounds, T_hor, epsi, NNorm)` for one `(width, alpha_hi, cx, cy)` config; same structure as `model_faulty_car.m`, parametrised. |
| `generate_config_table.m` | Builds the 84-row config grid -> `configs.mat`/`configs.csv`. |
| `run_one_config_pace.m` | Solves one config, saves `results/result_<idx>.mat`. |
| `run_faulty_car_amd_sweep.sbatch` | Slurm job array (`--array=1-84`), one task per config. |
| `aggregate_success_rate.m` | Scans `results/`, prints/writes the success-rate summary. |

## Why this differs from the local (workstation) run

The local run of this same MILP (24-core i7-13700HX, Gurobi 13.0.1
academic, 60s hard cap per config) showed the solver's branch-and-bound
gap stuck at 100% (no bound progress at all) for the full 60s window on
every config tried, so **every config observed locally timed out** at the
60s cap. That's a real result for the 60s question, but it leaves open
whether these instances are solvable at all given enough time. Rather
than serialize 84 configs behind one long walltime (which would need
~168h at 2h/config -- over the 72h ceiling this repo's other sbatch jobs
already ran into), this uses a Slurm **job array**: each config gets its
own independent time budget and Slurm schedules them concurrently, so
wall time is bounded by the slowest config rather than the sum of all 84.

## ASSUMPTIONS TO VERIFY before submitting

1. **Gurobi license.** The workstation license
   (`/opt/gurobi1301/linux64/gurobi.lic`) is `TYPE=ACADEMIC`,
   **node-locked to this workstation's HOSTID** -- it will not work on a
   PACE compute node. Get either PACE's own Gurobi module/site license,
   a fresh node-locked license issued for PACE hardware, or a
   floating/named-user license reachable from PACE's network, then point
   `GUROBI_MATLAB_DIR` (and however PACE wants `GRB_LICENSE_FILE` set) at
   it in the sbatch script.
2. **MATLAB module name/version** -- `module avail matlab` on PACE and
   update `MATLAB_MODULE` in the sbatch script (defaults to `matlab`).
3. **YALMIP** isn't a PACE module; clone it into the repo once (e.g.
   `git clone https://github.com/yalmip/YALMIP.git thirdparty/YALMIP`)
   and confirm `YALMIP_DIR` resolves to it.
4. **Account string** `gts-scoogan3-fy20phase3` was copied from this
   repo's existing GPU sbatch jobs (`pace_runtime_scaling_package`) --
   confirm it's still valid/appropriate for a CPU-only job.
5. **`--cpus-per-task=8` / `--mem=16G`** are not measured for this
   workload -- they're carried over from other CPU-ish defaults in this
   repo, scaled down from the GPU jobs' 8 cores. `Exact_get_u_respon.m`'s
   `sdpsettings` call has no `Threads` option set (left unmodified per
   the requirement not to touch the algorithm/optimizer), so watch the
   Gurobi log's "Thread count" line against the actual cgroup allocation.
6. **Per-config time limit** (`#SBATCH --time`, default `02:00:00`, with
   an internal `timeout` a few minutes under that) is a starting point,
   not evidence it's enough -- given the 100%-gap behavior seen locally,
   treat the first array's success rate as a probe. If most configs still
   time out at 2h, that itself is a meaningful empirical result about
   this MILP formulation on this problem, not necessarily a
   mis-configuration.

## Running it (once the assumptions above are confirmed)

```bash
cd examples/faulty_car/amd_matlab_separating_input
mkdir -p logs results
matlab -batch "generate_config_table"        # regenerates configs.mat if needed
sbatch --array=1-84%12 run_faulty_car_amd_sweep.sbatch   # %12 = example throttle, tune to your QOS
# ... wait for the array to finish (or partially finish) ...
matlab -batch "aggregate_success_rate"       # prints success rate, writes results_summary.csv
```

The sweep is resume-safe: re-submitting the same array skips any config
that already has a `results/result_<idx>.mat`.
