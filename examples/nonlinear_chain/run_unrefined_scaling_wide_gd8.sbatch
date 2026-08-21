#!/bin/bash
# =============================================================================
# run_unrefined_scaling_wide_gd8.sbatch -- single-node, single-GPU (any
# available type) job that runs ONLY the unrefined-multistep SCENARIO-COUNT
# runtime/RAM/VRAM scaling sweep (unrefined_scenario_scaling.py), using the
# widest multi-start budget tried anywhere in this directory (matching
# run_unrefined_horizon_scaling_zeroth_wide.sbatch's restart count) but with
# 8 gradient-descent iterations instead of zeroth order.
#
# Why this config: the horizon "best measured runtime" analysis
# (plot_horizon_runtime_scaling_best.py) found that across every non-zeroth
# config actually run for horizon scaling (20x15, 256x8, 4096x5, 4096x3),
# 256 restarts x 8 iterations ("n256i8", i.e. *_v2.csv) won on run time at
# 15 of 17 horizon steps -- more GD iterations at moderate restart width beat
# more restarts at fewer iterations. This job asks whether pairing that same
# num_iters=8 with the much wider 1.2M-restart budget used in the horizon
# zeroth-order sweep does even better, or instead trades a wider multi-start
# cloud for infeasible memory -- this exact (1.2M restarts, 8 iters)
# combination has not been run before, for either scaling axis.
#
# Config vs. the other unrefined_scenario_scaling*.sbatch variants:
#   - NLCHAIN_NUM_RESTARTS=1200000 (v2: 256; bw_zeroth/bw_gd5/ms4096: 4096)
#   - NLCHAIN_NUM_ITERS=8 (matches v2's iteration count, NOT zeroth order --
#     see run_unrefined_scaling_zeroth_wide.sbatch's counterpart on the
#     horizon axis for the true num_iters=0 config, which this is not)
#   - Scenario-count sweep ONLY, no horizon-sweep counterpart here (that one
#     is run_unrefined_horizon_scaling.sbatch's family).
#   - Any GPU accepted (gpu-h100/l40s/a100/rtx6000/v100 partition list, no
#     --constraint), matching the pattern validated for the admire timing job.
#
# SEVERE MEMORY WARNING: the horizon zeroth_wide job (same 1.2M restarts,
# num_iters=0, forward-pass-only) already device-OOM'd at its single largest
# point (RESOURCE_EXHAUSTED trying to allocate ~37GB on a 48GB L40S, at
# num_segments=50 with C(7,2)=21 fixed scenario pairs). This job is a strictly
# harder ask on BOTH axes at once:
#   (1) num_iters=8 means jax.grad is taken through the whole restart batch --
#       reverse-mode autodiff must retain forward-pass activations, typically
#       several times the pure-forward memory footprint zeroth order needed.
#   (2) the scenario-count sweep goes up to 21 scenarios (C(21,2)=210 pairs),
#       10x the 21 pairs fixed in the horizon sweep.
# It is entirely possible most or all points in this sweep device-OOM even on
# the largest GPUs in the partition list (h100/a100 80GB). That is an
# acceptable, informative outcome, not a failure of the job: per-point
# subprocess isolation (WORKER_TIMEOUT_S/WORKER_MEM_LIMIT_MB = math.inf in
# unrefined_scenario_scaling.py) means each scenario-count point that OOMs is
# recorded as status='error' and flushed to CSV, while any point that DOES
# fit is preserved -- the sweep as a whole tells you where the real
# restart-width ceiling is for this config, which is the point of running it.
#
# PREREQUISITES / submission -- see run_unrefined_scaling.sbatch's header for
# the full explanation (conda env setup, why you must
# `cd <repo_root> && sbatch examples/nonlinear_chain/<this file>` rather than
# submitting from inside examples/nonlinear_chain/).
# =============================================================================
#SBATCH --job-name=nlchain-unrefined-scaling-wide-gd8
#SBATCH --account=gts-scoogan3-fy20phase3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-h100,gpu-l40s,gpu-a100,gpu-rtx6000,gpu-v100
#SBATCH --mem=120G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/nlchain_unrefined_scaling_wide_gd8_%j.out
#SBATCH --error=logs/nlchain_unrefined_scaling_wide_gd8_%j.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
EXAMPLE_DIR="${REPO_ROOT}/examples/nonlinear_chain"
cd "${REPO_ROOT}"
mkdir -p logs

echo "=== Slurm job ${SLURM_JOB_ID:-local} on $(hostname) at $(date) ==="
echo "REPO_ROOT: ${REPO_ROOT}"
echo "GPUs visible: ${CUDA_VISIBLE_DEVICES:-none}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true

# ---------------------------------------------------------------------------
# Environment setup -- conda env with immrax installed.
# ---------------------------------------------------------------------------
module load anaconda3
CONDA_ENV="${CONDA_ENV:-${REPO_ROOT}/envs/immrax}"
PY="${CONDA_ENV}/bin/python"
if [ ! -x "${PY}" ]; then
    echo "ERROR: conda env python not found at ${PY}."
    echo "       Create it first -- see run_unrefined_scaling.sbatch's PREREQUISITES header."
    exit 1
fi
echo "Python: $("${PY}" --version) @ ${PY}"

# Keep pip/conda caches off the (small, quota-limited) home filesystem.
export PIP_CACHE_DIR="${REPO_ROOT}/.cache/pip"
mkdir -p "${PIP_CACHE_DIR}"

# Force the GPU backend. The scaling script uses os.environ.setdefault, so
# it will NOT override this once it's set here; its own worker subprocesses
# inherit this env var too.
export JAX_PLATFORMS=cuda

# Launch-overhead mitigation -- see run_unrefined_scaling.sbatch's comment
# for the full rationale (kernel-launch-bound workload, command buffers
# capture+replay the launch sequence as a CUDA graph).
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_command_buffer=FUSION,CUBLAS,CUSTOM_CALL"

# ---------------------------------------------------------------------------
# WIDE restarts + 8 GD iterations. See header SEVERE MEMORY WARNING above.
#
# Runs unrefined_scenario_scaling_wide_gd8.py -- a STANDALONE script, not
# unrefined_scenario_scaling.py with env overrides. This job's first attempt
# (env-var override of the shared script) got its config silently reverted
# mid-run by another process running git checkouts against this same working
# directory; see unrefined_scenario_scaling_wide_gd8.py's module docstring
# for the full incident. NUM_RESTARTS/NUM_ITERS are hardcoded in that
# standalone file, so no NLCHAIN_* env vars are needed here.
# ---------------------------------------------------------------------------
"${PY}" -c "import jax; print('JAX devices:', jax.devices())"

cd "${EXAMPLE_DIR}"

echo ""
echo "================================================================"
echo "  UNREFINED scenario-count runtime/RAM/VRAM sweep (wide restarts, 8 GD iters)"
echo "  Started: $(date)"
echo "================================================================"
"${PY}" unrefined_scenario_scaling_wide_gd8.py
echo "Unrefined scenario sweep complete at $(date)"
echo "CSV written to: ${EXAMPLE_DIR}/unrefined_scenario_scaling_wide_gd8.csv"
