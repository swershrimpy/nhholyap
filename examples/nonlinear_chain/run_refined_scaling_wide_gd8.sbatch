#!/bin/bash
# =============================================================================
# run_refined_scaling_wide_gd8.sbatch -- single-node, single-GPU (any
# available type) job that runs ONLY the intersection-refinement
# SCENARIO-COUNT runtime/RAM/VRAM scaling sweep (refined_scenario_scaling.py),
# using the widest multi-start budget tried anywhere in this directory
# (matching run_refined_horizon_scaling_zeroth_wide.sbatch's restart count)
# but with 8 gradient-descent iterations instead of zeroth order.
#
# Why this config -- see run_unrefined_scaling_wide_gd8.sbatch's header for
# the full rationale (n256i8 = 256 restarts x 8 iters won on run time at 15
# of 17 horizon steps in the "best measured runtime" analysis; this job
# tests num_iters=8 at the much wider 200k-restart budget used in the
# horizon zeroth-order sweep -- an untested combination on either axis).
#
# Config vs. the other refined_scenario_scaling*.sbatch variants:
#   - NLCHAIN_NUM_RESTARTS=200000 (v2: 256; bw_zeroth/bw_gd5/ms4096: 4096).
#     Smaller than the companion unrefined job's 1.2M restarts, same
#     asymmetry as the horizon zeroth_wide pair -- refined's per-restart cost
#     is much higher, so 200k is the analogous "as wide as still tractable"
#     budget here.
#   - NLCHAIN_NUM_ITERS=8 (matches v2's iteration count, NOT zeroth order)
#   - Scenario-count sweep ONLY, no horizon-sweep counterpart here.
#   - Any GPU accepted (gpu-h100/l40s/a100/rtx6000/v100 partition list, no
#     --constraint), matching the pattern validated for the admire timing job.
#
# MEMORY NOTE: refined's per-restart device footprint is the pair-stacked
# refinement carry, which scales with C(n,2) pairs -- up to C(21,2)=210 at
# the top of this sweep's scenario-count range (10x the 21 pairs the horizon
# sweep runs at). The horizon zeroth_wide refined job (same 200k restarts,
# num_iters=0) completed all 17 points cleanly with no OOM, so the
# per-restart footprint at this width has real headroom -- but num_iters=8
# means jax.grad is taken through the whole restart batch this time
# (reverse-mode autodiff retains forward-pass activations, unlike zeroth
# order's forward-only evaluation), and 210 pairs at the largest scenario
# counts is meaningfully more than the horizon sweep's fixed 21, so some risk
# of device-OOM at the largest scenario counts remains, just less than the
# unrefined companion job's. Per-point subprocess isolation
# (WORKER_TIMEOUT_S/WORKER_MEM_LIMIT_MB = math.inf in
# refined_scenario_scaling.py) means any point that does OOM only loses that
# one point.
#
# PREREQUISITES / submission -- see run_refined_scaling.sbatch's header for
# the full explanation (conda env setup, why you must
# `cd <repo_root> && sbatch examples/nonlinear_chain/<this file>` rather than
# submitting from inside examples/nonlinear_chain/).
# =============================================================================
#SBATCH --job-name=nlchain-refined-scaling-wide-gd8
#SBATCH --account=gts-scoogan3-fy20phase3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-h100,gpu-l40s,gpu-a100,gpu-rtx6000,gpu-v100
#SBATCH --mem=120G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/nlchain_refined_scaling_wide_gd8_%j.out
#SBATCH --error=logs/nlchain_refined_scaling_wide_gd8_%j.err
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
    echo "       Create it first -- see run_refined_scaling.sbatch's PREREQUISITES header."
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

# Launch-overhead mitigation -- see run_refined_scaling.sbatch's comment for
# the full rationale (kernel-launch-bound workload, command buffers
# capture+replay the launch sequence as a CUDA graph).
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_command_buffer=FUSION,CUBLAS,CUSTOM_CALL"

# ---------------------------------------------------------------------------
# WIDE restarts + 8 GD iterations. See header MEMORY NOTE above.
# ---------------------------------------------------------------------------
export NLCHAIN_NUM_RESTARTS=200000
export NLCHAIN_NUM_ITERS=8
export NLCHAIN_CSV=refined_scenario_scaling_wide_gd8.csv
"${PY}" -c "import jax; print('JAX devices:', jax.devices())"

cd "${EXAMPLE_DIR}"

echo ""
echo "================================================================"
echo "  REFINED scenario-count runtime/RAM/VRAM sweep (wide restarts, 8 GD iters)"
echo "  Started: $(date)"
echo "================================================================"
"${PY}" refined_scenario_scaling.py
echo "Refined scenario sweep complete at $(date)"
echo "CSV written to: ${EXAMPLE_DIR}/refined_scenario_scaling_wide_gd8.csv"
