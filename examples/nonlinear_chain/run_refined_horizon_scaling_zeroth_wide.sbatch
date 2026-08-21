#!/bin/bash
# =============================================================================
# run_refined_horizon_scaling_zeroth_wide.sbatch -- single-node, single-GPU
# (any available type) job that runs ONLY the intersection-refinement
# PLANNING-HORIZON runtime/RAM/VRAM scaling sweep (refined_horizon_scaling.py),
# zeroth-order only, at a wider multi-start budget than the other variants in
# this directory.
#
# Config vs. the other refined_horizon_scaling*.sbatch variants:
#   - NLCHAIN_NUM_RESTARTS=200000 (v2 default: 256; bw_zeroth/bw_gd5: 4096).
#     Smaller than the companion unrefined job's 1.2M restarts -- refined's
#     per-restart cost is much higher (pairwise intersection-refinement over
#     C(7,2)=21 pairs every step, vs. unrefined's plain propagate-and-compare),
#     so 200k here is the analogous "as wide as is still tractable" budget,
#     not a like-for-like restart count with the unrefined job.
#   - NLCHAIN_NUM_ITERS=0 (zeroth order only -- no gradient-descent variant
#     submitted alongside this one)
#   - Horizon sweep ONLY, no scenario-count-sweep counterpart.
#   - u0 is already sampled independently per restart AND per horizon step
#     (jax.random.normal(key, (num_restarts, num_steps, N)) in
#     optimize_refined_gpu, nonlinear_chain_separating_input.py) -- every
#     step of every restart's candidate control sequence is an independent
#     draw already; there is no constant-across-steps mode to disable here.
#   - Any GPU accepted (gpu-h100/l40s/a100/rtx6000/v100 partition list, no
#     --constraint), matching the pattern validated for the admire timing job.
#
# MEMORY NOTE: refined's per-restart device footprint is the pair-stacked
# refinement carry, (P=21, 2, 2N=20) float32 = ~3.4KB, so even at 200k
# restarts the peak device array is on the order of ~700MB per horizon point
# -- comfortably within VRAM on every partition in the list. Per-point
# subprocess isolation (WORKER_TIMEOUT_S / WORKER_MEM_LIMIT_MB = math.inf in
# refined_horizon_scaling.py) still applies, so a failure at any one horizon
# point only loses that point.
#
# PREREQUISITES / submission -- see run_refined_horizon_scaling.sbatch's
# header for the full explanation (conda env setup, why you must
# `cd <repo_root> && sbatch examples/nonlinear_chain/<this file>` rather than
# submitting from inside examples/nonlinear_chain/).
# =============================================================================
#SBATCH --job-name=nlchain-refined-horizon-zeroth-wide
#SBATCH --account=gts-scoogan3-fy20phase3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-h100,gpu-l40s,gpu-a100,gpu-rtx6000,gpu-v100
#SBATCH --mem=120G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/nlchain_refined_horizon_zeroth_wide_%j.out
#SBATCH --error=logs/nlchain_refined_horizon_zeroth_wide_%j.err
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
    echo "       Create it first -- see run_refined_horizon_scaling.sbatch's PREREQUISITES header."
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

# Launch-overhead mitigation -- see run_refined_horizon_scaling.sbatch's
# comment for the full rationale (kernel-launch-bound workload, command
# buffers capture+replay the launch sequence as a CUDA graph).
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_command_buffer=FUSION,CUBLAS,CUSTOM_CALL"

# ---------------------------------------------------------------------------
# WIDE zeroth-order config: 200k restarts, num_iters=0.
# ---------------------------------------------------------------------------
export NLCHAIN_NUM_RESTARTS=200000
export NLCHAIN_NUM_ITERS=0
export NLCHAIN_CSV=refined_horizon_scaling_zeroth_wide.csv
"${PY}" -c "import jax; print('JAX devices:', jax.devices())"

cd "${EXAMPLE_DIR}"

echo ""
echo "================================================================"
echo "  REFINED planning-horizon runtime/RAM/VRAM sweep (zeroth order, wide)"
echo "  Started: $(date)"
echo "================================================================"
"${PY}" refined_horizon_scaling.py
echo "Refined horizon sweep complete at $(date)"
echo "CSV written to: ${EXAMPLE_DIR}/refined_horizon_scaling_zeroth_wide.csv"
