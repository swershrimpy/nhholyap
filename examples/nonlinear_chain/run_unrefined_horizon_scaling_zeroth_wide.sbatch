#!/bin/bash
# =============================================================================
# run_unrefined_horizon_scaling_zeroth_wide.sbatch -- single-node, single-GPU
# (any available type) job that runs ONLY the unrefined-multistep PLANNING-
# HORIZON runtime/RAM/VRAM scaling sweep (unrefined_horizon_scaling.py),
# zeroth-order only, at a much wider multi-start budget than the other
# variants in this directory.
#
# Config vs. the other unrefined_horizon_scaling*.sbatch variants:
#   - NLCHAIN_NUM_RESTARTS=1200000 (v2 default: 256; bw_zeroth/bw_gd5: 4096)
#   - NLCHAIN_NUM_ITERS=0 (zeroth order only -- no gradient-descent variant
#     submitted alongside this one; see unrefined_horizon_scaling_bw_zeroth's
#     comment for why num_iters=0 takes a dedicated Python-level branch
#     rather than a zero-length loop)
#   - Horizon sweep ONLY -- this does not have a scenario-count-sweep
#     counterpart (unlike the bw_zeroth/bw_gd5/ms4096 families, which each
#     ship both).
#   - u0 is already sampled independently per restart AND per horizon step
#     (jax.random.normal(key, (num_restarts, num_segments, N)) in
#     optimize_multistep_gpu, nonlinear_chain_separating_input.py) -- there
#     is no "hold one constant value across every step" mode to disable here;
#     every step of every restart's candidate control sequence is an
#     independent draw already.
#   - Any GPU accepted (gpu-h100/l40s/a100/rtx6000/v100 partition list, no
#     --constraint), matching the pattern validated for the admire timing job
#     (examples/admire/run_diagnosis_timing.sbatch) after the V100-only
#     constraint there was found to just add queue wait with no other benefit.
#
# MEMORY WARNING: at num_segments=50 (the widest horizon point), the
# separation_loss_multistep intermediate (n=7 scenarios x num_segments x N=10,
# lower+upper, vmapped over 1.2M restarts) is on the order of 30+ GB of device
# memory for that one point alone -- likely to exceed VRAM on the smaller
# partitions in the list (v100/rtx6000) even though it is comfortable on
# h100/a100/l40s. This is expected and handled by the existing per-point
# subprocess isolation in unrefined_horizon_scaling.py: WORKER_TIMEOUT_S and
# WORKER_MEM_LIMIT_MB are math.inf (no self-imposed kill), so a real
# device-OOM on a large horizon point only fails THAT point (recorded as
# status='error' and flushed to CSV) -- smaller horizon points, which need
# much less memory, still complete and are never lost.
#
# PREREQUISITES / submission -- see run_unrefined_horizon_scaling.sbatch's
# header for the full explanation (conda env setup, why you must
# `cd <repo_root> && sbatch examples/nonlinear_chain/<this file>` rather than
# submitting from inside examples/nonlinear_chain/).
# =============================================================================
#SBATCH --job-name=nlchain-unrefined-horizon-zeroth-wide
#SBATCH --account=gts-scoogan3-fy20phase3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-h100,gpu-l40s,gpu-a100,gpu-rtx6000,gpu-v100
#SBATCH --mem=120G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/nlchain_unrefined_horizon_zeroth_wide_%j.out
#SBATCH --error=logs/nlchain_unrefined_horizon_zeroth_wide_%j.err
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
    echo "       Create it first -- see run_unrefined_horizon_scaling.sbatch's PREREQUISITES header."
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

# Launch-overhead mitigation -- see run_unrefined_horizon_scaling.sbatch's
# comment for the full rationale (kernel-launch-bound workload, command
# buffers capture+replay the launch sequence as a CUDA graph).
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_command_buffer=FUSION,CUBLAS,CUSTOM_CALL"

# ---------------------------------------------------------------------------
# WIDE zeroth-order config: 1.2M restarts, num_iters=0. See header MEMORY
# WARNING above -- large-horizon points may device-OOM on the smaller GPUs in
# the partition list; each point is an isolated subprocess so that only fails
# the one point, never the ones before it.
# ---------------------------------------------------------------------------
export NLCHAIN_NUM_RESTARTS=1200000
export NLCHAIN_NUM_ITERS=0
export NLCHAIN_CSV=unrefined_horizon_scaling_zeroth_wide.csv
"${PY}" -c "import jax; print('JAX devices:', jax.devices())"

cd "${EXAMPLE_DIR}"

echo ""
echo "================================================================"
echo "  UNREFINED planning-horizon runtime/RAM/VRAM sweep (zeroth order, wide)"
echo "  Started: $(date)"
echo "================================================================"
"${PY}" unrefined_horizon_scaling.py
echo "Unrefined horizon sweep complete at $(date)"
echo "CSV written to: ${EXAMPLE_DIR}/unrefined_horizon_scaling_zeroth_wide.csv"
