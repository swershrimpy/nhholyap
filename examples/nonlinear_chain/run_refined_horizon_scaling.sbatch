#!/bin/bash
# =============================================================================
# run_refined_horizon_scaling.sbatch -- single-node, single-V100 GPU job that
# runs ONLY the intersection-refinement PLANNING-HORIZON runtime/RAM/VRAM
# scaling sweep (refined_horizon_scaling.py) on PACE Phoenix.
#
# Companion to run_refined_scaling.sbatch (which sweeps scenario count at a
# fixed horizon); this one sweeps horizon at a fixed scenario count. Submit
# alongside run_unrefined_horizon_scaling.sbatch to run both sweeps
# concurrently on two separate GPU allocations instead of sequentially in
# one job:
#   mkdir -p logs
#   sbatch run_unrefined_horizon_scaling.sbatch
#   sbatch run_refined_horizon_scaling.sbatch
#
# sbatch detaches the job from your shell (fire-and-forget); you may log out
# while it runs.
#
# Resource sizing rationale -- see run_unrefined_horizon_scaling.sbatch's
# header for the full explanation (same partition, same --cpus-per-task=8 /
# --mem=80G / --time=3-00:00:00 reasoning, same math.inf per-point
# timeout/RSS watchdog, same incremental per-point CSV flush so a walltime
# kill or crash only ever loses the ONE horizon point in flight).
#
# PREREQUISITES (on the login node, before submitting):
#   Create the conda env (Python 3.11, needed by immrax's linrax dependency)
#   and install immrax + pytest into it:
#     module load anaconda3
#     conda create -y -p <repo>/envs/immrax python=3.11
#     <repo>/envs/immrax/bin/python -m pip install "immrax[cuda]" pytest
#
# Override directives on the CLI (CLI flags beat #SBATCH lines):
#   sbatch -A gts-scoogan3-fy20phase3 --time=05:00:00 run_refined_horizon_scaling.sbatch
#
# Env-var config (optional): CONDA_ENV=<repo>/envs/immrax
#
# IMPORTANT -- submit from the repo root (SLURM_SUBMIT_DIR becomes
# REPO_ROOT below), NOT from this script's own directory:
#   cd <repo_root> && sbatch examples/nonlinear_chain/run_refined_horizon_scaling.sbatch
# Submitting from inside examples/nonlinear_chain/ instead makes REPO_ROOT
# resolve to that directory, so CONDA_ENV would look for a nonexistent
# examples/nonlinear_chain/envs/immrax and the job fails immediately.
# =============================================================================
#SBATCH --job-name=nlchain-refined-horizon
#SBATCH --account=gts-scoogan3-fy20phase3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=80G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/nlchain_refined_horizon_%j.out
#SBATCH --error=logs/nlchain_refined_horizon_%j.err
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
    echo "       Create it first -- see this script's PREREQUISITES header."
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

# Launch-overhead mitigation. This workload is a long sequence of very small
# kernels -- a propagation step touches a couple of kilobytes at these batch
# sizes -- so wall time is dominated by per-kernel launch overhead rather
# than by arithmetic. Command buffers capture the launch sequence into a CUDA
# graph and replay it, which is exactly that case. Verified present in the
# installed jaxlib 0.6.2 (unknown XLA flags are rejected, so this is checked
# rather than assumed); FUSION is the one that matters here, the other two
# are harmless and cover the occasional gemm/custom call.
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_command_buffer=FUSION,CUBLAS,CUSTOM_CALL"
"${PY}" -c "import jax; print('JAX devices:', jax.devices())"

cd "${EXAMPLE_DIR}"

echo ""
echo "================================================================"
echo "  REFINED planning-horizon runtime/RAM/VRAM sweep"
echo "  Started: $(date)"
echo "================================================================"
"${PY}" refined_horizon_scaling.py
echo "Refined horizon sweep complete at $(date)"
echo "CSV written to: ${EXAMPLE_DIR}/refined_horizon_scaling.csv"
