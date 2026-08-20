#!/bin/bash
# =============================================================================
# run_refined_scaling.sbatch -- single-node, single-V100 GPU job that runs
# ONLY the intersection-refinement fault-scenario runtime/memory scaling
# sweep (refined_scenario_scaling.py) on PACE Phoenix.
#
# Submit this alongside run_unrefined_scaling.sbatch to run both sweeps
# concurrently on two separate GPU allocations instead of sequentially in
# one job:
#   mkdir -p logs
#   sbatch run_unrefined_scaling.sbatch
#   sbatch run_refined_scaling.sbatch
#
# sbatch detaches the job from your shell (fire-and-forget); you may log out
# while it runs.
#
# Resource sizing rationale (see gpu-v100 partition on this cluster):
#   - Each gpu-v100 node has 24 CPUs / 191000MB RAM / 2 GPUs (confirmed via
#     `scontrol show node`), so a single GPU's share is ~12 CPUs / ~95GB.
#     --cpus-per-task=8 and --mem=80G both sit inside that share, leaving
#     room for a co-tenant job on the node's other GPU -- e.g. the sibling
#     run_unrefined_scaling.sbatch job, if Slurm happens to pack both onto
#     the same node.
#   - The scaling script's own per-point worker watchdog
#     (WORKER_TIMEOUT_S / WORKER_MEM_LIMIT_MB in refined_scenario_scaling.py)
#     is intentionally disabled (math.inf) -- every scenario-count point is
#     left to run to completion, however long/large that turns out to be,
#     instead of being killed by an arbitrary bound. --mem=80G raises the
#     real ceiling well above what any point has been observed to need
#     (measured: the refined sweep's own 21-scenario point peaked at
#     ~14.4GB RSS), so it's the genuine OS/cgroup OOM killer (not us) that
#     would end a point early, and each completed point's CSV row is
#     flushed to disk immediately, so that can't lose earlier results.
#   - --time=3-00:00:00 is this partition's max wall-clock (`sinfo`) -- for
#     the same reason: no artificial time box on how long a point may take.
#
# PREREQUISITES (on the login node, before submitting):
#   Create the conda env (Python 3.11, needed by immrax's linrax dependency)
#   and install immrax + pytest into it:
#     module load anaconda3
#     conda create -y -p <repo>/envs/immrax python=3.11
#     <repo>/envs/immrax/bin/python -m pip install "immrax[cuda]" pytest
#
# Override directives on the CLI (CLI flags beat #SBATCH lines):
#   sbatch -A gts-scoogan3-fy20phase3 --time=05:00:00 run_refined_scaling.sbatch
#
# Env-var config (optional): CONDA_ENV=<repo>/envs/immrax
# =============================================================================
#SBATCH --job-name=nlchain-refined
#SBATCH --account=gts-scoogan3-fy20phase3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=80G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/nlchain_refined_%j.out
#SBATCH --error=logs/nlchain_refined_%j.err
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
echo "  REFINED (intersection-refinement) scenario-count runtime/memory sweep"
echo "  Started: $(date)"
echo "================================================================"
"${PY}" refined_scenario_scaling.py
echo "Refined sweep complete at $(date)"
echo "CSV written to: ${EXAMPLE_DIR}/refined_scenario_scaling_v2.csv"
