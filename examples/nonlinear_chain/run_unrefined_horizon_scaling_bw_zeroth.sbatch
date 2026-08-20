#!/bin/bash
# =============================================================================
# run_unrefined_horizon_scaling.sbatch -- single-node, single-V100 GPU job
# that runs ONLY the unrefined-multistep PLANNING-HORIZON runtime/RAM/VRAM
# scaling sweep (unrefined_horizon_scaling.py) on PACE Phoenix.
#
# Companion to run_unrefined_scaling.sbatch (which sweeps scenario count at
# a fixed horizon); this one sweeps horizon at a fixed scenario count. Submit
# alongside run_refined_horizon_scaling.sbatch to run both sweeps
# concurrently on two separate GPU allocations instead of sequentially in
# one job:
#   mkdir -p logs
#   sbatch run_unrefined_horizon_scaling.sbatch
#   sbatch run_refined_horizon_scaling.sbatch
#
# sbatch detaches the job from your shell (fire-and-forget); you may log out
# while it runs.
#
# Resource sizing rationale (see gpu-v100 partition on this cluster):
#   - Each gpu-v100 node has 24 CPUs / 191000MB RAM / 2 GPUs (confirmed via
#     `scontrol show node`), so a single GPU's share is ~12 CPUs / ~95GB.
#     --cpus-per-task=8 and --mem=80G both sit inside that share, leaving
#     room for a co-tenant job on the node's other GPU.
#   - The scaling script's own per-point worker watchdog
#     (WORKER_TIMEOUT_S / WORKER_MEM_LIMIT_MB in unrefined_horizon_scaling.py)
#     is intentionally disabled (math.inf), same as the scenario-count
#     sweeps -- every horizon point is left to run to completion, however
#     long that turns out to be, instead of being killed by an arbitrary
#     bound. --mem=80G raises the real ceiling well above what any point is
#     expected to need (see that script's docstring: the combinatorial risk
#     that motivated the watchdog was scenario PAIRS, fixed here at
#     C(7,2)=21, not horizon length), so it's the genuine OS/cgroup OOM
#     killer (not us) that would end a point early -- and each completed
#     point's CSV row is flushed to disk immediately (append_csv_row), so
#     that -- or a walltime kill of this whole job -- can only ever lose the
#     ONE point that was in flight, never the ones already completed.
#   - --time=3-00:00:00 is this partition's max wall-clock (`sinfo`) -- for
#     the same reason: no artificial time box on how long the sweep may take.
#
# PREREQUISITES (on the login node, before submitting):
#   Create the conda env (Python 3.11, needed by immrax's linrax dependency)
#   and install immrax + pytest into it:
#     module load anaconda3
#     conda create -y -p <repo>/envs/immrax python=3.11
#     <repo>/envs/immrax/bin/python -m pip install "immrax[cuda]" pytest
#
# Override directives on the CLI (CLI flags beat #SBATCH lines):
#   sbatch -A gts-scoogan3-fy20phase3 --time=05:00:00 run_unrefined_horizon_scaling.sbatch
#
# Env-var config (optional): CONDA_ENV=<repo>/envs/immrax
#
# IMPORTANT -- submit from the repo root (SLURM_SUBMIT_DIR becomes
# REPO_ROOT below), NOT from this script's own directory:
#   cd <repo_root> && sbatch examples/nonlinear_chain/run_unrefined_horizon_scaling.sbatch
# Submitting from inside examples/nonlinear_chain/ instead makes REPO_ROOT
# resolve to that directory, so CONDA_ENV would look for a nonexistent
# examples/nonlinear_chain/envs/immrax and the job fails immediately.
# =============================================================================
#SBATCH --job-name=nlchain-unrefined-horizon-bw-zeroth
#SBATCH --account=gts-scoogan3-fy20phase3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:rtx_pro_6000_blackwell:1
#SBATCH --partition=gpu-rtxpro-blackwell
#SBATCH --mem=160G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/nlchain_unrefined_horizon_bw_zeroth_%j.out
#SBATCH --error=logs/nlchain_unrefined_horizon_bw_zeroth_%j.err
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

# ---------------------------------------------------------------------------
# Blackwell (RTX Pro 6000) variant. Checked before writing this: the account
# reaches gpu-rtxpro-blackwell via the pace-phoenix-slurm group, and the
# installed jaxlib 0.6.2 carries sm_120/sm_120a targets, so the backend has
# real Blackwell codegen rather than a PTX fallback. The partition's three
# nodes are 64 CPUs / ~2TB / 8 GPUs each, so 8 CPUs and 160G is one GPU's
# share with room for co-tenants. Picked over gpu-v100 because that partition
# had 2 free GPUs against ~134 pending jobs while this one had 24 idle.
#
# The larger per-GPU memory also defuses the 4096-restart concern that applies
# on a 32GB V100 (see the ms4096 scripts).
# ---------------------------------------------------------------------------

# Launch-overhead mitigation. This workload is a long sequence of very small
# kernels -- a propagation step touches a couple of kilobytes at these batch
# sizes -- so wall time is dominated by per-kernel launch overhead rather
# than by arithmetic. Command buffers capture the launch sequence into a CUDA
# graph and replay it, which is exactly that case. Verified present in the
# installed jaxlib 0.6.2 (unknown XLA flags are rejected, so this is checked
# rather than assumed); FUSION is the one that matters here, the other two
# are harmless and cover the occasional gemm/custom call.
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_command_buffer=FUSION,CUBLAS,CUSTOM_CALL"

# ---------------------------------------------------------------------------
# ZEROTH ORDER: no gradient descent at all. num_iters=0 takes a Python-level
# branch in the optimiser that scores the 4096-sample random control cloud and
# returns the best feasible one, with no reverse-mode graph built anywhere --
# a zero-length lax.scan would still have traced and compiled one. This is the
# baseline the gradient-based runs have to beat: same multi-start budget, zero
# gradient work, so the difference isolates what the descent actually buys.
export NLCHAIN_NUM_RESTARTS=4096
export NLCHAIN_NUM_ITERS=0
export NLCHAIN_CSV=unrefined_horizon_scaling_bw_zeroth.csv
"${PY}" -c "import jax; print('JAX devices:', jax.devices())"

cd "${EXAMPLE_DIR}"

echo ""
echo "================================================================"
echo "  UNREFINED planning-horizon runtime/RAM/VRAM sweep"
echo "  Started: $(date)"
echo "================================================================"
"${PY}" unrefined_horizon_scaling.py
echo "Unrefined horizon sweep complete at $(date)"
echo "CSV written to: ${EXAMPLE_DIR}/unrefined_horizon_scaling_bw_zeroth.csv"
