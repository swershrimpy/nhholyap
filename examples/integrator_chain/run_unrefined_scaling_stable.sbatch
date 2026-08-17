#!/bin/bash
# =============================================================================
# run_unrefined_scaling_stable.sbatch -- single-node, single-V100 GPU job
# that runs the N=2..100 UNREFINED-ONLY sweep (unrefined_demo_stable.py) on
# PACE Phoenix.
#
# Companion to run_refinement_scaling_stable.sbatch, which times only the
# REFINED optimizer across N=2..100 (job 11915477, completed in 8h41m --
# see unrefined_demo_stable.py's module docstring for why that sweep never
# measured the unrefined path even though it also runs it every order).
# This job times ONLY the unrefined multistart optimizer instead, same
# N=2..100 range, same per-order resettable host-RAM peak and VRAM
# instrumentation, so the two jobs' logs are directly comparable point for
# point via plot_unrefined_runtime_scaling.py / plot_refinement_runtime_scaling.py.
#
# Like the refined sweep, this has no per-point subprocess isolation,
# watchdog, or CSV output -- it runs the whole sweep in one process and
# prints to stdout, so this job's log file is the only record of results.
# PYTHONUNBUFFERED=1 below keeps that log current line-by-line.
#
# sbatch detaches the job from your shell (fire-and-forget); you may log out
# while it runs.
#
# Resource sizing (see run_refinement_scaling_stable.sbatch for the node
# topology this is based on):
#   - Each gpu-v100 node has 24 CPUs / 191000MB RAM / 2 GPUs (confirmed via
#     `scontrol show node`), so a single GPU's share is ~12 CPUs / ~95GB.
#     --cpus-per-task=8 and --mem=80G both sit inside that share.
#   - --time=1-02:00:00 (26h) is the same budget the refined sweep used,
#     which completed in 8h41m; the unrefined optimizer has no refinement
#     loop (O(1) per step vs. the refined path's O(n_pairs)), so it is
#     expected to be cheaper, not more expensive, at every N -- this budget
#     has ample margin either way. Check `scontrol show reservation` before
#     submitting regardless: a walltime request whose worst case doesn't
#     fit before a maintenance window makes Slurm silently defer the job
#     until after maintenance ends rather than refuse it outright (this bit
#     the original, pre-fix refinement sweep once already).
#
# PREREQUISITES (on the login node, before submitting):
#   Create the conda env (Python 3.11, needed by immrax's linrax dependency)
#   and install immrax + pytest into it (same env as the other PACE jobs):
#     module load anaconda3
#     conda create -y -p <repo>/envs/immrax python=3.11
#     <repo>/envs/immrax/bin/python -m pip install "immrax[cuda]" pytest
#
# Override directives on the CLI (CLI flags beat #SBATCH lines):
#   sbatch -A gts-scoogan3-fy20phase3 --time=20:00:00 run_unrefined_scaling_stable.sbatch
#
# Env-var config (optional): CONDA_ENV=<repo>/envs/immrax
#
# IMPORTANT -- submit from the repo root (SLURM_SUBMIT_DIR becomes
# REPO_ROOT below), NOT from this script's own directory:
#   cd <repo_root> && sbatch examples/integrator_chain/run_unrefined_scaling_stable.sbatch
# Submitting from inside examples/integrator_chain/ instead makes REPO_ROOT
# resolve to that directory, so CONDA_ENV would look for a nonexistent
# examples/integrator_chain/envs/immrax and the job fails immediately (this
# happened once already on the refined sweep's first submission attempt).
# =============================================================================
#SBATCH --job-name=integrator-unrefined-scaling-stable
#SBATCH --account=gts-scoogan3-fy20phase3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=80G
#SBATCH --time=1-02:00:00
#SBATCH --output=logs/integrator_unrefined_scaling_stable_%j.out
#SBATCH --error=logs/integrator_unrefined_scaling_stable_%j.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
EXAMPLE_DIR="${REPO_ROOT}/examples/integrator_chain"
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

# Force the GPU backend, and keep stdout line-buffered so the log reflects
# progress in real time instead of only flushing in large blocks.
export JAX_PLATFORMS=cuda
export PYTHONUNBUFFERED=1
"${PY}" -c "import jax; print('JAX devices:', jax.devices())"

cd "${EXAMPLE_DIR}"

echo ""
echo "================================================================"
echo "  integrator_chain UNREFINED-only scaling sweep (N=2..100)"
echo "  Started: $(date)"
echo "================================================================"
"${PY}" unrefined_demo_stable.py
echo ""
echo "Unrefined scaling sweep complete at $(date)"
