#!/bin/bash
# =============================================================================
# run_refinement_scaling.sbatch -- single-node, single-V100 GPU job that runs
# integrator_chain's refinement scaling script (refinement_demo.py) on PACE
# Phoenix.
#
# refinement_demo.py sweeps integrator-chain order N=2..99, at each N
# optimizing both the unrefined and intersection-refinement multistep
# separating-input sequences and printing a loss/compile-time/run-time
# comparison -- see that file's module docstring. Unlike
# nonlinear_chain's *_scenario_scaling.py, this script has no per-point
# subprocess isolation, watchdog, or CSV output: it runs the whole N=2..99
# sweep in one process and prints to stdout, so this job's log file
# (logs/integrator_refinement_scaling_%j.out) is the only record of
# results -- there's no persisted-per-point row to fall back on if a later
# N fails/hangs. PYTHONUNBUFFERED=1 below keeps that log current line-by-
# line rather than only flushing in large blocks (see nlchain-scaling's
# equivalent buffering issue).
#
# sbatch detaches the job from your shell (fire-and-forget); you may log out
# while it runs.
#
# Resource sizing rationale (see gpu-v100 partition on this cluster, and
# examples/nonlinear_chain/run_*_scaling.sbatch for the node topology this
# is based on):
#   - Each gpu-v100 node has 24 CPUs / 191000MB RAM / 2 GPUs (confirmed via
#     `scontrol show node`), so a single GPU's share is ~12 CPUs / ~95GB.
#     --cpus-per-task=8 and --mem=80G both sit inside that share, leaving
#     room for a co-tenant job on the node's other GPU.
#   - --time=3-00:00:00 is this partition's max wall-clock (`sinfo`). No
#     artificial time box is imposed here either -- nonlinear_chain's
#     scenario-count sweep showed JIT compile time growing steeply as
#     problem size grows, and this script's N goes up to 99 with no
#     resume/checkpoint mechanism, so it needs all the wall-clock room the
#     partition allows to have a chance of reaching the end of the sweep.
#
# PREREQUISITES (on the login node, before submitting):
#   Create the conda env (Python 3.11, needed by immrax's linrax dependency)
#   and install immrax + pytest into it (same env as nonlinear_chain's jobs):
#     module load anaconda3
#     conda create -y -p <repo>/envs/immrax python=3.11
#     <repo>/envs/immrax/bin/python -m pip install "immrax[cuda]" pytest
#
# Override directives on the CLI (CLI flags beat #SBATCH lines):
#   sbatch -A gts-scoogan3-fy20phase3 --time=1-00:00:00 run_refinement_scaling.sbatch
#
# Env-var config (optional): CONDA_ENV=<repo>/envs/immrax
# =============================================================================
#SBATCH --job-name=integrator-refinement-scaling
#SBATCH --account=gts-scoogan3-fy20phase3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=80G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/integrator_refinement_scaling_%j.out
#SBATCH --error=logs/integrator_refinement_scaling_%j.err
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
echo "  integrator_chain refinement scaling sweep (N=2..99)"
echo "  Started: $(date)"
echo "================================================================"
"${PY}" refinement_demo.py
echo ""
echo "Refinement scaling sweep complete at $(date)"
