#!/bin/bash
# =============================================================================
# run_refinement_scaling_stable.sbatch -- single-node, single-V100 GPU job
# that runs the numerically-stable N=2..100 refinement sweep
# (refinement_demo_stable.py) on PACE Phoenix.
#
# refinement_demo_stable.py is a nan-safe variant of refinement_demo.py --
# see its module docstring for the full diagnosis. Original run 11754504
# (refinement_demo.py, N=2..99) went `nan` on the refined optimizer's
# reported loss for every N from 65 through 99 (JAX's documented "nan from
# an unselected `where` branch" gotcha in propagate_with_refinement's
# per-pair refinement step); this script runs the fixed version instead,
# extended by one order to N=100 to also cover the point the original run
# didn't reach.
#
# Like the original, this has no per-point subprocess isolation, watchdog,
# or CSV output -- it runs the whole sweep in one process and prints to
# stdout, so this job's log file is the only record of results.
# PYTHONUNBUFFERED=1 below keeps that log current line-by-line.
#
# TIME BUDGET -- must finish before Tuesday 2026-08-11 06:00 EDT
# ------------------------------------------------------------------
# There's a cluster-wide maintenance reservation (md-26-08, MAINT flag,
# essentially all nodes) from 2026-08-11 06:00 to 2026-08-16 06:00
# (`scontrol show reservation`). Requesting this partition's normal max
# walltime (3-00:00:00) would make Slurm refuse to start the job at all
# right now (its 3-day worst case wouldn't fit before the reservation, so
# it would instead queue the job to start only *after* maintenance ends --
# this is exactly what happened on the first submission of the original,
# unpatched sweep). --time=1-02:00:00 (26h) below both (a) fits in the
# ~30h50m gap between submission time and the reservation start with ~5h
# of margin, so Slurm starts the job immediately instead of deferring it a
# week, and (b) is still >3x the original N=2..99 sweep's actual walltime
# (8h16m, run 11754504) despite covering N=100 too, so there's no reason to
# expect this margin to be tight.
#
# sbatch detaches the job from your shell (fire-and-forget); you may log out
# while it runs.
#
# Resource sizing (see run_refinement_scaling.sbatch / nonlinear_chain's
# run_*_scaling.sbatch for the node topology this is based on):
#   - Each gpu-v100 node has 24 CPUs / 191000MB RAM / 2 GPUs (confirmed via
#     `scontrol show node`), so a single GPU's share is ~12 CPUs / ~95GB.
#     --cpus-per-task=8 and --mem=80G both sit inside that share.
#
# PREREQUISITES (on the login node, before submitting):
#   Create the conda env (Python 3.11, needed by immrax's linrax dependency)
#   and install immrax + pytest into it (same env as the other PACE jobs):
#     module load anaconda3
#     conda create -y -p <repo>/envs/immrax python=3.11
#     <repo>/envs/immrax/bin/python -m pip install "immrax[cuda]" pytest
#
# Override directives on the CLI (CLI flags beat #SBATCH lines):
#   sbatch -A gts-scoogan3-fy20phase3 --time=20:00:00 run_refinement_scaling_stable.sbatch
#
# Env-var config (optional): CONDA_ENV=<repo>/envs/immrax
# =============================================================================
#SBATCH --job-name=integrator-refinement-scaling-stable
#SBATCH --account=gts-scoogan3-fy20phase3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=80G
#SBATCH --time=1-02:00:00
#SBATCH --output=logs/integrator_refinement_scaling_stable_%j.out
#SBATCH --error=logs/integrator_refinement_scaling_stable_%j.err
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
echo "  integrator_chain refinement scaling sweep, nan-safe (N=2..100)"
echo "  Started: $(date)"
echo "================================================================"
"${PY}" refinement_demo_stable.py
echo ""
echo "Refinement scaling sweep (stable) complete at $(date)"
