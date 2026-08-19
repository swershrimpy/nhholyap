#!/bin/bash
# =============================================================================
# run_diagnosis_timing.sbatch -- single-node, single-GPU job that runs
# admire_diagnosis_timing.py: unrefined-multistep and refined-multistep
# ADMIRE fault diagnosis, single config, small NUM_ITERS sweep, looking for
# an operating point under 50ms post-compile run time.
#
# GPU type is intentionally NOT pinned (no --constraint, generic
# --gres=gpu:1, and --partition lists several GPU partitions) -- the
# original target was a single 16GB V100, but at submission time every
# V100-16GB GPU on gpu-v100 was allocated (0 free; the partition's one
# free GPU was a V100-32GB) while gpu-h100 and gpu-l40s each had a fully
# idle 8-GPU node, so this instead lets Slurm start on whichever of these
# partitions can run it soonest. Any of these has more compute and VRAM
# than a 16GB V100, so this only makes the 50ms target easier to hit --
# but the resulting numbers are NOT directly comparable to a V100 figure;
# check the job's log for which GPU it actually landed on
# (`nvidia-smi --query-gpu=name` output near the top) before quoting a
# result as "on a V100."
#
# This is a MUCH lighter workload than run_success_rate_analysis.sbatch's
# 27-config x 20-restart x 100-iter batched sweep (that job needed
# --mem=150G and still saw 60GB+ host RAM on its cheapest combo -- see that
# script's own header). admire_diagnosis_timing.py batches only 1 config at
# NUM_RESTARTS=20 and at most 50 GD iterations, at the cheapest horizon
# (0.5s) in this folder. --mem=120G is a middle ground between the
# project's standard single-V100 allocation (80G, used by integrator_chain/
# nonlinear_chain) and the success-rate sweep's 150G: this workload is far
# smaller in config/restart/iter count, but a local smoke test (CPU, login
# node, num_restarts=2/num_iters=2 -- far smaller than what this job
# actually runs) still got OOM-killed around ~3GB RSS on the login node's
# own tight limits, so ADMIRE's compile-time memory cost is confirmed
# real at even trivial scale and not something to assume away without a
# margin. If this still OOMs, that's itself useful signal (see
# run_success_rate_analysis.sbatch's header for the same reasoning) --
# check the .err file for RESOURCE_EXHAUSTED (device) vs. a host OOM kill
# before just requesting more memory.
#
# --constraint=V100-16GB pins to the 16GB V100 variant specifically (not
# the 32GB one this partition also has) -- see run_success_rate_analysis.sbatch's
# header for why this partition's 16GB nodes are both the target hardware
# here and the more available ones to schedule on.
#
# sbatch detaches the job from your shell (fire-and-forget); you may log out
# while it runs.
#
# PREREQUISITES (on the login node, before submitting):
#   Create the conda env (Python 3.11, needed by immrax's linrax dependency)
#   and install immrax + pytest into it (same env as the other PACE jobs):
#     module load anaconda3
#     conda create -y -p <repo>/envs/immrax python=3.11
#     <repo>/envs/immrax/bin/python -m pip install "immrax[cuda]" pytest
#
# Override directives on the CLI (CLI flags beat #SBATCH lines):
#   sbatch -A gts-scoogan3-fy20phase3 --time=05:00:00 run_diagnosis_timing.sbatch
#
# Env-var config (optional): CONDA_ENV=<repo>/envs/immrax
#
# IMPORTANT -- submit from the repo root (SLURM_SUBMIT_DIR becomes
# REPO_ROOT below), NOT from this script's own directory:
#   cd <repo_root> && sbatch examples/admire/run_diagnosis_timing.sbatch
# Submitting from inside examples/admire/ instead makes REPO_ROOT resolve
# to that directory, so CONDA_ENV would look for a nonexistent
# examples/admire/envs/immrax and the job fails immediately (this has
# happened before in this project -- see integrator_chain's sbatch scripts).
# =============================================================================
#SBATCH --job-name=admire-diagnosis-timing
#SBATCH --account=gts-scoogan3-fy20phase3
#SBATCH --partition=gpu-h100,gpu-l40s,gpu-a100,gpu-rtx6000,gpu-v100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --time=03:00:00
#SBATCH --output=logs/admire_diagnosis_timing_%j.out
#SBATCH --error=logs/admire_diagnosis_timing_%j.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
EXAMPLE_DIR="${REPO_ROOT}/examples/admire"
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
echo "  admire unrefined vs. refined fault-diagnosis timing probe"
echo "  Started: $(date)"
echo "================================================================"
"${PY}" admire_diagnosis_timing.py
echo ""
echo "Diagnosis timing probe complete at $(date)"
