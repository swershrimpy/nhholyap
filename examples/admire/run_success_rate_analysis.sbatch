#!/bin/bash
# =============================================================================
# run_success_rate_analysis.sbatch -- single-node, single-V100 GPU job that
# runs ONE method of admire_success_rate_analysis.py (all 3 horizons:
# 0.5s/1.0s/2.0s) on PACE Phoenix.
#
# Submit this three times, once per METHOD value, to run all three methods
# concurrently on three separate GPU allocations instead of one sequential
# `--method all` job. --job-name on the CLI (not just --export) so %x in the
# log filename pattern below actually distinguishes the three jobs' logs:
#   mkdir -p logs
#   sbatch --export=METHOD=single_step         --job-name=admire-single_step         run_success_rate_analysis.sbatch
#   sbatch --export=METHOD=multistep_unrefined --job-name=admire-multistep_unrefined run_success_rate_analysis.sbatch
#   sbatch --export=METHOD=multistep_refined   --job-name=admire-multistep_refined   run_success_rate_analysis.sbatch
#
# RESOURCE ESTIMATE -- see the module docstring of admire_success_rate_analysis.py
# ------------------------------------------------------------------------------
# The script's own docstring flags real risk here: ADMIRE's dynamics divide
# 5x per Euler step, the same category of operation that made
# quadrotor_fault_diagnosis's gradient compile go from ~2s/1 step to
# ~56s/4.3GB at 5 steps for a SINGLE (non-batched) config -- and this script
# double-vmaps 9 configs x 20 restarts = 180 instances through that same
# kind of gradient, at up to 20 steps (4x the quadrotor precedent), all
# within one jax.jit per (method, horizon).
#
# An initial probe (all 4 heaviest (method, horizon) combos run back-to-back
# in one long-lived process, sharing one JAX allocator/compilation cache)
# was OOM-killed at --mem=40G. A follow-up probe isolating each combo into
# its own subprocess (so nothing carries over between them) was still
# running past 7 minutes on the *cheapest* combo (single_step, 0.5s/5
# steps) with --mem=100G, without having OOM'd or finished -- i.e. even the
# lightest case is expensive, consistent with the docstring's own warning,
# and we do not have a clean per-combo peak number in hand.
#
# Given that and the hard Tuesday-maintenance deadline (no time left to
# probe further), this requests a generous-but-not-exhaustive ceiling
# instead of an evidence-backed exact number:
#   --mem=150G            -- well above the 40G that already failed, and
#                             within what gpu-v100 nodes have on hand (node
#                             RAM on this partition ranges from ~191GB to
#                             ~772GB per `scontrol show node`; 150G fits
#                             even the smaller nodes with room to spare).
#   --constraint=V100-16GB -- pins to the 16GB V100 variant, NOT the 32GB
#                             one. This partition has both, tagged as
#                             distinct Slurm features, but they're not
#                             equally available: at submission time, 23/30
#                             16GB nodes were idle/free vs. only 2/10 32GB
#                             nodes (`sinfo`), and an initial attempt
#                             requesting V100-32GB queued for 10+ minutes
#                             with no start in sight -- a real cost against
#                             the maintenance deadline. A live probe on a
#                             16GB node showed 76% VRAM used (12.4/16GB) for
#                             just the single cheapest (method, horizon)
#                             combo (single_step, 5 steps) -- high, but not
#                             yet OOM -- so 16GB is a calculated bet in
#                             favor of actually starting soon over a
#                             theoretically-safer card that isn't
#                             available. If a specific (method, horizon)
#                             combo genuinely exceeds 16GB (a JAX/XLA
#                             RESOURCE_EXHAUSTED error, not a host OOM --
#                             see the error text in the job's .err file to
#                             tell them apart), the fix is either request
#                             V100-32GB for just that combo, or shard its
#                             batch dimension across 2 GPUs via
#                             jax.sharding -- not attempted preemptively
#                             here since the dominant number actually
#                             measured so far (60GB+ HOST ram for the
#                             cheapest combo, still climbing) points at
#                             compile-time host memory, not device VRAM, as
#                             the more likely first failure mode, and
#                             sharding across devices doesn't reduce that.
#   --time=1-02:00:00 (26h) -- see TIME BUDGET below.
#
# If a job still OOMs or times out at these settings, that itself is
# useful signal that this sweep needs to be restructured (e.g. split by
# horizon too, or reduce NUM_RESTARTS/batch the vmap) rather than just
# resourced harder -- see this job's log before requesting more.
#
# TIME BUDGET -- must finish before Tuesday 2026-08-11 06:00 EDT
# ------------------------------------------------------------------
# Cluster-wide maintenance reservation (md-26-08, MAINT flag, effectively
# all nodes) runs 2026-08-11 06:00 to 2026-08-16 06:00
# (`scontrol show reservation`). Requesting this partition's normal max
# walltime (3-00:00:00) would make Slurm defer the job's start to AFTER
# maintenance ends instead of running it now (its 3-day worst case doesn't
# fit in the gap before the reservation) -- exactly what happened on an
# earlier, unrelated job this session. --time=1-02:00:00 fits inside the
# ~30h gap between submission and the reservation start with several hours
# of margin for Slurm scheduling / node startup delay.
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
# Env-var config: METHOD (required -- one of single_step, multistep_unrefined,
#                 multistep_refined, all), CONDA_ENV=<repo>/envs/immrax (optional)
# =============================================================================
#SBATCH --job-name=admire-success-rate
#SBATCH --account=gts-scoogan3-fy20phase3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:v100:1
#SBATCH --constraint=V100-16GB
#SBATCH --mem=150G
#SBATCH --time=1-02:00:00
#SBATCH --output=logs/admire_success_rate_%x_%j.out
#SBATCH --error=logs/admire_success_rate_%x_%j.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail

if [ -z "${METHOD:-}" ]; then
    echo "ERROR: METHOD env var not set. Submit with e.g.:"
    echo "  sbatch --export=METHOD=single_step run_success_rate_analysis.sbatch"
    exit 1
fi
case "${METHOD}" in
    single_step|multistep_unrefined|multistep_refined|all) ;;
    *) echo "ERROR: METHOD='${METHOD}' not one of single_step, multistep_unrefined, multistep_refined, all"; exit 1 ;;
esac

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
EXAMPLE_DIR="${REPO_ROOT}/examples/admire"
cd "${REPO_ROOT}"
mkdir -p logs

echo "=== Slurm job ${SLURM_JOB_ID:-local} (METHOD=${METHOD}) on $(hostname) at $(date) ==="
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
echo "  admire success-rate analysis -- method=${METHOD}, all horizons"
echo "  Started: $(date)"
echo "================================================================"
"${PY}" admire_success_rate_analysis.py --method "${METHOD}"
echo ""
echo "admire success-rate analysis (method=${METHOD}) complete at $(date)"
