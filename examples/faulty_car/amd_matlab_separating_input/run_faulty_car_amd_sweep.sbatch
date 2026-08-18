#!/bin/bash
# =============================================================================
# run_faulty_car_amd_sweep.sbatch -- Slurm JOB ARRAY that solves the exact
# AMD separating-input MILP (Ding et al. 2018, unmodified
# StateSpace.m / Exact_get_u_respon.m -- see amd_lib/, byte-identical to
# AMD-code's originals) for the faulty-nonholonomic-car 3-mode
# discrimination problem, once per sweep config, on PACE Phoenix.
#
# WHY A JOB ARRAY (not one long serial job)
# ------------------------------------------------------------------------
# A local run of this same MILP (workstation, 24-core i7-13700HX, Gurobi
# 13.0.1 academic) showed the solver's branch-and-bound gap stuck at 100%
# (zero measurable progress on the bound) for the full 60s test window on
# every config tried -- see amd_lib/Exact_get_u_respon.m's SOS-1
# reformulation (876 SOS constraints for this 3-mode/10-step instance).
# That means a much longer per-config budget is needed to find out whether
# these instances solve at all, and 84 configs x (say) 2h each would be
# ~168h serial -- past most partitions' walltime ceiling (the gpu-v100
# partition used elsewhere in this repo caps at 3-00:00:00 = 72h, see
# nonlinear_chain's run_unrefined_scaling.sbatch). A 1-84 job array instead
# gives EACH config its own independent wall-clock budget and lets Slurm
# schedule them concurrently across the cluster, so total wall time is
# bounded by the slowest single config, not the sum of all 84.
#
# A config that is killed at the time limit leaves no results/result_%03d.mat
# file -- the aggregation script (aggregate_success_rate.m) treats a missing
# file as a timeout/non-success, exactly like the local 60s sweep did.
#
# RESOURCE SIZING -- NOT YET VALIDATED ON PACE, adjust before relying on it
# ------------------------------------------------------------------------
#   --cpus-per-task=8   : Exact_get_u_respon.m's sdpsettings() does not set
#                         a Gurobi Threads limit (unmodified, per the task
#                         requirement to leave the algorithm/optimizer
#                         alone), so Gurobi auto-selects thread count from
#                         what it can see. On a shared PACE node under
#                         cgroups this should self-limit to the allocation,
#                         but that has NOT been confirmed on this cluster --
#                         watch the "Thread count" line in a job's .out for
#                         a mismatch (e.g. it reporting far more threads
#                         than --cpus-per-task) and lower --cpus-per-task or
#                         request an exclusive node if so.
#   --mem=16G            : the local run's single MATLAB+Gurobi process
#                         peaked under 3GB RSS on a much smaller instance
#                         count than some sweep configs may reach; 16G is
#                         headroom, not a measured ceiling.
#   --time (per task)    : set via PER_CONFIG_TIME below (default 02:00:00
#                         = 120x the local 60s smoke-test cap). Given the
#                         100%-gap behavior observed locally, there is no
#                         evidence yet that 2h is enough to prove optimality
#                         on any config -- treat the first array's success
#                         rate as a probe, not a final answer, and consider
#                         raising this (or accepting non-convergence as the
#                         real empirical result) based on what comes back.
#   --array=1-84%<N>     : the %<N> throttle (max concurrent array tasks)
#                         is intentionally left for you to set based on
#                         fair-share/QOS limits for this account -- an
#                         unthrottled array could request up to 84 x 8 =
#                         672 cores at once, which will just queue-wait
#                         rather than harm anything, but a throttle will
#                         likely start returning results sooner.
#
# PREREQUISITES (on the PACE login node, before submitting)
# ------------------------------------------------------------------------
# 1. Gurobi license: the workstation this was drafted on uses a
#    NODE-LOCKED academic license (HOSTID-tied) -- it will NOT work on a
#    PACE compute node. You need either:
#      a) PACE's own site Gurobi module/license (check `module avail gurobi`
#         and PACE's software pages), or
#      b) a fresh node-locked academic license requested for the specific
#         PACE login/compute-node hardware ID, or
#      c) a floating/named-user license reachable from PACE's network.
#    Set GRB_LICENSE_FILE / GUROBI_HOME accordingly below, or `module load`
#    whatever PACE provides.
# 2. MATLAB module: `module avail matlab` on PACE and update MATLAB_MODULE
#    below to match (this script assumes a module named "matlab").
# 3. YALMIP: not a PACE module. Clone it once into this repo, e.g.:
#      git clone https://github.com/yalmip/YALMIP.git thirdparty/YALMIP
#    and point YALMIP_DIR below at it (or override via --export).
# 4. Generate configs.mat once (from the login node, or any machine, since
#    it's pure MATLAB with no toolbox deps):
#      matlab -batch "generate_config_table"
#    (run from this directory) before submitting the array.
#
# SUBMISSION
# ------------------------------------------------------------------------
#   mkdir -p logs results
#   matlab -batch "generate_config_table"     # once, produces configs.mat
#   sbatch --array=1-84%12 run_faulty_car_amd_sweep.sbatch
#
# After the array completes (or partially completes):
#   matlab -batch "aggregate_success_rate"
# =============================================================================
#SBATCH --job-name=faulty-car-amd-sweep
#SBATCH --account=gts-scoogan3-fy20phase3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --array=1-84
#SBATCH --output=logs/config_%a_%A.out
#SBATCH --error=logs/config_%a_%A.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail

# Per-config safety cap, kept a few minutes under the #SBATCH --time above so
# MATLAB gets to save partial diagnostics before Slurm's own harder kill.
# Override with: sbatch --export=PER_CONFIG_TIMEOUT_S=6600 ...
PER_CONFIG_TIMEOUT_S="${PER_CONFIG_TIMEOUT_S:-7000}"   # ~1h56m, under a 2h #SBATCH --time

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
EXAMPLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${EXAMPLE_DIR}"
mkdir -p logs results

echo "=== Slurm array task ${SLURM_ARRAY_TASK_ID:-?} (job ${SLURM_ARRAY_JOB_ID:-local}) on $(hostname) at $(date) ==="
echo "EXAMPLE_DIR: ${EXAMPLE_DIR}"

if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
    echo "ERROR: this script must be submitted as a job array (--array=1-84)."
    exit 1
fi

if [ -f "results/result_$(printf '%03d' "${SLURM_ARRAY_TASK_ID}").mat" ]; then
    echo "Config ${SLURM_ARRAY_TASK_ID} already has a result file -- skipping (resume-safe)."
    exit 0
fi

# ---------------------------------------------------------------------------
# Environment setup -- MATLAB + YALMIP + Gurobi.
# ADAPT THESE THREE LINES to what's actually on PACE (see PREREQUISITES).
# ---------------------------------------------------------------------------
MATLAB_MODULE="${MATLAB_MODULE:-matlab}"
module load "${MATLAB_MODULE}"

export YALMIP_DIR="${YALMIP_DIR:-${REPO_ROOT}/thirdparty/YALMIP}"
export GUROBI_MATLAB_DIR="${GUROBI_MATLAB_DIR:-${GUROBI_HOME:-/opt/gurobi1301/linux64}/matlab}"

if [ ! -d "${YALMIP_DIR}" ]; then
    echo "ERROR: YALMIP_DIR=${YALMIP_DIR} not found -- see PREREQUISITES step 3."
    exit 1
fi
if [ ! -d "${GUROBI_MATLAB_DIR}" ]; then
    echo "ERROR: GUROBI_MATLAB_DIR=${GUROBI_MATLAB_DIR} not found -- see PREREQUISITES step 1-2."
    exit 1
fi
if [ ! -f "configs.mat" ]; then
    echo "ERROR: configs.mat not found -- run 'matlab -batch \"generate_config_table\"' first."
    exit 1
fi

echo "YALMIP_DIR: ${YALMIP_DIR}"
echo "GUROBI_MATLAB_DIR: ${GUROBI_MATLAB_DIR}"
echo "PER_CONFIG_TIMEOUT_S: ${PER_CONFIG_TIMEOUT_S}"

echo ""
echo "================================================================"
echo "  Config ${SLURM_ARRAY_TASK_ID}/84 -- exact AMD MILP solve"
echo "  Started: $(date)"
echo "================================================================"
timeout "${PER_CONFIG_TIMEOUT_S}" matlab -batch "run_one_config_pace(${SLURM_ARRAY_TASK_ID})"
rc=$?
if [ "$rc" -eq 124 ]; then
    echo "Config ${SLURM_ARRAY_TASK_ID} TIMED OUT after ${PER_CONFIG_TIMEOUT_S}s (no result file written -- counts as non-success)."
else
    echo "Config ${SLURM_ARRAY_TASK_ID} finished (matlab exit code ${rc}) at $(date)"
fi
