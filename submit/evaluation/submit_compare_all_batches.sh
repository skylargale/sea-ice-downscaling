#!/bin/bash
# ==============================================================
# PBS batch submission script for compare_all_batches.ipynb (Version4) on Casper
#
# Executes the notebook non-interactively via `jupyter nbconvert --execute
# --inplace` on a dedicated PBS-allocated node, so it isn't competing with
# other users' jobs for CPU/memory the way an interactive login-node shell
# is -- that contention is what made ad-hoc background runs of this
# notebook unreliable (workers/whole process killed partway through a
# 147-run member-metrics computation).
#
# Submit with:  qsub submit_compare_all_batches.sh
# Check status: qstat -u $USER
# Rerun after an interruption picks up where it left off: results are
# cached per-run in .member_metrics_cache.json (keyed by each run's
# eval_data/fields.npz mtime+size), so a second submission only
# (re)computes runs that changed or were never cached.
# ==============================================================

#PBS -N compare_all_batches
#PBS -A P93300065
#PBS -q casper
#PBS -l select=1:ncpus=8:mem=128GB
#PBS -l walltime=04:00:00
#PBS -j oe
#PBS -o logs/
#PBS -m abe
#PBS -M skycgale@uw.edu

# CPU-only (no ngpus/gpu_type): this notebook's metric computation
# (member_metrics.py) never moves tensors to GPU, so requesting a GPU
# chunk would only make the job wait longer to schedule for no benefit.

set -euo pipefail

module load conda
conda activate downscaling_env

cd "$PBS_O_WORKDIR"

# --ExecutePreprocessor.timeout=-1 disables nbconvert's own per-cell
# timeout (the member-metrics cell alone can legitimately run for a long
# time across 147 runs) -- PBS's walltime above is the real outer bound.
jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=-1 \
    compare_all_batches.ipynb

echo "Job finished at $(date)"
