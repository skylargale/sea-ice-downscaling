#!/bin/bash
# ==============================================================
# PBS batch submission script for evaluation/build_4window_aggregate_figures.py --
# pools the 4-window batch's raw per-cell data into single 4-window-aggregate
# distributional figures (rank histogram, reliability diagram, spread-skill, CRPS, PSD).
# CPU-only, no GPU. Mem sized like submit_daily_eval_batch.sh -- this loads one window's
# full fields.npz (preds_all_phys etc, ~4-5GB decompressed) at a time, not all 4 at once,
# but headroom matters since this was OOM-killed running on a login node with no PBS
# memory guarantee at all.
#
# Submit with:
#   qsub -v BATCH_DIR="results/MESA_stochastic_refine_sweep_avg_sharedbias" submit_4window_aggregate_figures.sh
# ==============================================================

#PBS -N agg_4window
#PBS -A P93300065
#PBS -q casper
#PBS -l select=1:ncpus=4:mem=192GB
#PBS -l walltime=01:00:00
#PBS -j oe
#PBS -o logs/
#PBS -m abe
#PBS -M skycgale@uw.edu

set -euo pipefail

BATCH_DIR="${BATCH_DIR:?Must set BATCH_DIR, e.g. -v BATCH_DIR=results/MESA_stochastic_refine_sweep_avg_sharedbias}"

module load conda
conda activate downscaling_env

# Hardcoded absolute path -- see submit_daily_eval_batch.sh's matching comment for why
# (PBS spools a submitted script at qsub time; BASH_SOURCE/$PBS_O_WORKDIR-based resolution
# is fragile across invocation conventions).
cd "/glade/work/skygale/projects/SeaIceDownscaling/Version5"

BATCH_NAME="$(basename "$BATCH_DIR")"
SAVE_DIR="saved_figs/${BATCH_NAME}/_4window_aggregate"

echo "Batch dir: $BATCH_DIR"
echo "Save dir: $SAVE_DIR"
echo "Job started at $(date)"

python -u evaluation/build_4window_aggregate_figures.py --batch-dir "$BATCH_DIR" --save-dir "$SAVE_DIR"

echo "Job finished at $(date)"
