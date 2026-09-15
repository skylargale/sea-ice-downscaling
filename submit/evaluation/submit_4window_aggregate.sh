#!/bin/bash
# ==============================================================
# PBS batch submission script for evaluation/build_4window_aggregate_figures.py --
# pools the 4 train-window runs in a batch (all evaluated on the same test=2021 period)
# into single 4-window-aggregate distributional-evaluation figures, rather than picking
# one window's figures arbitrarily for the paper. CPU-only, no GPU.
#
# Submit with:
#   qsub -v BATCH_DIR="results/MESA_stochastic_refine_sweep_avg_sharedbias" submit_4window_aggregate.sh
# ==============================================================

#PBS -N 4window_agg
#PBS -A P93300065
#PBS -q casper
#PBS -l select=1:ncpus=4:mem=64GB
#PBS -l walltime=00:30:00
#PBS -j oe
#PBS -o logs/
#PBS -m abe
#PBS -M skycgale@uw.edu

set -euo pipefail

BATCH_DIR="${BATCH_DIR:?Must set BATCH_DIR, e.g. -v BATCH_DIR=results/MESA_stochastic_refine_sweep_avg_sharedbias}"

module load conda
conda activate downscaling_env

# Hardcoded absolute path -- see submit_daily_eval_batch.sh's matching comment for why
# (PBS spools a submitted script at qsub time; BASH_SOURCE-based self-location breaks
# under real execution).
cd "/glade/work/skygale/projects/SeaIceDownscaling/Version5"

BATCH_NAME="$(basename "$BATCH_DIR")"
SAVE_DIR="saved_figs/${BATCH_NAME}/_4window_aggregate"

echo "Batch dir: $BATCH_DIR"
echo "Save dir: $SAVE_DIR"
echo "Job started at $(date)"

python evaluation/build_4window_aggregate_figures.py --batch-dir "$BATCH_DIR" --save-dir "$SAVE_DIR"

echo "Job finished at $(date)"
