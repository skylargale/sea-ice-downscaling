#!/bin/bash
# ==============================================================
# PBS batch submission script for run_daily_eval_batch.py (Version4) on Casper --
# headless version of evaluation_plots_daily.ipynb's "Batch mode" section (section 16).
#
# CPU-only (no training, just loading eval_data/fields.npz and regenerating figures/tables),
# so unlike submit_engressnet*.sh this requests no GPU. Memory sized generously: the MESA
# daily-combo batches' fields.npz run up to ~4.4GB compressed, and preds_all_phys (the
# K_eval-member ensemble) decompresses much larger than that in memory -- single-run peaks of
# 15-25GB were observed interactively, so 128GB leaves ample headroom across a whole batch's
# runs processed serially (each run's arrays are freed before the next one loads).
#
# Submit with:  qsub -v BATCH_DIR="results/FOSI_daily_combo_avg" submit_daily_eval_batch.sh
# Check status: qstat -u $USER
# (normally invoked via submit_daily_eval_sweep.sh, which loops over every batch directory)
# ==============================================================

#PBS -N daily_eval
#PBS -A P93300065
#PBS -q casper
#PBS -l select=1:ncpus=4:mem=128GB
#PBS -l walltime=04:00:00
#PBS -j oe
#PBS -o logs/
#PBS -m abe
#PBS -M skycgale@uw.edu

set -euo pipefail

BATCH_DIR="${BATCH_DIR:?Must set BATCH_DIR, e.g. -v BATCH_DIR=results/FOSI_daily_combo_avg}"

module load conda
conda activate downscaling_env

# cd to wherever qsub was run from -- run_daily_eval_batch.py and its --save-root default
# ("saved_figs", relative) both need to resolve against the project root, so submit this
# from the same directory as run_daily_eval_batch.py (e.g. `cd evaluation/ && qsub ...`).
cd "$PBS_O_WORKDIR"

echo "Batch dir: $BATCH_DIR"
echo "Job started at $(date)"

python run_daily_eval_batch.py --batch-dir "$BATCH_DIR"

echo "Job finished at $(date)"
