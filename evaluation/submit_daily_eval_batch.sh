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
#PBS -o .logs/
#PBS -m abe
#PBS -M skycgale@uw.edu

set -euo pipefail

BATCH_DIR="${BATCH_DIR:?Must set BATCH_DIR, e.g. -v BATCH_DIR=results/FOSI_daily_combo_avg}"

module load conda
conda activate seaice-downscaling

# cd to the Version6 root (not $PBS_O_WORKDIR, which varies by invocation convention --
# see Version6/README.md): --save-root defaults to the relative path "evaluation/saved_figs",
# and this cwd-relative path is only correct if cwd is the Version6 root -- without this cd,
# a run invoked from elsewhere could silently create a stray saved_figs/ in the wrong place
# instead of writing into the real Version6/evaluation/saved_figs/. Hardcoded absolute path, not resolved relative to
# $PBS_O_WORKDIR or this script's own BASH_SOURCE location -- both were tried elsewhere in
# this project (submit_engressnet_daily*.sh) and BASH_SOURCE-based resolution silently
# broke under real PBS execution, since PBS spools/copies a submitted script at qsub time
# and BASH_SOURCE[0] then points at that spool copy, not the real file on disk.
cd "/glade/work/skygale/projects/SeaIceDownscaling/Version6"

echo "Batch dir: $BATCH_DIR"
echo "Job started at $(date)"

python evaluation/run_daily_eval_batch.py --batch-dir "$BATCH_DIR"

echo "Job finished at $(date)"
