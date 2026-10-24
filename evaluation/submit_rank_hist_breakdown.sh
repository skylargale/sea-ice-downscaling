#!/bin/bash
# Open-water/frazil CRPS breakdown (rank_hist_breakdown.py).
# Submit with: qsub -v BATCH_DIRS="results/A results/B" submit_open_water_crps_breakdown.sh
#PBS -N rank_bd
#PBS -A P93300065
#PBS -q casper
#PBS -l select=1:ncpus=2:mem=48GB
#PBS -l walltime=02:00:00
#PBS -j oe
#PBS -o .logs/

set -uo pipefail
BATCH_DIRS="${BATCH_DIRS:?Must set BATCH_DIRS}"
OUT="${OUT:-evaluation/corrected_calibration/rank_hist_breakdown.txt}"
module load conda
conda activate seaice-downscaling
cd "/glade/work/skygale/projects/SeaIceDownscaling"
PYTHONPATH=evaluation python -u evaluation/rank_hist_breakdown.py $BATCH_DIRS 2>&1 | tee "$OUT"
