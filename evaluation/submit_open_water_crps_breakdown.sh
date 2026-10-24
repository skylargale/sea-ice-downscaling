#!/bin/bash
# Open-water/frazil CRPS breakdown (open_water_crps_breakdown.py).
# Submit with: qsub -v BATCH_DIRS="results/A results/B" submit_open_water_crps_breakdown.sh
#PBS -N ow_crps_bd
#PBS -A P93300065
#PBS -q casper
#PBS -l select=1:ncpus=2:mem=48GB
#PBS -l walltime=02:00:00
#PBS -j oe
#PBS -o .logs/

set -uo pipefail
BATCH_DIRS="${BATCH_DIRS:?Must set BATCH_DIRS}"
OUT="${OUT:-evaluation/corrected_calibration/open_water_crps_breakdown.txt}"
module load conda
conda activate seaice-downscaling
cd "/glade/work/skygale/projects/SeaIceDownscaling"
PYTHONPATH=evaluation python -u evaluation/open_water_crps_breakdown.py $BATCH_DIRS 2>&1 | tee "$OUT"
