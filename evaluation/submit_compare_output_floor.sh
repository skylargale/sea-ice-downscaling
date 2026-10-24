#!/bin/bash
# Per-regime comparison of --output-floor variants (compare_output_floor.py); optionally emails the table.
# Submit with: qsub -v BATCH_DIRS="results/A results/B",MAIL_TO="a@b" submit_compare_output_floor.sh
#PBS -N cmp_floor
#PBS -A P93300065
#PBS -q casper
#PBS -l select=1:ncpus=2:mem=48GB
#PBS -l walltime=02:00:00
#PBS -j oe
#PBS -o .logs/

set -uo pipefail
BATCH_DIRS="${BATCH_DIRS:?Must set BATCH_DIRS}"
MAIL_TO="${MAIL_TO:-}"
OUT="${OUT:-evaluation/corrected_calibration/output_floor_compare.txt}"
module load conda
conda activate seaice-downscaling
cd "/glade/work/skygale/projects/SeaIceDownscaling"
python -u evaluation/compare_output_floor.py $BATCH_DIRS 2>&1 | tee "$OUT"
for a in $MAIL_TO; do
  mailx -s "EngressNet regime comparison" "$a" < "$OUT"
done
