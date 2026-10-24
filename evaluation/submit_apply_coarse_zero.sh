#!/bin/bash
# Apply the coarse-zero rule to saved eval outputs (apply_coarse_zero_fields.py; originals backed up to scratch first).
# Submit with: qsub -v BATCH_DIRS="results/A results/B" [-l select=1:ncpus=2:mem=400GB] submit_apply_coarse_zero.sh
#PBS -N apply_cz
#PBS -A P93300065
#PBS -q casper
#PBS -l select=1:ncpus=2:mem=128GB
#PBS -l walltime=12:00:00
#PBS -j oe
#PBS -o .logs/

set -euo pipefail
BATCH_DIRS="${BATCH_DIRS:?Must set BATCH_DIRS}"
module load conda
conda activate seaice-downscaling
cd "/glade/work/skygale/projects/SeaIceDownscaling"
python -u evaluation/apply_coarse_zero_fields.py $BATCH_DIRS
