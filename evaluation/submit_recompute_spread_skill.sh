#!/bin/bash
# Recompute ocean-only spread-skill ratio / CRPS / deterministic MAE from saved fields.npz.
# Submit with: qsub -v BATCH_DIRS="results/A results/B" submit_recompute_spread_skill.sh
#PBS -N recompute_ssr
#PBS -A P93300065
#PBS -q casper
#PBS -l select=1:ncpus=2:mem=96GB
#PBS -l walltime=03:00:00
#PBS -j oe
#PBS -o .logs/

set -euo pipefail
BATCH_DIRS="${BATCH_DIRS:?Must set BATCH_DIRS}"
module load conda
conda activate seaice-downscaling
cd "/glade/work/skygale/projects/SeaIceDownscaling/Version6"
python -u evaluation/recompute_spread_skill.py --out-dir evaluation/corrected_calibration $BATCH_DIRS
