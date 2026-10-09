#!/bin/bash
# ==============================================================
# Coastal-fill (v4) rebuild of the area-averaged PIOMAS-obs X for 2021 (2026-09-25).
# build_X_PIOMAS_obs_2021_daily_v3_avg.py with PIOMAS_COAST_FILL=1 -> X_PIOMAS_obs_<year>_daily_v4_avg_coastfill.nc.
# Y is unchanged (Y_PIOMAS_obs_*_daily_v2.nc).
# Needs each year's X_PIOMAS_obs_<year>_daily_v2_interp.nc (already built).
# Submit with: qsub submit_build_X_PIOMAS_obs_coastfill.sh
# ==============================================================
#PBS -N build_PIOMAS_coastfill
#PBS -A P93300065
#PBS -q casper
#PBS -l select=1:ncpus=8:mem=64GB
#PBS -l walltime=02:00:00
#PBS -j oe
#PBS -o .logs/
set -euo pipefail
module load conda
conda activate seaice-downscaling
cd /glade/work/skygale/projects/SeaIceDownscaling/processing
export PIOMAS_COAST_FILL=1
for year in 2021; do
    echo "=== ${year} ==="
    PIOMAS_YEAR=$year python build_X_PIOMAS_obs_2021_daily_v3_avg.py
done
echo "Job finished at $(date)"
