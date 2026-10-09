#!/bin/bash
# ==============================================================
# PBS batch submission script for the 2021 PIOMAS observational-application
# grid build (base regrid + land-bleed fix), Version6 counterpart of
# Version5/processing/submit_build_X_Y_PIOMAS_obs_2020.sh.
#
# Runs both steps in sequence in one job:
#   1. build_X_Y_PIOMAS_obs_2021_daily.py           -> X/Y_PIOMAS_obs_2021_daily*.nc
#   2. build_X_Y_PIOMAS_obs_2021_daily_v2_landfixed.py -> X/Y_PIOMAS_obs_2021_daily_v2*.nc
# Step 2 requires step 1's output (OLD_X_PATH), so they must run in order --
# not parallel jobs.
#
# Pure CPU/memory regridding work (xESMF), no GPU needed.
#
# Submit with:  qsub submit_build_X_Y_PIOMAS_obs_2021.sh
# Check status: qstat -u $USER
# ==============================================================

#PBS -N build_PIOMAS_obs_2021
#PBS -A P93300065
#PBS -q casper
#PBS -l select=1:ncpus=8:mem=128GB
#PBS -l walltime=04:00:00
#PBS -j oe
#PBS -o .logs/
#PBS -m abe
#PBS -M skycgale@uw.edu

set -euo pipefail

echo "Job started on $(hostname) at $(date)"
echo "PBS_JOBID: ${PBS_JOBID:-not set}"

module load conda
conda activate seaice-downscaling

cd "$PBS_O_WORKDIR"
mkdir -p .logs

echo "=== Step 1: base daily build ==="
python build_X_Y_PIOMAS_obs_2021_daily.py

echo "=== Step 2: land-bleed fix ==="
python build_X_Y_PIOMAS_obs_2021_daily_v2_landfixed.py

echo "Job finished at $(date)"
