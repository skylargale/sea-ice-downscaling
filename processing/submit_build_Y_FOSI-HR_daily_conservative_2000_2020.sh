#!/bin/bash
# ==============================================================
# PBS batch submission script for build_Y_FOSI-HR_daily_conservative_2000_2020.py
# on Casper.
#
# Builds a conservative-regridded (instead of the production bilinear) daily FOSI Y file,
# scoped to 2000-2020 (all 4 standard training windows + TEST_YEARS=2020), so the
# recommended architecture can be trained and compared against a matched bilinear baseline
# across the full standard grid, not just one split. Sibling of
# submit_build_Y_FOSI-HR_daily_conservative.sh (2015-2021 only) -- does not touch that
# script's output. Saves to Y_FOSI_HR_JRA55_daily_conservative_2000_2020.nc -- does NOT
# touch the production Y_FOSI_HR_JRA55_daily.nc. Pure CPU/memory regridding work (xESMF),
# no GPU needed.
#
# Walltime bumped to 06:00:00 (vs. the 2015-2021 script's 02:00:00): 21 years is ~3x the
# original 7-year scope, and per-file regridding scales with year count even though the
# regridder-build step itself is fixed-cost (and reused from cache here, since the
# 2015-2021 script already built weights for the same grids).
#
# Submit with:  qsub submit_build_Y_FOSI-HR_daily_conservative_2000_2020.sh
# Check status: qstat -u $USER
# ==============================================================

#PBS -N build_Y_cons_2000_2020
#PBS -A P93300065
#PBS -q casper
#PBS -l select=1:ncpus=8:mem=64GB
#PBS -l walltime=06:00:00
#PBS -j oe
#PBS -o logs/
#PBS -m abe
#PBS -M skycgale@uw.edu

set -euo pipefail

echo "Job started on $(hostname) at $(date)"
echo "PBS_JOBID: ${PBS_JOBID:-not set}"

module load conda
conda activate downscaling_env

cd "$PBS_O_WORKDIR"
mkdir -p logs

python build_Y_FOSI-HR_daily_conservative_2000_2020.py

echo "Job finished at $(date)"
