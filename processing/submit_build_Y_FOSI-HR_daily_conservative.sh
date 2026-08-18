#!/bin/bash
# ==============================================================
# PBS batch submission script for build_Y_FOSI-HR_daily_conservative.py on Casper.
#
# Builds a conservative-regridded (instead of the production bilinear) daily FOSI Y file,
# scoped to 2015-2021 only (train 2015-2020 + test 2021 -- the baseline used for the
# coastal_ch/classif/attn_end comparison), as a targeted test of coastal-bias cause 1
# (truth-side regridding noise) that actually gets used to train a model, unlike the earlier
# standalone re-evaluation-only test. Saves to
# Y_FOSI_HR_JRA55_daily_conservative_2015_2021.nc -- does NOT touch the production
# Y_FOSI_HR_JRA55_daily.nc. Pure CPU/memory regridding work (xESMF), no GPU needed.
#
# Submit with:  qsub submit_build_Y_FOSI-HR_daily_conservative.sh
# Check status: qstat -u $USER
# ==============================================================

#PBS -N build_Y_cons
#PBS -A P93300065
#PBS -q casper
#PBS -l select=1:ncpus=8:mem=64GB
#PBS -l walltime=02:00:00
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

python build_Y_FOSI-HR_daily_conservative.py

echo "Job finished at $(date)"
