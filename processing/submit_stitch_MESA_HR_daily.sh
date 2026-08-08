#!/bin/bash
# ==============================================================
# PBS batch submission script for stitch_MESA_HR_daily_hist_rcp85.py on
# Casper. Run this AFTER build_X_Y_from_MESA-HR_daily_rcp85.py has finished
# and produced X_MESA_HR_RCP85_daily_{interp,avg}.nc / Y_MESA_HR_RCP85_daily.nc.
#
# Concatenates the existing HIST daily files with the new RCP8.5 files
# (6-member intersection, time >= 1999) into single continuous
# X_MESA_HR_daily_{interp,avg}.nc / Y_MESA_HR_daily.nc, ready for
# submit_engressnet_daily_mesa.sh.
#
# Submit with:  qsub submit_stitch_MESA_HR_daily.sh
# ==============================================================

#PBS -N stitch_MESA_HR_daily
#PBS -A P93300065
#PBS -q casper
#PBS -l select=1:ncpus=4:mem=256GB
#PBS -l walltime=04:00:00
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

python stitch_MESA_HR_daily_hist_rcp85.py

echo "Job finished at $(date)"
