#!/bin/bash
# ==============================================================
# PBS batch submission script for build_X_Y_from_MESA-HR_daily_rcp85.py on
# Casper.
#
# Builds the RCP8.5 daily MESA-HR (CESM-LE d651009) X/Y perfect-model pair,
# 2006-2021 subset only: X_MESA_HR_RCP85_daily_{interp,avg}.nc /
# Y_MESA_HR_RCP85_daily.nc in /glade/derecho/scratch/skygale/Downscaling_Data.
# This is the scenario continuation of the existing HIST daily files (which
# stop at 2006), needed so MESA daily training windows/test years past 2006
# (2010-2015, 2015-2020, test 2021) have data to draw on.
#
# Pure CPU/memory regridding work (xESMF + pop_tools), no GPU needed. Much
# smaller than the HIST build (6 members x ~16 years vs. 9 members x 86
# years) -- walltime/mem sized down accordingly but kept generous.
#
# Submit with:  qsub submit_build_X_Y_from_MESA-HR_daily_rcp85.sh
# Check status: qstat -u $USER
# ==============================================================

#PBS -N build_MESA_RCP85_daily
#PBS -A P93300065
#PBS -q casper
#PBS -l select=1:ncpus=8:mem=256GB
#PBS -l walltime=12:00:00
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

python build_X_Y_from_MESA-HR_daily_rcp85.py

echo "Job finished at $(date)"
