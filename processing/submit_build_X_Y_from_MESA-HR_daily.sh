#!/bin/bash
# ==============================================================
# PBS batch submission script for build_X_Y_from_MESA-HR_daily.py on Casper.
#
# Builds the daily MESA-HR (CESM-LE d651007/d651030) X/Y perfect-model pair:
# X_MESA_HR_HIST_daily_{interp,avg}.nc / Y_MESA_HR_HIST_daily.nc in
# /glade/derecho/scratch/skygale/Downscaling_Data. Pure CPU/memory regridding
# work (xESMF + pop_tools), no GPU needed.
#
# Submit with:  qsub submit_build_X_Y_from_MESA-HR_daily.sh
# Check status: qstat -u $USER
# ==============================================================

#PBS -N build_MESA_HR_daily
#PBS -A P93300065
#PBS -q casper
#PBS -l select=1:ncpus=64:mem=256GB
#PBS -l walltime=24:00:00
#PBS -j oe
#PBS -o logs/
#PBS -m abe
#PBS -M skycgale@uw.edu

# -o logs/ (trailing slash) keeps PBS's own default filename, routed into logs/
# instead of the process_data/ root. Resolved relative to the submission
# directory, so logs/ must exist there before qsub runs (created below too, in
# case this is submitted before the directory has been created interactively).

set -euo pipefail

echo "Job started on $(hostname) at $(date)"
echo "PBS_JOBID: ${PBS_JOBID:-not set}"

module load conda
conda activate downscaling_env

cd "$PBS_O_WORKDIR"
mkdir -p logs

python build_X_Y_from_MESA-HR_daily.py

echo "Job finished at $(date)"
