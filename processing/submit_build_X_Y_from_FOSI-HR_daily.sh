#!/bin/bash
# ==============================================================
# PBS batch submission script for build_X_Y_from_FOSI-HR_daily.py on Casper.
#
# Builds the daily FOSI_BGC HR (JRA55-forced t13) X/Y perfect-model pair:
# X_FOSI_HR_JRA55_daily_{interp,avg}.nc / Y_FOSI_HR_JRA55_daily.nc in
# /glade/derecho/scratch/skygale/Downscaling_Data. Deliberately a distinct
# run_name/filename from the existing monthly X_FOSI_HR_JRA55_*.nc /
# Y_FOSI_HR_JRA55.nc, which functions_engressnet.py's DEFAULT_X_PATH/
# DEFAULT_Y_PATH currently point at -- this job does NOT overwrite that live
# training data; swap the default paths yourself once you want to train on
# the daily version. Pure CPU/memory regridding work (xESMF), no GPU needed.
#
# Submit with:  qsub submit_build_X_Y_from_FOSI-HR_daily.sh
# Check status: qstat -u $USER
# ==============================================================

#PBS -N build_FOSI_HR_daily
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

python build_X_Y_from_FOSI-HR_daily.py

echo "Job finished at $(date)"
