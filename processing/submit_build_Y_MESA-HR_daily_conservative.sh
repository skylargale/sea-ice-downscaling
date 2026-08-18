#!/bin/bash
# ==============================================================
# PBS batch submission script for build_Y_MESA-HR_daily_conservative.py on Casper.
#
# MESA twin of submit_build_Y_FOSI-HR_daily_conservative.sh -- builds a conservative-
# regridded (instead of production bilinear) daily MESA Y file, scoped to 2015-2021,
# 6-member ensemble (matching the production Y_MESA_HR_daily.nc structure). More memory/
# time than the FOSI version since each of the 6 RCP8.5 members' ice files come in
# multi-year chunks covering more than the needed window (trimmed to 2015-2021 in-memory
# after regridding, not before). Pure CPU/memory regridding work (xESMF/pop_tools), no GPU.
#
# Submit with:  qsub submit_build_Y_MESA-HR_daily_conservative.sh
# Check status: qstat -u $USER
# ==============================================================

#PBS -N build_Y_cons_mesa
#PBS -A P93300065
#PBS -q casper
#PBS -l select=1:ncpus=8:mem=128GB
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

python build_Y_MESA-HR_daily_conservative.py

echo "Job finished at $(date)"
