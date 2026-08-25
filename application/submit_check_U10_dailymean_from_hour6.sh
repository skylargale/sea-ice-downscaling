#!/bin/bash
# ==============================================================
# submit_check_U10_dailymean_from_hour6.sh
#
# Runs build_check_U10_dailymean_from_hour6.py: builds a true daily-mean
# 10m wind speed channel (from hour_6 CAM U10 snapshots) for the 6 usable
# RCP8.5 MESA members over 2016-2020, and compares it against the existing
# daily-max U10 channel in X_MESA_HR_RCP85_daily_avg.nc -- vetting step
# before deciding whether to swap MESA's wind channel to a daily-mean
# convention consistent with PIOMAS/FOSI's JRA55 u_10/v_10.
#
# Pure CPU/memory work (no xesmf/pop_tools/GPU needed -- atm regridding
# uses the same plain bin-average function as the production MESA build
# scripts). Reads hour_6 files directly from campaign storage; no download.
#
# Submit with:  qsub submit_check_U10_dailymean_from_hour6.sh
# ==============================================================

#PBS -N check_U10_dailymean
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

python build_check_U10_dailymean_from_hour6.py

echo "Job finished at $(date)"
