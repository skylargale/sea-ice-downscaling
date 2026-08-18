#!/bin/bash
# ==============================================================
# PBS batch submission script for build_Y_MESA-HR_daily_conservative_2000_2020.py
# on Casper.
#
# Builds a conservative-regridded (instead of production bilinear) daily MESA Y file,
# scoped to 2000-2020 -- the harder MESA counterpart to
# submit_build_Y_FOSI-HR_daily_conservative_2000_2020.sh. Unlike the existing 2015-2021
# MESA conservative script (RCP8.5-only), this replicates the full HIST+RCP8.5 stitching
# pipeline with conservative regridding, since the standard grid's earliest window
# (2000-2004) starts before the 2006 HIST/RCP8.5 boundary. Sibling of
# submit_build_Y_MESA-HR_daily_conservative.sh (2015-2021 only) -- does not touch that
# script's output. Saves to Y_MESA_HR_daily_conservative_2000_2020.nc -- does NOT touch the
# production Y_MESA_HR_daily.nc. Pure CPU/memory regridding work (xESMF/pop_tools), no GPU.
#
# Resources scaled up from the 2015-2021 script (ncpus=8:mem=128GB, walltime=04:00:00):
# this processes both HIST (6 members, 2000-2005) and RCP8.5 (6 members, 2006-2020, ~2x
# the year-span of the 2015-2021 script) eras, plus the stitch step -- ncpus=8:mem=192GB,
# walltime=10:00:00.
#
# Submit with:  qsub submit_build_Y_MESA-HR_daily_conservative_2000_2020.sh
# Check status: qstat -u $USER
# ==============================================================

#PBS -N build_Y_cons_mesa_2000_2020
#PBS -A P93300065
#PBS -q casper
#PBS -l select=1:ncpus=8:mem=192GB
#PBS -l walltime=10:00:00
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

python build_Y_MESA-HR_daily_conservative_2000_2020.py

echo "Job finished at $(date)"
