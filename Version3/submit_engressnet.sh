#!/bin/bash
# ==============================================================
# PBS batch submission script for train_engressnet.py on Casper.
#
# Submit with:  qsub submit_engressnet.sh
# Try a different train/test split without editing this file:
#   qsub -v TRAIN_YEARS="1920-1950",TEST_YEARS="2020-2040" submit_engressnet.sh
# Check status: qstat -u $USER
# ==============================================================

#PBS -N engressnet_1980-2010_train_2020-2040_test
#PBS -A P93300065
#PBS -q casper
#PBS -l select=1:ncpus=16:ngpus=1:mem=64GB:gpu_type=v100
#PBS -l walltime=06:00:00
#PBS -j oe
#PBS -o engressnet_train.log
#PBS -m abe
#PBS -M skycgale@uw.edu

set -euo pipefail

# ==============================================================
# CONFIG -- each var can be overridden at submit time instead of editing
# this file, e.g.:
#   qsub -v TRAIN_YEARS="1920-1950",TEST_YEARS="2000-2020" submit_engressnet.sh
# ==============================================================

# Years used for training / testing. Accepts "YYYY-YYYY" ranges and/or
# comma-separated years, e.g. "1980-2000" or "1980-1990,1995,2000-2005".
# Leave BOTH blank ("") to fall back to a random 80/20 train/test split.
#
# Historical data (X_perfmodexp*.nc) covers 1920-02 to 2006-01.
# SSP/future data (X_perfmodexp_SSP*.nc, spliced in below via USE_SSP)
# covers 2006-02 to 2101-01. Test years can therefore:
#   - land entirely in the future, e.g. TEST_YEARS="2020-2040", or
#   - straddle the 2006 boundary ("spliced" test data), e.g. TEST_YEARS="2000-2020"
#     (2000-2006 comes from the historical file, 2006-2020 from SSP).
TRAIN_YEARS="${TRAIN_YEARS:-1980-2010}"
TEST_YEARS="${TEST_YEARS:-2020-2040}"

# Splice the SSP/future continuation onto the historical time axis before
# splitting (see DEFAULT_X_PATH_SSP/DEFAULT_Y_PATH_SSP in
# functions_engressnet.py). Set to false to train/test on historical data
# only (TEST_YEARS must then stay <= 2006 or the split will error/warn about
# missing years).
USE_SSP="${USE_SSP:-true}"
X_PATH_FUTURE="${X_PATH_FUTURE:-/glade/derecho/scratch/skygale/Downscaling_Data/Kivalina/X_perfmodexp_SSP_interp.nc}"
Y_PATH_FUTURE="${Y_PATH_FUTURE:-/glade/derecho/scratch/skygale/Downscaling_Data/Kivalina/Y_perfmodexp_SSP.nc}"

# "true"  -> sliding-window patch extraction (original behavior)
# "false" -> train directly on one lat/lon sub-domain, no tiling, and also
#            write out a domain-mean SIT time series for the test period
USE_PATCHES="${USE_PATCHES:-false}"

# Only used when USE_PATCHES=false. Must fall fully within the ML domain
# (lat 60 to 80, lon -190 to -140).
LAT_MIN="${LAT_MIN:-65}"
LAT_MAX="${LAT_MAX:-72}"
LON_MIN="${LON_MIN:--170}"
LON_MAX="${LON_MAX:--155}"

# ==============================================================

echo "Job started on $(hostname) at $(date)"
echo "PBS_JOBID: ${PBS_JOBID:-not set}"
echo "Train years: ${TRAIN_YEARS:-<random split>}   Test years: ${TEST_YEARS:-<random split>}"
echo "USE_SSP: ${USE_SSP}"

module load conda
conda activate downscaling_env

cd "$PBS_O_WORKDIR"

ARGS=()
[ -n "$TRAIN_YEARS" ] && ARGS+=(--train-years "$TRAIN_YEARS")
[ -n "$TEST_YEARS" ] && ARGS+=(--test-years "$TEST_YEARS")

if [ "$USE_SSP" = true ]; then
    ARGS+=(--x-path-future "$X_PATH_FUTURE" --y-path-future "$Y_PATH_FUTURE")
fi

if [ "$USE_PATCHES" = true ]; then
    ARGS+=(--patches)
else
    ARGS+=(--no-patches --lat-min "$LAT_MIN" --lat-max "$LAT_MAX" --lon-min "$LON_MIN" --lon-max "$LON_MAX")
fi

python train_engressnet.py "${ARGS[@]}"

echo "Job finished at $(date)"
