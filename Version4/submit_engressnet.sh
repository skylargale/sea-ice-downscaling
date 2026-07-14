#!/bin/bash
# ==============================================================
# PBS batch submission script for train_engressnet.py (Version4) on Casper.
#
# Submit with:  qsub submit_engressnet.sh
# Try a different train/test split without editing this file:
#   qsub -v TRAIN_YEARS="1958-1990",TEST_YEARS="1991-2005" submit_engressnet.sh
# Check status: qstat -u $USER
# ==============================================================

#PBS -N engressnet_fosi
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
#   qsub -v TRAIN_YEARS="1958-1980",TEST_YEARS="1981-1995" submit_engressnet.sh
# ==============================================================

# FOSI_BGC HR record covers 1958-2022 (single realization -- see
# 3d_models/build_X_Y_from_HR.ipynb). Accepts "YYYY-YYYY" ranges and/or
# comma-separated years, e.g. "1980-2000" or "1980-1990,1995,2000-2005".
# Leave BOTH blank ("") to fall back to a random 80/20 train/test split.
TRAIN_YEARS="${TRAIN_YEARS:-1958-2000}"
TEST_YEARS="${TEST_YEARS:-2001-2022}"

# "true"  -> sliding-window patch extraction
# "false" -> train directly on one lat/lon sub-domain (required for the
#            domain-mean SIT time series and the candidate-point
#            [Kivalina/Shishmaref/Kotzebue/Nome] time series -- both are
#            skipped under patches=True, see save_evaluation_data() in
#            functions_engressnet.py)
USE_PATCHES="${USE_PATCHES:-false}"

# Sub-domain covering all 4 candidate coastal communities (Kivalina,
# Shishmaref, Kotzebue, Nome), chosen so the 1-degree low-res crop is
# exactly 8x16 -- the UNet's 3 stride-2 pooling layers require both crop
# dims to be a multiple of 8 (and the resulting bottleneck to have more
# than 1 total spatial element, i.e. NOT both dims exactly 8 -- see
# extract_full_domain() in functions_engressnet.py). Must fall fully
# within the ML domain (lat 60-80, lon -190 to -140).
LAT_MIN="${LAT_MIN:-62}"
LAT_MAX="${LAT_MAX:-69}"
LON_MIN="${LON_MIN:--172}"
LON_MAX="${LON_MAX:--157}"

NUM_EPOCHS="${NUM_EPOCHS:-20}"

# ==============================================================

echo "Job started on $(hostname) at $(date)"
echo "PBS_JOBID: ${PBS_JOBID:-not set}"
echo "Train years: ${TRAIN_YEARS:-<random split>}   Test years: ${TEST_YEARS:-<random split>}"
echo "USE_PATCHES: ${USE_PATCHES}"

module load conda
conda activate downscaling_env

cd "$PBS_O_WORKDIR"

ARGS=(--num-epochs "$NUM_EPOCHS")
[ -n "$TRAIN_YEARS" ] && ARGS+=(--train-years "$TRAIN_YEARS")
[ -n "$TEST_YEARS" ] && ARGS+=(--test-years "$TEST_YEARS")

if [ "$USE_PATCHES" = true ]; then
    ARGS+=(--patches)
else
    ARGS+=(--no-patches --lat-min "$LAT_MIN" --lat-max "$LAT_MAX" --lon-min "$LON_MIN" --lon-max "$LON_MAX")
fi

python train_engressnet.py "${ARGS[@]}"

echo "Job finished at $(date)"
