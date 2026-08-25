#!/bin/bash
# ==============================================================
# submit_coastal_bias_experiments.sh
#
# Three single-axis tests of the three still-unresolved candidate causes of the coastal
# bias documented in the Version4 CLAUDE.md / project memory, all run against the SAME
# baseline config so they're directly comparable to each other and to the already-computed
# coastal_ch/classif/attn_end numbers:
#   FOSI, DATA_VARIANT=interp, medium domain (lat60-75/lon-182to-151),
#   TRAIN_YEARS=2015-2020, TEST_YEARS=2021, NUM_EPOCHS=20, K=20, K_EVAL=20, BETA=0.8
#   (== FOSI_el0_dmed_es0_2015-2020_2021, baseline Coastal RMSE: Stochastic UNet Mean
#   0.3978, Deterministic UNet 0.4396).
#
# Cause 1 -- truth-side regridding noise (land-threshold lets mostly-land cells count as
#   ocean, and bilinear regridding onto a mostly-land destination cell is noisy): tested
#   for real this time by actually training against a conservative-regridded Y (previous
#   test only re-evaluated an already-trained model against cleaner truth, which just
#   measures against a moved target). Needs processing/build_Y_FOSI-HR_daily_conservative.py
#   to run first (submitted here too, with the training job chained after it via
#   `qsub -W depend=afterok`) -- see that script for why it's scoped to 2015-2021 only.
#   Job: FOSI_consregrid (Y_PATH_OVERRIDE points at the new conservative Y file).
#
# Cause 1b -- land-mask threshold itself (0.1 vs. the original 0.5, independent of
#   regridding method): never re-tested before. Job: FOSI_landthresh05.
#
# Cause 2 -- decoder over-smoothing (bilinear-upsample `final_up`, chosen to avoid a
#   ConvTranspose2d checkerboard artifact): attn_end/coastal_ch only add signal on top of
#   the existing smoothing stage rather than replacing it. EnScaleNet actually replaces the
#   whole decoder's upsampling with per-location-adaptive LocallyConnected2d stages, so it's
#   the one architecture that attacks this directly -- previously only sensitivity-tested on
#   the monthly pipeline, never checked against coastal RMSE. Job: FOSI_enscalenet.
#
# Usage:
#   ./submit_coastal_bias_experiments.sh              # dry run: prints every qsub command
#   ./submit_coastal_bias_experiments.sh --submit      # actually submits everything
# ==============================================================

set -euo pipefail
cd "$(dirname "$0")"

SUBMIT=false
if [ "${1:-}" = "--submit" ]; then
    SUBMIT=true
fi

mkdir -p logs processing/logs

BATCH_NAME="FOSI_coastal_bias_experiments"
COMMON="BATCH_NAME=${BATCH_NAME},DATA_VARIANT=interp,TRAIN_YEARS=2015-2020,TEST_YEARS=2021"

n_jobs=0

echo "--- Data prep: conservative-regridded Y (2015-2021) ---"
echo "(cd processing && qsub submit_build_Y_FOSI-HR_daily_conservative.sh)"
DATA_JOB_ID=""
if [ "$SUBMIT" = true ]; then
    # cd into processing/ first so PBS_O_WORKDIR (and this script's own "cd
    # $PBS_O_WORKDIR") resolves there, not Version5/ root -- the python script and its
    # logs/ dir both live in processing/, not here.
    DATA_JOB_ID=$(cd processing && qsub submit_build_Y_FOSI-HR_daily_conservative.sh)
    echo "  -> ${DATA_JOB_ID}"
fi
n_jobs=$((n_jobs + 1))

echo ""
echo "--- Cause 1b: land-mask threshold 0.5 (vs. production default 0.1) ---"
echo "qsub -N FOSI_landthresh05 -v ${COMMON},LAND_THRESHOLD=0.5 submit_engressnet_daily.sh"
if [ "$SUBMIT" = true ]; then
    qsub -N "FOSI_landthresh05" -v "${COMMON},LAND_THRESHOLD=0.5" submit_engressnet_daily.sh
fi
n_jobs=$((n_jobs + 1))

echo ""
echo "--- Cause 2: EnScaleNet decoder (replaces bilinear-upsample final_up entirely) ---"
echo "qsub -N FOSI_enscalenet -v ${COMMON},ENSCALE_NET=true,NOISE_SIGMA=1.0 submit_engressnet_daily.sh"
if [ "$SUBMIT" = true ]; then
    qsub -N "FOSI_enscalenet" -v "${COMMON},ENSCALE_NET=true,NOISE_SIGMA=1.0" submit_engressnet_daily.sh
fi
n_jobs=$((n_jobs + 1))

echo ""
echo "--- Cause 1: conservative-regridded truth (chained after the data-prep job above) ---"
Y_OVERRIDE="/glade/derecho/scratch/skygale/Downscaling_Data/Y_FOSI_HR_JRA55_daily_conservative_2015_2021.nc"
if [ "$SUBMIT" = true ]; then
    echo "qsub -N FOSI_consregrid -W depend=afterok:${DATA_JOB_ID} -v ${COMMON},Y_PATH_OVERRIDE=${Y_OVERRIDE} submit_engressnet_daily.sh"
    qsub -N "FOSI_consregrid" -W "depend=afterok:${DATA_JOB_ID}" -v "${COMMON},Y_PATH_OVERRIDE=${Y_OVERRIDE}" submit_engressnet_daily.sh
else
    echo "qsub -N FOSI_consregrid -W depend=afterok:<data-prep-jobid> -v ${COMMON},Y_PATH_OVERRIDE=${Y_OVERRIDE} submit_engressnet_daily.sh"
fi
n_jobs=$((n_jobs + 1))

echo ""
echo "Total jobs: ${n_jobs} (1 data-prep + 3 training, one per candidate cause)"
if [ "$SUBMIT" = false ]; then
    echo "(dry run -- rerun with --submit to actually qsub these)"
fi
