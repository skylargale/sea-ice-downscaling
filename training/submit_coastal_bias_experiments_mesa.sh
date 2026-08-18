#!/bin/bash
# ==============================================================
# submit_coastal_bias_experiments_mesa.sh
#
# MESA twin of submit_coastal_bias_experiments.sh -- same three single-axis tests of the
# three still-unresolved candidate causes of coastal bias, against the MESA baseline:
#   MESA, DATA_VARIANT=interp, medium domain (lat60-75/lon-182to-151),
#   TRAIN_YEARS=2015-2020, TEST_YEARS=2021, NUM_EPOCHS=20, K=20, K_EVAL=20, BETA=0.8
#   (== MESA_el0_dmed_es0_2015-2020_2021, from results/MESA_daily_combo_interp/).
#
# See submit_coastal_bias_experiments.sh for the full per-cause rationale (regridding
# noise / land-mask threshold / decoder over-smoothing). Differences here are purely
# MESA-specific plumbing:
#   - Data-prep job runs process_data/build_Y_MESA-HR_daily_conservative.py (6-member
#     ensemble, RCP8.5 files, needs pop_tools) instead of the FOSI version.
#   - Training jobs go through submit_engressnet_daily_mesa.sh instead of
#     submit_engressnet_daily.sh (256GB mem, since MESA's 6 ensemble members multiply the
#     per-timestep sample count ~6x vs. FOSI's single realization).
#
# Usage:
#   ./submit_coastal_bias_experiments_mesa.sh              # dry run
#   ./submit_coastal_bias_experiments_mesa.sh --submit      # actually submit
# ==============================================================

set -euo pipefail
cd "$(dirname "$0")"

SUBMIT=false
if [ "${1:-}" = "--submit" ]; then
    SUBMIT=true
fi

mkdir -p logs process_data/logs

BATCH_NAME="MESA_coastal_bias_experiments"
COMMON="BATCH_NAME=${BATCH_NAME},DATA_VARIANT=interp,TRAIN_YEARS=2015-2020,TEST_YEARS=2021"

n_jobs=0

echo "--- Data prep: conservative-regridded Y (2015-2021, 6-member ensemble) ---"
echo "(cd process_data && qsub submit_build_Y_MESA-HR_daily_conservative.sh)"
DATA_JOB_ID=""
if [ "$SUBMIT" = true ]; then
    DATA_JOB_ID=$(cd process_data && qsub submit_build_Y_MESA-HR_daily_conservative.sh)
    echo "  -> ${DATA_JOB_ID}"
fi
n_jobs=$((n_jobs + 1))

echo ""
echo "--- Cause 1b: land-mask threshold 0.5 (vs. production default 0.1) ---"
echo "qsub -N MESA_landthresh05 -v ${COMMON},LAND_THRESHOLD=0.5 submit_engressnet_daily_mesa.sh"
if [ "$SUBMIT" = true ]; then
    qsub -N "MESA_landthresh05" -v "${COMMON},LAND_THRESHOLD=0.5" submit_engressnet_daily_mesa.sh
fi
n_jobs=$((n_jobs + 1))

echo ""
echo "--- Cause 2: EnScaleNet decoder (replaces bilinear-upsample final_up entirely) ---"
echo "qsub -N MESA_enscalenet -v ${COMMON},ENSCALE_NET=true,NOISE_SIGMA=1.0 submit_engressnet_daily_mesa.sh"
if [ "$SUBMIT" = true ]; then
    qsub -N "MESA_enscalenet" -v "${COMMON},ENSCALE_NET=true,NOISE_SIGMA=1.0" submit_engressnet_daily_mesa.sh
fi
n_jobs=$((n_jobs + 1))

echo ""
echo "--- Cause 1: conservative-regridded truth (chained after the data-prep job above) ---"
Y_OVERRIDE="/glade/derecho/scratch/skygale/Downscaling_Data/Y_MESA_HR_daily_conservative_2015_2021.nc"
if [ "$SUBMIT" = true ]; then
    echo "qsub -N MESA_consregrid -W depend=afterok:${DATA_JOB_ID} -v ${COMMON},Y_PATH_OVERRIDE=${Y_OVERRIDE} submit_engressnet_daily_mesa.sh"
    qsub -N "MESA_consregrid" -W "depend=afterok:${DATA_JOB_ID}" -v "${COMMON},Y_PATH_OVERRIDE=${Y_OVERRIDE}" submit_engressnet_daily_mesa.sh
else
    echo "qsub -N MESA_consregrid -W depend=afterok:<data-prep-jobid> -v ${COMMON},Y_PATH_OVERRIDE=${Y_OVERRIDE} submit_engressnet_daily_mesa.sh"
fi
n_jobs=$((n_jobs + 1))

echo ""
echo "Total jobs: ${n_jobs} (1 data-prep + 3 training, one per candidate cause)"
if [ "$SUBMIT" = false ]; then
    echo "(dry run -- rerun with --submit to actually qsub these)"
fi
