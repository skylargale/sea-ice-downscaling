#!/bin/bash
# ==============================================================
# submit_stochastic_refine_consregrid_2000_2020.sh
#
# Trains the recommended config (STOCHASTIC_REFINE=true, NOISE_SIGMA=1.0) against the new
# conservative-regridded truth (Y_FOSI_HR_JRA55_daily_conservative_2000_2020.nc, built by
# ../../processing/build_Y_FOSI-HR_daily_conservative_2000_2020.py) across the full
# standard grid, retraining (not just re-evaluating -- the earlier single-split
# re-evaluation-only test was inconclusive because it measured an already-bilinear-trained
# model against a moved target). Matched bilinear-truth baseline arm is
# submit_stochastic_refine_bilinear_2000_2020.sh (same windows/config, production Y).
#
# Windows/test-year match submit_daily_length_sweep.sh/submit_season_sweep.sh's convention
# (TEST_YEARS=2020, not 2021, so TRAIN_YEARS ends 2019 -- overlaps PIOMAS coverage), since
# that's what the conservative data was built to cover.
#
# Each job is CHAINED after the FOSI conservative-regrid data-prep job (qsub -W
# depend=afterok) so it only starts once that data actually exists -- pass the data-prep
# job's ID as $1, e.g.:
#   ./submit_stochastic_refine_consregrid_2000_2020.sh 5622474              # dry run
#   ./submit_stochastic_refine_consregrid_2000_2020.sh 5622474 --submit     # actually submit
#
# 4 jobs, into results/FOSI_stochastic_refine_consregrid_2000_2020/.
#
# MESA twin: submit_stochastic_refine_consregrid_2000_2020_mesa.sh.
# ==============================================================

set -euo pipefail
cd "$(dirname "$0")"

if [ -z "${1:-}" ]; then
    echo "Usage: $0 <data_prep_job_id> [--submit]" >&2
    exit 1
fi
DATA_JOB_ID="$1"

SUBMIT=false
if [ "${2:-}" = "--submit" ]; then
    SUBMIT=true
fi

mkdir -p logs

SPLITS=(
    "2000-2004"
    "2005-2009"
    "2010-2014"
    "2015-2019"
)
TEST_YEARS="2020"
BATCH_NAME="FOSI_stochastic_refine_consregrid_2000_2020"
Y_OVERRIDE="/glade/derecho/scratch/skygale/Downscaling_Data/Y_FOSI_HR_JRA55_daily_conservative_2000_2020.nc"

BASE_NUM_EPOCHS=20
BASE_K=20
BASE_K_EVAL=20
BASE_BETA=0.8
LAT_MIN=60
LAT_MAX=75
LON_MIN=-182
LON_MAX=-151

n_jobs=0

submit_job () {
    local job_name="$1"; shift
    local vlist="$1"; shift
    echo "qsub -N ${job_name} -W depend=afterok:${DATA_JOB_ID} -v ${vlist} submit_engressnet_daily.sh"
    if [ "$SUBMIT" = true ]; then
        qsub -N "${job_name}" -W "depend=afterok:${DATA_JOB_ID}" -v "${vlist}" submit_engressnet_daily.sh
    fi
    n_jobs=$((n_jobs + 1))
}

COMMON="BATCH_NAME=${BATCH_NAME},DATA_VARIANT=interp,TEST_YEARS=${TEST_YEARS},NUM_EPOCHS=${BASE_NUM_EPOCHS},K=${BASE_K},K_EVAL=${BASE_K_EVAL},BETA=${BASE_BETA},LAT_MIN=${LAT_MIN},LAT_MAX=${LAT_MAX},LON_MIN=${LON_MIN},LON_MAX=${LON_MAX},STOCHASTIC_REFINE=true,NOISE_SIGMA=1.0,Y_PATH_OVERRIDE=${Y_OVERRIDE}"

for train_years in "${SPLITS[@]}"; do
    submit_job "FOSI_refine_cons" "${COMMON},TRAIN_YEARS=${train_years}"
done

echo ""
echo "Total jobs: ${n_jobs}  (chained after data-prep job ${DATA_JOB_ID})"
if [ "$SUBMIT" = false ]; then
    echo "(dry run -- rerun with --submit to actually qsub these)"
fi
