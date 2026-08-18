#!/bin/bash
# ==============================================================
# submit_daily_sweep.sh
#
# Standalone driver script (NOT part of the core pipeline) that qsub-submits
# the daily-FOSI EngressNet sweep: 5-year train windows, each tested on the
# most recent complete year of daily data (2021), for both daily X variants
# (interp, avg). Every job goes through submit_engressnet_daily.sh, with
# every parameter matching the FOSI_2conv baseline (K=20, K_EVAL=20,
# BETA=0.8, NUM_EPOCHS=20, no-patches, domain lat 60-75 / lon -182 to -151)
# -- only the train window and data variant change.
#
# Splits (all tested on 2021):
#   2000-2005, 2005-2010, 2010-2015, 2015-2020
#
# Batches (results/<batch_name>/, via BATCH_NAME):
#   FOSI_daily_interp -- X_FOSI_HR_JRA55_daily_interp.nc
#   FOSI_daily_avg    -- X_FOSI_HR_JRA55_daily_avg.nc
#
# Usage:
#   ./submit_daily_sweep.sh              # dry run: prints every qsub command
#   ./submit_daily_sweep.sh --submit      # actually submits all jobs
#
# Submits 4 splits x 2 variants = 8 jobs total. Each is a 1-GPU v100 Casper
# job, walltime up to 6h, mem 128GB (see submit_engressnet_daily.sh).
# ==============================================================

set -euo pipefail
cd "$(dirname "$0")"

SUBMIT=false
if [ "${1:-}" = "--submit" ]; then
    SUBMIT=true
fi

mkdir -p logs

SPLITS=(
    "2000-2005"
    "2005-2010"
    "2010-2015"
    "2015-2020"
)
TEST_YEARS="2021"
VARIANTS=("interp" "avg")

# Baseline (FOSI_2conv) values, held fixed.
BASE_NUM_EPOCHS=20
BASE_K=20
BASE_K_EVAL=20
BASE_BETA=0.8
BASE_LAT_MIN=60
BASE_LAT_MAX=75
BASE_LON_MIN=-182
BASE_LON_MAX=-151

n_jobs=0

submit_job () {
    # $1 = PBS job name, remaining args = NAME=VALUE PBS -v pairs
    local job_name="$1"; shift
    local vlist="$1"; shift
    echo "qsub -N ${job_name} -v ${vlist} submit_engressnet_daily.sh"
    if [ "$SUBMIT" = true ]; then
        qsub -N "${job_name}" -v "${vlist}" submit_engressnet_daily.sh
    fi
    n_jobs=$((n_jobs + 1))
}

for variant in "${VARIANTS[@]}"; do
    batch_name="FOSI_daily_${variant}"
    for train_years in "${SPLITS[@]}"; do
        submit_job "FOSI_daily" \
            "BATCH_NAME=${batch_name},DATA_VARIANT=${variant},TRAIN_YEARS=${train_years},TEST_YEARS=${TEST_YEARS},NUM_EPOCHS=${BASE_NUM_EPOCHS},K=${BASE_K},K_EVAL=${BASE_K_EVAL},BETA=${BASE_BETA},LAT_MIN=${BASE_LAT_MIN},LAT_MAX=${BASE_LAT_MAX},LON_MIN=${BASE_LON_MIN},LON_MAX=${BASE_LON_MAX}"
    done
done

echo ""
echo "Total jobs: ${n_jobs}"
if [ "$SUBMIT" = false ]; then
    echo "(dry run -- rerun with --submit to actually qsub these)"
fi
