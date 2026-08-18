#!/bin/bash
# ==============================================================
# submit_cross_mesa_train_fosi_test.sh
#
# Cross-dataset generalization test, opposite direction of
# submit_cross_fosi_train_mesa_test.sh: train on MESA-HR (CESM1
# ihesp-hires, 6-member ensemble), evaluate the resulting checkpoint on
# FOSI (single-realization JRA55-forced) -- recommended_config.md
# architecture (2026-08-18): STOCHASTIC_REFINE=true, NOISE_SIGMA=1.0,
# DATA_VARIANT=avg, everything else off. Uses train_engressnet.py's
# --test-x-path/--test-y-path (added for this test) via
# submit_engressnet_daily_mesa.sh's TEST_X_PATH/TEST_Y_PATH passthrough.
#
# Same 4-window methodology as recommended_config.md's evidence table
# (train in {2000-2005, 2005-2010, 2010-2015, 2015-2020}), tested on FOSI's
# 2021 (the standard held-out year used throughout this project's daily
# batches -- both datasets cover it).
#
# 4 jobs, into results/MESA_train_FOSI_test_recommended/.
# FOSI-train/MESA-test twin: submit_cross_fosi_train_mesa_test.sh.
#
# Usage:
#   ./submit_cross_mesa_train_fosi_test.sh              # dry run
#   ./submit_cross_mesa_train_fosi_test.sh --submit      # actually submit
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
BATCH_NAME="MESA_train_FOSI_test_recommended"

TEST_X_PATH="/glade/derecho/scratch/skygale/Downscaling_Data/X_FOSI_HR_JRA55_daily_avg.nc"
TEST_Y_PATH="/glade/derecho/scratch/skygale/Downscaling_Data/Y_FOSI_HR_JRA55_daily.nc"

BASE_NUM_EPOCHS=20
BASE_K=20
BASE_K_EVAL=20
BASE_BETA=0.8
LAT_MIN=60
LAT_MAX=75
LON_MIN=-182
LON_MAX=-151

# submit_engressnet_daily_mesa.sh already requests the 256GB/8h tier for
# MESA's own full X/Y load; bump walltime a bit further since this also
# loads FOSI's full X/Y on top of it for the cross-dataset eval.
WALLTIME_OVERRIDE="10:00:00"

n_jobs=0

submit_job () {
    local job_name="$1"; shift
    local vlist="$1"; shift
    echo "qsub -N ${job_name} -l walltime=${WALLTIME_OVERRIDE} -v ${vlist} submit_engressnet_daily_mesa.sh"
    if [ "$SUBMIT" = true ]; then
        qsub -N "${job_name}" -l "walltime=${WALLTIME_OVERRIDE}" -v "${vlist}" submit_engressnet_daily_mesa.sh
    fi
    n_jobs=$((n_jobs + 1))
}

COMMON="BATCH_NAME=${BATCH_NAME},DATA_VARIANT=avg,TEST_YEARS=${TEST_YEARS},TEST_X_PATH=${TEST_X_PATH},TEST_Y_PATH=${TEST_Y_PATH},NUM_EPOCHS=${BASE_NUM_EPOCHS},K=${BASE_K},K_EVAL=${BASE_K_EVAL},BETA=${BASE_BETA},LAT_MIN=${LAT_MIN},LAT_MAX=${LAT_MAX},LON_MIN=${LON_MIN},LON_MAX=${LON_MAX},STOCHASTIC_REFINE=true,NOISE_SIGMA=1.0"

for train_years in "${SPLITS[@]}"; do
    submit_job "MESA_train_FOSI_test" "${COMMON},TRAIN_YEARS=${train_years}"
done

echo ""
echo "Total jobs: ${n_jobs}"
if [ "$SUBMIT" = false ]; then
    echo "(dry run -- rerun with --submit to actually qsub these)"
fi
