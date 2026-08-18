#!/bin/bash
# ==============================================================
# submit_cross_fosi_train_mesa_test.sh
#
# Cross-dataset generalization test: train on FOSI (single-realization
# JRA55-forced), evaluate the resulting checkpoint on MESA-HR (CESM1
# ihesp-hires, 6-member ensemble) -- using the recommended_config.md
# architecture (2026-08-18): STOCHASTIC_REFINE=true, NOISE_SIGMA=1.0,
# DATA_VARIANT=avg, everything else off. Uses train_engressnet.py's
# --test-x-path/--test-y-path (added for this test) to point evaluation at
# a different dataset than training, via submit_engressnet_daily.sh's
# TEST_X_PATH/TEST_Y_PATH passthrough.
#
# Same 4-window methodology as recommended_config.md's evidence table
# (train in {2000-2005, 2005-2010, 2010-2015, 2015-2020}), tested on MESA's
# 2021 (the standard held-out year used throughout this project's daily
# batches -- both datasets cover it). Both the FOSI (train) and MESA (test)
# full X/Y arrays get loaded into memory (run_pipeline loads each dataset
# whole before filtering by year), so this needs MESA-tier memory (256GB)
# even though FOSI is the smaller dataset -- overridden here via `qsub -l`
# since submit_engressnet_daily.sh's own #PBS line only requests 128GB.
#
# 4 jobs, into results/FOSI_train_MESA_test_recommended/.
# MESA-train/FOSI-test twin: submit_cross_mesa_train_fosi_test.sh.
#
# Usage:
#   ./submit_cross_fosi_train_mesa_test.sh              # dry run
#   ./submit_cross_fosi_train_mesa_test.sh --submit      # actually submit
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
BATCH_NAME="FOSI_train_MESA_test_recommended"

TEST_X_PATH="/glade/derecho/scratch/skygale/Downscaling_Data/X_MESA_HR_daily_avg.nc"
TEST_Y_PATH="/glade/derecho/scratch/skygale/Downscaling_Data/Y_MESA_HR_daily.nc"

BASE_NUM_EPOCHS=20
BASE_K=20
BASE_K_EVAL=20
BASE_BETA=0.8
LAT_MIN=60
LAT_MAX=75
LON_MIN=-182
LON_MAX=-151

# Cross-dataset eval loads FOSI's full X/Y (128GB-tier) AND MESA's full X/Y
# (256GB-tier) into memory at once -- resource-override to the larger tier
# rather than editing submit_engressnet_daily.sh's own #PBS line, since
# same-dataset FOSI runs still only need 128GB.
RESOURCE_OVERRIDE="select=1:ncpus=16:ngpus=1:mem=256GB:gpu_type=v100"
WALLTIME_OVERRIDE="10:00:00"

n_jobs=0

submit_job () {
    local job_name="$1"; shift
    local vlist="$1"; shift
    echo "qsub -N ${job_name} -l ${RESOURCE_OVERRIDE} -l walltime=${WALLTIME_OVERRIDE} -v ${vlist} submit_engressnet_daily.sh"
    if [ "$SUBMIT" = true ]; then
        qsub -N "${job_name}" -l "${RESOURCE_OVERRIDE}" -l "walltime=${WALLTIME_OVERRIDE}" -v "${vlist}" submit_engressnet_daily.sh
    fi
    n_jobs=$((n_jobs + 1))
}

COMMON="BATCH_NAME=${BATCH_NAME},DATA_VARIANT=avg,TEST_YEARS=${TEST_YEARS},TEST_X_PATH=${TEST_X_PATH},TEST_Y_PATH=${TEST_Y_PATH},NUM_EPOCHS=${BASE_NUM_EPOCHS},K=${BASE_K},K_EVAL=${BASE_K_EVAL},BETA=${BASE_BETA},LAT_MIN=${LAT_MIN},LAT_MAX=${LAT_MAX},LON_MIN=${LON_MIN},LON_MAX=${LON_MAX},STOCHASTIC_REFINE=true,NOISE_SIGMA=1.0"

for train_years in "${SPLITS[@]}"; do
    submit_job "FOSI_train_MESA_test" "${COMMON},TRAIN_YEARS=${train_years}"
done

echo ""
echo "Total jobs: ${n_jobs}"
if [ "$SUBMIT" = false ]; then
    echo "(dry run -- rerun with --submit to actually qsub these)"
fi
