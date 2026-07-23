#!/bin/bash
# ==============================================================
# submit_sensitivity_tests.sh
#
# Standalone driver script (NOT part of the core pipeline) that qsub-submits
# a battery of EngressNet sensitivity tests across the standard 5-way
# train/test sweep (train 1960-1970 / 1970-1980 / 1980-1990 / 1990-2000 /
# 2000-2009, all tested on 2010-2020). Every job goes through the existing
# submit_engressnet.sh, only varying one axis per batch; every other
# parameter matches the FOSI_2conv baseline (K=20, K_EVAL=20, BETA=0.8,
# NUM_EPOCHS=20, domain lat 60-75 / lon -182 to -151) so each batch isolates
# exactly one change relative to that baseline.
#
# Batches (results/<batch_name>/, via BATCH_NAME):
#   FOSI_epochs40        -- num_epochs 20 -> 40
#   FOSI_beta0.5         -- beta 0.8 -> 0.5
#   FOSI_beta0.75        -- beta 0.8 -> 0.75
#   FOSI_beta1.5         -- beta 0.8 -> 1.5
#   FOSI_extralayer1024  -- +1 UNet downsample/upsample stage (1024ch bottleneck)
#   FOSI_enscale_lite    -- neighbor-conditioned stochastic noise refinement
#   FOSI_domain_tiny     -- domain shrunk to 8x16 (lat 62-69, lon -172 to -157)
#   FOSI_domain_large    -- domain grown to 16x48 (lat 60-75, lon -188 to -141)
# (the "medium" domain size is the baseline itself -- already run as FOSI_2conv,
# not resubmitted here.)
#
# Usage:
#   ./submit_sensitivity_tests.sh              # dry run: prints every qsub command
#   ./submit_sensitivity_tests.sh --submit      # actually submits all jobs
#
# Submits 40 jobs total (5 splits x 8 batch arms). Each is a 1-GPU v100
# Casper job, walltime up to 6h (see submit_engressnet.sh's #PBS -l walltime).
# ==============================================================

set -euo pipefail
cd "$(dirname "$0")"

SUBMIT=false
if [ "${1:-}" = "--submit" ]; then
    SUBMIT=true
fi

mkdir -p logs

SPLITS=(
    "1960-1970"
    "1970-1980"
    "1980-1990"
    "1990-2000"
    "2000-2009"
)
TEST_YEARS="2010-2020"

# Baseline (FOSI_2conv) values held fixed across every sensitivity arm unless
# that arm is specifically testing the value.
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
    # $1 = PBS job name / batch name, remaining args = NAME=VALUE PBS -v pairs
    local job_name="$1"; shift
    local vlist="$1"; shift
    echo "qsub -N ${job_name} -v ${vlist} submit_engressnet.sh"
    if [ "$SUBMIT" = true ]; then
        qsub -N "${job_name}" -v "${vlist}" submit_engressnet.sh
    fi
    n_jobs=$((n_jobs + 1))
}

for split in "${SPLITS[@]}"; do
    train_years="$split"

    # ---- (1) More training epochs ----
    submit_job "FOSI_epochs40" \
        "BATCH_NAME=FOSI_epochs40,TRAIN_YEARS=${train_years},TEST_YEARS=${TEST_YEARS},NUM_EPOCHS=40,K=${BASE_K},K_EVAL=${BASE_K_EVAL},BETA=${BASE_BETA},LAT_MIN=${BASE_LAT_MIN},LAT_MAX=${BASE_LAT_MAX},LON_MIN=${BASE_LON_MIN},LON_MAX=${BASE_LON_MAX}"

    # ---- (2) Beta sweep ----
    for beta in 0.5 0.75 1.5; do
        submit_job "FOSI_beta${beta}" \
            "BATCH_NAME=FOSI_beta${beta},TRAIN_YEARS=${train_years},TEST_YEARS=${TEST_YEARS},NUM_EPOCHS=${BASE_NUM_EPOCHS},K=${BASE_K},K_EVAL=${BASE_K_EVAL},BETA=${beta},LAT_MIN=${BASE_LAT_MIN},LAT_MAX=${BASE_LAT_MAX},LON_MIN=${BASE_LON_MIN},LON_MAX=${BASE_LON_MAX}"
    done

    # ---- (3) Extra 1024-channel layer ----
    submit_job "FOSI_extralayer1024" \
        "BATCH_NAME=FOSI_extralayer1024,TRAIN_YEARS=${train_years},TEST_YEARS=${TEST_YEARS},NUM_EPOCHS=${BASE_NUM_EPOCHS},K=${BASE_K},K_EVAL=${BASE_K_EVAL},BETA=${BASE_BETA},LAT_MIN=${BASE_LAT_MIN},LAT_MAX=${BASE_LAT_MAX},LON_MIN=${BASE_LON_MIN},LON_MAX=${BASE_LON_MAX},EXTRA_LAYER=true"

    # ---- (4) Neighbor-conditioned stochastic refinement (EnScale-lite) ----
    submit_job "FOSI_enscale_lite" \
        "BATCH_NAME=FOSI_enscale_lite,TRAIN_YEARS=${train_years},TEST_YEARS=${TEST_YEARS},NUM_EPOCHS=${BASE_NUM_EPOCHS},K=${BASE_K},K_EVAL=${BASE_K_EVAL},BETA=${BASE_BETA},LAT_MIN=${BASE_LAT_MIN},LAT_MAX=${BASE_LAT_MAX},LON_MIN=${BASE_LON_MIN},LON_MAX=${BASE_LON_MAX},STOCHASTIC_REFINE=true"

    # ---- (5) Domain size: tiny (8x16) and large (16x48); medium = FOSI_2conv (already run) ----
    submit_job "FOSI_domain_tiny" \
        "BATCH_NAME=FOSI_domain_tiny,TRAIN_YEARS=${train_years},TEST_YEARS=${TEST_YEARS},NUM_EPOCHS=${BASE_NUM_EPOCHS},K=${BASE_K},K_EVAL=${BASE_K_EVAL},BETA=${BASE_BETA},LAT_MIN=62,LAT_MAX=69,LON_MIN=-172,LON_MAX=-157"

    submit_job "FOSI_domain_large" \
        "BATCH_NAME=FOSI_domain_large,TRAIN_YEARS=${train_years},TEST_YEARS=${TEST_YEARS},NUM_EPOCHS=${BASE_NUM_EPOCHS},K=${BASE_K},K_EVAL=${BASE_K_EVAL},BETA=${BASE_BETA},LAT_MIN=60,LAT_MAX=75,LON_MIN=-188,LON_MAX=-141"
done

echo ""
echo "Total jobs: ${n_jobs}"
if [ "$SUBMIT" = false ]; then
    echo "(dry run -- rerun with --submit to actually qsub these)"
fi
