#!/bin/bash
# ==============================================================
# submit_new_features_sweep_avg.sh
#
# "avg" data-variant twin of submit_new_features_sweep.sh. The original
# batch (results/FOSI_new_features/) deliberately scoped coastal_channel/
# classification_head/attention_end to the interp data variant only, to
# keep the batch a manageable size -- never checked against avg, so any
# conclusion about these 3 toggles is currently interp-only and unverified
# on the other data variant every other architecture sweep
# (daily_combo, daily_length) covers for both.
#
# Same 3 toggles x 4 training windows (test 2021), FOSI daily avg, into
# results/FOSI_new_features_avg/. Baseline arm to compare against:
# FOSI_daily_combo_avg/FOSI_el0_dmed_es0_* (already exists, not resubmitted
# here).
#
# MESA twin: submit_new_features_sweep_avg_mesa.sh.
#
# Usage:
#   ./submit_new_features_sweep_avg.sh              # dry run
#   ./submit_new_features_sweep_avg.sh --submit      # actually submit
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
BATCH_NAME="FOSI_new_features_avg"

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
    echo "qsub -N ${job_name} -v ${vlist} submit_engressnet_daily.sh"
    if [ "$SUBMIT" = true ]; then
        qsub -N "${job_name}" -v "${vlist}" submit_engressnet_daily.sh
    fi
    n_jobs=$((n_jobs + 1))
}

COMMON="BATCH_NAME=${BATCH_NAME},DATA_VARIANT=avg,TEST_YEARS=${TEST_YEARS},NUM_EPOCHS=${BASE_NUM_EPOCHS},K=${BASE_K},K_EVAL=${BASE_K_EVAL},BETA=${BASE_BETA},LAT_MIN=${LAT_MIN},LAT_MAX=${LAT_MAX},LON_MIN=${LON_MIN},LON_MAX=${LON_MAX}"

for train_years in "${SPLITS[@]}"; do
    submit_job "FOSI_coastal_ch_avg" "${COMMON},TRAIN_YEARS=${train_years},COASTAL_CHANNEL=true"
    submit_job "FOSI_classif_avg" "${COMMON},TRAIN_YEARS=${train_years},CLASSIFICATION_HEAD=true"
    submit_job "FOSI_attn_end_avg" "${COMMON},TRAIN_YEARS=${train_years},ATTENTION_END=true"
done

echo ""
echo "Total jobs: ${n_jobs}"
if [ "$SUBMIT" = false ]; then
    echo "(dry run -- rerun with --submit to actually qsub these)"
fi
