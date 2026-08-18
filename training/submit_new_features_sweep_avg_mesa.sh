#!/bin/bash
# ==============================================================
# submit_new_features_sweep_avg_mesa.sh
#
# MESA-HR twin of submit_new_features_sweep_avg.sh: coastal_channel/
# classification_head/attention_end on the avg data variant, never tested
# before (results/MESA_new_features/ is interp-only, matching the FOSI
# version's original scope-limiting choice).
#
# Same 3 toggles x 4 training windows (test 2021), MESA daily avg, into
# results/MESA_new_features_avg/.
#
# Usage:
#   ./submit_new_features_sweep_avg_mesa.sh              # dry run
#   ./submit_new_features_sweep_avg_mesa.sh --submit      # actually submit
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
BATCH_NAME="MESA_new_features_avg"

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
    echo "qsub -N ${job_name} -v ${vlist} submit_engressnet_daily_mesa.sh"
    if [ "$SUBMIT" = true ]; then
        qsub -N "${job_name}" -v "${vlist}" submit_engressnet_daily_mesa.sh
    fi
    n_jobs=$((n_jobs + 1))
}

COMMON="BATCH_NAME=${BATCH_NAME},DATA_VARIANT=avg,TEST_YEARS=${TEST_YEARS},NUM_EPOCHS=${BASE_NUM_EPOCHS},K=${BASE_K},K_EVAL=${BASE_K_EVAL},BETA=${BASE_BETA},LAT_MIN=${LAT_MIN},LAT_MAX=${LAT_MAX},LON_MIN=${LON_MIN},LON_MAX=${LON_MAX}"

for train_years in "${SPLITS[@]}"; do
    submit_job "MESA_coastal_ch_avg" "${COMMON},TRAIN_YEARS=${train_years},COASTAL_CHANNEL=true"
    submit_job "MESA_classif_avg" "${COMMON},TRAIN_YEARS=${train_years},CLASSIFICATION_HEAD=true"
    submit_job "MESA_attn_end_avg" "${COMMON},TRAIN_YEARS=${train_years},ATTENTION_END=true"
done

echo ""
echo "Total jobs: ${n_jobs}"
if [ "$SUBMIT" = false ]; then
    echo "(dry run -- rerun with --submit to actually qsub these)"
fi
