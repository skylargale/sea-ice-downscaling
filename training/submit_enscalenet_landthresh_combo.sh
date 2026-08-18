#!/bin/bash
# ==============================================================
# submit_enscalenet_landthresh_combo.sh
#
# Combines the two real, independently-positive findings from
# submit_coastal_bias_experiments.sh into one config and sweeps it across
# the standard 4-window grid, replacing the single one-off manual run
# under results/FOSI_coastal_bias_combo/ (not a reproducible sweep, no
# driver script) with a proper matched comparison:
#   - ENSCALE_NET=true   (decoder replacement; best calibration result so
#     far, Spread/Error ~1.05-1.13 vs. baseline ~1.5-1.7 in the one split
#     tested)
#   - LAND_THRESHOLD=0.5 (vs. production default 0.1; real but modest
#     coastal-RMSE improvement, e.g. FOSI 0.3978->0.3821)
# Everything else at the FOSI_2conv baseline (no extra_layer, no
# stochastic_refine, medium domain, interp data variant, TEST_YEARS=2021).
#
# 1 arm x 4 windows = 4 jobs, into results/FOSI_enscalenet_landthresh_combo/.
#
# MESA twin: submit_enscalenet_landthresh_combo_mesa.sh.
#
# Usage:
#   ./submit_enscalenet_landthresh_combo.sh              # dry run
#   ./submit_enscalenet_landthresh_combo.sh --submit      # actually submit
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
BATCH_NAME="FOSI_enscalenet_landthresh_combo"

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

COMMON="BATCH_NAME=${BATCH_NAME},DATA_VARIANT=interp,TEST_YEARS=${TEST_YEARS},NUM_EPOCHS=${BASE_NUM_EPOCHS},K=${BASE_K},K_EVAL=${BASE_K_EVAL},BETA=${BASE_BETA},LAT_MIN=${LAT_MIN},LAT_MAX=${LAT_MAX},LON_MIN=${LON_MIN},LON_MAX=${LON_MAX},ENSCALE_NET=true,NOISE_SIGMA=1.0,LAND_THRESHOLD=0.5"

for train_years in "${SPLITS[@]}"; do
    submit_job "FOSI_enscnet_lt05" "${COMMON},TRAIN_YEARS=${train_years}"
done

echo ""
echo "Total jobs: ${n_jobs}"
if [ "$SUBMIT" = false ]; then
    echo "(dry run -- rerun with --submit to actually qsub these)"
fi
