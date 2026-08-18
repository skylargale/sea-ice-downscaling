#!/bin/bash
# ==============================================================
# submit_enscalenet_sweep.sh
#
# EnScaleNet, standard-grid coverage. Prior evidence for EnScaleNet
# (--enscale-net, decoder-replacement architecture) is a single split
# (2015-2020->2021, see submit_coastal_bias_experiments.sh's FOSI_enscalenet
# job) plus a one-off manual run (results/FOSI_coastal_bias_combo/) -- it's
# the strongest calibration signal found so far (Spread/Error 1.52-1.67 ->
# 1.05-1.13 in that one split) but has never been checked across the same
# 4-window grid every other architecture toggle (new_features, daily_combo)
# was swept over. This closes that gap: same windows/domain/data-variant
# convention as submit_new_features_sweep.sh, ENSCALE_NET=true in place of
# the individual toggles, everything else at the FOSI_2conv baseline
# (no extra_layer, no stochastic_refine, land_threshold default 0.1).
#
# 1 arm x 4 windows = 4 jobs, into results/FOSI_enscalenet_sweep/.
#
# MESA twin: submit_enscalenet_sweep_mesa.sh.
#
# Usage:
#   ./submit_enscalenet_sweep.sh              # dry run
#   ./submit_enscalenet_sweep.sh --submit      # actually submit
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
BATCH_NAME="FOSI_enscalenet_sweep"

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

COMMON="BATCH_NAME=${BATCH_NAME},DATA_VARIANT=interp,TEST_YEARS=${TEST_YEARS},NUM_EPOCHS=${BASE_NUM_EPOCHS},K=${BASE_K},K_EVAL=${BASE_K_EVAL},BETA=${BASE_BETA},LAT_MIN=${LAT_MIN},LAT_MAX=${LAT_MAX},LON_MIN=${LON_MIN},LON_MAX=${LON_MAX},ENSCALE_NET=true,NOISE_SIGMA=1.0"

for train_years in "${SPLITS[@]}"; do
    submit_job "FOSI_enscalenet" "${COMMON},TRAIN_YEARS=${train_years}"
done

echo ""
echo "Total jobs: ${n_jobs}"
if [ "$SUBMIT" = false ]; then
    echo "(dry run -- rerun with --submit to actually qsub these)"
fi
