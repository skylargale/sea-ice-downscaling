#!/bin/bash
# ==============================================================
# submit_enscalenet_sweep_mesa.sh
#
# MESA-HR twin of submit_enscalenet_sweep.sh: same standard-grid coverage
# gap-fill for EnScaleNet (--enscale-net), MESA daily interp data instead
# of FOSI. Never run before -- the only MESA EnScaleNet evidence so far is
# the one-off results/MESA_coastal_bias_combo/ run.
#
# 1 arm x 4 windows = 4 jobs, into results/MESA_enscalenet_sweep/.
#
# Usage:
#   ./submit_enscalenet_sweep_mesa.sh              # dry run
#   ./submit_enscalenet_sweep_mesa.sh --submit      # actually submit
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
BATCH_NAME="MESA_enscalenet_sweep"

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

COMMON="BATCH_NAME=${BATCH_NAME},DATA_VARIANT=interp,TEST_YEARS=${TEST_YEARS},NUM_EPOCHS=${BASE_NUM_EPOCHS},K=${BASE_K},K_EVAL=${BASE_K_EVAL},BETA=${BASE_BETA},LAT_MIN=${LAT_MIN},LAT_MAX=${LAT_MAX},LON_MIN=${LON_MIN},LON_MAX=${LON_MAX},ENSCALE_NET=true,NOISE_SIGMA=1.0"

for train_years in "${SPLITS[@]}"; do
    submit_job "MESA_enscalenet" "${COMMON},TRAIN_YEARS=${train_years}"
done

echo ""
echo "Total jobs: ${n_jobs}"
if [ "$SUBMIT" = false ]; then
    echo "(dry run -- rerun with --submit to actually qsub these)"
fi
