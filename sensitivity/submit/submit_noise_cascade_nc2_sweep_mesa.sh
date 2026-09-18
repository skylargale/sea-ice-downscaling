#!/bin/bash
# ==============================================================
# submit_noise_cascade_nc2_sweep_mesa.sh
#
# noise_channels sensitivity check (2026-09-17): the settled architecture
# uses 1 noise channel per decoder stage, reasoned (not tested) to be the
# right choice for this single-target-variable model -- see
# GaussianNoiseStage's docstring. The one architectural concern with 1
# channel is that a single-channel input to a 1x1 projection conv can only
# produce scaled/shifted copies of the same field in every output channel
# ("rank-1" projected noise), which the ReLU immediately downstream was
# judged to adequately soften but never empirically checked. This sweep
# tests --noise-channels=2 against the existing 1-channel baseline
# (results/MESA_noise_cascade_avg/) directly.
#
# Same 4-window/domain/hyperparameter convention as every other comparison
# batch in this project.
#
# 4 jobs, into results/MESA_noise_cascade_nc2_avg/.
#
# Usage:
#   ./submit_noise_cascade_nc2_sweep_mesa.sh              # dry run
#   ./submit_noise_cascade_nc2_sweep_mesa.sh --submit      # actually submit
# ==============================================================

set -euo pipefail
cd "$(dirname "$0")"

SUBMIT=false
if [ "${1:-}" = "--submit" ]; then
    SUBMIT=true
fi

mkdir -p .logs

SPLITS=(
    "2000-2005"
    "2005-2010"
    "2010-2015"
    "2015-2020"
)
TEST_YEARS="2021"

BASE_NUM_EPOCHS=20
BASE_K=20
BASE_K_EVAL=20
BASE_BETA=0.8
LAT_MIN=60
LAT_MAX=75
LON_MIN=-182
LON_MAX=-151

n_jobs=0
JOB_IDS=()

submit_job () {
    local job_name="$1"; shift
    local vlist="$1"; shift
    echo "qsub -N ${job_name} -v ${vlist} ../../training/submit_engressnet_daily_mesa.sh"
    if [ "$SUBMIT" = true ]; then
        jid=$(qsub -N "${job_name}" -v "${vlist}" ../../training/submit_engressnet_daily_mesa.sh)
        echo "$jid"
        JOB_IDS+=("$jid")
    fi
    n_jobs=$((n_jobs + 1))
}

batch_name="MESA_noise_cascade_nc2_avg"
COMMON="BATCH_NAME=${batch_name},DATA_VARIANT=avg,TEST_YEARS=${TEST_YEARS},NUM_EPOCHS=${BASE_NUM_EPOCHS},K=${BASE_K},K_EVAL=${BASE_K_EVAL},BETA=${BASE_BETA},LAT_MIN=${LAT_MIN},LAT_MAX=${LAT_MAX},LON_MIN=${LON_MIN},LON_MAX=${LON_MAX},NOISE_CHANNELS=2"
for train_years in "${SPLITS[@]}"; do
    submit_job "MESA_noisecasc_nc2" "${COMMON},TRAIN_YEARS=${train_years}"
done

echo ""
echo "Total jobs: ${n_jobs}"
if [ "$SUBMIT" = false ]; then
    echo "(dry run -- rerun with --submit to actually qsub these)"
else
    echo "Job IDs: ${JOB_IDS[*]}"
fi
