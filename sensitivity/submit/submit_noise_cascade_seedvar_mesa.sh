#!/bin/bash
# ==============================================================
# submit_noise_cascade_seedvar_mesa.sh
#
# Seed-variance check for the per-stage noise-cascade architecture
# (2026-09-16/17): how much does RMSE/Spread-Error move across seeds alone,
# on one window, current (default) architecture -- same methodology as this
# project's earlier seed-variance work on the old architecture (5 seeds,
# one window: train=2000-2005, test=2021), but never measured for this
# architecture, which has far fewer stochastic parameters than either
# predecessor. Directly answers whether the "Stochastic UNet Mean beats
# Deterministic UNet on RMSE" result seen in results/MESA_noise_cascade_avg/
# is a real property of this architecture or this particular seed's luck.
#
# No architecture toggles -- current (settled) noise cascade throughout.
# metrics.csv alone (already written by the training job itself) answers
# this; no separate eval-batch/aggregate job needed.
#
# 5 jobs (seed 0-4), into results/MESA_noise_cascade_seedvar/.
#
# Usage:
#   ./submit_noise_cascade_seedvar_mesa.sh              # dry run
#   ./submit_noise_cascade_seedvar_mesa.sh --submit      # actually submit
# ==============================================================

set -euo pipefail
cd "$(dirname "$0")"

SUBMIT=false
if [ "${1:-}" = "--submit" ]; then
    SUBMIT=true
fi

mkdir -p .logs

SEEDS=(0 1 2 3 4)
TRAIN_YEARS="2000-2005"
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

batch_name="MESA_noise_cascade_seedvar"
COMMON="BATCH_NAME=${batch_name},DATA_VARIANT=avg,TRAIN_YEARS=${TRAIN_YEARS},TEST_YEARS=${TEST_YEARS},NUM_EPOCHS=${BASE_NUM_EPOCHS},K=${BASE_K},K_EVAL=${BASE_K_EVAL},BETA=${BASE_BETA},LAT_MIN=${LAT_MIN},LAT_MAX=${LAT_MAX},LON_MIN=${LON_MIN},LON_MAX=${LON_MAX}"
for seed in "${SEEDS[@]}"; do
    submit_job "MESA_noisecasc_seed${seed}" "${COMMON},SEED=${seed}"
done

echo ""
echo "Total jobs: ${n_jobs}"
if [ "$SUBMIT" = false ]; then
    echo "(dry run -- rerun with --submit to actually qsub these)"
else
    echo "Job IDs: ${JOB_IDS[*]}"
fi
