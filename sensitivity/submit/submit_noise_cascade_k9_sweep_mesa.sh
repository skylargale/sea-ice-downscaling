#!/bin/bash
# ==============================================================
# submit_noise_cascade_k9_sweep_mesa.sh
#
# Kernel-size sensitivity check (2026-09-17), upper bracket: the settled
# architecture fixes GaussianNoiseStage's kernel footprint at 5x5 for all
# four decoder stages -- a single scalar shared width (log_width) is
# learnable within that fixed footprint, but the footprint itself isn't.
# This checks kernel_size=9 (this project's own historical value from the removed
# GaussianNoiseMix class) against the 5x5 baseline (results/MESA_noise_cascade_avg/)
# -- see submit_noise_cascade_k3_sweep_mesa.sh for the lower-bracket companion run.
#
# Same 4-window/domain/hyperparameter convention as every other comparison
# batch in this project.
#
# 4 jobs, into results/MESA_noise_cascade_k9_avg/.
#
# Usage:
#   ./submit_noise_cascade_k9_sweep_mesa.sh              # dry run
#   ./submit_noise_cascade_k9_sweep_mesa.sh --submit      # actually submit
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

batch_name="MESA_noise_cascade_k9_avg"
COMMON="BATCH_NAME=${batch_name},DATA_VARIANT=avg,TEST_YEARS=${TEST_YEARS},NUM_EPOCHS=${BASE_NUM_EPOCHS},K=${BASE_K},K_EVAL=${BASE_K_EVAL},BETA=${BASE_BETA},LAT_MIN=${LAT_MIN},LAT_MAX=${LAT_MAX},LON_MIN=${LON_MIN},LON_MAX=${LON_MAX},NOISE_KERNEL_SIZE=9"
for train_years in "${SPLITS[@]}"; do
    submit_job "MESA_noisecasc_k9" "${COMMON},TRAIN_YEARS=${train_years}"
done

echo ""
echo "Total jobs: ${n_jobs}"
if [ "$SUBMIT" = false ]; then
    echo "(dry run -- rerun with --submit to actually qsub these)"
else
    echo "Job IDs: ${JOB_IDS[*]}"
fi
