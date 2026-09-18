#!/bin/bash
# ==============================================================
# submit_noise_cascade_sweep_fosi.sh
#
# FOSI equivalent of submit_noise_cascade_sweep_mesa.sh (2026-09-17): every
# noise-cascade comparison so far has been MESA-only. Worth checking
# whether FOSI's smooth, single-realization truth reacts the same way to
# this architecture as it did (or didn't) to past architecture changes --
# the old shared-bias fix, for instance, was a real ~3% RMSE win on MESA
# but essentially flat on FOSI (0.1802->0.1805), plausibly because FOSI's
# truth has no per-member texture artifact to be sensitive to in the first
# place. No architecture toggles needed -- current (settled) noise cascade,
# same as the MESA batch.
#
# Same 4-window/domain/hyperparameter convention as every other comparison
# batch in this project (uses submit_engressnet_daily.sh, the FOSI
# template, not the MESA one).
#
# 4 jobs, into results/FOSI_noise_cascade_avg/.
#
# Usage:
#   ./submit_noise_cascade_sweep_fosi.sh              # dry run
#   ./submit_noise_cascade_sweep_fosi.sh --submit      # actually submit
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
    echo "qsub -N ${job_name} -v ${vlist} submit_engressnet_daily.sh"
    if [ "$SUBMIT" = true ]; then
        jid=$(qsub -N "${job_name}" -v "${vlist}" submit_engressnet_daily.sh)
        echo "$jid"
        JOB_IDS+=("$jid")
    fi
    n_jobs=$((n_jobs + 1))
}

batch_name="FOSI_noise_cascade_avg"
COMMON="BATCH_NAME=${batch_name},DATA_VARIANT=avg,TEST_YEARS=${TEST_YEARS},NUM_EPOCHS=${BASE_NUM_EPOCHS},K=${BASE_K},K_EVAL=${BASE_K_EVAL},BETA=${BASE_BETA},LAT_MIN=${LAT_MIN},LAT_MAX=${LAT_MAX},LON_MIN=${LON_MIN},LON_MAX=${LON_MAX}"
for train_years in "${SPLITS[@]}"; do
    submit_job "FOSI_noisecasc_avg" "${COMMON},TRAIN_YEARS=${train_years}"
done

echo ""
echo "Total jobs: ${n_jobs}"
if [ "$SUBMIT" = false ]; then
    echo "(dry run -- rerun with --submit to actually qsub these)"
else
    echo "Job IDs: ${JOB_IDS[*]}"
fi
