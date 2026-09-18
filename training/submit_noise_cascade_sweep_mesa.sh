#!/bin/bash
# ==============================================================
# submit_noise_cascade_sweep_mesa.sh
#
# First training sweep under the per-stage independent-noise-cascade
# architecture (2026-09-16): each decoder stage (d3/d2/d1/final) draws its
# own fresh noise at its own resolution (GaussianNoiseStage), correlated by
# a shared 5x5 Gaussian kernel with a learnable width, renormalized, and
# scaled by a learnable per-stage amplitude -- replacing the single global
# latent z (drawn once, resized into every stage) that the
# MESA_no_stochastic_refine_avg/MESA_z_inject_final_avg comparison batches
# used. No architecture toggles needed here: this is now the only
# architecture there is (no --stochastic-refine/--z-inject-final/
# --deep-mask-head/--latent-channels/--noise-smooth-kernel-size -- all
# removed, see Version6/README.md).
#
# Same 4-window/domain/hyperparameter convention as every prior comparison
# batch in this project: train ∈ {2000-2005, 2005-2010, 2010-2015,
# 2015-2020}, test=2021, DATA_VARIANT=avg, medium domain (lat60-75/
# lon-182to-151), K=20/K_EVAL=20/BETA=0.8/NUM_EPOCHS=20.
#
# 4 jobs, into results/MESA_noise_cascade_avg/.
#
# Usage:
#   ./submit_noise_cascade_sweep_mesa.sh              # dry run
#   ./submit_noise_cascade_sweep_mesa.sh --submit      # actually submit
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
    echo "qsub -N ${job_name} -v ${vlist} submit_engressnet_daily_mesa.sh"
    if [ "$SUBMIT" = true ]; then
        jid=$(qsub -N "${job_name}" -v "${vlist}" submit_engressnet_daily_mesa.sh)
        echo "$jid"
        JOB_IDS+=("$jid")
    fi
    n_jobs=$((n_jobs + 1))
}

batch_name="MESA_noise_cascade_avg"
COMMON="BATCH_NAME=${batch_name},DATA_VARIANT=avg,TEST_YEARS=${TEST_YEARS},NUM_EPOCHS=${BASE_NUM_EPOCHS},K=${BASE_K},K_EVAL=${BASE_K_EVAL},BETA=${BASE_BETA},LAT_MIN=${LAT_MIN},LAT_MAX=${LAT_MAX},LON_MIN=${LON_MIN},LON_MAX=${LON_MAX}"
for train_years in "${SPLITS[@]}"; do
    submit_job "MESA_noisecasc_avg" "${COMMON},TRAIN_YEARS=${train_years}"
done

echo ""
echo "Total jobs: ${n_jobs}"
if [ "$SUBMIT" = false ]; then
    echo "(dry run -- rerun with --submit to actually qsub these)"
else
    echo "Job IDs: ${JOB_IDS[*]}"
fi
