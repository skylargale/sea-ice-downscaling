#!/bin/bash
# ==============================================================
# submit_stochastic_refine_sweep.sh
#
# Standard-grid coverage for the RECOMMENDED config (see
# ../../recommended_config.md, compiled 2026-08-14): STOCHASTIC_REFINE=true
# (EnScale-lite refiner, now with the unconditional smooth_noise() fix baked
# into functions_engressnet.py) is the single best-performing config found
# in the full gap-filling sweep -- but it was only ever tested on ONE split
# per dataset (2000-2005->2021, FOSI_enscale_lite_smooth/
# MESA_enscale_lite_smooth), unlike every other candidate in the evidence
# table (4 splits). This closes that gap: same 4-window grid convention as
# submit_new_features_sweep.sh, both data variants (interp and avg, since
# the smooth-noise-fixed refiner has never been checked on avg at all),
# FOSI only -- MESA twin is submit_stochastic_refine_sweep_mesa.sh.
#
# 2 variants x 4 windows = 8 jobs, into
# results/FOSI_stochastic_refine_sweep_<variant>/.
#
# Usage:
#   ./submit_stochastic_refine_sweep.sh              # dry run
#   ./submit_stochastic_refine_sweep.sh --submit      # actually submit
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
VARIANTS=("interp" "avg")

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

for variant in "${VARIANTS[@]}"; do
    batch_name="FOSI_stochastic_refine_sweep_${variant}"
    COMMON="BATCH_NAME=${batch_name},DATA_VARIANT=${variant},TEST_YEARS=${TEST_YEARS},NUM_EPOCHS=${BASE_NUM_EPOCHS},K=${BASE_K},K_EVAL=${BASE_K_EVAL},BETA=${BASE_BETA},LAT_MIN=${LAT_MIN},LAT_MAX=${LAT_MAX},LON_MIN=${LON_MIN},LON_MAX=${LON_MAX},STOCHASTIC_REFINE=true,NOISE_SIGMA=1.0"
    for train_years in "${SPLITS[@]}"; do
        submit_job "FOSI_refine_${variant}" "${COMMON},TRAIN_YEARS=${train_years}"
    done
done

echo ""
echo "Total jobs: ${n_jobs}"
if [ "$SUBMIT" = false ]; then
    echo "(dry run -- rerun with --submit to actually qsub these)"
fi
