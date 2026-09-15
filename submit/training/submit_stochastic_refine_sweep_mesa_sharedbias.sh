#!/bin/bash
# ==============================================================
# submit_stochastic_refine_sweep_mesa_sharedbias.sh
#
# Version6's primary training sweep: the same recommended-config toggles as
# Version5's identically-named script (STOCHASTIC_REFINE=true, NOISE_SIGMA=1.0,
# NOISE_SHARED_BIAS=true, DATA_VARIANT=avg, same 4 windows/domain), but run
# through Version6's rewritten functions_engressnet.py -- which unconditionally
# trains against a standardized *physical increment* (Y_phys - baseline_phys,
# normalized by the increment's own train-set mean/std, no internal
# bilinear-base residual add inside the model) instead of the standardized raw
# value, matching Brajard et al. 2026 (EGUsphere, Arctic SIT super-resolution
# via conditional diffusion)'s Eq. 4 / NVIDIA CorrDiff's residual construction.
# There is no toggle for this in Version6 -- it's simply how the model trains
# now, the same way Version5 always added an (uncleanly-scaled) internal
# residual. Compare this batch's output directly against Version5's
# `results/MESA_stochastic_refine_sweep_avg_sharedbias/` for a clean before/
# after of the residual/normalization change alone, with every other toggle
# held fixed.
#
# (NOISE_SHARED_BIAS=true is still a real, separate fix in this config -- see
# Version5/recommended_config.md's "Known issue" section -- unrelated to the
# increment-vs-value question; kept here since it's part of the recommended
# config, not because this script is specifically testing it.)
#
# 1 variant (avg only -- the recommended data variant) x 4 windows = 4 jobs,
# into results/MESA_stochastic_refine_sweep_avg_sharedbias/.
#
# Usage:
#   ./submit_stochastic_refine_sweep_mesa_sharedbias.sh              # dry run
#   ./submit_stochastic_refine_sweep_mesa_sharedbias.sh --submit      # actually submit
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

batch_name="MESA_stochastic_refine_sweep_avg_sharedbias"
COMMON="BATCH_NAME=${batch_name},DATA_VARIANT=avg,TEST_YEARS=${TEST_YEARS},NUM_EPOCHS=${BASE_NUM_EPOCHS},K=${BASE_K},K_EVAL=${BASE_K_EVAL},BETA=${BASE_BETA},LAT_MIN=${LAT_MIN},LAT_MAX=${LAT_MAX},LON_MIN=${LON_MIN},LON_MAX=${LON_MAX},STOCHASTIC_REFINE=true,NOISE_SIGMA=1.0,NOISE_SHARED_BIAS=true"
for train_years in "${SPLITS[@]}"; do
    submit_job "MESA_refine_avg_sharedbias" "${COMMON},TRAIN_YEARS=${train_years}"
done

echo ""
echo "Total jobs: ${n_jobs}"
if [ "$SUBMIT" = false ]; then
    echo "(dry run -- rerun with --submit to actually qsub these)"
fi
