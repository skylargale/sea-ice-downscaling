#!/bin/bash
# ==============================================================
# submit_stochastic_refine_consregrid_2000_2020_avg.sh
#
# `avg`-variant twin of submit_stochastic_refine_consregrid_2000_2020.sh. The interp
# version confirmed conservative regridding is not worth adopting (real RMSE cost for a
# smaller Coastal RMSE gain) -- this checks whether that conclusion holds on `avg`, the
# now-recommended data variant (see recommended_config.md's "Still open" item 2). No new
# data-prep needed -- Y_FOSI_HR_JRA55_daily_conservative_2000_2020.nc already exists
# (built 2026-08-17), so this submits directly with no qsub dependency, unlike the
# original interp version which had to chain after the data-prep job.
#
# Matched bilinear-truth baseline arm (same windows/config, production Y, avg variant) is
# submit_stochastic_refine_bilinear_2000_2020_avg.sh.
#
# 4 jobs, into results/FOSI_stochastic_refine_consregrid_2000_2020_avg/.
#
# MESA twin: submit_stochastic_refine_consregrid_2000_2020_avg_mesa.sh.
#
# Usage:
#   ./submit_stochastic_refine_consregrid_2000_2020_avg.sh              # dry run
#   ./submit_stochastic_refine_consregrid_2000_2020_avg.sh --submit      # actually submit
# ==============================================================

set -euo pipefail
cd "$(dirname "$0")"

SUBMIT=false
if [ "${1:-}" = "--submit" ]; then
    SUBMIT=true
fi

mkdir -p logs

SPLITS=(
    "2000-2004"
    "2005-2009"
    "2010-2014"
    "2015-2019"
)
TEST_YEARS="2020"
BATCH_NAME="FOSI_stochastic_refine_consregrid_2000_2020_avg"
Y_OVERRIDE="/glade/derecho/scratch/skygale/Downscaling_Data/Y_FOSI_HR_JRA55_daily_conservative_2000_2020.nc"

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

COMMON="BATCH_NAME=${BATCH_NAME},DATA_VARIANT=avg,TEST_YEARS=${TEST_YEARS},NUM_EPOCHS=${BASE_NUM_EPOCHS},K=${BASE_K},K_EVAL=${BASE_K_EVAL},BETA=${BASE_BETA},LAT_MIN=${LAT_MIN},LAT_MAX=${LAT_MAX},LON_MIN=${LON_MIN},LON_MAX=${LON_MAX},STOCHASTIC_REFINE=true,NOISE_SIGMA=1.0,Y_PATH_OVERRIDE=${Y_OVERRIDE}"

for train_years in "${SPLITS[@]}"; do
    submit_job "FOSI_refine_cons_avg" "${COMMON},TRAIN_YEARS=${train_years}"
done

echo ""
echo "Total jobs: ${n_jobs}"
if [ "$SUBMIT" = false ]; then
    echo "(dry run -- rerun with --submit to actually qsub these)"
fi
