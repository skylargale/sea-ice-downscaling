#!/bin/bash
# ==============================================================
# submit_season_sweep_mesa.sh
#
# MESA-HR twin of submit_season_sweep.sh -- same seasonal-focus experiment
# (March-July, --months "3-7") against the stitched HIST+RCP8.5 daily MESA
# data instead of FOSI. See that script for the full rationale (matched
# season/full-year baseline pairing, why TEST_YEARS=2020, fixed
# architecture: extra_layer=true, medium domain, avg variant, no
# enscale-lite/enscale-net).
#
# MESA daily data starts 1999-01-01, so all 4 windows below (ending 2019)
# fit within it, same as submit_daily_length_sweep_mesa.sh.
#
# 4 windows x 2 (season/full-year) = 8 jobs, into
# results/MESA_season_mjj_avg/ (season) and results/MESA_fullyear_avg/
# (baseline).
#
# Usage:
#   ./submit_season_sweep_mesa.sh              # dry run
#   ./submit_season_sweep_mesa.sh --submit      # actually submit
# ==============================================================

set -euo pipefail
cd "$(dirname "$0")"

SUBMIT=false
if [ "${1:-}" = "--submit" ]; then
    SUBMIT=true
fi

mkdir -p logs

TEST_YEARS="2020"
DATA_VARIANT="avg"
MONTHS="3-7"

SPLITS=(
    "2000-2004"
    "2005-2009"
    "2010-2014"
    "2015-2019"
)

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

for train_years in "${SPLITS[@]}"; do
    submit_job "MESA_season_${train_years}" \
        "BATCH_NAME=MESA_season_mjj_avg,DATA_VARIANT=${DATA_VARIANT},TRAIN_YEARS=${train_years},TEST_YEARS=${TEST_YEARS},MONTHS=${MONTHS},LAT_MIN=${LAT_MIN},LAT_MAX=${LAT_MAX},LON_MIN=${LON_MIN},LON_MAX=${LON_MAX},EXTRA_LAYER=true,STOCHASTIC_REFINE=false,ENSCALE_NET=false"

    submit_job "MESA_fullyear_${train_years}" \
        "BATCH_NAME=MESA_fullyear_avg,DATA_VARIANT=${DATA_VARIANT},TRAIN_YEARS=${train_years},TEST_YEARS=${TEST_YEARS},LAT_MIN=${LAT_MIN},LAT_MAX=${LAT_MAX},LON_MIN=${LON_MIN},LON_MAX=${LON_MAX},EXTRA_LAYER=true,STOCHASTIC_REFINE=false,ENSCALE_NET=false"
done

echo ""
echo "Total jobs: ${n_jobs}"
if [ "$SUBMIT" = false ]; then
    echo "(dry run -- rerun with --submit to actually qsub these)"
fi
