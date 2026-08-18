#!/bin/bash
# ==============================================================
# submit_season_sweep.sh
#
# FOSI seasonal-focus experiment: does restricting training/testing to a
# single season (March-July, --months "3-7", covering spring melt onset
# through mid-summer) improve accuracy and/or cut training time relative to
# training on all 12 months? Architecture held fixed at extra_layer=true
# (4th UNet stage), medium domain (lat60-75/lon-182to-151), avg data
# variant, no enscale-lite/enscale-net (stochastic_refine=false,
# enscale_net=false -- a fix for enscale-lite is being tested separately,
# kept out of this sweep).
#
# Same 4 non-overlapping 5-year train-window positions as
# submit_daily_length_sweep.sh's "05" arm convention, ending 2019 so none
# leak into the TEST_YEARS=2020 holdout (2020 is used, not 2021, so the
# test period overlaps PIOMAS's 1978-2020 coverage -- see
# evaluation_plots_daily.ipynb's 2026-08-10 PIOMAS-overlap fix):
#   2000-2004, 2005-2009, 2010-2014, 2015-2019
#
# Each window is submitted twice -- once with MONTHS=3-7 (season), once
# with no month filter (full year) -- as a matched baseline pair, so the
# season-vs-full-year comparison isn't confounded by anything except the
# month filter itself. 4 windows x 2 (season/full-year) = 8 jobs, into
# results/FOSI_season_mjj_avg/ (season) and results/FOSI_fullyear_avg/
# (baseline).
#
# MESA twin: submit_season_sweep_mesa.sh (same windows/logic; MESA daily
# data starts 1999-01-01, so all 4 windows here fit within it).
#
# Usage:
#   ./submit_season_sweep.sh              # dry run: prints every qsub command
#   ./submit_season_sweep.sh --submit      # actually submits all jobs
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

# Architecture, held fixed for every arm: extra_layer on, medium domain,
# no enscale-lite/enscale-net.
LAT_MIN=60
LAT_MAX=75
LON_MIN=-182
LON_MAX=-151

n_jobs=0

submit_job () {
    # $1 = PBS job name, $2 = comma-separated NAME=VALUE PBS -v pairs
    local job_name="$1"; shift
    local vlist="$1"; shift
    echo "qsub -N ${job_name} -v ${vlist} submit_engressnet_daily.sh"
    if [ "$SUBMIT" = true ]; then
        qsub -N "${job_name}" -v "${vlist}" submit_engressnet_daily.sh
    fi
    n_jobs=$((n_jobs + 1))
}

for train_years in "${SPLITS[@]}"; do
    # Seasonal arm (March-July only)
    submit_job "FOSI_season_${train_years}" \
        "BATCH_NAME=FOSI_season_mjj_avg,DATA_VARIANT=${DATA_VARIANT},TRAIN_YEARS=${train_years},TEST_YEARS=${TEST_YEARS},MONTHS=${MONTHS},LAT_MIN=${LAT_MIN},LAT_MAX=${LAT_MAX},LON_MIN=${LON_MIN},LON_MAX=${LON_MAX},EXTRA_LAYER=true,STOCHASTIC_REFINE=false,ENSCALE_NET=false"

    # Matched full-year baseline (same window/architecture, no month filter)
    submit_job "FOSI_fullyear_${train_years}" \
        "BATCH_NAME=FOSI_fullyear_avg,DATA_VARIANT=${DATA_VARIANT},TRAIN_YEARS=${train_years},TEST_YEARS=${TEST_YEARS},LAT_MIN=${LAT_MIN},LAT_MAX=${LAT_MAX},LON_MIN=${LON_MIN},LON_MAX=${LON_MAX},EXTRA_LAYER=true,STOCHASTIC_REFINE=false,ENSCALE_NET=false"
done

echo ""
echo "Total jobs: ${n_jobs}"
if [ "$SUBMIT" = false ]; then
    echo "(dry run -- rerun with --submit to actually qsub these)"
fi
