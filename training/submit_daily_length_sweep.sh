#!/bin/bash
# ==============================================================
# submit_daily_length_sweep.sh
#
# FOSI daily training-LENGTH sensitivity sweep: previous sweeps
# (submit_daily_combo_sweep.sh) varied the *position* of a fixed 5-year
# training window (2000-2005, 2005-2010, 2010-2015, 2015-2020, tested on
# 2021) and found position didn't matter much. This sweep instead holds the
# window's END fixed and varies its LENGTH, to find how short a training
# period can be before accuracy degrades -- one axis at a time, architecture
# held at the FOSI_2conv baseline (no extra_layer/stochastic_refine/enscale,
# medium domain: lat60-75/lon-182to-151) so the length effect isn't
# confounded with an architecture change.
#
# Train windows all end 2019-12-31 and TEST_YEARS is fixed at 2020 (not the
# usual 2021) specifically so the test period overlaps PIOMAS's 1978-2020
# coverage -- see evaluation_plots_daily.ipynb's 2026-08-10 PIOMAS-overlap
# fix. Ending training at 2019 (not 2020) keeps 2020 a clean, unseen holdout
# instead of leaking it into training.
#
# LENGTHS (years, ending 2019-12-31):
#   1  -> 2019
#   2  -> 2018-2019
#   3  -> 2017-2019
#   5  -> 2015-2019
#   10 -> 2010-2019
# x 2 data variants (interp, avg) = 10 jobs. Batches land in
# results/FOSI_daily_length_<variant>/.
#
# MESA twin: submit_daily_length_sweep_mesa.sh (same lengths/logic, MESA
# daily data starts 1999-01-01 so all lengths here fit within it too).
#
# Usage:
#   ./submit_daily_length_sweep.sh              # dry run: prints every qsub command
#   ./submit_daily_length_sweep.sh --submit      # actually submits all jobs
# ==============================================================

set -euo pipefail
cd "$(dirname "$0")"

SUBMIT=false
if [ "${1:-}" = "--submit" ]; then
    SUBMIT=true
fi

mkdir -p logs

TEST_YEARS="2020"
VARIANTS=("interp" "avg")
# length_years:train_years
LENGTHS=(
    "01:2019"
    "02:2018-2019"
    "03:2017-2019"
    "05:2015-2019"
    "10:2010-2019"
)

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

for variant in "${VARIANTS[@]}"; do
    batch_name="FOSI_daily_length_${variant}"
    for length_spec in "${LENGTHS[@]}"; do
        IFS=":" read -r len_tag train_years <<< "$length_spec"
        job_name="FOSI_len${len_tag}"
        submit_job "$job_name" \
            "BATCH_NAME=${batch_name},DATA_VARIANT=${variant},TRAIN_YEARS=${train_years},TEST_YEARS=${TEST_YEARS}"
    done
done

echo ""
echo "Total jobs: ${n_jobs}"
if [ "$SUBMIT" = false ]; then
    echo "(dry run -- rerun with --submit to actually qsub these)"
fi
