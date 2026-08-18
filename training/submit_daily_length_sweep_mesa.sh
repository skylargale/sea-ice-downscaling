#!/bin/bash
# ==============================================================
# submit_daily_length_sweep_mesa.sh
#
# MESA-HR twin of submit_daily_length_sweep.sh -- same training-LENGTH
# sensitivity sweep (find how short a training period can be, architecture
# held at the FOSI_2conv baseline), against the stitched HIST+RCP8.5 daily
# MESA data instead of FOSI. See that script for the full rationale
# (position-vs-length distinction, why TEST_YEARS=2020 for PIOMAS overlap).
#
# MESA daily data starts 1999-01-01 (vs. FOSI's 1958), so the 10-year window
# (2010-2019) still fits comfortably; no length here needs data MESA doesn't
# have.
#
# LENGTHS (years, ending 2019-12-31): 1, 2, 3, 5, 10 -> same train_years as
# the FOSI sweep. x 2 data variants (interp, avg) = 10 jobs. Batches land in
# results/MESA_daily_length_<variant>/.
#
# Usage:
#   ./submit_daily_length_sweep_mesa.sh              # dry run
#   ./submit_daily_length_sweep_mesa.sh --submit      # actually submit
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
    local job_name="$1"; shift
    local vlist="$1"; shift
    echo "qsub -N ${job_name} -v ${vlist} submit_engressnet_daily_mesa.sh"
    if [ "$SUBMIT" = true ]; then
        qsub -N "${job_name}" -v "${vlist}" submit_engressnet_daily_mesa.sh
    fi
    n_jobs=$((n_jobs + 1))
}

for variant in "${VARIANTS[@]}"; do
    batch_name="MESA_daily_length_${variant}"
    for length_spec in "${LENGTHS[@]}"; do
        IFS=":" read -r len_tag train_years <<< "$length_spec"
        job_name="MESA_len${len_tag}"
        submit_job "$job_name" \
            "BATCH_NAME=${batch_name},DATA_VARIANT=${variant},TRAIN_YEARS=${train_years},TEST_YEARS=${TEST_YEARS}"
    done
done

echo ""
echo "Total jobs: ${n_jobs}"
if [ "$SUBMIT" = false ]; then
    echo "(dry run -- rerun with --submit to actually qsub these)"
fi
