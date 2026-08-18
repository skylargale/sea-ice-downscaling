#!/bin/bash
# ==============================================================
# submit_daily_eval_sweep.sh
#
# Submits one PBS job per results/<BATCH_NAME>/ directory, each regenerating every
# evaluation_plots_daily.ipynb figure/table (section 16 batch mode, now with the
# 2026-08-10 PIOMAS-overlap fix + rolling-mean smoothing) for every run in that batch, via
# submit_daily_eval_batch.sh -> run_daily_eval_batch.py.
#
# BATCH_DIRS below is every results/ subdirectory produced by the daily-cadence sweeps
# (submit_daily_combo_sweep*.sh, submit_new_features_sweep*.sh, submit_calibrate_daily.sh,
# submit_daily_length_sweep*.sh). A directory with no run containing eval_data/ (e.g.
# MESA_new_features, whose runs currently have no eval_data/ subfolder at all, or the
# FOSI/MESA_daily_length_* directories before their training jobs finish) still gets a job --
# run_daily_eval_batch.py finds 0 matching runs, prints as much, and exits immediately, so
# it's a no-op rather than an error.
#
# Usage:
#   ./submit_daily_eval_sweep.sh              # dry run: prints every qsub command
#   ./submit_daily_eval_sweep.sh --submit      # actually submits all jobs
# ==============================================================

set -euo pipefail
cd "$(dirname "$0")"

SUBMIT=false
if [ "${1:-}" = "--submit" ]; then
    SUBMIT=true
fi

mkdir -p logs

BATCH_DIRS=(
    # Original 7 -- already evaluated (2026-08-10), commented out so a re-run of this
    # script doesn't redundantly regenerate them. Uncomment any that need a fresh pass
    # (e.g. after evaluation_plots_daily.ipynb's batch-mode logic changes again).
    # "FOSI_daily_combo_avg"
    # "FOSI_daily_combo_interp"
    # "FOSI_new_features"
    # "MESA_daily_combo_avg"
    # "MESA_daily_combo_interp"
    # "MESA_new_features"
    # "FOSI_calibrate"
    "FOSI_daily_length_avg"
    "FOSI_daily_length_interp"
    "MESA_daily_length_avg"
    "MESA_daily_length_interp"
    "FOSI_coastal_bias_experiments"
    "MESA_coastal_bias_experiments"
)

n_jobs=0
for batch in "${BATCH_DIRS[@]}"; do
    if [ ! -d "results/${batch}" ]; then
        echo "# skipping ${batch}: results/${batch} does not exist"
        continue
    fi
    job_name="eval_${batch}"
    echo "qsub -N ${job_name} -v BATCH_DIR=results/${batch} submit_daily_eval_batch.sh"
    if [ "$SUBMIT" = true ]; then
        qsub -N "${job_name}" -v "BATCH_DIR=results/${batch}" submit_daily_eval_batch.sh
    fi
    n_jobs=$((n_jobs + 1))
done

echo ""
echo "Total jobs: ${n_jobs}"
if [ "$SUBMIT" = false ]; then
    echo "(dry run -- rerun with --submit to actually qsub these)"
fi
