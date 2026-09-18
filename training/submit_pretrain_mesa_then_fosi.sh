#!/bin/bash
# ==============================================================
# submit_pretrain_mesa_then_fosi.sh
#
# Transfer-learning experiment (2026-09-17): initialize a FOSI training run's weights from
# the matching-window MESA checkpoint instead of random init, then continue training on FOSI.
# Motivation: MESA's 6-member ensemble gives ~6x more (member, time) training samples per
# window than FOSI's single-realization record, so pretraining on MESA's larger sample count
# before fine-tuning on FOSI's smaller, single-realization one is a natural transfer-learning
# setup -- and this project already has the exact checkpoints needed for free, since
# submit_noise_cascade_sweep_mesa.sh's 4-window MESA sweep already completed
# (results/MESA_noise_cascade_avg/, jobs 5956146-5956149) under the *current* architecture
# (per-stage noise cascade -- see Version6/README.md), so no MESA retraining is needed here.
#
# Channel-schema mismatch: FOSI's X carries vector wind (u_10, v_10, 4 predictor channels)
# while MESA's carries only wind speed (U10, 3 channels) -- see
# functions_engressnet.collapse_wind_vector_channel. Without reconciling this, the FOSI run's
# first conv layer would be sized for 4 channels and --init-checkpoint's strict state_dict load
# would fail immediately. COLLAPSE_WIND_VECTOR=true (train_engressnet.py --collapse-wind-vector)
# fixes this by collapsing FOSI's u_10/v_10 into a derived wind-speed channel before training,
# matching the MESA checkpoint's 3-channel schema exactly (confirmed via a direct
# strict-load smoke test before this script was written -- see the 2026-09-17 session).
#
# Each of the 4 windows fine-tunes from *its own matching* MESA window's checkpoint (e.g. the
# FOSI 2000-2005 run initializes from MESA's own 2000-2005 checkpoint), keeping the pretrain and
# fine-tune stages aligned on the same nominal training period, differing only in data source.
# Same epoch count / hyperparameters as every other FOSI daily batch in this project (no
# separate shorter fine-tune-only schedule) -- if the fine-tuned runs turn out to need less
# training than a from-scratch run, that's worth revisiting once results are in, not assumed
# up front.
#
# Full eval pipeline auto-chained same as every other comparison batch in this project (see
# project memory / prior sessions' MESA_no_stochastic_refine_avg etc.): 4 training jobs -> held
# per-run eval-figures job (afterok all 4) -> held 4-window-aggregate job (afterok the eval job).
# No manual follow-up needed once training finishes.
#
# 4 jobs, into results/FOSI_pretrained_from_mesa_avg/. Compare its metrics.csv/Spread-Error and
# evaluation/saved_figs against results/FOSI_noise_cascade_avg/ (the from-scratch FOSI baseline, same
# architecture, no pretraining) once both are done.
#
# Usage:
#   ./submit_pretrain_mesa_then_fosi.sh              # dry run
#   ./submit_pretrain_mesa_then_fosi.sh --submit      # actually submit
# ==============================================================

set -euo pipefail
cd "$(dirname "$0")"

SUBMIT=false
if [ "${1:-}" = "--submit" ]; then
    SUBMIT=true
fi

mkdir -p .logs

MESA_BATCH_DIR="../results/MESA_noise_cascade_avg"

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

batch_name="FOSI_pretrained_from_mesa_avg"

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

for train_years in "${SPLITS[@]}"; do
    mesa_ckpt_dir=$(find "$MESA_BATCH_DIR" -maxdepth 1 -type d -name "MESA_noisecasc_avg_${train_years}_${TEST_YEARS}_*" | head -n1)
    if [ -z "$mesa_ckpt_dir" ]; then
        echo "ERROR: no matching MESA checkpoint dir found for train_years=${train_years} under ${MESA_BATCH_DIR}" >&2
        exit 1
    fi
    ckpt_path="${mesa_ckpt_dir}/model_state_dict.pt"
    if [ ! -f "$ckpt_path" ]; then
        echo "ERROR: ${ckpt_path} does not exist (MESA run may not have finished)" >&2
        exit 1
    fi
    echo "train_years=${train_years} -> pretrained init from ${ckpt_path}"

    COMMON="BATCH_NAME=${batch_name},DATA_VARIANT=avg,TEST_YEARS=${TEST_YEARS},NUM_EPOCHS=${BASE_NUM_EPOCHS},K=${BASE_K},K_EVAL=${BASE_K_EVAL},BETA=${BASE_BETA},LAT_MIN=${LAT_MIN},LAT_MAX=${LAT_MAX},LON_MIN=${LON_MIN},LON_MAX=${LON_MAX},TRAIN_YEARS=${train_years},INIT_CHECKPOINT=${ckpt_path},COLLAPSE_WIND_VECTOR=true"
    submit_job "FOSI_pretrmesa_avg" "${COMMON}"
done

echo ""
echo "Training jobs: ${n_jobs}"

if [ "$SUBMIT" = true ]; then
    echo "Training job IDs: ${JOB_IDS[*]}"
    train_depend="afterok"
    for jid in "${JOB_IDS[@]}"; do
        train_depend="${train_depend}:${jid}"
    done

    cd ../evaluation
    eval_jid=$(qsub -N "eval_pretrmesa" -W depend="${train_depend}" -v BATCH_DIR="results/${batch_name}" submit_daily_eval_batch.sh)
    echo "Eval job (held on dependency): $eval_jid (depend=${train_depend})"
    agg_jid=$(qsub -N "agg_pretrmesa" -W depend="afterok:${eval_jid}" -v BATCH_DIR="results/${batch_name}" submit_4window_aggregate_figures.sh)
    echo "Aggregate job (held on dependency): $agg_jid (depend=afterok:${eval_jid})"
else
    echo "(dry run -- rerun with --submit to actually qsub these, including the chained eval/aggregate jobs)"
fi
