#!/bin/bash
# ==============================================================
# PBS batch submission script for train_engressnet.py (Version4) on Casper,
# using the DAILY MESA-HR dataset (stitched HIST+RCP8.5) instead of FOSI.
#
# Mirrors submit_engressnet_daily.sh (same architecture/loss/domain defaults
# -- matching the FOSI_2conv baseline: K=20, K_EVAL=20, BETA=0.8,
# NUM_EPOCHS=20, no-patches, domain lat60-75/lon-182to-151). DATA_VARIANT
# selects which daily X file to train against:
#   interp -> X_MESA_HR_daily_interp.nc
#   avg    -> X_MESA_HR_daily_avg.nc
# Y is always Y_MESA_HR_daily.nc.
#
# This file is the STITCHED HIST (1920-2006, subset to >=1999) + RCP8.5
# (2006-2021ish) record, restricted to the 6-member ensemble intersection
# that has complete daily data in both periods (.004/.005/.006/.007/.008/
# .010 -- see processing/build_X_Y_from_MESA-HR_daily_rcp85.py and
# processing/stitch_MESA_HR_daily_hist_rcp85.py for why 3 of the original
# 9 HIST members were dropped). Unlike FOSI's single-realization daily data,
# every MESA sample has a real ensemble-member axis, so at a given time span
# there are ~6x more (member, time) training/eval samples than FOSI would
# have -- mem bumped further accordingly (256GB vs FOSI daily's 128GB).
#
# Submit with:  qsub submit_engressnet_daily_mesa.sh
# Check status: qstat -u $USER
#
# Try a different train/test split or data variant without editing this file:
# qsub -v TRAIN_YEARS="2005-2010",TEST_YEARS="2021",DATA_VARIANT="avg" submit_engressnet_daily_mesa.sh
# ==============================================================

#PBS -N MESA_daily
#PBS -A P93300065
#PBS -q casper
#PBS -l select=1:ncpus=16:ngpus=1:mem=256GB:gpu_type=v100
#PBS -l walltime=08:00:00
#PBS -j oe
#PBS -o .logs/
#PBS -m abe
#PBS -M skycgale@uw.edu

set -euo pipefail

# Which daily X variant to train against -- "interp" or "avg".
DATA_VARIANT="${DATA_VARIANT:-interp}"
case "$DATA_VARIANT" in
    interp) X_PATH="/glade/derecho/scratch/skygale/Downscaling_Data/X_MESA_HR_daily_interp.nc" ;;
    avg)    X_PATH="/glade/derecho/scratch/skygale/Downscaling_Data/X_MESA_HR_daily_avg.nc" ;;
    *) echo "Unknown DATA_VARIANT: $DATA_VARIANT (expected 'interp' or 'avg')" >&2; exit 1 ;;
esac
# Opt-in override so a one-off experiment (e.g. a conservative-regridded Y, testing
# whether coastal-bias truth-regridding noise is the real driver) can point at an
# alternate Y file without needing a separate submission script.
Y_PATH="${Y_PATH_OVERRIDE:-/glade/derecho/scratch/skygale/Downscaling_Data/Y_MESA_HR_daily.nc}"

# Cross-dataset evaluation: set both to evaluate the checkpoint trained on
# MESA against a *different* dataset's X/Y files (e.g. FOSI) instead of
# MESA's own test split. Leave both blank (default) for normal same-dataset
# evaluation. Must be given together, and TRAIN_YEARS/TEST_YEARS must both
# be set (see train_engressnet.py --test-x-path/--test-y-path).
TEST_X_PATH="${TEST_X_PATH:-}"
TEST_Y_PATH="${TEST_Y_PATH:-}"

# Accepts "YYYY-YYYY" ranges and/or comma-separated years, e.g. "2000-2005" or "2021"
# Leave BOTH blank ("") to fall back to a random 80/20 train/test split
TRAIN_YEARS="${TRAIN_YEARS:-2000-2005}"
TEST_YEARS="${TEST_YEARS:-2021}"

# Optional seasonal focus: restrict both train and test samples to these
# calendar months before the year split, e.g. "3-7" for March-July. Comma/
# range syntax like TRAIN_YEARS. Leave blank ("") for all months (default).
MONTHS="${MONTHS:-}"

# "true"  -> sliding-window patch extraction
# "false" -> train directly on one lat/lon sub-domain (required for the
#            domain-mean SIT time series and the candidate-point
#            [Kivalina/Shishmaref/Kotzebue/Nome] time series -- both are
#            skipped under patches=True, see save_evaluation_data() in
#            functions_engressnet.py)
USE_PATCHES="${USE_PATCHES:-false}"

# Same sub-domain as the FOSI_2conv baseline (see submit_engressnet.sh for
# the full derivation): lat 60-75, lon -182 to -151, an 8x16-multiple crop
# covering all 5 candidate coastal communities. Same 1deg grid as the daily
# FOSI files, so the same bounds apply unchanged.
LAT_MIN="${LAT_MIN:-60}"
LAT_MAX="${LAT_MAX:-75}"
LON_MIN="${LON_MIN:--182}"
LON_MAX="${LON_MAX:--151}"

NUM_EPOCHS="${NUM_EPOCHS:-20}"

# Same as the FOSI_2conv baseline.
K="${K:-20}"
K_EVAL="${K_EVAL:-20}"
BETA="${BETA:-0.8}"

# Random seed (torch.manual_seed + train/test split RNG), default matches
# train_engressnet.py's own default -- override to run repeated-seed
# variance checks without editing this file, e.g.
# qsub -v SEED=1 submit_engressnet_daily_mesa.sh
SEED="${SEED:-0}"

# Coastal-focused training (unchanged from the monthly baseline).
COASTAL_WIDTH="${COASTAL_WIDTH:-5}"
COASTAL_BOOST="${COASTAL_BOOST:-2.0}"
# Windowed attention at the decoder end. Default off, unchanged architecture.
ATTENTION_END="${ATTENTION_END:-false}"
ATTN_WINDOW_SIZE="${ATTN_WINDOW_SIZE:-8}"
ATTN_NUM_HEADS="${ATTN_NUM_HEADS:-4}"

# 2026-09-17 sensitivity-test knobs (see --noise-channels/--noise-kernel-size/
# --late-mask-fusion help text) -- all default to the settled architecture.
NOISE_CHANNELS="${NOISE_CHANNELS:-1}"
NOISE_KERNEL_SIZE="${NOISE_KERNEL_SIZE:-5}"
LATE_MASK_FUSION="${LATE_MASK_FUSION:-false}"

# Optional batch folder name. When set, --output-dir is passed explicitly so
# this run's output lands under results/<BATCH_NAME>/<run_tag> instead of the
# default flat results/<run_tag>.
BATCH_NAME="${BATCH_NAME:-}"

# Pretraining / transfer learning: set INIT_CHECKPOINT to a prior run's model_state_dict.pt to
# initialize this run's weights from it instead of random init. See submit_engressnet_daily.sh's
# matching comment -- COLLAPSE_WIND_VECTOR is a no-op on MESA (already 3-channel, no vector wind)
# but kept here for symmetry/consistency with the FOSI template.
INIT_CHECKPOINT="${INIT_CHECKPOINT:-}"
COLLAPSE_WIND_VECTOR="${COLLAPSE_WIND_VECTOR:-false}"

# ==============================================================

echo "Job started on $(hostname) at $(date)"
echo "PBS_JOBID: ${PBS_JOBID:-not set}"
echo "Data variant: ${DATA_VARIANT}   X path: ${X_PATH}"
echo "Y path: ${Y_PATH}"
echo "Test X path override: ${TEST_X_PATH:-<none, same-dataset eval>}"
echo "Test Y path override: ${TEST_Y_PATH:-<none, same-dataset eval>}"
echo "Train years: ${TRAIN_YEARS:-<random split>}   Test years: ${TEST_YEARS:-<random split>}"
echo "Months: ${MONTHS:-<all months>}"
echo "USE_PATCHES: ${USE_PATCHES}"
echo "K (train ensemble size): ${K}   K_EVAL (eval ensemble size): ${K_EVAL}"
echo "Beta: ${BETA}"
echo "Seed: ${SEED}"
echo "Coastal width / boost: ${COASTAL_WIDTH} / ${COASTAL_BOOST}"
echo "Sub-domain: lat ${LAT_MIN}-${LAT_MAX}, lon ${LON_MIN}-${LON_MAX}"
echo "Attention end: ${ATTENTION_END} (window ${ATTN_WINDOW_SIZE}, heads ${ATTN_NUM_HEADS})"
echo "Noise channels: ${NOISE_CHANNELS}   Noise kernel size: ${NOISE_KERNEL_SIZE}   Late mask fusion: ${LATE_MASK_FUSION}"
echo "Batch name: ${BATCH_NAME:-<none, flat results/>}"
echo "Init checkpoint: ${INIT_CHECKPOINT:-<none, random init>}   Collapse wind vector: ${COLLAPSE_WIND_VECTOR}"

module load conda
conda activate downscaling_env

# functions_engressnet.py needs evaluation/member_metrics.py as a sibling import
# (MESACLIP per-member-averaged metrics), but this script's own directory (wherever
# $PBS_O_WORKDIR ends up being, depending on how it was invoked -- see Version6/README.md)
# doesn't put evaluation/ on sys.path. Confirmed missing 2026-09-11: every MESA training
# job submitted through this template since the 2026-08-25 stage reorg
# (functions_engressnet.py/member_metrics.py split into separate folders) would have
# failed with ModuleNotFoundError at import time -- no GPU time wasted (fails before
# training starts), but silent until someone actually ran one.
#
# Hardcoded absolute path (matching every other path in this script -- X_PATH, Y_PATH,
# WEIGHTED_GRIDS_DIR are all absolute too), not resolved relative to this script's own
# file location: a first attempt using "$(dirname "${BASH_SOURCE[0]}")" worked when
# tested interactively but still failed inside the actual PBS job, since PBS commonly
# copies/spools the submitted script before executing it -- BASH_SOURCE[0] then points
# at that spool copy, not the real file under training/.
export PYTHONPATH="/glade/work/skygale/projects/SeaIceDownscaling/Version6/evaluation:${PYTHONPATH:-}"

cd "$PBS_O_WORKDIR"

ARGS=(--x-path "$X_PATH" --y-path "$Y_PATH" --num-epochs "$NUM_EPOCHS" --k "$K" --k-eval "$K_EVAL" --beta "$BETA" --seed "$SEED" --coastal-width "$COASTAL_WIDTH" --coastal-boost "$COASTAL_BOOST")
[ -n "$TRAIN_YEARS" ] && ARGS+=(--train-years "$TRAIN_YEARS")
[ -n "$TEST_YEARS" ] && ARGS+=(--test-years "$TEST_YEARS")
[ -n "$MONTHS" ] && ARGS+=(--months "$MONTHS")
if [ -n "$TEST_X_PATH" ]; then
    ARGS+=(--test-x-path "$TEST_X_PATH" --test-y-path "$TEST_Y_PATH")
fi

if [ "$USE_PATCHES" = true ]; then
    ARGS+=(--patches)
else
    ARGS+=(--no-patches --lat-min "$LAT_MIN" --lat-max "$LAT_MAX" --lon-min "$LON_MIN" --lon-max "$LON_MAX")
fi


if [ "$ATTENTION_END" = true ]; then
    ARGS+=(--attention-end --attn-window-size "$ATTN_WINDOW_SIZE" --attn-num-heads "$ATTN_NUM_HEADS")
fi

ARGS+=(--noise-channels "$NOISE_CHANNELS" --noise-kernel-size "$NOISE_KERNEL_SIZE")
[ "$LATE_MASK_FUSION" = true ] && ARGS+=(--late-mask-fusion)
[ -n "$INIT_CHECKPOINT" ] && ARGS+=(--init-checkpoint "$INIT_CHECKPOINT")
[ "$COLLAPSE_WIND_VECTOR" = true ] && ARGS+=(--collapse-wind-vector)

if [ -n "$BATCH_NAME" ]; then
    RUN_TAG="${PBS_JOBNAME}_${TRAIN_YEARS}_${TEST_YEARS}_${PBS_JOBID}"
    ARGS+=(--output-dir "results/${BATCH_NAME}/${RUN_TAG}")
fi

python train_engressnet.py "${ARGS[@]}"

echo "Job finished at $(date)"
