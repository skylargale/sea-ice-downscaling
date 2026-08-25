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
#PBS -o logs/
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
LAND_THRESHOLD="${LAND_THRESHOLD:-0.1}"

# Architecture toggles -- all default off, matching FOSI_2conv.
EXTRA_LAYER="${EXTRA_LAYER:-false}"
STOCHASTIC_REFINE="${STOCHASTIC_REFINE:-false}"
ENSCALE_NET="${ENSCALE_NET:-false}"
NOISE_SIGMA="${NOISE_SIGMA:-1.0}"

# Newer sensitivity-test toggles (2026-08-06): coastal input channel,
# auxiliary classification head, windowed attention at the decoder end.
# All default off, unchanged architecture.
COASTAL_CHANNEL="${COASTAL_CHANNEL:-false}"
CLASSIFICATION_HEAD="${CLASSIFICATION_HEAD:-false}"
CLASSIF_WEIGHT="${CLASSIF_WEIGHT:-0.1}"
ATTENTION_END="${ATTENTION_END:-false}"
ATTN_WINDOW_SIZE="${ATTN_WINDOW_SIZE:-8}"
ATTN_NUM_HEADS="${ATTN_NUM_HEADS:-4}"

# Optional batch folder name. When set, --output-dir is passed explicitly so
# this run's output lands under results/<BATCH_NAME>/<run_tag> instead of the
# default flat results/<run_tag>.
BATCH_NAME="${BATCH_NAME:-}"

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
echo "Land threshold: ${LAND_THRESHOLD}"
echo "Sub-domain: lat ${LAT_MIN}-${LAT_MAX}, lon ${LON_MIN}-${LON_MAX}"
echo "Extra layer (1024ch): ${EXTRA_LAYER}   Stochastic refine (EnScale-lite): ${STOCHASTIC_REFINE}"
echo "EnScaleNet: ${ENSCALE_NET}   Noise sigma: ${NOISE_SIGMA}"
echo "Coastal channel: ${COASTAL_CHANNEL}   Classification head: ${CLASSIFICATION_HEAD} (weight ${CLASSIF_WEIGHT})"
echo "Attention end: ${ATTENTION_END} (window ${ATTN_WINDOW_SIZE}, heads ${ATTN_NUM_HEADS})"
echo "Batch name: ${BATCH_NAME:-<none, flat results/>}"

module load conda
conda activate downscaling_env

cd "$PBS_O_WORKDIR"

ARGS=(--x-path "$X_PATH" --y-path "$Y_PATH" --num-epochs "$NUM_EPOCHS" --k "$K" --k-eval "$K_EVAL" --beta "$BETA" --seed "$SEED" --coastal-width "$COASTAL_WIDTH" --coastal-boost "$COASTAL_BOOST" --land-threshold "$LAND_THRESHOLD")
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

[ "$EXTRA_LAYER" = true ] && ARGS+=(--extra-layer)
[ "$STOCHASTIC_REFINE" = true ] && ARGS+=(--stochastic-refine)
[ "$ENSCALE_NET" = true ] && ARGS+=(--enscale-net)
ARGS+=(--noise-sigma "$NOISE_SIGMA")

[ "$COASTAL_CHANNEL" = true ] && ARGS+=(--coastal-channel)
if [ "$CLASSIFICATION_HEAD" = true ]; then
    ARGS+=(--classification-head --classif-weight "$CLASSIF_WEIGHT")
fi
if [ "$ATTENTION_END" = true ]; then
    ARGS+=(--attention-end --attn-window-size "$ATTN_WINDOW_SIZE" --attn-num-heads "$ATTN_NUM_HEADS")
fi

if [ -n "$BATCH_NAME" ]; then
    RUN_TAG="${PBS_JOBNAME}_${TRAIN_YEARS}_${TEST_YEARS}_${PBS_JOBID}"
    ARGS+=(--output-dir "results/${BATCH_NAME}/${RUN_TAG}")
fi

python train_engressnet.py "${ARGS[@]}"

echo "Job finished at $(date)"
