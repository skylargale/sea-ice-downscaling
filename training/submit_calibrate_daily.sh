#!/bin/bash
# ==============================================================
# PBS batch submission script for stage-2 "freeze backbone, calibrate noise
# pathway" fine-tuning (Version4), on top of an already-trained daily-FOSI
# or daily-MESA checkpoint.
#
# Loads CHECKPOINT (a model_state_dict.pt from a completed run), freezes
# every parameter except the noise-injection pathway (z_proj_*/concat_d*/
# local_noise_mix/refiner -- see functions_engressnet.set_noise_only_trainable),
# and continues training for a short NUM_EPOCHS at a smaller LR, aimed at
# recalibrating ensemble spread (Spread/Error) without touching the
# parameters that determine mean accuracy/sharpness.
#
# IMPORTANT: every architecture/domain flag below (EXTRA_LAYER,
# STOCHASTIC_REFINE, coastal channel, classification head, attention end,
# domain bounds, data variant/path, train/test years) must match the run
# that produced CHECKPOINT exactly, or load_state_dict will fail on a shape
# mismatch, or (worse) silently train on a different train/test split than
# the checkpoint was evaluated on. Not supported with ENSCALE_NET=true (see
# set_noise_only_trainable's docstring -- the job will error out cleanly if
# you try).
#
# Submit with:
#   qsub -v CHECKPOINT="results/FOSI_daily_combo_interp/<run>/model_state_dict.pt",\
# DATA_VARIANT="interp",TRAIN_YEARS="2000-2005",TEST_YEARS="2021" submit_calibrate_daily.sh
# ==============================================================

#PBS -N calibrate_daily
#PBS -A P93300065
#PBS -q casper
#PBS -l select=1:ncpus=16:ngpus=1:mem=128GB:gpu_type=v100
#PBS -l walltime=02:00:00
#PBS -j oe
#PBS -o logs/
#PBS -m abe
#PBS -M skycgale@uw.edu

set -euo pipefail

if [ -z "${CHECKPOINT:-}" ]; then
    echo "CHECKPOINT is required (path to a model_state_dict.pt from a completed run)." >&2
    exit 1
fi

# Which dataset/variant this checkpoint was trained against.
DATASET="${DATASET:-FOSI}"
DATA_VARIANT="${DATA_VARIANT:-interp}"
case "$DATASET" in
    FOSI) DATA_PREFIX="FOSI_HR_JRA55_daily" ;;
    MESA) DATA_PREFIX="MESA_HR_daily" ;;
    *) echo "Unknown DATASET: $DATASET (expected 'FOSI' or 'MESA')" >&2; exit 1 ;;
esac
case "$DATA_VARIANT" in
    interp) X_PATH="/glade/derecho/scratch/skygale/Downscaling_Data/X_${DATA_PREFIX}_interp.nc" ;;
    avg)    X_PATH="/glade/derecho/scratch/skygale/Downscaling_Data/X_${DATA_PREFIX}_avg.nc" ;;
    *) echo "Unknown DATA_VARIANT: $DATA_VARIANT (expected 'interp' or 'avg')" >&2; exit 1 ;;
esac
Y_PATH="/glade/derecho/scratch/skygale/Downscaling_Data/Y_${DATA_PREFIX}.nc"

# Must match the checkpoint's original train/test split.
TRAIN_YEARS="${TRAIN_YEARS:-2000-2005}"
TEST_YEARS="${TEST_YEARS:-2021}"

USE_PATCHES="${USE_PATCHES:-false}"
LAT_MIN="${LAT_MIN:-60}"
LAT_MAX="${LAT_MAX:-75}"
LON_MIN="${LON_MIN:--182}"
LON_MAX="${LON_MAX:--151}"

K="${K:-20}"
K_EVAL="${K_EVAL:-20}"
BETA="${BETA:-0.8}"
COASTAL_WIDTH="${COASTAL_WIDTH:-5}"
COASTAL_BOOST="${COASTAL_BOOST:-2.0}"
LAND_THRESHOLD="${LAND_THRESHOLD:-0.1}"

# Must match the checkpoint's architecture exactly.
EXTRA_LAYER="${EXTRA_LAYER:-false}"
STOCHASTIC_REFINE="${STOCHASTIC_REFINE:-false}"
ENSCALE_NET="${ENSCALE_NET:-false}"
NOISE_SIGMA="${NOISE_SIGMA:-1.0}"
COASTAL_CHANNEL="${COASTAL_CHANNEL:-false}"
CLASSIFICATION_HEAD="${CLASSIFICATION_HEAD:-false}"
CLASSIF_WEIGHT="${CLASSIF_WEIGHT:-0.1}"
ATTENTION_END="${ATTENTION_END:-false}"
ATTN_WINDOW_SIZE="${ATTN_WINDOW_SIZE:-8}"
ATTN_NUM_HEADS="${ATTN_NUM_HEADS:-4}"

# Calibration-stage hyperparameters -- short + small-LR by design (fine-
# tuning a small already-good parameter subset, not training from scratch).
NUM_EPOCHS="${NUM_EPOCHS:-5}"
LR="${LR:-0.0001}"

BATCH_NAME="${BATCH_NAME:-}"

# ==============================================================

echo "Job started on $(hostname) at $(date)"
echo "PBS_JOBID: ${PBS_JOBID:-not set}"
echo "Checkpoint: ${CHECKPOINT}"
echo "Dataset: ${DATASET}   Data variant: ${DATA_VARIANT}   X path: ${X_PATH}"
echo "Train years: ${TRAIN_YEARS}   Test years: ${TEST_YEARS}"
echo "Calibration epochs: ${NUM_EPOCHS}   LR: ${LR}"
echo "Sub-domain: lat ${LAT_MIN}-${LAT_MAX}, lon ${LON_MIN}-${LON_MAX}"
echo "Extra layer: ${EXTRA_LAYER}   Stochastic refine: ${STOCHASTIC_REFINE}   EnScaleNet: ${ENSCALE_NET}"
echo "Coastal channel: ${COASTAL_CHANNEL}   Classification head: ${CLASSIFICATION_HEAD}   Attention end: ${ATTENTION_END}"

module load conda
conda activate downscaling_env

cd "$PBS_O_WORKDIR"

ARGS=(--x-path "$X_PATH" --y-path "$Y_PATH" --num-epochs "$NUM_EPOCHS" --lr "$LR" --k "$K" --k-eval "$K_EVAL" --beta "$BETA" --coastal-width "$COASTAL_WIDTH" --coastal-boost "$COASTAL_BOOST" --land-threshold "$LAND_THRESHOLD" --train-years "$TRAIN_YEARS" --test-years "$TEST_YEARS")

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

ARGS+=(--calibrate-from "$CHECKPOINT" --freeze-backbone)

if [ -n "$BATCH_NAME" ]; then
    RUN_TAG="${PBS_JOBNAME}_${TRAIN_YEARS}_${TEST_YEARS}_${PBS_JOBID}"
    ARGS+=(--output-dir "results/${BATCH_NAME}/${RUN_TAG}")
fi

python train_engressnet.py "${ARGS[@]}"

echo "Job finished at $(date)"
