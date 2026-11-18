#!/bin/bash
# ==============================================================
# PBS template for train_engressnet.py on Casper: daily MESACLIP (CESM high-res, HIST+RCP8.5 stitched, 6 members).
#
# Every option is an environment variable (qsub -v ...); the defaults reproduce the earliest daily runs, so the final
# model's flags are passed explicitly. Final model (FINAL2, recommended_config.md):
#   qsub -v DATA_VARIANT=avg,TRAIN_YEARS=2000-2020,TEST_YEARS=2021,TRAIN_RANDOM_DAYS=1825,RANDOM_DAYS_SEED=0,NOISE_AMP_MAP=true,\
#           ENERGY_PATCH_SIZE=10,NOISE_LR_MULT=10,EMA_DECAY=0.9995,COARSE_ZERO=true,BATCH_NAME=... submit_engressnet_daily_mesa.sh
# (training/submit_final2_training.sh submits all FINAL2 runs). Submit from training/ so "#PBS -o .logs/" resolves.
# Members .004/.005/.006/.007/.008/.010 (complete daily data in both periods; see processing/). Each date has 6
# samples, hence 256 GB. X carries wind speed only (FOSI has u_10, v_10).
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

# Which daily X variant: "avg" (area-mean coarse input, final model) or "interp".
DATA_VARIANT="${DATA_VARIANT:-interp}"
case "$DATA_VARIANT" in
    interp) X_PATH="/glade/derecho/scratch/skygale/downscaling/training_data/X_MESA_HR_daily_interp.nc" ;;
    avg)    X_PATH="/glade/derecho/scratch/skygale/downscaling/training_data/X_MESA_HR_daily_avg.nc" ;;
    *) echo "Unknown DATA_VARIANT: $DATA_VARIANT (expected 'interp' or 'avg')" >&2; exit 1 ;;
esac
# Opt-in overrides of the X / Y files (one-off experiments, e.g. a conservative-regridded Y).
X_PATH="${X_PATH_OVERRIDE:-$X_PATH}"
Y_PATH="${Y_PATH_OVERRIDE:-/glade/derecho/scratch/skygale/downscaling/training_data/Y_MESA_HR_daily.nc}"

# Cross-dataset evaluation / inference on other inputs: set both to evaluate against a different X/Y pair (e.g. PIOMAS
# inputs with NUM_EPOCHS=0 and INIT_CHECKPOINT). Must be given together, with TRAIN_YEARS and TEST_YEARS both set
# (see train_engressnet.py --test-x-path/--test-y-path).
TEST_X_PATH="${TEST_X_PATH:-}"
TEST_Y_PATH="${TEST_Y_PATH:-}"

# "YYYY-YYYY" ranges and/or comma-separated years, e.g. "2000-2005" or "2021".
# Leave BOTH blank ("") to fall back to a random 80/20 train/test split.
TRAIN_YEARS="${TRAIN_YEARS:-2000-2005}"
TEST_YEARS="${TEST_YEARS:-2021}"
# Optional seasonal focus, e.g. "3-7" for March-July (blank = all months).
MONTHS="${MONTHS:-}"

# Sub-domain: lat 60-75, lon -182 to -151 (an 8-multiple crop covering all 5 coastal communities).
LAT_MIN="${LAT_MIN:-60}"
LAT_MAX="${LAT_MAX:-75}"
LON_MIN="${LON_MIN:--182}"
LON_MAX="${LON_MAX:--151}"

NUM_EPOCHS="${NUM_EPOCHS:-20}"
K="${K:-20}"
K_EVAL="${K_EVAL:-20}"
BETA="${BETA:-0.8}"
SEED="${SEED:-0}"                 # torch seed (+ random-split / random-days seed unless RANDOM_DAYS_SEED is set)
LR="${LR:-}"                      # blank = train_engressnet.py default
COASTAL_WIDTH="${COASTAL_WIDTH:-5}"
COASTAL_BOOST="${COASTAL_BOOST:-2.0}"

# Appendix ablations (default = final architecture)
NOISE_CHANNELS="${NOISE_CHANNELS:-1}"
NOISE_KERNEL_SIZE="${NOISE_KERNEL_SIZE:-5}"
LATE_MASK_FUSION="${LATE_MASK_FUSION:-false}"
# Deterministic U-Net baseline (noise off, coastal-weighted MSE)
DETERMINISTIC_MSE="${DETERMINISTIC_MSE:-false}"

# Final-model options (see recommended_config.md; all default off so older runs reproduce)
NOISE_AMP_MAP="${NOISE_AMP_MAP:-false}"                # state-dependent noise amplitude maps
ENERGY_PATCH_SIZE="${ENERGY_PATCH_SIZE:-0}"            # energy score over N x N px tiles (final: 10)
NOISE_LR_MULT="${NOISE_LR_MULT:-1}"                    # lr multiplier for the noise parameters (final: 10)
EMA_DECAY="${EMA_DECAY:-0}"                            # EMA of weights (final: 0.997 FOSI / 0.9995 MESACLIP); 0 = off
COARSE_ZERO="${COARSE_ZERO:-false}"                    # at evaluation, members exactly 0 where coarse SIT is 0 (final: true)
COARSE_ZERO_LOSS="${COARSE_ZERO_LOSS:-false}"          # also apply the coarse-zero rule in the training loss (final: false)
TRAIN_RANDOM_DAYS="${TRAIN_RANDOM_DAYS:-0}"            # N random dates from TRAIN_YEARS (the pool) instead of all (final: 1825); 0 = off
RANDOM_DAYS_SEED="${RANDOM_DAYS_SEED:-}"               # seed of the date draw (blank = SEED)
RANDOM_DAYS_EXCLUDE_YEARS="${RANDOM_DAYS_EXCLUDE_YEARS:-}"   # inference of a random-days model on other years: the training run's test years
NOISE_ABLATE="${NOISE_ABLATE:-}"                       # stages to switch off at inference, e.g. "d3" or "d2,d1"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"              # same gradient in N chunks (less GPU memory); 1 = off
LAND_THRESHOLD="${LAND_THRESHOLD:-}"                   # final-round sensitivity: ocean-fraction land cutoff (empty = 0.1)
GLOBAL_NOISE="${GLOBAL_NOISE:-false}"                  # final-round sensitivity: one noise field shared by all stages
ENCODER_DEPTH="${ENCODER_DEPTH:-}"                     # final-round sensitivity: 5 = one more encoder block (empty = 4)
ATTENTION_END="${ATTENTION_END:-false}"                # final-round sensitivity: end-of-decoder windowed attention

# Optional batch folder: output goes to results/<BATCH_NAME>/<jobname>_<train>_<test>_<jobid> instead of flat results/.
BATCH_NAME="${BATCH_NAME:-}"

# Initialize from a prior run's model_state_dict.pt (inference: NUM_EPOCHS=0). COLLAPSE_WIND_VECTOR=true turns FOSI's
# (u_10, v_10) into one wind-speed channel to match a MESACLIP checkpoint (no-op on MESACLIP).
INIT_CHECKPOINT="${INIT_CHECKPOINT:-}"
COLLAPSE_WIND_VECTOR="${COLLAPSE_WIND_VECTOR:-false}"

# ==============================================================

echo "Job started on $(hostname) at $(date)"
echo "PBS_JOBID: ${PBS_JOBID:-not set}"
echo "Data variant: ${DATA_VARIANT}   X path: ${X_PATH}"
echo "Y path: ${Y_PATH}"
echo "Test X/Y: ${TEST_X_PATH:-<same dataset>} ${TEST_Y_PATH:-}"
echo "Train years: ${TRAIN_YEARS:-<random split>}   Test years: ${TEST_YEARS:-<random split>}   Months: ${MONTHS:-<all>}"
echo "K / K_EVAL: ${K} / ${K_EVAL}   Beta: ${BETA}   Seed: ${SEED}   Epochs: ${NUM_EPOCHS}"
echo "Sub-domain: lat ${LAT_MIN}-${LAT_MAX}, lon ${LON_MIN}-${LON_MAX}   Coastal width / boost: ${COASTAL_WIDTH} / ${COASTAL_BOOST}"
echo "Noise channels / kernel: ${NOISE_CHANNELS} / ${NOISE_KERNEL_SIZE}   Late mask fusion: ${LATE_MASK_FUSION}   Deterministic MSE: ${DETERMINISTIC_MSE}"
echo "Final-model flags: amp_map=${NOISE_AMP_MAP} patch=${ENERGY_PATCH_SIZE} noise_lr_mult=${NOISE_LR_MULT} ema=${EMA_DECAY} coarse_zero=${COARSE_ZERO} lr=${LR:-default}"
echo "Random days: ${TRAIN_RANDOM_DAYS} (seed ${RANDOM_DAYS_SEED:-SEED}, exclude ${RANDOM_DAYS_EXCLUDE_YEARS:-TEST_YEARS})   Noise ablate: ${NOISE_ABLATE:-none}   Grad accum: ${GRAD_ACCUM_STEPS}"
echo "Batch name: ${BATCH_NAME:-<none, flat results/>}"
echo "Init checkpoint: ${INIT_CHECKPOINT:-<none, random init>}   Collapse wind vector: ${COLLAPSE_WIND_VECTOR}"

module load conda
conda activate seaice-downscaling

# functions_engressnet.py imports evaluation/member_metrics.py. Absolute path on purpose: PBS spools the script before
# running it, so BASH_SOURCE-relative paths point at the spool copy.
export PYTHONPATH="/glade/work/skygale/projects/SeaIceDownscaling/evaluation:${PYTHONPATH:-}"

cd "$PBS_O_WORKDIR"

ARGS=(--x-path "$X_PATH" --y-path "$Y_PATH" --num-epochs "$NUM_EPOCHS" --k "$K" --k-eval "$K_EVAL" --beta "$BETA" --seed "$SEED"
      --coastal-width "$COASTAL_WIDTH" --coastal-boost "$COASTAL_BOOST"
      --lat-min "$LAT_MIN" --lat-max "$LAT_MAX" --lon-min "$LON_MIN" --lon-max "$LON_MAX"
      --noise-channels "$NOISE_CHANNELS" --noise-kernel-size "$NOISE_KERNEL_SIZE")
[ -n "$TRAIN_YEARS" ] && ARGS+=(--train-years "$TRAIN_YEARS")
[ -n "$TEST_YEARS" ] && ARGS+=(--test-years "$TEST_YEARS")
[ -n "$MONTHS" ] && ARGS+=(--months "$MONTHS")
[ -n "$TEST_X_PATH" ] && ARGS+=(--test-x-path "$TEST_X_PATH" --test-y-path "$TEST_Y_PATH")
[ -n "$LR" ] && ARGS+=(--lr "$LR")
[ "$LATE_MASK_FUSION" = true ] && ARGS+=(--late-mask-fusion)
[ -n "$LAND_THRESHOLD" ] && ARGS+=(--land-threshold "$LAND_THRESHOLD")
[ "$GLOBAL_NOISE" = true ] && ARGS+=(--global-noise)
[ -n "$ENCODER_DEPTH" ] && ARGS+=(--encoder-depth "$ENCODER_DEPTH")
[ "$ATTENTION_END" = true ] && ARGS+=(--attention-end)
[ "$DETERMINISTIC_MSE" = true ] && ARGS+=(--deterministic-mse)
[ "$NOISE_AMP_MAP" = true ] && ARGS+=(--noise-amp-map)
[ "$ENERGY_PATCH_SIZE" != 0 ] && ARGS+=(--energy-patch-size "$ENERGY_PATCH_SIZE")
[ "$NOISE_LR_MULT" != 1 ] && ARGS+=(--noise-lr-mult "$NOISE_LR_MULT")
[ "$EMA_DECAY" != 0 ] && ARGS+=(--ema-decay "$EMA_DECAY")
[ "$COARSE_ZERO" = true ] && ARGS+=(--coarse-zero)
[ "$COARSE_ZERO_LOSS" = true ] && ARGS+=(--coarse-zero-loss)
[ "$TRAIN_RANDOM_DAYS" != 0 ] && ARGS+=(--train-random-days "$TRAIN_RANDOM_DAYS")
[ -n "$RANDOM_DAYS_SEED" ] && ARGS+=(--random-days-seed "$RANDOM_DAYS_SEED")
[ -n "$RANDOM_DAYS_EXCLUDE_YEARS" ] && ARGS+=(--random-days-exclude-years "$RANDOM_DAYS_EXCLUDE_YEARS")
[ -n "$NOISE_ABLATE" ] && ARGS+=(--noise-ablate "$NOISE_ABLATE")
[ "$GRAD_ACCUM_STEPS" != 1 ] && ARGS+=(--grad-accum-steps "$GRAD_ACCUM_STEPS")
[ -n "$INIT_CHECKPOINT" ] && ARGS+=(--init-checkpoint "$INIT_CHECKPOINT")
[ "$COLLAPSE_WIND_VECTOR" = true ] && ARGS+=(--collapse-wind-vector)

if [ -n "$BATCH_NAME" ]; then
    RUN_TAG="${PBS_JOBNAME}_${TRAIN_YEARS}_${TEST_YEARS}_${PBS_JOBID}"
    ARGS+=(--output-dir "results/${BATCH_NAME}/${RUN_TAG}")
fi

python train_engressnet.py "${ARGS[@]}"

echo "Job finished at $(date)"
