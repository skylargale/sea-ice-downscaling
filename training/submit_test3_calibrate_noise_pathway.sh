#!/bin/bash
# ==============================================================
# submit_test3_calibrate_noise_pathway.sh
#
# Test 3 (narrow noise-pathway calibration): fine-tunes ONLY the
# noise-injection pathway (--freeze-backbone, functions_engressnet.
# set_noise_only_trainable) of the recommended FOSI-trained checkpoint on
# real PIOMAS-input/CryoSat-2-truth pairs from Jan-Feb 2020 (land-fixed
# PIOMAS X, real CryoSat-2 Y -- not PIOMAS-as-its-own-truth). Mar-Apr 2020
# is held out entirely for validation (see submit_test3_validate.sh),
# using the new --test-months override so the calibrated checkpoint's
# normalization stats (computed from this Jan-Feb run) can be reproduced
# exactly while evaluating on genuinely unseen months.
#
# This targets the calibration problem specifically (ensemble coverage
# ~1.5-2.8% vs. ideal ~90%), not the RMSE-vs-bilinear gap -- a much
# smaller, better-posed learning problem than a full retrain, per the
# reasoning that ruled out a full PIOMAS+CryoSat retrain earlier
# (too little real coastal data in this domain to safely relearn coastal
# texture, but plenty to fit a handful of noise-pathway parameters).
#
# Submit with:  qsub submit_test3_calibrate_noise_pathway.sh
# ==============================================================

#PBS -N test3_calibrate_janfeb
#PBS -A P93300065
#PBS -q casper
#PBS -l select=1:ncpus=16:ngpus=1:mem=128GB:gpu_type=v100
#PBS -l walltime=02:00:00
#PBS -j oe
#PBS -o logs/
#PBS -m abe
#PBS -M skycgale@uw.edu

set -euo pipefail

echo "Job started on $(hostname) at $(date)"
echo "PBS_JOBID: ${PBS_JOBID:-not set}"

module load conda
conda activate downscaling_env

cd "$PBS_O_WORKDIR/../.."
mkdir -p logs

DATA_DIR="/glade/derecho/scratch/skygale/Downscaling_Data"
CHECKPOINT="results/FOSI_stochastic_refine_bilinear_2000_2020_avg/FOSI_refine_bilin_avg_2015-2019_2020_5631173.casper-pbs/model_state_dict.pt"

PYTHONPATH="$PWD/evaluation:${PYTHONPATH:-}" python training/train_engressnet.py \
    --x-path "${DATA_DIR}/X_PIOMAS_obs_2020_daily_v2_interp.nc" \
    --y-path "${DATA_DIR}/Y_CRYOSAT_2020_daily.nc" \
    --months "1-2" \
    --no-patches --lat-min 60 --lat-max 75 --lon-min -182 --lon-max -151 \
    --stochastic-refine --noise-sigma 1.0 \
    --calibrate-from "$CHECKPOINT" --freeze-backbone \
    --num-epochs 8 --lr 0.0001 --k 20 --k-eval 20 --beta 0.8 \
    --coastal-width 5 --coastal-boost 2.0 --land-threshold 0.1 \
    --seed 0 \
    --output-dir "results/PIOMAS_obs_2020/test3_calibrate_janfeb_${PBS_JOBID:-local}"

echo "Job finished at $(date)"
