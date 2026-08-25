#!/bin/bash
# ==============================================================
# submit_stochastic_refine_bilinear_2015_2020_avg_sitbiasaug.sh
#
# Test 2 (domain-randomization): retrains the exact recommended config
# (same as submit_stochastic_refine_bilinear_2015_2020_avg_coastalboost1.sh's
# baseline, coastal_boost left at its default 2.0 this time) with a new
# augmentation: --sit-bias-aug-prob 0.5 --sit-bias-aug-mag 1.5 -- each
# training sample has a 50% chance of getting its SIT input channel shifted
# by a random negative bias up to 1.5m (matching the empirically-measured
# PIOMAS coastal thin-bias magnitude, see
# results/PIOMAS_obs_2020/piomas_vs_cryosat_raw*/). Goal: teach the network
# not to over-trust a systematically-too-thin low-res input, since FOSI's
# own training distribution never exposes it to that failure mode.
#
# See functions_engressnet.apply_sit_bias_augmentation for the mechanism.
# Smoke-tested standalone (unit test + 1-epoch CPU run reaching the
# training loop with no crash) before this submission.
#
# After this finishes, rerun the same PIOMAS-observational inference +
# CryoSat-2 comparison used for every other test in this chain.
#
# Submit with:  qsub submit_stochastic_refine_bilinear_2015_2020_avg_sitbiasaug.sh
# ==============================================================

#PBS -N FOSI_refine_bilin_avg_sitaug
#PBS -A P93300065
#PBS -q casper
#PBS -l select=1:ncpus=16:ngpus=1:mem=128GB:gpu_type=v100
#PBS -l walltime=08:00:00
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

PYTHONPATH="$PWD/evaluation:${PYTHONPATH:-}" python training/train_engressnet.py \
    --x-path "${DATA_DIR}/X_FOSI_HR_JRA55_daily_avg.nc" \
    --y-path "${DATA_DIR}/Y_FOSI_HR_JRA55_daily.nc" \
    --train-years "2015-2019" \
    --test-years "2020" \
    --no-patches --lat-min 60 --lat-max 75 --lon-min -182 --lon-max -151 \
    --stochastic-refine --noise-sigma 1.0 \
    --num-epochs 20 --k 20 --k-eval 20 --beta 0.8 \
    --coastal-width 5 --coastal-boost 2.0 --land-threshold 0.1 \
    --sit-bias-aug-prob 0.5 --sit-bias-aug-mag 1.5 \
    --seed 0 \
    --output-dir "results/FOSI_stochastic_refine_bilinear_2000_2020_avg_sitbiasaug/FOSI_refine_bilin_avg_sitaug_2015-2019_2020_${PBS_JOBID:-local}"

echo "Job finished at $(date)"
